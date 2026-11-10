"""Joint state thread - reads joint state and sends commands."""

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from arms.configs import (
    JOINT_STATE_STREAMING_RATE,
    MAX_JOINT_SLEW_DEG_S,
    SPEED_SCALE,
    USE_AGILEX_TELEOP,
)
from arms.data_manager import DataManager, RobotActivityState, activity_mode_code
from arms.logging_setup import get_logger

from arms.piper_controller import PiperController

log = get_logger("joint")

_GRIPPER_TOGGLE_PRESS_THRESHOLD = 0.5


def _apply_controller_stop_latch(
    data_manager: DataManager, robot_controller: PiperController, hand: str
) -> None:
    """Reflect a hardware-boundary Teach/Stop in operator state.

    The controller remains the safety authority. This retires the old clutch
    and makes dashboard Enable available after a physical Teach intervention.
    """
    latched = getattr(robot_controller, "is_stop_hold_latched", None)
    if latched is None or not latched():
        return
    if data_manager.get_teleop_active(hand):
        data_manager.set_teleop_state(hand, False, None, None)
    if data_manager.get_brain_control_armed(hand):
        brain = data_manager.get_brain_control(hand)
        data_manager.set_brain_control_params(
            armed=False,
            diameter=brain.diameter,
            slider_norm=brain.slider_norm,
            clamp_mode=brain.clamp_mode,
            active_group=brain.active_group,
            selected_alpha=brain.selected_alpha,
            unselected_alpha=brain.unselected_alpha,
            hand=hand,
        )
    state = data_manager.get_robot_activity_state(hand)
    if state not in (RobotActivityState.PAUSED, RobotActivityState.DISABLED):
        data_manager.set_robot_activity_state(RobotActivityState.PAUSED, hand)


def joint_state_thread(
    data_manager: DataManager,
    robot_controller: PiperController,
    hand: str = "right",
    stop_event: threading.Event | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> None:
    """Joint state thread - reads joint state and sends commands."""
    log.info(f"Joint state thread started ({hand})")
    diagnostics_enabled = os.getenv(
        "PIPER_RECORDING_DIAGNOSTICS", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    dt: float = 1.0 / JOINT_STATE_STREAMING_RATE
    # Base slew rate (deg/s) for the clean "command" telemetry (see the emission
    # block below). The per-tick cap is this * the MEASURED loop dt, CLAMPED to
    # _MAX_SLEW_DT_S so a stalled loop (VR tracking loss) still can't emit one huge
    # step into the command signal -- the exact failure the IK thread's UNclamped
    # variable dt had. Using the real (clamped) dt, not the nominal 1/100 s, keeps
    # genuine fast motion from being over-smoothed on hosts where the loop actually
    # runs slower than JOINT_STATE_STREAMING_RATE (e.g. ~50 Hz on the Jetson).
    slew_deg_s: np.ndarray = np.asarray(MAX_JOINT_SLEW_DEG_S, dtype=np.float64)
    _MAX_SLEW_DT_S: float = dt * 2.5
    prev_teleop_active: bool = False
    prev_agilex_generation: int | None = None
    prev_grip_pressed: bool = False
    prev_gripper_mode: str | None = None
    toggle_gripper_open: bool | None = None
    prev_current_joint_angles: np.ndarray | None = None
    prev_target_joint_angles: np.ndarray | None = None
    # Anchor for the glitch-free joint command telemetry. It advances EVERY loop
    # tick (the slew reconstruction), and the new signals are LOGGED every tick too
    # -- exactly like joint_positions. We deliberately do NOT throttle here: the
    # BigQuery sink already rate-limits every signal per-hand to --gcp-telemetry-hz
    # (gcp_telemetry.py), so a second client-side throttle only double-quantizes the
    # stream (30 Hz gate on top of the sink's 30 Hz gate cascaded to ~20 Hz). Logging
    # every tick makes these signals sample on the SAME iterations as joint_positions
    # -> phase-aligned action/state, and they scale together if the sink cap changes.
    last_logged_command: np.ndarray | None = None
    last_iter_ts: float | None = None
    last_feedback_rx_ts: float = 0.0
    last_feedback_warn_ts: float = 0.0
    last_control_warn_ts: float = 0.0
    last_clamp_warn_ts: float = 0.0
    last_motion_ts: float = 0.0
    last_motion_warn_ts: float = 0.0
    last_joint_telemetry_poll_ts: float = 0.0
    last_fault_flag_poll_ts: float = 0.0
    fault_flag_poll_period_s: float = 1.0  # fault flags drift slowly; ~1 Hz
    diag_last_ts = time.time()
    diag_loops = 0
    diag_overruns = 0
    diag_max_loop_ms = 0.0
    diag_motor_poll_ms = 0.0
    diag_motor_poll_count = 0
    diag_fault_poll_ms = 0.0
    diag_fault_poll_count = 0

    def _format_clamp_details(
        requested: np.ndarray, clamped: np.ndarray, diff: np.ndarray
    ) -> str:
        joint_indices = np.flatnonzero(diff > 0.5)
        try:
            limits = robot_controller.get_effective_joint_limits_deg()
        except Exception:
            limits = None

        details: list[str] = []
        for idx in joint_indices:
            if limits is not None and limits.shape == (requested.size, 2):
                details.append(
                    f"joint{idx + 1}: requested {requested[idx]:.1f}° -> "
                    f"{clamped[idx]:.1f}° "
                    f"(limit {limits[idx, 0]:.1f}°..{limits[idx, 1]:.1f}°)"
                )
            else:
                details.append(
                    f"joint{idx + 1}: requested {requested[idx]:.1f}° -> "
                    f"{clamped[idx]:.1f}°"
                )
        return "; ".join(details)

    try:
        while not data_manager.is_shutdown_requested() and not (
            stop_event is not None and stop_event.is_set()
        ):
            iteration_start: float = time.time()

            # Per-tick slew cap for the command telemetry, from the MEASURED loop
            # dt clamped to _MAX_SLEW_DT_S (tracks real loop speed; a stall is
            # capped so it can't leak a huge step). See slew_deg_s above.
            if last_iter_ts is not None:
                _cmd_dt = iteration_start - last_iter_ts
                if _cmd_dt > _MAX_SLEW_DT_S or _cmd_dt <= 0.0:
                    _cmd_dt = _MAX_SLEW_DT_S
            else:
                _cmd_dt = dt
            last_iter_ts = iteration_start
            max_cmd_step_deg = slew_deg_s * _cmd_dt

            # Get current joint angles and gripper value
            joint_timing = None
            if hasattr(type(robot_controller), "get_current_joint_feedback"):
                current_joint_angles, joint_timing = (
                    robot_controller.get_current_joint_feedback()
                )
            else:
                current_joint_angles = robot_controller.get_current_joint_angles()
            if current_joint_angles is not None:
                data_manager.set_current_joint_angles(
                    current_joint_angles, hand, timing=joint_timing
                )
                last_feedback_rx_ts = iteration_start
                try:
                    if (
                        prev_current_joint_angles is None
                        or np.max(
                            np.abs(current_joint_angles - prev_current_joint_angles)
                        )
                        > 0.05
                    ):
                        last_motion_ts = iteration_start
                    prev_current_joint_angles = current_joint_angles
                except Exception:
                    prev_current_joint_angles = current_joint_angles

            if data_manager.joint_telemetry_streams_enabled() and (
                iteration_start - last_joint_telemetry_poll_ts
                >= data_manager.joint_telemetry_min_period_s()
            ):
                last_joint_telemetry_poll_ts = iteration_start
                motor_poll_start = time.perf_counter()
                currents_a, velocities_rad_s = (
                    robot_controller.get_current_joint_motor_telemetry(
                        non_blocking=True
                    )
                )
                if diagnostics_enabled:
                    diag_motor_poll_ms += (
                        time.perf_counter() - motor_poll_start
                    ) * 1000.0
                    diag_motor_poll_count += 1
                if currents_a is not None:
                    data_manager.set_current_joint_currents_a(currents_a, hand)
                if velocities_rad_s is not None:
                    data_manager.set_current_joint_velocities_rad_s(
                        velocities_rad_s, hand
                    )

            # Low-rate poll of per-joint fault/status flags (overheating,
            # overcurrent, stall, etc.). Kept separate from the higher-rate
            # current/velocity stream so Prometheus can observe faults without
            # adding meaningful load to the control path.
            piper_feedback_enabled = (
                data_manager.joint_telemetry_streams_enabled()
                or data_manager.piper_feedback_streams_enabled()
            )
            piper_feedback_period_s = max(
                fault_flag_poll_period_s,
                data_manager.piper_feedback_min_period_s(),
            )
            if piper_feedback_enabled and (
                iteration_start - last_fault_flag_poll_ts
                >= piper_feedback_period_s
            ):
                last_fault_flag_poll_ts = iteration_start
                fault_poll_start = time.perf_counter()
                piper_status = robot_controller.get_piper_feedback_status(
                    non_blocking=True
                )
                if piper_status is not None:
                    for field, values in piper_status.items():
                        if field.startswith("motor_fault_"):
                            data_manager.log_telemetry_signal(
                                f"log_fault_{field[len('motor_fault_'):]}",
                                values,
                                hand,
                            )
                        elif field in ("motor_temperature", "driver_temperature"):
                            data_manager.log_telemetry_signal(
                                f"log_{field}", values, hand
                            )
                        else:
                            data_manager.log_telemetry_signal(
                                f"log_piper_{field}", values, hand
                            )
                if diagnostics_enabled:
                    diag_fault_poll_ms += (
                        time.perf_counter() - fault_poll_start
                    ) * 1000.0
                    diag_fault_poll_count += 1

            # Get current gripper open value and set in state
            gripper_timing = None
            if hasattr(type(robot_controller), "get_current_gripper_feedback"):
                gripper_open_value, gripper_timing = (
                    robot_controller.get_current_gripper_feedback()
                )
            else:
                gripper_open_value = robot_controller.get_current_gripper_open_value()
            if gripper_open_value is not None:
                data_manager.set_current_gripper_open_value(
                    gripper_open_value, hand, timing=gripper_timing
                )

            _apply_controller_stop_latch(data_manager, robot_controller, hand)
            target_joint_angles = data_manager.get_target_joint_angles(hand)
            _, grip_value, _ = data_manager.get_controller_data(hand)
            teleop_active = data_manager.get_teleop_active(hand)
            brain_armed = data_manager.get_brain_control_armed(hand)
            # Set true by the ENABLED/PAUSED arm below when operator targets are
            # streaming; stays False in every other state (policy, preset, homing,
            # disabled) so the control-source label at the end of the tick is defined.
            driven = False

            # Check if robot is homing
            robot_activity_state = data_manager.get_robot_activity_state(hand)
            if robot_activity_state == RobotActivityState.HOMING:
                if robot_controller.is_robot_homed():
                    data_manager.set_robot_activity_state(
                        RobotActivityState.ENABLED, hand
                    )
                    log.info(f"Robot reached home position and is re-enabled ({hand})")

            elif robot_activity_state in (RobotActivityState.ENABLED, RobotActivityState.PAUSED):
                # If teleop just deactivated, actively hold the current measured position.
                # Without this, the controller may keep streaming a stale old target from the
                # last time teleop was active, which can look like "random delayed motion".
                if robot_activity_state == RobotActivityState.ENABLED and prev_teleop_active and not teleop_active:
                    held = (robot_controller.hold_teleop_position() is not None
                            if USE_AGILEX_TELEOP else robot_controller.hold_current_position())
                    if not held:
                        log.warning(
                            f"Failed to hold current position on teleop stop ({hand})"
                        )

                # If teleop is active and targets are changing but feedback is stale,
                # warn the operator (CAN RX stalled / robot-side protection / E-stop).
                if teleop_active and robot_activity_state == RobotActivityState.ENABLED:
                    # Take ONE coherent read of the controller's health. Each
                    # individual accessor re-reads the shared-memory snapshot, so
                    # checking freshness and then reading two more values lets the
                    # snapshot go stale in between -- which is how a loop streaming
                    # at 100 Hz got logged as "not running" with "No command TX for
                    # infs". None means "too stale to conclude anything": stay quiet
                    # rather than assert a fault. In-process controllers have no such
                    # method and are always coherent, so they fall back to the
                    # individual accessors.
                    bundle = getattr(robot_controller, "get_health_snapshot", None)
                    health = bundle() if bundle is not None else None
                    if bundle is not None and health is None:
                        loop_running = enabled = True  # unknown -> do not warn
                        cmd_tx_age_s = 0.0
                    elif health is not None:
                        loop_running = health["loop_running"]
                        enabled = health["enabled"]
                        cmd_tx_age_s = health["cmd_tx_age_s"]
                    else:
                        loop_running = robot_controller.is_control_loop_running()
                        enabled = robot_controller.is_robot_enabled()
                        cmd_tx_age_s = robot_controller.get_last_command_tx_age_s()

                    # 0) Warn if the control loop isn't running / hasn't transmitted recently.
                    if (iteration_start - last_control_warn_ts) > 1.0:
                        if not loop_running:
                            last_control_warn_ts = iteration_start
                            log.warning(
                                f"PiperController control_loop thread is not running ({hand}). "
                                "Targets may update in the GUI but no commands are being streamed to hardware."
                            )
                        elif cmd_tx_age_s > 0.25:
                            last_control_warn_ts = iteration_start
                            log.warning(
                                f"No command TX for {cmd_tx_age_s:.2f}s while teleop ACTIVE ({hand}). "
                                "This often means the control loop is blocked inside SDK I/O (CAN TX backpressure) "
                                "or the robot is ignoring commands."
                            )
                            robot_controller.log_hardware_diagnostics(
                                f"no command TX while teleop ACTIVE ({hand})"
                            )

                    # 1) Warn if the controller loop isn't actually enabled at the hardware layer.
                    if (
                        (not enabled)
                        and (iteration_start - last_feedback_warn_ts) > 1.0
                    ):
                        last_feedback_warn_ts = iteration_start
                        log.warning(
                            f"RobotActivityState is ENABLED but PiperController reports disabled ({hand}). "
                            "Press A (resume) or check E-stop/robot enable."
                        )
                        robot_controller.log_hardware_diagnostics(
                            f"software enabled but controller disabled ({hand})"
                        )

                    # 2) Warn if feedback is missing entirely (common when CAN RX stalls).
                    feedback_rx_age_s = (
                        iteration_start - last_feedback_rx_ts
                        if last_feedback_rx_ts > 0.0
                        else float("inf")
                    )
                    if feedback_rx_age_s > 0.5 and (
                        iteration_start - last_feedback_warn_ts
                    ) > 1.0:
                        last_feedback_warn_ts = iteration_start
                        log.warning(
                            f"No joint feedback for {feedback_rx_age_s:.2f}s while teleop ACTIVE ({hand}). "
                            "Ghost may move but actual robot/joints won't update. "
                            "Check SocketCAN (`ip -details -statistics link show <canX>`), cabling, and robot faults."
                        )
                        robot_controller.log_hardware_diagnostics(
                            f"no joint feedback while teleop ACTIVE ({hand})"
                        )

                    # Auto-hold: if feedback is stale for >1s, hold position and deactivate teleop.
                    if feedback_rx_age_s > 1.0:
                        data_manager.set_teleop_state(hand, False, None, None)
                        robot_controller.hold_current_position()
                        log.warning(
                            f"AUTO-HOLD: Feedback stale for {feedback_rx_age_s:.2f}s, "
                            f"deactivated teleop and holding position ({hand})."
                        )
                        robot_controller.log_hardware_diagnostics(
                            f"AUTO-HOLD after stale feedback ({hand})",
                            force=True,
                        )

                    # 3) Warn if targets differ a lot but joints aren't moving.
                    if (
                        current_joint_angles is not None
                        and target_joint_angles is not None
                        and robot_controller.is_robot_enabled()
                    ):
                        max_err_deg = float(
                            np.max(np.abs(target_joint_angles - current_joint_angles))
                        )
                        motion_age_s = (
                            iteration_start - last_motion_ts
                            if last_motion_ts > 0.0
                            else float("inf")
                        )
                        if (
                            max_err_deg > 3.0
                            and motion_age_s > 0.5
                            and (iteration_start - last_motion_warn_ts) > 1.0
                        ):
                            last_motion_warn_ts = iteration_start
                            try:
                                cmd_tx_age_s = robot_controller.get_last_command_tx_age_s()
                                loop_age_s = robot_controller.get_control_loop_age_s()
                                loop_ok = robot_controller.is_control_loop_running()
                                motionctrl_age_s = robot_controller.get_motionctrl_age_s()
                                motionctrl_mode = robot_controller.get_motionctrl_move_mode()
                            except Exception:
                                cmd_tx_age_s = float("nan")
                                loop_age_s = float("nan")
                                loop_ok = False
                                motionctrl_age_s = float("nan")
                                motionctrl_mode = None
                            log.warning(
                                f"Targets differ by {max_err_deg:.1f}° but joints haven't moved for "
                                f"{motion_age_s:.2f}s ({hand}). "
                                f"(control_loop_running={loop_ok}, control_loop_age={loop_age_s:.2f}s, cmd_tx_age={cmd_tx_age_s:.2f}s, "
                                f"motionctrl_age={motionctrl_age_s:.2f}s, motionctrl_mode={motionctrl_mode}) "
                                "Likely joint limit/torque protection, E-stop/fault, or CAN/SDK issues not surfacing as exceptions."
                            )
                            robot_controller.log_hardware_diagnostics(
                                f"targets differ but no joint motion ({hand})"
                            )

                joystick_y = data_manager.get_joystick_y(hand)
                joystick_active = abs(joystick_y) > 0.1
                should_stream_joints = (teleop_active and robot_activity_state == RobotActivityState.ENABLED) or (
                    brain_armed and (not teleop_active)
                ) or (joystick_active and robot_activity_state == RobotActivityState.ENABLED)
                if USE_AGILEX_TELEOP and teleop_active:
                    # Wait for IK to seed this clutch; the shared target can
                    # otherwise still contain the previous movement's endpoint.
                    _, generation, _, _ = data_manager.get_teleop_snapshot(hand)
                    if generation != prev_agilex_generation:
                        prev_target_joint_angles = None
                        held_target = robot_controller.prepare_teleop(generation)
                        if held_target is None:
                            data_manager.log_telemetry_signal(
                                "log_teleop_reengagement_event", [generation, 4], hand
                            )
                            data_manager.set_teleop_state(hand, False, None, None)
                        elif data_manager.prepare_teleop_baseline(hand, generation, held_target):
                            prev_agilex_generation = generation
                    target_joint_angles = data_manager.get_ready_teleop_target(hand)
                    if target_joint_angles is None:
                        should_stream_joints = False
                        prev_target_joint_angles = None
                if target_joint_angles is not None and should_stream_joints:
                    # Apply speed scaling: interpolate between previous and new target
                    if SPEED_SCALE < 1.0 and prev_target_joint_angles is not None:
                        target_joint_angles = (
                            prev_target_joint_angles
                            + SPEED_SCALE * (target_joint_angles - prev_target_joint_angles)
                        )
                    prev_target_joint_angles = target_joint_angles.copy()
                    if USE_AGILEX_TELEOP and teleop_active and robot_activity_state == RobotActivityState.ENABLED:
                        robot_controller.set_target_joint_angles(
                            target_joint_angles,
                            use_agilex_teleop=True,
                            teleop_generation=prev_agilex_generation,
                        )
                    else:
                        robot_controller.set_target_joint_angles(target_joint_angles)
                    # Detect joint-limit clamping (helps explain "ghost moves but arm won't").
                    requested_target = target_joint_angles.copy()
                    try:
                        clamped_target = robot_controller.get_target_joint_angles()
                        # Prefer a delta the controller measured itself. Under
                        # --robot-process the set and the read-back cross different
                        # transports, so differencing them here reports one tick of
                        # motion as clamping -- and the write-back below would then
                        # feed that stale value into the next tick's interpolation.
                        measured_clamp = getattr(
                            robot_controller, "get_target_clamp_deltas", None
                        )
                        clamp_diff_deg = (
                            measured_clamp() if measured_clamp is not None else None
                        )
                        if clamp_diff_deg is None and measured_clamp is None:
                            clamp_diff_deg = np.abs(clamped_target - requested_target)
                        if clamp_diff_deg is None:
                            clamp_err_deg = 0.0
                        else:
                            clamp_err_deg = float(np.max(clamp_diff_deg))
                    except Exception:
                        clamped_target = None
                        clamp_diff_deg = None
                        clamp_err_deg = 0.0
                    if clamp_diff_deg is None:
                        clamp_diff_deg = np.zeros(6, dtype=np.float64)

                    if clamped_target is not None and clamp_err_deg > 1e-9:
                        data_manager.set_target_joint_angles(clamped_target, hand)
                        target_joint_angles = clamped_target
                        prev_target_joint_angles = clamped_target.copy()

                    if (
                        clamp_err_deg > 0.5
                        and (iteration_start - last_clamp_warn_ts) > 1.0
                    ):
                        last_clamp_warn_ts = iteration_start
                        clamp_details = _format_clamp_details(
                            requested_target, clamped_target, clamp_diff_deg
                        )
                        log.warning(
                            "Joint targets are being clamped by up to "
                            f"{clamp_err_deg:.1f}° ({hand}). "
                            f"{clamp_details}. "
                            "If motion stops near limits, this is expected. "
                            "If it happens far from limits, verify SDK joint limits units (deg vs rad)."
                        )
                        robot_controller.log_hardware_diagnostics(
                            f"joint target clamp ({hand})"
                        )

                # Gripper updates only during teleop (out of scope for brain control).
                if (
                    target_joint_angles is not None
                    and grip_value is not None
                    and teleop_active
                    and robot_activity_state == RobotActivityState.ENABLED
                ):
                    settings = data_manager.get_teleop_gripper_settings(hand)
                    open_limit = float(np.clip(settings.open_limit, 0.0, 1.0))
                    mode = settings.mode
                    grip = float(np.clip(float(grip_value), 0.0, 1.0))
                    grip_pressed = grip >= _GRIPPER_TOGGLE_PRESS_THRESHOLD
                    mode_changed = mode != prev_gripper_mode

                    if mode == "toggle":
                        if toggle_gripper_open is None or mode_changed:
                            existing_target = data_manager.get_target_gripper_open_value(
                                hand
                            )
                            existing = (
                                existing_target
                                if existing_target is not None
                                else gripper_open_value
                            )
                            toggle_gripper_open = (
                                bool(existing is not None)
                                and float(existing) > max(open_limit * 0.5, 1e-6)
                            )
                        if (
                            prev_teleop_active
                            and not mode_changed
                            and grip_pressed
                            and not prev_grip_pressed
                        ):
                            toggle_gripper_open = not bool(toggle_gripper_open)
                        target_gripper_open_value = (
                            open_limit if toggle_gripper_open else 0.0
                        )
                    else:
                        target_gripper_open_value = open_limit * (1.0 - grip)
                        toggle_gripper_open = target_gripper_open_value > (
                            open_limit * 0.5
                        )

                    data_manager.set_target_gripper_open_value(
                        target_gripper_open_value, hand
                    )
                    robot_controller.set_gripper_open_value(target_gripper_open_value)
                    prev_grip_pressed = grip_pressed
                    prev_gripper_mode = mode

                # --- Clean action telemetry for FUTURE LeRobot datasets --------
                # ADDITIVE ONLY: nothing here changes what is sent to the arm.
                # Emits three per-arm signals the exporter prefers when present
                # (old data lacks them and exports exactly as before):
                #   * arm_driven                : explicit per-arm driven flag.
                #   * joint_command_positions   : the IK target re-slewed at the
                #     FIXED loop rate -> glitch-free (fixes Defect 1: raw pre-guard
                #     IK glitches that leak into joint_target_positions when a VR
                #     tracking stall inflates the IK-thread's variable-dt clamp).
                #   * gripper_command_open_amount : the commanded gripper when
                #     driven, else the MEASURED (held) position -> a real position,
                #     never a stale 0 (fixes Defect 2: idle gripper reads closed).
                driven = bool(should_stream_joints)
                measured_j = (
                    np.asarray(current_joint_angles, dtype=np.float64).reshape(-1)[:6]
                    if current_joint_angles is not None
                    else None
                )
                if driven and target_joint_angles is not None:
                    tgt_j = np.asarray(
                        target_joint_angles, dtype=np.float64
                    ).reshape(-1)[:6]
                    anchor = (
                        last_logged_command
                        if last_logged_command is not None
                        else (measured_j if measured_j is not None else tgt_j)
                    )
                    cmd_j = anchor + np.clip(
                        tgt_j - anchor, -max_cmd_step_deg, max_cmd_step_deg
                    )
                else:
                    # Idle: command == measured -> action == state (matches the
                    # joints' harmless idle-hold and re-anchors the slew on resume).
                    cmd_j = measured_j
                # Log every tick, like joint_positions -- the BigQuery sink applies
                # the single per-signal rate cap (--gcp-telemetry-hz). No client
                # throttle here (see the note by last_logged_command above).
                if cmd_j is not None:
                    last_logged_command = cmd_j
                    data_manager.log_telemetry_signal(
                        "log_joint_command_positions", cmd_j.copy(), hand
                    )
                data_manager.log_telemetry_signal(
                    "log_arm_driven", 1.0 if driven else 0.0, hand
                )
                if driven and grip_value is not None:
                    _tgt_g = data_manager.get_target_gripper_open_value(hand)
                    grip_cmd = (
                        float(_tgt_g)
                        if _tgt_g is not None
                        else (
                            float(gripper_open_value)
                            if gripper_open_value is not None
                            else None
                        )
                    )
                else:
                    grip_cmd = (
                        float(gripper_open_value)
                        if gripper_open_value is not None
                        else None
                    )
                if grip_cmd is not None:
                    data_manager.log_telemetry_signal(
                        "log_gripper_command_open_amount", grip_cmd, hand
                    )

            # Per-hand control source (operator / preset / policy / ...), emitted
            # EVERY tick and OUTSIDE the state branches above. It used to sit inside
            # the ENABLED/PAUSED arm, so the one state it exists to record --
            # POLICY_CONTROLLED -- never produced a row: the 14 policy runs of
            # 2026-09-07 reached BigQuery with no activity_mode at all, the episode
            # exporter's _policy_session() found no code 7, and its fail-closed drop
            # of policy-commanded frames therefore never fired.
            # arm_driven and joint_command_positions stay inside that arm on purpose:
            # a policy logs its own policy_command_positions, and the OPERATOR's
            # command signal must not be synthesised while the operator is not
            # commanding the arm.
            data_manager.log_telemetry_signal(
                "log_activity_mode",
                float(
                    activity_mode_code(robot_activity_state, teleop_active, driven)
                ),
                hand,
            )

            # Sleep to maintain streaming rate
            elapsed = time.time() - iteration_start
            if diagnostics_enabled:
                diag_now = time.time()
                diag_loops += 1
                diag_max_loop_ms = max(diag_max_loop_ms, elapsed * 1000.0)
                if elapsed > dt:
                    diag_overruns += 1
                if diag_now - diag_last_ts >= 5.0:
                    motor_avg = (
                        diag_motor_poll_ms / diag_motor_poll_count
                        if diag_motor_poll_count
                        else 0.0
                    )
                    fault_avg = (
                        diag_fault_poll_ms / diag_fault_poll_count
                        if diag_fault_poll_count
                        else 0.0
                    )
                    log.debug(
                        f"joint diagnostics ({hand}): "
                        f"recording={data_manager.is_recording_active()} "
                        f"loops={diag_loops} overruns={diag_overruns} "
                        f"max_loop={diag_max_loop_ms:.1f}ms "
                        f"motor_poll_avg={motor_avg:.1f}ms/{diag_motor_poll_count} "
                        f"fault_poll_avg={fault_avg:.1f}ms/{diag_fault_poll_count}"
                    )
                    diag_last_ts = diag_now
                    diag_loops = 0
                    diag_overruns = 0
                    diag_max_loop_ms = 0.0
                    diag_motor_poll_ms = 0.0
                    diag_motor_poll_count = 0
                    diag_fault_poll_ms = 0.0
                    diag_fault_poll_count = 0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            prev_teleop_active = teleop_active

    except Exception as e:
        log.error("joint state thread error (%s): %s", hand, e, exc_info=True)
        if error_callback is not None:
            try:
                error_callback(e)
            except Exception:
                pass
        if stop_event is not None:
            stop_event.set()
        else:
            data_manager.request_shutdown()
    finally:
        log.info(f"Joint state thread stopped ({hand})")
