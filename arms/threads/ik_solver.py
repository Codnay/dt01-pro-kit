"""IK solver thread - solves IK and updates state."""

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path to import pink_ik_solver
from arms.configs import (
    CONTROLLER_DATA_RATE,
    IK_SOLVER_RATE,
    J6_JOYSTICK_DEADZONE,
    J6_JOYSTICK_SPEED,
    MAX_JOINT_SLEW_DEG_S,
    ROTATION_SCALE,
    TELEOP_CONTROLLER_JUMP_REJECT_ENABLED,
    TELEOP_CONTROLLER_MAX_SPEED_M_S,
    TELEOP_JOINT_SLEW_CLAMP_ENABLED,
    TELEOP_ACCEL_LIMIT_ENABLED,
    TELEOP_MAX_JOINT_ACCEL_DEG_S2,
    TELEOP_MAX_TRACKING_ERROR_DEG,
    TRANSLATION_SCALE,
    USE_AGILEX_TELEOP,
)
from arms.data_manager import DataManager, RobotActivityState
from arms.logging_setup import get_logger
from arms.joint_motion import JointAccelerationLimiter, limit_tracking_step
from arms.utils import limit_joint_step, scale_and_add_delta_transform

from arms.pink_ik_solver import PinkIKSolver
from arms.joint_limits import clamp_recovery_step, joint_limit_details, recovery_joint_bounds

log = get_logger("ik")

_TWO_PI = 2.0 * np.pi
_ROTATION_UNWRAP_EPS = 1e-12


def _tracking_error_violation(
    target_deg: np.ndarray, measured_deg: np.ndarray, limits_deg: np.ndarray
) -> tuple[int, float, float] | None:
    """Return the worst joint's index, error and limit if teleop must hold."""
    errors = np.abs(np.asarray(target_deg) - np.asarray(measured_deg))
    # A nonfinite angle must fail closed even if other joints are within limits.
    ratios = np.where(np.isfinite(errors), errors / limits_deg, np.inf)
    joint = int(np.argmax(ratios))
    if not np.isfinite(errors[joint]) or errors[joint] > limits_deg[joint]:
        return joint, float(errors[joint]), float(limits_deg[joint])
    return None


def _closest_equivalent_rotvec(
    candidate: np.ndarray, previous_rotvec: np.ndarray
) -> np.ndarray:
    """Shift a rotvec candidate by full windings to be closest to previous."""
    candidate = np.asarray(candidate, dtype=np.float64).reshape(3)
    previous_rotvec = np.asarray(previous_rotvec, dtype=np.float64).reshape(3)
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm < _ROTATION_UNWRAP_EPS:
        return candidate.copy()

    axis = candidate / candidate_norm
    previous_angle_on_axis = float(np.dot(previous_rotvec, axis))
    windings = round((previous_angle_on_axis - candidate_norm) / _TWO_PI)
    return candidate + (_TWO_PI * windings) * axis


def _select_continuous_rotvec(
    rotvec: np.ndarray, previous_rotvec: np.ndarray | None
) -> np.ndarray:
    """Choose the equivalent rotvec representation closest to the last cycle."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    if previous_rotvec is None:
        return rotvec.copy()

    previous_rotvec = np.asarray(previous_rotvec, dtype=np.float64).reshape(3)
    rotvec_norm = float(np.linalg.norm(rotvec))
    previous_norm = float(np.linalg.norm(previous_rotvec))

    if rotvec_norm < _ROTATION_UNWRAP_EPS:
        if previous_norm < _ROTATION_UNWRAP_EPS:
            return rotvec.copy()
        previous_axis = previous_rotvec / previous_norm
        windings = round(previous_norm / _TWO_PI)
        return previous_axis * (_TWO_PI * windings)

    # Same rotation, represented with the opposite-side winding. This avoids
    # SciPy's principal rotvec flipping sign at +/-pi.
    rotvec_alt = rotvec * (1.0 - (_TWO_PI / rotvec_norm))
    candidates = (
        _closest_equivalent_rotvec(rotvec, previous_rotvec),
        _closest_equivalent_rotvec(rotvec_alt, previous_rotvec),
    )
    return min(
        candidates,
        key=lambda candidate: float(np.linalg.norm(candidate - previous_rotvec)),
    ).copy()


def ik_solver_thread(
    data_manager: DataManager,
    ik_solver: PinkIKSolver,
    hand: str = "right",
    stop_event: threading.Event | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> None:
    """IK solver thread - solves IK and updates state."""
    log.info(f"IK solver thread started ({hand})")
    diagnostics_enabled = os.getenv(
        "PIPER_RECORDING_DIAGNOSTICS", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    dt: float = 1.0 / IK_SOLVER_RATE
    BRAIN_GAINS = np.array([1.0, 1.0, 1.0, 0.35, 0.35, 0.35], dtype=np.float64)
    MAX_CONSECUTIVE_IK_FAILURES = 10
    consecutive_ik_failures: int = 0
    rpy_offset_deg_cached = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    frame_correction = np.eye(3, dtype=np.float64)
    use_frame_correction = False
    diag_last_ts = time.time()
    diag_loops = 0
    diag_overruns = 0
    diag_max_loop_ms = 0.0
    previous_delta_rotvec: np.ndarray | None = None
    previous_controller_initial: np.ndarray | None = None
    previous_teleop_generation: int | None = None
    teleop_reference_deg: np.ndarray | None = None
    agilex_ik = None
    agilex_robot_anchor = None
    agilex_controller_anchor = None
    agilex_clutch_baseline = None
    if USE_AGILEX_TELEOP:
        from arms.agilex_teleop import AgilexTeleopIK
        agilex_ik = AgilexTeleopIK(ik_solver)
        teleop_soft_limits = ik_solver.get_joint_position_limits_deg()
        teleop_hard_limits = ik_solver.get_joint_position_limits_deg(hard=True)
    previous_controller_position: np.ndarray | None = None
    prev_iteration_ts: float | None = None

    # Teleop safety guards (prevent VR-glitch -> arm disarm/drop).
    max_joint_slew_deg_s = np.asarray(
        MAX_JOINT_SLEW_DEG_S, dtype=np.float64
    ).reshape(-1)[:6]
    max_tracking_error_deg = np.asarray(
        TELEOP_MAX_TRACKING_ERROR_DEG, dtype=np.float64
    )
    motion_limiter = JointAccelerationLimiter(
        max_joint_slew_deg_s, TELEOP_MAX_JOINT_ACCEL_DEG_S2
    )
    # Controller samples arrive at CONTROLLER_DATA_RATE; a teleport is a jump
    # between consecutive samples, so size the gate to the sample interval.
    controller_sample_dt = 1.0 / CONTROLLER_DATA_RATE
    max_controller_jump_m = TELEOP_CONTROLLER_MAX_SPEED_M_S * controller_sample_dt
    warn_last_ts: dict[str, float] = {}

    def _warn_throttled(key: str, message: str, period_s: float = 2.0) -> None:
        now = time.time()
        if now - warn_last_ts.get(key, 0.0) >= period_s:
            warn_last_ts[key] = now
            log.warning(message)

    def _active_mask(active_group: str) -> np.ndarray:
        mask = np.zeros(6, dtype=bool)
        if active_group == "all":
            mask[:] = True
        elif active_group == "J1-3":
            mask[:3] = True
        elif active_group == "J4-6":
            mask[3:] = True
        elif active_group in ("J1", "J2", "J3", "J4", "J5", "J6"):
            idx = int(active_group[1:]) - 1
            if 0 <= idx < 6:
                mask[idx] = True
        else:
            mask[:] = True
        return mask

    def _clamp_joint_angles_deg(joint_angles_deg: np.ndarray) -> np.ndarray:
        clamp_fn = getattr(ik_solver, "clamp_joint_angles_deg", None)
        if clamp_fn is None:
            return np.asarray(joint_angles_deg, dtype=np.float64).copy()
        return clamp_fn(np.asarray(joint_angles_deg, dtype=np.float64))

    def _clamp_teleop_step(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if USE_AGILEX_TELEOP:
            return clamp_recovery_step(
                reference, target, teleop_soft_limits, teleop_hard_limits
            )
        return _clamp_joint_angles_deg(target)

    def _log_reengagement(generation: int, status: int, held: np.ndarray) -> None:
        # One event keeps the hold and limits associated with the same clutch.
        data_manager.log_telemetry_signal(
            "log_teleop_reengagement_event",
            np.concatenate(
                (
                    [generation, status],
                    held,
                    teleop_soft_limits.ravel(),
                    teleop_hard_limits.ravel(),
                )
            ),
            hand,
        )

    def _joint_target_baseline_deg(
        current_joint_angles: np.ndarray | None,
        current_ik_joint_angles: np.ndarray,
    ) -> np.ndarray:
        target = data_manager.get_target_joint_angles(hand)
        if target is not None:
            return np.asarray(target, dtype=np.float64).reshape(-1)[:6].copy()
        if current_joint_angles is not None:
            return np.asarray(current_joint_angles, dtype=np.float64).reshape(-1)[:6].copy()
        return np.asarray(current_ik_joint_angles, dtype=np.float64).reshape(-1)[:6].copy()

    try:
        while not data_manager.is_shutdown_requested() and not (
            stop_event is not None and stop_event.is_set()
        ):
            iteration_start: float = time.time()
            iteration_monotonic = time.monotonic()

            # Actual elapsed time since the previous iteration. The slew clamp
            # uses this (not the nominal 1/IK_SOLVER_RATE) so its budget and the
            # logged rate reflect real wall-clock time -- correct under loop
            # overruns instead of attributing a long tick's motion to a 4 ms tick.
            # Clamped to a sane range so a first tick or a long pause cannot
            # produce a pathological budget.
            loop_dt = (
                dt
                if prev_iteration_ts is None
                else float(np.clip(iteration_monotonic - prev_iteration_ts, 1e-4, 0.1))
            )
            prev_iteration_ts = iteration_monotonic

            # Get current robot joint angles from state
            current_joint_angles = data_manager.get_current_joint_angles(hand)
            # Compute ACTUAL FK from feedback. The active teleop solve below
            # restores the accepted desired reference after publishing this FK;
            # restarting each solve from feedback injects encoder steps and
            # actuator lag into the commanded trajectory.
            if current_joint_angles is not None:
                ik_solver.set_configuration_no_task_update(np.radians(current_joint_angles))
            current_ik_joint_angles = np.degrees(ik_solver.get_current_configuration())

            # Get current end effector pose from IK solver and set in state
            ik_ee_pose = ik_solver.get_current_end_effector_pose()
            data_manager.set_current_end_effector_pose(ik_ee_pose, hand)

            # Get current state
            robot_activity_state = data_manager.get_robot_activity_state(hand)
            controller_transform, _, _ = data_manager.get_controller_data(hand)
            teleop_active = data_manager.get_teleop_active(hand)
            if not teleop_active:
                # Clutch released. Drop the jump-gate baseline so that
                # re-engaging teleop -- where the hand has usually moved while
                # declutched, and the clutch re-anchors controller_initial anyway
                # -- does not read the first sample as a glitch and reject it.
                previous_controller_position = None
            brain = data_manager.get_brain_control(hand)
            rpy_offset_deg = data_manager.get_teleop_delta_rpy_offset_deg(hand)
            if not np.allclose(rpy_offset_deg, rpy_offset_deg_cached):
                rpy_offset_deg_cached = rpy_offset_deg
                previous_delta_rotvec = None
                frame_correction = Rotation.from_euler(
                    "xyz", rpy_offset_deg_cached, degrees=True
                ).as_matrix()
                use_frame_correction = not np.allclose(rpy_offset_deg_cached, 0.0)
                if use_frame_correction:
                    log.info(
                        "Teleop delta RPY correction set "
                        f"({hand}): roll={rpy_offset_deg_cached[0]:.1f} deg, "
                        f"pitch={rpy_offset_deg_cached[1]:.1f} deg, "
                        f"yaw={rpy_offset_deg_cached[2]:.1f} deg"
                    )
                else:
                    log.info(f"Teleop delta RPY correction disabled ({hand})")

            joystick_y = data_manager.get_joystick_y(hand)
            j6_joystick_active = (
                robot_activity_state == RobotActivityState.ENABLED
                and abs(joystick_y) > J6_JOYSTICK_DEADZONE
            )
            if not (
                teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
                and controller_transform is not None
                and not j6_joystick_active
            ):
                teleop_reference_deg = None

            # Skip teleop-based IK if another command source owns joint targets.
            # We only update IK solver configuration to keep it in sync, but don't override targets.
            if j6_joystick_active:
                previous_delta_rotvec = None
                previous_controller_initial = None
                consecutive_ik_failures = 0

                target = _joint_target_baseline_deg(
                    current_joint_angles, current_ik_joint_angles
                )
                target[5] += joystick_y * J6_JOYSTICK_SPEED * dt
                target = _clamp_joint_angles_deg(target)

                data_manager.set_target_joint_angles(target, hand)
                data_manager.set_ik_solve_time_ms(0.0, hand)
                data_manager.set_ik_success(True, hand)
                ik_solver.set_configuration_no_task_update(np.radians(target))

                accepted_fk_pose = ik_solver.get_current_end_effector_pose()
                data_manager.set_target_pose(accepted_fk_pose, hand)

                if teleop_active and controller_transform is not None:
                    controller_anchor = np.asarray(
                        controller_transform, dtype=np.float64
                    ).copy()
                    previous_controller_position = controller_anchor[:3, 3].copy()
                    data_manager.set_teleop_state(
                        hand, True, controller_anchor, accepted_fk_pose
                    )

            elif robot_activity_state in (
                RobotActivityState.POLICY_CONTROLLED,
                RobotActivityState.PRESET_CONTROLLED,
            ):
                previous_delta_rotvec = None
                previous_controller_initial = None
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    current_end_effector_pose = (
                        data_manager.get_current_end_effector_pose(hand)
                    )
                    data_manager.set_target_pose(current_end_effector_pose, hand)
                    data_manager.set_ik_solve_time_ms(0.0, hand)
                    data_manager.set_ik_success(True, hand)

            # Brain control: joint-level nudges when teleop is inactive.
            elif (
                brain.armed
                and (not teleop_active)
                and robot_activity_state
                not in (
                    RobotActivityState.DISABLED,
                    RobotActivityState.HOMING,
                    RobotActivityState.POLICY_CONTROLLED,
                    RobotActivityState.PRESET_CONTROLLED,
                )
            ):
                previous_delta_rotvec = None
                previous_controller_initial = None
                baseline = brain.baseline_deg
                if baseline is None:
                    if current_joint_angles is not None:
                        baseline = np.array(current_joint_angles, dtype=np.float64)
                    else:
                        baseline = np.array(current_ik_joint_angles, dtype=np.float64)
                    baseline = baseline.reshape(-1)[:6]
                    data_manager.set_brain_control_baseline_deg(baseline, hand)

                slider = np.array(brain.slider_norm, dtype=np.float64).reshape(-1)[:6]
                slider = np.clip(slider, -1.0, 1.0)
                diameter = float(max(0.0, brain.diameter))
                offset_deg = slider * diameter * BRAIN_GAINS

                active = _active_mask(brain.active_group)
                clamp_mode = brain.clamp_mode
                if clamp_mode == "soft_damp":
                    instant = baseline.copy()
                    instant[active] = baseline[active] + offset_deg[active]
                else:
                    instant = baseline + offset_deg

                last_cmd = brain.last_commanded_deg
                if last_cmd is None:
                    last_cmd = baseline.copy()

                selected_alpha = float(np.clip(brain.selected_alpha, 0.0, 1.0))
                unselected_alpha = float(np.clip(brain.unselected_alpha, 0.0, 1.0))
                alpha = np.where(active, selected_alpha, unselected_alpha).astype(
                    np.float64
                )
                cmd = last_cmd + alpha * (instant - last_cmd)
                cmd = _clamp_joint_angles_deg(cmd)

                data_manager.set_target_joint_angles(cmd, hand)
                data_manager.set_ik_solve_time_ms(0.0, hand)
                data_manager.set_ik_success(True, hand)
                data_manager.set_brain_control_last_commanded_deg(cmd, hand)

                # Keep visualization target frame sensible (hold current EE pose).
                current_end_effector_pose = data_manager.get_current_end_effector_pose(
                    hand
                )
                data_manager.set_target_pose(current_end_effector_pose, hand)

                # Sync IK solver internal configuration for consistent FK.
                try:
                    ik_solver.set_configuration_no_task_update(np.radians(cmd))
                except Exception:
                    ik_solver.set_configuration(np.radians(cmd))

            elif (
                teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
                and controller_transform is not None
            ):
                # Guard 1: reject teleported controller poses (VR tracking
                # glitches). A jump larger than a human can move between samples
                # is dropped -- hold the previous target instead of slewing the
                # arm toward a garbage goal. The pre-jump reference is kept so a
                # sustained teleport stays rejected until tracking returns.
                controller_position = np.asarray(
                    controller_transform[:3, 3], dtype=np.float64
                )
                if (
                    TELEOP_CONTROLLER_JUMP_REJECT_ENABLED
                    and previous_controller_position is not None
                ):
                    jump_m = float(
                        np.linalg.norm(
                            controller_position - previous_controller_position
                        )
                    )
                    if jump_m > max_controller_jump_m:
                        _warn_throttled(
                            f"ctrl_jump_{hand}",
                            f"controller pose jump rejected ({hand}): "
                            f"{jump_m * 100.0:.1f} cm between samples "
                            f"(> {TELEOP_CONTROLLER_MAX_SPEED_M_S:.1f} m/s); "
                            "holding target",
                        )
                        elapsed = time.time() - iteration_start
                        sleep_time = dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue
                previous_controller_position = controller_position

                clutch_active, clutch_generation, controller_initial, robot_initial = (
                    data_manager.get_teleop_snapshot(hand)
                )
                if (
                    not clutch_active
                    or controller_initial is None
                    or robot_initial is None
                ):
                    teleop_reference_deg = None
                    previous_delta_rotvec = None
                    previous_controller_initial = None
                    elapsed = time.time() - iteration_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # Calculate delta transform
                delta_position = controller_transform[:3, 3] - controller_initial[:3, 3]
                delta_orientation = (
                    controller_transform[:3, :3] @ controller_initial[:3, :3].T
                )
                reanchored = (
                    previous_teleop_generation != clutch_generation
                    or previous_controller_initial is None
                )
                if reanchored and USE_AGILEX_TELEOP:
                    agilex_clutch_baseline = data_manager.get_teleop_baseline(hand, clutch_generation)
                    if agilex_clutch_baseline is None:
                        time.sleep(max(0.0, dt - (time.time() - iteration_start)))
                        continue
                    try:
                        recovery_joint_bounds(
                            agilex_clutch_baseline,
                            teleop_soft_limits,
                            teleop_hard_limits,
                        )
                    except ValueError as exc:
                        _log_reengagement(clutch_generation, 2, agilex_clutch_baseline)
                        data_manager.set_teleop_state(hand, False, None, None)
                        data_manager.set_ik_success(False, hand)
                        log.warning(
                            "Teleop re-engagement refused (%s, clutch=%d): %s",
                            hand,
                            clutch_generation,
                            exc,
                        )
                        continue
                    details = joint_limit_details(
                        agilex_clutch_baseline, teleop_soft_limits
                    )
                    _log_reengagement(
                        clutch_generation, 1 if details else 0, agilex_clutch_baseline
                    )
                    if details:
                        log.warning(
                            "Teleop soft-limit recovery (%s, clutch=%d): %s; "
                            "holding unchanged, operator movement may recover inward only",
                            hand,
                            clutch_generation,
                            details,
                        )
                if reanchored:
                    previous_delta_rotvec = None
                    previous_controller_initial = controller_initial.copy()
                    previous_teleop_generation = clutch_generation
                    teleop_reference_deg = None
                    if USE_AGILEX_TELEOP:
                        # Anchor to the acknowledged held command, so a physical
                        # tracking offset cannot ratchet the target on each press.
                        agilex_robot_anchor = ik_solver.get_pose_for_configuration_deg(
                            agilex_clutch_baseline
                        )
                        agilex_controller_anchor = controller_transform.copy()

                if USE_AGILEX_TELEOP:
                    robot_initial = agilex_robot_anchor
                    delta_position = controller_transform[:3, 3] - agilex_controller_anchor[:3, 3]
                    delta_orientation = controller_transform[:3, :3] @ agilex_controller_anchor[:3, :3].T

                if current_joint_angles is None:
                    # A desired reference must never be seeded without feedback.
                    data_manager.set_teleop_state(hand, False, None, None)
                    data_manager.set_ik_success(False, hand)
                    teleop_reference_deg = None
                    log.warning("Teleop has no joint feedback; deactivating (%s)", hand)
                    time.sleep(max(0.0, dt - (time.time() - iteration_start)))
                    continue

                if teleop_reference_deg is None:
                    teleop_reference_deg = np.asarray(
                        agilex_clutch_baseline if USE_AGILEX_TELEOP else current_joint_angles,
                        dtype=np.float64
                    ).copy()
                    motion_limiter.reset(teleop_reference_deg)
                else:
                    # The joint-state worker may have corrected a hardware joint
                    # limit. Respect that accepted target on the next solve.
                    accepted = data_manager.get_target_joint_angles(hand)
                    if accepted is not None:
                        teleop_reference_deg = np.asarray(
                            accepted, dtype=np.float64
                        ).copy()
                        if not np.array_equal(motion_limiter.position, teleop_reference_deg):
                            motion_limiter.reset(teleop_reference_deg)

                tracking_violation = _tracking_error_violation(
                    teleop_reference_deg, current_joint_angles, 2.0 * max_tracking_error_deg
                )
                if tracking_violation is not None:
                    joint, tracking_error, tracking_limit = tracking_violation
                    data_manager.set_teleop_state(hand, False, None, None)
                    if np.all(np.isfinite(current_joint_angles)):
                        data_manager.set_target_joint_angles(current_joint_angles, hand)
                    data_manager.set_ik_success(False, hand)
                    teleop_reference_deg = None
                    log.warning(
                        "Teleop J%d tracking fault: %.2f deg exceeds %.2f deg "
                        "(2x tracking limit); "
                        "deactivating and holding (%s). Release and re-engage the trigger.",
                        joint + 1, tracking_error, tracking_limit, hand,
                    )
                    time.sleep(max(0.0, dt - (time.time() - iteration_start)))
                    continue

                if use_frame_correction:
                    # Convert controller deltas into the arm's teleop reference frame.
                    delta_position = frame_correction @ delta_position
                    delta_orientation = (
                        frame_correction @ delta_orientation @ frame_correction.T
                    )

                raw_delta_rotvec = Rotation.from_matrix(delta_orientation).as_rotvec()
                delta_rotvec = _select_continuous_rotvec(
                    raw_delta_rotvec, previous_delta_rotvec
                )
                previous_delta_rotvec = delta_rotvec.copy()

                T_robot_target = scale_and_add_delta_transform(
                    delta_position,
                    delta_orientation,
                    TRANSLATION_SCALE,
                    ROTATION_SCALE,
                    robot_initial,
                    delta_rotvec=delta_rotvec,
                )

                ik_solver.set_target_pose(T_robot_target[:3, 3], T_robot_target[:3, :3])
                data_manager.set_target_pose(T_robot_target, hand)

                # Advance the continuous desired configuration, independently of
                # small feedback oscillations. Measured FK above stays measured.
                ik_solver.set_configuration_no_task_update(
                    np.radians(teleop_reference_deg)
                )
                agilex_solve_ms = None
                if agilex_ik is not None:
                    solve_started = time.monotonic()
                    try:
                        candidate = (
                            teleop_reference_deg.copy()
                            if reanchored
                            else agilex_ik.solve(
                                T_robot_target,
                                teleop_reference_deg,
                                bounds_deg=recovery_joint_bounds(
                                    teleop_reference_deg,
                                    teleop_soft_limits,
                                    teleop_hard_limits,
                                ),
                            )
                        )
                    except Exception as exc:
                        candidate = None
                        _warn_throttled("agilex_ik", f"AgileX teleop IK failed ({hand}): {exc}")
                    agilex_solve_ms = (time.monotonic() - solve_started) * 1000.0
                    success = candidate is not None
                    if success:
                        ik_solver.set_configuration_no_task_update(np.radians(candidate))
                else:
                    success = ik_solver.solve_ik()

                if success:
                    consecutive_ik_failures = 0
                    # Pink's configuration setter clips to its normal soft
                    # bounds. Never round-trip an AgileX recovery through it.
                    raw_joint_config = (
                        np.asarray(candidate, dtype=np.float64)
                        if agilex_ik is not None
                        else np.degrees(ik_solver.get_current_configuration())
                    )
                    joint_config = _clamp_teleop_step(
                        raw_joint_config, teleop_reference_deg
                    )
                    if not np.allclose(joint_config, raw_joint_config):
                        ik_solver.set_configuration_no_task_update(
                            np.radians(joint_config)
                        )
                    # Bound the destination before acceleration shaping so it
                    # can brake toward the tracking boundary. Keep clutch active.
                    tracking_binding = None
                    if np.all(np.isfinite(joint_config)):
                        joint_config, _, tracking_binding = limit_tracking_step(
                            teleop_reference_deg, joint_config,
                            current_joint_angles, max_tracking_error_deg,
                        )
                    # Guard 2: hardware-safe joint slew clamp. Rate-limit the
                    # commanded setpoint relative to the last command so a bad or
                    # unreachable IK target cannot advance the joints faster than
                    # the firmware tolerates -- which would trip a clean motor
                    # disarm and drop the arm. Uses the same limit_joint_step()
                    # limiter as the preset/trajectory path, and references the
                    # last committed target so it stays continuous across
                    # teleop/brain/home modes.
                    if TELEOP_JOINT_SLEW_CLAMP_ENABLED or USE_AGILEX_TELEOP:
                        last_target = teleop_reference_deg
                        slewed, was_limited = limit_joint_step(
                            last_target, joint_config, max_joint_slew_deg_s, loop_dt
                        )
                        if was_limited:
                            # Report the binding joint (largest requested-vs-cap
                            # ratio) as requested deg/s vs its cap, measured over
                            # the ACTUAL loop interval (loop_dt) so the logged rate
                            # is the true joint speed, not a nominal-period
                            # artifact inflated by loop overruns.
                            requested_rate = np.abs(joint_config - last_target) / loop_dt
                            binding = int(np.argmax(requested_rate / max_joint_slew_deg_s))
                            _warn_throttled(
                                f"slew_{hand}",
                                f"joint slew clamp active ({hand}): J{binding + 1} "
                                f"requested {requested_rate[binding]:.0f} deg/s, "
                                f"capped to {max_joint_slew_deg_s[binding]:.0f} deg/s",
                            )
                            joint_config = slewed

                    if (TELEOP_ACCEL_LIMIT_ENABLED or USE_AGILEX_TELEOP) and np.all(np.isfinite(joint_config)):
                        # Use the same monotonic timestep as the slew guard;
                        # nominal dt would distort velocity on uneven ticks.
                        # loop_dt already caps long-pause integration at 100 ms.
                        shaped = motion_limiter.step(joint_config, loop_dt)
                        joint_config = _clamp_teleop_step(shaped, teleop_reference_deg)
                        if not np.array_equal(shaped, joint_config):
                            motion_limiter.reset(joint_config)

                    # Release/re-engagement can use IDENTICAL anchors. Its
                    # generation still changes, so discard the old solve.
                    latest_active, latest_generation, _, _ = (
                        data_manager.get_teleop_snapshot(hand)
                    )
                    if (
                        not latest_active
                        or latest_generation != clutch_generation
                        or data_manager.get_robot_activity_state(hand)
                        != RobotActivityState.ENABLED
                    ):
                        teleop_reference_deg = None
                        time.sleep(max(0.0, dt - (time.time() - iteration_start)))
                        continue

                    if not np.all(np.isfinite(joint_config)):
                        data_manager.set_teleop_state(hand, False, None, None)
                        data_manager.set_target_joint_angles(current_joint_angles, hand)
                        data_manager.set_ik_success(False, hand)
                        teleop_reference_deg = None
                        log.warning(
                            "Nonfinite teleop joint target; "
                            "deactivating and holding (%s). Release and re-engage the trigger.",
                            hand,
                        )
                        time.sleep(max(0.0, dt - (time.time() - iteration_start)))
                        continue

                    # Momentum from the acceleration filter must not carry a
                    # target outside the envelope. Scale the full step, then
                    # retain the velocity of the accepted step in the filter.
                    # A moving feedback boundary can override acceleration;
                    # reference containment has priority in that case.
                    joint_config, tracking_scale, final_binding = limit_tracking_step(
                        teleop_reference_deg, joint_config,
                        current_joint_angles, max_tracking_error_deg,
                    )
                    if tracking_scale < 1.0:
                        motion_limiter.position = joint_config.copy()
                        motion_limiter.velocity = (joint_config - teleop_reference_deg) / loop_dt
                        tracking_binding = final_binding
                    if tracking_binding is not None:
                        _warn_throttled(
                            f"tracking_{hand}",
                            f"Teleop tracking limit active ({hand}): J{tracking_binding + 1} "
                            f"limit {max_tracking_error_deg[tracking_binding]:.2f} deg; "
                            "target constrained, teleop remains engaged",
                        )

                    teleop_reference_deg = joint_config.copy()
                    ik_solver.set_configuration_no_task_update(
                        np.radians(teleop_reference_deg)
                    )

                    stats = ik_solver.get_statistics()
                    solve_time_ms = stats["last_solve_time_ms"]
                    if agilex_solve_ms is not None:
                        solve_time_ms = agilex_solve_ms

                    if USE_AGILEX_TELEOP:
                        data_manager.commit_teleop_target(hand, clutch_generation, joint_config)
                    else:
                        data_manager.set_target_joint_angles(joint_config, hand)
                    data_manager.set_ik_solve_time_ms(solve_time_ms, hand)
                    data_manager.set_ik_success(success, hand)
                else:
                    motion_limiter.reset(teleop_reference_deg)
                    consecutive_ik_failures += 1
                    data_manager.set_ik_solve_time_ms(0.0, hand)
                    data_manager.set_ik_success(False, hand)

                    if consecutive_ik_failures >= MAX_CONSECUTIVE_IK_FAILURES:
                        data_manager.set_teleop_state(hand, False, None, None)
                        consecutive_ik_failures = 0
                        log.warning(
                            f"IK failed {MAX_CONSECUTIVE_IK_FAILURES} consecutive times, "
                            f"deactivating teleop ({hand}). "
                            "Target pose may be unreachable -- reduce scale or reposition."
                        )

            else:  # robot is HOMING or DISABLED
                previous_delta_rotvec = None
                previous_controller_initial = None
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    current_end_effector_pose = (
                        data_manager.get_current_end_effector_pose(hand)
                    )
                    data_manager.set_target_pose(current_end_effector_pose, hand)
                    data_manager.set_target_joint_angles(current_joint_angles, hand)
                    data_manager.set_ik_solve_time_ms(0.0, hand)
                    data_manager.set_ik_success(True, hand)

            # Sleep to maintain loop rate
            elapsed = time.time() - iteration_start
            if diagnostics_enabled:
                diag_now = time.time()
                diag_loops += 1
                diag_max_loop_ms = max(diag_max_loop_ms, elapsed * 1000.0)
                if elapsed > dt:
                    diag_overruns += 1
                if diag_now - diag_last_ts >= 5.0:
                    log.debug(
                        f"IK diagnostics ({hand}): "
                        f"recording={data_manager.is_recording_active()} "
                        f"loops={diag_loops} overruns={diag_overruns} "
                        f"max_loop={diag_max_loop_ms:.1f}ms"
                    )
                    diag_last_ts = diag_now
                    diag_loops = 0
                    diag_overruns = 0
                    diag_max_loop_ms = 0.0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        log.error("IK solver thread error (%s): %s", hand, e, exc_info=True)
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
        log.info(f"IK solver thread stopped ({hand})")
