"""Bimanual teleop runtime for the operator stack.

This module owns robot lifecycle, Quest input, IK, joint-state forwarding, and
the viser UI. It deliberately uses a teleop-scoped stop event so runtime
failures do not tear down service-owned camera/dashboard processes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from arms.camera_preview import CameraPreview
from arms.configs import (
    ARM_CONFIG_LOADED,
    ARM_CONFIG_PATH,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    DRIVER_TEMP_PAUSE_THRESHOLD_C,
    DRIVER_TEMP_REARM_THRESHOLD_C,
    FRAME_TASK_GAIN,
    GRIPPER_FRAME_NAME,
    HOME_SPEED_PERCENT,
    IK_SOLVER_RATE,
    JOINT_LIMIT_MARGIN_DEG_BY_JOINT,
    LM_DAMPING,
    MAX_JOINT_SLEW_DEG_S,
    MOTOR_SPEED_RAW_BY_HAND,
    MOTOR_TEMP_PAUSE_THRESHOLD_C,
    MOTOR_TEMP_REARM_THRESHOLD_C,
    NEUTRAL_JOINT_ANGLES_LEFT,
    NEUTRAL_JOINT_ANGLES_RIGHT,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TEMPERATURE_WATCHDOG_ENABLED,
    TEMPERATURE_WATCHDOG_POLL_PERIOD_S,
    TRANSLATION_SCALE,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from arms.data_manager import DataManager, RobotActivityState
from arms.logging_setup import get_logger

log = get_logger("teleop")


def _fmt_vec(values: Any, places: int) -> str:
    """Render a numeric vector compactly for one log line: ``[1.0, 2.0]``."""
    try:
        items = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]
    except Exception:  # noqa: BLE001 - a log line must never raise
        return str(values)
    return "[" + ", ".join(f"{v:.{places}f}" for v in items) + "]"


def _inc_operator_command(*_a: Any, **_k: Any) -> None:
    return None


from arms.operator_services import OperatorServices
# This kit ships no preset dashboard; the hooks below stay inert.
ArmPresetDashboardController = Any

# Operating-mode codes: the table lives in common.data_manager (shared with the
# per-hand "activity_mode" signal the joint-state loop emits every tick). This
# runtime still emits its stack-wide summary, under "stack_activity_mode".
from arms.data_manager import ACTIVITY_MODE_CODES  # noqa: E402


@dataclass
class BimanualTeleopRuntimeConfig:
    """Configuration for the robot teleop runtime."""

    active_hands: tuple[str, ...] = ("right", "left")
    can_right: str = "can_right"
    can_left: str = "can_left"
    robot_process: bool = False
    ip_address: str | None = None
    quest_source: str = "local"
    # LiveKit control transport (quest_source == "livekit"). Empty strings fall
    # back to LIVEKIT_* env / .env resolution in livekit_bridge.
    livekit_url: str = ""
    livekit_room: str = ""
    livekit_identity: str = ""
    livekit_connect_timeout: float = 15.0
    livekit_stats_sec: float = 5.0
    livekit_input_timeout: float = 0.25
    enable_visualizer: bool = True
    show_camera_preview: bool = False
    preview_camera_name: str = "rgb"
    preset_dashboard_url: str | None = None


class BimanualTeleopRuntime:
    """Own robot teleop lifecycle while services remain independent."""

    def __init__(
        self,
        *,
        data_manager: DataManager,
        config: BimanualTeleopRuntimeConfig,
        services: OperatorServices | None = None,
        preset_dashboard_controller: ArmPresetDashboardController | None = None,
        episode_controller: Any | None = None,
    ) -> None:
        self.data_manager = data_manager
        self.config = config
        self.services = services
        self.preset_dashboard_controller = preset_dashboard_controller
        self.episode_controller = episode_controller
        self.stop_event = threading.Event()
        self.thread_errors: list[str] = []
        self._last_rj_press_ts = 0.0
        self._rj_debounce_s = 0.75

        self.active_hands = tuple(
            h for h in config.active_hands if h in ("right", "left")
        )
        if not self.active_hands:
            raise ValueError("BimanualTeleopRuntime needs at least one active hand")
        self.inactive_hands = tuple(
            h for h in ("right", "left") if h not in self.active_hands
        )
        self.primary_hand = (
            "right" if "right" in self.active_hands else self.active_hands[0]
        )
        self.arm_label = (
            "both arms" if len(self.active_hands) == 2 else f"{self.primary_hand} arm"
        )
        self.can_interface_by_hand = {
            "right": config.can_right,
            "left": config.can_left,
        }
        self.neutral_joint_angles_by_hand = {
            "right": NEUTRAL_JOINT_ANGLES_RIGHT,
            "left": NEUTRAL_JOINT_ANGLES_LEFT,
        }
        self.robot_controllers: dict[str, Any] = {}
        self.joint_state_threads: list[threading.Thread] = []
        self.quest_thread: threading.Thread | None = None
        self.quest_reader: Any | None = None
        self.ik_solvers: dict[str, Any] = {}
        self.dashboard_ik_solvers: dict[str, Any] = {}
        self.ik_threads: list[threading.Thread] = []
        self.temperature_watchdog_thread: threading.Thread | None = None
        self._over_temp_latched = False
        self.visualizer: Any | None = None
        self.camera_preview: CameraPreview | None = None
        self.secondary_status_controls: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        """Initialize hardware, threads, callbacks, and viser."""
        self.stop_event.clear()
        self.data_manager.set_controller_filter_params(
            CONTROLLER_MIN_CUTOFF,
            CONTROLLER_BETA,
            CONTROLLER_D_CUTOFF,
        )
        self._create_robot_controllers()
        self._apply_arm_config_to_controllers()
        self._start_robot_control_loops()
        self._start_joint_state_threads()
        self._start_temperature_watchdog()
        self._start_quest_reader()
        self._create_ik_solvers()
        self._start_ik_threads()
        self._attach_dashboard_controllers()
        self._setup_visualizer()
        self._set_active_activity(RobotActivityState.DISABLED)
        self._set_inactive_disabled()
        self._clear_active_teleop()
        if self.visualizer is not None:
            self.visualizer.update_toggle_robot_enabled_status(False)
        self._print_operator_controls()

    def _run_headless_loop(self) -> None:
        """Keep-alive main loop when the visualizer is disabled.

        Teleop runs entirely on the background threads (quest reader IK         control); here we just stay alive until stop/shutdown. Teleop filter/pink/frame params use their
        configured defaults (no GUI to read live values from).
        """
        log.info("Running headless teleop loop. Ctrl+C to stop.")
        dt = 1.0 / VISUALIZATION_RATE
        while (
            not self.stop_event.is_set()
            and not self.data_manager.is_shutdown_requested()
        ):
            start = time.time()
            sleep_t = dt - (time.time() - start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def run_visualizer_loop(self) -> None:
        """Run the blocking viser/camera-preview update loop."""
        # Headless first: this MUST come before importing brain_control_integration,
        # which pulls in robot_visualizer → yourdfpy → trimesh (whose native libs
        # are incompatible in-process with Agora's RTM libs and would segfault).
        if self.visualizer is None:
            self._run_headless_loop()
            return

        from arms.brain_control_integration import sync_brain_control_from_gui

        dt = 1.0 / VISUALIZATION_RATE
        prev_brain_armed_by_hand: dict[str, bool] = {"right": False, "left": False}
        prev_activity_mode = -1
        last_activity_mode_emit_s = 0.0

        while (
            not self.stop_event.is_set()
            and not self.data_manager.is_shutdown_requested()
        ):
            iteration_start = time.time()
            min_cutoff, beta, d_cutoff = self.visualizer.get_controller_filter_params()
            self.data_manager.set_controller_filter_params(min_cutoff, beta, d_cutoff)

            pink_params = self.visualizer.get_pink_parameters()
            for solver in self.ik_solvers.values():
                solver.update_task_parameters(**pink_params)

            for hand in self.active_hands:
                self.data_manager.set_teleop_delta_rpy_offset_deg(
                    self.visualizer.get_teleop_frame_offset_rpy_deg(key=hand),
                    hand=hand,
                )
                prev_brain_armed_by_hand[hand] = sync_brain_control_from_gui(
                    data_manager=self.data_manager,
                    visualizer=self.visualizer,
                    hand=hand,
                    key=hand,
                    prev_armed=prev_brain_armed_by_hand[hand],
                )

            self._update_primary_visualizer()
            self._update_secondary_visualizer()

            # Emit the operating mode (teleop/preset/trajectory/homing/idle/…)
            # as a telemetry signal -> BigQuery, on change or every 0.5 s, so
            # logs can be chunked by mode. Cheap; sink/episode-gated downstream.
            mode_code = self._current_activity_mode()
            if (
                mode_code != prev_activity_mode
                or (iteration_start - last_activity_mode_emit_s) >= 0.5
            ):
                self.data_manager.log_telemetry_signal(
                    "log_stack_activity_mode", float(mode_code)
                )
                prev_activity_mode = mode_code
                last_activity_mode_emit_s = iteration_start
            if self.camera_preview is not None:
                camera_frame = self.data_manager.get_camera_frame(
                    self.config.preview_camera_name
                )
                self.camera_preview.update(
                    camera_frame.rgb_image,
                    camera_frame.depth_image_mm,
                )
            sleep_time = dt - (time.time() - iteration_start)
            if sleep_time > 0:
                self.stop_event.wait(sleep_time)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop teleop-owned threads and robot resources."""
        self.stop_event.set()
        if self.preset_dashboard_controller is not None:
            if self.preset_dashboard_controller.is_recording():
                try:
                    log.info("Saving active dashboard trajectory recording...")
                    self.preset_dashboard_controller.stop_recording()
                except Exception as exc:
                    log.warning(f"Failed to save active dashboard recording: {exc}")
            try:
                self.preset_dashboard_controller.stop_motion()
                self.preset_dashboard_controller.wait_for_idle(timeout_s=2.0)
            except Exception:
                pass
        self._detach_dashboard_controllers()
        self._set_active_activity(RobotActivityState.DISABLED)
        self._clear_active_teleop()

        if self.quest_reader is not None:
            try:
                self.quest_reader.stop()
            except Exception:
                pass
        if self.quest_thread is not None:
            self.quest_thread.join(timeout=timeout_s)
            self.quest_thread = None
        for thread in self.ik_threads:
            thread.join(timeout=timeout_s)
        self.ik_threads = []
        for thread in self.joint_state_threads:
            thread.join(timeout=timeout_s)
        self.joint_state_threads = []
        if self.temperature_watchdog_thread is not None:
            self.temperature_watchdog_thread.join(timeout=timeout_s)
            self.temperature_watchdog_thread = None
        if self.camera_preview is not None:
            self.camera_preview.close()
            self.camera_preview = None
        for controller in self.robot_controllers.values():
            try:
                controller.cleanup()
            except Exception as exc:
                log.warning(f"Controller cleanup failed: {exc}")
        self.robot_controllers = {}
        if self.visualizer is not None:
            try:
                self.visualizer.stop()
            except Exception:
                pass
            self.visualizer = None

    def _on_thread_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.thread_errors.append(message)
        self.stop_event.set()

    def _create_robot_controllers(self) -> None:
        from arms.piper_controller import PiperController

        for hand in self.active_hands:
            can_interface = self.can_interface_by_hand[hand]
            kwargs = dict(
                can_interface=can_interface,
                robot_rate=ROBOT_RATE,
                control_mode=PiperController.ControlMode.JOINT_SPACE,
                neutral_joint_angles=self.neutral_joint_angles_by_hand[hand],
                debug_mode=False,
                joint_limit_margin_deg=JOINT_LIMIT_MARGIN_DEG_BY_JOINT,
            )
            if self.config.robot_process:
                # Move the SDK's CAN threads + the control loop into their own
                # interpreter so they stop competing with IK for this one's GIL
                # (see common/perf/piper_process for the measurements).
                from arms.perf.piper_process import PiperControllerProxy

                log.info(
                    f"Initializing {hand} arm controller ({can_interface}) "
                    "in a separate process..."
                )
                self.robot_controllers[hand] = PiperControllerProxy(**kwargs)
            else:
                log.info(f"Initializing {hand} arm controller ({can_interface})...")
                self.robot_controllers[hand] = PiperController(**kwargs)

    def _apply_arm_config_to_controllers(self) -> None:
        """Apply per-deployment settings from ``config/configs.py``.

        Homing pose, mount orientation and the motion scales are plain Python
        values, already overlaid onto :mod:`common.configs` at import. Two
        things need the live controller instead, and are done here.
        """
        for controller in self.robot_controllers.values():
            try:
                controller.set_home_speed_percent(HOME_SPEED_PERCENT)
            except Exception as exc:
                log.warning(f"Could not set homing speed: {exc}")
        self._apply_motor_speed_caps()
        self._log_arm_config()

    def _log_arm_config(self) -> None:
        """Log the arm settings actually in effect for this session.

        Worth the handful of lines at startup: these values decide how the arms
        feel, they differ per cell, and one of them (the motor speed ceiling)
        lives in motor firmware where nothing else would ever show it. A session
        that feels wrong should be diagnosable from the log it already wrote,
        without anyone having to go read a file or query the SDK by hand.

        The motor ceiling is READ BACK from the arm rather than echoed from the
        config, so this reports what the hardware will actually do -- including
        the case where the write did not take. This read is the SINGLE authority
        on that number: it both prints the value and decides whether to warn, so
        the log can never carry a warning that disagrees with the value printed
        beside it.
        """
        source = (
            ARM_CONFIG_PATH
            if ARM_CONFIG_LOADED
            else f"shipped defaults (not loaded: {ARM_CONFIG_PATH})"
        )
        log.info("arm config: source=%s", source)
        log.info(
            "arm config: translation_scale=%.2f  rotation_scale=%.2f  "
            "home_speed=%d%%  slew_ceiling=%s deg/s",
            TRANSLATION_SCALE,
            ROTATION_SCALE,
            HOME_SPEED_PERCENT,
            _fmt_vec(MAX_JOINT_SLEW_DEG_S, 0),
        )
        for hand in self.active_hands:
            controller = self.robot_controllers.get(hand)
            raw = self._read_motor_cap(controller)
            log.info(
                "arm config (%s): home_pose=%s deg  mount_rpy=%s deg  " "motor_cap=%s",
                hand,
                _fmt_vec(self.neutral_joint_angles_by_hand[hand], 1),
                # From the DataManager, not the config constant: that is the
                # live value the IK thread actually reads each tick.
                _fmt_vec(self.data_manager.get_teleop_delta_rpy_offset_deg(hand), 1),
                self._describe_motor_cap(raw),
            )
            self._warn_if_motor_cap_differs(hand, raw)

    @staticmethod
    def _read_motor_cap(controller: Any) -> np.ndarray | None:
        """The arm's live per-motor speed ceiling, or None if it cannot be read.

        None is a distinct answer from any vector of numbers and must stay that
        way: "we could not ask the arm" has to never be reported as "the arm
        holds the configured value".
        """
        if controller is None:
            return None
        try:
            return np.asarray(
                controller.get_joint_velocity_limits_raw(refresh=True), dtype=np.int64
            ).reshape(-1)[:6]
        except Exception:
            return None

    @staticmethod
    def _describe_motor_cap(raw: np.ndarray | None) -> str:
        """Render the live ceiling as 'raw (deg/s)', or say it is unknown."""
        if raw is None:
            return "unknown (could not read it back)"
        deg_s = np.degrees(raw.astype(np.float64) * 0.001)
        if len(set(raw.tolist())) == 1:
            # The usual case: one value across all six. Say it once.
            return f"{int(raw[0])} raw ({deg_s[0]:.1f} deg/s)"
        return f"{_fmt_vec(raw, 0)} raw ({_fmt_vec(deg_s, 1)} deg/s)"

    @staticmethod
    def _warn_if_motor_cap_differs(hand: str, raw: np.ndarray | None) -> None:
        """Warn only when the authoritative read-back disagrees with the config.

        Deliberately driven by the same read the banner just printed, so the two
        can never contradict each other. A speed ceiling nobody can trust is
        worse than no report at all -- this is the number that silently decides
        how fast the arm moves.
        """
        wanted = MOTOR_SPEED_RAW_BY_HAND.get(hand)
        if not wanted:
            return  # nothing configured; whatever the arm holds is by definition it
        if raw is None:
            log.warning(
                "Could not read back the %s arm's motor speed cap. It may not "
                "match configs.py (wanted %s raw) -- treat the speed "
                "ceiling above as unverified.",
                hand,
                list(wanted),
            )
            return
        differs = {
            f"J{joint + 1}": (int(raw[joint]), int(wanted[joint]))
            for joint in range(min(len(raw), len(wanted)))
            if int(raw[joint]) != int(wanted[joint])
        }
        if differs:
            # Carefully NOT "the arm kept its old ceiling". The SDK's speed
            # feedback is a struct filled from per-motor CAN replies and it has
            # no freshness stamp: a reply that never arrived leaves the previous
            # value in place, and a refresh returns it looking perfectly valid.
            # A startup once reported J5 stuck on both arms that a later read
            # showed had applied all along. So this says what is true -- the
            # value could not be confirmed -- and points at the live check,
            # rather than asserting a hardware state it cannot actually see.
            log.warning(
                "Could not confirm the %s arm's motor speed cap matches "
                "configs.py (read back, wanted): %s. Either the write did "
                "not take, or the SDK is still serving a stale feedback frame "
                "for those joints. Check the live value with "
                "'curl localhost:8082/api/motor-speed-limits'.",
                hand,
                differs,
            )

    def _apply_motor_speed_caps(self) -> None:
        """Re-assert the firmware per-motor speed ceiling on every start.

        ``MotorMaxSpdSet`` writes the motor controller's non-volatile memory: it
        survives a stack restart AND a power cycle. So an arm silently keeps
        whatever ceiling the last thing to touch it left behind, and nothing in
        a log or a config file says what that is -- which is exactly how a cell
        ends up teleoperating at a fraction of its speed with no visible cause.
        Asserting the configured value here makes the arm's speed a property of
        this deployment's config file rather than of its history.

        Writes are verified and retried because the SDK call reports success
        whether or not the firmware took the value; only a read-back proves it.

        This method deliberately does NOT pass judgement on the result. The
        verdict belongs to :meth:`_log_arm_config`, which re-reads a moment
        later and is the single authority on what the arm actually holds --
        because the speed feedback lags the write, an "it did not apply" decided
        here could contradict the value reported there, and two conflicting
        numbers for the arm's speed ceiling are worse than a slow startup.
        """
        for hand, controller in self.robot_controllers.items():
            desired = MOTOR_SPEED_RAW_BY_HAND.get(hand)
            if not desired:
                continue  # no configured value: leave the motors as they are
            try:
                self._assert_motor_speeds(hand, controller, list(desired))
            except Exception as exc:
                # Never block startup on this, and keep it at debug: a mismatch
                # that is still a mismatch by the time the banner reads it gets
                # the loud warning there, once, with the authoritative numbers.
                log.debug(
                    "motor speed cap on the %s arm not confirmed yet (%s); "
                    "the startup banner read-back decides",
                    hand,
                    exc,
                )

    def _assert_motor_speeds(
        self, hand: str, controller: Any, desired: list[int]
    ) -> None:
        """Write and verify the six per-motor speed ceilings for one arm.

        The thing that makes this awkward is the READ side, not the write side.
        ``GetAllMotorAngleLimitMaxSpd`` returns an SDK struct that is filled
        progressively from per-motor CAN frames, so a refresh issued just after
        a write routinely reports the value from *before* it. Treating that as a
        failed write is wrong twice over: it writes again unnecessarily, and if
        it happens to be the last joint checked it reports a failure for a
        setting that did in fact take -- which is exactly what a startup log
        showed, warning that J4 was still 600 immediately before the banner read
        back 450 on all six.

        So: write the joints that differ, then POLL for the whole vector to
        agree, and only call it stuck once the feedback has had time to catch up
        and still disagrees.
        """
        want = np.asarray(desired, dtype=np.int64).reshape(-1)[:6]
        rounds = 2
        settle_s = 3.0
        # Deliberately unhurried. Each read is a SearchAllMotorMaxAngleSpd
        # broadcast that asks all six motors to reply, so polling it hard puts
        # dozens of frames a second on a bus that is also carrying the control
        # loop -- and a reply that gets lost in that traffic leaves the SDK
        # serving its previous value, which is the very staleness this loop is
        # trying to see through. Reading gently is part of reading correctly.
        read_every_s = 0.5
        writes = 0

        def read() -> np.ndarray:
            return np.asarray(
                controller.get_joint_velocity_limits_raw(refresh=True), dtype=np.int64
            ).reshape(-1)[:6]

        def settle() -> np.ndarray:
            """Re-read until the feedback matches, or the window runs out."""
            deadline = time.monotonic() + settle_s
            time.sleep(read_every_s)  # let the write's reply frame land first
            current = read()
            while bool(np.any(current != want)) and time.monotonic() < deadline:
                time.sleep(read_every_s)
                current = read()
            return current

        current = read()
        for _ in range(rounds):
            pending = [j for j in range(6) if int(current[j]) != int(want[j])]
            if not pending:
                break
            for joint in pending:
                controller.set_joint_velocity_limit_raw(joint + 1, int(want[joint]))
                writes += 1
                time.sleep(0.05)  # don't stack frames back to back on the bus
            current = settle()

        stuck = {
            f"J{joint + 1}": (int(current[joint]), int(want[joint]))
            for joint in range(6)
            if int(current[joint]) != int(want[joint])
        }
        if stuck:
            raise RuntimeError(
                f"after {rounds} rounds and a {settle_s:g}s settle these still "
                f"differ (actual, wanted): {stuck}"
            )
        if writes:
            log.info(
                f"Motor speed cap applied to the {hand} arm ({writes} writes): "
                f"{desired}"
            )
        else:
            log.info(f"Motor speed cap on the {hand} arm already correct: {desired}")

    def _start_robot_control_loops(self) -> None:
        log.info("Starting robot control loops...")
        for controller in self.robot_controllers.values():
            controller.start_control_loop()

    def _start_joint_state_threads(self) -> None:
        from arms.threads.joint_state import joint_state_thread

        log.info("Starting joint state threads...")
        for hand, controller in self.robot_controllers.items():
            thread = threading.Thread(
                target=joint_state_thread,
                args=(self.data_manager, controller, hand),
                kwargs={
                    "stop_event": self.stop_event,
                    "error_callback": self._on_thread_error,
                },
                daemon=True,
                name=f"joint-state-{hand}",
            )
            thread.start()
            self.joint_state_threads.append(thread)

    def _register_quest_callbacks(self, event_names: list[str], callback: Any) -> None:
        if self.quest_reader is None:
            return
        # Count every operator (Meta Quest) command for Grafana visibility. The
        # reader fires one alias per press, so wrapping the callback counts once
        # per actual press, labelled by the canonical event name.
        command_label = event_names[0] if event_names else "unknown"

        def _counted(*args: Any, **kwargs: Any) -> Any:
            _inc_operator_command(command_label)
            return callback(*args, **kwargs)

        for event_name in event_names:
            try:
                self.quest_reader.on(event_name, _counted)
            except Exception:
                continue

    def _start_quest_reader(self) -> None:
        from arms.threads.quest_reader import quest_reader_thread

        if self.config.quest_source == "livekit":
            # Read control DIRECTLY in this process over LiveKit data channels.
            # No bridge, no UDP hop: LiveKit's native libs load alongside
            # viser/trimesh/pinocchio, so the visualizer can stay on.
            from arms.livekit_rt.livekit_quest_reader import (
                LiveKitQuestReader,
            )

            log.info("Initializing LiveKit Quest reader (in-process data channels)...")
            self.quest_reader = LiveKitQuestReader(
                url=self.config.livekit_url,
                room=self.config.livekit_room,
                identity=self.config.livekit_identity,
                connect_timeout=self.config.livekit_connect_timeout,
                stats_period_s=self.config.livekit_stats_sec,
                input_timeout_s=self.config.livekit_input_timeout,
            )
        else:
            from meta_quest_teleop.reader import MetaQuestReader

            log.info("Initializing Meta Quest reader...")
            self.quest_reader = MetaQuestReader(
                ip_address=self.config.ip_address,
                port=5555,
                run=True,
            )
        self._register_quest_callbacks(
            ["button_a_pressed", "button_a", "a_pressed"],
            self.on_button_a_pressed,
        )
        self._register_quest_callbacks(
            ["button_b_pressed", "button_b", "b_pressed"],
            self.on_button_b_pressed,
        )
        self._register_quest_callbacks(
            ["button_x_pressed", "button_x", "x_pressed"],
            self.on_button_x_pressed,
        )
        self._register_quest_callbacks(
            ["button_y_pressed", "button_y", "y_pressed"],
            self.on_button_y_pressed,
        )
        if self.episode_controller is not None:
            # The RJ button drives a shared GCP episode (telemetry + RGBD video).
            self._register_quest_callbacks(
                ["button_rj_pressed", "button_rj", "rj_pressed"],
                lambda: self._on_button_rj_pressed(
                    lambda: self.episode_controller.toggle("Button RJ")
                ),
            )

        log.info("Starting quest reader thread...")
        self.quest_thread = threading.Thread(
            target=quest_reader_thread,
            args=(self.data_manager, self.quest_reader, self.active_hands),
            kwargs={
                "stop_event": self.stop_event,
                "error_callback": self._on_thread_error,
            },
            daemon=True,
            name="quest-reader",
        )
        self.quest_thread.start()

    def _on_button_rj_pressed(self, toggle_recording: Callable[[], None]) -> None:
        """Toggle recording without letting the stick press nudge J6."""
        now = time.monotonic()
        if now - self._last_rj_press_ts < self._rj_debounce_s:
            log.warning("Ignoring duplicate RJ recording toggle")
            return
        self._last_rj_press_ts = now
        suppress = getattr(self.data_manager, "suppress_joystick_y", None)
        if callable(suppress):
            suppress("right", duration_s=0.35)
        toggle_recording()

    def _initial_configuration_from_robot(
        self,
        controller: Any,
        neutral_joint_angles_deg: list[float],
    ) -> np.ndarray:
        angles_deg = controller.get_current_joint_angles()
        if angles_deg is None:
            return np.radians(neutral_joint_angles_deg)
        return np.radians(angles_deg)

    def _create_ik_solver_for_hand(self, hand: str, solver_cls: Any) -> Any:
        controller = self.robot_controllers[hand]
        return solver_cls(
            urdf_path=URDF_PATH,
            end_effector_frame=GRIPPER_FRAME_NAME,
            solver_name=SOLVER_NAME,
            position_cost=POSITION_COST,
            orientation_cost=ORIENTATION_COST,
            frame_task_gain=FRAME_TASK_GAIN,
            lm_damping=LM_DAMPING,
            damping_cost=DAMPING_COST,
            solver_damping_value=SOLVER_DAMPING_VALUE,
            integration_time_step=1 / IK_SOLVER_RATE,
            initial_configuration=self._initial_configuration_from_robot(
                controller,
                self.neutral_joint_angles_by_hand[hand],
            ),
            posture_cost_vector=np.array(POSTURE_COST_VECTOR),
            joint_position_limits_deg=controller.get_effective_joint_limits_deg(),
            joint_velocity_limits_deg_s=np.asarray(
                MAX_JOINT_SLEW_DEG_S, dtype=np.float64
            ),
        )

    def _create_ik_solvers(self) -> None:
        from arms.pink_ik_solver import PinkIKSolver

        log.info(f"Creating Pink IK solver(s) ({', '.join(self.active_hands)})...")
        for hand in self.active_hands:
            self.ik_solvers[hand] = self._create_ik_solver_for_hand(hand, PinkIKSolver)
            self.dashboard_ik_solvers[hand] = self._create_ik_solver_for_hand(
                hand, PinkIKSolver
            )

    def _start_ik_threads(self) -> None:
        from arms.threads.ik_solver import ik_solver_thread

        log.info("Starting IK solver threads...")
        for hand, solver in self.ik_solvers.items():
            thread = threading.Thread(
                target=ik_solver_thread,
                args=(self.data_manager, solver, hand),
                kwargs={
                    "stop_event": self.stop_event,
                    "error_callback": self._on_thread_error,
                },
                daemon=True,
                name=f"ik-solver-{hand}",
            )
            thread.start()
            self.ik_threads.append(thread)

    def _attach_dashboard_controllers(self) -> None:
        if self.services is not None:
            self.services.attach_robot_controllers(
                self.robot_controllers,
                active_hands=self.active_hands,
                dashboard_ik_solvers=self.dashboard_ik_solvers,
            )
        elif self.preset_dashboard_controller is not None:
            self.preset_dashboard_controller.attach_robot_controllers(
                self.robot_controllers,
                active_hands=self.active_hands,
                dashboard_ik_solvers=self.dashboard_ik_solvers,
            )

    def _detach_dashboard_controllers(self) -> None:
        if self.services is not None:
            self.services.detach_robot_controllers()
        elif self.preset_dashboard_controller is not None:
            self.preset_dashboard_controller.detach_robot_controllers()

    def _setup_visualizer(self) -> None:
        if not self.config.enable_visualizer:
            # Headless: do NOT import the visualizer (viser/trimesh). Teleop
            # runs on the background threads; the main thread uses a keep-alive
            # loop.
            log.info("Visualization disabled (--no-visualizer); running headless.")
            self.visualizer = None
            return

        from arms.brain_control_integration import register_brain_control_callbacks
        from arms.robot_visualizer import RobotVisualizer

        log.info("Starting visualization...")
        self.visualizer = RobotVisualizer(urdf_path=URDF_PATH)
        self.visualizer.add_basic_controls()
        self.visualizer.add_teleop_controls()
        self.visualizer.add_gripper_status_controls()
        self.visualizer.add_homing_controls()
        self.visualizer.add_toggle_robot_enabled_status_button()
        self.visualizer.add_controller_filter_controls(
            initial_min_cutoff=CONTROLLER_MIN_CUTOFF,
            initial_beta=CONTROLLER_BETA,
            initial_d_cutoff=CONTROLLER_D_CUTOFF,
        )
        self.visualizer.add_scaling_controls(
            initial_translation_scale=TRANSLATION_SCALE,
            initial_rotation_scale=ROTATION_SCALE,
        )
        for hand in self.active_hands:
            self.visualizer.add_teleop_frame_controls(
                key=hand,
                label_prefix=hand.capitalize(),
                initial_rpy_deg=self.data_manager.get_teleop_delta_rpy_offset_deg(hand),
            )
        self.visualizer.add_pink_parameter_controls(
            position_cost=POSITION_COST,
            orientation_cost=ORIENTATION_COST,
            frame_task_gain=FRAME_TASK_GAIN,
            lm_damping=LM_DAMPING,
            damping_cost=DAMPING_COST,
            solver_damping_value=SOLVER_DAMPING_VALUE,
            posture_cost_vector=POSTURE_COST_VECTOR,
        )
        self.visualizer.add_controller_visualization()
        self.visualizer.add_target_frame_visualization()
        for hand in self.active_hands:
            self.visualizer.add_brain_control_controls(
                key=hand,
                label_prefix=hand.capitalize(),
            )

        self.secondary_status_controls = {}
        for hand in self.active_hands:
            if hand == self.primary_hand:
                continue
            label = hand.capitalize()
            self.secondary_status_controls[hand] = {
                "robot_status": self.visualizer.server.gui.add_text(
                    f"{label} Arm Status",
                    "Initializing...",
                ),
                "teleop_status": self.visualizer.server.gui.add_text(
                    f"{label} Teleop",
                    "Inactive",
                ),
                "joint_angles": self.visualizer.server.gui.add_text(
                    f"{label} Joint Angles",
                    "Waiting...",
                ),
                "grip_value": self.visualizer.server.gui.add_number(
                    f"{label} Grip Value",
                    0.0,
                    disabled=True,
                ),
                "trigger_value": self.visualizer.server.gui.add_number(
                    f"{label} Trigger Value",
                    0.0,
                    disabled=True,
                ),
            }
        if self.config.preset_dashboard_url is not None:
            self.visualizer.server.gui.add_text(
                "Preset Dashboard URL",
                self.config.preset_dashboard_url,
            )
        self.visualizer.set_toggle_robot_enabled_status_callback(
            self.on_button_a_pressed
        )
        self.visualizer.set_go_home_callback(self.on_button_b_pressed)
        for hand in self.active_hands:
            register_brain_control_callbacks(
                data_manager=self.data_manager,
                visualizer=self.visualizer,
                hand=hand,
                key=hand,
            )

        self.camera_preview = CameraPreview(
            enabled=self.config.show_camera_preview,
            window_name=f"RealSense - {self.config.preview_camera_name} - Example 7",
        )
        if self.camera_preview.enabled:
            log.info("Live camera preview enabled (OpenCV window)")

    def _print_operator_controls(self) -> None:
        parts = [
            f"Starting {self.arm_label} teleoperation with REAL ROBOT CONTROL...",
            "CONTROLS:",
            f"   1. Press BUTTON A to enable/pause (hold) {self.arm_label}",
        ]
        for idx, hand in enumerate(self.active_hands, start=2):
            parts.append(
                f"   {idx}. Hold {hand.upper()} GRIP to teleop {hand.upper()} arm"
            )
        parts.append("   Use the active hand TRIGGER to close that arm's gripper")
        parts.append(f"   Press BUTTON X to send {self.arm_label} home")
        if self.config.preset_dashboard_url is not None:
            parts.append(f"   Preset dashboard: {self.config.preset_dashboard_url}")
            parts.append(
                "   Policy runs start ONLY from the dashboard's Policy panel; "
                "squeeze a trigger to take over (arms hold, teleop arms on that "
                "squeeze); Button A / STOP also ends a run"
            )
        if self.episode_controller is not None:
            parts.append(
                "   6. Press RIGHT JOYSTICK to start/stop GCP recording "
                "(telemetry + RGBD)"
            )
        parts.append("Press Ctrl+C to exit")
        log.info("\n".join(parts))

    def _set_active_activity(self, state: RobotActivityState) -> None:
        for hand in self.active_hands:
            self.data_manager.set_robot_activity_state(state, hand)

    def _set_inactive_disabled(self) -> None:
        for hand in self.inactive_hands:
            self.data_manager.set_robot_activity_state(
                RobotActivityState.DISABLED, hand
            )
            self.data_manager.set_teleop_state(hand, False, None, None)

    def _clear_active_teleop(self) -> None:
        for hand in self.active_hands:
            self.data_manager.set_teleop_state(hand, False, None, None)

    def _pause_and_hold_active(self, reason: str) -> bool:
        """Pause + hold all arms (no torque cut), latch state to PAUSED.

        Shared by the Button A/GUI toggle and the temperature watchdog. Stops
        teleop driving, commands each controller to freeze at its measured pose
        (motors stay powered), and sets the active hands to ``PAUSED`` so IK/
        teleop cannot overwrite the hold and the arms do not re-arm on their own
        — an operator must re-enable. Returns True if every arm held.
        """
        self._clear_active_teleop()
        hold_results = {
            hand: controller.hold_current_position()
            for hand, controller in self.robot_controllers.items()
        }
        if not all(hold_results.values()):
            failed = [hand for hand, ok in hold_results.items() if not ok]
            log.warning(f"Failed to hold arm(s) {failed} ({reason})")
            return False
        self._set_active_activity(RobotActivityState.PAUSED)
        if self.visualizer is not None:
            self.visualizer.update_toggle_robot_enabled_status(False)
        log.info(f"{self.arm_label.capitalize()} paused + holding ({reason})")
        return True

    def _start_temperature_watchdog(self) -> None:
        """Start the motor over-temperature auto-pause watchdog thread."""
        if not TEMPERATURE_WATCHDOG_ENABLED:
            return
        thread = threading.Thread(
            target=self._temperature_watchdog_loop,
            name="temperature_watchdog",
            daemon=True,
        )
        self.temperature_watchdog_thread = thread
        thread.start()

    def _temperature_watchdog_loop(self) -> None:
        """Poll per-joint motor/driver temps; auto-pause on a breach."""
        period = max(0.1, TEMPERATURE_WATCHDOG_POLL_PERIOD_S)
        while (
            not self.stop_event.is_set()
            and not self.data_manager.is_shutdown_requested()
        ):
            try:
                self._check_temperatures_once()
            except Exception as exc:  # never let the safety watchdog die
                log.error(f"Temperature watchdog error: {exc}")
            # Interruptible sleep so stop() joins promptly.
            self.stop_event.wait(period)

    def _check_temperatures_once(self) -> None:
        """Pause + hold the arms if any active-arm joint is over temperature.

        Hysteresis: an arm trips (pauses) when a joint reaches the PAUSE limit
        and only clears the latch once every joint cools below the (lower) REARM
        limit, so a manual re-enable sticks instead of instantly re-pausing.
        Re-enabling while a joint is still above the PAUSE limit re-pauses by
        design — the operator must let it cool below REARM first.
        """
        any_enabled = any(
            self.data_manager.get_robot_activity_state(hand)
            == RobotActivityState.ENABLED
            for hand in self.active_hands
        )

        breaches: list[str] = []
        all_recovered = True
        have_reading = False
        for hand, controller in self.robot_controllers.items():
            temps = controller.get_motor_temperatures(non_blocking=True)
            if temps is None:
                continue
            have_reading = True
            for label, values, trip, rearm in (
                (
                    "motor",
                    temps.get("motor_temperature"),
                    MOTOR_TEMP_PAUSE_THRESHOLD_C,
                    MOTOR_TEMP_REARM_THRESHOLD_C,
                ),
                (
                    "driver",
                    temps.get("driver_temperature"),
                    DRIVER_TEMP_PAUSE_THRESHOLD_C,
                    DRIVER_TEMP_REARM_THRESHOLD_C,
                ),
            ):
                if values is None or len(values) == 0:
                    continue
                peak = float(max(values))
                if peak >= rearm:
                    all_recovered = False
                hot_joints = [idx + 1 for idx, t in enumerate(values) if t >= trip]
                if hot_joints:
                    joints = ",".join(f"J{j}" for j in hot_joints)
                    breaches.append(f"{hand} {label} {joints} {peak:.0f}C>={trip:.0f}C")

        # Recovery: once every joint has cooled below its re-arm threshold, drop
        # the latch so the operator's next enable will hold.
        if have_reading and all_recovered and self._over_temp_latched:
            self._over_temp_latched = False
            log.info("Motor temperatures recovered — arms may be re-enabled")

        if not breaches:
            return

        reason = "over-temperature (" + "; ".join(breaches) + ")"
        first_trip = not self._over_temp_latched
        self._over_temp_latched = True

        if not any_enabled:
            # Already paused/disabled — nothing to hold. Stay latched quietly so
            # we don't spam the console while it cools.
            if first_trip:
                log.warning(f"{reason} — arms already paused; holding until cooled")
            return

        if first_trip:
            log.warning(f"OVER-TEMPERATURE — auto-pausing arms: {reason}")
            self._log_active_hardware_diagnostics(reason, force=True)
        else:
            log.warning(f"Still too hot to run — re-pausing arms: {reason}")
        self._pause_and_hold_active("over-temp auto-pause")

    def _log_active_hardware_diagnostics(
        self,
        reason: str,
        *,
        force: bool = False,
    ) -> None:
        for hand, controller in self.robot_controllers.items():
            controller.log_hardware_diagnostics(f"{reason} ({hand})", force=force)

    def on_button_a_pressed(self) -> None:
        """Toggle active arms enabled/pause (hold)."""
        states = {
            hand: self.data_manager.get_robot_activity_state(hand)
            for hand in self.active_hands
        }
        state_msg = ", ".join(
            f"{hand}_state={state.value}" for hand, state in states.items()
        )
        self._log_active_hardware_diagnostics(
            f"Button A/GUI enable toggle pressed; {state_msg}",
            force=True,
        )
        if any(
            state
            in (RobotActivityState.PRESET_CONTROLLED, RobotActivityState.POLICY_CONTROLLED)
            for state in states.values()
        ):
            # Dashboard motion or a policy run: stop it and hold (the dashboard's
            # STOP path also ends the policy runner).
            if self.preset_dashboard_controller is not None:
                self.preset_dashboard_controller.stop_motion()
                log.info(
                    f"{self.arm_label.capitalize()} dashboard motion / policy "
                    "stopped (Button A)"
                )
                return

        any_enabled = any(
            state == RobotActivityState.ENABLED for state in states.values()
        )
        if any_enabled:
            self._pause_and_hold_active("Button A")
            return

        resume_results = {
            hand: controller.resume_robot(force=True)
            for hand, controller in self.robot_controllers.items()
        }
        hold_results = {
            hand: controller.hold_current_position() if resume_results[hand] else False
            for hand, controller in self.robot_controllers.items()
        }
        if all(resume_results.values()) and all(hold_results.values()):
            self._set_active_activity(RobotActivityState.ENABLED)
            if self.visualizer is not None:
                self.visualizer.update_toggle_robot_enabled_status(True)
            log.info(f"{self.arm_label.capitalize()} enabled (Button A)")
        else:
            for hand in self.active_hands:
                if not resume_results[hand]:
                    log.warning(f"Failed forced resume for {hand} robot (Button A)")
                if resume_results[hand] and not hold_results[hand]:
                    log.warning(f"Failed to hold {hand} robot after resume (Button A)")
            self._set_active_activity(RobotActivityState.DISABLED)
            self._clear_active_teleop()
            if self.visualizer is not None:
                self.visualizer.update_toggle_robot_enabled_status(False)
            log.warning(f"Failed to enable {self.arm_label} (Button A)")

    def on_button_x_pressed(self) -> None:
        """Home active arms when enabled."""
        if any(
            self.data_manager.get_robot_activity_state(hand)
            != RobotActivityState.ENABLED
            for hand in self.active_hands
        ):
            log.info(f"Button X pressed but {self.arm_label} is not enabled")
            return

        log.info(f"Button X pressed - Homing {self.arm_label}...")
        self._set_active_activity(RobotActivityState.HOMING)
        self._clear_active_teleop()

        home_results = {
            hand: controller.move_to_home(
                speed_percent=controller.get_home_speed_percent()
            )
            for hand, controller in self.robot_controllers.items()
        }
        if not all(home_results.values()):
            failed = [hand for hand, ok in home_results.items() if not ok]
            log.warning(f"Failed to initiate home move on arm(s) {failed}")
            self._set_active_activity(RobotActivityState.ENABLED)

    def on_button_b_pressed(self) -> None:
        """Advance the right preset-dashboard workflow human wait step."""
        if self.preset_dashboard_controller is None:
            return
        motion = self.preset_dashboard_controller.confirm_human_step(
            "Quest B",
            builder="right",
        )
        if (
            motion.get("kind") == "workflow"
            and motion.get("state") == "waiting"
            and motion.get("workflow_step_kind") == "human_wait"
        ):
            log.info("Right preset workflow human wait confirmed (Button B)")

    def on_button_y_pressed(self) -> None:
        """Advance the left preset-dashboard workflow human wait step."""
        if self.preset_dashboard_controller is None:
            return
        motion = self.preset_dashboard_controller.confirm_human_step(
            "Quest Y",
            builder="left",
        )
        if (
            motion.get("kind") == "workflow"
            and motion.get("state") == "waiting"
            and motion.get("workflow_step_kind") == "human_wait"
        ):
            log.info("Left preset workflow human wait confirmed (Button Y)")

    def _current_activity_mode(self) -> int:
        """Derive the current operating-mode code (see ACTIVITY_MODE_CODES).

        Priority: teleop > homing > trajectory > preset > paused > disabled >
        idle. Reads live DataManager state plus the preset dashboard's active
        motion kind; never raises.
        """
        states = {
            hand: self.data_manager.get_robot_activity_state(hand)
            for hand in self.active_hands
        }
        if any(self.data_manager.get_teleop_active(hand) for hand in self.active_hands):
            return ACTIVITY_MODE_CODES["teleop"]
        if any(state == RobotActivityState.HOMING for state in states.values()):
            return ACTIVITY_MODE_CODES["homing"]
        kind = None
        if self.preset_dashboard_controller is not None:
            try:
                kind = self.preset_dashboard_controller.active_motion_kind()
            except Exception:  # noqa: BLE001
                kind = None
        if kind == "policy" or any(
            state == RobotActivityState.POLICY_CONTROLLED for state in states.values()
        ):
            return ACTIVITY_MODE_CODES["policy"]
        if kind == "trajectory":
            return ACTIVITY_MODE_CODES["trajectory"]
        if kind == "preset" or any(
            state == RobotActivityState.PRESET_CONTROLLED for state in states.values()
        ):
            return ACTIVITY_MODE_CODES["preset"]
        if states and all(
            state == RobotActivityState.DISABLED for state in states.values()
        ):
            return ACTIVITY_MODE_CODES["disabled"]
        if any(state == RobotActivityState.PAUSED for state in states.values()):
            return ACTIVITY_MODE_CODES["paused"]
        return ACTIVITY_MODE_CODES["idle"]

    def _update_primary_visualizer(self) -> None:
        assert self.visualizer is not None
        controller_transform, grip_value, trigger_value = (
            self.data_manager.get_controller_data(self.primary_hand)
        )
        teleop_active = self.data_manager.get_teleop_active(self.primary_hand)
        robot_activity_state = self.data_manager.get_robot_activity_state(
            self.primary_hand
        )
        current_joint_angles = self.data_manager.get_current_joint_angles(
            self.primary_hand
        )
        target_joint_angles = self.data_manager.get_target_joint_angles(
            self.primary_hand
        )
        solve_time_ms = self.data_manager.get_ik_solve_time_ms(self.primary_hand)
        target_pose = self.data_manager.get_target_pose(self.primary_hand)

        self.visualizer.set_grip_value(grip_value)
        self.visualizer.set_trigger_value(trigger_value)
        self.visualizer.update_timing(solve_time_ms)
        self.visualizer.update_controller_visualization(controller_transform)
        if controller_transform is not None:
            self.visualizer.update_controller_status_display(
                controller_transform[:3, 3],
                connected=True,
            )
        else:
            self.visualizer.update_controller_status_display(None, connected=False)
        self.visualizer.update_teleop_status(teleop_active)
        self.visualizer.update_target_visualization(target_pose)

        if current_joint_angles is not None:
            current_joint_rad = np.radians(current_joint_angles)
            self.visualizer.update_robot_pose(current_joint_rad)
            self.visualizer.update_joint_angles_display(current_joint_rad)

        if target_joint_angles is not None and robot_activity_state in (
            RobotActivityState.ENABLED,
            RobotActivityState.PRESET_CONTROLLED,
        ):
            target_joint_rad = np.radians(target_joint_angles)
            self.visualizer.update_ghost_robot_visibility(True)
            self.visualizer.update_ghost_robot_pose(target_joint_rad)
        else:
            self.visualizer.update_ghost_robot_visibility(False)

        if robot_activity_state == RobotActivityState.ENABLED:
            self.visualizer.update_robot_status(
                f"{self.primary_hand.capitalize()} Robot: Enabled"
            )
        elif robot_activity_state == RobotActivityState.PRESET_CONTROLLED:
            self.visualizer.update_robot_status(
                f"{self.primary_hand.capitalize()} Robot: Preset"
            )
        elif robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
            self.visualizer.update_robot_status(
                f"{self.primary_hand.capitalize()} Robot: Policy"
            )
        elif robot_activity_state == RobotActivityState.HOMING:
            self.visualizer.update_robot_status(
                f"{self.primary_hand.capitalize()} Robot: Homing"
            )
        else:
            self.visualizer.update_robot_status(
                f"{self.primary_hand.capitalize()} Robot: Disabled"
            )

        self.visualizer.update_gripper_status(
            trigger_value,
            robot_enabled=robot_activity_state
            in (RobotActivityState.ENABLED, RobotActivityState.PRESET_CONTROLLED),
        )

    def _update_secondary_visualizer(self) -> None:
        for hand, controls in self.secondary_status_controls.items():
            _controller_transform, hand_grip, hand_trigger = (
                self.data_manager.get_controller_data(hand)
            )
            hand_teleop = self.data_manager.get_teleop_active(hand)
            hand_activity = self.data_manager.get_robot_activity_state(hand)
            hand_joints = self.data_manager.get_current_joint_angles(hand)

            controls["grip_value"].value = float(hand_grip)
            controls["trigger_value"].value = float(hand_trigger)
            controls["teleop_status"].value = "Active" if hand_teleop else "Inactive"
            if hand_activity == RobotActivityState.ENABLED:
                controls["robot_status"].value = "Enabled"
            elif hand_activity == RobotActivityState.PRESET_CONTROLLED:
                controls["robot_status"].value = "Preset"
            elif hand_activity == RobotActivityState.POLICY_CONTROLLED:
                controls["robot_status"].value = "Policy"
            elif hand_activity == RobotActivityState.HOMING:
                controls["robot_status"].value = "Homing"
            else:
                controls["robot_status"].value = "Disabled"

            if hand_joints is not None:
                controls["joint_angles"].value = (
                    "[" + ",".join(f"{x:.1f}" for x in hand_joints) + "]"
                )
            else:
                controls["joint_angles"].value = "Waiting..."


def run_runtime_until_interrupt(runtime: BimanualTeleopRuntime) -> None:
    """Run a runtime and always clean it up on exit."""
    try:
        runtime.start()
        runtime.run_visualizer_loop()
    except KeyboardInterrupt:
        log.info("interrupt received — stopping teleop runtime...")
    except Exception as exc:
        log.error("teleop runtime error: %s", exc, exc_info=True)
        raise
    finally:
        runtime.stop()


__all__ = [
    "BimanualTeleopRuntime",
    "BimanualTeleopRuntimeConfig",
    "run_runtime_until_interrupt",
]
