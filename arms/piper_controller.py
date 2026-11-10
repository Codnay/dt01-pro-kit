#!/usr/bin/env python3
"""Core Piper Robot Controller."""

import os
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from enum import Enum
from typing import Any, Callable

import numpy as np
from piper_sdk import C_PiperInterface_V2  # type: ignore[attr-defined]
from scipy.spatial.transform import Rotation

from arms.joint_limits import clamp_recovery_step, recovery_joint_bounds

from arms.feedback_timing import FeedbackTiming, read_sdk_feedback

from arms.perf.gil_tuning import try_realtime_priority

# Metrics hooks: no exporter in this kit, so these are no-ops.
def inc_can_io_error(*_a: Any, **_k: Any) -> None:
    return None


def set_can_bus_state(*_a: Any, **_k: Any) -> None:
    return None


from arms.logging_setup import get_logger as _get_logger


MAX_CONFIGURABLE_JOINT_SPEED_RAW = 1500


class PiperController:
    """Core PiPER robot controller that manages robot state and communication.

    This class is independent of any UI framework and can be used with
    GUI, VR, or other control interfaces through a command queue.
    """

    class ControlMode(Enum):
        """Control mode enumeration."""

        END_EFFECTOR = "end_effector"
        JOINT_SPACE = "joint_space"

    def __init__(
        self,
        can_interface: str = "can0",
        robot_rate: float = 100.0,
        control_mode: "PiperController.ControlMode" = ControlMode.JOINT_SPACE,
        neutral_joint_angles: np.ndarray | None = None,
        neutral_end_effector_pose: np.ndarray | None = None,
        debug_mode: bool = False,
        joint_limit_margin_deg: float | Sequence[float] | Mapping[str, float] = 0.0,
        enable_on_init: bool = False,
    ) -> None:
        """Initialize the robot controller.

        Args:
            can_interface: CAN interface for robot communication (default: 'can0')
            robot_rate: Robot control loop rate in Hz (default: 100.0)
            control_mode: Initial control mode (END_EFFECTOR or JOINT_SPACE)
            neutral_joint_angles: Neutral joint angles [j1, j2, j3, j4, j5, j6] in degrees (default: None)
            neutral_end_effector_pose: Neutral end effector pose as 4x4 transformation matrix (default: None)
            debug_mode: Enable debug logging (default: False)
            joint_limit_margin_deg: Inward margin (degrees) applied to SDK joint limits.
                Accepts:
                - scalar float for all joints,
                - sequence of 6 values ordered [j1..j6],
                - mapping with named keys (joint1..joint6 or j1..j6).
                Unspecified mapping entries default to 0.0 degrees.
                Use non-negative values to avoid firmware torque protection at
                mechanical extremes (default: 0.0)
            enable_on_init: When True, explicitly run the resume path after
                connecting. The default False keeps motors disabled on startup.
        """
        self.can_interface = can_interface
        # Per-arm logger; the component renders as e.g. [can_left] / [can_right].
        self._log = _get_logger(can_interface)
        self.robot_rate = robot_rate
        self.debug_mode = debug_mode
        self._enable_on_init = bool(enable_on_init)
        self._joint_limit_margin_deg = self._normalize_joint_limit_margins_deg(
            joint_limit_margin_deg
        )

        # SDK I/O is not guaranteed to be thread-safe. Protect all SDK calls.
        self._sdk_lock = threading.Lock()

        # Best-effort diagnostics / error throttling for intermittent CAN issues.
        self._io_error_count: int = 0
        self._last_io_error_ts: float = 0.0
        self._last_can_diag_ts: float = 0.0
        self._last_hardware_diag_ts: float = 0.0

        # Control-loop / command TX diagnostics (monotonic clock).
        self._last_control_loop_iter_ts: float = time.monotonic()
        self._last_any_cmd_tx_ts: float = 0.0
        self._last_joint_cmd_tx_ts: float = 0.0
        self._last_pose_cmd_tx_ts: float = 0.0
        self._last_gripper_cmd_tx_ts: float = 0.0
        self._warn_last_ts_by_key: dict[str, float] = {}

        # Optional tuning knobs via env vars (safe defaults).
        self._diagnostics_enabled: bool = os.getenv("PIPER_DIAGNOSTICS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self._slow_sdk_call_warn_s: float = float(
            os.getenv("PIPER_SLOW_SDK_WARN_S", "0.05")
        )
        self._can_recover_on_init: bool = os.getenv(
            "PIPER_CAN_RECOVER_ON_INIT", "1"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._can_recover_on_resume: bool = os.getenv(
            "PIPER_CAN_RECOVER_ON_RESUME", "1"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._can_recover_restart_ms: int = int(
            os.getenv("PIPER_CAN_RECOVER_RESTART_MS", "100")
        )
        self._can_recover_default_bitrate: int = int(
            os.getenv("PIPER_CAN_DEFAULT_BITRATE", "1000000")
        )
        self._startup_feedback_sample_count: int = max(
            1, int(os.getenv("PIPER_STARTUP_FEEDBACK_SAMPLES", "3"))
        )
        self._startup_feedback_sample_interval_s: float = max(
            0.0, float(os.getenv("PIPER_STARTUP_FEEDBACK_SAMPLE_INTERVAL_S", "0.02"))
        )
        self._startup_feedback_max_age_s: float = max(
            0.0, float(os.getenv("PIPER_STARTUP_FEEDBACK_MAX_AGE_S", "0.5"))
        )
        self._startup_feedback_stability_tolerance_deg: float = max(
            0.0,
            float(os.getenv("PIPER_STARTUP_FEEDBACK_STABILITY_TOLERANCE_DEG", "0.5")),
        )
        self._resume_hardware_limit_epsilon_deg: float = max(
            0.0,
            float(os.getenv("PIPER_RESUME_HARDWARE_LIMIT_EPS_DEG", "0.001")),
        )
        self._hold_current_position_speed_percent: int = int(
            np.clip(
                int(os.getenv("PIPER_HOLD_CURRENT_POSITION_SPEED_PERCENT", "1")),
                1,
                100,
            )
        )
        # Speed cap applied to X-button / operator-triggered homing so the arms
        # crawl back to the home pose. Default 1%; override via env.
        self._home_speed_percent: int = int(
            np.clip(
                int(os.getenv("PIPER_HOME_SPEED_PERCENT", "1")),
                1,
                100,
            )
        )

        # Reduce CAN load: don't spam MotionCtrl_2 unless requested.
        self._motionctrl_every_loop: bool = os.getenv(
            "PIPER_MOTIONCTRL_EVERY_LOOP", ""
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._motionctrl_keepalive_s: float = float(
            os.getenv("PIPER_MOTIONCTRL_KEEPALIVE_S", "1.0")
        )
        self._last_motionctrl_ts: float = 0.0
        self._last_motionctrl_move_mode: int | None = None
        self._last_motionctrl_speed_percent: int | None = None
        self._last_motionctrl_ok_ts: float = 0.0
        self._motionctrl_speed_percent: int = 100
        self._active_motion_speed_cap_percent: int | None = None
        self._joint_velocity_limits_deg_s: np.ndarray | None = None
        self._joint_velocity_limits_raw: np.ndarray | None = None

        # Physical Teach owns the arm until a fresh dashboard Enable. Check the
        # SDK's cached status every control tick and at motion I/O, without a
        # CAN round trip or the old 100 ms window for stale target transmission.
        self._teach_guard_enabled: bool = os.getenv(
            "PIPER_TEACH_MODE_GUARD", "1"
        ).strip().lower() not in {"0", "false", "no", "n", "off"}
        self._teach_mode_active: bool = False
        self._was_teaching: bool = False
        self._teach_status_valid: bool = False
        self._teach_interlock: bool = False

        # Reduce CAN load: gripper doesn't need 100 Hz streaming in most cases.
        self._gripper_tx_hz: float = float(os.getenv("PIPER_GRIPPER_TX_HZ", "20.0"))
        self._gripper_tx_period_s: float = (
            1.0 / self._gripper_tx_hz if self._gripper_tx_hz > 0 else 0.0
        )
        self._gripper_tx_epsilon_deg: float = float(
            os.getenv("PIPER_GRIPPER_TX_EPS_DEG", "0.1")
        )

        self._sdk_error_sentinel = object()

        # CAN TX circuit breaker — stops hammering the bus when it enters error state.
        self._cb_state: str = "CLOSED"          # "CLOSED", "OPEN", "HALF_OPEN"
        self._cb_consecutive_ok: int = 0
        self._cb_consecutive_fail: int = 0
        self._cb_open_since: float = 0.0
        self._cb_backoff_s: float = 0.0
        self._cb_last_bus_state: str = "active"

        self._cb_fail_threshold: int = int(os.getenv("PIPER_CB_FAIL_THRESHOLD", "3"))
        self._cb_ok_threshold: int = int(os.getenv("PIPER_CB_OK_THRESHOLD", "5"))
        self._cb_min_backoff_s: float = float(os.getenv("PIPER_CB_MIN_BACKOFF_S", "0.1"))
        self._cb_max_backoff_s: float = float(os.getenv("PIPER_CB_MAX_BACKOFF_S", "2.0"))

        # Thread synchronization
        self.position_lock = threading.Lock()
        self._command_handoff_lock = threading.RLock()
        self._stop_hold_lock = threading.RLock()
        self._stop_hold_latched = threading.Event()
        self._stop_hold_generation = 0
        self._last_stop_hold_tx = 0.0
        # Immutable stop destination, separate from normal/late worker targets.
        self._stop_hold_target: np.ndarray | None = None
        self._stop_hold_mit_mode = 0x00
        self.state_lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()

        # Robot operational state
        self._robot_enabled = False

        self._control_loop_thread = threading.Thread(
            target=self.control_loop, daemon=True
        )

        # Control mode
        self._control_mode = control_mode

        # HOME positions in end effector space and joint space
        if neutral_end_effector_pose is not None:
            if neutral_end_effector_pose.shape == (4, 4):
                self.HOME_POSE = neutral_end_effector_pose.copy().astype(np.float64)
            else:
                raise ValueError(
                    "neutral_end_effector_pose must be a 4x4 transformation matrix"
                )
        else:
            # Convert default 6D pose to 4x4 matrix
            default_6d_pose = np.array(
                [-3.123, -125.085, 382.251, -78.132, 84.303, -169.496], dtype=np.float64
            )
            self.HOME_POSE = self._pose_6d_to_4x4(default_6d_pose)

        if neutral_joint_angles is not None:
            self.HOME_JOINT_ANGLES = np.array(neutral_joint_angles, dtype=np.float64)
        else:
            self.HOME_JOINT_ANGLES = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        # Gripper range in degrees (for internal SDK communication)
        self.GRIPPER_DEGREES_MIN = 0.000
        self.GRIPPER_DEGREES_MAX = 95.00
        self.GRIPPER_DEGREES_RANGE = self.GRIPPER_DEGREES_MAX - self.GRIPPER_DEGREES_MIN

        # Home gripper value in normalized form (53.8 / 95.0)
        self.HOME_GRIPPER_OPEN_VALUE_DEGREES = 53.800

        # End-effector target pose
        self._target_pose = self.HOME_POSE.copy()

        # Joint space target angles
        self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()

        # Gripper target open value
        self._gripper_open_value_degrees = self.HOME_GRIPPER_OPEN_VALUE_DEGREES
        self._last_gripper_cmd_degrees: float = self._gripper_open_value_degrees

        # Initialize robot connection
        self._initialize_robot()
        if self._enable_on_init and not self.resume_robot(force=True):
            raise RuntimeError(
                f"{self._log_prefix()} Failed to enable robot during initialization. "
                "Startup feedback validation failed or EnablePiper did not succeed."
            )

        if self.debug_mode:
            self._log.info(f"PiperController initialized at {robot_rate} Hz")

    @staticmethod
    def _normalize_joint_limit_margins_deg(
        margin: float | Sequence[float] | Mapping[str, float],
    ) -> np.ndarray:
        """Normalize per-joint margins into a 6-element array in degrees."""
        if isinstance(margin, (int, float)):
            values = np.full(6, float(margin), dtype=np.float64)
        elif isinstance(margin, Mapping):
            # Allow common naming variants. Missing joints default to no margin.
            key_to_idx = {
                "j1": 0, "joint1": 0, "joint_1": 0,
                "j2": 1, "joint2": 1, "joint_2": 1,
                "j3": 2, "joint3": 2, "joint_3": 2,
                "j4": 3, "joint4": 3, "joint_4": 3,
                "j5": 4, "joint5": 4, "joint_5": 4,
                "j6": 5, "joint6": 5, "joint_6": 5,
            }
            values = np.zeros(6, dtype=np.float64)
            unknown_keys: list[str] = []
            for raw_key, raw_value in margin.items():
                key = str(raw_key).strip().lower()
                if key not in key_to_idx:
                    unknown_keys.append(str(raw_key))
                    continue
                idx = key_to_idx[key]
                values[idx] = float(raw_value)

            if unknown_keys:
                raise ValueError(
                    "Unknown joint_limit_margin_deg keys: "
                    + ", ".join(sorted(unknown_keys))
                    + ". Expected joint1..joint6 (or j1..j6)."
                )
        else:
            values = np.array(margin, dtype=np.float64)
            if values.shape != (6,):
                raise ValueError(
                    f"joint_limit_margin_deg sequence must have shape (6,), got {values.shape}"
                )

        if np.any(~np.isfinite(values)):
            raise ValueError("joint_limit_margin_deg contains non-finite values")
        if np.any(values < 0.0):
            raise ValueError("joint_limit_margin_deg must be non-negative for all joints")
        return values

    def _log_prefix(self) -> str:
        return f"[{self.can_interface}]"

    def _warn_rate_limited(
        self, key: str, message: str, interval_s: float = 1.0
    ) -> None:
        now = time.monotonic()
        last = self._warn_last_ts_by_key.get(key, 0.0)
        if now - last < interval_s:
            return
        self._warn_last_ts_by_key[key] = now
        # Callers pass messages already prefixed with "[can_x] "; the logger
        # re-adds that as the component, so strip it to avoid double prefixing.
        prefix = self._log_prefix() + " "
        if message.startswith(prefix):
            message = message[len(prefix):]
        self._log.warning(message)

    def _sdk_call(self, op: str, fn: Callable[[], Any]) -> Any:
        start = time.monotonic()
        try:
            motion = op in {
                "MotionCtrl_2", "JointCtrl", "EndPoseCtrl", "GripperCtrl",
                "EnablePiper", "JointConfig(clear_err)",
                "MotionCtrl_2(stop_hold)", "JointCtrl(stop_hold)",
                "MotionCtrl_1(clear_stopped_plan)", "MotionCtrl_1(stop_hold)",
            }
            # Match Stop's lock order. Inspect even commands copied before the
            # loop noticed Teach, including a previous Stop's frozen target.
            with self._stop_hold_lock if motion else nullcontext():
                with self._sdk_lock:
                    if motion and self._teach_guard_enabled:
                        self._poll_teach_mode(sdk_locked=True)
                        if self._teach_mode_active or self._was_teaching:
                            return self._sdk_error_sentinel
                    # A command copied before STOP must not reach CAN after
                    # the pause. Reads and safety-owned writes remain available.
                    if self._stop_hold_latched.is_set() and op in {
                        "MotionCtrl_2", "JointCtrl", "EndPoseCtrl", "GripperCtrl",
                        "EnablePiper", "JointConfig(clear_err)",
                    }:
                        return self._sdk_error_sentinel
                    ret = fn()
        except Exception as e:
            self._record_io_error(op, e)
            return self._sdk_error_sentinel
        finally:
            dur_s = time.monotonic() - start
            if dur_s >= self._slow_sdk_call_warn_s:
                self._warn_rate_limited(
                    f"slow_{op}",
                    f"{self._log_prefix()} Slow SDK call {op}: {dur_s * 1000.0:.1f}ms",
                    interval_s=1.0,
                )

        # Some SDK functions may return a boolean success flag; surface failures.
        if isinstance(ret, bool) and (not ret):
            self._warn_rate_limited(
                f"false_{op}",
                f"{self._log_prefix()} SDK call {op} returned False (no exception)",
                interval_s=1.0,
            )

        return ret

    def _sdk_call_nonblocking(self, op: str, fn: Callable[[], Any]) -> Any:
        """Call optional SDK op only if no command/feedback call is in progress."""
        start = time.monotonic()
        if not self._sdk_lock.acquire(blocking=False):
            return self._sdk_error_sentinel
        try:
            ret = fn()
        except Exception as e:
            self._record_io_error(op, e)
            return self._sdk_error_sentinel
        finally:
            self._sdk_lock.release()
            dur_s = time.monotonic() - start
            if dur_s >= self._slow_sdk_call_warn_s:
                self._warn_rate_limited(
                    f"slow_{op}",
                    f"{self._log_prefix()} Slow SDK call {op}: {dur_s * 1000.0:.1f}ms",
                    interval_s=1.0,
                )

        if isinstance(ret, bool) and (not ret):
            self._warn_rate_limited(
                f"false_{op}",
                f"{self._log_prefix()} SDK call {op} returned False (no exception)",
                interval_s=1.0,
            )

        return ret

    def _sdk_ok(self, op: str, fn: Callable[[], Any]) -> bool:
        """Call SDK op and return True on best-effort success."""
        ret = self._sdk_call(op, fn)
        if ret is self._sdk_error_sentinel:
            return False
        if isinstance(ret, bool):
            return ret
        return True

    def _safe_sdk_snapshot_call(self, op: str, fn: Callable[[], Any]) -> Any:
        """Call SDK for diagnostics without recursing through I/O error logging."""
        if not self._sdk_lock.acquire(blocking=False):
            return f"<{op} skipped: SDK busy>"
        try:
            return fn()
        except Exception as e:
            return f"<{op} error: {e}>"
        finally:
            self._sdk_lock.release()

    def _compact_sdk_value(self, value: Any, depth: int = 0) -> Any:
        """Convert SDK feedback objects into compact log-friendly structures."""
        if depth > 3:
            return repr(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._compact_sdk_value(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {
                str(k): self._compact_sdk_value(v, depth + 1)
                for k, v in value.items()
            }

        raw = getattr(value, "__dict__", None)
        if isinstance(raw, dict):
            public = {
                k: self._compact_sdk_value(v, depth + 1)
                for k, v in raw.items()
                if not k.startswith("_")
            }
            if public:
                return public

        return repr(value)

    def log_hardware_diagnostics(self, reason: str, *, force: bool = False) -> None:
        """Log PiPER hardware status/protection feedback from the SDK."""
        now = time.time()
        if (not force) and (now - self._last_hardware_diag_ts < 2.0):
            return
        self._last_hardware_diag_ts = now

        if not hasattr(self, "piper") or self.piper is None:
            self._log.debug(
                f"Hardware diagnostics ({reason}): "
                "no SDK object"
            )
            return

        snapshot: dict[str, Any] = {}
        for op in (
            "GetArmStatus",
            "GetArmEnableStatus",
            "GetDriverStates",
            "GetArmLowSpdInfoMsgs",
        ):
            fn = getattr(self.piper, op, None)
            if callable(fn):
                snapshot[op] = self._compact_sdk_value(
                    self._safe_sdk_snapshot_call(op, fn)
                )

        if not snapshot:
            self._log.debug(
                f"Hardware diagnostics ({reason}): "
                "SDK exposes no status/protection methods"
            )
            return

        self._log.debug(f"Hardware diagnostics ({reason}): {snapshot}")

    def _sdk_connected(self) -> bool:
        """Return True only when SDK explicitly reports an active connection."""
        if (not hasattr(self, "piper")) or self.piper is None:
            return False
        status = self._sdk_call("get_connect_status", self.piper.get_connect_status)
        if status is self._sdk_error_sentinel:
            return False
        return bool(status)

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit-breaker bookkeeping after transport recovery."""
        self._cb_state = "CLOSED"
        self._cb_consecutive_ok = 0
        self._cb_consecutive_fail = 0
        self._cb_open_since = 0.0
        self._cb_backoff_s = 0.0
        self._cb_last_bus_state = "active"

    def _get_can_link_state(self) -> tuple[str | None, int | None]:
        """Best-effort SocketCAN state + bitrate from `ip link show`."""
        try:
            proc = subprocess.run(
                ["ip", "-details", "-statistics", "link", "show", self.can_interface],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return None, None

            out = proc.stdout or ""
            state_match = re.search(r"\bstate\s+([A-Z-]+)\b", out)
            bitrate_match = re.search(r"\bbitrate\s+(\d+)\b", out)
            state = state_match.group(1) if state_match else None
            bitrate = int(bitrate_match.group(1)) if bitrate_match else None
            return state, bitrate
        except Exception:
            return None, None

    def _recover_can_link(self, reason: str) -> bool:
        """Best-effort CAN interface reset (down/type/up with restart-ms)."""
        state, detected_bitrate = self._get_can_link_state()
        bitrate = detected_bitrate or self._can_recover_default_bitrate
        self._log.info(
            f"Attempting CAN link recovery ({reason}). "
            f"state={state or 'unknown'}, bitrate={bitrate}, "
            f"restart-ms={self._can_recover_restart_ms}"
        )

        ok = True
        down_cmd = ["ip", "link", "set", self.can_interface, "down"]
        down_proc = subprocess.run(
            down_cmd, capture_output=True, text=True, check=False
        )
        if down_proc.returncode != 0:
            ok = False
            err = (down_proc.stderr or down_proc.stdout or "").strip()
            self._log.warning(
                f"CAN recovery step failed: {' '.join(down_cmd)} "
                f"(exit={down_proc.returncode}) {err}"
            )
        else:
            type_cmd_with_restart = [
                "ip",
                "link",
                "set",
                self.can_interface,
                "type",
                "can",
                "bitrate",
                str(bitrate),
                "restart-ms",
                str(self._can_recover_restart_ms),
            ]
            type_proc = subprocess.run(
                type_cmd_with_restart, capture_output=True, text=True, check=False
            )
            if type_proc.returncode != 0:
                err = (type_proc.stderr or type_proc.stdout or "").strip()
                self._log.warning(
                    f"CAN recovery: restart-ms unsupported or failed "
                    f"(exit={type_proc.returncode}) {err}. Falling back to bitrate-only."
                )
                type_cmd_no_restart = [
                    "ip",
                    "link",
                    "set",
                    self.can_interface,
                    "type",
                    "can",
                    "bitrate",
                    str(bitrate),
                ]
                type_fallback_proc = subprocess.run(
                    type_cmd_no_restart, capture_output=True, text=True, check=False
                )
                if type_fallback_proc.returncode != 0:
                    ok = False
                    fb_err = (
                        type_fallback_proc.stderr or type_fallback_proc.stdout or ""
                    ).strip()
                    self._log.warning(
                        f"CAN recovery fallback failed: "
                        f"{' '.join(type_cmd_no_restart)} "
                        f"(exit={type_fallback_proc.returncode}) {fb_err}"
                    )

            if ok:
                up_cmd = ["ip", "link", "set", self.can_interface, "up"]
                up_proc = subprocess.run(
                    up_cmd, capture_output=True, text=True, check=False
                )
                if up_proc.returncode != 0:
                    ok = False
                    err = (up_proc.stderr or up_proc.stdout or "").strip()
                    self._log.warning(
                        f"CAN recovery step failed: {' '.join(up_cmd)} "
                        f"(exit={up_proc.returncode}) {err}"
                    )

        post_state, _ = self._get_can_link_state()
        self._log.info(
            f"CAN link recovery result: "
            f"{'ok' if ok else 'failed'} (post_state={post_state or 'unknown'})"
        )
        return ok

    def _recover_transport(self, reason: str, reset_can_link: bool) -> bool:
        """Disconnect/reconnect SDK and optionally reset CAN link."""
        self._log.info(
            f"Attempting transport recovery ({reason}), "
            f"reset_can_link={reset_can_link}"
        )
        self._set_robot_status_enabled(False)

        if hasattr(self, "piper") and self.piper is not None:
            self._sdk_ok("DisconnectPort", self.piper.DisconnectPort)

        if reset_can_link:
            self._recover_can_link(reason=reason)

        if hasattr(self, "piper") and self.piper is not None:
            time.sleep(0.05)
            connected = self._sdk_ok("ConnectPort", self.piper.ConnectPort)
            if connected:
                self._reset_circuit_breaker()
            return connected
        return False

    def is_control_loop_running(self) -> bool:
        return self._control_loop_thread.is_alive() and self.running.is_set()

    def get_control_loop_age_s(self) -> float:
        """Seconds since last control_loop iteration started (monotonic)."""
        return max(0.0, time.monotonic() - self._last_control_loop_iter_ts)

    def get_last_command_tx_age_s(self) -> float:
        """Seconds since last successful-ish command TX (monotonic)."""
        if self._last_any_cmd_tx_ts <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_any_cmd_tx_ts)

    def get_motionctrl_age_s(self) -> float:
        if self._last_motionctrl_ok_ts <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_motionctrl_ok_ts)

    def get_motionctrl_move_mode(self) -> int | None:
        return self._last_motionctrl_move_mode

    def _maybe_print_can_diagnostics(self, reason: str) -> None:
        """Print best-effort SocketCAN state when we hit I/O errors (rate-limited)."""
        now = time.time()
        if now - self._last_can_diag_ts < 2.0:
            return
        self._last_can_diag_ts = now

        try:
            proc = subprocess.run(
                ["ip", "-details", "-statistics", "link", "show", self.can_interface],
                capture_output=True,
                text=True,
                check=False,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if out:
                self._log.debug(f"CAN diagnostics ({reason}):\n{out}")
            elif err:
                self._log.warning(
                    f"CAN diagnostics ({reason}) failed: {err}"
                )
        except Exception as e:
            if self.debug_mode:
                self._log.debug(f"CAN diagnostics exception: {e}")

    def _record_io_error(self, op: str, err: Exception) -> None:
        now = time.time()
        self._io_error_count += 1
        self._last_io_error_ts = now
        inc_can_io_error(self.can_interface)
        self._log.error("I/O error during %s: %s", op, err)
        self._maybe_print_can_diagnostics(reason=f"{op} error")

    # ------------------------------------------------------------------
    # CAN bus health probe & circuit breaker
    # ------------------------------------------------------------------

    def _is_can_bus_healthy(self) -> bool:
        """Check CAN bus state via the SDK.  Logs transitions."""
        try:
            can_obj = self.piper.GetCanBus()
            if can_obj is None:
                state = "unavailable"
            else:
                # is_can_bus_ok() returns an IntEnum; BUS_STATE_ACTIVE == 100019.
                raw = can_obj.is_can_bus_ok()
                # Map to readable string.
                _STATE_MAP = {100019: "active", 100020: "passive",
                              100021: "error", 100022: "unknown"}
                state = _STATE_MAP.get(int(raw), f"raw({raw})")
        except Exception:
            state = "unavailable"

        prev = self._cb_last_bus_state
        self._cb_last_bus_state = state
        set_can_bus_state(self.can_interface, state)

        # Log transitions.
        if state != prev:
            if state == "active":
                self._log.info("CAN bus recovered -> ACTIVE")
            else:
                self._log.warning(
                    "CAN bus state changed: %s -> %s (io_errors=%s)",
                    prev, state, self._io_error_count,
                )
        elif state != "active":
            # Rate-limited logging while unhealthy.
            self._warn_rate_limited(
                "can_unhealthy",
                f"{self._log_prefix()} CAN bus unhealthy: state={state}, "
                f"cb={self._cb_state}, consecutive_fail={self._cb_consecutive_fail}, "
                f"io_errors={self._io_error_count}",
                interval_s=1.0,
            )

        return state == "active"

    def _circuit_breaker_check(self) -> bool:
        """Return True if the control loop should attempt TX this iteration."""
        now = time.monotonic()

        if self._cb_state == "CLOSED":
            return True

        if self._cb_state == "OPEN":
            if (now - self._cb_open_since) >= self._cb_backoff_s:
                self._cb_state = "HALF_OPEN"
                self._cb_consecutive_ok = 0
                self._log.info(
                    "circuit breaker OPEN -> HALF_OPEN: probing CAN bus "
                    "(backoff was %.2fs)", self._cb_backoff_s,
                )
                return True
            return False

        # HALF_OPEN: allow sending so we can probe.
        return True

    def _circuit_breaker_record(self, success: bool) -> None:
        """Update circuit breaker state after a TX iteration."""
        if success:
            self._cb_consecutive_fail = 0
            self._cb_consecutive_ok += 1

            if self._cb_state == "HALF_OPEN" and self._cb_consecutive_ok >= self._cb_ok_threshold:
                self._cb_state = "CLOSED"
                self._cb_backoff_s = self._cb_min_backoff_s
                self._log.info(
                    "circuit breaker HALF_OPEN -> CLOSED: CAN bus recovered "
                    "(%s consecutive OK)", self._cb_consecutive_ok,
                )
        else:
            self._cb_consecutive_ok = 0
            self._cb_consecutive_fail += 1

            if self._cb_state == "CLOSED" and self._cb_consecutive_fail >= self._cb_fail_threshold:
                self._cb_state = "OPEN"
                self._cb_open_since = time.monotonic()
                self._cb_backoff_s = self._cb_min_backoff_s
                self._log.warning(
                    "circuit breaker CLOSED -> OPEN: %s consecutive CAN failures, "
                    "bus_state=%s, pausing TX for %.2fs",
                    self._cb_consecutive_fail, self._cb_last_bus_state,
                    self._cb_backoff_s,
                )
                self._maybe_print_can_diagnostics("circuit breaker opened")

            elif self._cb_state == "HALF_OPEN":
                self._cb_state = "OPEN"
                self._cb_open_since = time.monotonic()
                self._cb_backoff_s = min(self._cb_backoff_s * 2.0, self._cb_max_backoff_s)
                self._log.warning(
                    "circuit breaker HALF_OPEN -> OPEN: probe failed, "
                    "bus_state=%s, backing off %.2fs",
                    self._cb_last_bus_state, self._cb_backoff_s,
                )

    def start_control_loop(self) -> None:
        """Start the control loop thread."""
        # Thread objects cannot be restarted once stopped; recreate if needed.
        if self._control_loop_thread.is_alive():
            return
        if not self.running.is_set():
            self.running.set()
        self._control_loop_thread = threading.Thread(
            target=self.control_loop, daemon=True
        )
        self._control_loop_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the control loop thread."""
        self.running.clear()
        if self._control_loop_thread.is_alive():
            self._control_loop_thread.join()
            self._log.info("control loop thread joined")
        else:
            self._log.debug("control loop thread is not running")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources and disconnect from robot."""
        self._log.info("cleaning up robot controller...")
        self.stop_control_loop()

        if hasattr(self, "piper") and self.piper is not None and self._sdk_connected():
            self.graceful_stop()
            self._sdk_ok("DisconnectPort", self.piper.DisconnectPort)
            self._log.info("robot disconnected")

        self._log.info("robot controller cleanup completed")

    def is_robot_enabled(self) -> bool:
        """Check if robot is enabled and ready to accept commands.

        Returns:
            True if robot is enabled, False otherwise
        """
        with self.state_lock:
            return self._robot_enabled

    def _set_robot_status_enabled(self, enabled: bool) -> None:
        """Set the robot status enabled state (internal method).

        Args:
            enabled: True to enable robot, False to disable
        """
        with self.state_lock:
            old_state = self._robot_enabled
            self._robot_enabled = enabled
            if self.debug_mode:
                self._log.info("robot enabled state changed: %s -> %s", old_state, enabled)

    def _initialize_robot(self) -> None:
        """Initialize robot connection without enabling motors."""
        self._log.info("initializing robot...")
        self.piper = C_PiperInterface_V2(
            self.can_interface, start_sdk_joint_limit=True, start_sdk_gripper_limit=True
        )
        connect_ok = self._sdk_ok("ConnectPort", self.piper.ConnectPort)
        if (not connect_ok) and self._can_recover_on_init:
            connect_ok = self._recover_transport(
                reason="ConnectPort failed during initialization", reset_can_link=True
            )
        if (not connect_ok) and (not self._sdk_connected()):
            connect_ok = self._sdk_ok("ConnectPort", self.piper.ConnectPort)
        if (not connect_ok) and (not self._sdk_connected()):
            raise RuntimeError(
                f"{self._log_prefix()} Failed to connect to robot over CAN interface "
                f"{self.can_interface}. Check CAN link state (`ip -details -statistics link "
                f"show {self.can_interface}`), permissions, and robot power."
            )

        self._sdk_ok(
            "SetSDKGripperRangeParam",
            lambda: self.piper.SetSDKGripperRangeParam(0.0, 0.1),
        )

        joint_limits_raw = [
            self._sdk_call(
                "GetSDKJointLimitParam(j1)",
                lambda: self.piper.GetSDKJointLimitParam("j1"),
            ),
            self._sdk_call(
                "GetSDKJointLimitParam(j2)",
                lambda: self.piper.GetSDKJointLimitParam("j2"),
            ),
            self._sdk_call(
                "GetSDKJointLimitParam(j3)",
                lambda: self.piper.GetSDKJointLimitParam("j3"),
            ),
            self._sdk_call(
                "GetSDKJointLimitParam(j4)",
                lambda: self.piper.GetSDKJointLimitParam("j4"),
            ),
            self._sdk_call(
                "GetSDKJointLimitParam(j5)",
                lambda: self.piper.GetSDKJointLimitParam("j5"),
            ),
            self._sdk_call(
                "GetSDKJointLimitParam(j6)",
                lambda: self.piper.GetSDKJointLimitParam("j6"),
            ),
        ]

        if any(x is None for x in joint_limits_raw):
            raise RuntimeError(
                f"{self._log_prefix()} Failed to read joint limits from SDK; cannot safely clamp joint commands."
            )
        if any(x is self._sdk_error_sentinel for x in joint_limits_raw):
            raise RuntimeError(
                f"{self._log_prefix()} SDK error reading joint limits; cannot safely clamp joint commands."
            )

        joint_limits = np.array(joint_limits_raw, dtype=np.float64)
        if joint_limits.shape != (6, 2):
            raise RuntimeError(
                f"{self._log_prefix()} Unexpected joint limit shape from SDK: {joint_limits.shape} (expected (6,2))"
            )

        self.JOINT_LIMITS = np.degrees(joint_limits)

        # Effective limits: shrink by the soft margin to keep motors away from
        # firmware torque protection at mechanical extremes.
        self.EFFECTIVE_JOINT_LIMITS = self.JOINT_LIMITS.copy()
        self.EFFECTIVE_JOINT_LIMITS[:, 0] += self._joint_limit_margin_deg
        self.EFFECTIVE_JOINT_LIMITS[:, 1] -= self._joint_limit_margin_deg
        if np.any(self.EFFECTIVE_JOINT_LIMITS[:, 0] > self.EFFECTIVE_JOINT_LIMITS[:, 1]):
            raise RuntimeError(
                f"{self._log_prefix()} joint_limit_margin_deg too large; effective limits inverted."
            )

        self._log.debug(f"SDK joint limits (deg): {self.JOINT_LIMITS}")
        self._log.debug(
            f"Per-joint soft margin (deg): "
            f"joint1={self._joint_limit_margin_deg[0]:.3f}, "
            f"joint2={self._joint_limit_margin_deg[1]:.3f}, "
            f"joint3={self._joint_limit_margin_deg[2]:.3f}, "
            f"joint4={self._joint_limit_margin_deg[3]:.3f}, "
            f"joint5={self._joint_limit_margin_deg[4]:.3f}, "
            f"joint6={self._joint_limit_margin_deg[5]:.3f}"
        )
        self._log.debug(
            f"Effective joint limits (deg): "
            f"{self.EFFECTIVE_JOINT_LIMITS}"
        )

        self.log_initial_hardware_snapshot(reason="startup")
        self._set_robot_status_enabled(False)

        self._log.info(
            f"Robot initialized with motors disabled. "
            "Call resume_robot() to enable command streaming."
        )

    def _enable_robot(self, timeout_s: float = 5.0) -> bool:
        """Enable the robot.

        Args:
            timeout_s: Maximum time to wait for enable to succeed.

        Returns:
            True if enabled, False on timeout or error.
        """
        if not hasattr(self, "piper") or self.piper is None:
            return False

        start_time = time.time()
        try:
            while True:
                if self._stop_hold_latched.is_set():
                    return False
                ok = self._sdk_call("EnablePiper", self.piper.EnablePiper)
                if ok is not self._sdk_error_sentinel and bool(ok):
                    break
                if time.time() - start_time > timeout_s:
                    self.log_hardware_diagnostics(
                        f"EnablePiper timed out after {timeout_s:.1f}s",
                        force=True,
                    )
                    return False
                time.sleep(0.01)
        except Exception as e:
            self.log_hardware_diagnostics(
                f"EnablePiper exception: {e}",
                force=True,
            )
            return False

        self._set_robot_status_enabled(True)
        self._log.info("robot enabled successfully")
        return True

    def _disable_robot(self, timeout_s: float = 5.0) -> bool:
        """Disable the robot.

        Args:
            timeout_s: Maximum time to wait for disable to succeed.

        Returns:
            True if disabled, False on timeout or error.
        """
        if not hasattr(self, "piper") or self.piper is None:
            return False

        start_time = time.time()
        try:
            while True:
                ok = self._sdk_call("DisablePiper", self.piper.DisablePiper)
                if ok is not self._sdk_error_sentinel and bool(ok):
                    break
                if time.time() - start_time > timeout_s:
                    return False
                time.sleep(0.01)
        except Exception:
            return False

        self._set_robot_status_enabled(False)
        self._log.info("robot disabled successfully")
        return True

    def log_initial_hardware_snapshot(self, reason: str = "startup") -> None:
        """Log raw SDK feedback values without using them as command targets."""
        if not hasattr(self, "piper") or self.piper is None:
            self._log.debug(
                f"Initial hardware feedback ({reason}): "
                "no SDK object"
            )
            return

        snapshot: dict[str, Any] = {}
        raw_joint_msgs: Any = None
        for op in (
            "GetArmJointMsgs",
            "GetArmGripperMsgs",
            "GetArmEndPoseMsgs",
            "GetArmStatus",
            "GetArmEnableStatus",
            "GetDriverStates",
            "GetArmLowSpdInfoMsgs",
        ):
            fn = getattr(self.piper, op, None)
            if not callable(fn):
                continue
            raw_value = self._safe_sdk_snapshot_call(op, fn)
            if op == "GetArmJointMsgs":
                raw_joint_msgs = raw_value
            snapshot[op] = self._compact_sdk_value(raw_value)

        if raw_joint_msgs is not None and not isinstance(raw_joint_msgs, str):
            joints_deg, error = self._joint_feedback_to_degrees(raw_joint_msgs)
            if joints_deg is not None:
                snapshot["joint_angles_deg"] = self._format_joint_angles_deg(
                    joints_deg
                )
            elif error:
                snapshot["joint_feedback_parse_error"] = error

        self._log.debug(f"Initial hardware feedback ({reason}): {snapshot}")

    def _format_joint_angles_deg(self, joints_deg: np.ndarray) -> str:
        """Format a 6-joint degree vector with fixed decimals for logs."""
        return "[" + ", ".join(f"{float(value):.3f}" for value in joints_deg) + "]"

    def _joint_feedback_to_degrees(
        self, arm_joint_msgs: Any
    ) -> tuple[np.ndarray | None, str | None]:
        """Extract SDK joint feedback as a 6-element degree vector."""
        if arm_joint_msgs is self._sdk_error_sentinel:
            return None, "GetArmJointMsgs returned SDK error"
        if arm_joint_msgs is None:
            return None, "GetArmJointMsgs returned None"

        joint_msg = getattr(arm_joint_msgs, "joint_state", None)
        if joint_msg is None:
            return None, "GetArmJointMsgs result has no joint_state"

        raw_values: list[float] = []
        for joint_idx in range(1, 7):
            field = f"joint_{joint_idx}"
            if not hasattr(joint_msg, field):
                return None, f"joint_state is missing {field}"
            try:
                raw_values.append(float(getattr(joint_msg, field)))
            except Exception as exc:
                return None, f"joint_state.{field} is not numeric: {exc}"

        joints_deg = np.asarray(raw_values, dtype=np.float64) / 1000.0
        if joints_deg.shape != (6,):
            return None, f"joint feedback shape is {joints_deg.shape}, expected (6,)"
        if np.any(~np.isfinite(joints_deg)):
            return None, f"joint feedback contains non-finite values: {joints_deg}"
        return joints_deg, None

    def _coerce_timestamp_seconds(self, value: Any) -> float | None:
        """Best-effort conversion of SDK timestamp-like values to seconds."""
        if value is None:
            return None
        for sec_name in ("sec", "secs", "seconds"):
            if hasattr(value, sec_name):
                try:
                    seconds = float(getattr(value, sec_name))
                except Exception:
                    return None
                nanos = 0.0
                for ns_name in ("nanosec", "nsec", "nsecs", "nanoseconds"):
                    if hasattr(value, ns_name):
                        try:
                            nanos = float(getattr(value, ns_name)) * 1e-9
                        except Exception:
                            nanos = 0.0
                        break
                return seconds + nanos
        try:
            return float(value)
        except Exception:
            return None

    def _feedback_timestamp_age_s(self, arm_joint_msgs: Any) -> float | None:
        """Return feedback age when the SDK exposes an interpretable timestamp."""
        timestamp_value: Any = None
        for owner in (arm_joint_msgs, getattr(arm_joint_msgs, "joint_state", None)):
            if owner is None:
                continue
            for field in ("time_stamp", "timestamp", "timeStamp", "stamp"):
                if hasattr(owner, field):
                    timestamp_value = getattr(owner, field)
                    break
            if timestamp_value is not None:
                break

        timestamp_s = self._coerce_timestamp_seconds(timestamp_value)
        if timestamp_s is None:
            return None
        if not np.isfinite(timestamp_s) or timestamp_s <= 0.0:
            return float("inf")

        now_wall = time.time()
        now_monotonic = time.monotonic()
        wall_candidates: list[float] = []
        for divisor in (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0):
            scaled = timestamp_s / divisor
            if 946684800.0 <= scaled <= now_wall + 5.0:
                wall_candidates.append(now_wall - scaled)

        if wall_candidates:
            return max(0.0, min(wall_candidates))

        monotonic_candidates: list[float] = []
        for divisor in (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0):
            scaled = timestamp_s / divisor
            if max(0.0, now_monotonic - 3600.0) <= scaled <= now_monotonic + 5.0:
                monotonic_candidates.append(now_monotonic - scaled)

        if not monotonic_candidates:
            return None
        return max(0.0, min(monotonic_candidates))

    def _log_joint_feedback_rejection(
        self,
        reason: str,
        message: str,
        raw_samples: list[Any],
        joint_samples_deg: list[str] | None = None,
    ) -> None:
        converted_samples = (
            f" Converted joint samples deg: {joint_samples_deg}."
            if joint_samples_deg
            else ""
        )
        self._log.error(
            f"Refusing to enable robot ({reason}): {message}. "
            f"Raw joint feedback samples: {raw_samples}."
            f"{converted_samples} "
            f"SDK hardware joint limits (deg): {self.JOINT_LIMITS}. "
            f"Effective joint limits (deg): {self.EFFECTIVE_JOINT_LIMITS}"
        )
        self.log_hardware_diagnostics(
            f"invalid joint feedback during {reason}", force=True
        )

    def _read_validated_joint_feedback(self, reason: str) -> np.ndarray | None:
        """Read stable in-limit joint feedback before allowing enable/streaming."""
        if not hasattr(self, "piper") or self.piper is None:
            return None

        samples: list[np.ndarray] = []
        raw_samples: list[Any] = []
        joint_samples_deg: list[str] = []
        errors: list[str] = []

        for sample_idx in range(self._startup_feedback_sample_count):
            raw_value = self._sdk_call("GetArmJointMsgs", self.piper.GetArmJointMsgs)
            raw_samples.append(self._compact_sdk_value(raw_value))

            joints_deg, error = self._joint_feedback_to_degrees(raw_value)
            if error:
                errors.append(f"sample {sample_idx + 1}: {error}")
            else:
                age_s = self._feedback_timestamp_age_s(raw_value)
                if (
                    age_s is not None
                    and age_s > self._startup_feedback_max_age_s
                ):
                    errors.append(
                        f"sample {sample_idx + 1}: stale feedback age "
                        f"{age_s:.3f}s > {self._startup_feedback_max_age_s:.3f}s"
                    )
                else:
                    samples.append(joints_deg)
                    joint_samples_deg.append(
                        self._format_joint_angles_deg(joints_deg)
                    )

            if sample_idx + 1 < self._startup_feedback_sample_count:
                time.sleep(self._startup_feedback_sample_interval_s)

        if errors:
            self._log_joint_feedback_rejection(
                reason, "; ".join(errors), raw_samples, joint_samples_deg
            )
            return None
        if not samples:
            self._log_joint_feedback_rejection(
                reason, "no usable joint feedback samples", raw_samples
            )
            return None

        sample_array = np.vstack(samples)
        hardware_lower_limits = self.JOINT_LIMITS[:, 0]
        hardware_upper_limits = self.JOINT_LIMITS[:, 1]
        hardware_lower_guard = hardware_lower_limits - self._resume_hardware_limit_epsilon_deg
        hardware_upper_guard = hardware_upper_limits + self._resume_hardware_limit_epsilon_deg
        outside_hardware_limits = (
            (sample_array < hardware_lower_guard)
            | (sample_array > hardware_upper_guard)
        )
        if np.any(outside_hardware_limits):
            bad_sample_idx, bad_joint_idx = np.argwhere(outside_hardware_limits)[0]
            bad_value = sample_array[bad_sample_idx, bad_joint_idx]
            self._log_joint_feedback_rejection(
                reason,
                (
                    f"sample {bad_sample_idx + 1} joint{bad_joint_idx + 1}="
                    f"{bad_value:.3f} deg outside hardware limit "
                    f"[{hardware_lower_limits[bad_joint_idx]:.3f}, "
                    f"{hardware_upper_limits[bad_joint_idx]:.3f}] deg "
                    f"(guard epsilon {self._resume_hardware_limit_epsilon_deg:.3f} deg)"
                ),
                raw_samples,
                joint_samples_deg,
            )
            return None

        if sample_array.shape[0] > 1:
            max_delta = float(np.max(np.ptp(sample_array, axis=0)))
            if max_delta > self._startup_feedback_stability_tolerance_deg:
                self._log_joint_feedback_rejection(
                    reason,
                    (
                        f"joint feedback changed by up to {max_delta:.3f} deg "
                        "between startup samples "
                        f"(limit {self._startup_feedback_stability_tolerance_deg:.3f} deg)"
                    ),
                    raw_samples,
                    joint_samples_deg,
                )
                return None

        return sample_array[-1].copy()

    def _lock_joint_target_without_clamping(self, joints_deg: np.ndarray) -> None:
        """Install a pre-validated measured joint target without applying clamps."""
        joints = np.asarray(joints_deg, dtype=np.float64)
        if joints.shape != (6,) or np.any(~np.isfinite(joints)):
            raise ValueError("validated joint target must be finite with shape (6,)")
        with self.position_lock:
            self._target_joint_angles = joints.copy()
            self._teleop_generation = None
            self._agilex_teleop_until = 0.0
            self._agilex_teleop_holding = False
            self._active_motion_speed_cap_percent = None

    def _set_hold_speed_cap_locked(self) -> None:
        """Cap movement speed for targets created by hold_current_position()."""
        current_speed = int(self._motionctrl_speed_percent)
        self._active_motion_speed_cap_percent = min(
            current_speed,
            self._hold_current_position_speed_percent,
        )

    def _freeze_teleop_target_locked(self) -> None:
        """Drop unsent motion, preserving the last transmitted target and mode."""
        sent = getattr(self, "_last_sent_joint_angles", None)
        if sent is not None:
            self._target_joint_angles = sent.copy()
        self._agilex_teleop_until = 0.0
        # A queued first JS target may never have reached CAN. Preserve the
        # last transmitted mode too, not merely the mode requested by the parent.
        self._agilex_teleop_holding = (
            getattr(self, "_last_sent_joint_mit_mode", 0) == 0xAD
        )
        self._active_motion_speed_cap_percent = (
            None
            if self._agilex_teleop_holding
            else getattr(
                self,
                "_last_sent_joint_speed_cap",
                self._hold_current_position_speed_percent,
            )
        )

    def _teleop_joint_mode_locked(self) -> int:
        """An expired streaming lease freezes motion; it never swaps controllers."""
        deadline = getattr(self, "_agilex_teleop_until", 0.0)
        if deadline > 0.0 and time.monotonic() >= deadline:
            self._freeze_teleop_target_locked()
        active = deadline > time.monotonic() or getattr(self, "_agilex_teleop_holding", False)
        return 0xAD if active and self._active_motion_speed_cap_percent is None else 0x00

    def hold_teleop_position(self) -> np.ndarray | None:
        """Hold without a mode/target jump; never enables a disabled arm.

        Return the controller's accepted hold target for the next clutch anchor.
        Physical settling is still possible; this freezes commanded motion.
        """
        # An in-flight command from the old clutch must not resume this hold.
        self._teleop_generation = None
        if not self.is_robot_enabled():
            return None
        with self._command_handoff_lock:
            if self.get_control_mode() != PiperController.ControlMode.JOINT_SPACE:
                return None
            with self.position_lock:
                active = (getattr(self, "_agilex_teleop_until", 0.0) > 0.0
                          or getattr(self, "_agilex_teleop_holding", False))
                if active:
                    self._freeze_teleop_target_locked()
                    return self._target_joint_angles.copy()
            if not self.hold_current_position(allow_resume=False):
                return None
            return self.get_target_joint_angles()

    def prepare_teleop(self, generation: int) -> np.ndarray | None:
        """Fence a clutch and acknowledge a hardware-valid hold with fresh feedback."""
        with self._command_handoff_lock:
            held = self.hold_teleop_position()
            if held is None:
                return None
            raw = self._sdk_call("GetArmJointMsgs", self.piper.GetArmJointMsgs)
            measured, error = self._joint_feedback_to_degrees(raw)
            age = self._feedback_timestamp_age_s(raw)
            try:
                if (
                    error
                    or measured is None
                    or age is None
                    or age > self._startup_feedback_max_age_s
                ):
                    raise ValueError(f"unusable joint feedback: {error}, age={age}")
                recovery_joint_bounds(
                    measured, self.EFFECTIVE_JOINT_LIMITS, self.JOINT_LIMITS
                )
                recovery_joint_bounds(
                    held, self.EFFECTIVE_JOINT_LIMITS, self.JOINT_LIMITS
                )
            except ValueError as exc:
                self._log.warning("Teleop clutch %s refused: %s", generation, exc)
                return None
            with self.position_lock:
                if self._stop_hold_latched.is_set():
                    return None
                self._teleop_generation = int(generation)
            return held

    def _set_hold_joint_target_angles(self, joint_angles: np.ndarray) -> None:
        """Set a hold target and cap any resulting movement speed."""
        with self.position_lock:
            self._teleop_generation = None
            self._agilex_teleop_until = 0.0
            self._agilex_teleop_holding = False
            angles = np.asarray(joint_angles, dtype=np.float64)
            # A soft-margin clamp would command a move, not a hold. Callers
            # must validate feedback against the SDK's actual limits first.
            self._target_joint_angles = angles.copy()
            self._set_hold_speed_cap_locked()

    def _set_hold_target_pose(self, transform: np.ndarray) -> None:
        """Set a hold pose target and cap any resulting movement speed."""
        with self.position_lock:
            if transform.shape != (4, 4):
                raise ValueError("transform must be a 4x4 transformation matrix")
            self._target_pose = transform.copy().astype(np.float64)
            self._set_hold_speed_cap_locked()

    def _sync_targets_to_measured_state(self) -> None:
        """Read the robot's current joint/gripper state and set it as the target.

        This prevents the control loop from snapping the arm to HOME on startup
        when the robot is already in an arbitrary position.
        """
        try:
            current_mode = self.get_control_mode()
            if current_mode == PiperController.ControlMode.JOINT_SPACE:
                joints = self.get_current_joint_angles()
                if joints is not None and len(joints) >= 6:
                    self.set_target_joint_angles(joints)
                    self._log.info(f"Synced initial joint targets to measured state")
            elif current_mode == PiperController.ControlMode.END_EFFECTOR:
                pose = self.get_current_end_effector_pose()
                if pose is not None and pose.shape == (4, 4):
                    self.set_target_pose(pose)
                    self._log.info(f"Synced initial pose target to measured state")

            gripper = self.get_current_gripper_open_value()
            if gripper is not None:
                self.set_gripper_open_value(gripper)
        except Exception as e:
            self._log.warning(f"Warning: could not sync targets to measured state: {e}")

    def get_control_mode(self) -> "PiperController.ControlMode":
        """Get the current control mode.

        Returns:
            Current control mode (PiperController.ControlMode.END_EFFECTOR or PiperController.ControlMode.JOINT_SPACE)
        """
        with self.state_lock:
            return self._control_mode

    def set_control_mode(self, mode: "PiperController.ControlMode") -> None:
        """Set the control mode.

        Args:
            mode: Control mode (PiperController.ControlMode.END_EFFECTOR or PiperController.ControlMode.JOINT_SPACE)
        """
        with self.state_lock:
            old_mode = self._control_mode
            self._control_mode = mode
            if self.debug_mode:
                self._log.info(f"Control mode changed: {old_mode.value} -> {mode.value}")

    @staticmethod
    def _pose_6d_to_4x4(pose_6d: np.ndarray) -> np.ndarray:
        """Convert 6D pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix.

        Args:
            pose_6d: 6D pose [x, y, z in mm, rx, ry, rz in degrees]

        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = pose_6d[:3]  # Position in mm
        # Convert Euler angles (degrees) to rotation matrix
        rot = Rotation.from_euler("xyz", pose_6d[3:6], degrees=True)
        transform[:3, :3] = rot.as_matrix()
        return transform

    @staticmethod
    def _pose_4x4_to_6d(transform: np.ndarray) -> np.ndarray:
        """Convert 4x4 transformation matrix to 6D pose [x, y, z, rx, ry, rz].

        Args:
            transform: 4x4 transformation matrix

        Returns:
            6D pose [x, y, z in mm, rx, ry, rz in degrees]
        """
        pose_6d = np.zeros(6, dtype=np.float64)
        pose_6d[:3] = transform[:3, 3]  # Position in mm
        # Convert rotation matrix to Euler angles (degrees)
        rot = Rotation.from_matrix(transform[:3, :3])
        pose_6d[3:6] = rot.as_euler("xyz", degrees=True)
        return pose_6d

    def get_target_pose(self) -> np.ndarray:
        """Get the current target pose.

        Returns:
            4x4 transformation matrix representing the target pose
        """
        with self.position_lock:
            return self._target_pose.copy()

    def set_target_pose(
        self,
        transform: np.ndarray,
    ) -> None:
        """Set target pose from a 4x4 transformation matrix.

        Args:
            transform: 4x4 transformation matrix
        """
        with self.position_lock:
            if transform.shape != (4, 4):
                raise ValueError("transform must be a 4x4 transformation matrix")
            self._target_pose = transform.copy().astype(np.float64)
            self._active_motion_speed_cap_percent = None

            if self.debug_mode:
                pos = self._target_pose[:3, 3]
                rot_euler = Rotation.from_matrix(self._target_pose[:3, :3]).as_euler(
                    "xyz", degrees=True
                )
                self._log.info(
                    f"Target pose set: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], rot=[{rot_euler[0]:.1f}, {rot_euler[1]:.1f}, {rot_euler[2]:.1f}]"
                )

    def update_target_pose(
        self,
        linear_delta: np.ndarray,
        angular_delta: np.ndarray,
    ) -> None:
        """Update target position with relative deltas.

        Args:
            linear_delta: Change in position [dx, dy, dz] in millimeters
            angular_delta: Change in orientation [droll, dpitch, dyaw] in degrees
        """
        with self.position_lock:
            lin_delta = np.array(linear_delta, dtype=np.float64)
            ang_delta = np.array(angular_delta, dtype=np.float64)
            self._active_motion_speed_cap_percent = None

            # Update position (in mm)
            self._target_pose[:3, 3] += lin_delta

            # Update orientation: apply rotation delta to current rotation
            current_rot = Rotation.from_matrix(self._target_pose[:3, :3])
            delta_rot = Rotation.from_euler("xyz", ang_delta, degrees=True)
            new_rot = current_rot * delta_rot
            self._target_pose[:3, :3] = new_rot.as_matrix()

            if self.debug_mode:
                pos = self._target_pose[:3, 3]
                rot_euler = new_rot.as_euler("xyz", degrees=True)
                self._log.info(
                    f"Pose updated: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                    f"rot=[{rot_euler[0]:.1f}, {rot_euler[1]:.1f}, {rot_euler[2]:.1f}]"
                )

    def get_gripper_open_value(self) -> float:
        """Get the current gripper open value.

        Returns:
            Current gripper open value normalized (0.0 to 1.0).
        """
        with self.position_lock:
            # Convert from degrees to normalized (0.0 to 1.0)
            normalized = (
                self._gripper_open_value_degrees - self.GRIPPER_DEGREES_MIN
            ) / self.GRIPPER_DEGREES_RANGE
            return float(np.clip(normalized, 0.0, 1.0))

    def set_gripper_open_value(self, gripper_open_value: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value: Gripper open value normalized (0.0 to 1.0).
        """
        with self.position_lock:
            # Clamp normalized value to [0.0, 1.0]
            normalized = float(np.clip(gripper_open_value, 0.0, 1.0))
            # Convert to degrees for internal storage
            self._gripper_open_value_degrees = (
                normalized * self.GRIPPER_DEGREES_RANGE + self.GRIPPER_DEGREES_MIN
            )

            if self.debug_mode:
                self._log.info(f"Gripper updated: {normalized:.3f} (normalized)")

    def update_gripper_open_value(self, gripper_open_value_delta: float) -> None:
        """Update target gripper position.

        Args:
            gripper_open_value_delta: Gripper open value delta normalized (-1.0 to 1.0).
        """
        with self.position_lock:
            # Get current normalized value
            current_normalized = (
                self._gripper_open_value_degrees - self.GRIPPER_DEGREES_MIN
            ) / self.GRIPPER_DEGREES_RANGE
            # Add delta and clamp to [0.0, 1.0]
            new_normalized = float(
                np.clip(current_normalized + gripper_open_value_delta, 0.0, 1.0)
            )
            # Convert back to degrees for internal storage
            self._gripper_open_value_degrees = (
                new_normalized * self.GRIPPER_DEGREES_RANGE + self.GRIPPER_DEGREES_MIN
            )

            if self.debug_mode:
                self._log.info(f"Gripper updated: {new_normalized:.3f} (normalized)")

    def get_target_joint_angles(self) -> np.ndarray:
        """Get the current target joint angles.

        Returns:
            Current target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            stop_target = self._stop_hold_target
            if self._stop_hold_latched.is_set() and stop_target is not None:
                return stop_target.copy()
            return self._target_joint_angles.copy()

    def get_effective_joint_limits_deg(self) -> np.ndarray:
        """Get SDK joint limits after application-level soft margins.

        Returns:
            Effective joint limits as a (6, 2) array in degrees, where each row
            is [lower, upper] for joint1..joint6.
        """
        return self.EFFECTIVE_JOINT_LIMITS.copy()

    def set_target_joint_angles(
        self,
        joint_angles: np.ndarray,
        *,
        use_agilex_teleop: bool = False,
        teleop_generation: int | None = None,
    ) -> None:
        """Set target joint angles.

        Args:
            joint_angles: Target joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        with self.position_lock:
            angles = np.array(joint_angles, dtype=np.float64)

            if teleop_generation is not None:
                if (
                    not use_agilex_teleop
                    or teleop_generation != getattr(self, "_teleop_generation", None)
                    or self._stop_hold_latched.is_set()
                ):
                    raise ValueError("Teleop target belongs to an inactive clutch")
                # Validate hard limits before containment. Only this acknowledged
                # clutch may recover inside a margin; dashboard/preset commands
                # retain their ordinary soft-limit clamp.
                recovery_joint_bounds(
                    angles, self.EFFECTIVE_JOINT_LIMITS, self.JOINT_LIMITS
                )
                clamped_angles = clamp_recovery_step(
                    self._target_joint_angles,
                    angles,
                    self.EFFECTIVE_JOINT_LIMITS,
                    self.JOINT_LIMITS,
                )
                # Merely squeezing the trigger must not change motor mode or
                # remove a hold's speed cap. Compare actual SDK command units.
                unchanged = np.array_equal(
                    np.rint(clamped_angles * 1000),
                    np.rint(self._target_joint_angles * 1000),
                )
                active = getattr(self, "_agilex_teleop_until", 0.0) > 0.0 or getattr(
                    self, "_agilex_teleop_holding", False
                )
                if unchanged and not active:
                    return
            else:
                self._teleop_generation = None

                # Ordinary command sources do not gain access to the soft margin.
                clamped_angles = np.clip(
                    angles,
                    self.EFFECTIVE_JOINT_LIMITS[:, 0],
                    self.EFFECTIVE_JOINT_LIMITS[:, 1],
                )

            self._target_joint_angles = clamped_angles
            self._agilex_teleop_until = time.monotonic() + 0.1 if use_agilex_teleop else 0.0
            self._agilex_teleop_holding = False
            self._active_motion_speed_cap_percent = None

            if self.debug_mode:
                self._log.info(f"Target joint angles set: {self._target_joint_angles}")

    def update_target_joint_angles(self, joint_deltas: np.ndarray) -> None:
        """Update target joint angles with relative deltas.

        Args:
            joint_deltas: Change in joint angles [dj1, dj2, dj3, dj4, dj5, dj6] in degrees
        """
        with self.position_lock:
            self._teleop_generation = None
            self._agilex_teleop_until = 0.0
            deltas = np.array(joint_deltas, dtype=np.float64)
            self._agilex_teleop_holding = False
            new_joint_angles = self._target_joint_angles + deltas

            # Clamp joint angles to effective limits (SDK limits minus soft margin).
            self._target_joint_angles = np.clip(
                new_joint_angles, self.EFFECTIVE_JOINT_LIMITS[:, 0], self.EFFECTIVE_JOINT_LIMITS[:, 1]
            )
            self._active_motion_speed_cap_percent = None

            if self.debug_mode:
                self._log.info(f"Joint angles updated: {self._target_joint_angles}")

    def move_to_home(self, speed_percent: int | None = None) -> bool:
        """Move robot to home position based on current control mode.

        Args:
            speed_percent: Optional 1..100 cap on the homing motion speed. When
                None (the default), homing runs at the current global motion
                speed. The cap self-clears on the next teleop target set, so a
                slow homing speed does not persist into subsequent teleop.
        """
        try:
            # Ensure robot is enabled
            if not self._enable_robot(timeout_s=2.0):
                self._log.error(f"Failed to enable robot before homing")
                self.log_hardware_diagnostics("homing enable failure", force=True)
                return False

            speed_cap = (
                None
                if speed_percent is None
                else int(np.clip(int(speed_percent), 1, 100))
            )

            current_mode = self.get_control_mode()
            with self.position_lock:
                self._teleop_generation = None
                self._agilex_teleop_until = 0.0
                self._agilex_teleop_holding = False
                if current_mode == PiperController.ControlMode.END_EFFECTOR:
                    self._target_pose = self.HOME_POSE.copy()
                    self._active_motion_speed_cap_percent = speed_cap
                    if self.debug_mode:
                        self._log.info("Moved to home pose (end effector mode)")
                elif current_mode == PiperController.ControlMode.JOINT_SPACE:
                    self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()
                    self._active_motion_speed_cap_percent = speed_cap
                    if self.debug_mode:
                        self._log.info("Moved to home joint angles (joint space mode)")
                else:
                    self._log.warning(
                        f"Unknown control mode; cannot move to home position"
                    )
                    return False

            self._log.info(f"Home position set")
            return True

        except Exception as e:
            self._log.error(f"Home position error: {e}")
            return False

    def _poll_teach_mode(self, *, sdk_locked: bool = False) -> bool:
        """Latch physical Teach; only fresh normal status can establish exit.

        ctrl_mode is the live mode (0x02 = Teach). teach_status can remain 1
        after a button-driven teach, so it is not an exit signal. A missing,
        stale or nonfinite status must never turn Teach into normal motion.
        The SDK getter only reads its CAN receive cache.
        """
        with self._stop_hold_lock:
            self._teach_status_valid = False
            if not (hasattr(self, "piper") and self.piper is not None):
                return self._teach_mode_active
            msg = (
                self.piper.GetArmStatus() if sdk_locked
                else self._sdk_call_nonblocking("GetArmStatus", self.piper.GetArmStatus)
            )
            try:
                mode = self._sdk_enum_value(msg.arm_status.ctrl_mode)
            except (AttributeError, TypeError, ValueError):
                return self._teach_mode_active
            if not np.isfinite(mode) or mode not in (0, 1, self.CTRL_MODE_TEACH):
                return self._teach_mode_active
            # Even a stale Teach frame is enough to fence motion. Exiting
            # requires fresh status, because replaying a cached normal frame
            # must not reclaim a hand-moved arm.
            if mode == self.CTRL_MODE_TEACH:
                if not self._teach_mode_active:
                    self._stop_hold_generation += 1
                    self._stop_hold_latched.set()
                    self._teleop_generation = None
                    self._stop_hold_target = None
                    self._was_teaching = True
                    self._teach_interlock = True
                    self._log.warning(
                        "Physical Teach entered; software motion latched off. "
                        "Exit Teach, then use dashboard Enable to resume."
                    )
                self._teach_mode_active = True
                return True
            age = self._feedback_timestamp_age_s(msg)
            if age is None or not np.isfinite(age) or age > 0.1:
                return self._teach_mode_active
            self._teach_status_valid = True
            self._teach_mode_active = False
            return False

    def _handle_teach_transition(self) -> bool:
        """Leave Teach in a fixed hold, with all ordinary motion still fenced."""
        if not self._teach_guard_enabled:
            return False
        with self._stop_hold_lock:
            if self._poll_teach_mode():
                return True  # Do not fight the operator with any stored hold.
            if self._was_teaching:
                if not self._teach_status_valid:
                    return True
                # Consume once. Invalid joint feedback must not be retried
                # into an automatically enabled hold at a later position.
                self._was_teaching = False
                self._capture_stop_hold_target()
                self._log.warning(
                    "Physical Teach exited; %s. Software motion remains "
                    "latched off until dashboard Enable.",
                    "fixed measured hold requested; physical hold unverified"
                    if self._send_stop_hold()
                    else "hold UNCONFIRMED; check arm before proceeding",
                )
                return True
            return False

    def control_loop(self) -> None:
        """Main robot control loop running at robot_rate Hz.

        This should be run in a separate thread.
        Only sends commands when robot is in ENABLED state.
        """
        # Ask for real-time scheduling from inside the loop thread (the call
        # applies to the caller). Best-effort: without an rtprio limit this is a
        # no-op and the loop runs on CFS exactly as before.
        rt_prio = try_realtime_priority()
        if rt_prio is not None:
            self._log.info("control loop running SCHED_FIFO prio=%d", rt_prio)
        else:
            self._log.debug(
                "control loop on default scheduler "
                "(grant 'rtprio' in limits.conf for deterministic timing)"
            )

        loop_period = 1.0 / max(1e-6, self.robot_rate)
        next_deadline = time.monotonic()

        while self.running.is_set():
            iter_start = time.monotonic()
            self._last_control_loop_iter_ts = iter_start
            try:
                # Teach must also supersede an existing Stop target and must
                # be observed while disabled or while the CAN breaker is open.
                if self._handle_teach_transition():
                    pass
                elif self._stop_hold_latched.is_set():
                    # Retry independently of the motion circuit breaker and
                    # software-enabled flag (enable may have been in flight).
                    if (self._stop_hold_target is not None
                            or time.monotonic() - self._last_stop_hold_tx >= 0.1):
                        with self._stop_hold_lock:
                            if self._stop_hold_latched.is_set():
                                self._send_stop_hold()
                elif self.is_robot_enabled():
                    # Circuit breaker gate: skip TX when bus is known-bad.
                    if not self._circuit_breaker_check():
                        pass  # Breaker open — fall through to sleep.
                    elif not self._is_can_bus_healthy():
                        # Bus unhealthy before sending — don't add more TX errors.
                        self._circuit_breaker_record(success=False)
                    else:
                        with self._command_handoff_lock:
                            current_mode = self.get_control_mode()

                            # Copy targets without holding the lock across (potentially slow) SDK I/O.
                            with self.position_lock:
                                target_pose = self._target_pose.copy()
                                agilex_mit = self._teleop_joint_mode_locked()
                                target_joint_angles = self._target_joint_angles.copy()
                                gripper_degrees = float(self._gripper_open_value_degrees)
                                motion_speed_percent = int(self._motionctrl_speed_percent)
                                if self._active_motion_speed_cap_percent is not None:
                                    motion_speed_percent = min(
                                        motion_speed_percent,
                                        int(self._active_motion_speed_cap_percent),
                                    )

                            if current_mode == PiperController.ControlMode.END_EFFECTOR:
                                self._ensure_motionctrl(
                                    move_mode=0x00,
                                    speed_percent=motion_speed_percent,
                                )
                                self._send_end_effector_command(target_pose)
                            elif current_mode == PiperController.ControlMode.JOINT_SPACE:
                                self._ensure_motionctrl(
                                    move_mode=0x01,
                                    speed_percent=motion_speed_percent,
                                    mit_mode=agilex_mit,
                                )
                                self._send_joint_command(target_joint_angles)
                            else:
                                self._warn_rate_limited(
                                    "unknown_control_mode",
                                    f"{self._log_prefix()} Unknown control mode: {current_mode}",
                                    interval_s=1.0,
                                )

                            self._send_gripper_command(gripper_degrees)

                        # Post-check: did the bus degrade during our sends?
                        iter_ok = self._is_can_bus_healthy()
                        self._circuit_breaker_record(success=iter_ok)
                else:
                    # Robot is not enabled, just sleep without sending commands
                    if self.debug_mode:
                        self._log.debug("Control loop: Robot not enabled, skipping command")

                # Maintain loop rate (monotonic); detect overruns.
                next_deadline += loop_period
                sleep_s = next_deadline - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # Overrun: resync schedule and optionally warn.
                    next_deadline = time.monotonic()
                    if self._diagnostics_enabled or self.debug_mode:
                        self._warn_rate_limited(
                            "control_loop_overrun",
                            f"{self._log_prefix()} control_loop overrun by {-sleep_s * 1000.0:.1f}ms",
                            interval_s=1.0,
                        )

            except Exception as e:
                self._record_io_error("control_loop", e)
                self._circuit_breaker_record(success=False)
                time.sleep(0.01)

    def _ensure_motionctrl(
        self, move_mode: int, speed_percent: int | None = None, mit_mode: int = 0x00
    ) -> None:
        """Set MotionCtrl_2 mode with optional keepalive throttling."""
        now = time.monotonic()
        raw_speed = (
            self._motionctrl_speed_percent
            if speed_percent is None
            else speed_percent
        )
        speed = int(
            np.clip(
                raw_speed,
                1,
                100,
            )
        )
        should_send = self._motionctrl_every_loop
        if getattr(self, "_last_motionctrl_mit_mode", None) != mit_mode:
            should_send = True
        if self._last_motionctrl_move_mode != move_mode:
            should_send = True
        if self._last_motionctrl_speed_percent != speed:
            should_send = True
        if (
            (not should_send)
            and self._motionctrl_keepalive_s > 0
            and (now - self._last_motionctrl_ts) >= self._motionctrl_keepalive_s
        ):
            should_send = True

        if not should_send:
            return

        speed_changed = self._last_motionctrl_speed_percent != speed
        try:
            ok = self._sdk_ok(
                "MotionCtrl_2",
                lambda: self.piper.MotionCtrl_2(0x01, move_mode, speed, mit_mode),
            )
            if not ok:
                # A late command fenced by STOP did not change hardware mode.
                # Keep the last transmitted mode for the stop-owned hold.
                return
            if speed_changed or getattr(self, "_last_motionctrl_mit_mode", None) != mit_mode:
                # Only on change, so this is the ground truth for "what speed is
                # the arm actually moving at" without spamming the loop rate.
                self._log.info(
                    "%s MotionCtrl_2 speed %d%% (move_mode=0x%02x, base=%d%%, "
                    "cap=%s, mit=0x%02x)",
                    self._log_prefix(),
                    speed,
                    move_mode,
                    int(self._motionctrl_speed_percent),
                    self._active_motion_speed_cap_percent,
                    mit_mode,
                )
            self._last_motionctrl_ts = now
            self._last_motionctrl_move_mode = move_mode
            self._last_motionctrl_mit_mode = mit_mode
            self._last_motionctrl_speed_percent = speed
            if ok:
                self._last_motionctrl_ok_ts = now
        except Exception as e:
            self._record_io_error("MotionCtrl_2", e)

    def _send_end_effector_command(self, transform: np.ndarray) -> None:
        """Send end-effector pose command to the robot.

        Args:
            transform: 4x4 transformation matrix
        """
        try:
            # Convert 4x4 matrix to 6D pose for SDK
            pose_6d = self._pose_4x4_to_6d(transform)

            # Convert from mm/degrees to piper SDK units (0.001mm/0.001degrees)
            X = round(float(pose_6d[0] * 1000))
            Y = round(float(pose_6d[1] * 1000))
            Z = round(float(pose_6d[2] * 1000))
            RX = round(float(pose_6d[3] * 1000))
            RY = round(float(pose_6d[4] * 1000))
            RZ = round(float(pose_6d[5] * 1000))

            ok = self._sdk_ok(
                "EndPoseCtrl",
                lambda: self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ),
            )
            if ok:
                now = time.monotonic()
                self._last_any_cmd_tx_ts = now
                self._last_pose_cmd_tx_ts = now

        except Exception as e:
            self._record_io_error("EndPoseCtrl", e)

    def _send_joint_command(self, joint_angles: np.ndarray) -> None:
        """Send joint angles command to the robot.

        Args:
            joint_angles: Joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        try:
            # Convert from degrees to piper SDK units (0.001degrees)
            joint_1 = round(float(joint_angles[0] * 1000))
            joint_2 = round(float(joint_angles[1] * 1000))
            joint_3 = round(float(joint_angles[2] * 1000))
            joint_4 = round(float(joint_angles[3] * 1000))
            joint_5 = round(float(joint_angles[4] * 1000))
            joint_6 = round(float(joint_angles[5] * 1000))

            ok = self._sdk_ok(
                "JointCtrl",
                lambda: self.piper.JointCtrl(
                    joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
                ),
            )
            if ok:
                now = time.monotonic()
                self._last_any_cmd_tx_ts = now
                self._last_joint_cmd_tx_ts = now
                with self.position_lock:
                    self._last_sent_joint_angles = np.asarray(joint_angles).copy()
                    self._last_sent_joint_mit_mode = getattr(
                        self, "_last_motionctrl_mit_mode", 0
                    )
                    self._last_sent_joint_speed_cap = getattr(
                        self, "_last_motionctrl_speed_percent", None
                    )

        except Exception as e:
            self._record_io_error("JointCtrl", e)

    def _send_gripper_command(self, gripper_open_value_degrees: float) -> None:
        """Send gripper command to the robot.

        Args:
            gripper_open_value_degrees: Gripper open value in degrees
        """
        try:
            now = time.monotonic()
            # Throttle gripper TX to reduce bus load, but still keep it updated periodically.
            if self._gripper_tx_period_s > 0:
                if (
                    (now - self._last_gripper_cmd_tx_ts) < self._gripper_tx_period_s
                    and abs(gripper_open_value_degrees - self._last_gripper_cmd_degrees)
                    < self._gripper_tx_epsilon_deg
                ):
                    return

            # Convert from degrees to piper SDK units (0.001degrees)
            gripper_value = round(gripper_open_value_degrees * 1000)
            ok = self._sdk_ok(
                "GripperCtrl",
                lambda: self.piper.GripperCtrl(gripper_value, 1000, 0x01, 0),
            )
            if ok:
                now = time.monotonic()
                self._last_any_cmd_tx_ts = now
                self._last_gripper_cmd_tx_ts = now
                self._last_gripper_cmd_degrees = float(gripper_open_value_degrees)

        except Exception as e:
            self._record_io_error("GripperCtrl", e)

    def _read_stop_hold_feedback(self) -> tuple[np.ndarray | None, float | None]:
        """Read once, without waiting for a moving arm to become stationary."""
        raw = self._sdk_call("GetArmJointMsgs", self.piper.GetArmJointMsgs)
        joints, error = self._joint_feedback_to_degrees(raw)
        age = self._feedback_timestamp_age_s(raw)
        stamp = self._coerce_timestamp_seconds(getattr(raw, "time_stamp", None))
        epsilon = self._resume_hardware_limit_epsilon_deg
        if (error or joints is None or age is None or age > 0.1
                or stamp is None or not np.isfinite(stamp)
                or np.any(joints < self.JOINT_LIMITS[:, 0] - epsilon)
                or np.any(joints > self.JOINT_LIMITS[:, 1] + epsilon)):
            return None, None
        # Only resolve sub-encoder-count rounding at the hardware bounds.
        # Applying the normal soft margin here would turn a hold into a move.
        return np.clip(joints, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1]), stamp

    def _send_stop_hold(self) -> bool:
        """Transmit only the immutable stop target while the latch is held.

        A trajectory pause alone did not interrupt US MOVE J presets. Replace
        that destination, bypassing only the normal motion fence, never motor
        enable/reset. Keep MIT's existing servo mode when stopping MIT teleop.
        Caller holds _stop_hold_lock; normal motion remains fenced at SDK I/O.
        """
        if self._teach_guard_enabled:
            self._poll_teach_mode()
        if (not self._stop_hold_latched.is_set()
                or self._teach_mode_active or self._was_teaching):
            return False
        self._last_stop_hold_tx = time.monotonic()
        if self._stop_hold_target is None:
            self._sdk_ok(
                "MotionCtrl_1(stop_hold)", lambda: self.piper.MotionCtrl_1(0, 1, 0)
            )
            return False  # Pause was attempted; an active hold is unavailable.
        target = tuple(round(float(q) * 1000) for q in self._stop_hold_target)
        # Install the replacement BEFORE selecting joint mode, so switching
        # from Cartesian control cannot expose an old stored joint target.
        if not self._sdk_ok("JointCtrl(stop_hold)", lambda: self.piper.JointCtrl(*target)):
            return False
        return self._sdk_ok(
            "MotionCtrl_2(stop_hold)",
            lambda: self.piper.MotionCtrl_2(1, 1, 10, self._stop_hold_mit_mode),
        )

    def _capture_stop_hold_target(self) -> None:
        """Replace the interrupted plan with one fresh, hardware-valid target.

        Caller holds _stop_hold_lock and has already latched ordinary motion.
        """
        self._sdk_ok(
            "MotionCtrl_1(stop_hold)", lambda: self.piper.MotionCtrl_1(0, 1, 0)
        )
        if self._teach_mode_active or self._was_teaching:
            return
        self._stop_hold_target, _ = self._read_stop_hold_feedback()
        self._stop_hold_mit_mode = (
            0xAD if getattr(self, "_last_motionctrl_mit_mode", 0) == 0xAD else 0
        )
        if self._stop_hold_target is not None:
            self._sdk_ok(
                "MotionCtrl_1(clear_stopped_plan)",
                lambda: self.piper.MotionCtrl_1(0, 3, 0),
            )

    def stop_and_hold(self) -> bool:
        """Fence motion, interrupt the plan, and freeze one fresh measured pose.

        Always attempts pause before reading feedback. Invalid feedback leaves
        the latch on and returns failure, never inventing an angle or cutting
        motor power. A successful send still requires verify_stop_hold().
        """
        with self._stop_hold_lock:
            if self._teach_guard_enabled:
                self._poll_teach_mode()
            already_stopped = self._stop_hold_latched.is_set()
            self._stop_hold_generation += 1
            self._stop_hold_latched.set()
            self._teleop_generation = None
            if not already_stopped:
                self._capture_stop_hold_target()
            ok = self._send_stop_hold()
        self._log.warning(
            "STOP & HOLD latched; fixed-position command %s",
            "sent; feedback verification required" if ok
            else "UNAVAILABLE; check arm and use physical emergency stop if moving",
        )
        return ok

    def verify_stop_hold(self, timeout_s: float = 0.6) -> bool:
        """Require fresh, settled feedback near the frozen pose; never move.

        This is an encoder-based observation, not independent physical proof.
        Three new samples spanning 150 ms must stay within 0.05 degrees of one
        another and 0.5 degrees of the captured target. Missing/frozen feedback
        and continued travel must not produce a successful dashboard response.
        """
        with self._stop_hold_lock:
            generation = self._stop_hold_generation
            target = self._stop_hold_target
            target = None if target is None else target.copy()
        if target is None:
            return False
        deadline = time.monotonic() + max(0.0, min(timeout_s, 2.0))
        last_stamp = None
        samples: list[np.ndarray] = []
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            if (not self._stop_hold_latched.is_set()
                    or generation != self._stop_hold_generation):
                return False
            joints, stamp = self._read_stop_hold_feedback()
            if joints is None:
                return False
            if last_stamp is None or stamp > last_stamp:
                last_stamp = stamp
                if np.max(np.abs(joints - target)) > 0.5:
                    samples.clear()
                    stable_since = time.monotonic()
                else:
                    if not samples:
                        stable_since = time.monotonic()
                    samples.append(joints)
                    if np.max(np.ptp(np.asarray(samples), axis=0)) > 0.05:
                        samples = [joints]
                        stable_since = time.monotonic()
                    if len(samples) >= 3 and time.monotonic() - stable_since >= 0.15:
                        return (self._stop_hold_latched.is_set()
                                and generation == self._stop_hold_generation)
            time.sleep(0.02)
        self._log.error("STOP & HOLD not confirmed: feedback stale or still moving")
        return False

    def get_stop_hold_generation(self) -> int:
        """Token used to prevent a queued Enable from releasing a newer stop."""
        return self._stop_hold_generation

    def is_stop_hold_latched(self) -> bool:
        """Whether physical Teach or Stop has fenced ordinary software motion."""
        return self._stop_hold_latched.is_set()

    def emergency_stop(self) -> bool:
        """Emergency stop - immediately halt all motion."""
        try:
            self._log.warning("EMERGENCY STOP")

            # Send emergency stop commands to hardware
            self._sdk_ok(
                "MotionCtrl_1", lambda: self.piper.MotionCtrl_1(0x02, 0, 0)
            )
            self._sdk_ok(
                "GripperCtrl(emergency_stop)",
                lambda: self.piper.GripperCtrl(0, 0, 0x00, 0),
            )

            # Disable robot hardware
            if not self._disable_robot(timeout_s=5.0):
                self._log.error(
                    f"Timed out disabling robot during emergency stop"
                )
                return False

            self._log.info("emergency stop completed")
            return True

        except Exception as e:
            self._log.error(f"Emergency stop error: {e}")
            return False

    def _uninterruptible_sleep(self, duration_s: float) -> bool:
        """Sleep ~``duration_s`` seconds, absorbing Ctrl+C.

        A repeated SIGINT during the shutdown settle must not abort the
        sequence, otherwise the motors are left in gravity-compensation and the
        arm drops abruptly. Returns True if at least one interrupt was absorbed.
        """
        deadline = time.monotonic() + duration_s
        interrupted = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                time.sleep(min(remaining, 0.2))
            except KeyboardInterrupt:
                interrupted = True
        return interrupted

    def graceful_stop(self) -> bool:
        """Gracefully stop the robot and move to home position.

        The descent + hardware-disable sequence is resilient to repeated SIGINT
        (Ctrl+C). ``KeyboardInterrupt`` is a ``BaseException`` (not an
        ``Exception``), so a second Ctrl+C during the 5 s settle previously
        propagated past the handler and skipped the disable steps, leaving the
        arm in gravity-comp to drop abruptly. The disable steps now always run
        via the ``finally`` block, even when interrupted.
        """
        interrupted = False
        settled = False
        start_ts = time.monotonic()
        try:
            self._log.warning(
                "graceful stop activated on %s — beginning safe descent",
                self.can_interface,
            )

            # Disable robot state (this will stop the control loop from sending commands)
            self._set_robot_status_enabled(False)
            self._log.info("control loop disarmed; no further teleop/IK commands will stream")

            self._uninterruptible_sleep(0.1)

            # Motion control to go home
            self._sdk_ok(
                "MotionCtrl_1", lambda: self.piper.MotionCtrl_1(0x01, 0, 0)
            )
            self._sdk_ok(
                "GripperCtrl(graceful_stop)",
                lambda: self.piper.GripperCtrl(0, 0, 0x00, 0),
            )
            self._log.info("home/gripper commands sent; entering 5 s settle window")
            # NOTE: use the logger (not print) for every line in this shutdown
            # path. The logger swallows write errors, whereas a raw print to a
            # broken pipe (e.g. `tee` killed by the same Ctrl-C) raises
            # BrokenPipeError — which previously skipped the disable below.
            self._log.warning(
                "gracefully falling forward for 5 seconds — please hold the robot "
                "if it doesn't reach home after this time"
            )
            if self._uninterruptible_sleep(5.0):
                interrupted = True
                self._log.warning("absorbed Ctrl+C during settle; descent continues uninterrupted")
            settled = True

        except KeyboardInterrupt:
            # An interrupt landed in an unprotected window (an SDK call, the
            # warning log, or the gap before the 5 s settle) rather than inside
            # _uninterruptible_sleep. Do NOT fall through to disable yet -- that
            # cuts driver current (MotionCtrl_1(0x02) below) while the arm is
            # still elevated and lets it drop. Absorb it and still give the
            # operator the full grab window before disabling. The settle is
            # itself interrupt-proof, so continued Ctrl+C mashing is safe.
            interrupted = True
            self._log.warning(
                "Ctrl+C caught before settle completed — absorbing and running the "
                "full 5 s grab window anyway to avoid an abrupt drop"
            )
            self._uninterruptible_sleep(5.0)
            settled = True
        except Exception as e:
            self._log.error("graceful stop error before settle (%s); disabling now", e)
        finally:
            if interrupted:
                self._log.warning("interrupt(s) during shutdown ignored; completing safe disable")
            if not settled:
                self._log.warning("settle window did NOT complete — arm may drop abruptly on disable")
            # Always disable motion + hardware, even if interrupted above.
            try:
                self._sdk_ok(
                    "MotionCtrl_1", lambda: self.piper.MotionCtrl_1(0x02, 0, 0)
                )
                self._log.info("disabling robot drivers...")
                disabled = self._disable_robot(timeout_s=5.0)
            except Exception as e:
                self._log.error("graceful stop disable error: %s", e)
                disabled = False

        elapsed = time.monotonic() - start_ts
        if not disabled:
            self._log.error("timed out disabling robot after %.1f s", elapsed)
            return False

        self._log.info("graceful stop completed in %.1f s (drivers disabled)", elapsed)
        return True

    def resume_robot(
        self,
        force: bool = False,
        *,
        clear_stop_hold: bool = False,
        expected_stop_generation: int | None = None,
    ) -> bool:
        """Resume robot operation by re-enabling it.

        This method can be called after emergency_stop() or graceful_stop()
        to re-enable the robot and resume normal operation. It validates and
        locks measured joint feedback before command streaming is allowed.

        Args:
            force: When True, always run full resume/recovery + target sync path
                even if software state currently reports enabled.
            clear_stop_hold: Explicit operator Enable after STOP & HOLD. Other
                motion owners must not release the stop latch through resume.
            expected_stop_generation: Stop token captured when Enable was requested.
                A queued request cannot release a stop issued after that token.

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._stop_hold_lock:
                if self._teach_guard_enabled:
                    self._poll_teach_mode()
                if (self._teach_mode_active or self._was_teaching
                        or (self._teach_interlock and not self._teach_status_valid)):
                    self._log.warning("Enable refused: exit physical Teach with fresh status first")
                    return False
                stop_generation = self._stop_hold_generation
                if (
                    expected_stop_generation is not None
                    and expected_stop_generation != stop_generation
                ):
                    return False
                stopped = self._stop_hold_latched.is_set()
                if stopped and not clear_stop_hold:
                    self._log.warning("STOP & HOLD is latched; use dashboard Enable")
                    return False
            was_enabled = self.is_robot_enabled()
            if was_enabled and not force and not stopped:
                self._log.info(f"Robot is already enabled")
                return True

            self._log.info("resuming robot...")
            if was_enabled:
                # Stop the control loop from streaming while we validate the
                # measured pose. Hardware may still be enabled until validation
                # succeeds or we explicitly disable on failure.
                self._set_robot_status_enabled(False)

            need_enable = not was_enabled
            transport_recovered = False

            # If the SDK considers itself disconnected (or status read fails),
            # attempt a reconnect first.
            if not self._sdk_connected():
                if not self._recover_transport(
                    reason="SDK disconnected before resume",
                    reset_can_link=self._can_recover_on_resume,
                ):
                    self._log.error(f"Failed to reconnect SDK transport")
                    return False
                transport_recovered = True
                need_enable = True

            validated_joints = self._read_validated_joint_feedback("resume")
            if stop_generation != self._stop_hold_generation:
                return False
            if validated_joints is None:
                if was_enabled and not stopped:
                    # Bad encoder feedback must not turn a failed resume into
                    # a power cut, or leave the previous firmware plan running.
                    self.stop_and_hold()
                else:
                    self._set_robot_status_enabled(False)
                return False

            self._lock_joint_target_without_clamping(validated_joints)
            gripper = self.get_current_gripper_open_value()
            if gripper is not None:
                self.set_gripper_open_value(gripper)
            else:
                self._log.warning(
                    f"Warning: gripper feedback unavailable "
                    "during resume; preserving existing gripper target"
                )

            with self._command_handoff_lock, self._stop_hold_lock:
                if stop_generation != self._stop_hold_generation:
                    return False
                if stopped:
                    # Discard the interrupted plan; never "continue" it. A new
                    # MOVE J starts from validated feedback after explicit Enable.
                    if not self._sdk_ok(
                        "MotionCtrl_1(clear_stopped_plan)",
                        lambda: self.piper.MotionCtrl_1(0, 0x03, 0),
                    ):
                        return False
                    self.set_control_mode(PiperController.ControlMode.JOINT_SPACE)
                    self._last_motionctrl_move_mode = None
                    self._stop_hold_latched.clear()
                    self._stop_hold_target = None
                    self._teach_interlock = False

            # Forced resume (or successful transport reconnect) should clear stale TX state.
            if force or transport_recovered:
                self._reset_circuit_breaker()

            if need_enable:
                if not self._enable_robot(timeout_s=5.0):
                    if self._stop_hold_latched.is_set():
                        return False
                    if self._can_recover_on_resume:
                        self._recover_transport(
                            reason="EnablePiper failed on resume", reset_can_link=True
                        )
                        validated_joints = self._read_validated_joint_feedback(
                            "resume after transport recovery"
                        )
                        if validated_joints is None:
                            self._set_robot_status_enabled(False)
                            return False
                        self._lock_joint_target_without_clamping(validated_joints)
                    if not self._enable_robot(timeout_s=8.0):
                        self._log.error(f"Timed out enabling robot")
                        self._set_robot_status_enabled(False)
                        return False
            elif force or stopped:
                self._log.info(
                    f"Force resume: already enabled; "
                    "validated and locked measured joint target"
                )
                self._set_robot_status_enabled(True)

            if stop_generation != self._stop_hold_generation:
                return False
            self._log.info("robot resumed successfully")
            return True

        except Exception as e:
            self._log.error(f"Resume robot error: {e}")
            return False

    # GetArmStatus (CAN 0x2A1): Teach uses the live control mode, not the latched
    # teach_status field. It installs the same persistent motion fence as Stop.
    CTRL_MODE_TEACH = 0x02

    # Human-readable arm_status codes from GetArmStatus (CAN 0x2A1).
    ARM_STATUS_LABELS = {
        0x00: "normal",
        0x01: "emergency stop",
        0x02: "no solution",
        0x03: "singularity",
        0x06: "joint brake not released",
        0x07: "collision",
        0x09: "joint abnormal",
        0x0A: "other abnormal",
        0x0E: "main-control NTC over-temp",
        0x0F: "brake-resistor NTC over-temp",
    }

    # Subset of MOTOR_FAULT_FLAG_FIELDS that indicate an actual fault (the
    # remaining flag, driver_enable_status, is normal operating state).
    FAULT_FLAG_FIELDS = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "driver_error_status",
        "stall_status",
    )

    def describe_fault_state(self) -> dict[str, Any]:
        """Snapshot the current fault state for logging/diagnostics.

        Reads the arm-level ``arm_status``/``err_code`` and the per-joint driver
        fault flags, and returns ``{has_fault, summary, arm_status_code,
        arm_status_label, err_code, active_flags}``. ``has_fault`` is False when
        the arm reports normal status with no active fault flags.
        """
        feedback = self.get_piper_feedback_status(non_blocking=True) or {}
        flags = self.get_motor_fault_flags(non_blocking=True) or {}

        raw_status = feedback.get("arm_arm_status")
        arm_status_code = None if raw_status is None else int(raw_status)
        raw_err = feedback.get("arm_err_code")
        err_code = None if raw_err is None else int(raw_err)
        arm_status_label = (
            self.ARM_STATUS_LABELS.get(arm_status_code, f"0x{arm_status_code:02X}")
            if arm_status_code is not None
            else "unknown"
        )

        active_flags: list[str] = []
        for field in self.FAULT_FLAG_FIELDS:
            arr = flags.get(field)
            if arr is None:
                continue
            joints = [idx + 1 for idx, val in enumerate(arr) if val]
            if joints:
                active_flags.append(
                    f"{field}@j{','.join(str(j) for j in joints)}"
                )
        for key, label in (
            ("arm_joint_angle_limit_error", "angle_limit"),
            ("arm_joint_communication_error", "comm_error"),
        ):
            arr = feedback.get(key)
            if arr is None:
                continue
            joints = [idx + 1 for idx, val in enumerate(np.asarray(arr)) if val]
            if joints:
                active_flags.append(f"{label}@j{','.join(str(j) for j in joints)}")

        has_fault = (
            (arm_status_code not in (None, 0x00))
            or bool(err_code)
            or bool(active_flags)
        )
        if has_fault:
            parts = [f"arm_status={arm_status_label}"]
            if err_code:
                parts.append(f"err_code=0x{err_code:X}")
            if active_flags:
                parts.append("flags=[" + ", ".join(active_flags) + "]")
            summary = "; ".join(parts)
        else:
            summary = "no fault"
        return {
            "has_fault": has_fault,
            "summary": summary,
            "arm_status_code": arm_status_code,
            "arm_status_label": arm_status_label,
            "err_code": err_code,
            "active_flags": active_flags,
        }

    def log_fault_state(self, step: str, arm_name: str | None = None) -> dict[str, Any]:
        """Print the current fault state + measured position with a step label.

        Used by :meth:`recover_from_fault` to log, at every step, what the fault
        was (or "no fault at arm <name>"). Returns the snapshot dict.
        """
        name = arm_name or self.can_interface
        snap = self.describe_fault_state()
        joints = self.get_current_joint_angles()
        pos = (
            self._format_joint_angles_deg(joints)
            if joints is not None
            else "unknown"
        )
        if snap["has_fault"]:
            self._log.warning(
                f"[recover:{step}] fault at arm {name}: "
                f"{snap['summary']} | pos={pos}"
            )
        else:
            self._log.info(
                f"[recover:{step}] no fault at arm {name} "
                f"| pos={pos}"
            )
        return snap

    def clear_joint_errors(self) -> bool:
        """Clear latched joint driver fault codes (JointConfig clear_err, CAN 0x475).

        ``joint_num=7`` targets all joints; ``clear_err=0xAE`` is the SDK's
        documented "valid" clear value. This clears the latched
        overcurrent/stall/comm-error codes that pin the arm WITHOUT cutting motor
        power, so the arm keeps holding its position. Returns True if the SDK
        accepted the command.
        """
        return self._sdk_ok(
            "JointConfig(clear_err)",
            lambda: self.piper.JointConfig(joint_num=7, clear_err=0xAE),
        )

    def recover_from_fault(
        self, timeout_s: float = 8.0, arm_name: str | None = None
    ) -> bool:
        """Clear latched joint faults and re-enable WITHOUT dropping the arm.

        This is the software equivalent of the manual power-cut recovery, but it
        never removes motor power, so it is safe to use with the arm at height.

        If the arm has no active fault this is a no-op -- a healthy arm is never
        disturbed (disturbing a holding arm is what caused abrupt drops).

        Anti-drop ordering matters here:
          1. Reconnect transport only if the SDK dropped (keeps the bus alive).
          2. Validate a *stable* measured pose and lock it as a speed-capped hold
             target, keeping the arm enabled so the loop streams "stay here" the
             whole time. If joints can't be read stably we abort rather than
             clearing blind -- holding a bad target could fling/drop the arm.
          3. Clear latched joint error codes (JointConfig clear_err -- no power
             cut, so torque is retained throughout).
          4. Re-enable the motors with EnablePiper. We deliberately do NOT send
             MotionCtrl_1(0x02): in this codebase that is the emergency-STOP (it
             disables the drivers and drops the arm), not a resume.
          5. Re-arm the active control mode. The control loop is already locked
             onto the measured pose, so the arm holds in place instead of sagging
             or snapping HOME.

        For a fault that ``clear_err`` cannot release, a physical power-cycle
        (with the arm supported so it does not fall) is the last resort.

        Returns:
            True if faults were cleared and the arm re-enabled and is holding.
        """
        name = arm_name or self.can_interface
        try:
            if self._stop_hold_latched.is_set():
                self._log.warning("Recovery blocked by STOP & HOLD; inspect before Enable")
                return False
            # Step 0: NEVER touch a healthy arm. Disturbing an arm that is already
            # holding fine is what caused the abrupt drop -- so if there is no
            # fault, do nothing at all.
            before = self.log_fault_state("before", arm_name=name)
            if not before["has_fault"]:
                self._log.info(
                    f"No fault at arm {name}; "
                    "skipping recovery (arm left untouched)"
                )
                return True

            self._log.warning("recovering from fault (clear_err, no power cut)...")

            if not self._sdk_connected():
                if not self._recover_transport(
                    reason="SDK disconnected before fault recovery",
                    reset_can_link=self._can_recover_on_resume,
                ):
                    self._log.error(f"Transport reconnect failed; aborting recovery")
                    return False

            # Capture a stable pose to hold. Abort if unavailable -- never clear blind.
            validated = self._read_validated_joint_feedback("fault recovery")
            if validated is None:
                self._log.error(
                    f"No stable joint feedback; aborting recovery "
                    "(refusing to clear faults without a safe hold target)"
                )
                return False

            # Lock the measured pose as the hold target AND keep the arm enabled so
            # the control loop is *actively streaming* "stay here" the entire time.
            # We intentionally do NOT disable the robot status (that would stop the
            # hold stream and let the arm sag), and we cap the hold speed so the arm
            # cannot snap to the target on re-enable.
            self._set_hold_joint_target_angles(validated)
            gripper = self.get_current_gripper_open_value()
            if gripper is not None:
                self.set_gripper_open_value(gripper)
            self._set_robot_status_enabled(True)

            # Clear latched joint error codes (no power cut -> arm keeps torque).
            if not self.clear_joint_errors():
                self._log.warning(f"JointConfig clear_err was not acknowledged")
            # Step 2: log the fault state immediately after clearing.
            self.log_fault_state("after-clear", arm_name=name)

            # Re-enable the motors. NOTE: we deliberately do NOT send
            # MotionCtrl_1(0x02) here -- in this codebase that call is the
            # emergency-STOP (see emergency_stop()), and it disables the drivers
            # and drops the arm. EnablePiper alone re-arms after a clear_err.
            self._reset_circuit_breaker()
            if not self._enable_robot(timeout_s=timeout_s):
                self._log.error(f"Re-enable failed after clear_err")
                self.log_fault_state("enable-failed", arm_name=name)
                # Do NOT disable -- leave the arm enabled/holding so it does not drop.
                return False

            # Re-arm the active control mode so streamed commands take effect. The
            # loop is locked onto the measured pose, so this holds, not moves.
            move_mode = (
                0x01
                if self.get_control_mode() == PiperController.ControlMode.JOINT_SPACE
                else 0x00
            )
            self._ensure_motionctrl(move_mode=move_mode)

            # Step 3: final fault state after re-enable (should read "no fault").
            self.log_fault_state("after-enable", arm_name=name)
            self._log.info(
                f"Fault cleared and arm re-enabled "
                "(holding current position)"
            )
            return True
        except Exception as e:
            self._log.error("fault recovery error: %s", e)
            return False

    def get_current_end_effector_pose(self) -> np.ndarray | None:
        """Get the current measured end effector pose from the robot, if available.

        Returns:
            4x4 transformation matrix or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                end_pose_msg = self._sdk_call(
                    "GetArmEndPoseMsgs", self.piper.GetArmEndPoseMsgs
                )
                if end_pose_msg and end_pose_msg is not self._sdk_error_sentinel:
                    # Get 6D pose from SDK
                    pose_6d = np.array(
                        [
                            end_pose_msg.end_pose.X_axis / 1000,
                            end_pose_msg.end_pose.Y_axis / 1000,
                            end_pose_msg.end_pose.Z_axis / 1000,
                            end_pose_msg.end_pose.RX_axis / 1000,
                            end_pose_msg.end_pose.RY_axis / 1000,
                            end_pose_msg.end_pose.RZ_axis / 1000,
                        ],
                        dtype=np.float64,
                    )
                    # Convert to 4x4 matrix
                    return self._pose_6d_to_4x4(pose_6d)
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get current end pose: {e}")
        return None

    def get_current_joint_angles(self) -> np.ndarray | None:
        """Get the current measured joint angles from the robot, if available.

        Returns:
            numpy array or None: [j1, j2, j3, j4, j5, j6] or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                arm_joint_msgs = self._sdk_call(
                    "GetArmJointMsgs", self.piper.GetArmJointMsgs
                )
                if arm_joint_msgs is self._sdk_error_sentinel:
                    joint_msg = None
                else:
                    joint_msg = arm_joint_msgs.joint_state if arm_joint_msgs else None
                if joint_msg:
                    return np.array(
                        [
                            joint_msg.joint_1 / 1000,
                            joint_msg.joint_2 / 1000,
                            joint_msg.joint_3 / 1000,
                            joint_msg.joint_4 / 1000,
                            joint_msg.joint_5 / 1000,
                            joint_msg.joint_6 / 1000,
                        ],
                        dtype=np.float64,
                    )
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get current joint angles: {e}")
        return None

    def get_current_joint_feedback(
        self,
    ) -> tuple[np.ndarray | None, FeedbackTiming | None]:
        """Measured degrees and their coherent SDK receive timestamp."""
        if not hasattr(self, "piper") or self.piper is None:
            return None, None
        result = self._sdk_call(
            "GetArmJointMsgs", lambda: read_sdk_feedback(self.piper, "joint")
        )
        if result is self._sdk_error_sentinel:
            return None, None
        raw, timing = result
        joints = np.asarray(raw, dtype=np.float64) / 1000.0
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            return None, None
        return joints, timing

    def get_current_gripper_feedback(
        self,
    ) -> tuple[float | None, FeedbackTiming | None]:
        """Measured normalized opening and its coherent SDK receive timestamp."""
        if not hasattr(self, "piper") or self.piper is None:
            return None, None
        result = self._sdk_call(
            "GetArmGripperMsgs", lambda: read_sdk_feedback(self.piper, "gripper")
        )
        if result is self._sdk_error_sentinel:
            return None, None
        raw, timing = result
        value = (
            raw[0] / 1000.0 - self.GRIPPER_DEGREES_MIN
        ) / self.GRIPPER_DEGREES_RANGE
        if not np.isfinite(value):
            return None, None
        return float(np.clip(value, 0.0, 1.0)), timing

    def get_current_joint_motor_telemetry(
        self,
        *,
        non_blocking: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get measured joint currents and velocities from high-speed feedback.

        Returns:
            Tuple of (currents_a, velocities_rad_s). Each array has shape (6,)
            when feedback is available; otherwise that entry is None.
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                call = (
                    self._sdk_call_nonblocking
                    if non_blocking
                    else self._sdk_call
                )
                high_speed_msgs = call(
                    "GetArmHighSpdInfoMsgs",
                    self.piper.GetArmHighSpdInfoMsgs,
                )
                if (
                    high_speed_msgs is None
                    or high_speed_msgs is self._sdk_error_sentinel
                ):
                    return None, None
                currents_a = []
                velocities_rad_s = []
                for joint_idx in range(1, 7):
                    motor = getattr(high_speed_msgs, f"motor_{joint_idx}")
                    currents_a.append(float(getattr(motor, "current")) * 0.001)
                    velocities_rad_s.append(
                        float(getattr(motor, "motor_speed")) * 0.001
                    )
                return (
                    np.asarray(currents_a, dtype=np.float64),
                    np.asarray(velocities_rad_s, dtype=np.float64),
                )
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get current joint motor telemetry: {e}")
        return None, None

    # Per-motor fault/status flag fields exposed by the low-speed feedback.
    MOTOR_FAULT_FLAG_FIELDS = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "driver_error_status",
        "driver_enable_status",
        "stall_status",
    )

    def get_motor_fault_flags(
        self, *, non_blocking: bool = False
    ) -> dict[str, np.ndarray] | None:
        """Get per-joint fault/status flags from low-speed driver feedback.

        Returns a dict mapping each flag name (see ``MOTOR_FAULT_FLAG_FIELDS``)
        to a length-6 array of 0/1 values, one per joint, or None when the
        feedback is unavailable.
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                call = (
                    self._sdk_call_nonblocking if non_blocking else self._sdk_call
                )
                low_speed_msgs = call(
                    "GetArmLowSpdInfoMsgs", self.piper.GetArmLowSpdInfoMsgs
                )
                if (
                    low_speed_msgs is None
                    or low_speed_msgs is self._sdk_error_sentinel
                ):
                    return None
                flags: dict[str, np.ndarray] = {}
                for field in self.MOTOR_FAULT_FLAG_FIELDS:
                    values = [
                        int(
                            bool(
                                getattr(
                                    getattr(
                                        low_speed_msgs, f"motor_{joint_idx}"
                                    ).foc_status,
                                    field,
                                )
                            )
                        )
                        for joint_idx in range(1, 7)
                    ]
                    flags[field] = np.asarray(values, dtype=np.float64)
                return flags
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get motor fault flags: {e}")
        return None

    @staticmethod
    def _sdk_enum_value(value: Any) -> float:
        raw = getattr(value, "value", value)
        try:
            return float(raw)
        except Exception:
            return float("nan")

    def get_piper_feedback_status(
        self, *, non_blocking: bool = False
    ) -> dict[str, float | np.ndarray] | None:
        """Get low-rate PiPER status/fault fields for observability.

        This is intentionally read-only and intended for slow diagnostics
        polling, not the control loop. It uses the same non-blocking SDK path as
        the existing optional telemetry streams when requested.
        """
        if not (hasattr(self, "piper") and self.piper is not None):
            return None
        try:
            call = self._sdk_call_nonblocking if non_blocking else self._sdk_call
            low_speed_msgs = call(
                "GetArmLowSpdInfoMsgs", self.piper.GetArmLowSpdInfoMsgs
            )
            gripper_msg = call("GetArmGripperMsgs", self.piper.GetArmGripperMsgs)
            arm_status_msg = call("GetArmStatus", self.piper.GetArmStatus)
            if (
                low_speed_msgs is None
                or low_speed_msgs is self._sdk_error_sentinel
            ):
                return None

            status: dict[str, float | np.ndarray] = {}
            status["motor_foc_status_code"] = np.asarray(
                [
                    float(
                        getattr(
                            getattr(low_speed_msgs, f"motor_{joint_idx}"),
                            "foc_status_code",
                        )
                    )
                    for joint_idx in range(1, 7)
                ],
                dtype=np.float64,
            )
            status["motor_temperature"] = np.asarray(
                [
                    float(getattr(low_speed_msgs, f"motor_{joint_idx}").motor_temp)
                    for joint_idx in range(1, 7)
                ],
                dtype=np.float64,
            )
            status["driver_temperature"] = np.asarray(
                [
                    float(getattr(low_speed_msgs, f"motor_{joint_idx}").foc_temp)
                    for joint_idx in range(1, 7)
                ],
                dtype=np.float64,
            )
            for field in self.MOTOR_FAULT_FLAG_FIELDS:
                status[f"motor_fault_{field}"] = np.asarray(
                    [
                        float(
                            int(
                                bool(
                                    getattr(
                                        getattr(
                                            low_speed_msgs, f"motor_{joint_idx}"
                                        ).foc_status,
                                        field,
                                    )
                                )
                            )
                        )
                        for joint_idx in range(1, 7)
                    ],
                    dtype=np.float64,
                )

            if gripper_msg is not None and gripper_msg is not self._sdk_error_sentinel:
                gripper = gripper_msg.gripper_state
                status["gripper_status_code"] = float(
                    getattr(gripper, "status_code", float("nan"))
                )
                for field in (
                    "voltage_too_low",
                    "motor_overheating",
                    "driver_overcurrent",
                    "driver_overheating",
                    "sensor_status",
                    "driver_error_status",
                    "driver_enable_status",
                    "homing_status",
                ):
                    status[f"gripper_fault_{field}"] = float(
                        int(bool(getattr(gripper.foc_status, field)))
                    )

            if arm_status_msg is None or arm_status_msg is self._sdk_error_sentinel:
                return status

            arm_status = arm_status_msg.arm_status
            for field in (
                "ctrl_mode",
                "arm_status",
                "mode_feed",
                "teach_status",
                "motion_status",
            ):
                status[f"arm_{field}"] = self._sdk_enum_value(
                    getattr(arm_status, field)
                )
            status["arm_trajectory_num"] = float(arm_status.trajectory_num)
            status["arm_err_code"] = float(arm_status.err_code)

            angle_limit_errors = np.zeros(6, dtype=np.float64)
            communication_errors = np.zeros(6, dtype=np.float64)
            for joint_idx in range(1, 7):
                angle_limit_errors[joint_idx - 1] = float(
                    int(
                        bool(
                            getattr(
                                arm_status.err_status,
                                f"joint_{joint_idx}_angle_limit",
                            )
                        )
                    )
                )
                communication_errors[joint_idx - 1] = float(
                    int(
                        bool(
                            getattr(
                                arm_status.err_status,
                                f"communication_status_joint_{joint_idx}",
                            )
                        )
                    )
                )
            status["arm_joint_angle_limit_error"] = angle_limit_errors
            status["arm_joint_communication_error"] = communication_errors
            return status
        except Exception as e:
            if self.debug_mode:
                self._log.warning(f"Failed to get PiPER feedback status: {e}")
        return None

    def get_motor_temperatures(
        self, *, non_blocking: bool = False
    ) -> dict[str, np.ndarray] | None:
        """Get per-joint motor and driver temperatures from low-speed feedback.

        Returns a dict with ``motor_temperature`` and ``driver_temperature``,
        each a length-6 array of degrees Celsius (one per joint), or None when
        the feedback is unavailable. ``driver_temperature`` is the FOC/driver
        temperature (SDK ``foc_temp``).
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                call = (
                    self._sdk_call_nonblocking if non_blocking else self._sdk_call
                )
                low_speed_msgs = call(
                    "GetArmLowSpdInfoMsgs", self.piper.GetArmLowSpdInfoMsgs
                )
                if (
                    low_speed_msgs is None
                    or low_speed_msgs is self._sdk_error_sentinel
                ):
                    return None
                motor_temp = [
                    float(getattr(low_speed_msgs, f"motor_{j}").motor_temp)
                    for j in range(1, 7)
                ]
                driver_temp = [
                    float(getattr(low_speed_msgs, f"motor_{j}").foc_temp)
                    for j in range(1, 7)
                ]
                return {
                    "motor_temperature": np.asarray(motor_temp, dtype=np.float64),
                    "driver_temperature": np.asarray(driver_temp, dtype=np.float64),
                }
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get motor temperatures: {e}")
        return None

    def get_current_gripper_open_value(self) -> float | None:
        """Get the current measured gripper open value from the robot, if available.

        Returns:
            float or None: Current gripper open value normalized (0.0 to 1.0) or None if not available
        """
        if hasattr(self, "piper") and self.piper is not None:
            try:
                gripper_msg = self._sdk_call(
                    "GetArmGripperMsgs", self.piper.GetArmGripperMsgs
                )
                if gripper_msg and gripper_msg is not self._sdk_error_sentinel:
                    gripper_degrees = gripper_msg.gripper_state.grippers_angle / 1000
                    # Convert from degrees to normalized (0.0 to 1.0)
                    normalized = (
                        gripper_degrees - self.GRIPPER_DEGREES_MIN
                    ) / self.GRIPPER_DEGREES_RANGE
                    return float(np.clip(normalized, 0.0, 1.0))
            except Exception as e:
                if self.debug_mode:
                    self._log.warning(f"Failed to get current gripper open value: {e}")
        return None

    def get_robot_status(self) -> dict[str, Any]:
        """Get comprehensive robot status information.

        Returns:
            dict: Dictionary containing robot status information
        """
        try:
            status = {
                "enabled": self.is_robot_enabled(),
                "control_mode": self.get_control_mode(),
                "target_pose": self.get_target_pose(),
                "target_joint_angles": self.get_target_joint_angles(),
                "gripper_open_value": self.get_gripper_open_value(),
                "current_end_pose": self.get_current_end_effector_pose(),
                "current_joint_angles": self.get_current_joint_angles(),
                "current_gripper_open_value": self.get_current_gripper_open_value(),
            }
            return status

        except Exception as e:
            self._log.warning(f"Failed to get robot status: {e}")
            return {
                "enabled": None,
                "control_mode": None,
                "target_pose": None,
                "target_joint_angles": None,
                "gripper_open_value": None,
                "current_end_pose": None,
                "current_joint_angles": None,
                "current_gripper_open_value": None,
            }

    def is_robot_homed(self, tolerance_degrees: float = 2.0) -> bool:
        """Check if robot is at home position within tolerance.

        Args:
            tolerance_degrees: Tolerance in degrees

        Returns:
            True if robot is at home position within tolerance, False otherwise
        """
        current_joints = self.get_current_joint_angles()
        if current_joints is None:
            return False
        home_joints = self.HOME_JOINT_ANGLES
        differences = np.abs(current_joints - home_joints)
        return np.all(differences < tolerance_degrees)

    def update_robot_rate(self, rate: float) -> None:
        """Update the robot rate.

        Args:
            rate: The new robot rate in Hz
        """
        self.robot_rate = rate

    def set_motion_speed_percent(self, speed_percent: int | float) -> None:
        """Set SDK MotionCtrl_2 speed percentage for subsequent MOVE commands.

        Args:
            speed_percent: Speed percentage in the SDK's 1..100 range.
        """
        speed = int(round(float(speed_percent)))
        speed = max(1, min(100, speed))
        with self.position_lock:
            self._motionctrl_speed_percent = speed

    def get_motion_speed_percent(self) -> int:
        """Get the current SDK MotionCtrl_2 speed percentage."""
        with self.position_lock:
            return int(self._motionctrl_speed_percent)

    def get_home_speed_percent(self) -> int:
        """Get the speed cap (1..100) applied to operator-triggered homing."""
        return int(self._home_speed_percent)

    def set_home_speed_percent(self, speed_percent: int | float) -> None:
        """Set the speed cap applied to operator-triggered homing.

        The construction-time default comes from ``PIPER_HOME_SPEED_PERCENT``.
        This lets the runtime apply the per-deployment value from
        ``config/configs.py`` after the controller exists, instead of the
        caller having to set an environment variable before construction.

        Args:
            speed_percent: Speed percentage, clamped into the SDK's 1..100 range.
        """
        speed = int(round(float(speed_percent)))
        self._home_speed_percent = int(np.clip(speed, 1, 100))

    def get_joint_velocity_limits_deg_s(
        self, *, refresh: bool = False, timeout_s: float = 0.5
    ) -> np.ndarray:
        """Return SDK-reported max joint speeds in degrees/second.

        The SDK feedback is reported as ``max_joint_spd`` in 0.001 rad/s.
        This helper only exposes the hardware speed limits; dashboard-level
        safety factors and MotionCtrl speed percentages are applied by callers.
        """
        if (not refresh) and self._joint_velocity_limits_deg_s is not None:
            return self._joint_velocity_limits_deg_s.copy()
        if not hasattr(self, "piper") or self.piper is None:
            raise RuntimeError("SDK object is not initialized")
        if not self._sdk_connected():
            raise RuntimeError("SDK is not connected")

        search_all = getattr(self.piper, "SearchAllMotorMaxAngleSpd", None)
        get_all = getattr(self.piper, "GetAllMotorAngleLimitMaxSpd", None)
        if search_all is None or get_all is None:
            raise RuntimeError("SDK does not expose motor max speed feedback helpers")

        if not self._sdk_ok("SearchAllMotorMaxAngleSpd", search_all):
            raise RuntimeError("SDK failed to query motor max angle/speed limits")

        deadline = time.monotonic() + max(0.01, float(timeout_s))
        last_seen: list[int] = []
        while time.monotonic() < deadline:
            feedback = self._sdk_call(
                "GetAllMotorAngleLimitMaxSpd",
                get_all,
            )
            if feedback is self._sdk_error_sentinel or feedback is None:
                time.sleep(0.02)
                continue

            try:
                motors = feedback.all_motor_angle_limit_max_spd.motor
                speeds_raw = [
                    int(getattr(motors[idx], "max_joint_spd"))
                    for idx in range(1, 7)
                ]
                motor_nums = [
                    int(getattr(motors[idx], "motor_num"))
                    for idx in range(1, 7)
                ]
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected SDK motor speed feedback shape: {exc}"
                ) from exc

            last_seen = speeds_raw
            if all(num == idx for idx, num in enumerate(motor_nums, start=1)) and all(
                speed > 0 for speed in speeds_raw
            ):
                speeds_raw_arr = np.asarray(speeds_raw, dtype=np.int64)
                speeds_deg_s = speeds_raw_arr.astype(np.float64) * 0.001
                speeds_deg_s = np.degrees(speeds_deg_s)
                with self.position_lock:
                    self._joint_velocity_limits_raw = speeds_raw_arr.copy()
                    self._joint_velocity_limits_deg_s = speeds_deg_s.copy()
                return speeds_deg_s

            time.sleep(0.02)

        if self._joint_velocity_limits_deg_s is not None:
            return self._joint_velocity_limits_deg_s.copy()
        raise RuntimeError(
            "Timed out waiting for SDK motor speed limits "
            f"(last raw max_joint_spd={last_seen})"
        )

    def get_joint_velocity_limits_raw(
        self, *, refresh: bool = False, timeout_s: float = 0.5
    ) -> np.ndarray:
        """Return SDK-reported raw max joint speeds in 0.001 rad/s units."""
        if (not refresh) and self._joint_velocity_limits_raw is not None:
            return self._joint_velocity_limits_raw.copy()
        self.get_joint_velocity_limits_deg_s(refresh=refresh, timeout_s=timeout_s)
        if self._joint_velocity_limits_raw is not None:
            return self._joint_velocity_limits_raw.copy()
        if self._joint_velocity_limits_deg_s is not None:
            raw = np.rint(np.radians(self._joint_velocity_limits_deg_s) * 1000.0)
            return raw.astype(np.int64)
        raise RuntimeError("SDK motor speed limits are unavailable")

    def set_joint_velocity_limit_raw(
        self,
        motor_num: int,
        max_joint_spd_raw: int,
        *,
        timeout_s: float = 0.5,
    ) -> np.ndarray:
        """Set one SDK motor max speed, then return refreshed raw speed limits."""
        motor = int(motor_num)
        speed_raw = int(max_joint_spd_raw)
        if motor < 1 or motor > 6:
            raise ValueError("motor_num must be in 1..6")
        if speed_raw < 1 or speed_raw > MAX_CONFIGURABLE_JOINT_SPEED_RAW:
            raise ValueError(
                "max_joint_spd_raw must be in "
                f"1..{MAX_CONFIGURABLE_JOINT_SPEED_RAW}"
            )
        if not hasattr(self, "piper") or self.piper is None:
            raise RuntimeError("SDK object is not initialized")
        if not self._sdk_connected():
            raise RuntimeError("SDK is not connected")

        set_speed = getattr(self.piper, "MotorMaxSpdSet", None)
        if set_speed is None:
            set_with_limits = getattr(self.piper, "MotorAngleLimitMaxSpdSet", None)
            if set_with_limits is None:
                raise RuntimeError("SDK does not expose motor max speed setter")
            ok = self._sdk_ok(
                "MotorAngleLimitMaxSpdSet",
                lambda: set_with_limits(motor, 0x7FFF, 0x7FFF, speed_raw),
            )
        else:
            ok = self._sdk_ok("MotorMaxSpdSet", lambda: set_speed(motor, speed_raw))
        if not ok:
            raise RuntimeError("SDK failed to set motor max speed")

        with self.position_lock:
            self._joint_velocity_limits_raw = None
            self._joint_velocity_limits_deg_s = None
        time.sleep(0.05)
        return self.get_joint_velocity_limits_raw(refresh=True, timeout_s=timeout_s)

    def hold_current_position(self, *, allow_resume: bool = True) -> bool:
        """Freeze the robot at its current measured state (without disabling motors).

        This is useful for "pause/hold" behavior: stop teleop updates but keep the arm
        actively holding its current pose.

        Args:
            allow_resume: When True, a disabled controller may resume before
                installing the hold target. Shutdown/cleanup paths should pass
                False so stopping a service never implicitly enables hardware.

        Returns:
            True if targets were updated successfully, False otherwise.
        """
        if self._stop_hold_latched.is_set():
            with self._stop_hold_lock:
                if self._stop_hold_latched.is_set():
                    return self._send_stop_hold()
        if not self.is_robot_enabled():
            if not allow_resume:
                self._log.info(
                    f"Hold current position skipped: "
                    "robot disabled and resume not allowed"
                )
                return False
            if not self.resume_robot():
                return False

        # Prefer holding in the currently active control mode.
        current_mode = self.get_control_mode()

        if current_mode == PiperController.ControlMode.JOINT_SPACE:
            raw = self._sdk_call("GetArmJointMsgs", self.piper.GetArmJointMsgs)
            joints, error = self._joint_feedback_to_degrees(raw)
            age = self._feedback_timestamp_age_s(raw)
            # Feedback lands on the firmware's 0.001 deg grid and the SDK's own bounds
            # are not on that grid (j5 = 1.22 rad = 69.900851 deg), so an arm parked at a
            # stop reports 69.901: 0.000149 deg "out of range". Rejecting that latched
            # STOP & HOLD and made Enable fail with "could not hold position" on both arms
            # (2026-09-20). Guard by the same epsilon _read_validated_joint_feedback uses,
            # so resume and hold agree, then hold at a legal in-range target. Feedback
            # further out than the epsilon is still a real fault and still latches.
            limit_epsilon = self._resume_hardware_limit_epsilon_deg
            if (
                error or joints is None
                or (age is not None and age > self._startup_feedback_max_age_s)
                or np.any(joints < self.JOINT_LIMITS[:, 0] - limit_epsilon)
                or np.any(joints > self.JOINT_LIMITS[:, 1] + limit_epsilon)
            ):
                self.stop_and_hold()
                return False
            self._set_hold_joint_target_angles(
                np.clip(joints, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])
            )
        elif current_mode == PiperController.ControlMode.END_EFFECTOR:
            pose = self.get_current_end_effector_pose()
            if pose is None or pose.shape != (4, 4):
                return False
            self._set_hold_target_pose(pose)
        else:
            return False

        gripper = self.get_current_gripper_open_value()
        if gripper is not None:
            self.set_gripper_open_value(gripper)

        return True
