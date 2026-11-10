#!/usr/bin/env python3
"""Thread-safe teleoperation state management.

This module provides shared state classes for teleoperation systems that need
to coordinate between multiple threads (data collection, IK solving, visualization).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, overload

import numpy as np

from .camera_timing import CameraFrameTiming, offer_timed_frame
from .configs import TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND
from .feedback_timing import FeedbackTiming
from .one_euro_filter import OneEuroFilterTransform

def _set_robot_activity_state(*_a: Any, **_k: Any) -> None:
    return None


# Numeric codes for the activity-state gauge (matches exporter doc string).
_ACTIVITY_STATE_CODES = {
    "DISABLED": 0.0,
    "ENABLED": 1.0,
    "PAUSED": 2.0,
    "HOMING": 3.0,
    "POLICY_CONTROLLED": 4.0,
    "PRESET_CONTROLLED": 5.0,
}

Hand = Literal["left", "right"]
ClampMode = Literal["off", "soft_damp"]
TeleopGripperMode = Literal["hold", "toggle"]


@dataclass(frozen=True)
class BrainControlSnapshot:
    """Immutable snapshot of brain-control state (thread-safe copy)."""

    armed: bool
    diameter: float
    slider_norm: np.ndarray  # shape (6,)
    clamp_mode: ClampMode
    active_group: str
    selected_alpha: float
    unselected_alpha: float
    baseline_deg: np.ndarray | None  # shape (6,)
    last_commanded_deg: np.ndarray | None  # shape (6,)


@dataclass(frozen=True)
class TeleopGripperSettings:
    """Shared Quest grip-button behavior for teleop gripper control."""

    open_limit: float
    mode: TeleopGripperMode


def _normalize_hand(hand: str | None) -> Hand:
    if hand is None:
        return "right"
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be 'left' or 'right' (got {hand!r})")
    return hand  # type: ignore[return-value]


class RobotActivityState(Enum):
    """Robot activity state enumeration."""

    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    HOMING = "HOMING"
    DISABLED = "DISABLED"
    POLICY_CONTROLLED = "POLICY_CONTROLLED"
    PRESET_CONTROLLED = "PRESET_CONTROLLED"


# Who is commanding an arm, as the per-hand "activity_mode" telemetry signal
# (BigQuery `values[0]`), emitted every joint-state tick next to arm_driven so a
# dataset frame can be labelled operator / preset / policy after the fact.
# Keys are shared with the runtime's stack-wide mode; "policy" is the code the
# episode exporter excludes from training data (DAgger: never learn from the
# policy's own actions).
ACTIVITY_MODE_CODES: dict[str, int] = {
    "idle": 0,
    "teleop": 1,
    "preset": 2,
    "trajectory": 3,
    "homing": 4,
    "paused": 5,
    "disabled": 6,
    "policy": 7,
}
ACTIVITY_MODE_NAMES: dict[int, str] = {v: k for k, v in ACTIVITY_MODE_CODES.items()}


def activity_mode_code(
    state: "RobotActivityState", teleop_active: bool, driven: bool = False
) -> int:
    """Per-hand control source for one control tick.

    ``driven`` is the joint-state loop's "operator targets are streaming" flag
    (teleop, brain nudges or the J6 joystick); it counts as teleop.
    """
    if state == RobotActivityState.POLICY_CONTROLLED:
        return ACTIVITY_MODE_CODES["policy"]
    if state == RobotActivityState.PRESET_CONTROLLED:
        return ACTIVITY_MODE_CODES["preset"]
    if state == RobotActivityState.HOMING:
        return ACTIVITY_MODE_CODES["homing"]
    if state == RobotActivityState.PAUSED:
        return ACTIVITY_MODE_CODES["paused"]
    if state == RobotActivityState.DISABLED:
        return ACTIVITY_MODE_CODES["disabled"]
    if teleop_active or driven:
        return ACTIVITY_MODE_CODES["teleop"]
    return ACTIVITY_MODE_CODES["idle"]


class ControllerState:
    """Controller input state - Quest Reader writes, IK/Joint reads."""

    def __init__(self) -> None:
        """Initialize ControllerState with default values."""
        self._lock = threading.Lock()

        # 1€ Filter parameters
        self.min_cutoff: float = 1.0  # Minimum cutoff frequency
        self.beta: float = 0.0  # Speed coefficient
        self.d_cutoff: float = 1.0  # Derivative cutoff frequency

        # Controller data
        self.transform_raw: np.ndarray | None = None
        self.transform: np.ndarray | None = None  # Smoothed transform
        self.grip_value: float = 0.0
        self.trigger_value: float = 0.0
        self.joystick_y: float = 0.0
        self.joystick_y_suppress_until: float = 0.0
        self._filter: OneEuroFilterTransform | None = None


class TeleopState:
    """Teleop activation state - manages teleop start/stop."""

    def __init__(self, hand: Hand) -> None:
        """Initialize TeleopState with default values."""
        self._lock = threading.Lock()

        self.active: bool = False
        self.generation: int = 0
        self.ready_generation: int = -1
        self.baseline_generation: int = -1
        self.command_baseline: np.ndarray | None = None
        self.controller_initial_transform: np.ndarray | None = None
        self.robot_initial_transform: np.ndarray | None = None

        # Per-hand fixed frame correction [roll, pitch, yaw] in degrees.
        raw_offset = TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND.get(hand, [0.0, 0.0, 0.0])
        arr = np.asarray(raw_offset, dtype=np.float64).reshape(-1)
        self.delta_rpy_offset_deg: np.ndarray = (
            arr.copy() if arr.size == 3 else np.zeros(3, dtype=np.float64)
        )


class RobotState:
    """Current robot state - joint angles, end effector pose, activity state."""

    def __init__(self) -> None:
        """Initialize RobotState with default values."""
        self._lock = threading.Lock()

        self.joint_angles: np.ndarray | None = None
        self.joint_currents_a: np.ndarray | None = None
        self.joint_velocities_rad_s: np.ndarray | None = None
        self.end_effector_pose: np.ndarray | None = None
        self.current_gripper_open_value: float | None = None
        self.target_gripper_open_value: float | None = None
        self.activity_state: RobotActivityState = RobotActivityState.DISABLED


class IKState:
    """IK solution state - target joint angles, pose, metrics."""

    def __init__(self) -> None:
        """Initialize IKState with default values."""
        self._lock = threading.Lock()

        self.target_joint_angles: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.solve_time_ms: float = 0.0
        self.success: bool = True


class BrainControlState:
    """Brain control UI state - joint offset nudges and clamp selection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.armed: bool = False
        self.diameter: float = 0.0
        self.slider_norm: np.ndarray = np.zeros(6, dtype=np.float64)
        self.clamp_mode: ClampMode = "off"
        self.active_group: str = "all"
        self.selected_alpha: float = 0.2
        self.unselected_alpha: float = 0.01
        self.baseline_deg: np.ndarray | None = None
        self.last_commanded_deg: np.ndarray | None = None


@dataclass(frozen=True)
class CameraFrameSnapshot:
    """Immutable camera frame snapshot returned by DataManager getters."""

    rgb_image: np.ndarray | None
    depth_image_mm: np.ndarray | None
    rgb_timestamp_ns: int | None
    depth_timestamp_ns: int | None
    rgb_timing: CameraFrameTiming | None = None
    depth_timing: CameraFrameTiming | None = None


class CameraFrameState:
    """Mutable camera frame storage for a single camera."""

    def __init__(self) -> None:
        self.rgb_image: np.ndarray | None = None
        self.depth_image_mm: np.ndarray | None = None
        self.rgb_timestamp_ns: int | None = None
        self.depth_timestamp_ns: int | None = None
        self.rgb_timing: CameraFrameTiming | None = None
        self.depth_timing: CameraFrameTiming | None = None


class CameraState:
    """Camera state storage for multiple cameras."""

    def __init__(self) -> None:
        """Initialize CameraState with default values."""
        self._lock = threading.Lock()
        self.frames_by_name: dict[str, CameraFrameState] = {}


class DataManager:
    """Main state container coordinating all state groups.

    This class manages shared data between threads:
    - Data collection thread: updates controller data
    - IK solver thread: reads controller data, updates joint solutions
    - Main thread: reads everything for visualization

    Uses separate locks for each state group to reduce contention.
    """

    def __init__(self) -> None:
        """Initialize DataManager with default values."""
        # State groups with individual locks (per hand)
        self._controller_state_by_hand: dict[Hand, ControllerState] = {
            "left": ControllerState(),
            "right": ControllerState(),
        }
        self._teleop_state_by_hand: dict[Hand, TeleopState] = {
            "left": TeleopState("left"),
            "right": TeleopState("right"),
        }
        self._robot_state_by_hand: dict[Hand, RobotState] = {
            "left": RobotState(),
            "right": RobotState(),
        }
        self._ik_state_by_hand: dict[Hand, IKState] = {
            "left": IKState(),
            "right": IKState(),
        }
        self._brain_control_state_by_hand: dict[Hand, BrainControlState] = {
            "left": BrainControlState(),
            "right": BrainControlState(),
        }
        self._camera_state = CameraState()
        self._teleop_gripper_lock = threading.Lock()
        self._teleop_gripper_open_limit_by_hand: dict[Hand, float] = {
            "left": 1.0,
            "right": 1.0,
        }
        self._teleop_gripper_mode_by_hand: dict[Hand, TeleopGripperMode] = {
            "left": "hold",
            "right": "hold",
        }

        self._operator_connection_provider = None

        # System state
        self._shutdown_event = threading.Event()

        # Operator take-over requests filed by the quest reader when the teleop
        # trigger is squeezed while a hand is POLICY_CONTROLLED; the policy
        # runner's 100 Hz loop consumes them (see common.policy.runner).
        self._policy_takeover_lock = threading.Lock()
        self._policy_takeover_hands: set[Hand] = set()

        # Callback for state changes (RGB, target joints, current joints)
        # the callable takes arguments: (stream_name: str, data: Any, timestamp: float)
        self._on_change_callback: Callable[[str, Any, float], None] | None = None
        self._joint_telemetry_streams_enabled = False
        self._joint_telemetry_min_period_s = 0.05
        self._piper_feedback_streams_enabled = False
        self._piper_feedback_min_period_s = 1.0
        # Whether an episode is being recorded. Used to gate CPU-heavy depth
        # processing so it runs only while recording. Defaults True so paths
        # that never set it (LeRobot/dashboard) keep depth always-on.
        self._recording_active = True
        # Optional frame sink (separate-process GCS recorder); see
        # set_camera_frame_sink.
        self._camera_frame_sink: Any = None
        # When True (and the sink can accept raw depth), the camera thread ships
        # the *raw* aligned depth to the recorder process and skips the CPU-heavy
        # scale/clip/rotate in-process -- that work moves to the recorder's own
        # OS process so it never steals GIL time from the control loops. Only
        # safe when nothing in this process consumes processed depth (no preview,
        # no LeRobot/dashboard); see set_depth_processing_offload.
        self._depth_offload_requested = False
        self._depth_raw_capable = False

    def set_recording_active(self, active: bool) -> None:
        """Mark whether an episode is currently recording (lock-free bool)."""
        self._recording_active = bool(active)

    def set_camera_frame_sink(self, sink: Any) -> None:
        """Register a frame sink (e.g. the separate-process GCS recorder).

        The sink must expose ``offer_rgb(cam, image)`` and
        ``offer_depth(cam, depth)``. They are called from the camera thread and
        must return immediately (the recorder copies into shared memory only
        while recording), so they never block capture or control.

        A sink may additionally expose ``offer_depth_raw(cam, raw_u16, scale_m)``
        to accept unprocessed depth (see set_depth_processing_offload).
        """
        self._camera_frame_sink = sink
        self._depth_raw_capable = hasattr(sink, "offer_depth_raw")

    def set_depth_processing_offload(self, enabled: bool) -> None:
        """Request that depth scale/clip/rotate be done in the recorder process.

        Only takes effect when the registered sink can accept raw depth. Callers
        must ensure no in-process consumer needs processed depth while enabled.
        """
        self._depth_offload_requested = bool(enabled)

    def depth_offload_active(self) -> bool:
        """Whether the camera thread should ship raw depth instead of processing."""
        return self._depth_offload_requested and self._depth_raw_capable

    def offer_camera_depth_raw(
        self,
        camera_name: str,
        raw_depth_u16: np.ndarray,
        depth_scale_m: float,
        timestamp_ns: int | None = None,
        *,
        timing: CameraFrameTiming | None = None,
    ) -> None:
        """Forward raw (pre-scale, pre-rotate) depth to the recorder sink.

        Cheap: the sink copies the buffer into shared memory and returns; all
        scaling/rotation/encoding happens in the recorder's own process.
        """
        sink = self._camera_frame_sink
        if sink is not None:
            offer_timed_frame(
                sink,
                "offer_depth_raw",
                camera_name,
                raw_depth_u16,
                depth_scale_m,
                timing.timestamp_ns if timing is not None else timestamp_ns,
                timing=timing,
            )

    def is_recording_active(self) -> bool:
        """Whether an episode is recording (gates depth processing)."""
        return self._recording_active

    def set_on_change_callback(
        self, on_change_callback: Callable[[str, Any, float], None]
    ) -> None:
        """Set on change callback (thread-safe)."""
        self._on_change_callback = on_change_callback

    def set_joint_telemetry_streams_enabled(
        self, enabled: bool, *, max_hz: float | None = None
    ) -> None:
        """Enable optional joint current/velocity callback streams."""
        self._joint_telemetry_streams_enabled = bool(enabled)
        if max_hz is not None and max_hz > 0.0:
            self._joint_telemetry_min_period_s = 1.0 / float(max_hz)

    def set_piper_feedback_streams_enabled(
        self, enabled: bool, *, max_hz: float | None = None
    ) -> None:
        """Enable low-rate PiPER fault/status feedback callback streams."""
        self._piper_feedback_streams_enabled = bool(enabled)
        if max_hz is not None and max_hz > 0.0:
            self._piper_feedback_min_period_s = 1.0 / float(max_hz)

    def log_telemetry_signal(
        self,
        name: str,
        values: Any,
        hand: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Emit an arbitrary diagnostic telemetry signal via the on-change callback.

        Unlike the ``set_*`` methods, this holds no state -- it is a passthrough
        for low-rate diagnostic streams (e.g. motor fault flags) that downstream
        sinks (such as the BigQuery telemetry sink) capture. Safe no-op when no
        on-change callback is registered.
        """
        if self._on_change_callback:
            ts = timestamp if timestamp is not None else time.time()
            full_name = f"{name}/{hand}" if hand else name
            self._on_change_callback(full_name, values, ts)

    def joint_telemetry_streams_enabled(self) -> bool:
        """Return whether optional joint current/velocity streams are enabled."""
        return self._joint_telemetry_streams_enabled

    def joint_telemetry_min_period_s(self) -> float:
        """Return minimum seconds between optional current/velocity SDK polls."""
        return self._joint_telemetry_min_period_s

    def piper_feedback_streams_enabled(self) -> bool:
        """Return whether low-rate PiPER fault/status streams are enabled."""
        return self._piper_feedback_streams_enabled

    def piper_feedback_min_period_s(self) -> float:
        """Return minimum seconds between low-rate PiPER feedback SDK polls."""
        return self._piper_feedback_min_period_s

    # ============================================================================
    # Camera State Methods
    # ============================================================================

    def _get_or_create_camera_frame_locked(self, camera_name: str) -> CameraFrameState:
        frame = self._camera_state.frames_by_name.get(camera_name)
        if frame is None:
            frame = CameraFrameState()
            self._camera_state.frames_by_name[camera_name] = frame
        return frame

    def _default_camera_name_locked(self) -> str | None:
        if "rgb" in self._camera_state.frames_by_name:
            return "rgb"
        if not self._camera_state.frames_by_name:
            return None
        return sorted(self._camera_state.frames_by_name.keys())[0]

    def get_camera_names(self) -> list[str]:
        """Get currently known camera names."""
        with self._camera_state._lock:
            return sorted(self._camera_state.frames_by_name.keys())

    def get_camera_frame(
        self, camera_name: str, *, copy: bool = True
    ) -> CameraFrameSnapshot:
        """Get the latest snapshot for a specific camera.

        The (expensive) array copy is done OUTSIDE the lock: writers rebind
        ``frame.rgb_image`` to a new array rather than mutating in place, so a
        reference grabbed under the lock stays valid to copy afterwards. This
        keeps the shared camera-buffer lock held for only microseconds even with
        many readers (Agora publisher + GCS recorder), so the camera thread and
        the real-time control loops are not serialized behind big memcopies.

        ``copy=False`` skips the copy entirely and hands back the stored arrays
        by reference. That same rebind-never-mutate contract makes this safe for
        a **read-only** consumer: the array a reader holds is never written to,
        it is only replaced by the next frame. Every RGB frame is ~1 MB, so at
        30 fps x N cameras each copying reader costs ~30N MB/s of GIL-held
        memcpy -- video sinks that only encode/transmit the frame should pass
        ``copy=False``. Pass the default ``copy=True`` if you intend to mutate
        the array in place or hold it beyond the next frame.
        """
        with self._camera_state._lock:
            frame = self._camera_state.frames_by_name.get(camera_name)
            if frame is None:
                return CameraFrameSnapshot(None, None, None, None)
            rgb_ref = frame.rgb_image
            depth_ref = frame.depth_image_mm
            rgb_ts = frame.rgb_timestamp_ns
            depth_ts = frame.depth_timestamp_ns
            rgb_timing = frame.rgb_timing
            depth_timing = frame.depth_timing
        if not copy:
            return CameraFrameSnapshot(
                rgb_image=rgb_ref,
                depth_image_mm=depth_ref,
                rgb_timestamp_ns=rgb_ts,
                depth_timestamp_ns=depth_ts,
                rgb_timing=rgb_timing,
                depth_timing=depth_timing,
            )
        return CameraFrameSnapshot(
            rgb_image=rgb_ref.copy() if rgb_ref is not None else None,
            depth_image_mm=depth_ref.copy() if depth_ref is not None else None,
            rgb_timestamp_ns=rgb_ts,
            depth_timestamp_ns=depth_ts,
            rgb_timing=rgb_timing,
            depth_timing=depth_timing,
        )

    def get_all_camera_frames(self) -> dict[str, CameraFrameSnapshot]:
        """Get snapshots for all known cameras (copies done outside the lock)."""
        with self._camera_state._lock:
            refs = [
                (
                    name,
                    frame.rgb_image,
                    frame.depth_image_mm,
                    frame.rgb_timestamp_ns,
                    frame.depth_timestamp_ns,
                    frame.rgb_timing,
                    frame.depth_timing,
                )
                for name, frame in self._camera_state.frames_by_name.items()
            ]
        out: dict[str, CameraFrameSnapshot] = {}
        for (
            name,
            rgb_ref,
            depth_ref,
            rgb_ts,
            depth_ts,
            rgb_timing,
            depth_timing,
        ) in refs:
            out[name] = CameraFrameSnapshot(
                rgb_image=rgb_ref.copy() if rgb_ref is not None else None,
                depth_image_mm=depth_ref.copy() if depth_ref is not None else None,
                rgb_timestamp_ns=rgb_ts,
                depth_timestamp_ns=depth_ts,
                rgb_timing=rgb_timing,
                depth_timing=depth_timing,
            )
        return out

    def get_rgb_image(self, camera_name: str | None = None) -> np.ndarray | None:
        """Get latest RGB image (thread-safe).

        Backward compatibility:
        - Without ``camera_name``, returns "rgb" camera if available, else first known camera.
        """
        with self._camera_state._lock:
            selected = camera_name or self._default_camera_name_locked()
            if selected is None:
                return None
            frame = self._camera_state.frames_by_name.get(selected)
            if frame is None or frame.rgb_image is None:
                return None
            return frame.rgb_image.copy()

    def get_depth_image(self, camera_name: str | None = None) -> np.ndarray | None:
        """Get latest depth image in millimeters (uint16)."""
        with self._camera_state._lock:
            selected = camera_name or self._default_camera_name_locked()
            if selected is None:
                return None
            frame = self._camera_state.frames_by_name.get(selected)
            if frame is None or frame.depth_image_mm is None:
                return None
            return frame.depth_image_mm.copy()

    def set_camera_rgb(
        self,
        camera_name: str,
        image: np.ndarray,
        timestamp_ns: int | None = None,
        *,
        copy: bool = True,
        timing: CameraFrameTiming | None = None,
    ) -> None:
        """Set RGB image for a named camera.

        ``copy=False`` stores *image* by reference instead of copying it. Only
        pass it when the caller has just produced a fresh array it will not
        touch again -- notably NOT a bare ``np.asanyarray(frame.get_data())``
        view, which aliases a RealSense buffer that the SDK recycles.
        """
        ts_ns = (
            timing.timestamp_ns
            if timing is not None
            else timestamp_ns if timestamp_ns is not None else time.time_ns()
        )
        image_copy = image if not copy else image.copy()
        with self._camera_state._lock:
            frame = self._get_or_create_camera_frame_locked(camera_name)
            frame.rgb_image = image_copy
            frame.rgb_timestamp_ns = ts_ns
            frame.rgb_timing = timing
        if self._on_change_callback:
            ts_sec = ts_ns / 1_000_000_000.0
            # Pass the stored array by reference (no extra copy): writers rebind
            # frame.rgb_image to a new array rather than mutating in place, so a
            # consumer can safely hold this. Avoids ~1 MB/frame/camera of wasted
            # GIL-held copying that otherwise starves the camera/control threads.
            self._on_change_callback(
                f"log_camera_rgb/{camera_name}", image_copy, ts_sec
            )
            if camera_name == "rgb":
                # Backward compatibility for existing single-camera callbacks.
                self._on_change_callback("log_rgb", image_copy, ts_sec)
        sink = self._camera_frame_sink
        if sink is not None:
            offer_timed_frame(
                sink, "offer_rgb", camera_name, image_copy, ts_ns, timing=timing
            )

    def set_camera_depth(
        self,
        camera_name: str,
        depth_image_mm: np.ndarray,
        timestamp_ns: int | None = None,
        *,
        copy: bool = True,
        timing: CameraFrameTiming | None = None,
    ) -> None:
        """Set depth image (uint16 millimeters) for a named camera.

        ``copy=False`` stores *depth_image_mm* by reference; see
        :meth:`set_camera_rgb` for the ownership contract it requires.
        """
        ts_ns = (
            timing.timestamp_ns
            if timing is not None
            else timestamp_ns if timestamp_ns is not None else time.time_ns()
        )
        depth_copy = depth_image_mm if not copy else depth_image_mm.copy()
        with self._camera_state._lock:
            frame = self._get_or_create_camera_frame_locked(camera_name)
            frame.depth_image_mm = depth_copy
            frame.depth_timestamp_ns = ts_ns
            frame.depth_timing = timing
        if self._on_change_callback:
            # Pass by reference (no extra copy) -- see set_camera_rgb.
            self._on_change_callback(
                f"log_camera_depth/{camera_name}",
                depth_copy,
                ts_ns / 1_000_000_000.0,
            )
        sink = self._camera_frame_sink
        if sink is not None:
            if getattr(sink, "supports_frame_timing", False):
                offer_timed_frame(
                    sink, "offer_depth", camera_name, depth_copy, ts_ns, timing=timing
                )
            else:
                sink.offer_depth(camera_name, depth_copy)

    def set_rgb_image(self, image: np.ndarray) -> None:
        """Set RGB image for the default single-camera path."""
        self.set_camera_rgb("rgb", image)

    # ============================================================================
    # Controller State Methods
    # ============================================================================

    def get_controller_data(
        self, hand: str | None = None
    ) -> tuple[np.ndarray | None, float, float]:
        """Get current controller data (thread-safe).

        Returns:
            Tuple of (controller_transform, grip_value, trigger_value)
        """
        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        with state._lock:
            return (
                state.transform.copy() if state.transform is not None else None,
                state.grip_value,
                state.trigger_value,
            )

    @overload
    def set_controller_data(
        self,
        hand: str,
        transform: np.ndarray | None,
        grip: float,
        trigger: float,
        *,
        pose_is_new: bool = True,
    ) -> None: ...

    @overload
    def set_controller_data(
        self,
        transform: np.ndarray | None,
        grip: float,
        trigger: float,
        *,
        pose_is_new: bool = True,
    ) -> None: ...

    def set_controller_data(self, *args: object, pose_is_new: bool = True) -> None:
        """Set controller data (thread-safe).

        Backward compatible overloads:
        - set_controller_data(transform, grip, trigger)               -> right hand
        - set_controller_data(hand, transform, grip, trigger)         -> specified hand

        ``pose_is_new=False`` means "this is the SAME pose you were already
        given" — the caller polled faster than the operator's frames arrive.
        Grip and trigger still update (the dead-man must never be stale), but the
        1 euro filter is left alone.

        That matters more than it looks. The filter is driven by the caller's
        wall clock, so re-feeding a duplicate sample makes it compute a near-zero
        derivative for that tick; ``cutoff = min_cutoff + beta*|dx|`` then
        collapses toward ``min_cutoff`` and the pose is smoothed hard — and the
        duplicates cluster exactly when the link is degraded, so the arm goes
        laggiest precisely when the operator most needs it crisp. Skipping keeps
        ``t_prev`` at the last real sample, so the next genuine frame is
        differentiated over the true interval.
        """
        if len(args) == 4 and isinstance(args[0], str):
            h = _normalize_hand(args[0])
            transform = args[1]
            grip = args[2]
            trigger = args[3]
        elif len(args) == 3:
            h = "right"
            transform = args[0]
            grip = args[1]
            trigger = args[2]
        else:
            raise TypeError(
                "set_controller_data expects (transform, grip, trigger) or (hand, transform, grip, trigger)"
            )

        if transform is not None:
            if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
                raise ValueError("Transform must be a 4x4 numpy array or None")
        if not isinstance(grip, (int, float)) or not (0.0 <= float(grip) <= 1.0):
            raise ValueError("Grip value must be between 0.0 and 1.0")
        if not isinstance(trigger, (int, float)) or not (0.0 <= float(trigger) <= 1.0):
            raise ValueError("Trigger value must be between 0.0 and 1.0")

        state = self._controller_state_by_hand[h]
        with state._lock:
            state.grip_value = float(grip)
            state.trigger_value = float(trigger)

            if transform is not None and not pose_is_new:
                # Same pose as last time: hold the filtered output and the
                # filter's clock. Raw is refreshed so debug views still match.
                state.transform_raw = transform.copy()
            elif transform is not None:
                current_time = time.time()

                # Store raw transform
                state.transform_raw = transform.copy()

                # Initialize filter if needed
                if state._filter is None:
                    state._filter = OneEuroFilterTransform(
                        current_time,
                        transform,
                        state.min_cutoff,
                        state.beta,
                        state.d_cutoff,
                    )
                    state.transform = transform.copy()
                else:
                    # Update filter parameters if they changed
                    state._filter.update_params(
                        state.min_cutoff, state.beta, state.d_cutoff
                    )

                    # Apply filter
                    state.transform = state._filter(current_time, transform)
            else:
                state.transform = None
                state.transform_raw = None
                state._filter = None  # Reset filter when transform is None

    def set_controller_filter_params(
        self, min_cutoff: float, beta: float, d_cutoff: float, hand: str | None = None
    ) -> None:
        """Update 1€ Filter parameters for controller transform (thread-safe).

        Args:
            min_cutoff: Minimum cutoff frequency (stabilizes when holding still)
            beta: Speed coefficient (reduces lag when moving)
            d_cutoff: Cutoff frequency for derivative filtering
            hand: "left" or "right" (optional). If None, applies to both hands.
        """
        if hand is None:
            for state in self._controller_state_by_hand.values():
                with state._lock:
                    state.min_cutoff = min_cutoff
                    state.beta = beta
                    state.d_cutoff = d_cutoff
            return

        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        with state._lock:
            state.min_cutoff = min_cutoff
            state.beta = beta
            state.d_cutoff = d_cutoff

    def get_controller_filter_params(
        self, hand: str | None = None
    ) -> tuple[float, float, float]:
        """Get 1€ Filter parameters for controller transform (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        with state._lock:
            return (state.min_cutoff, state.beta, state.d_cutoff)

    def set_joystick_y(self, value: float, hand: str | None = None) -> None:
        """Set joystick Y axis value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        with state._lock:
            state.joystick_y = (
                0.0 if time.time() < state.joystick_y_suppress_until else value
            )

    def get_joystick_y(self, hand: str | None = None) -> float:
        """Get joystick Y axis value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        with state._lock:
            return (
                0.0
                if time.time() < state.joystick_y_suppress_until
                else state.joystick_y
            )

    def suppress_joystick_y(
        self, hand: str | None = None, duration_s: float = 0.35
    ) -> None:
        """Temporarily ignore joystick Y input after stick-button actions."""
        h = _normalize_hand(hand)
        state = self._controller_state_by_hand[h]
        until = time.time() + max(0.0, float(duration_s))
        with state._lock:
            state.joystick_y = 0.0
            state.joystick_y_suppress_until = max(
                state.joystick_y_suppress_until, until
            )

    # ============================================================================
    # Teleop Gripper Settings
    # ============================================================================

    def get_teleop_gripper_settings(
        self, hand: str | None = None
    ) -> TeleopGripperSettings:
        """Get Quest grip-button gripper behavior for one hand."""
        h = _normalize_hand(hand)
        with self._teleop_gripper_lock:
            return TeleopGripperSettings(
                open_limit=float(self._teleop_gripper_open_limit_by_hand[h]),
                mode=self._teleop_gripper_mode_by_hand[h],
            )

    def get_teleop_gripper_settings_by_hand(
        self,
    ) -> dict[Hand, TeleopGripperSettings]:
        """Get Quest grip-button gripper behavior for both hands."""
        with self._teleop_gripper_lock:
            return {
                hand: TeleopGripperSettings(
                    open_limit=float(self._teleop_gripper_open_limit_by_hand[hand]),
                    mode=self._teleop_gripper_mode_by_hand[hand],
                )
                for hand in ("left", "right")
            }

    def set_teleop_gripper_settings(
        self,
        *,
        open_limit: float | None = None,
        mode: str | None = None,
        hand: str | None = None,
    ) -> TeleopGripperSettings:
        """Set Quest grip-button gripper behavior.

        Passing ``hand=None`` keeps the legacy API behavior and applies the
        update to both arms. Passing ``left`` or ``right`` updates only that
        arm's Quest grip behavior.
        """
        hands: tuple[Hand, ...]
        if hand is None:
            hands = ("left", "right")
            return_hand: Hand = "right"
        else:
            return_hand = _normalize_hand(hand)
            hands = (return_hand,)
        with self._teleop_gripper_lock:
            clipped_limit: float | None = None
            if open_limit is not None:
                try:
                    limit = float(open_limit)
                except (TypeError, ValueError) as exc:
                    raise ValueError("open_limit must be numeric") from exc
                if not np.isfinite(limit):
                    raise ValueError("open_limit must be finite")
                clipped_limit = float(np.clip(limit, 0.0, 1.0))

            normalized_mode: TeleopGripperMode | None = None
            if mode is not None:
                normalized = str(mode).strip().lower()
                if normalized not in ("hold", "toggle"):
                    raise ValueError("mode must be 'hold' or 'toggle'")
                normalized_mode = normalized  # type: ignore[assignment]

            for h in hands:
                if clipped_limit is not None:
                    self._teleop_gripper_open_limit_by_hand[h] = clipped_limit
                if normalized_mode is not None:
                    self._teleop_gripper_mode_by_hand[h] = normalized_mode

            return TeleopGripperSettings(
                open_limit=float(self._teleop_gripper_open_limit_by_hand[return_hand]),
                mode=self._teleop_gripper_mode_by_hand[return_hand],
            )

    # ============================================================================
    # Teleop State Methods
    # ============================================================================

    @overload
    def set_teleop_state(
        self,
        hand: str,
        active: bool,
        controller_initial: np.ndarray | None,
        robot_initial: np.ndarray | None,
    ) -> None: ...

    @overload
    def set_teleop_state(
        self,
        active: bool,
        controller_initial: np.ndarray | None,
        robot_initial: np.ndarray | None,
    ) -> None: ...

    def set_teleop_state(self, *args: object) -> None:
        """Set teleoperation state (thread-safe).

        Backward compatible overloads:
        - set_teleop_state(active, controller_initial, robot_initial)                  -> right hand
        - set_teleop_state(hand, active, controller_initial, robot_initial)           -> specified hand
        """
        if len(args) == 4 and isinstance(args[0], str):
            h = _normalize_hand(args[0])
            active = args[1]
            controller_initial = args[2]
            robot_initial = args[3]
        elif len(args) == 3:
            h = "right"
            active = args[0]
            controller_initial = args[1]
            robot_initial = args[2]
        else:
            raise TypeError(
                "set_teleop_state expects (active, controller_initial, robot_initial) or (hand, active, controller_initial, robot_initial)"
            )

        if not isinstance(active, bool):
            raise TypeError("active must be a bool")

        if controller_initial is not None and (
            not isinstance(controller_initial, np.ndarray)
            or controller_initial.shape != (4, 4)
        ):
            raise ValueError("controller_initial must be a 4x4 numpy array or None")
        if robot_initial is not None and (
            not isinstance(robot_initial, np.ndarray) or robot_initial.shape != (4, 4)
        ):
            raise ValueError("robot_initial must be a 4x4 numpy array or None")

        state = self._teleop_state_by_hand[h]
        with state._lock:
            state.generation += 1
            state.active = active
            state.controller_initial_transform = (
                controller_initial.copy() if controller_initial is not None else None
            )
            state.robot_initial_transform = (
                robot_initial.copy() if robot_initial is not None else None
            )

    def set_operator_connection_provider(self, provider: Callable[[], dict]) -> None:
        """Register a nonblocking, live transport snapshot for the dashboard."""
        self._operator_connection_provider = provider

    def get_operator_connection(self) -> dict:
        """Report current transport health independently of hardware availability."""
        provider = self._operator_connection_provider
        return provider() if provider is not None else {"state": "unavailable"}

    def get_teleop_active(self, hand: str | None = None) -> bool:
        """Get teleoperation active state (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._teleop_state_by_hand[h]
        with state._lock:
            return state.active

    def commit_teleop_target(self, hand: str, generation: int, target: np.ndarray) -> bool:
        """Publish only for this clutch; mark its first IK target as ready."""
        state = self._teleop_state_by_hand[_normalize_hand(hand)]
        with state._lock:
            if not state.active or state.generation != generation:
                return False
            self.set_target_joint_angles(target, hand)
            state.ready_generation = generation
            return True

    def prepare_teleop_baseline(self, hand: str, generation: int, target: np.ndarray) -> bool:
        """Acknowledge the controller-side hold before IK starts this clutch."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            return False
        state = self._teleop_state_by_hand[_normalize_hand(hand)]
        with state._lock:
            if not state.active or state.generation != generation:
                return False
            state.command_baseline = target.copy()
            state.baseline_generation = generation
            return True

    def get_teleop_baseline(self, hand: str, generation: int) -> np.ndarray | None:
        state = self._teleop_state_by_hand[_normalize_hand(hand)]
        with state._lock:
            if (not state.active or state.generation != generation
                    or state.baseline_generation != generation):
                return None
            return state.command_baseline.copy()

    def get_ready_teleop_target(self, hand: str) -> np.ndarray | None:
        """Never expose a target left over from a previous clutch."""
        state = self._teleop_state_by_hand[_normalize_hand(hand)]
        with state._lock:
            if not state.active or getattr(state, "ready_generation", -1) != state.generation:
                return None
            return self.get_target_joint_angles(hand)

    def get_initial_robot_controller_transforms(
        self, hand: str | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get initial robot and controller transforms (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._teleop_state_by_hand[h]
        with state._lock:
            return (
                (
                    state.controller_initial_transform.copy()
                    if state.controller_initial_transform is not None
                    else None
                ),
                (
                    state.robot_initial_transform.copy()
                    if state.robot_initial_transform is not None
                    else None
                ),
            )

    def get_teleop_snapshot(
        self, hand: str | None = None
    ) -> tuple[bool, int, np.ndarray | None, np.ndarray | None]:
        """Read one clutch generation and its anchors under the same lock."""
        state = self._teleop_state_by_hand[_normalize_hand(hand)]
        with state._lock:
            return (
                state.active,
                state.generation,
                (
                    None if state.controller_initial_transform is None
                    else state.controller_initial_transform.copy()
                ),
                (
                    None if state.robot_initial_transform is None
                    else state.robot_initial_transform.copy()
                ),
            )

    def set_teleop_delta_rpy_offset_deg(
        self,
        offset_deg: np.ndarray | list[float] | tuple[float, float, float],
        hand: str | None = None,
    ) -> None:
        """Set fixed teleop frame correction [roll, pitch, yaw] in degrees."""
        arr = np.asarray(offset_deg, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError("offset_deg must be a 3-element [roll, pitch, yaw] vector")
        arr = arr.copy()

        if hand is None:
            for state in self._teleop_state_by_hand.values():
                with state._lock:
                    state.delta_rpy_offset_deg = arr.copy()
            return

        h = _normalize_hand(hand)
        state = self._teleop_state_by_hand[h]
        with state._lock:
            state.delta_rpy_offset_deg = arr.copy()

    def get_teleop_delta_rpy_offset_deg(self, hand: str | None = None) -> np.ndarray:
        """Get fixed teleop frame correction [roll, pitch, yaw] in degrees."""
        h = _normalize_hand(hand)
        state = self._teleop_state_by_hand[h]
        with state._lock:
            return state.delta_rpy_offset_deg.copy()

    # ============================================================================
    # Robot State Methods
    # ============================================================================

    def get_robot_activity_state(self, hand: str | None = None) -> RobotActivityState:
        """Get robot activity state (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return state.activity_state

    def set_robot_activity_state(
        self, state: RobotActivityState, hand: str | None = None
    ) -> None:
        """Set robot activity state (thread-safe)."""
        h = _normalize_hand(hand)
        rstate = self._robot_state_by_hand[h]
        with rstate._lock:
            rstate.activity_state = state
        _set_robot_activity_state(h, _ACTIVITY_STATE_CODES.get(state.value, -1.0))

    def get_current_joint_angles(self, hand: str | None = None) -> np.ndarray | None:
        """Get current joint angles (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return state.joint_angles.copy() if state.joint_angles is not None else None

    def set_current_joint_angles(
        self,
        angles: np.ndarray,
        hand: str | None = None,
        *,
        timing: FeedbackTiming | None = None,
    ) -> None:
        """Set current joint angles (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.joint_angles = angles.copy()
        if self._on_change_callback:
            self._on_change_callback(
                f"log_joint_positions/{h}",
                state.joint_angles.copy(),
                timing.timestamp_s if timing is not None else time.time(),
            )
            if timing is not None:
                self.log_telemetry_signal(
                    "log_joint_feedback_timing",
                    timing.as_values(),
                    h,
                    timing.timestamp_s,
                )

    def get_current_joint_currents_a(
        self, hand: str | None = None
    ) -> np.ndarray | None:
        """Get current joint motor currents in amps (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return (
                state.joint_currents_a.copy()
                if state.joint_currents_a is not None
                else None
            )

    def set_current_joint_currents_a(
        self, currents_a: np.ndarray, hand: str | None = None
    ) -> None:
        """Set current joint motor currents in amps (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.joint_currents_a = currents_a.copy()
        if self._on_change_callback:
            self._on_change_callback(
                f"log_joint_currents/{h}",
                state.joint_currents_a.copy(),
                time.time(),
            )

    def get_current_joint_velocities_rad_s(
        self, hand: str | None = None
    ) -> np.ndarray | None:
        """Get current joint motor velocities in radians/second (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return (
                state.joint_velocities_rad_s.copy()
                if state.joint_velocities_rad_s is not None
                else None
            )

    def set_current_joint_velocities_rad_s(
        self, velocities_rad_s: np.ndarray, hand: str | None = None
    ) -> None:
        """Set current joint motor velocities in radians/second (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.joint_velocities_rad_s = velocities_rad_s.copy()
        if self._on_change_callback:
            self._on_change_callback(
                f"log_joint_velocities/{h}",
                state.joint_velocities_rad_s.copy(),
                time.time(),
            )

    def get_current_end_effector_pose(
        self, hand: str | None = None
    ) -> np.ndarray | None:
        """Get current end effector pose (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return (
                state.end_effector_pose.copy()
                if state.end_effector_pose is not None
                else None
            )

    def set_current_end_effector_pose(
        self, pose: np.ndarray, hand: str | None = None
    ) -> None:
        """Set current end effector pose (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.end_effector_pose = pose.copy()

    def get_current_gripper_open_value(self, hand: str | None = None) -> float | None:
        """Get current gripper open value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return state.current_gripper_open_value

    def set_current_gripper_open_value(
        self,
        value: float,
        hand: str | None = None,
        *,
        timing: FeedbackTiming | None = None,
    ) -> None:
        """Set current gripper open value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.current_gripper_open_value = value
        if self._on_change_callback:
            self._on_change_callback(
                f"log_parallel_gripper_open_amounts/{h}",
                value,
                timing.timestamp_s if timing is not None else time.time(),
            )
            if timing is not None:
                self.log_telemetry_signal(
                    "log_gripper_feedback_timing",
                    timing.as_values(),
                    h,
                    timing.timestamp_s,
                )

    def get_target_gripper_open_value(self, hand: str | None = None) -> float | None:
        """Get target gripper open value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            return state.target_gripper_open_value

    def set_target_gripper_open_value(
        self, value: float, hand: str | None = None
    ) -> None:
        """Set target gripper open value (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._robot_state_by_hand[h]
        with state._lock:
            state.target_gripper_open_value = value
        if self._on_change_callback:
            self._on_change_callback(
                f"log_parallel_gripper_target_open_amounts/{h}",
                state.target_gripper_open_value,
                time.time(),
            )

    # ============================================================================
    # IK State Methods
    # ============================================================================

    def get_target_joint_angles(self, hand: str | None = None) -> np.ndarray | None:
        """Get current target joint angles (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            return (
                state.target_joint_angles.copy()
                if state.target_joint_angles is not None
                else None
            )

    def set_target_joint_angles(
        self, angles: np.ndarray, hand: str | None = None
    ) -> None:
        """Set target joint angles (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            state.target_joint_angles = angles.copy()
        if self._on_change_callback:
            self._on_change_callback(
                f"log_joint_target_positions/{h}",
                state.target_joint_angles.copy(),
                time.time(),
            )

    def set_target_pose(
        self, transform: np.ndarray | None, hand: str | None = None
    ) -> None:
        """Set target transform for visualization (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            state.target_pose = transform.copy() if transform is not None else None

    def get_target_pose(self, hand: str | None = None) -> np.ndarray | None:
        """Get target transform for visualization (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            return state.target_pose.copy() if state.target_pose is not None else None

    def set_ik_solve_time_ms(self, time_ms: float, hand: str | None = None) -> None:
        """Set IK solve time (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            state.solve_time_ms = time_ms

    def set_ik_success(self, success: bool, hand: str | None = None) -> None:
        """Set IK success (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            state.success = success

    def get_ik_solve_time_ms(self, hand: str | None = None) -> float:
        """Get IK solve time (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            return state.solve_time_ms

    def get_ik_success(self, hand: str | None = None) -> bool:
        """Get IK success (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._ik_state_by_hand[h]
        with state._lock:
            return state.success

    # ============================================================================
    # System State Methods
    # ============================================================================

    def request_shutdown(self) -> None:
        """Request shutdown of all threads (lock-free using Event)."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown is requested (lock-free using Event)."""
        return self._shutdown_event.is_set()

    # ============================================================================
    # Policy take-over (teleop trigger squeezed while a hand is POLICY_CONTROLLED)
    # ============================================================================

    def request_policy_takeover(self, hand: str | None = None) -> None:
        """File an operator take-over request for ``hand`` (quest reader -> policy runner)."""
        h = _normalize_hand(hand)
        with self._policy_takeover_lock:
            self._policy_takeover_hands.add(h)

    def consume_policy_takeover(self) -> tuple[Hand, ...]:
        """Return and clear the pending take-over requests (policy runner side)."""
        order: tuple[Hand, ...] = ("right", "left")
        with self._policy_takeover_lock:
            hands = tuple(h for h in order if h in self._policy_takeover_hands)
            self._policy_takeover_hands.clear()
        return hands

    def policy_takeover_pending(self) -> tuple[Hand, ...]:
        """Hands with a take-over request not yet consumed (read-only, for status)."""
        order: tuple[Hand, ...] = ("right", "left")
        with self._policy_takeover_lock:
            return tuple(h for h in order if h in self._policy_takeover_hands)

    # ============================================================================
    # Brain Control State Methods
    # ============================================================================

    def get_brain_control(self, hand: str | None = None) -> BrainControlSnapshot:
        """Get brain control snapshot (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._brain_control_state_by_hand[h]
        with state._lock:
            slider = state.slider_norm.copy()
            baseline = (
                state.baseline_deg.copy() if state.baseline_deg is not None else None
            )
            last_cmd = (
                state.last_commanded_deg.copy()
                if state.last_commanded_deg is not None
                else None
            )
            return BrainControlSnapshot(
                armed=state.armed,
                diameter=float(state.diameter),
                slider_norm=slider,
                clamp_mode=state.clamp_mode,
                active_group=str(state.active_group),
                selected_alpha=float(state.selected_alpha),
                unselected_alpha=float(state.unselected_alpha),
                baseline_deg=baseline,
                last_commanded_deg=last_cmd,
            )

    def get_brain_control_armed(self, hand: str | None = None) -> bool:
        """Get brain control armed status (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._brain_control_state_by_hand[h]
        with state._lock:
            return bool(state.armed)

    def set_brain_control_params(
        self,
        *,
        armed: bool,
        diameter: float,
        slider_norm: np.ndarray,
        clamp_mode: ClampMode,
        active_group: str,
        selected_alpha: float,
        unselected_alpha: float,
        hand: str | None = None,
    ) -> None:
        """Set brain control parameters from UI (thread-safe).

        Notes:
            - Does not automatically capture baseline (call capture_brain_baseline()).
            - Resets last_commanded_deg when disarmed (prevents stale smoothing state).
        """
        h = _normalize_hand(hand)
        state = self._brain_control_state_by_hand[h]

        slider = np.array(slider_norm, dtype=np.float64).reshape(-1)
        if slider.size < 6:
            slider = np.pad(
                slider, (0, 6 - slider.size), mode="constant", constant_values=0.0
            )
        slider = slider[:6]
        slider = np.clip(slider, -1.0, 1.0)

        with state._lock:
            prev_armed = state.armed
            state.armed = bool(armed)
            state.diameter = float(diameter)
            state.slider_norm = slider
            state.clamp_mode = clamp_mode
            state.active_group = str(active_group)
            state.selected_alpha = float(selected_alpha)
            state.unselected_alpha = float(unselected_alpha)

            # Clear smoothing state on (re)arm/disarm to avoid jumps.
            if (not state.armed) or (state.armed and not prev_armed):
                state.last_commanded_deg = None

    def set_brain_control_baseline_deg(
        self, baseline_deg: np.ndarray | None, hand: str | None = None
    ) -> None:
        """Set baseline (deg) used by brain control (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._brain_control_state_by_hand[h]
        with state._lock:
            if baseline_deg is None:
                state.baseline_deg = None
                return
            baseline = np.array(baseline_deg, dtype=np.float64).reshape(-1)
            if baseline.size < 6:
                baseline = np.pad(baseline, (0, 6 - baseline.size), mode="edge")
            state.baseline_deg = baseline[:6].copy()

    def set_brain_control_last_commanded_deg(
        self, last_commanded_deg: np.ndarray | None, hand: str | None = None
    ) -> None:
        """Set last commanded joints (deg) for smoothing continuity (thread-safe)."""
        h = _normalize_hand(hand)
        state = self._brain_control_state_by_hand[h]
        with state._lock:
            if last_commanded_deg is None:
                state.last_commanded_deg = None
                return
            arr = np.array(last_commanded_deg, dtype=np.float64).reshape(-1)
            if arr.size < 6:
                arr = np.pad(arr, (0, 6 - arr.size), mode="edge")
            state.last_commanded_deg = arr[:6].copy()

    def capture_brain_baseline(self, hand: str | None = None) -> bool:
        """Capture baseline from current measured joint angles (deg)."""
        baseline = self.get_current_joint_angles(hand)
        if baseline is None:
            return False
        self.set_brain_control_baseline_deg(baseline, hand)
        self.set_brain_control_last_commanded_deg(baseline, hand)
        return True
