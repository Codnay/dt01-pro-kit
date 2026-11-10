"""Keep PiPER feedback values paired with the SDK's CAN receive time."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeedbackTiming:
    """Epoch seconds; SDK receive time is not a motor acquisition timestamp."""

    timestamp_s: float
    polled_at_s: float
    sdk_received_at_s: float | None
    source: str

    def as_values(self) -> list[float]:
        """Diagnostic row: event, poll, raw SDK time, source (1=SDK, 0=poll)."""
        return [
            self.timestamp_s,
            self.polled_at_s,
            self.sdk_received_at_s or 0.0,
            float(self.source == "sdk_receive"),
        ]


def feedback_timing(raw_timestamp: Any, polled_at_s: float) -> FeedbackTiming:
    """Preserve old feedback times; never freshen a stale sample at publication."""
    raw: float | None
    try:
        raw = float(raw_timestamp)
        if not math.isfinite(raw):
            raw = None
    except (TypeError, ValueError):
        raw = None
    valid = raw is not None and 946684800.0 <= raw <= polled_at_s + 0.001
    return FeedbackTiming(
        raw if valid and raw is not None else polled_at_s,
        polled_at_s,
        raw,
        "sdk_receive" if valid else "poll_fallback",
    )


def read_sdk_feedback(sdk: Any, kind: str) -> tuple[tuple[float, ...], FeedbackTiming]:
    """Copy values AND time under the pinned SDK's own feedback mutex.

    piper-sdk 0.6.1 public getters release that mutex before returning a mutable
    object. Reading its timestamp separately can pair a new message with old
    values. Keep this version-specific access isolated here. Other SDKs retain
    the old getter behavior, explicitly labelled as a poll-time fallback.
    The joint timestamp belongs to the most recently received joint-pair CAN
    message; the firmware does not expose simultaneous acquisition of all six.
    """
    if kind not in ("joint", "gripper"):
        raise ValueError(f"Unsupported feedback kind: {kind}")
    prefix = f"_C_PiperInterface_V2__arm_{kind}_msgs"
    lock = getattr(sdk, prefix + "_mtx", None)

    def values(raw: Any) -> tuple[float, ...]:
        if kind == "joint":
            return tuple(
                float(getattr(raw.joint_state, f"joint_{i}")) for i in range(1, 7)
            )
        return (float(raw.gripper_state.grippers_angle),)

    if lock is not None and hasattr(sdk, prefix):
        with lock:
            raw = getattr(sdk, prefix)
            copied = values(raw)
            stamp = getattr(raw, "time_stamp", None)
        return copied, feedback_timing(stamp, time.time())
    getter = sdk.GetArmJointMsgs if kind == "joint" else sdk.GetArmGripperMsgs
    copied = values(getter())
    return copied, feedback_timing(None, time.time())
