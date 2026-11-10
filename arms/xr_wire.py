"""Transport-neutral wire format for Quest controller teleoperation.

Ships Meta Quest controller state from the *leader* machine (the PC/headset the
operator drives from) to the *follower* machine (the robot host). The JSON
encoding here is deliberately transport-agnostic: it is carried verbatim over
Agora RTM message channels, LiveKit data channels, and the localhost UDP hop
between the Agora control bridge and the teleop process.

Both sides import this module so the schema stays in lock-step.
"""

from __future__ import annotations

import json
import time

import numpy as np

# -- Naming -------------------------------------------------------------------

DEFAULT_LEADER_NAME = "quest-leader"
DEFAULT_FOLLOWER_NAME = "piper-follower"

HANDS = ("left", "right")

# Buttons the runtime registers callbacks for (see bimanual_teleop_runtime).
BUTTON_NAMES = ("a", "b", "x", "y", "rj")


# -- XR controller frame ------------------------------------------------------


def encode_xr(
    transform: np.ndarray | None,
    grip: float,
    trigger: float,
    joystick_y: float,
) -> bytes:
    """Encode one hand's controller state to JSON bytes.

    ``transform`` is a 4x4 ROS-frame pose (as returned by
    ``MetaQuestReader.get_hand_controller_transform_ros``) or ``None`` when
    the pose is unavailable.
    """
    if transform is None:
        flat: list[float] = []
        valid = False
    else:
        arr = np.asarray(transform, dtype=np.float64)
        valid = arr.shape == (4, 4)
        flat = arr.ravel().tolist() if valid else []

    return json.dumps(
        {
            "transform": flat,
            "grip": float(grip),
            "trigger": float(trigger),
            "joystick_y": float(joystick_y),
            "stamp": time.time(),
            "valid": valid,
        },
        separators=(",", ":"),
    ).encode()


def decode_xr_obj(obj: dict) -> tuple[np.ndarray | None, float, float, float]:
    """Decode an ALREADY-PARSED per-hand frame dict.

    Split out from :func:`decode_xr` so a caller holding the parsed combined
    frame can decode each hand without re-serialising it. ``decode_xr_frame``
    used to do ``decode_xr(json.dumps(sub))`` per hand -- a dumps plus a loads
    purely to hand a dict to a function that immediately parsed it back. At
    50 Hz x 2 hands that was ~200 needless JSON operations a second on the
    process whose GIL the IK solvers and joint-state threads all share.
    """
    transform: np.ndarray | None = None
    if obj.get("valid") and len(obj.get("transform", [])) == 16:
        transform = np.asarray(obj["transform"], dtype=np.float64).reshape(4, 4)
    return (
        transform,
        float(obj.get("grip", 0.0)),
        float(obj.get("trigger", 0.0)),
        float(obj.get("joystick_y", 0.0)),
    )


def decode_xr(payload: bytes) -> tuple[np.ndarray | None, float, float, float]:
    """Decode an XR frame to ``(transform, grip, trigger, joystick_y)``."""
    return decode_xr_obj(json.loads(payload))


# -- Button event -------------------------------------------------------------


def encode_button(name: str) -> bytes:
    """Encode a button-press event to JSON bytes."""
    return json.dumps(
        {"name": str(name), "stamp": time.time()},
        separators=(",", ":"),
    ).encode()


def decode_button(payload: bytes) -> str:
    """Decode a button event, returning the button name (e.g. ``"a"``)."""
    return str(json.loads(payload).get("name", ""))
