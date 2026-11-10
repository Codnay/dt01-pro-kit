"""Utility functions for the examples."""

import numpy as np
from scipy.spatial.transform import Rotation


def scale_and_add_delta_transform(
    delta_position: np.ndarray,
    delta_orientation: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
    initial_transform: np.ndarray,
    delta_rotvec: np.ndarray | None = None,
) -> np.ndarray:
    """Scale and add delta position and orientation to an initial transform.

    Args:
        delta_position: 3D position vector
        delta_orientation: 3x3 rotation matrix
        translation_scale: translation scale
        rotation_scale: rotation scale
        initial_transform: 4x4 transformation matrix
        delta_rotvec: optional precomputed/unwrapped axis-angle delta

    Returns:
        4x4 transformation matrix
    """
    scaled_delta_position = delta_position * translation_scale

    # Scale the delta rotation by converting to axis-angle, scaling the angle,
    # then converting back.
    if delta_rotvec is None:
        delta_rotation = Rotation.from_matrix(delta_orientation)
        delta_axis_angle = delta_rotation.as_rotvec()
    else:
        delta_axis_angle = np.asarray(delta_rotvec, dtype=np.float64).reshape(3)
    scaled_delta_axis_angle = delta_axis_angle * rotation_scale
    scaled_delta_rotation = Rotation.from_rotvec(scaled_delta_axis_angle).as_matrix()

    # Apply controller delta in world frame (matches delta_orientation construction).
    initial_rotation = initial_transform[:3, :3]
    new_rotation = scaled_delta_rotation @ initial_rotation

    target_transform = np.eye(4)
    target_transform[:3, 3] = initial_transform[:3, 3] + scaled_delta_position
    target_transform[:3, :3] = new_rotation

    return target_transform


def limit_joint_step(
    previous: np.ndarray,
    target: np.ndarray,
    caps_deg_s: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, bool]:
    """Rate-limit a 6-joint target relative to the previous command.

    Scales the whole joint-delta vector by a single factor so the move keeps its
    direction (the arm follows the intended path, just slower) when any joint
    would exceed its per-joint speed cap. Shared by the teleop slew guard and the
    preset/trajectory replay path so both rate-limit identically.

    Args:
        previous: last commanded joint angles (degrees), 6 joints.
        target: requested joint angles (degrees), 6 joints.
        caps_deg_s: per-joint speed caps (deg/s), 6 joints.
        dt_s: time step (seconds) since the previous command.

    Returns:
        ``(limited_target_deg, was_limited)`` -- the (possibly) rate-limited
        target and whether any cap was hit.

    Raises:
        ValueError: if any input does not have exactly 6 joints.
    """
    previous_arr = np.asarray(previous, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    caps_arr = np.asarray(caps_deg_s, dtype=np.float64).reshape(-1)
    if previous_arr.size != 6 or target_arr.size != 6 or caps_arr.size != 6:
        raise ValueError("Joint speed limiting requires 6 joints")
    dt = max(float(dt_s), 1e-6)
    delta = target_arr - previous_arr
    abs_delta = np.abs(delta)
    max_delta = caps_arr * dt
    over = abs_delta > (max_delta + 1e-9)
    if not np.any(over):
        return target_arr.copy(), False
    scale = float(np.min(max_delta[over] / abs_delta[over]))
    scale = float(np.clip(scale, 0.0, 1.0))
    return previous_arr + delta * scale, True
