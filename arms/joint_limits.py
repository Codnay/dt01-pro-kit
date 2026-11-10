"""Joint-limit recovery rules shared by IK and the hardware controller."""

import numpy as np


def recovery_joint_bounds(
    reference_deg: np.ndarray,
    soft_limits_deg: np.ndarray,
    hard_limits_deg: np.ndarray,
) -> np.ndarray:
    """Allow a valid held reference to stay still or return inside soft limits.

    Bounds shrink with each accepted reference. A joint already inside its
    soft range can never enter the margin; one outside cannot move farther out.
    Hardware/model limits are never expanded, including for the initial hold.
    """
    reference = np.asarray(reference_deg, dtype=np.float64)
    soft = np.asarray(soft_limits_deg, dtype=np.float64)
    hard = np.asarray(hard_limits_deg, dtype=np.float64)
    if reference.shape != (6,) or soft.shape != (6, 2) or hard.shape != (6, 2):
        raise ValueError("Joint reference/limits must have shapes (6,) and (6, 2)")
    if not all(np.all(np.isfinite(value)) for value in (reference, soft, hard)):
        raise ValueError("Joint reference and limits must be finite")
    if (
        np.any(soft[:, 0] >= soft[:, 1])
        or np.any(soft[:, 0] < hard[:, 0])
        or np.any(soft[:, 1] > hard[:, 1])
    ):
        raise ValueError("Soft joint limits must be nonempty and inside hard limits")
    outside = (reference < hard[:, 0]) | (reference > hard[:, 1])
    if np.any(outside):
        raise ValueError(
            "Held target outside hard limits: " + joint_limit_details(reference, hard)
        )
    return np.column_stack(
        (np.minimum(soft[:, 0], reference), np.maximum(soft[:, 1], reference))
    )


def clamp_recovery_step(
    reference_deg: np.ndarray,
    target_deg: np.ndarray,
    soft_limits_deg: np.ndarray,
    hard_limits_deg: np.ndarray,
) -> np.ndarray:
    """Contain a step without snapping an outside reference to the soft limit."""
    target = np.asarray(target_deg, dtype=np.float64)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise ValueError("Joint target must be finite with shape (6,)")
    bounds = recovery_joint_bounds(reference_deg, soft_limits_deg, hard_limits_deg)
    return np.clip(target, bounds[:, 0], bounds[:, 1])


def joint_limit_details(target_deg: np.ndarray, limits_deg: np.ndarray) -> str:
    """Describe every out-of-range joint, including its signed excess."""
    target = np.asarray(target_deg, dtype=np.float64)
    limits = np.asarray(limits_deg, dtype=np.float64)
    excess = target - np.clip(target, limits[:, 0], limits[:, 1])
    return "; ".join(
        f"J{i + 1} held={target[i]:.6f} deg allowed="
        f"[{limits[i, 0]:.6f}, {limits[i, 1]:.6f}] deg "
        f"excess={excess[i]:+.6f} deg"
        for i in range(6)
        if excess[i] != 0 or not np.isfinite(target[i])
    )
