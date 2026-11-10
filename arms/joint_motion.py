"""Teleop reference shaping; degrees throughout, no hardware side effects."""

import numpy as np


def limit_tracking_step(reference, target, measured, limits):
    """Scale one joint-space step to fit the measured-angle tracking envelope.

    A common scale preserves the proposed step's direction. If feedback moved
    an existing reference outside the envelope, allow recovery toward it but
    never push that joint farther out or jump the reference to measured angles.
    Returns the accepted target, scale, and binding joint (None when unrestricted).
    """
    reference, target, measured, limits = (
        np.asarray(value, dtype=float).reshape(6)
        for value in (reference, target, measured, limits)
    )
    if not all(np.all(np.isfinite(value)) for value in (reference, target, measured, limits)):
        raise ValueError("Tracking angles and limits must be finite")
    if np.any(limits <= 0):
        raise ValueError("Tracking limits must be positive")
    lower = np.minimum(measured - limits, reference)
    upper = np.maximum(measured + limits, reference)
    step = target - reference
    fractions = np.ones(6)
    positive = step > 0
    negative = step < 0
    fractions[positive] = (upper[positive] - reference[positive]) / step[positive]
    fractions[negative] = (lower[negative] - reference[negative]) / step[negative]
    joint = int(np.argmin(fractions))
    scale = float(np.clip(fractions[joint], 0.0, 1.0))
    if scale >= 1.0:
        return target.copy(), 1.0, None
    return reference + scale * step, scale, joint


class JointAccelerationLimiter:
    """Bound discrete command velocity and acceleration between accepted samples.

    Reversals may pass the requested target while braking. Joint-limit clamps,
    clutch changes and emergency holds take precedence over smoothness; callers
    must reset after those discontinuities. This does not bound jerk.
    """

    def __init__(self, velocity_limits, acceleration_limits):
        self.velocity_limits = self._limits(velocity_limits)
        self.acceleration_limits = self._limits(acceleration_limits)
        self.position = None
        self.velocity = np.zeros(6)

    @staticmethod
    def _limits(values):
        values = np.asarray(values, dtype=float).reshape(6)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Joint motion limits must be finite and positive")
        return values.copy()

    def reset(self, position):
        position = np.asarray(position, dtype=float).reshape(6)
        if not np.all(np.isfinite(position)):
            raise ValueError("Joint reference must be finite")
        self.position = position.copy()
        self.velocity = np.zeros(6)

    def step(self, target, dt):
        target = np.asarray(target, dtype=float).reshape(6)
        if self.position is None:
            raise ValueError("Reset the limiter before stepping")
        if not np.all(np.isfinite(target)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Target and positive timestep must be finite")
        error = target - self.position
        # Discrete braking envelope: reserve one timestep's travel before the
        # continuous stopping distance, avoiding chatter around a fixed target.
        a = self.acceleration_limits
        braking_speed = 2 * a * np.abs(error) / (
            np.sqrt((a * dt) ** 2 + 2 * a * np.abs(error)) + a * dt
        )
        desired = np.sign(error) * np.minimum(
            self.velocity_limits, np.minimum(np.abs(error) / dt, braking_speed)
        )
        self.velocity += np.clip(desired - self.velocity, -a * dt, a * dt)
        self.position += self.velocity * dt
        return self.position.copy()
