"""Pick frames on an even time grid when a stream is downsampled."""

from __future__ import annotations

from fractions import Fraction

from arms.camera_timing import CameraFrameTiming


class CaptureRateGate:
    """Select unique captures on an absolute source-time grid, not arrival time."""

    def __init__(self, fps: float) -> None:
        """Keep fractional phase when downsampling to the requested rate."""
        rate = Fraction(str(max(1.0, fps)))
        self.numerator = rate.numerator
        self.denominator = rate.denominator * 1_000_000_000
        self.origin: int | None = None
        self.last_stamp = 0
        self.last_tick = -1
        self.last_number: int | None = None
        self.missing = 0
        self.resets = 0

    def select(self, stamp: int, timing: CameraFrameTiming | None) -> str:
        """Return the reason for retaining or rejecting this source sample."""
        if stamp == self.last_stamp:
            return "duplicates"
        if stamp < self.last_stamp:
            return "regressions"
        self.last_stamp = stamp
        number = timing.frame_number if timing is not None else None
        if number is not None and self.last_number is not None:
            self.missing += max(0, number - self.last_number - 1)
            self.resets += int(number < self.last_number)
        self.last_number = number
        if self.origin is None:
            self.origin = stamp
        # Nearest grid cell. Integer arithmetic carries phase across all frames.
        tick = ((stamp - self.origin) * 2 * self.numerator + self.denominator) // (
            2 * self.denominator
        )
        if tick <= self.last_tick:
            return "rate_skipped"
        self.last_tick = tick
        return "accept"
