"""Camera configuration parsing utilities for RealSense capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CameraSpec:
    """Configured camera name and RealSense serial, with optional exposure controls.

    ``exposure_us`` and/or ``gain`` opt a camera out of auto-exposure and pin
    manual color-sensor controls instead. Both default to ``None`` (keep the
    global auto-exposure behavior). Use this when cameras face permanently
    different lighting (e.g. one aimed at a dim scene) so each can be brightened
    independently without over-exposing the others.
    """

    name: str
    serial: str
    exposure_us: int | None = None
    gain: float | None = None

    @property
    def uses_manual_exposure(self) -> bool:
        """True when this camera pins manual exposure/gain (auto-exposure off)."""
        return self.exposure_us is not None or self.gain is not None


def _parse_camera_option(raw_item: str, token: str) -> tuple[str, float]:
    """Parse one ``key=value`` option token into a (key, float value) pair."""
    if "=" not in token:
        raise ValueError(
            f"Invalid --camera value {raw_item!r}. Option {token!r} must be 'key=value' "
            "(supported: exposure=<microseconds>, gain=<value>)."
        )
    key, _, value = token.partition("=")
    key = key.strip().lower()
    value = value.strip()
    if key not in ("exposure", "gain"):
        raise ValueError(
            f"Invalid --camera value {raw_item!r}. Unknown option {key!r} "
            "(supported: exposure, gain)."
        )
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --camera value {raw_item!r}. Option {key!r} needs a number, got {value!r}."
        ) from exc
    if number < 0:
        raise ValueError(
            f"Invalid --camera value {raw_item!r}. Option {key!r} must be non-negative."
        )
    return key, number


def parse_camera_specs(raw_specs: Iterable[str] | None) -> list[CameraSpec]:
    """Parse repeatable CLI specs of the form ``name:serial[:exposure=US][:gain=G]``.

    Extra colon-separated ``key=value`` tokens set manual color-sensor controls
    for that camera (``exposure`` in microseconds, ``gain`` unitless); supplying
    either disables auto-exposure for that camera only.

    Raises:
        ValueError: If any item is malformed or names/serials are duplicated.
    """
    if raw_specs is None:
        return []

    parsed: list[CameraSpec] = []
    seen_names: set[str] = set()
    seen_serials: set[str] = set()

    for raw_item in raw_specs:
        item = raw_item.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                f"Invalid --camera value {raw_item!r}. Expected format "
                "'name:serial[:exposure=US][:gain=G]'."
            )

        name_part, serial_part, *option_parts = item.split(":")
        name = name_part.strip()
        serial = serial_part.strip()

        if not name or not serial:
            raise ValueError(
                f"Invalid --camera value {raw_item!r}. Expected non-empty name and serial."
            )
        if "/" in name:
            raise ValueError(
                f"Invalid camera name {name!r}. '/' is not allowed in camera names."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate camera name {name!r} in --camera specs.")
        if serial in seen_serials:
            raise ValueError(f"Duplicate camera serial {serial!r} in --camera specs.")

        exposure_us: int | None = None
        gain: float | None = None
        for token in option_parts:
            token = token.strip()
            if not token:
                continue
            key, number = _parse_camera_option(raw_item, token)
            if key == "exposure":
                exposure_us = int(number)
            else:  # gain
                gain = number

        seen_names.add(name)
        seen_serials.add(serial)
        parsed.append(
            CameraSpec(name=name, serial=serial, exposure_us=exposure_us, gain=gain)
        )

    return parsed

