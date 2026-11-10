"""Camera timestamps and their provenance, independent of the RealSense runtime.

``get_timestamp()`` is milliseconds in the domain reported by the SDK. Only
``global_time`` is a camera-derived timestamp mapped to the host epoch. The
optional SENSOR_TIMESTAMP metadata is the exposure midpoint, in device-clock
microseconds; it must never be mistaken for UTC. Preserve both, even when the
SDK has not established a global-time mapping yet.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from typing import Any

# Fixed-size recorder IPC metadata, written with the pixels under one seqlock.
# The canonical epoch timestamp stays in the existing frame header.
TIMING_STRUCT = struct.Struct("<qqdqqqqq")
_DOMAINS = ("unknown", "hardware_clock", "system_time", "global_time")
_SOURCES = ("host_receipt", "sdk_global_time")


@dataclass(frozen=True)
class CameraFrameTiming:
    """Keep original device values separate from the timestamp used for joins."""

    timestamp_ns: int
    host_received_ts_ns: int
    host_received_monotonic_ns: int
    sdk_timestamp_ms: float | None
    timestamp_domain: str
    frame_number: int | None
    sensor_timestamp_us: int | None
    frame_timestamp_us: int | None
    timestamp_source: str

    def pack(self) -> bytes:
        """Encode the bounded metadata for the recorder's shared-memory slot."""
        return TIMING_STRUCT.pack(
            self.host_received_ts_ns,
            self.host_received_monotonic_ns,
            self.sdk_timestamp_ms if self.sdk_timestamp_ms is not None else math.nan,
            self.frame_number if self.frame_number is not None else -1,
            self.sensor_timestamp_us if self.sensor_timestamp_us is not None else -1,
            self.frame_timestamp_us if self.frame_timestamp_us is not None else -1,
            _DOMAINS.index(self.timestamp_domain),
            _SOURCES.index(self.timestamp_source),
        )

    @classmethod
    def unpack(cls, timestamp_ns: int, payload: bytes) -> CameraFrameTiming | None:
        """Decode a recorder slot; an all-zero slot is a legacy frame producer."""
        host, mono, sdk, number, sensor, frame, domain, source = TIMING_STRUCT.unpack(
            payload
        )
        if not host:
            return None
        return cls(
            timestamp_ns=timestamp_ns,
            host_received_ts_ns=host,
            host_received_monotonic_ns=mono,
            sdk_timestamp_ms=sdk if math.isfinite(sdk) else None,
            timestamp_domain=_DOMAINS[domain],
            frame_number=number if number >= 0 else None,
            sensor_timestamp_us=sensor if sensor >= 0 else None,
            frame_timestamp_us=frame if frame >= 0 else None,
            timestamp_source=_SOURCES[source],
        )

    def as_manifest(self) -> dict[str, Any]:
        """Return JSON-safe metadata paired positionally with one encoded frame."""
        return asdict(self)


def _metadata_us(frame: Any, sdk: Any, name: str) -> int | None:
    option = getattr(getattr(sdk, "frame_metadata_value", None), name, None)
    if option is None:
        return None
    try:
        if frame.supports_frame_metadata(option):
            value = int(frame.get_frame_metadata(option))
            if 0 <= value < 2**63:
                return value
    except Exception:
        pass  # Metadata availability varies by device, firmware and USB driver.
    return None


class CameraTimestampReader:
    """Read one stream's timestamps without ever interpreting device uptime as UTC.

    Preserve the raw values across domain changes and restarts. During global
    time warmup, unsupported metadata, implausible mappings or a backwards SDK
    correction, retain the old host-receipt fallback and label it explicitly.
    This fallback is not an estimate of exposure time.
    """

    def __init__(self, sdk: Any) -> None:
        """Keep timestamp-selection history separate for each camera stream."""
        self._sdk = sdk
        self._last_timestamp_ns: int | None = None
        self._last_frame: CameraFrameTiming | None = None

    def read(
        self, frame: Any, host_received_ts_ns: int, host_received_monotonic_ns: int
    ) -> CameraFrameTiming:
        """Read metadata before any alignment, conversion or image rotation."""
        sdk_ts = None
        domain = "unknown"
        number = None
        try:
            value = float(frame.get_timestamp())
            if math.isfinite(value) and value >= 0:
                sdk_ts = value
        except Exception:
            pass
        # Query AFTER get_timestamp(): some SDK versions establish the mapping
        # while reading the timestamp and only then report global_time.
        try:
            actual = frame.get_frame_timestamp_domain()
            for candidate in _DOMAINS[1:]:
                expected = getattr(self._sdk.timestamp_domain, candidate, None)
                if expected is not None and actual == expected:
                    domain = candidate
                    break
        except Exception:
            pass
        try:
            value = int(frame.get_frame_number())
            if 0 <= value < 2**63:
                number = value
        except Exception:
            pass

        previous = self._last_frame
        if (
            previous is not None
            and number is not None
            and sdk_ts is not None
            and number == previous.frame_number
            and sdk_ts == previous.sdk_timestamp_ms
            and domain == previous.timestamp_domain
        ):
            # RealSense may repeat one stream's image in successive framesets.
            # It is still the same capture, not a backwards clock. Re-stamping
            # it at receipt also pushes the next fresh frame into fallback.
            return previous

        timestamp_ns = host_received_ts_ns
        source = "host_receipt"
        if domain == "global_time" and sdk_ts is not None:
            candidate_ns = round(sdk_ts * 1_000_000)
            age_ns = host_received_ts_ns - candidate_ns
            # Sanity bounds, not an accuracy guarantee. Keep raw SDK values
            # even when its mapping is unusable for an epoch-based consumer.
            if 0 <= age_ns <= 10_000_000_000 and (
                self._last_timestamp_ns is None
                or candidate_ns > self._last_timestamp_ns
            ):
                timestamp_ns = candidate_ns
                source = "sdk_global_time"
        self._last_timestamp_ns = timestamp_ns
        result = CameraFrameTiming(
            timestamp_ns=timestamp_ns,
            host_received_ts_ns=host_received_ts_ns,
            host_received_monotonic_ns=host_received_monotonic_ns,
            sdk_timestamp_ms=sdk_ts,
            timestamp_domain=domain,
            frame_number=number,
            sensor_timestamp_us=_metadata_us(frame, self._sdk, "sensor_timestamp"),
            frame_timestamp_us=_metadata_us(frame, self._sdk, "frame_timestamp"),
            timestamp_source=source,
        )
        self._last_frame = result
        return result


def offer_timed_frame(
    sink: Any, method: str, *args: Any, timing: CameraFrameTiming | None = None
) -> None:
    """Forward provenance to capable sinks, preserving the legacy sink interface."""
    callback = getattr(sink, method)
    if timing is not None and getattr(sink, "supports_frame_timing", False):
        callback(*args, timing=timing)
    else:
        callback(*args)
