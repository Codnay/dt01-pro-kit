"""Camera thread - captures RGB/depth images from one or more RealSense cameras."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyrealsense2 as rs
from arms.camera_config import CameraSpec
from arms.camera_timing import CameraTimestampReader
from arms.configs import (
    CAMERA_AUTO_EXPOSURE_WARMUP_FRAMES,
    CAMERA_ENABLE_AUTO_EXPOSURE,
    CAMERA_ENABLE_AUTO_EXPOSURE_PRIORITY,
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_REALSENSE_VIEWER_JSON_PRESET_PATH,
    DEPTH_PROCESS_RATE,
)
from arms.data_manager import DataManager
from arms.logging_setup import console_logger

# cspell:words memcpys SIGABRT asic sysfs lsusb ASIC frameset


# Camera logs are pinned to the console unconditionally (all levels, no env/flags
# required) so multi-camera capture/reconnect events are always visible and can
# never be muted or raised by the global PIPER_LOG_* config.
log = console_logger("camera")

try:  # OpenCV gives a SIMD rotate; NumPy fallback keeps this import optional.
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]

def inc_camera_reconnect(*_a: Any, **_k: Any) -> None:
    return None


def init_camera_metrics(*_a: Any, **_k: Any) -> None:
    return None


def record_camera_frame(*_a: Any, **_k: Any) -> None:
    return None


def stop_camera_metrics(*_a: Any, **_k: Any) -> None:
    return None


def _rotate_90_cw(image: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise, always returning a fresh C-contiguous array.

    Equivalent to the ``np.rot90(image, k=3)`` this replaces, but ``np.rot90``
    returns a transposed *view*: every downstream consumer then pays a strided,
    cache-hostile copy that runs several times slower than a memcpy, once per
    frame per camera. Producing a contiguous array here makes those copies plain
    memcpys -- and lets the caller hand the result to ``DataManager`` with
    ``copy=False``, since this array is freshly allocated and never aliases the
    RealSense frame buffer (which the SDK recycles).
    """
    if cv2 is not None:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return np.ascontiguousarray(np.rot90(image, k=3))


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


# Every librealsense pipeline start/stop and device enumeration in this process
# goes through this lock. Capture runs one thread per camera, so without it a
# reconnect on one camera tears a pipeline down and re-enumerates the USB bus on
# a fresh context *while the other cameras are streaming on the same context* —
# which corrupts the heap ("corrupted double-linked list" -> SIGABRT) and kills
# every feed, not just the disturbed camera. Streaming (`wait_for_frames`) is NOT
# serialized; only the device-level operations are.
_DEVICE_OP_LOCK = threading.RLock()

# A RealSense that physically left the bus does not reliably come back in the
# same process, and retrying forever is what turns one loose cable into a crash.
# Past this many restarts the camera is left offline and its thread exits; the
# other cameras keep streaming to Agora and to the GCS recording.
CAMERA_MAX_RESTARTS = _env_int("PIPER_CAMERA_MAX_RESTARTS", 3)
# How long to wait for an unplugged camera's serial to reappear on the bus
# before giving up on it (0 disables the wait: give up as soon as it vanishes).
CAMERA_REATTACH_WAIT_S = _env_int("PIPER_CAMERA_REATTACH_WAIT_S", 60)


@dataclass
class _PipelineHandle:
    """Live capture state for one camera.

    Mutable by design: on a pipeline restart the capture loop swaps
    ``pipeline``/``align_to_color``/``depth_scale_m``/``color_fps``/
    ``depth_enabled`` in place (see the reconnect branch in
    ``_capture_camera_loop``) so the same handle keeps flowing to its owner.
    """

    spec: CameraSpec
    pipeline: rs.pipeline
    align_to_color: rs.align | None
    depth_scale_m: float
    color_fps: int
    depth_fps: int
    depth_enabled: bool
    # False once the capture loop has stopped this pipeline itself (unplug /
    # restart), so the thread's teardown does not stop an already-stopped
    # pipeline and log a spurious error for a camera it already gave up on.
    streaming: bool = True


def list_connected_camera_serials() -> list[str]:
    """Return serial numbers of currently connected RealSense devices.

    Enumeration builds a fresh librealsense context, so it is serialized with
    every other device-level operation (see ``_DEVICE_OP_LOCK``) — doing it
    concurrently with a pipeline start/stop on another camera thread is a heap
    corruption, not a race you get an exception for.
    """
    try:
        with _DEVICE_OP_LOCK:
            context = rs.context()
            serials: list[str] = []
            devices = context.query_devices()
            for index in range(len(devices)):
                try:
                    device = devices[index]
                    serials.append(device.get_info(rs.camera_info.serial_number))
                except Exception:
                    continue
        return sorted(set(serials))
    except Exception:
        return []


def describe_connected_cameras() -> dict[str, dict[str, str]]:
    """Map librealsense serial -> ``{asic_serial, model}`` for each connected device.

    The USB descriptor (and therefore sysfs, lsusb, and node_exporter) carries a
    RealSense's *ASIC* serial, not the serial passed to ``--camera``. This is the
    only place both identities are visible at once, so it is what lets the USB
    metrics be labelled with the serial the operator actually runs with.

    Best-effort: a device that refuses to answer is skipped, never raised.
    """
    out: dict[str, dict[str, str]] = {}
    # Serialized with pipeline start/stop for the same reason enumeration is in
    # list_connected_camera_serials() — including the per-device reads, which
    # also go through the context.
    with _DEVICE_OP_LOCK:
        try:
            devices = rs.context().query_devices()
        except Exception:
            return out
        for index in range(len(devices)):
            try:
                # Device construction can fail for a single camera (including
                # a DFU device without raw USB access). Keep siblings usable.
                device = devices[index]
            except Exception:
                continue

            def info(key: Any, dev: Any = device) -> str:
                try:
                    return str(dev.get_info(key)) if dev.supports(key) else ""
                except Exception:
                    return ""

            serial = info(rs.camera_info.serial_number)
            if not serial:
                continue
            out[serial] = {
                "asic_serial": info(rs.camera_info.asic_serial_number),
                "model": info(rs.camera_info.name),
            }
    return out


def _get_depth_scale_m(profile: rs.pipeline_profile) -> float:
    try:
        device = profile.get_device()
        for sensor in device.query_sensors():
            if sensor.is_depth_sensor():
                return float(sensor.as_depth_sensor().get_depth_scale())
    except Exception:
        pass
    # Typical RealSense default (meters per unit).
    return 0.001


def _configure_camera_global_time(
    profile: rs.pipeline_profile, spec: CameraSpec
) -> None:
    """Request SDK device-to-host time mapping on every supported sensor."""
    option = getattr(rs.option, "global_time_enabled", None)
    configured = False
    try:
        for sensor in profile.get_device().query_sensors():
            configured |= _set_sensor_option_if_supported(
                sensor=sensor,
                option=option,
                value=1.0,
                camera_name=spec.name,
                sensor_name=str(sensor),
                option_label="global_time_enabled",
            )
    except Exception as exc:
        log.warning("Camera '%s' global-time setup failed: %s", spec.name, exc)
    if not configured:
        log.warning(
            "Camera '%s' has no confirmed global-time mapping option; original "
            "timestamps will still be saved with their clock domain. Epoch time "
            "falls back to host receipt when the frame has no usable mapping.",
            spec.name,
        )


def _set_sensor_option_if_supported(
    *,
    sensor: rs.sensor,
    option: rs.option | None,
    value: float,
    camera_name: str,
    sensor_name: str,
    option_label: str,
) -> bool:
    """Set a RealSense option when present/supported; return whether it was applied."""
    if option is None:
        return False
    try:
        if not sensor.supports(option):
            return False
        sensor.set_option(option, float(value))
        configured_value = sensor.get_option(option)
        log.info(
            "Camera '%s' sensor '%s' %s=%.0f",
            camera_name,
            sensor_name,
            option_label,
            configured_value,
        )
        return True
    except Exception as exc:
        log.warning(
            "Camera '%s' sensor '%s' failed setting %s: %s",
            camera_name,
            sensor_name,
            option_label,
            exc,
        )
        return False


def _read_sensor_option_if_supported(
    *,
    sensor: rs.sensor,
    option: rs.option | None,
) -> float | None:
    """Read a RealSense option if supported; returns None when unavailable."""
    if option is None:
        return None
    try:
        if not sensor.supports(option):
            return None
        return float(sensor.get_option(option))
    except Exception:
        return None


def _sensor_name(sensor: rs.sensor) -> str:
    try:
        return sensor.get_info(rs.camera_info.name) or ""
    except Exception:
        return ""


def _sensor_has_color_stream(sensor: rs.sensor) -> bool:
    try:
        for stream_profile in sensor.get_stream_profiles():
            try:
                if stream_profile.stream_type() == rs.stream.color:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _sensor_supports_option(sensor: rs.sensor, option: rs.option | None) -> bool:
    if option is None:
        return False
    try:
        return bool(sensor.supports(option))
    except Exception:
        return False


def _find_color_sensor(profile: rs.pipeline_profile) -> rs.sensor | None:
    """Return the RGB/color sensor of the device, or None if not present."""
    # Most robust path: use the sensor that owns the active color stream.
    # Some devices/sensors may not classify as "color_sensor"but still provide
    # a color stream with exposure controls.
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        sensor_from_stream = color_stream.get_sensor()
        if sensor_from_stream:
            return sensor_from_stream
    except Exception:
        pass

    try:
        sensors = list(profile.get_device().query_sensors())
    except Exception:
        return None

    # Some RealSense devices expose a color stream from a sensor that does not
    # report true for is_color_sensor(). Select by stream ownership first.
    for sensor in sensors:
        if _sensor_has_color_stream(sensor):
            return sensor

    for sensor in sensors:
        try:
            if sensor.is_color_sensor():
                return sensor
        except Exception:
            pass

        name = _sensor_name(sensor)
        if "RGB" in name or "Color" in name:
            return sensor

    opt_auto_exposure = getattr(rs.option, "enable_auto_exposure", None)
    opt_exposure = getattr(rs.option, "exposure", None)
    opt_gain = getattr(rs.option, "gain", None)
    for sensor in sensors:
        if (
            _sensor_supports_option(sensor, opt_auto_exposure)
            and _sensor_supports_option(sensor, opt_exposure)
            and _sensor_supports_option(sensor, opt_gain)
        ):
            return sensor
    return None


def _configure_manual_exposure(
    *,
    color_sensor: rs.sensor,
    spec: CameraSpec,
    sensor_name: str,
    opt_auto_exposure: Any,
    opt_exposure: Any,
    opt_gain: Any,
) -> None:
    """Pin manual exposure/gain (auto-exposure off) for a camera that requests it.

    Used for cameras aimed at permanently dim scenes where global auto-exposure
    has no headroom to brighten (exposure already at the frame-rate ceiling, gain
    at its floor). Setting exposure and/or gain per camera brightens just that
    stream without over-exposing the others.
    """
    ae_off = _set_sensor_option_if_supported(
        sensor=color_sensor,
        option=opt_auto_exposure,
        value=0.0,
        camera_name=spec.name,
        sensor_name=sensor_name,
        option_label="enable_auto_exposure",
    )
    if opt_auto_exposure is not None and not ae_off:
        log.warning(
            "Camera '%s' sensor '%s' would not disable auto-exposure; "
            "manual exposure/gain may be ignored.",
            spec.name,
            sensor_name,
        )

    if spec.exposure_us is not None:
        _set_sensor_option_if_supported(
            sensor=color_sensor,
            option=opt_exposure,
            value=float(spec.exposure_us),
            camera_name=spec.name,
            sensor_name=sensor_name,
            option_label="exposure",
        )
    if spec.gain is not None:
        _set_sensor_option_if_supported(
            sensor=color_sensor,
            option=opt_gain,
            value=float(spec.gain),
            camera_name=spec.name,
            sensor_name=sensor_name,
            option_label="gain",
        )

    exposure = _read_sensor_option_if_supported(
        sensor=color_sensor, option=opt_exposure
    )
    gain = _read_sensor_option_if_supported(sensor=color_sensor, option=opt_gain)
    log.info(
        "Camera '%s' sensor '%s' MANUAL exposure: auto_exposure=off, "
        "exposure=%s (requested %s), gain=%s (requested %s)",
        spec.name,
        sensor_name,
        exposure,
        spec.exposure_us,
        gain,
        spec.gain,
    )


def _configure_camera_auto_controls(
    profile: rs.pipeline_profile, spec: CameraSpec
) -> None:
    """Replicate the realsense-viewer 'Enable Auto Exposure' toggle on the color sensor.

    Cameras that request manual exposure/gain (``spec.uses_manual_exposure``) are
    pinned to those values with auto-exposure off instead.
    """
    if not spec.uses_manual_exposure and not CAMERA_ENABLE_AUTO_EXPOSURE:
        return

    color_sensor = _find_color_sensor(profile)
    if color_sensor is None:
        log.info(
            "Camera '%s' has no color sensor; skipping exposure config.", spec.name
        )
        return

    try:
        sensor_name = color_sensor.get_info(rs.camera_info.name)
    except Exception:
        sensor_name = "color"

    opt_auto_exposure = getattr(rs.option, "enable_auto_exposure", None)
    opt_auto_exposure_priority = getattr(rs.option, "auto_exposure_priority", None)
    opt_exposure = getattr(rs.option, "exposure", None)
    opt_gain = getattr(rs.option, "gain", None)

    if spec.uses_manual_exposure:
        _configure_manual_exposure(
            color_sensor=color_sensor,
            spec=spec,
            sensor_name=sensor_name,
            opt_auto_exposure=opt_auto_exposure,
            opt_exposure=opt_exposure,
            opt_gain=opt_gain,
        )
        return

    ae_applied = _set_sensor_option_if_supported(
        sensor=color_sensor,
        option=opt_auto_exposure,
        value=1.0,
        camera_name=spec.name,
        sensor_name=sensor_name,
        option_label="enable_auto_exposure",
    )
    if opt_auto_exposure is not None and not ae_applied:
        log.warning(
            "Camera '%s' sensor '%s' does not support or rejected "
            "enable_auto_exposure.",
            spec.name,
            sensor_name,
        )

    if CAMERA_ENABLE_AUTO_EXPOSURE_PRIORITY:
        ae_priority_applied = _set_sensor_option_if_supported(
            sensor=color_sensor,
            option=opt_auto_exposure_priority,
            value=1.0,
            camera_name=spec.name,
            sensor_name=sensor_name,
            option_label="auto_exposure_priority",
        )
        if opt_auto_exposure_priority is not None and not ae_priority_applied:
            log.warning(
                "Camera '%s' sensor '%s' does not support or rejected "
                "auto_exposure_priority.",
                spec.name,
                sensor_name,
            )

    # Log the active auto-exposure-related controls for hardware verification.
    ae_enabled = _read_sensor_option_if_supported(
        sensor=color_sensor, option=opt_auto_exposure
    )
    ae_priority = _read_sensor_option_if_supported(
        sensor=color_sensor, option=opt_auto_exposure_priority
    )
    exposure = _read_sensor_option_if_supported(
        sensor=color_sensor, option=opt_exposure
    )
    gain = _read_sensor_option_if_supported(sensor=color_sensor, option=opt_gain)

    log.info(
        "Camera '%s' sensor '%s' AE status: enable_auto_exposure=%s, "
        "auto_exposure_priority=%s, exposure=%s, gain=%s",
        spec.name,
        sensor_name,
        ae_enabled,
        ae_priority,
        exposure,
        gain,
    )


def _read_color_ae_metadata(
    color_frame: rs.video_frame | None,
) -> tuple[float | None, float | None, float | None]:
    """Read key AE frame metadata as (auto_mode, actual_exposure_us, exposure_priority)."""
    if color_frame is None:
        return (None, None, None)
    try:
        md = rs.frame_metadata_value
        auto_mode = (
            float(color_frame.get_frame_metadata(md.auto_exposure))
            if color_frame.supports_frame_metadata(md.auto_exposure)
            else None
        )
        actual_exposure = (
            float(color_frame.get_frame_metadata(md.actual_exposure))
            if color_frame.supports_frame_metadata(md.actual_exposure)
            else None
        )
        exposure_priority = (
            float(color_frame.get_frame_metadata(md.exposure_priority))
            if color_frame.supports_frame_metadata(md.exposure_priority)
            else None
        )
        return (auto_mode, actual_exposure, exposure_priority)
    except Exception:
        return (None, None, None)


def _frame_metadata(frame: Any, name: str) -> float | None:
    """Safely read one RealSense frame-metadata value, or None if unsupported."""
    try:
        md = rs.frame_metadata_value
        key = getattr(md, name, None)
        if key is None or not frame.supports_frame_metadata(key):
            return None
        return float(frame.get_frame_metadata(key))
    except Exception:
        return None


def _depth_valid_pct(depth_raw: np.ndarray, already: float | None) -> float | None:
    """Percentage of depth pixels with a reading, computed at most once a window.

    Returns *already* untouched when this window has been sampled, so the scan
    runs at the ~1 Hz telemetry rate rather than per depth frame. Previously the
    caller kept ``depth_raw.copy()`` on every frame so this could be measured
    later -- ~614 KB per frame per camera (~37 MB/s at 15 fps x 4 cameras) of
    which 59 in 60 copies were overwritten before anything read them.

    Measured on the raw sensor frame: scaling to millimetres and rotating both
    preserve which pixels are zero, so the result is identical either way.
    """
    if already is not None:
        return already
    if depth_raw is None or depth_raw.size == 0:
        return None
    return float((depth_raw > 0).mean() * 100.0)


def _collect_camera_telemetry(
    color_frame: Any,
    depth_valid_pct: float | None,
    frames_in_interval: int,
    interval_s: float,
    prev_frame_counter: int | None,
) -> tuple[dict[str, float], int | None]:
    """Build camera-health signals from frame metadata + computed stats.

    Returns (signals, new_frame_counter). Signal keys become BigQuery signals
    prefixed ``camera_`` (e.g. ``camera_fps``, ``camera_drops``). Each value is
    a single number. Only signals the device actually supports are included.

    ``depth_valid_pct`` arrives pre-computed: the caller measures it straight
    off the live depth frame at the telemetry rate, rather than retaining a
    frame for this function to measure later.
    """
    signals: dict[str, float] = {}
    if interval_s > 0:
        # Measured throughput we actually stored -- drops below target FPS flag
        # USB/bandwidth problems.
        signals["fps"] = frames_in_interval / interval_s

    new_counter = prev_frame_counter
    if color_frame is not None:
        exposure = _frame_metadata(color_frame, "actual_exposure")
        if exposure is not None:
            signals["exposure_us"] = exposure
        gain = _frame_metadata(color_frame, "gain_level")
        if gain is not None:
            signals["gain"] = gain
        temperature = _frame_metadata(color_frame, "temperature")
        if temperature is not None:
            signals["temperature_c"] = temperature
        counter = _frame_metadata(color_frame, "frame_counter")
        if counter is not None:
            c = int(counter)
            if prev_frame_counter is not None and c >= prev_frame_counter:
                advanced = c - prev_frame_counter
                # Frames the sensor produced but we did not store = drops.
                signals["drops"] = float(max(0, advanced - frames_in_interval))
            new_counter = c

    if depth_valid_pct is not None:
        signals["depth_valid_pct"] = depth_valid_pct

    return signals, new_counter


def _read_sensor_option(sensor: Any, opt_name: str) -> float | None:
    """Read one RealSense sensor option by name. Guarded -> None if unsupported."""
    try:
        opt = getattr(rs.option, opt_name, None)
        if opt is not None and sensor.supports(opt):
            return float(sensor.get_option(opt))
    except Exception:
        return None
    return None


def _read_camera_temperature(pipeline: Any) -> float | None:
    """Read a RealSense sensor temperature in C via sensor options.

    Color-frame metadata does not carry temperature on most RealSense devices,
    so read the sensor's ASIC (and projector, as a fallback) temperature option
    instead. Fully guarded: returns None when unsupported. Cheap at ~1 Hz.
    """
    try:
        device = pipeline.get_active_profile().get_device()
    except Exception:
        return None
    for sensor in device.query_sensors():
        for opt_name in ("asic_temperature", "projector_temperature"):
            val = _read_sensor_option(sensor, opt_name)
            if val is not None and val > 0.0:
                return val
    return None


def _log_active_color_stream_profile(
    profile: rs.pipeline_profile, spec: CameraSpec
) -> None:
    """Log active color stream profile and device default color profile (viewer often uses defaults)."""
    try:
        active = profile.get_stream(rs.stream.color).as_video_stream_profile()
        log.info(
            "Camera '%s' active color stream: %dx%d@%d format=%s",
            spec.name,
            active.width(),
            active.height(),
            active.fps(),
            active.format(),
        )
    except Exception:
        pass

    # Log device-declared default color profile to highlight potential viewer/app mismatch.
    try:
        color_sensor = _find_color_sensor(profile)
        if color_sensor is None:
            return
        default_profiles: list[rs.video_stream_profile] = []
        for p in color_sensor.get_stream_profiles():
            try:
                vp = p.as_video_stream_profile()
                if vp and p.stream_type() == rs.stream.color and p.is_default():
                    default_profiles.append(vp)
            except Exception:
                continue
        if default_profiles:
            vp = default_profiles[0]
            log.info(
                "Camera '%s' device default color profile: %dx%d@%d format=%s",
                spec.name,
                vp.width(),
                vp.height(),
                vp.fps(),
                vp.format(),
            )
    except Exception:
        pass


def _apply_viewer_json_preset(profile: rs.pipeline_profile, spec: CameraSpec) -> None:
    """Load a viewer-exported JSON preset if configured."""
    preset_path = CAMERA_REALSENSE_VIEWER_JSON_PRESET_PATH
    if not preset_path:
        return
    path = os.path.expanduser(str(preset_path))
    if not os.path.isfile(path):
        log.warning(
            "Camera '%s' JSON preset path not found: %s. "
            "Skipping viewer preset load.",
            spec.name,
            path,
        )
        return

    try:
        with open(path, encoding="utf-8") as f:
            payload = f.read()
    except Exception as exc:
        log.warning("Camera '%s' failed reading preset %s: %s", spec.name, path, exc)
        return

    try:
        device = profile.get_device()
        as_serializable = getattr(device, "as_serializable_device", None)
        if as_serializable is None:
            log.warning(
                "Camera '%s' SDK has no serializable_device API; "
                "cannot load viewer preset.",
                spec.name,
            )
            return
        serializable = as_serializable()
        serializable.load_json(payload)
        log.info("Camera '%s' loaded viewer JSON preset: %s", spec.name, path)
    except Exception as exc:
        log.warning(
            "Camera '%s' failed loading viewer preset %s: %s", spec.name, path, exc
        )


def _warmup_auto_exposure(
    *,
    pipeline: rs.pipeline,
    spec: CameraSpec,
    num_frames: int,
) -> None:
    """Drop initial frames to let auto-exposure converge after stream start."""
    if num_frames <= 0:
        return

    log.info(
        "Camera '%s' warming up auto-exposure for %d frame(s)...",
        spec.name,
        num_frames,
    )

    warmed = 0
    last_color_frame: rs.video_frame | None = None
    for _ in range(num_frames):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if color_frame and color_frame.get_data_size() > 0:
                last_color_frame = color_frame
            warmed += 1
        except Exception as exc:
            log.warning("Camera '%s' AE warmup interrupted: %s", spec.name, exc)
            break

    auto_mode, actual_exposure, exposure_priority = _read_color_ae_metadata(
        last_color_frame
    )
    log.info(
        "Camera '%s' AE metadata after warmup: auto_exposure=%s, "
        "actual_exposure_us=%s, exposure_priority=%s",
        spec.name,
        auto_mode,
        actual_exposure,
        exposure_priority,
    )

    log.info(
        "Camera '%s' AE warmup complete (%d/%d frame(s))",
        spec.name,
        warmed,
        num_frames,
    )
    return


def _resolve_cameras(cameras: list[CameraSpec] | None) -> list[CameraSpec]:
    if cameras:
        return cameras
    return [CameraSpec(name="rgb", serial="")]


def _start_pipeline_with_fallbacks(
    spec: CameraSpec,
    requested_enable_depth: bool,
    requested_fps: int,
    requested_depth_fps: int | None = None,
) -> _PipelineHandle | None:
    """Start a camera's pipeline, serialized against every other device op.

    Thin lock wrapper around :func:`_start_pipeline_locked`; see
    ``_DEVICE_OP_LOCK`` for why starting a pipeline while another camera thread
    is starting/stopping one is not survivable.
    """
    with _DEVICE_OP_LOCK:
        return _start_pipeline_locked(
            spec=spec,
            requested_enable_depth=requested_enable_depth,
            requested_fps=requested_fps,
            requested_depth_fps=requested_depth_fps,
        )


def _stop_pipeline_quietly(handle: _PipelineHandle) -> None:
    """Stop a camera's pipeline under the device lock, swallowing any error.

    Used on the unplug/restart path, where the device may already be gone and
    ``stop()`` raising is expected rather than interesting.
    """
    handle.streaming = False
    stop_camera_metrics(handle.spec.name)
    try:
        with _DEVICE_OP_LOCK:
            handle.pipeline.stop()
    except Exception as exc:  # noqa: BLE001 - teardown must never raise
        log.debug("Pipeline stop failed [%s]: %s", handle.spec.name, exc)


def _start_pipeline_locked(
    spec: CameraSpec,
    requested_enable_depth: bool,
    requested_fps: int,
    requested_depth_fps: int | None = None,
) -> _PipelineHandle | None:
    """Start a RealSense pipeline with progressive fallbacks.

    Callers must hold ``_DEVICE_OP_LOCK`` (use ``_start_pipeline_with_fallbacks``).

    Fall back order prioritizes keeping RGB alive:
    1) requested FPS with requested depth mode
    2) 30 FPS with requested depth mode (if different)
    3) requested FPS RGB-only (if depth requested)
    4) 30 FPS RGB-only (if depth requested and different)
    """
    requested_depth_fps = int(requested_depth_fps or requested_fps)
    attempts: list[tuple[int, bool, int]] = []

    def _append_attempt(fps: int, depth_on: bool, depth_fps: int | None = None) -> None:
        pair = (int(fps), bool(depth_on), int(depth_fps or fps))
        if pair not in attempts:
            attempts.append(pair)

    _append_attempt(requested_fps, requested_enable_depth, requested_depth_fps)
    if requested_enable_depth and requested_depth_fps != requested_fps:
        _append_attempt(requested_fps, requested_enable_depth, requested_fps)
    _append_attempt(30, requested_enable_depth, min(requested_depth_fps, 30))
    if requested_enable_depth and requested_depth_fps != 30:
        _append_attempt(30, requested_enable_depth, 30)
    if requested_enable_depth:
        _append_attempt(requested_fps, False)
        _append_attempt(30, False)

    serial_label = spec.serial or "auto"
    last_error: Exception | None = None

    for fps, depth_on, depth_fps in attempts:
        pipeline = rs.pipeline()
        config = rs.config()
        if spec.serial:
            config.enable_device(spec.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)
        if depth_on:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, depth_fps)

        try:
            profile = pipeline.start(config)
            _log_active_color_stream_profile(profile, spec)
            _apply_viewer_json_preset(profile, spec)
            _configure_camera_auto_controls(profile, spec)
            _configure_camera_global_time(profile, spec)
            if spec.uses_manual_exposure:
                # Drop a few frames so the manual exposure/gain settles; the
                # AE-recovery path below is meaningless when AE is intentionally off.
                _warmup_auto_exposure(pipeline=pipeline, spec=spec, num_frames=10)
            elif CAMERA_ENABLE_AUTO_EXPOSURE:
                _warmup_auto_exposure(
                    pipeline=pipeline,
                    spec=spec,
                    num_frames=int(CAMERA_AUTO_EXPOSURE_WARMUP_FRAMES),
                )
                # Some firmware paths can reset controls at stream start; retry once if AE stayed off.
                color_sensor = _find_color_sensor(profile)
                opt_auto_exposure = getattr(rs.option, "enable_auto_exposure", None)
                ae_value = (
                    _read_sensor_option_if_supported(
                        sensor=color_sensor, option=opt_auto_exposure
                    )
                    if color_sensor is not None
                    else None
                )
                if ae_value is not None and ae_value < 0.5:
                    log.warning(
                        "Camera '%s' reports AE disabled after warmup; "
                        "reapplying auto controls once.",
                        spec.name,
                    )
                    _configure_camera_auto_controls(profile, spec)
                    _warmup_auto_exposure(pipeline=pipeline, spec=spec, num_frames=10)
            align_to_color = rs.align(rs.stream.color) if depth_on else None
            depth_scale_m = _get_depth_scale_m(profile)
            log.info(
                "Camera '%s' started (serial=%s, fps=%s, depth=%s)",
                spec.name,
                serial_label,
                fps,
                f"{depth_fps}fps" if depth_on else False,
            )
            return _PipelineHandle(
                spec=spec,
                pipeline=pipeline,
                align_to_color=align_to_color,
                depth_scale_m=depth_scale_m,
                color_fps=fps,
                depth_fps=depth_fps,
                depth_enabled=depth_on,
            )
        except Exception as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            if depth_on and depth_fps != fps:
                log.info(
                    "Camera '%s' does not support color@%s + depth@%s "
                    "(serial=%s): %s; trying fallback.",
                    spec.name,
                    fps,
                    depth_fps,
                    serial_label,
                    exc,
                )
            else:
                log.warning(
                    "Camera start failed [%s] (serial=%s, fps=%s, depth=%s): %s",
                    spec.name,
                    serial_label,
                    fps,
                    f"{depth_fps}fps" if depth_on else False,
                    exc,
                )

    log.error(
        "Could not start camera '%s' (serial=%s) after fallback attempts. "
        "Last error: %s",
        spec.name,
        serial_label,
        last_error,
    )
    return None


def _wait_for_serial(
    *,
    serial: str,
    cam_name: str,
    data_manager: DataManager,
    stop_event: threading.Event | None,
    timeout_s: float | None = None,
    poll_s: float = 5.0,
) -> bool:
    """Poll (slowly) for an unplugged camera's serial to reappear on the bus.

    Returns True if it came back within the timeout, False on timeout/shutdown.
    The poll is deliberately slow: each check re-enumerates the USB bus, and the
    caller's camera has nothing to stream in the meantime, so the cheap thing is
    to leave the other cameras' capture threads alone.
    """
    limit = CAMERA_REATTACH_WAIT_S if timeout_s is None else timeout_s
    if limit <= 0:
        return False
    log.warning(
        "Camera '%s' (serial=%s) dropped off the USB bus — waiting up to %gs "
        "for it to come back. Other cameras keep streaming.",
        cam_name,
        serial,
        limit,
    )
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if data_manager.is_shutdown_requested() or (
            stop_event is not None and stop_event.is_set()
        ):
            return False
        time.sleep(poll_s)
        if serial in list_connected_camera_serials():
            log.info("Camera '%s' (serial=%s) is back on the bus", cam_name, serial)
            return True
    return False


def _capture_camera_loop(
    *,
    handle: _PipelineHandle,
    data_manager: DataManager,
    depth_process_period_s: float,
    diagnostics_enabled: bool,
    stop_event: threading.Event | None,
) -> None:
    """Capture RGB(+depth) for ONE camera until shutdown, in its own thread.

    One of these runs per camera. A single shared loop used to service every
    RealSense pipeline sequentially with a blocking ``wait_for_frames``, so each
    camera was head-of-line blocked behind the others: N cameras ran well below
    their FPS (very laggy feeds) even while the CPU sat almost idle (the loop
    spent its time blocked, not computing). Giving each pipeline its own thread
    lets it be drained the instant a frame is ready, so ``DataManager`` always
    holds the freshest frame. ``wait_for_frames`` itself paces this loop at the
    sensor's FPS, so there is no extra sleep (which would only re-introduce
    staleness).
    """
    cam_name = handle.spec.name
    from arms.capture_rate import CaptureRateGate

    rgb_clock = CameraTimestampReader(rs)
    depth_clock = CameraTimestampReader(rs)
    depth_gate = CaptureRateGate(
        1.0 / depth_process_period_s if depth_process_period_s > 0 else handle.depth_fps
    )
    last_counted_rgb: int | None = None
    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    consecutive_failures = 0
    restarts = 0
    frames_logged = 0
    last_log_ts = 0.0
    # Camera-telemetry state (~1 Hz poll -> BigQuery via the sink).
    cam_tel_frames = 0
    cam_tel_last_ts = time.time()
    cam_tel_prev_counter: int | None = None
    cam_tel_depth_pct: float | None = None
    diag_last_ts = time.time()
    diag_loops = 0
    diag_overruns = 0
    diag_max_loop_ms = 0.0
    diag_rgb_ms = 0.0
    diag_depth_ms = 0.0
    diag_rgb_count = 0
    diag_depth_count = 0

    try:
        while not data_manager.is_shutdown_requested() and not (
            stop_event is not None and stop_event.is_set()
        ):
            iteration_start = time.time()
            # Bind before the capture attempt so the telemetry block below always
            # has a defined value even if a future edit adds an early path out of
            # the frame grab (the reconnect branch already `continue`s).
            color_frame = None
            color_ok = False

            try:
                frames = handle.pipeline.wait_for_frames(timeout_ms=5000)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                n = consecutive_failures
                log.warning(
                    "RealSense wait_for_frames error [%s] (#%d): %s", cam_name, n, exc
                )
                # Two timeouts in a row (~10 s) on a camera whose serial is no
                # longer enumerated means the cable/device physically went away.
                # Restarting a pipeline against a device that left the bus is the
                # path that used to abort the whole process, so instead: release
                # it, wait for it to come back, and only then restart.
                if n == 2 and handle.spec.serial:
                    if handle.spec.serial not in list_connected_camera_serials():
                        _stop_pipeline_quietly(handle)
                        if not _wait_for_serial(
                            serial=handle.spec.serial,
                            cam_name=cam_name,
                            data_manager=data_manager,
                            stop_event=stop_event,
                        ):
                            log.error(
                                "Camera '%s' (serial=%s) is off the USB bus — "
                                "leaving it offline. The other cameras keep "
                                "streaming (Agora + GCS); re-plug it and restart "
                                "the recorder to get this feed back.",
                                cam_name,
                                handle.spec.serial,
                            )
                            return
                        # Device is back: fall through to the restart below.
                        n = consecutive_failures = 5
                if n >= 5:
                    if restarts >= CAMERA_MAX_RESTARTS:
                        log.error(
                            "Camera '%s' failed %d restart(s) — leaving it "
                            "offline instead of retrying forever. The other "
                            "cameras keep streaming (Agora + GCS); restart the "
                            "recorder to recover this one.",
                            cam_name,
                            restarts,
                        )
                        _stop_pipeline_quietly(handle)
                        return
                    restarts += 1
                    log.warning(
                        "Restarting RealSense pipeline [%s] (%d/%d)...",
                        cam_name,
                        restarts,
                        CAMERA_MAX_RESTARTS,
                    )
                    _stop_pipeline_quietly(handle)
                    try:
                        restarted = _start_pipeline_with_fallbacks(
                            spec=handle.spec,
                            requested_enable_depth=handle.depth_enabled,
                            requested_fps=handle.color_fps,
                            requested_depth_fps=handle.depth_fps,
                        )
                        if restarted is None:
                            raise RuntimeError("pipeline restart fallback failed")
                        handle.pipeline = restarted.pipeline
                        handle.align_to_color = restarted.align_to_color
                        handle.depth_scale_m = restarted.depth_scale_m
                        handle.color_fps = restarted.color_fps
                        handle.depth_enabled = restarted.depth_enabled
                        handle.streaming = True
                        init_camera_metrics(cam_name)
                        consecutive_failures = 0
                        inc_camera_reconnect(cam_name)
                        log.info("Pipeline restarted [%s]", cam_name)
                    except Exception as restart_exc:
                        log.error(
                            "Pipeline restart failed [%s]: %s", cam_name, restart_exc
                        )
                        # The pipeline is stopped, so the next wait_for_frames
                        # raises instantly instead of blocking for its timeout.
                        # Without this pause the loop would burn the whole restart
                        # budget in milliseconds and spin a core doing it.
                        time.sleep(2.0)
                continue

            host_ts_ns = time.time_ns()
            host_mono_ns = time.monotonic_ns()

            # Capture color from the raw frameset BEFORE alignment.
            # align.process() can drop the color frame when depth/color
            # aren't paired in the same frameset, leaving rgb empty.
            color_frame = frames.get_color_frame()
            color_ok = bool(color_frame) and color_frame.get_data_size() > 0
            if color_ok:
                rgb_timing = rgb_clock.read(color_frame, host_ts_ns, host_mono_ns)
                color_image = np.asanyarray(color_frame.get_data())
                # Fresh contiguous array (match existing orientation), so it can
                # be stored by reference instead of re-copied under the GIL.
                color_image = _rotate_90_cw(color_image)
                rgb_set_start = time.perf_counter()
                data_manager.set_camera_rgb(
                    cam_name, color_image, timing=rgb_timing, copy=False
                )
                record_camera_frame(cam_name, rgb_timing)
                if diagnostics_enabled:
                    diag_rgb_ms += (time.perf_counter() - rgb_set_start) * 1000.0
                    diag_rgb_count += 1
                unique = rgb_timing.timestamp_ns != last_counted_rgb
                frames_logged += int(unique)
                cam_tel_frames += int(unique)
                last_counted_rgb = rgb_timing.timestamp_ns

            original_depth = frames.get_depth_frame() if handle.depth_enabled else None
            depth_timing = (
                depth_clock.read(original_depth, host_ts_ns, host_mono_ns)
                if original_depth
                else None
            )

            if (
                handle.depth_enabled
                and data_manager.is_recording_active()
                and depth_timing is not None
                and depth_gate.select(depth_timing.timestamp_ns, depth_timing)
                == "accept"
            ):
                depth_process_start = time.perf_counter()
                # Alignment produces synthetic depth frames. Preserve the
                # original depth sensor's metadata before transforming pixels.
                aligned = frames
                if handle.align_to_color is not None:
                    try:
                        aligned = handle.align_to_color.process(frames)
                    except Exception:
                        aligned = frames
                depth_frame = aligned.get_depth_frame()
                if (
                    depth_timing is not None
                    and depth_frame
                    and depth_frame.get_data_size() > 0
                ):
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    if data_manager.depth_offload_active():
                        # Ship raw aligned depth to the recorder process; the
                        # scale/clip/rotate happens there, off this thread's
                        # GIL, so it cannot jitter the control loops.
                        data_manager.offer_camera_depth_raw(
                            cam_name,
                            depth_raw,
                            handle.depth_scale_m,
                            timing=depth_timing,
                        )
                        cam_tel_depth_pct = _depth_valid_pct(
                            depth_raw, cam_tel_depth_pct
                        )
                    else:
                        depth_mm = np.clip(
                            depth_raw.astype(np.float32)
                            * handle.depth_scale_m
                            * 1000.0,
                            0.0,
                            65535.0,
                        ).astype(np.uint16)
                        depth_mm = _rotate_90_cw(depth_mm)
                        data_manager.set_camera_depth(
                            cam_name, depth_mm, timing=depth_timing, copy=False
                        )
                        cam_tel_depth_pct = _depth_valid_pct(
                            depth_raw, cam_tel_depth_pct
                        )
                if diagnostics_enabled:
                    diag_depth_ms += (
                        time.perf_counter() - depth_process_start
                    ) * 1000.0
                    diag_depth_count += 1

            now = time.time()
            if now - last_log_ts >= 5.0:
                log.debug(
                    "%s frames stored last 5s: %s (color_ok=%s)",
                    cam_name,
                    frames_logged,
                    color_ok,
                )
                frames_logged = 0
                last_log_ts = now

            # Camera telemetry (~1 Hz) -> BigQuery sink. No-op unless a sink/
            # recording is active (log_telemetry_signal gates on the on-change
            # callback; the sink gates on the episode).
            interval_s = now - cam_tel_last_ts
            if interval_s >= 1.0:
                signals, cam_tel_prev_counter = _collect_camera_telemetry(
                    color_frame if color_ok else None,
                    cam_tel_depth_pct,
                    cam_tel_frames,
                    interval_s,
                    cam_tel_prev_counter,
                )
                # Temperature lives in a sensor option, not frame metadata,
                # so most devices omit it above; read it directly here.
                if "temperature_c" not in signals:
                    cam_temp = _read_camera_temperature(handle.pipeline)
                    if cam_temp is not None:
                        signals["temperature_c"] = cam_temp
                cam_hand = f"cam:{cam_name}"
                for sig_name, value in signals.items():
                    data_manager.log_telemetry_signal(
                        f"log_camera_{sig_name}", float(value), hand=cam_hand
                    )
                cam_tel_frames = 0
                cam_tel_last_ts = now
                cam_tel_depth_pct = None

            elapsed = time.time() - iteration_start
            if diagnostics_enabled:
                diag_now = time.time()
                diag_loops += 1
                diag_max_loop_ms = max(diag_max_loop_ms, elapsed * 1000.0)
                if elapsed > dt:
                    diag_overruns += 1
                if diag_now - diag_last_ts >= 5.0:
                    rgb_avg = (
                        f"{diag_rgb_ms / diag_rgb_count:.1f}ms"
                        if diag_rgb_count
                        else "none"
                    )
                    depth_avg = (
                        f"{diag_depth_ms / diag_depth_count:.1f}ms"
                        if diag_depth_count
                        else "none"
                    )
                    log.info(
                        "[%s] recording diagnostics: recording=%s loops=%d "
                        "overruns=%d max_loop=%.1fms rgb_set_avg=%s depth_avg=%s",
                        cam_name,
                        data_manager.is_recording_active(),
                        diag_loops,
                        diag_overruns,
                        diag_max_loop_ms,
                        rgb_avg,
                        depth_avg,
                    )
                    diag_last_ts = diag_now
                    diag_loops = 0
                    diag_overruns = 0
                    diag_max_loop_ms = 0.0
                    diag_rgb_ms = 0.0
                    diag_depth_ms = 0.0
                    diag_rgb_count = 0
                    diag_depth_count = 0
            # No trailing sleep: wait_for_frames blocks until the next frame, so
            # it already paces this loop at the sensor's FPS. Sleeping here would
            # let frames pile up unread and re-introduce the staleness/lag this
            # per-camera-thread design exists to remove.
    except Exception:  # noqa: BLE001 - one camera must not kill the others
        log.exception("Camera capture loop crashed [%s]", cam_name)
    finally:
        stop_camera_metrics(cam_name)


def camera_thread(
    data_manager: DataManager,
    cameras: list[CameraSpec] | None = None,
    enable_depth: bool | set[str] = False,
    depth_process_rate_hz: float | None = None,
    stop_event: threading.Event | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> None:
    """Camera thread with optional multi-camera and depth support.

    ``enable_depth`` may be a bool (depth on/off for every camera) or a set of
    camera names that should capture depth (others stay RGB-only). Restricting
    depth to fewer cameras cuts the CPU cost of depth-to-color alignment.
    """
    log.info("Camera thread started")
    diagnostics_enabled = os.getenv(
        "PIPER_RECORDING_DIAGNOSTICS", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    requested_fps = int(CAMERA_FRAME_STREAMING_RATE)
    effective_depth_rate_hz = (
        float(depth_process_rate_hz)
        if depth_process_rate_hz is not None
        else float(DEPTH_PROCESS_RATE)
    )
    requested_depth_fps = (
        requested_fps
        if effective_depth_rate_hz <= 0.0
        else max(1, int(round(effective_depth_rate_hz)))
    )
    handles: list[_PipelineHandle] = []

    try:
        resolved_cameras = _resolve_cameras(cameras)
        for spec in resolved_cameras:
            init_camera_metrics(spec.name)
        connected = list_connected_camera_serials()

        if cameras:
            missing = [
                spec
                for spec in resolved_cameras
                if spec.serial and spec.serial not in connected
            ]
            if missing:
                # A camera that is unplugged (or was left wedged by a previous
                # run) must not stop the others from coming up: an operator with
                # one loose cable still wants the remaining feeds live on Agora
                # and still wants RGBD recorded. Skip it loudly and continue.
                log.error(
                    "Configured camera(s) not connected, starting WITHOUT them: "
                    "%s. Connected serials: %s",
                    ", ".join(f"{s.name} (serial={s.serial})" for s in missing),
                    connected or "[]",
                )
                skipped = {spec.name for spec in missing}
                resolved_cameras = [
                    spec for spec in resolved_cameras if spec.name not in skipped
                ]
            if not resolved_cameras:
                raise RuntimeError(
                    "None of the configured camera serial(s) are connected. "
                    f"Connected serials: {connected or '[]'}"
                )

        for spec in resolved_cameras:
            spec_enable_depth = (
                enable_depth
                if isinstance(enable_depth, bool)
                else (spec.name in enable_depth)
            )
            handle = _start_pipeline_with_fallbacks(
                spec=spec,
                requested_enable_depth=spec_enable_depth,
                requested_fps=requested_fps,
                requested_depth_fps=requested_depth_fps,
            )
            if handle is not None:
                handles.append(handle)

        if not handles:
            log.error(
                "Camera thread could not start any camera pipeline. "
                "Check USB bandwidth/power and stream profile constraints."
            )
            return

        # Seed each camera's metrics so a healthy camera reads 0 reconnects /
        # connected=1 instead of "no data"until the first event.
        for _h in handles:
            init_camera_metrics(_h.spec.name)

        # Throttle the CPU-heavy depth align+process so it
        # does not steal cycles from the real-time control loops.
        depth_process_period_s = (
            1.0 / effective_depth_rate_hz if effective_depth_rate_hz > 0 else 0.0
        )
        if any(h.depth_enabled for h in handles):
            log.info("Depth processing rate: %g Hz", effective_depth_rate_hz)

        # One capture thread PER camera. A single shared loop previously serviced
        # every pipeline sequentially with a blocking wait_for_frames, so each
        # camera was head-of-line blocked behind the others: N cameras ran well
        # below their FPS (very laggy feeds) even while the CPU sat almost idle.
        # Per-camera threads let each pipeline be drained the instant a frame is
        # ready, so DataManager always holds the freshest frame.
        workers: list[threading.Thread] = []
        for handle in handles:
            worker = threading.Thread(
                target=_capture_camera_loop,
                kwargs=dict(
                    handle=handle,
                    data_manager=data_manager,
                    depth_process_period_s=depth_process_period_s,
                    diagnostics_enabled=diagnostics_enabled,
                    stop_event=stop_event,
                ),
                name=f"camera-{handle.spec.name}",
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        # Block until shutdown; each worker owns its own capture + reconnect loop.
        # A worker that gave up on its camera exits on its own; the rest keep
        # streaming, and only when every camera is gone is there nothing left to
        # supervise.
        while not data_manager.is_shutdown_requested() and not (
            stop_event is not None and stop_event.is_set()
        ):
            if not any(w.is_alive() for w in workers):
                log.error(
                    "Every camera has stopped — no frames will be produced "
                    "until this process is restarted."
                )
                break
            time.sleep(0.2)

        for worker in workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                # Likely still inside a 5 s wait_for_frames; the pipeline stop in
                # `finally` will unblock it, but flag it so a wedged camera thread
                # is visible rather than silently leaked.
                log.warning("Camera worker '%s' did not stop within 10s", worker.name)

    except Exception as exc:
        log.exception("Camera thread error")
        if error_callback is not None:
            try:
                error_callback(exc)
            except Exception:
                pass
        # Camera is optional for teleop; do not crash whole system.
    finally:
        for handle in handles:
            if not handle.streaming:
                # Already stopped by its own capture loop (unplugged / gave up).
                continue
            try:
                with _DEVICE_OP_LOCK:
                    handle.pipeline.stop()
                log.info("RealSense pipeline stopped [%s]", handle.spec.name)
            except Exception as exc:
                log.warning("Error stopping pipeline [%s]: %s", handle.spec.name, exc)
        log.info("Camera thread stopped")
