"""Publish DataManager camera frames as LiveKit video tracks (follower side).

An **alternative** to :mod:`arms.agora_rt.agora_video`, not a
replacement: both read the same frames from the same :class:`DataManager` and
either, or both, can run. Nothing on the Agora path changes — the recorder
decides with flags.

Why it exists: Agora's Python/Linux SDK encodes in software, which is what keeps
the Jetson's CPU pinned. LiveKit's Rust core carries a **Jetson MMAPI** hardware
encoder (H.264/H.265/AV1) and selects it automatically when the platform offers
one for the negotiated codec, falling back to software otherwise. That is the
whole point of this module, and it has two hard requirements that are easy to
get wrong:

1. **The codec must be H.264.** Jetson MMAPI has no VP8/VP9 path, so publishing
   VP8 (the WebRTC default, and what the Agora viewer asks for) silently lands
   back on a software encoder — the exact thing being escaped. Hence the
   explicit ``video_codec``.
2. **Simulcast must be off.** Simulcast encodes the same camera two or three
   times at different resolutions. On a desktop that is a bandwidth feature; on
   a robot publishing N cameras it multiplies the encoder count by three.

**Video only.** No audio source, no audio track, and the token is minted with
``can_publish_sources=["camera"]`` so audio cannot be published even by mistake.
The room is joined with ``auto_subscribe=False``, so the Jetson never downloads
another participant's media either — there is no viewer whose video the robot
needs, and receiving it would cost CPU and egress for nothing.

Threading: the LiveKit SDK is asyncio and the recorder is threaded, so the room
lives on its own event loop in a daemon thread (same shape as
``livekit_quest_reader``), and frames are pumped by one asyncio task per camera.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from typing import Any

import numpy as np
from arms.data_manager import DataManager
from arms.logging_setup import console_logger

from arms.livekit_rt.livekit_bridge import (
    VIDEO_ROOM_SUFFIX,
    get_static_token,
    get_url,
    video_room,
)
from arms.livekit_rt.livekit_token import build_token

# Pinned to the console like the Agora publisher's log: an operator needs to see
# publish/connection state for the live feed without turning on a debug flag.
log = console_logger("livekit-video")

DEFAULT_FPS = 15
DEFAULT_BITRATE_KBPS = 1500
DEFAULT_IDENTITY = "piper-camera"
DEFAULT_CONNECT_TIMEOUT_S = 15.0

# A camera's last frame lives in the DataManager indefinitely, so "older than
# this" is how a dead camera is told apart from a live one. Matches the Agora
# publisher's default so the two paths report a dead camera at the same moment.
try:
    _STALE_FRAME_S = float(os.getenv("PIPER_LIVEKIT_STALE_FRAME_S", "5") or 5.0)
except ValueError:
    _STALE_FRAME_S = 5.0
_STALE_FRAME_NS = int(_STALE_FRAME_S * 1e9)


def add_livekit_video_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--livekit-video*`` flags on the camera recorder."""
    group = parser.add_argument_group("LiveKit live video (alternative to Agora)")
    group.add_argument(
        "--livekit-video",
        action="store_true",
        help="Publish live RGB to LiveKit (hardware-encoded on Jetson). "
        "Independent of --agora-video; both may run at once for an A/B.",
    )
    group.add_argument(
        "--livekit-video-camera",
        default="",
        help="Comma-separated camera names to publish (default: all cameras).",
    )
    group.add_argument(
        "--livekit-video-fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Publish frame rate (default: {DEFAULT_FPS}).",
    )
    group.add_argument(
        "--livekit-video-resolution",
        default="",
        help="WxH to publish (default: the camera's native size, which avoids "
        "a rescale on the way into the encoder).",
    )
    group.add_argument(
        "--livekit-video-bitrate",
        type=int,
        default=DEFAULT_BITRATE_KBPS,
        help=f"Max bitrate per camera in kbps (default: {DEFAULT_BITRATE_KBPS}).",
    )
    group.add_argument(
        "--livekit-video-room",
        default="",
        help="Room to publish into. Overrides $LIVEKIT_VIDEO_ROOM; defaults to "
        f"'<robot_id>{VIDEO_ROOM_SUFFIX}'. Give parallel deployments of the same "
        "robot distinct rooms so they do not evict each other.",
    )
    group.add_argument(
        "--livekit-video-identity",
        default="",
        help=f"Publisher participant identity (default: {DEFAULT_IDENTITY}).",
    )
    group.add_argument(
        "--livekit-video-codec",
        default="h264",
        choices=["h264", "h265", "av1", "vp8"],
        help="Negotiated codec. Only h264/h265/av1 can reach the Jetson "
        "hardware encoder; vp8 is software-only and is here for debugging.",
    )


def _sanitize_rgb_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Coerce a DataManager frame to a contiguous HxWx3 uint8 RGB of this size."""
    frame = image
    if frame.ndim == 2:  # grayscale -> RGB
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:  # RGBA -> RGB
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    if frame.shape[0] != height or frame.shape[1] != width:
        try:
            import cv2

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 - cv2 optional; pad/crop instead
            out = np.zeros((height, width, 3), dtype=np.uint8)
            h = min(height, frame.shape[0])
            w = min(width, frame.shape[1])
            out[:h, :w] = frame[:h, :w]
            frame = out
    return np.ascontiguousarray(frame)


def _placeholder(width: int, height: int, label: str) -> np.ndarray:
    """A dark frame carrying ``label``, published when a camera has no frames.

    Publishing something is deliberate: the LiveKit docs are explicit that a
    source should keep pushing frames even when the image does not change, and
    a viewer tile that goes black is indistinguishable from a broken link.
    """
    frame = np.full((height, width, 3), 16, dtype=np.uint8)
    try:
        import cv2

        cv2.putText(
            frame,
            label,
            (24, max(40, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.6, width / 1280.0),
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
    except Exception:  # noqa: BLE001 - a plain dark frame is an acceptable fallback
        pass
    return np.ascontiguousarray(frame)


class LiveKitVideoPublisher:
    """One LiveKit room connection publishing one video track per camera."""

    def __init__(
        self,
        data_manager: DataManager,
        camera_names: list[str],
        *,
        fps: int = DEFAULT_FPS,
        width: int = 640,
        height: int = 480,
        bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
        room: str | None = None,
        identity: str = "",
        url: str | None = None,
        codec: str = "h264",
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ) -> None:
        """Configure the publisher; nothing connects until :meth:`start`."""
        self._dm = data_manager
        self._cameras = list(camera_names)
        self._fps = max(1, int(fps))
        self._width = int(width)
        self._height = int(height)
        self._bitrate_bps = max(1, int(bitrate_kbps)) * 1000
        self._room_name = video_room(room)
        self._identity = identity.strip() or DEFAULT_IDENTITY
        self._url = url or get_url()
        self._codec = codec.lower()
        self._connect_timeout_s = float(connect_timeout_s)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._room: Any = None
        self._offline = {
            cam: _placeholder(self._width, self._height, f"{cam}: OFFLINE")
            for cam in self._cameras
        }
        self._waiting = {
            cam: _placeholder(self._width, self._height, f"{cam}: waiting…")
            for cam in self._cameras
        }

    @property
    def room_name(self) -> str:
        """The room this publisher joined — worth logging, it is per robot."""
        return self._room_name

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Connect, publish every camera, and start pumping frames.

        Blocks until the room is up (or raises), so the recorder's startup log
        tells the truth about whether the feed is live.
        """
        if not self._cameras:
            raise ValueError("no cameras to publish")
        self._thread = threading.Thread(
            target=self._run, name="livekit-video", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(self._connect_timeout_s):
            self.stop()
            raise TimeoutError(
                f"LiveKit video did not connect within {self._connect_timeout_s:g}s"
            )
        if self._error is not None:
            raise self._error

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop pumping and disconnect."""
        self._stop.set()
        loop, thread = self._loop, self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)  # wake the loop to notice _stop
        if thread is not None:
            thread.join(timeout=timeout_s)

    # -- asyncio side ----------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001 - reported through start()
            self._error = exc
            log.error("LiveKit video stopped: %r", exc)
        finally:
            self._ready.set()  # unblock start() on failure too
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _main(self) -> None:
        try:
            from livekit import rtc  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - surface a fixable message
            raise RuntimeError(
                "livekit is not installed. Run: pip install livekit"
            ) from exc

        # Mint, don't inherit. LIVEKIT_TOKEN exists for the XR control path and
        # is bound to THAT room with data-publish rights; using it here would
        # either land the cameras in the Quest's room or fail auth outright. It
        # is honoured only when there is no key/secret to mint with, and then
        # it is the operator's job to have scoped it to the video room.
        token = ""
        try:
            token = build_token(
                self._identity,
                room=self._room_name,
                name=self._identity,
                # Video only, and nothing else: no data packets (that is the XR
                # transport's job, in its own room) and no subscribing.
                can_publish=True,
                can_subscribe=False,
                can_publish_data=False,
                can_publish_sources=["camera"],
            )
        except RuntimeError:
            token = get_static_token()
            if not token:
                raise
            log.warning(
                "No LIVEKIT_API_KEY/SECRET — falling back to the static "
                "LIVEKIT_TOKEN. It must be scoped to room '%s', or the cameras "
                "will not appear where the viewer looks for them.",
                self._room_name,
            )

        room = rtc.Room()
        self._room = room
        # auto_subscribe=False: never pull a viewer's media onto the Jetson.
        await room.connect(self._url, token, rtc.RoomOptions(auto_subscribe=False))
        log.info(
            "LiveKit video connected: room '%s' as '%s' — %d camera(s) at "
            "%dx%d @ %d fps, %s, max %d kbps each, simulcast off, no audio.",
            self._room_name,
            self._identity,
            len(self._cameras),
            self._width,
            self._height,
            self._fps,
            self._codec.upper(),
            self._bitrate_bps // 1000,
        )

        sources: dict[str, Any] = {}
        for cam in self._cameras:
            source = rtc.VideoSource(self._width, self._height)
            track = rtc.LocalVideoTrack.create_video_track(cam, source)
            await room.local_participant.publish_track(
                track, self._publish_options(rtc)
            )
            sources[cam] = source
            log.info("LiveKit video publishing track '%s'", cam)

        self._ready.set()
        tasks = [
            asyncio.create_task(self._pump(rtc, cam, src))
            for cam, src in sources.items()
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            with_suppressed = asyncio.gather(*tasks, return_exceptions=True)
            await with_suppressed
            await room.disconnect()
            log.info("LiveKit video disconnected from room '%s'", self._room_name)

    def _publish_options(self, rtc: Any) -> Any:
        """Publish options: forced codec, no simulcast, capped bitrate."""
        codec = {
            "h264": rtc.VideoCodec.H264,
            "vp8": rtc.VideoCodec.VP8,
        }.get(self._codec)
        if codec is None:  # h265 / av1 only exist on newer SDKs
            codec = getattr(rtc.VideoCodec, self._codec.upper(), rtc.VideoCodec.H264)
        return rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_codec=codec,
            # See the module docstring: simulcast would triple the encoder count.
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_framerate=self._fps,
                max_bitrate=self._bitrate_bps,
            ),
        )

    async def _pump(self, rtc: Any, camera_name: str, source: Any) -> None:
        """Push this camera's latest frame at the publish rate, forever."""
        interval = 1.0 / self._fps
        was_stale = False
        loop = asyncio.get_running_loop()
        while not self._stop.is_set() and not self._dm.is_shutdown_requested():
            started = loop.time()
            try:
                # copy=False: this loop only reads the frame to encode it.
                snapshot = self._dm.get_camera_frame(camera_name, copy=False)
                stamp_ns = snapshot.rgb_timestamp_ns
                stale = (
                    stamp_ns is not None
                    and (time.time_ns() - stamp_ns) > _STALE_FRAME_NS
                )
                if snapshot.rgb_image is None:
                    rgb = self._waiting[camera_name]
                elif stale:
                    if not was_stale:
                        log.warning(
                            "Camera '%s' has not produced a frame in %gs — "
                            "publishing an OFFLINE tile; other cameras unaffected.",
                            camera_name,
                            _STALE_FRAME_S,
                        )
                        was_stale = True
                    rgb = self._offline[camera_name]
                else:
                    if was_stale:
                        log.info("Camera '%s' frames are flowing again", camera_name)
                        was_stale = False
                    rgb = _sanitize_rgb_frame(
                        snapshot.rgb_image, self._width, self._height
                    )
                # RGB24 straight in: the SDK converts to I420 in Rust during
                # capture_frame, which keeps the colour conversion off the
                # Python side entirely (the Agora path does it with cv2).
                frame = rtc.VideoFrame(
                    self._width, self._height, rtc.VideoBufferType.RGB24, rgb.tobytes()
                )
                source.capture_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the feed alive
                log.warning("LiveKit video send error [%s]: %r", camera_name, exc)
                await asyncio.sleep(0.1)

            remaining = interval - (loop.time() - started)
            await asyncio.sleep(max(0.0, remaining))


__all__ = [
    "DEFAULT_BITRATE_KBPS",
    "DEFAULT_FPS",
    "LiveKitVideoPublisher",
    "add_livekit_video_args",
    "video_room",
]
