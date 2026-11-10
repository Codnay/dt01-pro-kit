#!/usr/bin/env python3
"""Live camera feed: RealSense cameras -> LiveKit video.

    python -m arms.cameras --camera left-arm:<serial> --camera right-arm:<serial>

Opens the listed cameras and publishes one live video track per camera into
this robot's video room (``<ROBOT_ID>-video``). Watch it with the camera page
(``python -m arms.livekit_rt.livekit_camera_dashboard.server``).

``python -m arms.cameras --list`` prints the serial of every camera plugged in.

Nothing is recorded or uploaded. Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import faulthandler
import signal
import threading
import time

from arms.camera_config import CameraSpec, parse_camera_specs
from arms.data_manager import DataManager
from arms.livekit_rt.livekit_video import add_livekit_video_args
from arms.logging_setup import console_logger
from arms.perf.cv_threads import limit_opencv_threads
from arms.robot_identity import add_robot_id_arg, resolve_robot_id
from arms.threads.camera import (
    camera_thread,
    describe_connected_cameras,
    list_connected_camera_serials,
)

log = console_logger("cameras")

CAMERA_READY_TIMEOUT_S = 30.0
CAMERA_STOP_TIMEOUT_S = 15.0


def build_parser() -> argparse.ArgumentParser:
    """Build the camera command line."""
    parser = argparse.ArgumentParser(description="Publish the robot cameras live")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="name:serial[:exposure=US][:gain=G]",
        help="Repeatable. e.g. --camera top:123456789. Add exposure=<microseconds> "
        "and/or gain=<value> to pin manual exposure on a dim camera.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the cameras plugged in right now, then exit.",
    )
    add_robot_id_arg(parser)
    add_livekit_video_args(parser)
    return parser


def _print_connected() -> int:
    cameras = describe_connected_cameras()
    if not cameras:
        print("No RealSense camera found. Check the USB cables.")
        return 1
    for serial, info in sorted(cameras.items()):
        details = "  ".join(f"{k}={v}" for k, v in sorted(info.items()) if v)
        print(f"{serial}  {details}")
    return 0


def _parse_size(raw: str) -> tuple[int, int] | None:
    try:
        width, height = raw.lower().split("x", 1)
        return int(width), int(height)
    except (ValueError, AttributeError):
        return None


def _native_size(data_manager: DataManager, names: list[str]) -> tuple[int, int]:
    """Size the cameras actually produce (frames arrive rotated 90 degrees)."""
    for name in names:
        rgb = data_manager.get_camera_frame(name, copy=False).rgb_image
        if rgb is not None and rgb.ndim == 3:
            return int(rgb.shape[1]), int(rgb.shape[0])
    return 640, 480


def _wait_for_frames(
    data_manager: DataManager,
    names: list[str],
    capture: threading.Thread,
    absent: set[str],
) -> list[str]:
    """Wait until each present camera has produced a frame; return those that did."""
    pending = set(names) - absent
    deadline = time.monotonic() + CAMERA_READY_TIMEOUT_S
    while pending and time.monotonic() < deadline:
        for name in list(pending):
            if data_manager.get_camera_frame(name, copy=False).rgb_image is not None:
                pending.discard(name)
        if pending and not capture.is_alive():
            log.error(
                "The camera thread stopped before any frame arrived. Most often "
                "another program still holds the cameras."
            )
            break
        if pending:
            time.sleep(0.1)
    if pending:
        log.warning("No frame yet from: %s", ", ".join(sorted(pending)))
    return [name for name in names if name not in pending and name not in absent]


def _start_publisher(
    args: argparse.Namespace, data_manager: DataManager, names: list[str]
):
    # Imported late: the LiveKit client opens many file descriptors, and the
    # cameras must already hold theirs (RealSense breaks on high descriptors).
    from arms.livekit_rt.livekit_video import LiveKitVideoPublisher

    wanted = [c.strip() for c in args.livekit_video_camera.split(",") if c.strip()]
    publish = [name for name in names if not wanted or name in wanted]
    size = _parse_size(args.livekit_video_resolution) if args.livekit_video_resolution else None
    width, height = size or _native_size(data_manager, publish)
    publisher = LiveKitVideoPublisher(
        data_manager,
        publish,
        fps=args.livekit_video_fps,
        width=width,
        height=height,
        bitrate_kbps=args.livekit_video_bitrate,
        room=args.livekit_video_room,
        identity=args.livekit_video_identity,
        codec=args.livekit_video_codec,
    )
    publisher.start()
    log.info(
        "Live: %s at %dx%d in room %s",
        ", ".join(publish),
        width,
        height,
        publisher.room_name,
    )
    return publisher


def _absent(cameras: list[CameraSpec]) -> set[str]:
    connected = set(list_connected_camera_serials())
    return {c.name for c in cameras if c.serial and c.serial not in connected}


def run(args: argparse.Namespace) -> int:
    """Open the cameras, publish them, and wait for Ctrl+C."""
    cameras = parse_camera_specs(args.camera)
    if not cameras:
        log.error("No camera given. Pass --camera name:serial (see --list).")
        return 2
    names = [c.name for c in cameras]
    log.info("robot_id=%s  cameras=%s", resolve_robot_id(args), ", ".join(names))

    data_manager = DataManager()
    stop_event = threading.Event()
    capture = threading.Thread(
        target=camera_thread,
        kwargs=dict(data_manager=data_manager, cameras=cameras, stop_event=stop_event),
        name="camera",
        daemon=True,
    )
    capture.start()

    absent = _absent(cameras)
    if absent:
        log.warning("Not plugged in: %s", ", ".join(sorted(absent)))
    ready = _wait_for_frames(data_manager, names, capture, absent)

    publisher = None
    try:
        if not ready:
            log.error("No camera produced a frame; stopping.")
            return 1
        # Publish every configured camera: one that is plugged in later starts
        # streaming on its own.
        publisher = _start_publisher(args, data_manager, names)
        log.info("Cameras are live. Ctrl+C to stop.")
        while capture.is_alive():
            time.sleep(1.0)
        log.error("The camera thread stopped.")
        return 1
    except KeyboardInterrupt:
        log.info("Stopping...")
        return 0
    finally:
        # Stop producing frames before stopping the consumer that reads them.
        stop_event.set()
        data_manager.request_shutdown()
        capture.join(timeout=CAMERA_STOP_TIMEOUT_S)
        if publisher is not None:
            publisher.stop()
        log.info("Stopped.")


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    """Parse the command line and run."""
    faulthandler.enable()
    args = build_parser().parse_args()
    if args.list:
        return _print_connected()
    # OpenCV's thread pool spin-waits for no gain: capture is already one thread
    # per camera.
    limit_opencv_threads()
    # SIGINT is set explicitly as well: a process started in the background by
    # a script inherits "ignore SIGINT", and would then never stop on Ctrl+C.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _raise_interrupt)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
