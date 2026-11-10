"""Pin OpenCV to one thread per call.

OpenCV parallelises *inside* a single call (``resize``, ``cvtColor``, ``rotate``)
across a pool of ``ncores - 1`` worker threads. With the pthreads backend those
workers **spin-wait** between calls rather than sleeping, so the pool costs CPU
continuously once anything touches cv2 -- and this stack touches it 100+ times a
second (per-camera rotate, per-publisher resize/I420 convert).

That parallelism is worthless here: the frames are small (640x480), and the
pipeline is *already* parallel at a coarser grain -- one capture thread and one
publisher thread per camera. Intra-op threads just add spin and contention on
top of threads that are themselves the real unit of work.

Measured on the Jetson (8 cores, OpenCV 5.0.0/pthreads), replaying the agora
send loop's per-frame work on 4 threads at 30 fps:

    cv2.setNumThreads=8  ->  0.63 cores   (119.1 frames/s)
    cv2.setNumThreads=1  ->  0.07 cores   (119.3 frames/s)

Same throughput, ~9x less CPU. In the live recorder the pool was 7 threads at
~0.11 cores each (~0.8 cores, ~10% of the box) doing nothing useful.

Call :func:`limit_opencv_threads` once at process start, before the camera and
publisher threads spin up.
"""

from __future__ import annotations

import os

_DEFAULT_ENV = "PIPER_OPENCV_THREADS"


def limit_opencv_threads(threads: int | None = None) -> int | None:
    """Cap OpenCV's intra-op thread pool. Returns the value applied, or None.

    ``threads`` defaults to ``$PIPER_OPENCV_THREADS`` and then to 1. Set the env
    var to 0 to leave OpenCV's default pool alone (useful if a future caller
    does big single-frame work that genuinely wants intra-op parallelism).

    Safe to call when OpenCV is absent -- returns None rather than raising.
    """
    if threads is None:
        raw = os.getenv(_DEFAULT_ENV, "").strip()
        try:
            threads = int(raw) if raw else 1
        except ValueError:
            threads = 1
    if threads <= 0:
        return None  # explicitly opted out
    try:
        import cv2
    except Exception:  # noqa: BLE001 - cv2 is optional at runtime
        return None
    try:
        cv2.setNumThreads(threads)
        return threads
    except Exception:  # noqa: BLE001 - never let a tuning knob break startup
        return None


__all__ = ["limit_opencv_threads"]
