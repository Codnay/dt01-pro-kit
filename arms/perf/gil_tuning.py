"""Shorten CPython's GIL handoff interval so the control loops stay on time.

The teleop process is a *single* CPython process, so every hot thread -- the
piper_sdk ``ReadCan`` decoder, ``control_loop``, ``ik_solver``, ``joint_state``,
the quest reader -- serialises on one GIL. It can never use more than ~1 core of
Python execution no matter how many cores are idle.

CPython hands the GIL over lazily: a thread that wants it sets ``gil_drop_request``
and the holder only releases at its next bytecode check *after* the switch interval
has elapsed (default **5 ms**). With N hot threads the worst-case wait to reacquire
is roughly ``N x switchinterval`` -- so a 100 Hz loop with a 10 ms budget can be
stalled for 25 ms+ by pure handoff latency, having done nothing wrong.

Measured on the Jetson Orin NX (aarch64, 8 cores), 4 GIL-hungry threads plus one
fixed-rate 100 Hz loop, reporting that loop's period distribution:

    switchinterval = 5.00 ms   ->  p99 = 147.50 ms   max = 216.15 ms
    switchinterval = 1.00 ms   ->  p99 =  40.74 ms   max =  55.52 ms
    switchinterval = 0.50 ms   ->  p99 =  33.09 ms   max =  52.65 ms

~4x off the tail for one line. This is real on x86 too, but the Jetson's slower
per-core performance makes the same Python work occupy a much larger share of the
GIL, which is what turns the latency into visible arm jitter: irregular setpoint
arrival makes the joint firmware interpolate over uneven intervals, so the arm
rings and feels like it is resisting.

This is a *mitigation*, not a cure. It reduces handoff latency; it does not create
GIL capacity. The cure is to take Python work out of the process (see
``ReadCan`` -- ~45% of this process's GIL budget decoding ~3.4k CAN frames/s) or
to split the arms into separate processes so each gets its own GIL.

Call :func:`tune_gil_switch_interval` once at process start, before threads spin
up. Set ``PIPER_GIL_SWITCH_INTERVAL`` (seconds) to override, or ``0`` to leave
CPython's default alone.
"""

from __future__ import annotations

import os
import sys

_ENV = "PIPER_GIL_SWITCH_INTERVAL"

# 0.5 ms. Low enough to keep handoff latency well inside a 10 ms control tick,
# high enough that the extra context switching stays in the noise (the interval
# only bounds how long a *contended* handoff waits -- it does not force a switch
# when nothing else wants the GIL).
_DEFAULT_S = 0.0005


def tune_gil_switch_interval(interval_s: float | None = None) -> float | None:
    """Shorten the GIL handoff interval. Returns the value applied, or None.

    ``interval_s`` defaults to ``$PIPER_GIL_SWITCH_INTERVAL`` and then to 0.5 ms.
    A value of 0 (or a malformed env var) leaves CPython's default untouched.
    """
    if interval_s is None:
        raw = os.environ.get(_ENV, "").strip()
        if raw:
            try:
                interval_s = float(raw)
            except ValueError:
                return None
        else:
            interval_s = _DEFAULT_S

    if interval_s <= 0:
        return None

    try:
        sys.setswitchinterval(interval_s)
    except (ValueError, AttributeError):
        return None
    return interval_s


_RT_ENV = "PIPER_CONTROL_LOOP_RT"

# SCHED_FIFO priority for the control loop. Deliberately low: Linux kernel
# threads (irq/*, softirq) sit at 50, so 10 keeps the loop ahead of every normal
# task without ever outranking the kernel machinery that services the CAN USB
# device it depends on.
_RT_PRIORITY = 10


def try_realtime_priority(priority: int | None = None) -> int | None:
    """Put the CALLING thread on SCHED_FIFO. Returns the priority, or None.

    The 100 Hz control loop is the one thread whose lateness is felt directly:
    the joint firmware interpolates between setpoints assuming even spacing, so
    a tick that arrives late and then early becomes a velocity discontinuity --
    physical buzzing. Measured on the Jetson, that thread sits ~9.7 ms/s
    *runnable but not scheduled* and misses ~2% of its 10 ms deadlines, while
    the same code on a faster host misses 0%.

    Safe to make real-time because this loop sleeps ~85% of every period (its
    own tail sleep); it cannot monopolise a CPU. It is also strictly opt-in at
    the OS level -- without ``rtprio`` in limits.conf the call simply fails and
    we stay on CFS, so this can never be the reason the arm does not come up.

    Grant it with, in ``/etc/security/limits.conf``::

        <user>  -  rtprio  20
    """
    if os.environ.get(_RT_ENV, "").strip().lower() in {"0", "false", "no", "off"}:
        return None
    prio = _RT_PRIORITY if priority is None else int(priority)
    try:
        param = os.sched_param(prio)  # type: ignore[attr-defined]
        os.sched_setscheduler(0, os.SCHED_FIFO, param)  # type: ignore[attr-defined]
    except (AttributeError, OSError, PermissionError, ValueError):
        return None
    return prio


__all__ = ["try_realtime_priority", "tune_gil_switch_interval"]
