"""Linux ``/proc`` reader: the only way this package learns about processes.

Deliberately stdlib-only and free of any robot-stack import, so the supervisor
keeps working when the teleop venv is half-rebuilt or ``piper_sdk`` is broken --
the moment you most need a reliable "stop everything" button.

Every lookup here is *identity-checked*: a bare PID is not a process. Linux
recycles PIDs, and a supervisor that stores ``pid=2978560`` across a reboot and
later signals it will eventually SIGKILL something innocent. A process is
therefore identified by ``(boot_id, pid, starttime)``, where ``starttime`` is
field 22 of ``/proc/<pid>/stat`` -- the boot-relative clock tick the process
began at, which the kernel never reuses within a boot.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PROC = Path("/proc")


def boot_id() -> str:
    """Identifier that changes on every reboot; invalidates stored PIDs."""
    try:
        return (PROC / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class ProcInfo:
    """One live process, as of the moment ``read()`` walked ``/proc``."""

    pid: int
    ppid: int
    pgid: int
    sid: int
    state: str
    starttime: int
    rss_kb: int
    cpu_ticks: int

    @property
    def zombie(self) -> bool:
        """True for a reaped-but-not-waited child; it holds no resources."""
        return self.state == "Z"


def read(pid: int) -> ProcInfo | None:
    """Parse ``/proc/<pid>/stat``, or None if the process is gone."""
    try:
        raw = (PROC / str(pid) / "stat").read_text()
    except (OSError, ValueError):
        return None
    # Field 2 is comm, wrapped in parens and free to contain spaces *and*
    # parens ("(ffmpeg (2))"), so the only safe split is after the LAST ')'.
    close = raw.rfind(")")
    if close < 0:
        return None
    rest = raw[close + 2 :].split()
    if len(rest) < 22:
        return None
    try:
        return ProcInfo(
            pid=pid,
            state=rest[0],
            ppid=int(rest[1]),
            pgid=int(rest[2]),
            sid=int(rest[3]),
            cpu_ticks=int(rest[11]) + int(rest[12]),  # utime + stime
            starttime=int(rest[19]),
            rss_kb=int(rest[21]) * (os.sysconf("SC_PAGE_SIZE") // 1024),
        )
    except (ValueError, IndexError):
        return None


def cmdline(pid: int) -> list[str]:
    """Argv of ``pid``. Empty for kernel threads and for processes that died."""
    try:
        raw = (PROC / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\0") if a]


def cwd(pid: int) -> str:
    """Working directory of ``pid``, or "" when unreadable (other user)."""
    try:
        return os.readlink(PROC / str(pid) / "cwd")
    except OSError:
        return ""


def alive(pid: int, starttime: int | None = None) -> bool:
    """True only if ``pid`` is running *and* is the same process as before.

    A zombie counts as dead: it has already released the CAN socket, the
    cameras and its shared memory, which is all the supervisor cares about.
    """
    info = read(pid)
    if info is None or info.zombie:
        return False
    return starttime is None or info.starttime == starttime


def iter_procs() -> Iterator[ProcInfo]:
    """Yield every readable process on the host, skipping kernel threads."""
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        info = read(int(entry.name))
        if info is not None and not info.zombie:
            yield info


_snapshot: tuple[float, list[ProcInfo]] = (0.0, [])

# A live process can never change its own argv, so cmdline is read once per
# process and cached by identity. This is where the saving actually is: a walk
# is a few hundred cheap stat reads, but re-reading every cmdline on every poll
# is not. In steady state this cache serves all of them.
_cmd_cache: dict[tuple[int, int], list[str]] = {}


# A walk costs ~400 small reads on this host. Several browsers polling at once
# would each trigger one, so a short TTL collapses them into a single walk --
# but only because :func:`invalidate` is called the instant anything changes.
# Time-based staleness alone was not safe: a cached walk could predate the very
# job being described, which reported a just-started job as having -1 children.
SNAPSHOT_TTL = 2.5


def invalidate() -> None:
    """Drop the cached walk. Call after anything that starts or stops a process."""
    global _snapshot
    _snapshot = (0.0, [])


def snapshot(max_age: float = 0.0) -> list[ProcInfo]:
    """All live processes.

    Fresh by default. Pass ``max_age`` (typically :data:`SNAPSHOT_TTL`) only on
    read-only polling paths, which are safe precisely because every mutating
    path calls :func:`invalidate`.
    """
    global _snapshot
    now = time.monotonic()
    if max_age > 0 and now - _snapshot[0] <= max_age and _snapshot[1]:
        return _snapshot[1]
    procs = list(iter_procs())
    _snapshot = (now, procs)
    if len(_cmd_cache) > 4096:  # bound the cache on a long-lived dashboard
        live = {(p.pid, p.starttime) for p in procs}
        for key in [k for k in _cmd_cache if k not in live]:
            del _cmd_cache[key]
    return procs


def cmdline_of(info: ProcInfo) -> list[str]:
    """Cached argv for a process already described by ``info``."""
    key = (info.pid, info.starttime)
    cached = _cmd_cache.get(key)
    if cached is None:
        cached = cmdline(info.pid)
        _cmd_cache[key] = cached
    return cached


def group_members(pgid: int, snap: list[ProcInfo] | None = None) -> list[ProcInfo]:
    """Every live process in process group ``pgid``."""
    if pgid <= 1:
        return []
    return [p for p in (snap if snap is not None else snapshot()) if p.pgid == pgid]


def descendants(pid: int) -> list[ProcInfo]:
    """Every live descendant of ``pid`` (children, grandchildren, ...).

    Used as a *backstop* to the process-group sweep: a child that called
    ``setsid()`` leaves the group but keeps its parent, so it still shows up
    here. Anything that does both has orphaned itself and can only be found by
    signature -- see ``sweep`` in ``supervisor.py``.
    """
    by_parent: dict[int, list[ProcInfo]] = {}
    for proc in snapshot():
        by_parent.setdefault(proc.ppid, []).append(proc)
    out: list[ProcInfo] = []
    stack = [pid]
    seen = {pid}
    while stack:
        for child in by_parent.get(stack.pop(), []):
            if child.pid in seen:
                continue
            seen.add(child.pid)
            out.append(child)
            stack.append(child.pid)
    return out


def uptime_sec() -> float:
    """Seconds since boot, so a process's age can be derived from starttime."""
    try:
        return float((PROC / "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


_TICKS = os.sysconf("SC_CLK_TCK")


def age_sec(info: ProcInfo, host_uptime: float | None = None) -> float:
    """How long ``info`` has been running, in seconds.

    Pass ``host_uptime`` when ageing many processes at once so /proc/uptime is
    read once for the batch rather than once per process.
    """
    up = uptime_sec() if host_uptime is None else host_uptime
    return max(0.0, up - info.starttime / _TICKS)
