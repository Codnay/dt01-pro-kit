"""The lifting column, over RS-232 — the path that actually drives it.

The vendor's HTTP ``cmd/lift_ctrl`` was tried first and is a dead end on this
hardware: it answers ``successed: true`` and moves nothing (probed against the
real robot on 2026-09-02). The column is a separate device on a CP210x USB
serial adapter, speaking the protocol in ``lifting_platform_ros/``. That path is
also strictly better — it reports height, limit switches, a stop button and
drive errors, none of which the HTTP endpoint offers.

Wire format, from ``lifting_platform_ros_pkg.cc``::

    DE ED | len | cmd | data[8] | sum_lo sum_hi      len = 6 + len(data) = 14
    sum = 16-bit sum of every preceding byte, little-endian

The controller broadcasts ``state`` (0x02) and ``motion`` (0x03) frames
continuously at roughly 60 Hz, so reading needs no polling at all.

**The deadman is not optional.** ``pkgSetVelocity`` sends once and the column
*latches* the last velocity it was given — there is no device-side watchdog. A
held button whose release packet is lost would leave the column travelling until
a limit switch. So velocity is re-sent while held and explicitly zeroed on
release, exactly as :mod:`drive` does for the base, and for the same reason.

Stdlib only (``termios``), like the rest of the dashboard: no ``pyserial`` on
the robot's network.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any

try:                                   # POSIX only, and optional even there
    import termios
except ImportError:                    # pragma: no cover - Windows
    termios = None                     # type: ignore[assignment]

# The dashboard is handed to whoever's laptop is beside the robot, and the
# vendor ships a Windows .exe, so Windows is a real target. Importing this
# module must therefore never fail — the column is simply unavailable there,
# which is also true in the sense that matters: no /dev/ttyUSB0 either.
DEVICE = "/dev/ttyUSB0"

CMD_STATE, CMD_MOTION, CMD_ERROR = 0x02, 0x03, 0x04
CMD_VELOCITY, CMD_POSITION = 0x08, 0x09

RATE_HZ = 10.0           # velocity re-send rate while a button is held
DEADMAN_S = 0.6          # zero the column if no intent arrives within this
ZERO_BURST = 8           # zero frames sent after motion before going idle
STALE_S = 2.0            # feedback older than this is not to be trusted
MAX_PERCENT = 60         # ceiling the UI cannot exceed

# Seeking a height is closed-loop on the reported height, NOT the controller's
# own position mode. Position frames (0x09) are accepted and ignored by this
# hardware -- target_hight stays 0 and the column does not move -- while
# velocity frames work, so the loop is built on the primitive that works.
# Travel limits, measured on the hardware on 2026-09-02. The column's own limit
# switches remain the real protection; these stop the dashboard commanding into
# them, which is a different job — defence in depth, not a replacement.
MIN_HEIGHT_MM = 50
MAX_HEIGHT_MM = 850

TOLERANCE_MM = 2         # close enough to stop
SLOW_MM = 25             # approach distance where the speed is cut
SLOW_PERCENT = 12        # creep speed near the target, to limit overshoot
SEEK_TIMEOUT_S = 90.0    # give up rather than drive forever on a stuck column
SEEK_STALL_S = 4.0       # no progress for this long means something is wrong

ACTIONS = {0: "idle", 1: "down_finish", 2: "up_finish",
           3: "going_down", 4: "going_up"}


def encode(cmd: int, data: bytes = b"\x00" * 8) -> bytes:
    """One protocol frame, checksum included."""
    body = bytes([0xDE, 0xED, 6 + len(data), cmd]) + data
    total = sum(body) & 0xFFFF
    return body + bytes([total & 0xFF, total >> 8])


def velocity_frame(percent: int) -> bytes:
    """Velocity command. Positive raises, negative lowers, zero stops."""
    percent = max(-MAX_PERCENT, min(MAX_PERCENT, int(percent)))
    return encode(CMD_VELOCITY, percent.to_bytes(1, "little", signed=True) + b"\x00" * 7)


def position_frame(height_mm: int, percent: int) -> bytes:
    """Go to an absolute height at a given speed percentage."""
    percent = max(1, min(MAX_PERCENT, int(percent)))
    return encode(CMD_POSITION, bytes([percent])
                  + int(height_mm).to_bytes(2, "little", signed=True) + b"\x00" * 5)


def parse(buffer: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """Split a byte stream into ``(cmd, data)`` frames, keeping the remainder.

    Checksum failures are dropped silently rather than raised: on a serial line
    a corrupt frame is normal traffic, and the next good one is 16 ms away.
    """
    found: list[tuple[int, bytes]] = []
    while True:
        start = buffer.find(b"\xde\xed")
        if start < 0:
            return found, b""
        if len(buffer) - start < 4:
            return found, buffer[start:]
        length = buffer[start + 2]
        if length < 6 or length > 64:                  # resync past a bad header
            buffer = buffer[start + 2:]
            continue
        if len(buffer) - start < length:
            return found, buffer[start:]
        chunk = buffer[start:start + length]
        if (sum(chunk[:-2]) & 0xFFFF) == int.from_bytes(chunk[-2:], "little"):
            found.append((chunk[3], chunk[4:-2]))
        buffer = buffer[start + length:]


def _i16(data: bytes, at: int) -> int:
    return int.from_bytes(data[at:at + 2], "little", signed=True)


class LiftLink:
    """Reader thread plus a deadman'd writer for the lifting column."""

    def __init__(self, device: str = DEVICE, fd: int | None = None) -> None:
        self.device = device
        self._fd = fd
        self._lock = threading.Lock()
        self._motion: bytes | None = None
        self._state: bytes | None = None
        self._seen_at = 0.0
        # Recent (time, height) so status() can report real mm/s. The motor
        # percentage is a duty cycle, not a speed — nothing in the protocol says
        # how fast a given percent actually moves this column, so measure it.
        self._track: collections.deque = collections.deque(maxlen=240)
        self._error: str | None = None
        self._intent = 0
        self._intent_at = 0.0
        self._zeros_left = 0
        self._goto: bytes | None = None
        self._seek: int | None = None
        self._seek_percent = 30
        self._seek_deadline = 0.0
        self._seek_progress_at = 0.0
        self._seek_last_mm = 0
        self._seek_error: str | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- lifecycle ---------------------------------------------------------
    def open(self) -> bool:
        """Open the port. False (with .error set) rather than raising."""
        if termios is None:
            self._error = ("serial ports need POSIX termios, which this platform "
                           "does not have — the lifting column is unavailable here")
            return False
        if self._fd is None:
            try:
                fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            except OSError as exc:
                self._error = (f"{self.device}: {exc.strerror}. "
                               "The dashboard's user must be in the 'dialout' "
                               "group, and group changes need a fresh login.")
                return False
            try:
                attrs = termios.tcgetattr(fd)
                attrs[0] = attrs[1] = attrs[3] = 0      # raw in, raw out, no echo
                attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
                attrs[4] = attrs[5] = termios.B115200
                attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                termios.tcflush(fd, termios.TCIOFLUSH)
            except termios.error as exc:                # not a serial device
                os.close(fd)
                self._error = f"{self.device}: {exc}"
                return False
            self._fd = fd
        self._error = None
        for target in (self._read_loop, self._write_loop):
            thread = threading.Thread(target=target, daemon=True,
                                      name=target.__name__)
            thread.start()
            self._threads.append(thread)
        return True

    def close(self) -> None:
        # Stop the column before dropping the link, never after.
        try:
            self.stop()
        except OSError:
            pass
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    # --- reading -----------------------------------------------------------
    def _read_loop(self) -> None:
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = os.read(self._fd, 4096)         # type: ignore[arg-type]
            except (BlockingIOError, InterruptedError):
                chunk = b""
            except OSError as exc:
                with self._lock:
                    self._error = f"read failed: {exc}"
                return
            if chunk:
                buffer += chunk
            frames, buffer = parse(buffer)
            if frames:
                now = time.monotonic()
                with self._lock:
                    for cmd, data in frames:
                        if cmd == CMD_MOTION and len(data) >= 8:
                            self._motion = data
                        elif cmd == CMD_STATE and len(data) >= 4:
                            self._state = data
                    self._seen_at = now
                    if self._motion:
                        self._track.append((now, _i16(self._motion, 6)))
            self._stop.wait(0.01)

    def _speed_locked(self) -> float | None:
        """Measured mm/s over the last half second. Caller holds the lock."""
        if len(self._track) < 2:
            return None
        newest = self._track[-1]
        for stamp, height in reversed(self._track):
            if newest[0] - stamp >= 0.5:
                span = newest[0] - stamp
                return round((newest[1] - height) / span, 1) if span else None
        oldest = self._track[0]
        span = newest[0] - oldest[0]
        if span < 0.15:
            return None
        return round((newest[1] - oldest[1]) / span, 1)

    def status(self) -> dict[str, Any]:
        """Everything the UI and the gate need, or ``connected: False``."""
        with self._lock:
            motion, state, seen, error = (self._motion, self._state,
                                          self._seen_at, self._error)
            intent = self._intent
            speed = self._speed_locked()
            seek_target, seek_error = self._seek, self._seek_error
            seeking = seek_target is not None
        age = (time.monotonic() - seen) if seen else None
        # No feedback is not "all clear": a column whose readings stopped is one
        # whose limit switches are no longer being reported.
        fresh = age is not None and age <= STALE_S
        out: dict[str, Any] = {
            "connected": bool(motion) and fresh,
            "device": self.device,
            "error": error,
            "age_s": None if age is None else round(age, 2),
            "stale": not fresh,
            "commanded": intent,
            "seeking": seeking,
            "seek_target_mm": seek_target,
            "seek_error": seek_error,
            "speed_mms": speed,
            "min_mm": MIN_HEIGHT_MM,
            "max_mm": MAX_HEIGHT_MM,
        }
        if motion:
            out.update({
                "height_mm": _i16(motion, 6),
                "target_mm": _i16(motion, 4),
                "mode": "position" if motion[0] else "velocity",
                "target_velocity": int.from_bytes(motion[1:2], "little", signed=True),
                "velocity": int.from_bytes(motion[2:3], "little", signed=True),
                "action": ACTIONS.get(motion[3], str(motion[3])),
                "moving": motion[3] in (3, 4),
            })
        if state:
            out.update({
                "limit_up": bool(state[0]),
                "limit_down": bool(state[1]),
                "stop_button": bool(state[2]),
                "drive_error": bool(state[3]),
            })
        return out

    # --- writing -----------------------------------------------------------
    def _send(self, frame: bytes) -> None:
        if self._fd is None:
            raise OSError("lift port is not open")
        os.write(self._fd, frame)

    def hold(self, percent: int) -> dict[str, Any]:
        """Register an intent to move. Expires after DEADMAN_S without renewal."""
        reason = gate(self.status(), percent)
        if reason:
            self.stop()
            return {"accepted": False, "reason": reason}
        with self._lock:
            self._intent = int(percent)
            self._intent_at = time.monotonic()
            self._goto = None
            if percent:
                self._zeros_left = ZERO_BURST
        return {"accepted": True, "percent": int(percent)}

    def seek(self, height_mm: int, percent: int = 30) -> dict[str, Any]:
        """Drive to an absolute height, closed-loop on the reported height.

        Not the controller's position mode: that is accepted and ignored by this
        hardware. This watches the height it is already broadcasting and stops
        itself, which also means it cannot overshoot into a limit the way an
        open-loop timed move can.
        """
        status = self.status()
        here = status.get("height_mm")
        if here is None:
            return {"accepted": False, "reason": "no height reading from the column"}
        # Clamp rather than refuse: asking to go past the end should travel as
        # far as it can, but the caller is told so it can say what happened.
        asked = int(height_mm)
        target = max(MIN_HEIGHT_MM, min(MAX_HEIGHT_MM, asked))
        delta = target - here
        if abs(delta) <= TOLERANCE_MM:
            return {"accepted": True, "already_there": True, "height_mm": here,
                    "clamped": target != asked, "target_mm": target}
        reason = gate(status, 1 if delta > 0 else -1)
        if reason:
            return {"accepted": False, "reason": reason}
        now = time.monotonic()
        with self._lock:
            self._intent, self._intent_at = 0, 0.0
            self._seek = target
            self._seek_percent = max(SLOW_PERCENT, min(MAX_PERCENT, int(percent)))
            self._seek_deadline = now + SEEK_TIMEOUT_S
            self._seek_progress_at = now
            self._seek_last_mm = here
            self._seek_error = None
        return {"accepted": True, "height_mm": here, "target_mm": target,
                "delta_mm": delta, "clamped": target != asked,
                "asked_mm": asked}

    def nudge(self, delta_mm: int, percent: int = 30) -> dict[str, Any]:
        """Move a distance in mm relative to wherever the column is now."""
        here = self.status().get("height_mm")
        if here is None:
            return {"accepted": False, "reason": "no height reading from the column"}
        return self.seek(here + int(delta_mm), percent)

    def stop(self) -> dict[str, Any]:
        """Zero the column now, repeatedly. Safe from any exit path."""
        with self._lock:
            self._intent, self._intent_at, self._goto = 0, 0.0, None
            self._seek = None
            self._zeros_left = max(self._zeros_left, ZERO_BURST)
        errors = []
        for _ in range(3):
            try:
                self._send(velocity_frame(0))
            except OSError as exc:
                errors.append(str(exc))
        return {"stopped": not errors, "errors": errors}

    def _write_loop(self) -> None:
        period = 1.0 / RATE_HZ
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._tick(started)
            except OSError as exc:
                with self._lock:
                    self._error = f"write failed: {exc}"
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _tick(self, now: float) -> None:
        with self._lock:
            goto, intent, seek = self._goto, self._intent, self._seek
            expired = self._intent_at and (now - self._intent_at) > DEADMAN_S
            if expired:
                self._intent, self._intent_at = 0, 0.0
                intent = 0
            if goto is not None:
                self._goto = None
            zeros = self._zeros_left
            if not intent and zeros:
                self._zeros_left -= 1
        if goto is not None:
            self._send(goto)
            return
        if seek is not None:
            self._seek_tick(now, seek)
            return
        if intent:
            # An interlock that trips mid-hold must stop the column, not wait
            # for the operator to let go.
            if gate(self.status(), intent):
                self.stop()
                return
            self._send(velocity_frame(intent))
        elif zeros:
            self._send(velocity_frame(0))

    def _finish_seek(self, error: str | None = None) -> None:
        with self._lock:
            self._seek = None
            self._seek_error = error
            self._zeros_left = max(self._zeros_left, ZERO_BURST)
        self._send(velocity_frame(0))

    def _seek_tick(self, now: float, target: int) -> None:
        """One step of the closed loop: compare, creep, or stop."""
        status = self.status()
        here = status.get("height_mm")
        if here is None:
            self._finish_seek("lost the height reading")
            return
        delta = target - here
        direction = 1 if delta > 0 else -1
        if abs(delta) <= TOLERANCE_MM:
            self._finish_seek(None)
            return
        reason = gate(status, direction)
        if reason:
            self._finish_seek(reason)
            return
        with self._lock:
            if abs(here - self._seek_last_mm) >= 1:
                self._seek_last_mm, self._seek_progress_at = here, now
            stalled = (now - self._seek_progress_at) > SEEK_STALL_S
            timed_out = now > self._seek_deadline
            percent = self._seek_percent
        # A column that is being commanded but is not moving is jammed, at an
        # unreported limit, or held by something. Either way, stop asking.
        if stalled:
            self._finish_seek(f"stopped moving {abs(delta)}mm from the target")
            return
        if timed_out:
            self._finish_seek("took too long to reach the target")
            return
        # Creep over the last stretch so the loop's 10 Hz sampling cannot
        # overshoot past the tolerance band at full speed.
        if abs(delta) <= SLOW_MM:
            percent = SLOW_PERCENT
        self._send(velocity_frame(direction * percent))


def gate(status: dict[str, Any], percent: int = 0) -> str | None:
    """Why the column must not move, or None. Pure, so it is testable.

    Direction-aware: a tripped upper limit blocks raising but must still allow
    lowering, or the column would be stuck at the top with no way back.
    """
    if not status.get("connected"):
        error = status.get("error")
        return error or "no feedback from the lifting column"
    if status.get("stale"):
        return "lift feedback is stale — limit switches are not being reported"
    if status.get("stop_button"):
        return "the column's stop button is pressed"
    if status.get("drive_error"):
        return "the column reports a drive error"
    if percent > 0 and status.get("limit_up"):
        return "the column is at its upper limit"
    if percent < 0 and status.get("limit_down"):
        return "the column is at its lower limit"
    here = status.get("height_mm")
    if here is not None:
        if percent > 0 and here >= MAX_HEIGHT_MM:
            return f"the column is at the {MAX_HEIGHT_MM} mm software limit"
        if percent < 0 and here <= MIN_HEIGHT_MM:
            return f"the column is at the {MIN_HEIGHT_MM} mm software limit"
    return None
