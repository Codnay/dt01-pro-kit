"""The only place in the dashboard that commands base motion.

Everything about driving this robot that could hurt someone is concentrated
here so it can be reasoned about in one screen:

* **The deadman lives in the server, not the browser.** The UI sends an intent;
  this loop keeps it alive for ``DEADMAN_S`` and then zeroes. Close the tab,
  lose WiFi, or kill the browser and the robot stops on its own.
* **Stops are repeated.** One zero packet on a flaky link is not a stop, so
  going idle sends a burst of them.
* **Zero is not a hard stop.** The vendor is explicit that a zero-speed command
  may still coast to a halt, so the emergency path sends ``move_brake`` and
  only then zeroes (API doc p.259).
* **Motion is gated.** A nonzero command is refused while the software e-stop is
  engaged, while a bumper is pressed, or while the navigator is running its own
  task — two controllers fighting over the base is how a robot ends up in a
  wall.

The controller's own documentation says ``/cmd/move`` must be re-sent at no
less than about 3 Hz or the robot stops, while this repository's notes describe
it as latching until a zero arrives. The loop is built for the pessimistic
reading of both: it re-sends at 10 Hz *and* always sends explicit zeros.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from drobot_api import DRobotError, Velocity, explain

RATE_HZ = 10.0
DEADMAN_S = 0.6          # zero the base if the browser goes quiet for this long
ZERO_BURST = 8           # zero packets sent after motion before going idle
STOP_REPEAT = 3          # zero packets sent by an explicit stop

# Ceilings the UI cannot exceed. The vendor app tops out at 1.2 m/s in high
# gear; a browser dashboard has no business anywhere near that.
MAX_LINEAR = 0.60
MAX_ANGULAR = 1.00

SPEED_PRESETS = {
    "slow":   {"linear": 0.10, "angular": 0.25},
    "normal": {"linear": 0.20, "angular": 0.45},
    "fast":   {"linear": 0.40, "angular": 0.70},
}

# Navigator states in which teleop is refused outright.
BUSY_NAV_TYPES = ("navigation", "follow_path")


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class DriveStatus:
    armed: bool = False
    linear: float = 0.0
    angular: float = 0.0
    intent_age: float | None = None
    blocked_by: str | None = None
    last_error: str | None = None
    commands_sent: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "armed": self.armed,
            "linear": round(self.linear, 3),
            "angular": round(self.angular, 3),
            "intent_age": round(self.intent_age, 2) if self.intent_age is not None else None,
            "blocked_by": self.blocked_by,
            "last_error": self.last_error,
            "commands_sent": self.commands_sent,
            "deadman_s": DEADMAN_S,
            "max_linear": MAX_LINEAR,
            "max_angular": MAX_ANGULAR,
        }


def serialized(method):
    """Keep stop/guard transitions ordered with the final velocity send."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._command_lock:
            return method(self, *args, **kwargs)
    return wrapped


class DriveLoop:
    """Streams velocity to the base at a fixed rate, with a server-side deadman."""

    def __init__(self, client: Any, gate: Callable[[], str | None]) -> None:
        """``gate`` returns None when motion is allowed, or a reason string."""
        self._client = client
        self._gate = gate
        self._lock = threading.Lock()
        self._command_lock = threading.RLock()
        self._target = Velocity()
        self._intent_at = 0.0
        self._zeros_left = 0
        self._status = DriveStatus()
        self._stop = threading.Event()
        self._protection_latched: str | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="drive", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop driving, then stop the loop. Order matters."""
        try:
            self.stop()
        finally:
            self._stop.set()

    # --- commands from the UI ---------------------------------------------
    @serialized
    def request(self, linear: float, angular: float) -> dict[str, Any]:
        """Register a movement intent. Must be repeated or the deadman fires."""
        if TELEOP_DISABLED and (float(linear) or float(angular)):
            reason = "teleop is disabled in this build (drive.TELEOP_DISABLED)"
            with self._lock:
                self._target = Velocity()
                self._status.blocked_by = reason
            self.stop()
            return {"accepted": False, "reason": reason}
        if (float(linear) or float(angular)) and self._protection_latched:
            return {"accepted": False, "reason": self._protection_latched}
        velocity = Velocity(clamp(float(linear), MAX_LINEAR),
                            clamp(float(angular), MAX_ANGULAR))
        if not velocity.is_zero():
            reason = self._safe_gate()
            if reason:
                with self._lock:
                    self._target = Velocity()
                    self._status.blocked_by = reason
                if reason.startswith("obstacle protection:"):
                    self._protection_stop(reason)
                else:
                    self.stop()
                return {"accepted": False, "reason": reason}
        with self._lock:
            self._target = velocity
            self._intent_at = time.monotonic()
            self._status.blocked_by = None
            if not velocity.is_zero():
                self._zeros_left = ZERO_BURST
        return {"accepted": True, "linear": velocity.linear, "angular": velocity.angular}

    @serialized
    def stop(self) -> dict[str, Any]:
        """Zero the base now, repeatedly. Safe to call from any exit path."""
        with self._lock:
            self._target = Velocity()
            self._intent_at = 0.0
            self._zeros_left = max(self._zeros_left, ZERO_BURST)
        errors = []
        for _ in range(STOP_REPEAT):
            try:
                self._client.stop()
            except DRobotError as exc:
                errors.append(explain(exc))
        with self._lock:
            self._status.armed = False
            self._status.linear = self._status.angular = 0.0
            self._status.last_error = errors[-1] if errors else None
        return {"stopped": not errors, "errors": errors}

    def _safe_gate(self) -> str | None:
        try:
            return self._gate()
        except Exception as exc:
            return f"obstacle protection: safety check failed ({type(exc).__name__})"

    def _protection_stop(self, reason: str) -> None:
        self._protection_latched = reason + "; release controls and explicitly release the brake to acknowledge"
        self.emergency_stop()

    @serialized
    def emergency_stop(self) -> dict[str, Any]:
        """Lock the brakes, then zero. A zero-speed command alone still coasts."""
        result: dict[str, Any] = {"braked": False, "errors": []}
        try:
            self._client.set_brake(True)
            result["braked"] = True
        except DRobotError as exc:
            result["errors"].append(explain(exc))
        stopped = self.stop()
        result["errors"].extend(stopped["errors"])
        return result

    @serialized
    def release_brake(self) -> dict[str, Any]:
        if self._protection_latched:
            # Explicit release is the acknowledgement; never resume a held
            # joystick automatically after an obstacle disappears.
            self.stop()
        try:
            self._client.set_brake(False)
            self._protection_latched = None
            return {"released": True}
        except DRobotError as exc:
            return {"released": False, "error": explain(exc)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = self._status
            age = (time.monotonic() - self._intent_at) if self._intent_at else None
            status.intent_age = age
            result = status.as_dict()
            result["protection_latched"] = self._protection_latched
            return result

    # --- the loop ----------------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / RATE_HZ
        while not self._stop.is_set():
            started = time.monotonic()
            self._tick(started)
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, period - elapsed))

    @serialized
    def _tick(self, now: float) -> None:
        with self._lock:
            target = self._target
            stale = self._intent_at and (now - self._intent_at) > DEADMAN_S
            if stale and not target.is_zero():
                target = self._target = Velocity()
                self._zeros_left = max(self._zeros_left, ZERO_BURST)
                self._status.blocked_by = "deadman: no command from the browser"
            if target.is_zero():
                if self._zeros_left <= 0:
                    self._status.armed = False
                    self._status.linear = self._status.angular = 0.0
                    return                       # idle: stop talking to the base
                self._zeros_left -= 1
            else:
                self._zeros_left = ZERO_BURST

        if not target.is_zero():
            reason = self._safe_gate()
            if reason:
                with self._lock:
                    self._target = Velocity()
                    self._status.blocked_by = reason
                target = Velocity()
                if reason.startswith("obstacle protection:"):
                    self._protection_stop(reason)
                    return

        try:
            self._client.move(target)
        except DRobotError as exc:
            with self._lock:
                self._status.last_error = explain(exc)
            return
        with self._lock:
            self._status.last_error = None
            self._status.commands_sent += 1
            self._status.armed = not target.is_zero()
            self._status.linear, self._status.angular = target.linear, target.angular


# Master teleop lockout. While True the dashboard is read-only: every panel
# and all telemetry still work, but W/A/S/D and the on-screen pad cannot
# command motion. Set to True to lock driving out again.
TELEOP_DISABLED = False


def lift_gate(snapshot: dict[str, Any]) -> str | None:
    """Return why the lifting column must not move, or None if it may.

    A separate gate from ``motion_gate`` because the hazards are different.
    Bumpers and stale proximity data say nothing about whether it is safe to
    raise a column, and requiring them would make the lift unusable on firmware
    whose bumper endpoint is dead. What matters is that the base is not moving
    and nobody has called a stop:

    * **The base must be still.** Raising or lowering a load while driving is
      how a robot tips or snags what it is carrying.
    * **A declared stop means everything.** If the operator engaged the software
      e-stop, that is not "stop the wheels only".

    The column protects its own travel limits in hardware, so this gate is
    about the rest of the robot, not the ends of the stroke.
    """
    if snapshot.get("reachable") is False:
        return "the controller is not answering — check the link to the robot"
    if snapshot.get("brake", {}).get("engaged"):
        return "software emergency stop is engaged"
    if (snapshot.get("bumper") or {}).get("estop"):
        return "chassis emergency stop is active"
    drive = snapshot.get("drive") or {}
    if drive.get("armed") or drive.get("linear") or drive.get("angular"):
        return "the base is moving — stop it before moving the lift"
    nav_type = (snapshot.get("nav") or {}).get("nav_status_type")
    if nav_type in BUSY_NAV_TYPES:
        return f"the navigator is running a task ({nav_type}) — stop it first"
    # Nobody driving means nobody may drive. This is what makes an expired
    # lease safe: a holder whose tab closed stops the robot through the normal
    # gate rather than needing anything to notice and clean up.
    if not (snapshot.get("control") or {}).get("held"):
        return "nobody has taken control of the robot"
    return None


def motion_gate(snapshot: dict[str, Any]) -> str | None:
    """Return why motion must be refused right now, or None if it is allowed.

    Kept as a free function so it can be tested against a plain dict without a
    robot, a socket, or a running loop.
    """
    if TELEOP_DISABLED:
        return "teleop is disabled in this build (drive.TELEOP_DISABLED)"
    if snapshot.get("reachable") is False:
        return "the controller is not answering — check the link to the robot"
    if snapshot.get("brake", {}).get("engaged"):
        return "software emergency stop is engaged"
    bumper = snapshot.get("bumper") or {}
    if bumper.get("estop"):
        return "chassis emergency stop is active"
    if bumper.get("front") or bumper.get("rear"):
        return "a bumper is pressed"
    # Absent readings are not "all clear". On firmware that serves neither the
    # bumper endpoint nor the documented push path, `bumper` stays {} and every
    # check above passes by default — two of the four interlocks silently blind.
    # Refuse instead: not knowing whether a bumper is pressed is a reason not to
    # drive, not a reason to assume it is not.
    if snapshot.get("safety_stale"):
        return ("no bumper or e-stop reading from the robot — interlocks are blind "
                "(check the notice feed under Setup)")
    nav_type = (snapshot.get("nav") or {}).get("nav_status_type")
    if nav_type in BUSY_NAV_TYPES:
        return f"the navigator is running a task ({nav_type}) — stop it first"
    # Nobody driving means nobody may drive. This is what makes an expired
    # lease safe: a holder whose tab closed stops the robot through the normal
    # gate rather than needing anything to notice and clean up.
    if not (snapshot.get("control") or {}).get("held"):
        return "nobody has taken control of the robot"
    obstacles = snapshot.get("obstacles") or {}
    protection = obstacles.get("protection") or {}
    # Unavailable obstacle feedback is a visible warning, not a manual-drive
    # interlock. A detected hazard still blocks, even if another sensor is down.
    if obstacles.get("alarm") or protection.get("blocked_by"):
        reason = protection.get("blocked_by") or next(iter(obstacles.get("reasons") or []), "obstacle detected")
        return "obstacle protection: " + reason
    return None
