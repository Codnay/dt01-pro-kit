"""Who is allowed to command the robot — exactly one browser at a time.

The dashboard has no login, and anyone who can reach it can open it. That is
fine for watching; it is not fine for driving. Two people holding W at once is
not a merge conflict, it is a robot doing something neither of them asked for.

So commands need a **lease**. One client holds it, everyone else watches. The
holder keeps it alive with a heartbeat; if their tab closes, their laptop
sleeps, or the link drops, the lease expires on its own and the robot becomes
available to someone else. Nothing has to be cleaned up by hand.

Two decisions worth stating, because both are safety-relevant:

* **Stopping is never gated.** An observer who can see the robot about to hit
  something must be able to stop it, lease or no lease. The emergency stop and
  every explicit stop path stay open to everyone; only the commands that *start*
  motion are held behind the lease.
* **Losing the lease stops the robot.** Expiry is not just bookkeeping — the
  motion gate refuses while nobody holds control, so a holder who vanishes
  leaves a stationary robot rather than one still coasting on its last command.

The timeout is deliberately generous relative to the heartbeat. This dashboard
is used over a hotspot from another continent on a link that has already been
observed to drop; a holder should not lose control to one bad second. Losing it
is safe (the robot stops), but it is disruptive, so the tolerance is for the
network rather than against it.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

# The browser beats every HEARTBEAT_S; the lease survives roughly three missed
# beats. Keep the ratio: a timeout close to the heartbeat interval makes a live
# holder lose control on ordinary jitter.
HEARTBEAT_S = 2.5
TIMEOUT_S = 8.0

MAX_NAME = 40


def clean_name(name: Any) -> str:
    """A display name safe to show to every other viewer."""
    text = " ".join(str(name or "").split())
    return text[:MAX_NAME] or "someone"


class ControlLease:
    """Authoritative "who may command the robot" lease. Thread-safe.

    ``clock`` is injectable so expiry is testable without sleeping.
    """

    def __init__(self, timeout_s: float = TIMEOUT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._timeout = float(timeout_s)
        self._holder: str | None = None
        self._name = ""
        self._taken_at = 0.0
        self._seen_at = 0.0
        # Set whenever the holder changes for any reason. The server drains it
        # and stops the robot: a handover must never inherit the previous
        # operator's motion.
        self._changed = False

    # --- internals ---------------------------------------------------------
    def _expire_locked(self, now: float) -> None:
        if self._holder is not None and now - self._seen_at > self._timeout:
            self._holder, self._name = None, ""
            self._changed = True

    # --- queries -----------------------------------------------------------
    def holder(self) -> str | None:
        with self._lock:
            self._expire_locked(self._clock())
            return self._holder

    def is_holder(self, client: str | None) -> bool:
        """True only for the live holder. A blank client is never the holder."""
        if not client:
            return False
        with self._lock:
            self._expire_locked(self._clock())
            return self._holder == client

    def status(self) -> dict[str, Any]:
        """The shape every browser renders, and the motion gate reads."""
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._holder is None:
                return {"held": False, "holder": None, "name": "",
                        "held_for_s": 0.0, "expires_in_s": 0.0,
                        "heartbeat_s": HEARTBEAT_S}
            return {
                "held": True,
                "holder": self._holder,
                "name": self._name,
                "held_for_s": round(now - self._taken_at, 1),
                "expires_in_s": round(max(0.0, self._timeout - (now - self._seen_at)), 1),
                "heartbeat_s": HEARTBEAT_S,
            }

    def take_changed(self) -> bool:
        """Drain the "holder changed" flag. True at most once per change."""
        with self._lock:
            changed, self._changed = self._changed, False
            return changed

    # --- transitions -------------------------------------------------------
    def take(self, client: str, name: Any = "",
             force: bool = False) -> tuple[bool, str]:
        """Claim control. Returns ``(took_it, reason_if_not)``.

        A live holder is not displaced unless ``force`` is set. Stealing is
        deliberate rather than automatic: silently preempting whoever is driving
        is how two people end up believing they are in control.
        """
        if not client:
            return False, "no client id"
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._holder == client:            # already ours — just renew
                self._seen_at = now
                self._name = clean_name(name) or self._name
                return True, ""
            if self._holder is not None and not force:
                return False, f"{self._name} is controlling the robot"
            self._holder = client
            self._name = clean_name(name)
            self._taken_at = self._seen_at = now
            self._changed = True
            return True, ""

    def heartbeat(self, client: str) -> bool:
        """Renew the lease. False if the caller is not (still) the holder."""
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._holder != client or not client:
                return False
            self._seen_at = now
            return True

    def release(self, client: str) -> bool:
        """Give up control. Only the holder can; releasing twice is harmless."""
        with self._lock:
            self._expire_locked(self._clock())
            if self._holder != client or not client:
                return False
            self._holder, self._name = None, ""
            self._changed = True
            return True
