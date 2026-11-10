"""Transport-neutral take-control protocol for Quest teleoperation.

Layered on top of the raw controller wire format in :mod:`arms.xr_wire`,
this module adds what a *remote* operator link needs beyond per-hand poses:

* a combined two-hand frame carrying the operator's take-control identity
  (``cid``) and epoch (``ep``), so the follower can tell whose input it is;
* the take-control lease messages (claim / heartbeat / release); and
* :class:`ControlLease`, the follower-side state machine that decides which
  operator currently holds the arms and when a stale holder times out.

None of this is transport-specific — the same frames ride LiveKit data channels
today, and rode Agora RTM message channels before that. A transport module owns
only its own connection setup and topic/channel names.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import numpy as np

from arms.xr_wire import (  # noqa: F401
    BUTTON_NAMES,
    HANDS,
    decode_button,
    decode_xr,
    decode_xr_obj,
    encode_button,
    encode_xr,
)

# Reserved top-level keys carried alongside the per-hand entries in an XR frame.
# ``cid`` = the publishing operator's stable client/session id; ``ep`` = the
# take-control epoch the operator currently believes it holds. The follower gates
# on these; decoders MUST skip them when iterating hands (see ``decode_xr_frame``).
FRAME_CID_KEY = "cid"
FRAME_EPOCH_KEY = "ep"


# -- Combined two-hand frame (one message per tick) ---------------------------


def encode_xr_frame(
    per_hand: dict[str, tuple], *, cid: str = "", epoch: int = 0, seq: int | None = None
) -> str:
    """Encode both hands into one JSON string.

    ``per_hand`` maps hand -> ``(transform, grip, trigger, joystick_y)`` (the
    same tuple ``encode_xr`` takes). ``cid``/``epoch`` tag the frame with the
    publishing operator's take-control identity so the follower can enforce a
    single active controller. Returns a str (transports carry string payloads).

    ``seq`` is a monotonic per-tick counter stamped on BOTH hands (they share one
    tick). The follower latches it via ``get_seq()`` so it can DEDUP any
    re-sent state and count real transit drops as seq gaps. Without it the robot
    counts every duplicate as a fresh frame -- inflating "frames received" and making the
    liveness heartbeat never go stale even when the source has stopped. Omit
    (``None``) for legacy behaviour; the follower falls back to per-datagram.
    """
    payload: dict[str, object] = {
        hand: json.loads(encode_xr(*vals)) for hand, vals in per_hand.items()
    }
    if seq is not None:
        seq_int = int(seq)
        for hand in per_hand:
            sub = payload.get(hand)
            if isinstance(sub, dict):
                sub["seq"] = seq_int
    payload[FRAME_CID_KEY] = str(cid)
    payload[FRAME_EPOCH_KEY] = int(epoch)
    return json.dumps(payload, separators=(",", ":"))


def decode_xr_frame_obj(obj: dict) -> dict[str, tuple]:
    """Control-path decode from an ALREADY-PARSED combined frame.

    Lets a caller that must also read the frame's metadata (see
    :func:`decode_xr_frame_meta_obj`) pay for a single ``json.loads`` instead of
    one per consumer.
    """
    out: dict[str, tuple] = {}
    for hand, sub in obj.items():
        if hand not in HANDS:  # skip cid/ep and any future reserved keys
            continue
        out[hand] = decode_xr_obj(sub)
    return out


def decode_xr_frame(message: str | bytes) -> dict[str, tuple]:
    """Decode a combined frame back to ``{hand: (transform, grip, trigger, joystick_y)}``."""
    return decode_xr_frame_obj(json.loads(message))


def frame_control_id(message: str | bytes) -> tuple[str, int]:
    """Extract ``(cid, epoch)`` from an XR-frame or button message.

    Missing fields (legacy leaders that predate take-control) decode to
    ``("", 0)``, which the follower treats as the single legacy operator.
    """
    try:
        obj = json.loads(message)
    except Exception:
        return ("", 0)
    if not isinstance(obj, dict):
        return ("", 0)
    cid = obj.get(FRAME_CID_KEY, "")
    ep = obj.get(FRAME_EPOCH_KEY, 0)
    try:
        return (str(cid), int(ep))
    except (TypeError, ValueError):
        return ("", 0)


# -- Take-control lease messages (control channel) ----------------------------

CONTROL_CLAIM = "claim"
CONTROL_HEARTBEAT = "heartbeat"
CONTROL_RELEASE = "release"
# Follower -> leaders. The ONLY authoritative statement of who holds the arms and
# what epoch a new claim must beat. Leaders keep a page-local epoch counter that
# resets to 0 on every reload, while the follower's epoch floor survives a
# release (see ControlLease.release) -- so without this broadcast a reloaded
# browser claims epoch 1 against a floor of N and loses, silently, until the
# operator has clicked Take Control N times. Sent on join, on every lease change
# and as a slow keepalive.
CONTROL_STATE = "state"

# Leader -> follower -> leader round trip, for MEASURING THE LINK.
#
# There is no shared clock between a Quest browser and the robot, and nothing to
# make one: NTP on the headset is not ours to set, and the pose stamp
# (``Date.now()`` in the browser) minus the robot's receive time is dominated by
# skew, not transit. That difference ran ~-480 ms median historically -- a
# *negative* latency, which is the giveaway that it never measured transit at
# all.
#
# A round trip needs no shared clock. The leader stamps a ping with its own
# monotonic clock, the follower echoes the id back untouched, and the leader
# subtracts on the SAME clock it started on. Skew cancels exactly, because it
# appears once with each sign. The follower never reads or trusts a leader
# timestamp, so there is nothing left to be wrong about.
#
# One-way delay is reported as RTT/2, which assumes a symmetric path. On a
# satellite or a heavily asymmetric uplink that assumption is wrong, so RTT is
# the number to alert on and the one-way figure is a convenience.
CONTROL_PING = "ping"
CONTROL_PONG = "pong"
CONTROL_KINDS = (
    CONTROL_CLAIM,
    CONTROL_HEARTBEAT,
    CONTROL_RELEASE,
    CONTROL_STATE,
    CONTROL_PING,
    CONTROL_PONG,
)

# Key carrying the leader's opaque ping id. Echoed byte-for-byte; the follower
# assigns it no meaning, which is what keeps the measurement the leader's alone.
PING_ID_KEY = "pid"


def encode_button_msg(name: str, *, cid: str = "", epoch: int = 0) -> str:
    """Encode a button press tagged with the operator's take-control identity."""
    return json.dumps(
        {"name": str(name), "cid": str(cid), "ep": int(epoch)},
        separators=(",", ":"),
    )


def encode_control(kind: str, *, cid: str, epoch: int, name: str = "") -> str:
    """Encode a control-lease message for the ``ctl`` topic."""
    return json.dumps(
        {"kind": str(kind), "cid": str(cid), "ep": int(epoch), "name": str(name)},
        separators=(",", ":"),
    )


def encode_control_state(*, holder_cid: str, epoch: int, name: str = "") -> str:
    """Encode the follower's authoritative lease state for the ``ctl`` topic.

    ``holder_cid`` is the current holder's cid, or ``""`` when the lease is
    vacant. ``epoch`` is the floor a new claim must strictly beat -- it is
    retained across a release, which is exactly the value a freshly loaded
    leader page cannot otherwise know. Deliberately reuses the
    ``{kind, cid, ep, name}`` shape so :func:`decode_control` needs no change and
    older leaders ignore it harmlessly.
    """
    return json.dumps(
        {
            "kind": CONTROL_STATE,
            "cid": str(holder_cid or ""),
            "ep": int(epoch),
            "name": str(name),
        },
        separators=(",", ":"),
    )


def encode_pong(ping_id: str) -> str:
    """Encode the follower's echo of a leader ping.

    Carries the id and nothing else that matters: no timestamp, because a
    follower clock reading would reintroduce exactly the skew this round trip
    exists to avoid.
    """
    return json.dumps(
        {
            "kind": CONTROL_PONG,
            "cid": "",
            "ep": 0,
            "name": "",
            PING_ID_KEY: str(ping_id),
        },
        separators=(",", ":"),
    )


def ping_id(message: str | bytes) -> str:
    """Return the opaque ping id on a ping/pong message, or ``""``.

    Parsed separately from :func:`decode_control` so the lease decoder keeps its
    narrow ``{kind, cid, ep, name}`` contract, which both transports and every
    older leader already depend on.
    """
    try:
        obj = json.loads(message)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(obj, dict):
        return ""
    return str(obj.get(PING_ID_KEY, "") or "")


def decode_control(message: str | bytes) -> dict:
    """Decode a control-lease message to ``{kind, cid, ep, name}`` (best-effort)."""
    try:
        obj = json.loads(message)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    try:
        return {
            "kind": str(obj.get("kind", "")),
            "cid": str(obj.get("cid", "")),
            "ep": int(obj.get("ep", 0)),
            "name": str(obj.get("name", "")),
        }
    except (TypeError, ValueError):
        return {}


# -- Single-controller arbitration (pure, unit-testable) ----------------------

# Default duration of the post-handoff suppression window. Kept here (not in the
# reader) so the wire contract and its enforcement stay in one place.
DEFAULT_HANDOFF_SUPPRESS_SEC = 0.75

# If the active holder stops being seen (no XR frames AND no ctl heartbeats) for
# this long, the lease auto-vacates so a NEW operator can take over — even one
# whose epoch counter reset lower (e.g. a browser that reloaded). Without this a
# holder that disconnects without releasing would latch its epoch forever and
# silently block every future (lower-epoch) operator. Must comfortably exceed the
# browser heartbeat interval (~0.5 s) so a live holder never expires.
DEFAULT_HOLDER_TIMEOUT_SEC = 3.0


class ControlLease:
    """Authoritative "who is driving" lease for exactly-one active controller.

    Preemptive take-control: an operator ``(cid, epoch)`` becomes the holder when
    it carries a higher epoch than the current holder (or an equal epoch with a
    lexicographically greater cid — a deterministic tie-break so the follower and
    every browser converge on the same winner regardless of arrival order). A
    handoff arms a suppression window; while it is active the follower forces the
    teleop-facing controls neutral so the incoming operator must re-arm and
    control never jumps to their hand pose.

    Thread-safe and free of any Agora/SDK dependency so it can be unit-tested and
    reused by other transports. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        *,
        suppress_sec: float = DEFAULT_HANDOFF_SUPPRESS_SEC,
        holder_timeout_sec: float = DEFAULT_HOLDER_TIMEOUT_SEC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._suppress_sec = float(suppress_sec)
        self._holder_timeout_sec = float(holder_timeout_sec)
        self.active_cid: str | None = None
        self.active_epoch: int = -1
        self.active_name: str = ""
        self._suppress_until: float = 0.0
        self._last_seen: float = 0.0  # last time the active holder was confirmed
        # Set when a holder is vacated by timeout; drained by take_expired() so
        # the follower can log the expiry (distinct from a voluntary handoff).
        self._last_expired: tuple[str, int, str] | None = None

    def _maybe_expire(self, now: float) -> bool:
        """Vacate a stale holder (caller holds the lock). Returns True if expired.

        Resets the epoch floor to -1 so a fresh, lower-epoch operator can win —
        the dead holder's epoch must not keep blocking newcomers.
        """
        if self.active_cid is None:
            return False
        if now - self._last_seen <= self._holder_timeout_sec:
            return False
        self._last_expired = (self.active_cid, self.active_epoch, self.active_name)
        self.active_cid = None
        self.active_epoch = -1
        self.active_name = ""
        return True

    def take_expired(self) -> tuple[str, int, str] | None:
        """Return-and-clear the holder most recently vacated by timeout, if any."""
        with self._lock:
            expired = self._last_expired
            self._last_expired = None
            return expired

    def check_expiry(self) -> None:
        """Force a liveness check (e.g. from a watchdog) so a holder that goes
        silent with no successor still expires and gets logged."""
        with self._lock:
            self._maybe_expire(self._clock())

    @staticmethod
    def _wins(cid: str, ep: int, cur_cid: str | None, cur_ep: int) -> bool:
        # No current holder (initial state or a vacated lease after release):
        # require a STRICTLY higher epoch. This stops the just-released holder's
        # still-in-flight frames — or any observer — from re-acquiring at the same
        # epoch. The epoch is retained across a release precisely so it must be
        # out-bid to take control again.
        if cur_cid is None:
            return ep > cur_ep
        # Held lease: a higher epoch preempts; an equal epoch is resolved by the
        # lexicographically greater cid (deterministic race tie-break).
        if ep != cur_ep:
            return ep > cur_ep
        return cid > cur_cid

    def is_holder(self, cid: str, ep: int, name: str = "") -> bool:
        """True if ``(cid, ep)`` may drive the robot; latches a handoff (and opens
        the suppression window) if it preempts the current holder. Losing
        publishers return False and change nothing. Callers that need to detect
        the handoff itself can compare :meth:`snapshot` before/after.
        """
        with self._lock:
            now = self._clock()
            expired = self._maybe_expire(now)
            if not expired and cid == self.active_cid and ep == self.active_epoch:
                self._last_seen = now  # holder is alive; refresh liveness
                return True
            if not self._wins(cid, ep, self.active_cid, self.active_epoch):
                return False
            self.active_cid = cid
            self.active_epoch = ep
            self.active_name = name
            self._last_seen = now
            self._suppress_until = now + self._suppress_sec
            return True

    def release(self, cid: str, ep: int) -> bool:
        """Drop the lease if ``(cid, ep)`` is the current holder. Epoch stays
        monotonic so a later claim at a higher epoch still wins."""
        with self._lock:
            if cid == self.active_cid and ep == self.active_epoch:
                self.active_cid = None
                self.active_epoch = ep
                self.active_name = ""
                self._suppress_until = self._clock() + self._suppress_sec
                return True
            return False

    def suppressed(self) -> bool:
        """True while the post-handoff re-arm window is open."""
        with self._lock:
            return self._clock() < self._suppress_until

    def snapshot(self) -> tuple[str | None, int, str]:
        with self._lock:
            return (self.active_cid, self.active_epoch, self.active_name)

    def epoch_floor(self) -> int:
        """Epoch a new claim must strictly beat, held or not.

        Equal to ``snapshot()[1]``; named because when the lease is VACANT that
        number is still load-bearing (``_wins`` requires ``ep > cur_ep``) and
        reading it off a field called "active_epoch" reads like a no-op.
        """
        with self._lock:
            return self.active_epoch


def decode_xr_frame_meta(message: str | bytes) -> dict[str, dict]:
    """Decode a combined frame to per-hand dicts *with* operator metadata.

    Unlike :func:`decode_xr_frame` (the control path, which returns only the
    ``(transform, grip, trigger, joystick_y)`` tuple the IK needs), this keeps
    the leader-side fields the follower needs to log operator telemetry and do
    one-sided RCA:

        ``stamp``  leader capture time (s) -- ``recv - stamp`` is transport latency
        ``seq``    monotonic per-tick counter -- gaps mean frames dropped in transit
        ``valid``  whether controller tracking was live for this frame

    Returns ``{hand: {transform, grip, trigger, joystick_y, stamp, seq, valid}}``
    where ``transform`` is a 4x4 ndarray (or ``None`` when tracking was lost) and
    ``stamp``/``seq`` are ``None`` when the leader did not send them (older JS).
    """
    return decode_xr_frame_meta_obj(json.loads(message))


def decode_xr_frame_meta_obj(obj: dict) -> dict[str, dict]:
    """Metadata decode from an ALREADY-PARSED combined frame.

    See :func:`decode_xr_frame_obj` -- the follower taps every frame for both
    control and telemetry, so it parses once and calls both object forms.
    """
    out: dict[str, dict] = {}
    for hand, sub in obj.items():
        if not isinstance(sub, dict):
            continue
        flat = sub.get("transform", [])
        transform = None
        if sub.get("valid") and isinstance(flat, list) and len(flat) == 16:
            transform = np.asarray(flat, dtype=np.float64).reshape(4, 4)
        seq = sub.get("seq")
        stamp = sub.get("stamp")
        out[hand] = {
            "transform": transform,
            "grip": float(sub.get("grip", 0.0)),
            "trigger": float(sub.get("trigger", 0.0)),
            "joystick_y": float(sub.get("joystick_y", 0.0)),
            "stamp": float(stamp) if stamp else None,
            "seq": int(seq) if seq is not None else None,
            "valid": bool(sub.get("valid", False)),
        }
    return out
