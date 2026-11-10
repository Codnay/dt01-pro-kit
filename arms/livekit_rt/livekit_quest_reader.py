"""Quest XR input with a disposable network worker and parent-owned safety.

The SDK lives in a dedicated process. The parent keeps leases, fresh input and
callbacks; an independent watchdog can replace a hung SDK without touching CAN.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from arms.livekit_rt.livekit_bridge import (
    TOPIC_BTN,
    TOPIC_CTL,
    TOPIC_XR,
    get_identity,
    get_room,
    get_static_token,
    get_url,
)
from arms.livekit_rt.transport_supervisor import WorkerTransport
from arms.logging_setup import get_logger
from arms.xr_control import (
    CONTROL_CLAIM,
    CONTROL_PING,
    CONTROL_PONG,
    CONTROL_RELEASE,
    CONTROL_STATE,
    DEFAULT_HOLDER_TIMEOUT_SEC,
    ControlLease,
    decode_button,
    decode_control,
    decode_xr_frame_meta_obj,
    decode_xr_frame_obj,
    encode_control_state,
    encode_pong,
    frame_control_id,
    ping_id,
)

def _meta_record_frame(*_a: Any, **_k: Any) -> None:
    return None


def _meta_record_rtt(*_a: Any, **_k: Any) -> None:
    return None


log = get_logger("livekit-rtm")
# Same component name the Agora reader uses for lease events, so take-control
# logs read identically on both transports (PIPER_LOG_ONLY=take-control still
# isolates them) and an A/B diff is not confused by a renamed logger.
ctl_log = get_logger("take-control")

HANDS = ("left", "right")

# On a handoff the
# teleop-facing trigger is forced low so the previous operator deactivates and
# the incoming one must deliberately re-squeeze; the dead-man logic snapshots a
# fresh origin on that press edge, so control never jumps to the new operator's
# hand pose. See threads/quest_reader.py rising-edge handling.
#
# NOTE: this is a *suppression* window, not a deadline. For 0.75 s after a
# handoff the reader reports trigger=0 / pose=None no matter what the operator
# does; after it closes, the next trigger squeeze arms teleop whenever it
# happens. There is nothing to miss and no timer to beat.
HANDOFF_SUPPRESS_SEC = 0.75

DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_STATS_PERIOD_S = 5.0
DEFAULT_INPUT_TIMEOUT_S = 0.25

# How long a lease holder may heartbeat without sending a single pose frame
# before the reader calls it out. Heartbeats are reliable and pose is lossy, so
# "ctl flowing, xr=0" is the signature of a leader whose WebXR loop never
# started (or died) -- a failure that is otherwise invisible, because from the
# arm's side it looks exactly like an operator standing still.
SILENT_HOLDER_WARN_SEC = 3.0
SILENT_HOLDER_WARN_PERIOD_S = 10.0

# A backwards seq jump larger than this is read as "the leader restarted", not
# "this packet is late". Real reordering on a lossy channel spans a handful of
# frames; at ~45 Hz this is ~11 s, far beyond any plausible reorder window, and
# without it a leader restart would deafen the follower permanently.
SEQ_RESET_THRESHOLD = 500

# Aggregate floor between pongs. A leader pings at ~2 Hz and a cell rarely has
# more than a handful of browsers open, so this is far above normal traffic and
# exists only so a runaway page cannot turn the reliable ctl topic into a flood
# that competes with the lease.
PONG_MIN_INTERVAL_S = 0.05


def add_livekit_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--livekit-*`` flags (only meaningful with ``--quest-source livekit``)."""
    group = parser.add_argument_group("LiveKit XR control (experimental)")
    group.add_argument(
        "--livekit-url",
        default="",
        help="LiveKit server URL (wss://...). Overrides LIVEKIT_URL.",
    )
    group.add_argument(
        "--livekit-room",
        default="",
        help="LiveKit room to join. Overrides $LIVEKIT_ROOM, which itself "
        "defaults to $ROBOT_ID. Both ends of the link must use the same name.",
    )
    group.add_argument(
        "--livekit-identity",
        default="",
        help="Follower participant identity. Overrides LIVEKIT_IDENTITY.",
    )
    group.add_argument(
        "--livekit-connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_S,
        help="Deadline for each connection/resume attempt; retries continue offline (default: 15).",
    )
    group.add_argument(
        "--livekit-input-timeout",
        type=float,
        default=DEFAULT_INPUT_TIMEOUT_S,
        help="Seconds to retain the last valid controller input while connected "
        "(default: 0.25). Applies to pose, trigger, grip and joystick.",
    )
    group.add_argument(
        "--livekit-stats-sec",
        type=float,
        default=DEFAULT_STATS_PERIOD_S,
        help=(
            "Seconds between the terminal link-stats line "
            "(rx rates, gated frames, seq gaps). 0 disables. Default: 5."
        ),
    )


class LiveKitQuestReader:
    """MetaQuestReader-compatible reader fed by LiveKit data packets."""

    def __init__(
        self,
        *,
        url: str = "",
        token: str = "",
        room: str = "",
        identity: str = "",
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
        stats_period_s: float = DEFAULT_STATS_PERIOD_S,
        input_timeout_s: float = DEFAULT_INPUT_TIMEOUT_S,
    ) -> None:
        self._input_timeout_s = float(input_timeout_s)
        if not math.isfinite(self._input_timeout_s) or self._input_timeout_s <= 0:
            raise ValueError("LiveKit input timeout must be finite and positive")
        self._url = url or get_url()
        self._room_name = room or get_room()
        self._identity = identity or get_identity()
        # A pre-minted LIVEKIT_TOKEN is fixed for the life of the process; with
        # key+secret we mint a fresh one per connect attempt instead (see
        # _fresh_token), which is what lets a rejoin work after the default 6h
        # TTL has passed.
        self._static_token = token or get_static_token()
        self._connect_timeout = max(1.0, float(connect_timeout))
        self._stats_period_s = float(stats_period_s)

        self._lock = threading.Lock()
        # hand -> (transform, grip, trigger, joystick_y)
        self._state: dict[str, tuple] = {hand: (None, 0.0, 0.0, 0.0) for hand in HANDS}
        self._seq: dict[str, int | None] = {hand: None for hand in HANDS}
        self._button_callbacks: dict[str, list[Callable[[], None]]] = {}

        self._lease = ControlLease(
            suppress_sec=HANDOFF_SUPPRESS_SEC,
            holder_timeout_sec=max(DEFAULT_HOLDER_TIMEOUT_SEC, self._input_timeout_s),
        )
        # Set whenever the lease changes (grant / release / expiry) or a new
        # operator joins; the room loop publishes the authoritative state on the
        # next tick. A browser cannot know the epoch floor any other way -- its
        # own counter resets to 0 on reload while the floor survives a release.
        self._lease_announce_due = True
        self._last_lease_announce = 0.0

        # -- Link counters (all read under _lock by the stats line) -------------
        # Downstream "N pose msgs/s" cannot reveal a stalled link on its own, so
        # these are incremented in the packet handler BEFORE the take-control
        # gate. That keeps "arriving but gated" distinguishable from "subscription
        # went deaf" -- two failures that look identical from the arm's side.
        self._rx_xr = 0
        self._rx_btn = 0
        self._rx_ctl = 0
        self._rx_unknown = 0
        self._decode_errors = 0
        self._dropped_xr_frames = 0
        self._seq_gaps = 0
        self._last_seq_seen: int | None = None
        self._last_applied_seq: int | None = None
        self._reordered = 0
        self._last_message = 0.0
        self._ever_received_message = False
        self._last_drop_log = 0.0
        self._last_stats_counts = (0, 0, 0)
        # Leader-reported health, piggybacked on ctl heartbeats (see
        # _note_leader_diag). Empty for leaders that predate the extension.
        self._leader_diag: dict[str, Any] = {}
        self._last_ctl = 0.0
        self._last_silent_warn = 0.0
        self._last_pong = 0.0
        self._pongs = 0
        self._pongs_dropped = 0

        self._stop = threading.Event()
        self._input_key = None
        self._transport_state = "connecting"
        self._epoch_floor = -1
        self._lease_epoch_highwater = -1
        self._resume_cid = ""
        self._pose_at = {hand: 0.0 for hand in HANDS}
        self._packet_received_at = None
        self._input_revision = 0
        # Configuration errors fail clearly; network failures retry indefinitely.
        self._fresh_token()
        self._transport = WorkerTransport(
            self._url,
            self._room_name,
            self._identity,
            self._fresh_token,
            connect_timeout=self._connect_timeout,
        )
        self._transport.start()
        self._thread = threading.Thread(
            target=self._run_loop, name="livekit-quest-reader", daemon=True
        )
        self._thread.start()
        log.info(
            "connection recovery started: room=%s identity=%s input_timeout_s=%s",
            self._room_name,
            self._identity,
            self._input_timeout_s,
        )

    def _clear_input(self) -> None:
        with self._lock:
            self._state = {hand: (None, 0.0, 0.0, 0.0) for hand in HANDS}
            self._pose_at = {hand: 0.0 for hand in HANDS}
            self._seq = {hand: None for hand in HANDS}
            self._last_applied_seq = self._last_seq_seen = None
            self._input_revision += 1

    def _refresh_transport(self) -> dict:
        snapshot = self._transport.snapshot()
        key = (snapshot["generation"], snapshot["connection_epoch"])
        state = snapshot["state"]
        if key != self._input_key or state != self._transport_state:
            # Keep the floor even if the lease times out during an outage. Old
            # reliable messages and cached XR can never renew the former lease.
            cid, epoch, _ = self._lease.snapshot()
            self._epoch_floor = max(self._epoch_floor, self._lease_epoch_highwater)
            if cid:
                self._resume_cid = cid
                self._epoch_floor = max(self._epoch_floor, epoch)
                self._lease.release(cid, epoch)
            self._clear_input()
            self._input_key, self._transport_state = key, state
            self._lease_announce_due = True
            log.info("operator connection: %s (%s)", state, snapshot["last_error"])
        return snapshot

    def _run_loop(self) -> None:
        last_stats = time.monotonic()
        while not self._stop.is_set():
            try:
                self._refresh_transport()
                for event in self._transport.events():
                    snapshot = self._refresh_transport()
                    key = (event["generation"], event["connection_epoch"])
                    if snapshot["state"] != "connected" or key != self._input_key:
                        continue
                    if event["kind"] == "data":
                        if not 0 <= time.monotonic() - event["at"] <= 0.25:
                            continue
                        self._packet_received_at = event["at"]
                        try:
                            self._handle_packet(event["data"].encode(), event["topic"])
                        finally:
                            self._packet_received_at = None
                    else:
                        self._lease_announce_due = True
                self._lease.check_expiry()
                self._log_expiry()
                self._warn_silent_holder()
                self._announce_lease_state()
                now = time.monotonic()
                if (
                    self._stats_period_s > 0
                    and now - last_stats >= self._stats_period_s
                ):
                    self._log_stats(now - last_stats)
                    last_stats = now
            except Exception as exc:
                self._clear_input()
                log.warning("reader recovered from %s", type(exc).__name__)
            self._transport.changed.wait(0.01)

    def _fresh_token(self) -> str:
        """Return the token for the next connect attempt.

        Re-minted per attempt whenever we hold the API key. The SDK reuses
        whatever token it was handed for every resume and rejoin, and
        ``build_token``'s default TTL is 6h -- so a stack up longer than that
        could never rejoin after a drop, which is precisely the situation the
        reconnect loop exists to handle. Signing a JWT is microseconds and
        happens at most once every few seconds, on this thread, never on the
        control path.

        A pre-minted ``LIVEKIT_TOKEN`` cannot be refreshed; its expiry stays
        whoever minted it's problem, which is the cost the bridge docstring
        already warns about for that option.
        """
        if self._static_token:
            return self._static_token
        from arms.livekit_rt.livekit_token import build_token

        return build_token(self._identity, room=self._room_name)

    def _announce_lease_state(self) -> None:
        now = time.monotonic()
        if not self._lease_announce_due and now - self._last_lease_announce < 1.0:
            return
        cid, epoch, name = self._lease.snapshot()
        payload = json.loads(
            encode_control_state(
                holder_cid=cid or "", epoch=max(epoch, self._epoch_floor), name=name
            )
        )
        payload["resume_cid"] = self._resume_cid if not cid else ""
        payload["requires_rearm"] = True
        if self._transport.publish(json.dumps(payload).encode()):
            self._lease_announce_due = False
            self._last_lease_announce = now

    # -- Packet handling --------------------------------------------------------

    def _handle_packet(self, *args: Any) -> None:
        """Decode one packet on the parent reader thread, outside the SDK worker."""
        payload, topic = _unpack_data_packet(*args)
        if payload is None:
            return
        try:
            if topic == TOPIC_XR:
                self._on_xr(payload)
            elif topic == TOPIC_BTN:
                self._on_button(payload)
            elif topic == TOPIC_CTL:
                self._on_control(payload)
            else:
                with self._lock:
                    self._rx_unknown += 1
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill teleop
            with self._lock:
                self._decode_errors += 1
            log.debug("packet decode failed (topic=%r): %r", topic, exc)

    def _on_xr(self, payload: bytes) -> None:
        with self._lock:
            self._rx_xr += 1

        # Gate on take-control identity BEFORE touching robot state, so a
        # non-holder's frames never reach the arm.
        cid, ep = frame_control_id(payload)
        if not self._is_holder(cid, ep, acquire=False):
            self._note_dropped_xr(cid, ep)
            return

        # REJECT STALE POSES. The pose channel is deliberately lossy, and a lossy
        # WebRTC data channel is also UNORDERED -- so an older frame can arrive
        # after a newer one. Applying it drags the EE target backwards for one
        # tick and forwards on the next, which the arm expresses as buzzing
        # rather than as a position error. Agora RTM was ordered, so this guard
        # had no equivalent there; it is the price of choosing lossy, and worth
        # paying because retransmitting a stale pose is worse than dropping it.
        seq = _frame_seq(payload)
        if seq is not None:
            with self._lock:
                last = self._last_applied_seq
                if last is not None and seq <= last:
                    # A leader restart resets seq to 0. Treat a large backwards
                    # jump as a new stream rather than deafening ourselves to it
                    # forever; a genuine reorder is only ever a few frames.
                    if last - seq < SEQ_RESET_THRESHOLD:
                        self._reordered += 1
                        return
                    log.info(
                        "leader seq reset (%d -> %d); following the new stream",
                        last,
                        seq,
                    )
                self._last_applied_seq = seq

        # Parse the frame ONCE and feed both consumers. Control and telemetry
        # each used to json.loads the same payload independently, and the
        # control decode additionally round-tripped every hand through
        # dumps/loads -- six JSON operations per frame where two suffice, on
        # the process whose GIL the IK solvers and joint-state threads share.
        frame_obj = json.loads(payload)
        decoded = decode_xr_frame_obj(frame_obj)
        now = time.time()
        with self._lock:
            if not self._ever_received_message:
                ctl_log.info(
                    "control stream received <- %s (cid=%s epoch=%d)",
                    self._lease.active_name or cid,
                    cid,
                    ep,
                )
            self._last_message = now
            self._ever_received_message = True
            for hand, vals in decoded.items():
                if hand in self._state:
                    transform = vals[0]
                    if (
                        transform is None
                        or not np.isfinite(transform).all()
                        or not np.isfinite(vals[1:]).all()
                    ):
                        if self._pose_at[hand]:
                            self._input_revision += 1
                        self._state[hand] = (None, 0.0, 0.0, 0.0)
                        self._pose_at[hand] = 0.0
                    else:
                        received_at = (
                            self._packet_received_at
                            if self._packet_received_at is not None
                            else time.monotonic()
                        )
                        if (
                            self._pose_at[hand]
                            and received_at - self._pose_at[hand]
                            > self._input_timeout_s
                        ):
                            self._input_revision += 1
                        self._state[hand] = vals
                        self._pose_at[hand] = (
                            self._packet_received_at
                            if self._packet_received_at is not None
                            else time.monotonic()
                        )

        self._tap_operator_frame(frame_obj, now)

    def _on_button(self, payload: bytes) -> None:
        with self._lock:
            self._rx_btn += 1
        cid, ep = frame_control_id(payload)
        if not self._is_holder(cid, ep, acquire=False):
            return  # only the active operator's buttons fire actions
        if not any(self.input_available(hand) for hand in HANDS):
            return
        name = decode_button(payload)
        if not name:
            return
        now = time.time()
        # The wire carries a bare name ("a"); the runtime registers ALIASES
        # ("button_a_pressed", "button_a", "a_pressed") and binds the SAME
        # callback object to all three. Look up every alias but de-duplicate by
        # identity, so one physical press fires each distinct handler exactly
        # once -- matching the intent of _register_quest_callbacks ("counts once
        # per actual press") rather than firing it once per alias.
        aliases = (f"button_{name}_pressed", f"button_{name}", f"{name}_pressed")
        callbacks: list[Callable[[], None]] = []
        seen: set[int] = set()
        with self._lock:
            self._last_message = now
            self._ever_received_message = True
            for alias in aliases:
                for cb in self._button_callbacks.get(alias, ()):
                    if id(cb) not in seen:
                        seen.add(id(cb))
                        callbacks.append(cb)
        # Report the handler count, not just receipt: "button 'a' -> 0 handlers"
        # is the signature of an alias mismatch, which otherwise looks exactly
        # like a working button that does nothing.
        level = log.info if callbacks else log.warning
        level(
            "button '%s' from cid=%s -> %d handler(s)",
            name,
            cid or "<untagged>",
            len(callbacks),
        )
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                log.warning("button '%s' handler failed: %r", name, exc)

    def _on_control(self, payload: bytes) -> None:
        with self._lock:
            self._rx_ctl += 1
            self._last_ctl = time.time()
        self._note_leader_diag(payload)
        msg = decode_control(payload)
        if not msg:
            return
        kind = msg.get("kind", "")
        if kind == CONTROL_PING:
            self._answer_ping(payload)
            return
        if kind == CONTROL_PONG:
            return  # our own echo, seen because data packets are broadcast
        if kind == CONTROL_STATE:
            return  # our own broadcast reflected by another follower; never a claim
        cid, ep = msg.get("cid", ""), int(msg.get("ep", 0))
        name = msg.get("name", "") or cid
        if kind == CONTROL_RELEASE:
            if self._lease.release(cid, ep):
                self._resume_cid = ""
                self._clear_input()
                self._lease_announce_due = True
                ctl_log.info("%s released control. Nobody is driving.", name)
            return
        # claim / heartbeat: latch the holder early (before its first XR frame).
        won = self._is_holder(cid, ep, msg.get("name", ""))
        if not won and kind == CONTROL_CLAIM:
            # The claim LOST. Previously this was silent on both ends -- the
            # browser painted "holding" optimistically and the robot said
            # nothing -- which is what made a dead Take Control button so hard
            # to diagnose. Announce the floor again so the loser can retry with
            # a winning epoch instead of the operator clicking blindly.
            holder_cid, floor, holder_name = self._lease.snapshot()
            self._lease_announce_due = True
            if holder_cid:
                ctl_log.warning(
                    "%s asked for control but %s already has it. Ignoring.",
                    name,
                    holder_name or holder_cid,
                )
            else:
                ctl_log.warning(
                    "%s asked for control with a stale request (sent #%d, needs "
                    "more than #%d). Telling the browser the right number; it "
                    "will retry by itself.",
                    name,
                    ep,
                    floor,
                )

    # -- Take-control -----------------------------------------------------------

    def _is_holder(
        self, cid: str, ep: int, name: str = "", *, acquire: bool = True
    ) -> bool:
        if ep <= self._epoch_floor:
            return False
        before_cid, before_ep, _ = self._lease.snapshot()
        if not acquire and (before_cid, before_ep) != (cid, ep):
            return False
        holder = self._lease.is_holder(cid, ep, name)
        if holder and (before_cid != cid or before_ep != ep):
            self._clear_input()
            self._lease_epoch_highwater = max(self._lease_epoch_highwater, ep)
            self._resume_cid = cid
            self._lease_announce_due = True  # tell every browser who won
            # Phrased as a window, not a deadline: the reader forces trigger=0 /
            # pose=None for HANDOFF_SUPPRESS_SEC so the outgoing operator's
            # dead-man drops, then the NEXT squeeze arms teleop whenever it
            # comes. An earlier wording ("must re-squeeze within 0.75s") read as
            # a timer the operator could miss, which is not a thing that exists.
            ctl_log.info(
                "%s now has control (request #%d). Arms are held for %.2fs, then "
                "they follow on the next trigger squeeze — no rush, there is no "
                "deadline.",
                name or cid,
                ep,
                HANDOFF_SUPPRESS_SEC,
            )
        return holder

    def _answer_ping(self, payload: bytes) -> None:
        """Echo any leader's ping through bounded IPC without waiting on the SDK."""
        pid = ping_id(payload)
        if not pid:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_pong < PONG_MIN_INTERVAL_S:
                self._pongs_dropped += 1
                return
            self._last_pong = now
            self._pongs += 1

        self._transport.publish(encode_pong(pid).encode("utf-8"))

    def _note_leader_diag(self, payload: bytes) -> None:
        """Latch the optional leader-health fields carried on a ctl message.

        Parsed here rather than through :func:`decode_control` because that
        decoder is shared with the Agora path and deliberately keeps only the
        lease fields — the same reason :func:`_frame_seq` parses XR payloads
        itself. Every field is optional, so an older leader simply reports
        nothing and the stats line says ``leader=?``.

        The fields answer the question the follower cannot otherwise answer when
        ``xr=0``: is the operator's browser failing to *send*, or is the link
        failing to *deliver*? ``vr``/``pub`` come from the leader's own counters,
        so ``vr=no pub=0`` means the headset never started publishing and the
        transport is blameless.
        """
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001 - diagnostics must never break the lease
            return
        if not isinstance(obj, dict):
            return
        diag = {key: obj[key] for key in ("vr", "xr", "err", "rtt") if key in obj}
        if not diag:
            return
        with self._lock:
            self._leader_diag = diag
        # The leader measured this on one clock; we only carry it. Exported as
        # the operator link's real latency -- the first absolute number this
        # link has ever had.
        rtt_ms = diag.get("rtt")
        if rtt_ms is not None:
            try:
                _meta_record_rtt(float(rtt_ms) / 1000.0)
            except (TypeError, ValueError):
                pass

    def _describe_leader(self) -> str:
        """One-token summary of leader-reported health for the stats line."""
        with self._lock:
            diag = dict(self._leader_diag)
        if not diag:
            return "?"
        rtt = diag.get("rtt")
        return "vr:%s pub:%s err:%s rtt:%s" % (
            "yes" if diag.get("vr") else "no",
            diag.get("xr", "?"),
            diag.get("err", "?"),
            f"{float(rtt):.0f}ms" if rtt is not None else "?",
        )

    def _warn_silent_holder(self) -> None:
        """Warn when the lease holder is heartbeating but sending no pose.

        Rate-limited, and only while somebody actually holds control. This is the
        exact shape of the "operator connected, took control, entered VR, arm
        never moved" report: reliable ctl traffic keeps arriving while the lossy
        xr topic stays at zero.
        """
        cid, _ep, name = self._lease.snapshot()
        if cid is None:
            return
        now = time.time()
        with self._lock:
            last_ctl, last_xr = self._last_ctl, self._last_message
            ever_xr = self._ever_received_message
            if now - last_ctl > SILENT_HOLDER_WARN_SEC:
                return  # holder is silent on ctl too -> the lease will expire
            xr_age = (now - last_xr) if ever_xr else float("inf")
            if xr_age <= SILENT_HOLDER_WARN_SEC:
                return
            if now - self._last_silent_warn < SILENT_HOLDER_WARN_PERIOD_S:
                return
            self._last_silent_warn = now
        ctl_log.warning(
            "holder %s is heartbeating but has sent NO pose for %s — its "
            "browser is connected, its WebXR publish loop is not running "
            "(leader reports %s). Have the operator exit and re-enter VR from "
            "the dashboard and check the page's session log for an XR error; "
            "the transport is fine.",
            name or cid,
            "the whole session" if xr_age == float("inf") else f"{xr_age:.1f}s",
            self._describe_leader(),
        )

    def _log_expiry(self) -> None:
        """Log a holder that was just vacated by the liveness timeout."""
        expired = self._lease.take_expired()
        if expired is None:
            return
        e_cid, e_ep, e_name = expired
        self._epoch_floor = max(self._epoch_floor, e_ep)
        self._clear_input()
        self._lease_announce_due = True
        ctl_log.info(
            "%s went quiet for more than %.0fs (headset asleep, tab closed or "
            "link dropped). Control is free again — anyone can take it.",
            e_name or e_cid,
            DEFAULT_HOLDER_TIMEOUT_SEC,
        )

    def active_controller(self) -> tuple[str | None, int, str]:
        """Current lease holder as ``(cid, epoch, name)`` — for dashboards/logs."""
        return self._lease.snapshot()

    def _note_dropped_xr(self, cid: str, ep: int) -> None:
        """Record + rate-limit-log an XR frame rejected by the take-control gate."""
        now = time.time()
        with self._lock:
            self._dropped_xr_frames += 1
            dropped = self._dropped_xr_frames
            if now - self._last_drop_log < 2.0:
                return
            self._last_drop_log = now
        active_cid, active_ep, active_name = self._lease.snapshot()
        ctl_log.warning(
            "ignoring frames from cid=%s epoch=%d — not the holder "
            "(current: %s cid=%s epoch=%d); %d dropped",
            cid or "<untagged>",
            ep,
            active_name or "-",
            active_cid,
            active_ep,
            dropped,
        )

    # -- Telemetry + stats ------------------------------------------------------

    def _tap_operator_frame(self, frame_obj: dict, recv_ts: float) -> None:
        """Forward a received XR frame to operator telemetry (best-effort).

        Also latches the leader ``seq`` per hand and counts gaps. A gap means the
        frame was lost between the operator's browser and here — the only place
        that loss is observable, since by the time a pose reaches the arm the
        missing one simply never existed. Both are best-effort: a telemetry or
        decode hiccup must never delay or break teleop.
        """
        try:
            meta = decode_xr_frame_meta_obj(frame_obj)
            seqs = {hand: frame.get("seq") for hand, frame in meta.items()}
            with self._lock:
                for hand, seq in seqs.items():
                    if hand in self._seq and seq is not None:
                        self._seq[hand] = int(seq)
                # Both hands share one tick's seq, so track gaps once.
                newest = max(
                    (int(s) for s in seqs.values() if s is not None), default=None
                )
                if newest is not None:
                    prev = self._last_seq_seen
                    if prev is not None and newest > prev + 1:
                        self._seq_gaps += newest - prev - 1
                    self._last_seq_seen = newest
            for hand, frame in meta.items():
                if hand in self._state:
                    _meta_record_frame(hand, frame, recv_ts)
        except Exception:  # noqa: BLE001 - telemetry must never break teleop
            pass

    def _log_stats(self, elapsed_s: float) -> None:
        """Emit the periodic terminal link-stats line."""
        with self._lock:
            rx_xr, rx_btn, rx_ctl = self._rx_xr, self._rx_btn, self._rx_ctl
            errors, gated, gaps = (
                self._decode_errors,
                self._dropped_xr_frames,
                self._seq_gaps,
            )
            reordered = self._reordered
            last_msg, ever = self._last_message, self._ever_received_message
            prev_xr, prev_btn, prev_ctl = self._last_stats_counts
            self._last_stats_counts = (rx_xr, rx_btn, rx_ctl)

        rate = (rx_xr - prev_xr) / elapsed_s if elapsed_s > 0 else 0.0
        age = (time.time() - last_msg) if ever else float("inf")
        cid, ep, name = self._lease.snapshot()
        holder = f"{name or cid or '-'} ep={ep}" if cid else "none"

        connection = self.get_connection_status()
        link = "%s verified_age=%s restarts=%s" % (
            connection["state"],
            connection["verified_age_s"],
            connection["restarts"],
        )

        line = (
            "xr=%d (%.1f/s) btn=+%d ctl=+%d | gated=%d seqgap=%d reord=%d err=%d | "
            "age=%s | holder=%s | leader=%s | link=%s"
        )
        args = (
            rx_xr,
            rate,
            rx_btn - prev_btn,
            rx_ctl - prev_ctl,
            gated,
            gaps,
            reordered,
            errors,
            "never" if age == float("inf") else f"{age:.2f}s",
            holder,
            self._describe_leader(),
            link,
        )
        # Silence is the interesting case while an operator holds control: warn
        # so it is visible without --debug, but stay at info once healthy. A
        # holder that has NEVER sent a frame counts as stale too -- that is the
        # "entered VR but nothing arrives" case, and at info it scrolled past
        # looking like an idle link.
        if connection["state"] != "connected" or (cid and (not ever or age > 1.0)):
            log.warning(line + " — STALE", *args)
        else:
            log.info(line, *args)

    def seconds_since_last_frame(self) -> float:
        """Wall-clock seconds since the last real leader frame updated state."""
        with self._lock:
            if not self._ever_received_message:
                return float("inf")
            return max(0.0, time.time() - self._last_message)

    def get_seq(self, hand: str) -> int | None:
        """Latest leader sequence number for ``hand`` (None if unknown)."""
        with self._lock:
            return self._seq.get(hand)

    # -- MetaQuestReader-compatible surface ------------------------------------

    def get_connection_status(self) -> dict:
        result = self._transport.snapshot()
        verified = result["last_verified_at"]
        result["verified_age_s"] = (
            max(0.0, time.time() - verified) if verified else None
        )
        result["verified"] = verified is not None and result["verified_age_s"] < 15.0
        result["input_fresh"] = {hand: self.input_available(hand) for hand in HANDS}
        result["input_timeout_s"] = self._input_timeout_s
        return result

    def input_revision(self) -> tuple:
        """Change on a transport or ownership boundary, even between polls."""
        snapshot = self._transport.snapshot()
        return (
            snapshot["generation"],
            snapshot["connection_epoch"],
            self._input_revision,
        )

    def input_available(self, hand: str) -> bool:
        snapshot = self._transport.snapshot()
        key = (snapshot["generation"], snapshot["connection_epoch"])
        if (
            self._stop.is_set()
            or snapshot["state"] != "connected"
            or key != self._input_key
            or not self._lease.snapshot()[0]
            or self._lease.suppressed()
        ):
            return False
        with self._lock:
            return (
                0
                <= time.monotonic() - self._pose_at.get(hand, 0.0)
                <= self._input_timeout_s
            )

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray | None:
        if not self.input_available(hand):
            return None  # hide pose during the handoff window -> teleop drops
        with self._lock:
            transform = self._state.get(hand, (None, 0.0, 0.0, 0.0))[0]
        return None if transform is None else transform.copy()

    get_hand_transform_ros = get_hand_controller_transform_ros

    def get_grip_value(self, hand: str) -> float:
        if not self.input_available(hand):
            return 0.0
        with self._lock:
            return self._state.get(hand, (None, 0.0, 0.0, 0.0))[1]

    def get_trigger_value(self, hand: str) -> float:
        if not self.input_available(hand):
            return 0.0  # dead-man release -> incoming operator must re-press
        with self._lock:
            return self._state.get(hand, (None, 0.0, 0.0, 0.0))[2]

    def get_joystick_value(self, hand: str) -> tuple[float, float]:
        # This freshness check IS the frozen-thumbstick defence: a stuck axis
        # only reaches the robot while poses keep arriving under a held lease.
        # The polling thread deliberately adds no further gate -- see
        # ``threads/quest_reader.py``.
        if not self.input_available(hand):
            return (0.0, 0.0)
        with self._lock:
            joystick_y = self._state.get(hand, (None, 0.0, 0.0, 0.0))[3]
        return (0.0, joystick_y)

    def get_transformations_and_buttons(self) -> tuple[dict, dict]:
        """Debug surface used by ``--debug-inputs``; no headset pose on this path."""
        transforms = {
            hand: self.get_hand_controller_transform_ros(hand) for hand in HANDS
        }
        return (
            {hand: pose for hand, pose in transforms.items() if pose is not None},
            {},
        )

    def on(self, event_name: str, callback: Callable[[], None]) -> None:
        self._button_callbacks.setdefault(event_name, []).append(callback)

    # -- Lifecycle --------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._transport.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            rx_xr, gaps, gated = self._rx_xr, self._seq_gaps, self._dropped_xr_frames
        log.info(
            "closed: %d xr frames, %d seq gaps, %d gated by take-control, "
            "%d link drop(s) rejoined",
            rx_xr,
            gaps,
            gated,
            self._transport.snapshot()["restarts"],
        )


def _frame_seq(payload: bytes) -> int | None:
    """Return the tick seq stamped on an XR frame, or None if absent.

    Both hands share one tick's seq, so either hand answers. Kept separate from
    ``decode_xr_frame`` so the shared Agora decoder stays untouched; the extra
    parse is negligible at ~45 Hz against the cost of applying a stale pose.
    """
    try:
        obj = json.loads(payload)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    for hand in HANDS:
        sub = obj.get(hand)
        if isinstance(sub, dict) and sub.get("seq") is not None:
            try:
                return int(sub["seq"])
            except (TypeError, ValueError):
                return None
    return None


def _unpack_data_packet(*args: Any) -> tuple[bytes | None, str]:
    """Return ``(payload, topic)`` from whichever shape the SDK handed us.

    livekit-rtc has emitted ``data_received`` as a single ``DataPacket`` and, in
    older releases, as ``(payload, participant, kind, topic)``. Sniffing the
    shape here keeps the reader working across an SDK bump instead of failing
    silently with zero frames — the exact symptom that is hardest to diagnose
    from the robot's side.
    """
    if not args:
        return (None, "")
    first = args[0]

    payload = getattr(first, "data", None)
    if payload is not None:
        topic = getattr(first, "topic", "") or ""
        return (bytes(payload), str(topic))

    if isinstance(first, (bytes, bytearray)):
        topic = ""
        for extra in args[1:]:
            if isinstance(extra, str):
                topic = extra
        return (bytes(first), topic)

    return (None, "")


__all__ = ["HANDOFF_SUPPRESS_SEC", "LiveKitQuestReader", "add_livekit_args"]
