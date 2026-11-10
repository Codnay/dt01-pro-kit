"""Bounded recovery of an isolated LiveKit worker, independent of arm control."""

from __future__ import annotations

import json
import select
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable

from .livekit_bridge import get_api_key, get_api_secret
from .transport_worker import MAX_PACKET, encode_message


def check_membership(url: str, room: str, identity: str, sid: str) -> bool | None:
    """Independent cloud truth. None means unverified, never confirmed absent."""
    key, secret = get_api_key(), get_api_secret()
    if not key or not secret:
        return None  # static-token deployments have no room-admin capability
    from livekit import api

    token = (
        api.AccessToken(key, secret)
        .with_grants(api.VideoGrants(room_admin=True, room=room))
        .to_jwt()
    )
    endpoint = url.replace("wss://", "https://").replace("ws://", "http://")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/twirp/livekit.RoomService/GetParticipant",
        data=json.dumps({"room": room, "identity": identity}).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            participant = json.load(response)
        active = participant.get("state") in (2, "ACTIVE")
        return bool(active and sid and participant.get("sid") == sid)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        return None
    except Exception:
        return None


class WorkerTransport:
    """Supervise only a private SDK child; public methods never wait on network."""

    def __init__(
        self,
        url: str,
        room: str,
        identity: str,
        token_factory: Callable[[], str],
        connect_timeout: float = 15.0,
        *,
        worker_module: str | None = None,
        probe: Callable = check_membership,
        heartbeat_timeout: float = 2.0,
        shutdown_timeout: float = 2.0,
        probe_interval: float = 5.0,
        backoff_max: float = 5.0,
    ) -> None:
        self.url, self.room, self.identity = url, room, identity
        self.token_factory = token_factory
        self.connect_timeout = connect_timeout
        self.worker_module = (
            worker_module or "arms.livekit_rt.transport_worker"
        )
        self.probe = probe
        self.heartbeat_timeout, self.shutdown_timeout = (
            heartbeat_timeout,
            shutdown_timeout,
        )
        self.probe_interval, self.backoff_max = probe_interval, backoff_max
        self._stop = threading.Event()
        self.changed = threading.Event()
        self._lock = threading.Lock()
        self._events: deque[dict] = deque()
        self._outbound: deque[dict] = deque()
        self._fault = ""
        self._process: subprocess.Popen | None = None
        self._socket: socket.socket | None = None
        self._probe_thread: threading.Thread | None = None
        self._probe_result: tuple | None = None
        self._generation = 0
        self._retry_at = 0.0
        self._backoff = 0.0
        self._last_heartbeat = 0.0
        self._deadline = 0.0
        self._next_probe = 0.0
        self._missing = 0
        self._publish_failures = 0
        self._outage_at = time.monotonic()
        self._snapshot = dict(
            state="connecting",
            generation=0,
            connection_epoch=0,
            attempts=0,
            restarts=0,
            room=room,
            identity=identity,
            sid="",
            last_verified_at=None,
            last_error="",
            recovery_seconds=0.0,
            retry_in_s=0.0,
        )
        self._thread = threading.Thread(
            target=self._run, name="livekit-supervisor", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            result = dict(self._snapshot)
        result["retry_in_s"] = max(0.0, self._retry_at - time.monotonic())
        return result

    def events(self) -> list[dict]:
        with self._lock:
            result = list(self._events)
            self._events.clear()
            self.changed.clear()
        return result

    def publish(self, payload: bytes) -> bool:
        with self._lock:
            if self._snapshot["state"] != "connected":
                return False
            if len(self._outbound) >= 64:
                self._fault = "control send queue full"
                return False
            self._outbound.append(dict(kind="publish", data=payload.decode()))
        return True

    def _update(self, **fields) -> None:
        with self._lock:
            self._snapshot.update(fields)
            self.changed.set()

    def _spawn(self) -> None:
        self._generation += 1
        self._missing = self._publish_failures = 0
        self._next_probe = 0.0
        now = time.monotonic()
        self._deadline = now + self.connect_timeout
        self._last_heartbeat = now
        self._update(
            state="connecting",
            generation=self._generation,
            sid="",
            attempts=self.snapshot()["attempts"] + 1,
            last_verified_at=None,
        )
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket = parent
        parent.setblocking(False)
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", self.worker_module, str(child.fileno())],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            # Credentials stay in private IPC and are not exposed by ps.
            parent.send(encode_message(dict(url=self.url, token=self.token_factory())))
        finally:
            child.close()

    def _retire(self, reason: str) -> None:
        if self.snapshot()["state"] == "connected":
            self._outage_at = time.monotonic()
        self._update(state="reconnecting", last_error=reason, last_verified_at=None)
        with self._lock:
            self._events.clear()
            self._outbound.clear()
        proc, sock = self._process, self._socket
        self._process = self._socket = None
        if sock is not None:
            try:
                sock.send(encode_message(dict(kind="stop")))
            except OSError:
                pass
            sock.close()
        if proc is not None:
            try:
                proc.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                # This exact Popen child has no CAN or robot objects. Never killpg.
                proc.kill()
                proc.wait(timeout=2)
            if not self._stop.is_set():
                self._update(restarts=self.snapshot()["restarts"] + 1)
        self._backoff = min(
            self.backoff_max, self._backoff * 2 if self._backoff else 0.5
        )
        self._retry_at = time.monotonic() + self._backoff

    def _lost(self, now: float) -> None:
        if self.snapshot()["state"] == "connected":
            self._outage_at = now
            self._deadline = now + self.connect_timeout
            with self._lock:
                self._events.clear()
                self._outbound.clear()
        self._update(
            state="reconnecting", last_error="SDK reconnecting", last_verified_at=None
        )

    def _message(self, message: dict, now: float) -> None:
        kind = message.get("kind")
        if kind == "heartbeat":
            self._last_heartbeat = now
            if message.get("state") != 1 and self.snapshot()["state"] == "connected":
                self._lost(now)
        elif kind == "state":
            state = message.get("state")
            if state == "connected":
                if self.snapshot()["state"] != "connected":
                    with self._lock:
                        self._events.clear()
                        self._outbound.clear()
                        self._snapshot["connection_epoch"] += 1
                    self._missing = 0
                    self._next_probe = 0.0
                    self._update(
                        state="connected",
                        sid=message.get("sid", ""),
                        last_error="",
                        last_verified_at=None,
                        recovery_seconds=max(0.0, now - self._outage_at),
                    )
                    self._backoff = 0.0
            elif state == "reconnecting":
                self._lost(now)
            else:
                self._fault = "SDK disconnected"
        elif kind == "error":
            self._fault = "SDK connection failed: " + str(
                message.get("error", "unknown")
            )
        elif kind == "publish_ok":
            self._publish_failures = 0
        elif kind == "publish_failed":
            self._publish_failures += 1
            if self._publish_failures >= 3:
                self._fault = "control announcements failed"
        elif kind in ("data", "joined", "left"):
            snapshot = self.snapshot()
            if snapshot["state"] != "connected":
                return
            if kind == "data" and not 0 <= now - float(message.get("at", 0)) <= 0.25:
                return
            message["generation"] = self._generation
            message["connection_epoch"] = snapshot["connection_epoch"]
            with self._lock:
                if len(self._events) >= 256:
                    self._fault = "control receive queue full"
                elif (
                    message.get("topic") == "xr"
                    and self._events
                    and self._events[-1].get("topic") == "xr"
                ):
                    self._events[-1] = (
                        message  # newest pose wins while a callback is busy
                    )
                    self.changed.set()
                else:
                    self._events.append(message)
                    self.changed.set()

    def _check_cloud(self, now: float) -> None:
        with self._lock:
            result, self._probe_result = self._probe_result, None
        if result is not None:
            generation, epoch, sid, present = result
            current = self.snapshot()
            if (generation, epoch, sid) == (
                self._generation,
                current["connection_epoch"],
                current["sid"],
            ) and current["state"] == "connected":
                self._missing = self._missing + 1 if present is False else 0
                if present is True:
                    self._update(last_verified_at=time.time())
                if self._missing >= 2:
                    self._fault = "robot participant absent from LiveKit"
        if now < self._next_probe or (
            self._probe_thread and self._probe_thread.is_alive()
        ):
            return
        self._next_probe = now + self.probe_interval
        snapshot = self.snapshot()
        if snapshot["state"] != "connected":
            return
        generation, epoch, sid = (
            self._generation,
            snapshot["connection_epoch"],
            snapshot["sid"],
        )

        def check() -> None:
            try:
                result = self.probe(self.url, self.room, self.identity, sid)
            except Exception:
                result = None
            with self._lock:
                self._probe_result = (generation, epoch, sid, result)

        self._probe_thread = threading.Thread(
            target=check, name="livekit-membership", daemon=True
        )
        self._probe_thread.start()

    def _tick(self) -> None:
        now = time.monotonic()
        if self._process is None:
            if now >= self._retry_at:
                self._spawn()
            return
        sock = self._socket
        if sock is not None and select.select([sock], [], [], 0.002)[0]:
            for _ in range(128):
                try:
                    self._message(json.loads(sock.recv(MAX_PACKET)), time.monotonic())
                except BlockingIOError:
                    break
        if self._process.poll() is not None:
            self._fault = "network worker exited"
        if (
            now - self._last_heartbeat > self.heartbeat_timeout
            and now >= self._deadline
        ):
            self._fault = "network worker heartbeat expired"
        state = self.snapshot()["state"]
        if state != "connected" and now >= self._deadline:
            self._fault = "connection/resume deadline exceeded"
        if state == "connected" and now - self._last_heartbeat > self.heartbeat_timeout:
            self._fault = "network worker heartbeat expired"
        self._check_cloud(now)
        with self._lock:
            outbound = list(self._outbound)
            self._outbound.clear()
            fault, self._fault = self._fault, ""
        if fault:
            self._retire(fault)
            return
        for message in outbound:
            try:
                sock.send(encode_message(message))
            except OSError:
                self._retire("control send failed")
                return

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._tick()
                except Exception as exc:
                    self._retire("supervisor error: " + type(exc).__name__)
                if self._process is None:
                    self._stop.wait(0.02)
        finally:
            self._retire("stopped by operator")
            self._update(state="stopped")

    def close(self) -> None:
        """Stop retries and reap only the child owned by this supervisor."""
        self._stop.set()
        self.changed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.shutdown_timeout + 3)
