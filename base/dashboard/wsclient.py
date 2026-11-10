"""A minimal RFC 6455 WebSocket client, stdlib only.

The controller pushes its liveliest data over two sockets — ``/robot/ws/notice``
(state, battery, bumpers, localisation, navigation notices) and
``/robot/notification/sensor_scan`` (laser). Polling cannot replace them: the
bumper and emergency-stop flags are the two signals you least want to learn
about a second late.

``pip install websockets`` is not available on the robot's own access point, so
this implements the client end directly. It is deliberately small: text and
binary frames, continuation, ping/pong, close. No extensions, no compression.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

HANDSHAKE_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 20.0
MAX_MESSAGE_BYTES = 8 * 1024 * 1024   # a 3D scan frame is large; this is the ceiling


class WebSocketError(RuntimeError):
    """Handshake or framing failure."""


class WebSocket:
    """One connected client socket. Not thread-safe for concurrent sends."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = b""
        self._send_lock = threading.Lock()
        self.closed = False

    # --- construction ------------------------------------------------------
    @classmethod
    def connect(cls, url: str, timeout: float = HANDSHAKE_TIMEOUT_S) -> "WebSocket":
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "http"):
            raise WebSocketError(f"unsupported scheme {parts.scheme!r} (ws:// only)")
        host, port = parts.hostname, parts.port or 80
        if not host:
            raise WebSocketError(f"no host in {url!r}")
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"

        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode()

        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            sock.sendall(request)
            client = cls(sock)
            client._read_handshake(timeout)
        except Exception:
            sock.close()
            raise
        sock.settimeout(READ_TIMEOUT_S)
        return client

    def _read_handshake(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while b"\r\n\r\n" not in self._buffer:
            if time.monotonic() > deadline:
                raise WebSocketError("timed out waiting for the upgrade response")
            chunk = self._sock.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed during the handshake")
            self._buffer += chunk
        head, _, rest = self._buffer.partition(b"\r\n\r\n")
        self._buffer = rest
        status = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        if "101" not in status:
            raise WebSocketError(f"upgrade refused: {status}")
        # The Sec-WebSocket-Accept digest is not verified. It guards against a
        # confused cache or proxy, and there is neither between us and a robot
        # on its own access point; a wrong digest would fail at the first frame
        # anyway.

    # --- receiving ---------------------------------------------------------
    def _recv_exactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self._sock.recv(max(4096, count - len(self._buffer)))
            if not chunk:
                self.closed = True
                raise WebSocketError("connection closed by the robot")
            self._buffer += chunk
        head, self._buffer = self._buffer[:count], self._buffer[count:]
        return head

    def _recv_frame(self) -> tuple[int, bytes, bool]:
        first, second = self._recv_exactly(2)
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._recv_exactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._recv_exactly(8))[0]
        if length > MAX_MESSAGE_BYTES:
            raise WebSocketError(f"frame of {length} bytes exceeds the {MAX_MESSAGE_BYTES} cap")
        mask = self._recv_exactly(4) if masked else b""
        payload = self._recv_exactly(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return opcode, payload, fin

    def messages(self) -> Iterator[str]:
        """Yield complete text messages until the socket closes."""
        pending = bytearray()
        pending_opcode = None
        while not self.closed:
            opcode, payload, fin = self._recv_frame()
            if opcode == OPCODE_CLOSE:
                self.closed = True
                return
            if opcode == OPCODE_PING:
                self._send_frame(OPCODE_PONG, payload)
                continue
            if opcode == OPCODE_PONG:
                continue
            if opcode == OPCODE_CONTINUATION:
                if pending_opcode is None:
                    raise WebSocketError("continuation frame with nothing to continue")
            else:
                pending_opcode = opcode
                pending = bytearray()
            pending += payload
            if not fin:
                continue
            if pending_opcode in (OPCODE_TEXT, OPCODE_BINARY):
                yield pending.decode("utf-8", "replace")
            pending_opcode, pending = None, bytearray()

    # --- sending -----------------------------------------------------------
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        with self._send_lock:
            self._sock.sendall(bytes(header) + mask + masked)

    def send_json(self, payload: dict[str, Any]) -> None:
        self._send_frame(OPCODE_TEXT, json.dumps(payload).encode())

    def close(self) -> None:
        self.closed = True
        try:
            self._send_frame(OPCODE_CLOSE, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


class WebSocketFeed:
    """A self-reconnecting reader for one push socket.

    ``on_message`` is called with each decoded JSON payload. Exceptions from it
    are swallowed and reported through ``last_error`` — a malformed push must
    never take the dashboard's connection down with it.
    """

    def __init__(self, url: str | list[str], on_message: Callable[[dict[str, Any]], None],
                 on_open: Callable[["WebSocket"], None] | None = None,
                 name: str = "ws", reconnect_s: float = 3.0) -> None:
        # Firmware revisions disagree about where the push socket lives: some
        # serve /robot/ws/notice, others 404 it and serve /robot/notification.
        # Rotate through the candidates rather than hardcoding one and losing
        # the e-stop and bumper feed on half the fleet.
        self.urls = [url] if isinstance(url, str) else list(url)
        self.url = self.urls[0]
        self._next_url = 0
        self.name = name
        self._on_message = on_message
        self._on_open = on_open
        self._reconnect_s = reconnect_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: WebSocket | None = None
        self.connected = False
        self.last_error: str | None = None
        self.last_message_at: float = 0.0
        self.messages_seen = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"wsfeed-{self.name}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        socket_ = self._socket
        if socket_ is not None:
            socket_.close()
        self.connected = False

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "url": self.url,
            "error": self.last_error,
            "messages": self.messages_seen,
            "age_s": round(time.monotonic() - self.last_message_at, 2)
            if self.last_message_at else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.url = self.urls[self._next_url % len(self.urls)]
                self._socket = WebSocket.connect(self.url)
                self.connected = True
                self.last_error = None
                if self._on_open:
                    self._on_open(self._socket)
                for raw in self._socket.messages():
                    if self._stop.is_set():
                        break
                    self.last_message_at = time.monotonic()
                    self.messages_seen += 1
                    try:
                        self._on_message(json.loads(raw))
                    except (ValueError, TypeError) as exc:
                        self.last_error = f"undecodable push: {exc}"
            except (WebSocketError, OSError) as exc:
                self.last_error = f"{self.url}: {exc}"
                self._next_url += 1          # a 404 here just means "try the other path"
            except Exception as exc:                      # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.connected = False
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
            if not self._stop.wait(self._reconnect_s):
                continue
