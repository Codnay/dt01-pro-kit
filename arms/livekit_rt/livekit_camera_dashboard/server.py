"""Static server + subscriber-token minter for the LiveKit camera viewer.

The LiveKit counterpart of ``agora_camera_dashboard/server/token_server.py``,
deliberately a separate process on a separate port: the Agora dashboard is in
daily use and nothing here should be able to disturb it. Run both, compare, keep
whichever wins.

Endpoints:
    GET /                       the viewer page
    GET /api/config             {url, room, rooms?} — what to connect to
    GET /api/token?room=<name>  a **subscriber-only** join token
    GET /api/rooms              rooms currently live on this LiveKit project
    GET /api/health             {"ok": true}

Tokens minted here are `can_subscribe` only: a viewer cannot publish, cannot
send data, and — because the page never asks for a microphone and the publisher
cannot publish audio either (see ``livekit_video``) — no audio exists anywhere
in this path to be billed or encoded.

ONE SERVER, N ROBOTS: ``?room=`` selects which robot to watch, so a single
viewer host covers a fleet. The rooms it will mint for are an **allowlist**
(``LIVEKIT_VIEWER_ROOMS``), defaulting to this host's own
``<robot_id>-video``. That guard is the same one the XR server carries and for
the same reason: this process holds the API secret, so an unrestricted
``?room=`` would let anyone who can reach the port mint a valid token for any
room on the LiveKit project — including another deployment's, and including the
XR control rooms.

Config (env):
    LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET   (repo .env is read too)
    LIVEKIT_VIEWER_HOST   Bind address (default 0.0.0.0).
    LIVEKIT_VIEWER_PORT   Bind port (default 5175 — 5174 is the Agora viewer).
    LIVEKIT_VIEWER_TTL    Token lifetime in seconds (default 21600 = 6h).
    LIVEKIT_VIEWER_ROOMS  Comma-separated rooms this server may mint tokens for
                          (default: this host's ``<robot_id>-video``). Robot ids
                          are accepted too and get the ``-video`` suffix, so
                          ``LIVEKIT_VIEWER_ROOMS=dt01-pro-01,dt01-pro-02``
                          does what it looks like.
"""

from __future__ import annotations

import asyncio
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Deliberately imports nothing from livekit_video: this process only serves
# HTTP and mints tokens, and the publisher module pulls in numpy/cv2.
from arms.livekit_rt.livekit_bridge import (
    VIDEO_ROOM_SUFFIX,
    get_url,
    video_room,
)
from arms.livekit_rt.livekit_token import build_token

HOST = os.getenv("LIVEKIT_VIEWER_HOST", "0.0.0.0")
PORT = int(os.getenv("LIVEKIT_VIEWER_PORT", "5175"))
TTL_SECONDS = int(os.getenv("LIVEKIT_VIEWER_TTL", str(6 * 60 * 60)))
STATIC_DIR = Path(__file__).resolve().parent


def _as_room(name: str) -> str:
    """Accept a bare robot id where a room name is expected.

    Operators say "dt01-pro-02"; the room is "dt01-pro-02-video". Refusing the
    former would be an annoyance with no security value — the allowlist check is
    what actually guards access, and it runs on the normalised name.
    """
    name = name.strip()
    if not name or name.endswith(VIDEO_ROOM_SUFFIX):
        return name
    return f"{name}{VIDEO_ROOM_SUFFIX}"


def allowed_rooms() -> list[str]:
    """Rooms this server may mint tokens for.

    A bare robot id is accepted and gets the ``-video`` suffix, because that is
    how operators refer to a cell and forgetting the suffix would otherwise look
    like an access error.
    """
    raw = os.getenv("LIVEKIT_VIEWER_ROOMS", "")
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names:
        return [video_room(None)]
    return [_as_room(n) for n in names]


def _viewer_identity() -> str:
    """A per-tab identity.

    Never a shared one: LiveKit evicts the older session when an identity
    reconnects, so two operators watching the same robot under one identity
    would take turns kicking each other off — the same trap a static Agora RTM
    token sets, and the reason ``livekit_token`` mints per-session identities.
    """
    return f"viewer-{os.urandom(4).hex()}"


async def _list_rooms() -> list[dict]:
    """Rooms currently live on this project, newest first (best effort)."""
    from livekit import api  # type: ignore[import-not-found]

    url = get_url().replace("wss://", "https://").replace("ws://", "http://")
    lkapi = api.LiveKitAPI(
        url,
        os.getenv("LIVEKIT_API_KEY") or "",
        os.getenv("LIVEKIT_API_SECRET") or "",
    )
    try:
        resp = await lkapi.room.list_rooms(api.ListRoomsRequest())
        return [
            {
                "name": r.name,
                "participants": r.num_participants,
                "publishers": r.num_publishers,
                "createdAt": r.creation_time,
            }
            for r in resp.rooms
        ]
    finally:
        await lkapi.aclose()


class Handler(SimpleHTTPRequestHandler):
    """Static viewer files plus the token/config endpoints."""

    def __init__(self, *args, **kwargs):
        """Serve files out of this package directory."""
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        """Route the API endpoints; everything else is a static file."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        room = _as_room((query.get("room") or [""])[0])

        if parsed.path == "/api/health":
            return self._send_json(200, {"ok": True})

        if parsed.path == "/api/config":
            try:
                rooms = allowed_rooms()
                return self._send_json(
                    200,
                    {
                        "url": get_url(),
                        "room": room or rooms[0],
                        "rooms": rooms,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - host without a ROBOT_ID
                return self._send_json(500, {"error": str(exc)})

        if parsed.path == "/api/rooms":
            try:
                allowed = set(allowed_rooms())
                live = [r for r in asyncio.run(_list_rooms()) if r["name"] in allowed]
                return self._send_json(200, {"rooms": live})
            except Exception as exc:  # noqa: BLE001 - listing is a convenience
                return self._send_json(200, {"rooms": [], "error": str(exc)})

        if parsed.path == "/api/token":
            try:
                allowed = allowed_rooms()
                target = room or allowed[0]
                if target not in allowed:
                    # Never mint for a room this server was not told to serve.
                    return self._send_json(
                        403,
                        {
                            "error": (
                                f"room '{target}' is not served here. "
                                f"Allowed: {', '.join(allowed)}. Add it to "
                                "LIVEKIT_VIEWER_ROOMS to watch it."
                            )
                        },
                    )
                identity = _viewer_identity()
                token = build_token(
                    identity,
                    room=target,
                    ttl_seconds=TTL_SECONDS,
                    name=identity,
                    # Watch only. Nothing this page does can publish media or
                    # data into the robot's room.
                    can_publish=False,
                    can_subscribe=True,
                    can_publish_data=False,
                )
                return self._send_json(
                    200,
                    {
                        "url": get_url(),
                        "room": target,
                        "identity": identity,
                        "token": token,
                        "ttl": TTL_SECONDS,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - surface a fixable message
                return self._send_json(500, {"error": str(exc)})

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):  # noqa: A002 (stdlib signature)
        """Log token mints and errors only — static GETs are noise."""
        code = str(args[1]) if len(args) > 1 else ""
        if "/api/token" in (self.path or "") or code[:1] in ("4", "5"):
            super().log_message(fmt, *args)


def main() -> int:
    """Serve the viewer until interrupted."""
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        default_room = video_room(None)
    except Exception:  # noqa: BLE001 - no ROBOT_ID here is survivable
        default_room = "(set ROBOT_ID)"
    print(
        f"livekit camera viewer on http://{HOST}:{PORT}  (default room: {default_room})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
