#!/usr/bin/env python3
"""Serve the LiveKit XR operator dashboard and mint per-session join tokens.

Run this anywhere the Quest's browser can reach — your laptop, the Mumbai VM,
any box with TLS. It does not talk to the robot at all: the operator's browser
publishes to LiveKit, and the robot subscribes independently. That is the whole
point of moving control to a cloud transport, and it means this server can be
restarted mid-session without dropping the arm.

    python -m arms.livekit_rt.livekit_xr_server --generate-cert

WebXR requires a **secure context**, so the page must be served over HTTPS (or
from localhost). ``--generate-cert`` makes a throwaway self-signed cert for
bring-up; the Quest browser will show an interstitial you have to accept once.
For anything beyond a first test, put it behind a real certificate.

Every browser session is minted its OWN participant identity and a stable
``cid``. Two headsets can therefore be connected at once while exactly one holds
the take-control lease — a shared identity would make LiveKit evict the first
session when the second joins, which is the same trap a static
``AGORA_RTM_TOKEN`` sets on the Agora path.

ONE SERVER, N ROBOTS. The room is chosen per browser session, not per process,
so a single container serves every deployment: the operator picks their robot
from the dropdown (or opens ``?robot=dt01-pro-02`` directly) and gets a token
scoped to that room only. Set the allowlist with ``XR_ROBOTS`` (or ``--robots``);
with neither, it falls back to the single room this host's ``ROBOT_ID`` implies,
which is exactly the old behaviour.

The allowlist is not decoration. This process holds the LiveKit API secret and
will mint a token for any room it is asked for, so an unchecked ``?robot=``
would let anyone join any deployment's session.
"""

from __future__ import annotations

import argparse
import secrets
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from arms.livekit_rt.livekit_bridge import (  # noqa: E402
    _env,
    get_room,
    get_url,
)
from arms.livekit_rt.livekit_token import build_token  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "livekit_xr_dashboard"
DEFAULT_TTL = 6 * 60 * 60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument(
        "--port", type=int, default=8443, help="Bind port (default: 8443)."
    )
    parser.add_argument(
        "--room", default="", help="Single room to serve. Overrides LIVEKIT_ROOM."
    )
    parser.add_argument(
        "--robots",
        default="",
        help=(
            "Comma-separated robot ids this server may mint tokens for, e.g. "
            "'dt01-pro-01,dt01-pro-02'. Overrides $XR_ROBOTS. The operator "
            "picks one from the dashboard dropdown. Defaults to the single room "
            "implied by ROBOT_ID/LIVEKIT_ROOM."
        ),
    )
    parser.add_argument(
        "--token-ttl", type=int, default=DEFAULT_TTL, help="Token TTL in seconds."
    )
    parser.add_argument("--cert-file", default="", help="TLS certificate file.")
    parser.add_argument("--key-file", default="", help="TLS private key file.")
    parser.add_argument(
        "--generate-cert",
        action="store_true",
        help="Generate a temporary self-signed certificate with openssl.",
    )
    return parser.parse_args()


def _session_identity(prefix: str = "operator") -> tuple[str, str]:
    """Return ``(identity, cid)`` unique to one browser session."""
    suffix = secrets.token_hex(3)
    return (f"{prefix}-{suffix}", suffix)


def _resolve_robots(args: argparse.Namespace) -> list[str]:
    """Robot ids this server may mint tokens for, most explicit source first.

    ``--robots`` > ``$XR_ROBOTS`` (or the repo ``.env``) > the single room
    ``--room``/``LIVEKIT_ROOM``/``ROBOT_ID`` implies. The last branch keeps a
    single-deployment install working with no new configuration at all.

    Order matters: the first entry is what a browser gets when it does not ask
    for one, so it is also the dropdown's default.
    """
    raw = (args.robots or "").strip() or (_env("XR_ROBOTS") or "").strip()
    names: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in names:  # de-dup, keep author order
            names.append(name)
    if names:
        return names
    return [args.room or get_room()]


def _generate_self_signed() -> tuple[str, str]:
    """Create a throwaway self-signed cert/key pair; return their paths."""
    tmp = Path(tempfile.mkdtemp(prefix="livekit-xr-cert-"))
    cert, key = tmp / "cert.pem", tmp / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-subj",
            "/CN=livekit-xr-leader",
        ],
        check=True,
        capture_output=True,
    )
    print(f"[cert] self-signed certificate written to {tmp}")
    return (str(cert), str(key))


def main() -> int:
    args = _parse_args()
    try:
        from aiohttp import web
    except Exception as exc:  # noqa: BLE001
        print(f"aiohttp is required: pip install aiohttp ({exc})", file=sys.stderr)
        return 2

    try:
        robots = _resolve_robots(args)
        url = get_url()
    except RuntimeError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    async def handle_robots(_request: Any) -> Any:
        """The robots this server can connect to — populates the UI dropdown."""
        return web.json_response({"robots": robots, "default": robots[0]})

    async def handle_config(request: Any) -> Any:
        """Mint a fresh identity + token for one browser session.

        ``?robot=`` picks the room. Checked against the allowlist rather than
        passed through: this process holds the API secret, so an unchecked value
        would mint a valid token for ANY room, including another deployment's.
        """
        want = (request.query.get("robot") or "").strip() or robots[0]
        if want not in robots:
            print(f"[token] REFUSED robot={want!r} (not in {robots})")
            return web.json_response(
                {
                    "error": (
                        f"{want!r} is not a robot this server serves. "
                        f"Available: {', '.join(robots)}."
                    )
                },
                status=400,
            )
        identity, cid = _session_identity()
        try:
            token = build_token(identity, room=want, ttl_seconds=args.token_ttl)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=500)
        print(f"[token] issued identity={identity} cid={cid} robot={want}")
        return web.json_response(
            {
                "url": url,
                "token": token,
                "room": want,
                "robot": want,
                "identity": identity,
                "cid": cid,
            }
        )

    async def handle_log(request: Any) -> Any:
        """Print operator-page log lines on this terminal.

        Once the headset is on, the dashboard's own log pane is unreadable and
        the Quest browser's devtools need a tethered inspector — so leader-side
        failures ("XR start FAILED", a publish error storm) were visible to
        nobody. The page batches its log here so whoever runs this server sees
        the operator's session as it happens.

        Deliberately unauthenticated and print-only: it is a bring-up aid on the
        same host that already hands out join tokens, and it touches nothing but
        stdout. Do not grow it into a control surface.
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed POST must not 500 the page
            return web.json_response({"ok": False}, status=400)
        who = str(body.get("name") or body.get("identity") or "?")[:64]
        lines = body.get("lines") or []
        if not isinstance(lines, list):
            return web.json_response({"ok": False}, status=400)
        for line in lines[:200]:
            print(f"[operator {who}] {str(line)[:500]}", flush=True)
        return web.json_response({"ok": True})

    async def handle_index(_request: Any) -> Any:
        return web.FileResponse(STATIC_DIR / "index.html")

    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/config", handle_config)
    app.router.add_get("/api/robots", handle_robots)
    app.router.add_post("/api/log", handle_log)
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    ssl_context = None
    cert_file, key_file = args.cert_file, args.key_file
    if args.generate_cert and not cert_file:
        cert_file, key_file = _generate_self_signed()
    if cert_file and key_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_file, key_file)

    scheme = "https" if ssl_context else "http"
    print(f"[serve] LiveKit XR dashboard on {scheme}://{args.host}:{args.port}/")
    print(f"[serve] robots={', '.join(robots)} (default: {robots[0]})")
    print(f"[serve] livekit={url}")
    if ssl_context is None:
        print(
            "[warn]  WebXR needs a secure context — over plain HTTP the headset "
            "will refuse to enter VR unless you reach it via localhost. "
            "Use --generate-cert or a real certificate."
        )
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
