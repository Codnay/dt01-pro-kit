"""Shared LiveKit credentials/config resolution for the XR control transport.

Mirrors the env/.env resolution style of ``agora_bridge`` so every transport
reads configuration the same way. LiveKit carries **control only** here (Quest
pose, buttons, take-control lease); video stays on Agora RTC.

Required:
    LIVEKIT_URL            wss:// URL of the LiveKit server (Cloud or self-hosted),
                           e.g. ``wss://my-project.livekit.cloud``.
    LIVEKIT_API_KEY        API key   (LiveKit Cloud > Settings > Keys).
    LIVEKIT_API_SECRET     API secret for the same key.

Optional:
    LIVEKIT_ROOM           Room every operator + the robot join
                           (default: ``$ROBOT_ID``; there is no built-in
                           fallback -- an unidentified host errors out).
    LIVEKIT_IDENTITY       Follower (robot) participant identity
                           (default: ``piper-follower``).
    LIVEKIT_TOKEN          A pre-minted follower token. Skips local minting, so
                           API key/secret are not needed on the robot. Bound to
                           one identity and expires — prefer key+secret.

Unlike Agora RTM's three separate channels, LiveKit multiplexes all three
streams over ONE connection and distinguishes them by data-packet *topic*:

    xr   -> ~50 Hz controller pose      (lossy / unreliable: drop, never resend)
    btn  -> button presses              (reliable)
    ctl  -> take-control lease traffic  (reliable)

Sending pose lossy is the point: a retransmitted stale pose is worse than no
pose at all, because the arm would act on intent the operator has already moved
past. Agora RTM had no such option — every channel was guaranteed-delivery.
"""

from __future__ import annotations

import os
from pathlib import Path

from arms.robot_identity import transport_name

DEFAULT_IDENTITY = "piper-follower"

# Data-packet topics. These replace Agora's {ch}-xr / {ch}-btn / {ch}-ctl
# channels; the payloads on each are byte-identical to the Agora wire format so
# the two transports can be A/B'd without a decoder difference muddying results.
TOPIC_XR = "xr"
TOPIC_BTN = "btn"
TOPIC_CTL = "ctl"


def _env(name: str) -> str | None:
    """Return ``name`` from the environment or the repo ``.env`` (minimal parse).

    Same walk-up-to-find-.env behaviour as ``agora_bridge._env`` so both
    transports resolve config identically. Returns ``None`` when unset/empty.
    """
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    here = Path(__file__).resolve()
    for parent in here.parents:
        env_path = parent / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    parsed = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if parsed:
                        return parsed
            break
    return None


# Suffix for the default video room. Video deliberately does NOT default into
# the XR control room: that room's participants are Quest headsets, and every
# one of them would start paying for camera bitrate it never asked for. Point
# --livekit-video-room at the XR room only if you actually want that.
VIDEO_ROOM_SUFFIX = "-video"


def video_room(explicit: str | None = None) -> str:
    """Return the room the camera feed publishes into.

    ``--livekit-video-room`` > ``$LIVEKIT_VIDEO_ROOM`` > ``<robot_id>-video``.
    Deriving it from ``robot_id`` is what lets N robots publish at once without
    colliding, for the same reason the Agora channel does (see
    ``robot_identity.transport_name``, which raises rather than invent a shared
    default name).

    This lives here rather than in ``livekit_video`` on purpose: the viewer's
    token server needs to name the room, and importing it from the publisher
    would drag numpy (and cv2) into a process that only serves HTTP.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    from_env = _env("LIVEKIT_VIDEO_ROOM")
    if from_env:
        return from_env
    base = transport_name(None, env_var="ROBOT_ID", kind="LiveKit video room")
    return f"{base}{VIDEO_ROOM_SUFFIX}"


def get_url() -> str:
    """Return the LiveKit server URL, or raise if it is not configured."""
    url = _env("LIVEKIT_URL")
    if url:
        return url
    raise RuntimeError(
        "LIVEKIT_URL not set. Export it or add it to the repo .env file "
        "(e.g. wss://<project>.livekit.cloud)."
    )


def get_api_key() -> str:
    """Return the LiveKit API key, or "" when only a static token is available."""
    return _env("LIVEKIT_API_KEY") or ""


def get_api_secret() -> str:
    """Return the LiveKit API secret, or "" when only a static token is available."""
    return _env("LIVEKIT_API_SECRET") or ""


def get_room() -> str:
    """Return the room name every operator and the robot join.

    Defaults to this host's ``robot_id`` so the room is per-robot without
    anyone having to remember a second variable. ``LIVEKIT_ROOM`` still wins
    (and warns when it disagrees) — see :func:`transport_name`.
    """
    return transport_name(
        _env("LIVEKIT_ROOM"), env_var="LIVEKIT_ROOM", kind="LiveKit room"
    )


def get_identity() -> str:
    """Return the follower's participant identity."""
    return _env("LIVEKIT_IDENTITY") or DEFAULT_IDENTITY


def get_static_token() -> str:
    """Return a pre-minted follower token, or "".

    Empty is the normal case: the reader mints its own from key+secret.
    """
    return _env("LIVEKIT_TOKEN") or ""


def describe_config() -> str:
    """One-line, secret-free summary for startup logs."""
    key = get_api_key()
    return (
        f"url={get_url()} room={get_room()} identity={get_identity()} "
        f"auth={'api-key ' + key[:4] + '…' if key else 'static-token'}"
    )


__all__ = [
    "DEFAULT_IDENTITY",
    "TOPIC_BTN",
    "TOPIC_CTL",
    "TOPIC_XR",
    "describe_config",
    "get_api_key",
    "get_api_secret",
    "get_identity",
    "get_room",
    "get_static_token",
    "get_url",
]
