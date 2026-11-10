"""Mint LiveKit access tokens (room-join JWTs) for the robot and for browsers.

LiveKit issues ONE token per participant that grants join on a room, and all
three data topics ride that single grant.

Needs ``livekit-api`` (``pip install livekit-api``) plus ``LIVEKIT_API_KEY`` /
``LIVEKIT_API_SECRET``. The robot can skip both by exporting a pre-minted
``LIVEKIT_TOKEN`` instead, at the cost of a fixed identity and an expiry you
have to manage by hand.
"""

from __future__ import annotations

import datetime

from arms.livekit_rt.livekit_bridge import (
    get_api_key,
    get_api_secret,
    get_room,
)

DEFAULT_TTL_SECONDS = 6 * 60 * 60


def build_token(
    identity: str,
    *,
    room: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    name: str = "",
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True,
    can_publish_sources: list[str] | None = None,
) -> str:
    """Return a room-join JWT for ``identity``.

    Every participant — the robot follower and each operator browser — gets a
    distinct identity. That is what makes multi-operator take-control work: a
    shared identity would let LiveKit evict one session when the next connects,
    which is the same trap a static ``AGORA_RTM_TOKEN`` sets on the Agora path.

    The four capability arguments default to the XR control transport's needs
    (publish + subscribe + data), so existing callers are unchanged. The video
    path narrows them, and ``can_publish_sources=["camera"]`` is the one that
    matters most there: it makes an audio track **impossible to publish** rather
    than merely un-published, which is what keeps a camera feed from ever
    turning into an audio bill or an extra encoder on the Jetson.
    """
    api_key, api_secret = get_api_key(), get_api_secret()
    if not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set, so a token cannot be "
            "minted. Set both, or export a pre-minted LIVEKIT_TOKEN."
        )

    try:
        from livekit import api  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - surface a fixable message
        raise RuntimeError(
            "livekit-api is not installed. Run: pip install livekit-api"
        ) from exc

    grants = api.VideoGrants(
        room_join=True,
        room=room or get_room(),
        can_publish=can_publish,
        can_subscribe=can_subscribe,
        can_publish_data=can_publish_data,
    )
    if can_publish_sources:
        # Older livekit-api versions may not carry the field; a token without it
        # still works (it just does not narrow the sources), so never hard-fail.
        try:
            grants.can_publish_sources = list(can_publish_sources)
        except Exception:  # noqa: BLE001 - optional narrowing, not a hard error
            pass
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_grants(grants)
        .with_ttl(datetime.timedelta(seconds=int(ttl_seconds)))
    )
    return token.to_jwt()


__all__ = ["DEFAULT_TTL_SECONDS", "build_token"]
