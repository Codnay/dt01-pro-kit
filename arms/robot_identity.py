"""Data provenance: the one place `robot_id` and `collected_by` come from.

Every data path this repo writes — BigQuery rows, GCS video manifests,
Prometheus series, session/episode ids — must be attributable to the physical
arm pair that produced it, or a second Jetson silently merges its telemetry
into the first one's. That attribution comes from a single value, ``ROBOT_ID``,
resolved here and nowhere else.

Resolution order (first non-empty wins):

1. an explicit ``--robot-id`` CLI flag (see :func:`add_robot_id_arg`),
2. the ``ROBOT_ID`` process environment variable,
3. ``ROBOT_ID`` in the repo's ``.env`` (the normal way to set it — the value
   lives with the checkout, so it survives shells, systemd units and ops jobs
   without touching the system environment),
4. :data:`DEFAULT_ROBOT_ID` (``robot0``) — the historical default, kept so a
   host with no ``.env`` entry keeps writing the series it always has.

The id names the **robot cell** (arm pair + its Jetson), not the board, e.g.
``dt01-pro-01``. Keep it stable forever and never reuse it: it is the join key
for every past episode.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

try:  # pragma: no cover - thin shim; this module is imported both ways
    from arms.logging_setup import get_logger
except ImportError:  # noqa: BLE001
    from arms.logging_setup import get_logger

log = get_logger("robot-identity")

DEFAULT_ROBOT_ID = "robot0"

# Who operated the robot for a recording. Every dataset in the bucket up to
# 2026-08-27 was collected by one person, so that is the default a host keeps
# until its .env says otherwise -- an unset value must not silently orphan a
# session from its operator.
DEFAULT_COLLECTED_BY = "operator"

# robot_id ends up in BigQuery values, GCS object names, Prometheus label values
# and (derived) transport channel names, so keep it to the intersection of what
# all of those accept without escaping.
_VALID_ROBOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

# Warn at most once per process that this host has no configured identity.
_WARNED_DEFAULT = False


def _env(name: str) -> str | None:
    """Return ``name`` from the environment or the repo ``.env``.

    Mirrors the minimal resolver used by the GCP sinks and the transport
    bridges, so this module keeps the repo's "no python-dotenv dependency"
    convention and stays importable from any process (including the bridge
    subprocesses, which do not import the GCP package).
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


def _validated(candidate: str | None) -> str | None:
    """Return ``candidate`` if it is a usable id, else ``None`` (saying why)."""
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    if not candidate:
        return None
    if not _VALID_ROBOT_ID.match(candidate):
        log.warning(
            "ignoring invalid ROBOT_ID %r (allowed: letters, digits, '.', '_', "
            "'-'; max 63 chars)",
            candidate,
        )
        return None
    return candidate


def configured_robot_id(args: argparse.Namespace | None = None) -> str | None:
    """Return the EXPLICITLY configured robot id, or ``None`` when unset.

    The difference from :func:`resolve_robot_id` matters: a caller that needs a
    name two machines must agree on (a room, a channel) cannot accept a silent
    fallback, because every unidentified host would pick the SAME fallback and
    they would all meet inside it.
    """
    explicit = getattr(args, "robot_id", None) if args is not None else None
    return _validated(explicit) or _validated(_env("ROBOT_ID"))


def resolve_robot_id(args: argparse.Namespace | None = None) -> str:
    """Return this host's ``robot_id``, honouring flag > env > .env > default.

    Never raises: an unusable value is reported and replaced by
    :data:`DEFAULT_ROBOT_ID`, because a bad id must not stop a robot from
    running — it only makes its data harder to attribute.
    """
    global _WARNED_DEFAULT
    configured = configured_robot_id(args)
    if configured:
        return configured
    if not _WARNED_DEFAULT:
        # `.env` is gitignored, so a fresh host (or a `git pull` onto a second
        # Jetson) silently has no identity -- and everything it writes would be
        # filed under robot0 while the rest of the fleet's history is not. Say so
        # once, loudly, at startup rather than discovering it in a query later.
        _WARNED_DEFAULT = True
        log.warning(
            "no ROBOT_ID set -- writing data as %r. Add 'ROBOT_ID=<this-cell>' "
            "to the repo .env (it is gitignored, so it does not travel with a "
            "git pull).",
            DEFAULT_ROBOT_ID,
        )
    return DEFAULT_ROBOT_ID


def resolve_collected_by(args: argparse.Namespace | None = None) -> str:
    """Return who is collecting this data: flag > env > .env > default.

    Distinct from ``robot_id`` on purpose: the robot is a machine that stays
    put, while the operator changes between sessions. Both are stamped on every
    row and object so a dataset can be traced to a cell AND to a person.

    Never raises — unlike a rendezvous name, a wrong-but-present operator tag
    costs attribution, not correctness, so it degrades to
    :data:`DEFAULT_COLLECTED_BY` rather than stopping a recording.
    """
    explicit = getattr(args, "collected_by", None) if args is not None else None
    return (
        _validated(explicit) or _validated(_env("COLLECTED_BY")) or DEFAULT_COLLECTED_BY
    )


def add_collected_by_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--collected-by`` on a parser (idempotent across helpers)."""
    for action in parser._actions:  # noqa: SLF001 - argparse has no public probe
        if "--collected-by" in getattr(action, "option_strings", ()):
            return
    parser.add_argument(
        "--collected-by",
        type=str,
        default=None,
        help=(
            "Operator who collected this data, stamped on BigQuery rows and GCS "
            f"objects (default: $COLLECTED_BY / .env, else {DEFAULT_COLLECTED_BY})."
        ),
    )


def add_robot_id_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--robot-id`` on a parser (idempotent across helpers).

    Several ``add_*_args`` helpers may want the flag; argparse raises on a
    duplicate option string, so registering is a no-op when it already exists.
    """
    for action in parser._actions:  # noqa: SLF001 - argparse has no public probe
        if "--robot-id" in getattr(action, "option_strings", ()):
            return
    parser.add_argument(
        "--robot-id",
        type=str,
        default=None,
        help=(
            "Identity of this robot cell, stamped on BigQuery rows, GCS video "
            "manifests, Prometheus series and session ids "
            f"(default: $ROBOT_ID / .env, else {DEFAULT_ROBOT_ID})."
        ),
    )


# Mismatch is reported once per variable per process: it is a startup-config
# fact, not a per-call event, and these accessors run on hot paths.
_WARNED_TRANSPORTS: set[str] = set()


def transport_name(explicit: str | None, *, env_var: str, kind: str) -> str:
    """Return a transport rendezvous name, defaulting to this robot's id.

    The LiveKit room and the Agora channel name a MEETING POINT, so both ends of
    the link must choose the same string. Deriving them from ``robot_id`` makes
    that automatic and makes collisions impossible: robot #2 cannot land in
    robot #1's room by inheriting a shared default.

    An explicit value still wins — the far end may be pinned to a legacy name,
    and silently overriding it would break a working link. But a name that
    disagrees with ``robot_id`` is exactly how two robots end up sharing one
    room, so it is reported once, loudly, instead of passing unnoticed.

    Raises:
        RuntimeError: when neither ``env_var`` nor ``ROBOT_ID`` is configured.
            There is deliberately NO built-in default — a shared fallback name
            is precisely how an unidentified robot joins someone else's session.
    """
    robot_id = configured_robot_id()
    name = (explicit or "").strip()
    if not name:
        if robot_id is None:
            raise RuntimeError(
                f"{env_var} is unset and ROBOT_ID is not configured, so there is "
                f"no {kind} name to use — and there is no built-in default on "
                f"purpose, because a shared fallback is how two robots end up in "
                f"one session. Set ROBOT_ID in the repo .env (recommended: it "
                f"names every other data path too), or set {env_var} explicitly "
                f"on BOTH ends of the link."
            )
        return robot_id
    if robot_id is not None and name != robot_id and env_var not in _WARNED_TRANSPORTS:
        _WARNED_TRANSPORTS.add(env_var)
        log.warning(
            "%s=%r does not match ROBOT_ID=%r. The %s is NOT derived from this "
            "robot's identity, so the other end must be pinned to %r too. "
            "Unset %s to derive it from ROBOT_ID.",
            env_var,
            name,
            robot_id,
            kind,
            name,
            env_var,
        )
    return name


def make_session_id(robot_id: str, suffix: str | None = None) -> str:
    """Build this process's session id, or reuse a shared ``$PIPER_SESSION_ID``.

    Format is ``{%Y%m%dT%H%M%S}-{robot_id}-{suffix|pid}``. The timestamp stays
    LEADING on purpose: episode folders in GCS are named ``{session_id}-epNNNN``
    and the episode pipeline pre-filters manifests by parsing that leading
    stamp, so the robot id goes after it. Because ``episode_id`` now contains
    ``robot_id``, rows and objects from two robots stay separable even in
    queries and listings that predate the ``robot_id`` column.

    A shared ``$PIPER_SESSION_ID`` still wins, so the recorder process and the
    arms process can agree on one session (and therefore one ``episode_id``).
    """
    shared = _env("PIPER_SESSION_ID")
    if shared:
        return shared
    tail = suffix or str(os.getpid())
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{robot_id}-{tail}"


__all__ = [
    "DEFAULT_COLLECTED_BY",
    "DEFAULT_ROBOT_ID",
    "add_collected_by_arg",
    "add_robot_id_arg",
    "configured_robot_id",
    "make_session_id",
    "resolve_collected_by",
    "resolve_robot_id",
    "transport_name",
]
