"""Where this platform's fleet identity comes from — the only place it is derived.

One value, ``ROBOT_ID``, decides which robot a metric belongs to. Resolution
order, first hit wins:

1. ``--robot-id`` on the command line
2. ``$ROBOT_ID`` in the environment
3. ``ROBOT_ID=`` in a ``.env`` beside the dashboard or at the repository root
4. ``UNIDENTIFIED``

The fallback is deliberately not a plausible name. Two platforms that both
default to something like ``dt01-01`` merge into a single history that looks
correct and is not — the same trap as letting a scrape job overwrite the label.
``UNIDENTIFIED`` is visibly wrong in Grafana, so it gets fixed instead of
believed. The dashboard still starts: it is a control panel first, and refusing
to drive a robot because nobody wrote a config file would be the wrong trade.

Deliberately not a ``python-dotenv`` dependency. The robot's own network has no
internet, so the dashboard is stdlib-only; ``KEY=value`` needs no library.
"""

from __future__ import annotations

import os
from pathlib import Path

VARIABLE = "ROBOT_ID"
FALLBACK = "UNIDENTIFIED"


def _candidates() -> list[Path]:
    """``.env`` files to try, nearest first.

    The dashboard is also shipped as a standalone zip (``make_handoff.sh``),
    where there is no repository root above it — hence looking beside the
    package as well as above it, rather than assuming a checkout.
    """
    here = Path(__file__).resolve().parent
    return [
        here / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
        Path.cwd() / ".env",
    ]


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Never raises — an unreadable file is no file."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.split(" #")[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def resolve(explicit: str | None = None,
            env: dict[str, str] | None = None) -> tuple[str, str]:
    """Return ``(robot_id, where_it_came_from)``.

    The second element exists so startup can *say* which source won. An
    identity that silently came from the wrong place is how two robots end up
    sharing one history.
    """
    environ = os.environ if env is None else env
    if explicit and explicit.strip():
        return explicit.strip(), "--robot-id"
    from_env = (environ.get(VARIABLE) or "").strip()
    if from_env:
        return from_env, f"${VARIABLE}"
    for path in _candidates():
        value = read_env_file(path).get(VARIABLE, "").strip()
        if value:
            return value, str(path)
    return FALLBACK, "fallback"
