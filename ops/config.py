"""Load, validate and rewrite ``jobs.json``.

The dashboard re-reads this file on every start and every status poll, so the
operator can add a camera, switch ``--arm right`` to ``--arm both`` or define a
whole new job from the browser and have the next Start pick it up -- no service
restart, no SSH.

Rewrites go through :func:`save_raw`, which validates *before* touching the
file and keeps a ``.bak`` of the last good version. A dashboard that corrupts
its own config is a dashboard that can no longer stop the robot.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OPS_DIR / "jobs.json"
# Tracked template. jobs.json itself is deliberately NOT in git: camera serials,
# which arms exist and the venv path differ per robot cell, so a shared file
# would either be overwritten by every `git pull` or force a merge conflict on
# each deployment. The template is copied into place on first run and never
# read again -- edits from the dashboard only ever touch the local jobs.json.
EXAMPLE_PATH = OPS_DIR / "jobs.example.json"


class ConfigError(ValueError):
    """Raised for any config the supervisor refuses to run."""


@dataclass(frozen=True)
class StopPolicy:
    """How a job is asked to exit, and how far escalation may go on its own.

    ``signal`` is SIGINT for every job in this repo: the teleop stack and the
    recorder both hang their teardown off ``KeyboardInterrupt``, and neither
    installs a SIGTERM handler that reaches it.

    ``auto_force`` is the safety switch. When False the supervisor will send the
    stop signal and wait forever rather than escalate to SIGKILL, because for
    the teleop stack a SIGKILL mid-teardown latches the arm in E-stop and costs
    a physical power cycle. Escalating that far stays a human decision.
    """

    signal: str = "SIGINT"
    grace_sec: float = 120.0
    repeat_signal: bool = False
    repeat_after_sec: float = 60.0
    auto_force: bool = False
    force_warning: str = ""


@dataclass(frozen=True)
class Job:
    """One supervised command."""

    id: str
    name: str
    argv: list[str]
    cwd: str
    env: dict[str, str]
    stop: StopPolicy
    match: list[str] = field(default_factory=list)
    collateral: list[str] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    # Whether this job holds the CAN bus. Reconfiguring CAN under a running arm
    # would yank the socket out from under the control loop, so the default is
    # fail-safe: anything that forbids automatic SIGKILL is an arm job.
    uses_can: bool = False

    @property
    def command_line(self) -> str:
        """Argv rendered as a copy-pasteable shell command."""
        return " ".join(shlex.quote(a) for a in self.argv)


def _expand(value: str, subs: dict[str, str]) -> str:
    """Expand ``{project_dir}``-style placeholders, then ``~`` and ``$VAR``."""
    for _ in range(5):  # placeholders may nest one level ({python} -> {project_dir})
        nxt = value.format_map(_Missing(subs))
        if nxt == value:
            break
        value = nxt
    return os.path.expanduser(os.path.expandvars(value))


class _Missing(dict):
    """format_map helper: leave unknown placeholders untouched, never KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_common_env(raw: dict[str, Any], subs: dict[str, str]) -> dict[str, str]:
    """Environment every job and action inherits, with placeholders expanded."""
    return {
        str(k): _expand(str(v), subs) for k, v in (raw.get("common_env") or {}).items()
    }


def _seed_from_example(path: Path) -> None:
    """Create a missing ``jobs.json`` from the tracked template.

    A fresh clone has no ``jobs.json`` (it is gitignored), so without this the
    dashboard would refuse to start on a new robot until someone copied a file
    by hand. Copying is one-shot: once the local file exists it is authoritative
    and the template is never consulted again.
    """
    if path != CONFIG_PATH or not EXAMPLE_PATH.is_file():
        return
    tmp = path.with_suffix(".json.seed")
    tmp.write_text(EXAMPLE_PATH.read_text())
    try:
        # Lose the race rather than clobber: another process may have seeded or
        # written the real config between the exists() check and here.
        os.link(tmp, path)
    except FileExistsError:
        return
    finally:
        tmp.unlink(missing_ok=True)
    print(
        f"[ops] seeded {path.name} from {EXAMPLE_PATH.name} -- edit it for this "
        f"robot (camera serials, arms, venv path). It is local to this host and "
        f"is not tracked by git."
    )


def load_raw(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the config file as plain JSON, seeding it from the template once."""
    if not path.exists():
        _seed_from_example(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid JSON: {exc}") from exc


def parse(raw: dict[str, Any]) -> tuple[list[Job], dict[str, str]]:
    """Turn raw config into validated :class:`Job` objects.

    Returns the jobs plus the substitution map, so callers can report what
    ``{project_dir}`` resolved to without re-deriving it.
    """
    if not isinstance(raw, dict):
        raise ConfigError("top level of jobs.json must be an object")

    home = os.path.expanduser("~")
    project_dir = _expand(str(raw.get("project_dir") or "."), {"home": home})
    if not Path(project_dir).is_dir():
        raise ConfigError(f"project_dir does not exist: {project_dir}")
    subs = {"home": home, "project_dir": project_dir}
    subs["python"] = _expand(str(raw.get("python") or "python3"), subs)
    # Identity placeholders, so a link or an argv can name this cell without
    # anyone hardcoding it per deployment: {channel} is what the camera
    # dashboard and the transports key off, {robot_id} is the .env value it
    # falls back to. Empty when unset -- the header says so in amber.
    subs["robot_id"] = robot_id(project_dir) or ""
    subs["channel"] = channel(project_dir) or ""

    common_env = build_common_env(raw, subs)

    entries = raw.get("jobs")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("jobs.json must define a non-empty 'jobs' array")

    jobs: list[Job] = []
    seen: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"jobs[{idx}] must be an object")
        job_id = str(entry.get("id") or "").strip()
        if not job_id:
            raise ConfigError(f"jobs[{idx}] is missing an 'id'")
        if not all(c.isalnum() or c in "-_" for c in job_id):
            raise ConfigError(f"job id {job_id!r} must be alphanumeric/-/_")
        if job_id in seen:
            raise ConfigError(f"duplicate job id: {job_id}")
        seen.add(job_id)

        argv = [_expand(str(a), subs) for a in entry.get("argv") or []]
        if not argv:
            raise ConfigError(f"job {job_id!r} has an empty argv")

        cwd = _expand(str(entry.get("cwd") or project_dir), subs)
        if not Path(cwd).is_dir():
            raise ConfigError(f"job {job_id!r}: cwd does not exist: {cwd}")

        env = dict(common_env)
        env.update(
            {str(k): _expand(str(v), subs) for k, v in (entry.get("env") or {}).items()}
        )

        stop_raw = entry.get("stop") or {}
        signal_name = str(stop_raw.get("signal") or "SIGINT").upper()
        if signal_name not in {"SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"}:
            raise ConfigError(
                f"job {job_id!r}: stop.signal {signal_name!r} is not a graceful "
                "signal (SIGKILL is reachable only via an explicit force stop)"
            )
        stop = StopPolicy(
            signal=signal_name,
            grace_sec=max(1.0, float(stop_raw.get("grace_sec", 120))),
            repeat_signal=bool(stop_raw.get("repeat_signal", False)),
            repeat_after_sec=max(1.0, float(stop_raw.get("repeat_after_sec", 60))),
            auto_force=bool(stop_raw.get("auto_force", False)),
            force_warning=str(stop_raw.get("force_warning") or ""),
        )

        jobs.append(
            Job(
                id=job_id,
                name=str(entry.get("name") or job_id),
                argv=argv,
                cwd=cwd,
                env=env,
                stop=stop,
                match=[str(m) for m in entry.get("match") or []],
                collateral=[str(c) for c in entry.get("collateral") or []],
                links=list(entry.get("links") or []),
                uses_can=bool(entry.get("uses_can", not stop.auto_force)),
            )
        )
    return jobs, subs


def parse_links(raw: dict[str, Any], subs: dict[str, str]) -> list[dict[str, Any]]:
    """Host-level links shown in the header, e.g. other dashboards.

    A link is either an absolute ``url`` (a service on some other host) or a
    ``port`` on this one. Ports stay relative on purpose: the browser resolves
    them against whatever address the panel is open on, so one entry works over
    both Tailscale and the LAN without knowing either address.
    """
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("links") or []):
        if not isinstance(item, dict):
            raise ConfigError(f"links[{idx}] must be an object")
        label = str(item.get("label") or "").strip()
        if not label:
            raise ConfigError(f"links[{idx}] is missing a label")
        if item.get("url"):
            out.append({"label": label, "url": _expand(str(item["url"]), subs)})
        elif item.get("port"):
            out.append(
                {
                    "label": label,
                    "port": int(item["port"]),
                    "scheme": str(item.get("scheme") or "http"),
                }
            )
        else:
            raise ConfigError(f"link {label!r} needs either a url or a port")
    return out


@dataclass(frozen=True)
class Action:
    """A short, one-shot command the dashboard can run on request.

    Distinct from a :class:`Job`: an action runs to completion in seconds and
    its output is the whole point, whereas a job is a long-running process with
    a lifecycle. Used for recovery steps such as clearing a latched E-stop.
    """

    id: str
    name: str
    blurb: str
    argv: list[str]
    cwd: str
    env: dict[str, str]
    timeout_sec: float = 90.0
    requires_arms_stopped: bool = True
    confirm: str = ""

    @property
    def command_line(self) -> str:
        """Argv rendered as a copy-pasteable shell command."""
        return " ".join(shlex.quote(a) for a in self.argv)


def parse_actions(
    raw: dict[str, Any], subs: dict[str, str], common_env: dict[str, str]
) -> list[Action]:
    """One-shot commands exposed as buttons in the dashboard."""
    out: list[Action] = []
    seen: set[str] = set()
    project_dir = subs["project_dir"]
    for idx, entry in enumerate(raw.get("actions") or []):
        if not isinstance(entry, dict):
            raise ConfigError(f"actions[{idx}] must be an object")
        action_id = str(entry.get("id") or "").strip()
        if not action_id or not all(c.isalnum() or c in "-_" for c in action_id):
            raise ConfigError(f"actions[{idx}] needs an alphanumeric id")
        if action_id in seen:
            raise ConfigError(f"duplicate action id: {action_id}")
        seen.add(action_id)
        argv = [_expand(str(a), subs) for a in (entry.get("argv") or [])]
        if not argv:
            raise ConfigError(f"action {action_id!r} has an empty argv")
        cwd = _expand(str(entry.get("cwd") or project_dir), subs)
        if not Path(cwd).is_dir():
            raise ConfigError(f"action {action_id!r}: cwd does not exist: {cwd}")
        env = dict(common_env)
        env.update(
            {str(k): _expand(str(v), subs) for k, v in (entry.get("env") or {}).items()}
        )
        out.append(
            Action(
                id=action_id,
                name=str(entry.get("name") or action_id),
                blurb=str(entry.get("blurb") or ""),
                argv=argv,
                cwd=cwd,
                env=env,
                timeout_sec=max(5.0, float(entry.get("timeout_sec", 90))),
                # Defaults to True. An action that talks to an arm while teleop
                # holds the CAN socket is the dangerous case, so it has to be
                # opted out of explicitly rather than opted into.
                requires_arms_stopped=bool(entry.get("requires_arms_stopped", True)),
                confirm=str(entry.get("confirm") or ""),
            )
        )
    return out


def parse_can(raw: dict[str, Any]) -> dict[str, Any]:
    """CAN settings: the bitrate to configure and the role -> interface names.

    Roles are named here rather than hardcoded because which interface name the
    stack looks for is a property of this deployment, not of the dashboard.
    """
    section = raw.get("can") or {}
    if not isinstance(section, dict):
        raise ConfigError("'can' must be an object")
    bitrate = int(section.get("bitrate", 1_000_000))
    if not 10_000 <= bitrate <= 8_000_000:
        raise ConfigError(f"can.bitrate is implausible: {bitrate}")
    roles: dict[str, str] = {}
    for role, name in (section.get("roles") or {}).items():
        role, name = str(role).strip(), str(name).strip()
        if not role or not name:
            raise ConfigError("can.roles entries need both a role and a name")
        if not all(c.isalnum() or c == "_" for c in name):
            raise ConfigError(f"can.roles[{role}] is not a valid interface name")
        roles[role] = name
    if len(set(roles.values())) != len(roles):
        raise ConfigError("two can.roles cannot share one interface name")
    return {"bitrate": bitrate, "roles": roles}


def load(path: Path = CONFIG_PATH) -> tuple[list[Job], dict[str, str]]:
    """Read and validate the config in one step."""
    return parse(load_raw(path))


def save_raw(raw: dict[str, Any], path: Path = CONFIG_PATH) -> list[Job]:
    """Validate ``raw``, back up the current file, then write it atomically.

    Nothing is written unless :func:`parse` accepts the new config, so a typo in
    the browser cannot leave the dashboard unable to load its own jobs.
    """
    jobs, _ = parse(raw)  # raises ConfigError before anything is touched
    body = json.dumps(raw, indent=2) + "\n"
    if path.exists():
        # Derived from ``path``, not the module constant: a caller saving to
        # somewhere else (a test, a second config) must not overwrite the
        # real host's backup.
        path.with_name(path.name + ".bak").write_text(path.read_text())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(body)
    os.replace(tmp, path)  # atomic: readers see old or new, never a partial file
    return jobs


def update_job(
    job_id: str,
    *,
    command_line: str | None = None,
    env: dict[str, str] | None = None,
    path: Path = CONFIG_PATH,
) -> list[Job]:
    """Rewrite one job's argv and/or per-job env, preserving everything else."""
    raw = load_raw(path)
    for entry in raw.get("jobs", []):
        if isinstance(entry, dict) and entry.get("id") == job_id:
            break
    else:
        raise ConfigError(f"no such job: {job_id}")

    if command_line is not None:
        try:
            argv = shlex.split(command_line)
        except ValueError as exc:
            raise ConfigError(f"could not parse command: {exc}") from exc
        if not argv:
            raise ConfigError("command is empty")
        entry["argv"] = argv
    if env is not None:
        entry["env"] = {str(k): str(v) for k, v in env.items()}
    return save_raw(raw, path)


# -- robot identity ------------------------------------------------------
#
# Which robot cell this Jetson is shows up on every dashboard the operator has
# open at once, so the panel has to say it out loud: ten deployments running the
# same UI on the same port are otherwise indistinguishable, and stopping the
# wrong arm is a real mistake to make.
#
# Read straight out of the repo's .env rather than importing
# arms.robot_identity: ops is stdlib-only and must keep working when
# the robot venv is broken or half-installed, which is exactly when someone
# needs the dashboard. Resolution mirrors that module (env var > .env), minus
# the silent "robot0" fallback -- an unset id is reported as unset, not guessed.
_ROBOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

# The ONLY keys this module will ever read out of .env. That file also holds
# API keys and a service-account path, and these values are served to a
# dashboard that has no authentication, so the allowlist is the safety rail --
# not the caller's discretion.
_DOTENV_KEYS = ("ROBOT_ID", "AGORA_CHANNEL")

_dotenv_cache: tuple[str, float, int, dict[str, str]] | None = None


def _dotenv(project_dir: str) -> dict[str, str]:
    """Return the allowlisted keys from ``{project_dir}/.env``.

    Cached against the file's mtime and size: the status poll asks for this once
    a second per open browser, and the answer changes about once in a robot's
    lifetime. An edited .env is picked up without restarting the service.
    """
    global _dotenv_cache

    env_path = Path(project_dir) / ".env"
    try:
        st = env_path.stat()
        stamp = (project_dir, st.st_mtime, st.st_size)
    except OSError:
        stamp = (project_dir, 0.0, -1.0)
    if _dotenv_cache is not None and _dotenv_cache[:3] == stamp:
        return _dotenv_cache[3]

    found: dict[str, str] = {}
    try:
        text = env_path.read_text()
    except OSError:
        text = ""
    for line in text.splitlines():
        line = line.strip()
        for key in _DOTENV_KEYS:
            if line.startswith(f"{key}=") and key not in found:
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    found[key] = value
    _dotenv_cache = (*stamp, found)
    return found


def _identity(key: str, project_dir: str) -> str | None:
    """Resolve one identity value: process env > .env > None."""
    value = os.getenv(key) or _dotenv(project_dir).get(key)
    if not value:
        return None
    value = value.strip()
    # These end up in an HTML page, a channel name and a GCS path. Same charset
    # rule the robot stack enforces; anything else is treated as unset.
    return value if _ROBOT_ID_RE.match(value) else None


def robot_id(project_dir: str) -> str | None:
    """Return this host's configured robot id, or None when it is unset."""
    return _identity("ROBOT_ID", project_dir)


def channel(project_dir: str) -> str | None:
    """Return the transport channel this cell publishes video/control on.

    Mirrors ``transport_name()`` in the robot stack: ``AGORA_CHANNEL`` wins when
    set (a far end may be pinned to a legacy name), otherwise the channel *is*
    the robot id. The camera dashboard keys off this value, so a link built from
    ``robot_id`` alone would open the wrong feed on a host that overrides it.
    """
    return _identity("AGORA_CHANNEL", project_dir) or robot_id(project_dir)
