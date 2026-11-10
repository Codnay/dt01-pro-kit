"""Centralized, colorized, configurable logging for the teleop stack.

One logger, one line format, used by every module so the terminal is readable
and a run can be grepped or replayed line by line.

Terminal line (colorized on a TTY):

    15:10:09.088 INFO    [main]        opencv threads=1 gil switch=0.50ms
    15:10:15.067 INFO    [livekit-rtm] connected room='dt01-pro-01' identity=...
    15:10:57.001 WARNING [can_left]    stale feedback age 0.42s — holding

House rules (follow these when adding a line anywhere in the stack)
-------------------------------------------------------------------
1. **Everything goes through a logger.** No bare ``print()`` on the teleop path:
   a print has no timestamp, no level, no component, ignores ``--log-level`` and
   never reaches ``--log-file``. :func:`console_logger` is the ONLY exception and
   is reserved for the standalone recorder process, which does not parse the
   teleop CLI flags.
2. **One record renders as prefixed lines.** A multi-line message gets the
   timestamp/level/component prefix on *every* line (see :class:`_Formatter`), so
   no output is ever ambiguous about who wrote it or when.
3. **The level carries the severity — the message does not.** No warning/cross/
   tick emoji: WARNING and ERROR already colorize, align and sort. Decorative
   per-subsystem icons are out too — they were applied to a third of the lines
   of any given component, which reads as noise rather than structure.
4. **Startup logs state, not narration.** Prefer one ``key=value`` line naming
   what actually came up over N "Starting X..." lines with no matching "started".
   Per-object construction chatter belongs at DEBUG.
5. **Nothing repeats unchanged forever.** A periodic line must either change or
   back off (see the LiveKit link stats), so a quiet terminal means quiet, and a
   scrolling one means something happened.
6. **Errors log the exception**: ``log.error("...: %s", exc, exc_info=True)`` —
   never ``traceback.print_exc()``, which escapes the format entirely.

Usage
-----
    from arms.logging_setup import get_logger   # or: from common.logging_setup
    log = get_logger("livekit-rtm")
    log.info("connected to room %s", room)
    log.debug("per-frame heartbeat ...")   # silent unless level is DEBUG

Configuration (flags > env > default). The teleop CLI exposes flags
(``--log-level``, ``--log-only``, ``--log-mute``, ``--log-components``,
``--no-log-color``) that call :func:`configure_logging`; env vars are the
fallback for tools that don't parse those flags:

    PIPER_LOG_LEVEL=DEBUG                 # global threshold (default INFO)
    PIPER_LOG_ONLY=teleop,livekit-rtm     # show ONLY these components
    PIPER_LOG_MUTE=camera,joint           # hide these components
    PIPER_LOG_LEVELS=livekit-rtm=DEBUG,camera=WARNING   # per-component thresholds
    PIPER_LOG_COLOR=0                     # force-disable color (default: auto/TTY)
    PIPER_LOG_UTC=1                       # UTC stamps (default: local, like `ops`)
    PIPER_LOG_FILE=/path/log              # also append plain lines to a file

Components (``--log-only`` / ``--log-mute`` take these names)
-------------------------------------------------------------
``main`` startup/shutdown of the process · ``teleop`` runtime + quest reader ·
``ik`` IK solvers · ``joint`` joint-state threads · ``can_right``/``can_left``
per-arm controller · ``robot-proc`` arm subprocess supervision · ``services``
service lifecycle · ``preset`` preset dashboard · ``livekit-rtm`` /
``take-control`` XR transport · ``gcp-telemetry`` / ``meta-telemetry`` /
``gcs-video`` sinks · ``metrics`` Prometheus exporter · ``robot-state-beacon`` ·
``robot-identity`` · ``camera`` / ``recorder`` / ``agora-video`` /
``agora-manifest`` recorder process · ``py`` bridged third-party warnings.

Design notes
------------
* Only the ``piper`` logger tree is configured (propagate=False), so this never
  fights third-party libraries that log on the root logger; those are relayed in
  explicitly by :func:`install_third_party_bridge`.
* The handler writes to stdout exactly like ``print`` did — non-blocking, safe
  in the control loop. High-frequency heartbeats live at DEBUG (off by default).
* Importing this module has ZERO third-party dependencies (stdlib only) so it
  can never fail to import inside a subprocess.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime, timezone

_ROOT = "piper"
# Idempotency + shared config live on the logger object (not module globals)
# because this file is importable as both `common.logging_setup` and
# `arms.logging_setup` — two distinct module objects, one logging
# tree. The logger attribute is the single source of truth across both.
_SENTINEL = "_piper_configured"

_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_ALIASES = {"WARN": "WARNING", "ERR": "ERROR", "FATAL": "CRITICAL"}

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_COMP = "\x1b[36m"  # cyan
_LEVEL_COLOR = {
    "DEBUG": "\x1b[90m",      # grey
    "INFO": "\x1b[32m",       # green
    "WARNING": "\x1b[33m",    # yellow
    "ERROR": "\x1b[31m",      # red
    "CRITICAL": "\x1b[1;31m",  # bold red
}


def _component_of(name: str) -> str:
    return name[len(_ROOT) + 1 :] if name.startswith(_ROOT + ".") else "main"


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "false", "no", "off", ""}


def _utc_timestamps() -> bool:
    """Whether to stamp lines in UTC (default: local, like every other host tool).

    The ops supervisor stamps its ``=== ops start ... ===`` banner in local time
    and so does ``journalctl``/``date``; stamping the log in UTC meant the same
    event carried two clocks 5.5 h apart in one file. Local wins for reading a
    terminal. Set ``PIPER_LOG_UTC=1`` to line the console up with BigQuery
    ``event_ts`` instead.
    """
    return _truthy(os.environ.get("PIPER_LOG_UTC"))


def _norm_level(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().upper()
    v = _ALIASES.get(v, v)
    return v if v in _LEVELS else None


class _Formatter(logging.Formatter):
    """``HH:MM:SS.mmm LEVEL   [component] message`` (+ ANSI color if enabled)."""

    def __init__(self, color: bool) -> None:
        super().__init__()
        self.color = color

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            if _utc_timestamps()
            else datetime.fromtimestamp(record.created)
        )
        return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record)
        comp = _component_of(record.name)
        level = record.levelname
        msg = record.getMessage()

        if self.color:
            lc = _LEVEL_COLOR.get(level, "")
            # Color the whole message for WARNING+ so problems stand out.
            mc = lc if record.levelno >= logging.WARNING else ""
            prefix = (
                f"{_DIM}{ts}{_RESET} "
                f"{lc}{level:<7}{_RESET} "
                f"{_COMP}[{comp}]{_RESET} "
            )
            body = "\n".join(
                f"{prefix}{mc}{line}{_RESET if mc else ''}"
                for line in msg.split("\n")
            )
        else:
            prefix = f"{ts} {level:<7} [{comp}] "
            body = "\n".join(f"{prefix}{line}" for line in msg.split("\n"))

        if record.exc_info:
            # Tracebacks stay verbatim (indented) so they can be pasted into a
            # tool unmodified; every *message* line above carries the prefix.
            trace = self.formatException(record.exc_info)
            body += "\n" + "\n".join(f"    {ln}" for ln in trace.split("\n"))
        return body


class _ComponentFilter(logging.Filter):
    """Allow/deny records by component name (``--log-only`` / ``--log-mute``)."""

    def __init__(self, only: set[str] | None, mute: set[str]) -> None:
        super().__init__()
        self.only = only
        self.mute = mute

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        comp = _component_of(record.name)
        if self.only is not None and comp not in self.only:
            return False
        return comp not in self.mute


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {p.strip() for p in value.split(",") if p.strip()}


def _parse_component_levels(spec: str | None) -> dict[str, str]:
    """Parse ``"agora-rtm=DEBUG,camera=WARNING"`` -> {component: LEVEL}."""
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        comp, _, lvl = part.partition("=")
        comp, nlvl = comp.strip(), _norm_level(lvl)
        if comp and nlvl:
            out[comp] = nlvl
    return out


def configure_logging(
    level: str | None = None,
    *,
    only: "str | set[str] | None" = None,
    mute: "str | set[str] | None" = None,
    component_levels: "str | dict[str, str] | None" = None,
    color: bool | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Install the structured stdout handler on the ``piper`` logger.

    Idempotent: a second call is a no-op unless ``force=True`` (the teleop CLI
    calls it once with the parsed flags; lazy callers get the env-based default).
    All arguments fall back to ``PIPER_LOG_*`` env vars when ``None``.

    When ``log_file`` (or ``$PIPER_LOG_FILE``) is set, structured lines are ALSO
    written to that file (plain, no color). This lets the dashboard shipper tail
    the file directly, so teleop can run WITHOUT being piped through ``tee`` —
    keeping stdout a real terminal whose graceful Ctrl-C shutdown can't be
    aborted by a broken pipe.
    """
    logger = logging.getLogger(_ROOT)
    if getattr(logger, _SENTINEL, False) and not force:
        return

    level = _norm_level(level) or _norm_level(os.environ.get("PIPER_LOG_LEVEL")) or "INFO"

    only_set = _split_csv(only) if isinstance(only, str) else only
    if only_set is None:
        env_only = _split_csv(os.environ.get("PIPER_LOG_ONLY"))
        only_set = env_only or None
    mute_set = _split_csv(mute) if isinstance(mute, str) else (mute or set())
    if not mute_set:
        mute_set = _split_csv(os.environ.get("PIPER_LOG_MUTE"))

    levels = (
        _parse_component_levels(component_levels)
        if isinstance(component_levels, str) or component_levels is None
        else dict(component_levels)
    )
    if not isinstance(component_levels, dict):
        # merge env defaults under any explicit string spec already parsed above
        for comp, lvl in _parse_component_levels(os.environ.get("PIPER_LOG_LEVELS")).items():
            levels.setdefault(comp, lvl)

    if color is None:
        env_color = os.environ.get("PIPER_LOG_COLOR")
        if env_color is not None:
            color = env_color.strip().lower() not in {"0", "false", "no", "off", ""}
        else:
            color = sys.stdout.isatty()

    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
        logger.removeHandler(h)

    comp_filter = _ComponentFilter(only_set, mute_set)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)  # gating happens at the logger level
    handler.setFormatter(_Formatter(bool(color)))
    handler.addFilter(comp_filter)
    logger.addHandler(handler)

    log_file = log_file or os.environ.get("PIPER_LOG_FILE")
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(logging.NOTSET)
            fh.setFormatter(_Formatter(False))  # plain text for the shipper
            fh.addFilter(comp_filter)
            logger.addHandler(fh)
        except Exception as exc:  # noqa: BLE001 - logging must never break startup
            print(f"[logging] could not open log file {log_file!r}: {exc}")

    logger.setLevel(level)
    logger.propagate = False

    # Reset any previously-set per-component thresholds, then apply the new ones.
    for child in list(logging.Logger.manager.loggerDict):
        if child.startswith(_ROOT + "."):
            logging.getLogger(child).setLevel(logging.NOTSET)
    for comp, lvl in levels.items():
        logging.getLogger(f"{_ROOT}.{comp}").setLevel(lvl)

    setattr(logger, _SENTINEL, True)


# Extra handlers that must see EVERY piper record, wherever it was logged from.
# Kept on the module so `add_log_tap` and `console_logger` agree on the set.
_TAPS: list[logging.Handler] = []


def add_log_tap(handler: logging.Handler) -> None:
    """Also send every ``piper`` log record to *handler* (idempotent).

    Used by in-process log viewers — e.g. the preset dashboard's console — so the
    operator sees the same lines the terminal gets without switching windows.

    Attaching to the ``piper`` logger alone is NOT enough: components pinned with
    :func:`console_logger` (camera, recorder, ...) set ``propagate = False`` so
    their lines print exactly once, which also means a handler on the parent never
    sees them. So this attaches to the root *and* to every already-pinned
    component, and :func:`console_logger` attaches it to any pinned afterwards.
    """
    if handler in _TAPS:
        return
    _TAPS.append(handler)
    root = logging.getLogger(_ROOT)
    if not getattr(root, _SENTINEL, False):
        configure_logging()
    root.addHandler(handler)
    for name in list(logging.Logger.manager.loggerDict):
        if not name.startswith(_ROOT + "."):
            continue
        child = logging.getLogger(name)
        if not child.propagate:
            child.addHandler(handler)


def remove_log_tap(handler: logging.Handler) -> None:
    """Detach a handler previously installed with :func:`add_log_tap`."""
    if handler in _TAPS:
        _TAPS.remove(handler)
    logging.getLogger(_ROOT).removeHandler(handler)
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith(_ROOT + "."):
            logging.getLogger(name).removeHandler(handler)


def install_third_party_bridge(*, level: str = "WARNING") -> None:
    """Route ``warnings`` and root-logger output through the piper formatter.

    Third-party libraries (google-api-core, urllib3, pyrealsense, ...) write to
    ``warnings`` or the root logger, which land on stdout raw — no timestamp, no
    level, no component, unaffected by ``--log-level``/``--log-file``. That is why
    a bare multi-line ``FutureWarning`` used to appear mid-startup looking like a
    crash. Bridged here they become ordinary records under component ``py``.

    Idempotent; never raises (logging must not be able to stop the robot).
    """
    root = logging.getLogger()
    if getattr(root, "_piper_bridged", False):
        return
    try:
        piper_root = logging.getLogger(_ROOT)
        if not getattr(piper_root, _SENTINEL, False):
            configure_logging()

        class _Relay(logging.Handler):
            """Re-emit a foreign record on ``piper.py`` so it gets our format."""

            def emit(self, record: logging.LogRecord) -> None:
                dest = logging.getLogger(f"{_ROOT}.py")
                if not dest.isEnabledFor(record.levelno):
                    return
                dest.handle(
                    dest.makeRecord(
                        dest.name,
                        record.levelno,
                        record.pathname,
                        record.lineno,
                        f"{record.name}: {record.getMessage()}",
                        (),
                        record.exc_info,
                    )
                )

        relay = _Relay()
        relay.setLevel(logging.NOTSET)
        root.addHandler(relay)
        root.setLevel(_norm_level(level) or "WARNING")

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            # One line: the raw hook prints the source line and a blank line too.
            logging.getLogger(f"{_ROOT}.py").warning(
                "%s: %s (%s:%d)",
                getattr(category, "__name__", category),
                message,
                filename,
                lineno,
            )

        warnings.showwarning = _showwarning
        setattr(root, "_piper_bridged", True)
    except Exception:  # noqa: BLE001 - logging must never break startup
        pass


def get_logger(component: str) -> logging.Logger:
    """Return the logger for ``component`` (e.g. ``"agora-rtm"``, ``"teleop"``).

    Lazily configures logging on first use so call sites never need to call
    :func:`configure_logging` themselves.
    """
    if not getattr(logging.getLogger(_ROOT), _SENTINEL, False):
        configure_logging()
    return logging.getLogger(f"{_ROOT}.{component}")


def console_logger(component: str, level: str = "DEBUG") -> logging.Logger:
    """Return a component logger whose records ALWAYS reach the console.

    Unlike :func:`get_logger`, this does NOT depend on :func:`configure_logging`
    or the ``PIPER_LOG_*`` env/flags: it attaches a dedicated stdout handler to
    the component logger, pins the level to ``level`` (``DEBUG`` = everything),
    and disables propagation so the lines print exactly once and can never be
    muted or raised by the global config. If ``$PIPER_LOG_FILE`` is set, a
    matching plain-text file handler is added too so the dashboard log shipper
    still sees the lines.

    Use for components whose output must be visible unconditionally (e.g. the
    camera thread) rather than gated behind a configurable log level. Idempotent:
    re-calling for the same component does not stack handlers.
    """
    logger = logging.getLogger(f"{_ROOT}.{component}")
    if getattr(logger, "_piper_console_pinned", False):
        return logger

    env_color = os.environ.get("PIPER_LOG_COLOR")
    if env_color is not None:
        color = env_color.strip().lower() not in {"0", "false", "no", "off", ""}
    else:
        color = sys.stdout.isatty()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_Formatter(bool(color)))
    logger.addHandler(handler)

    log_file = os.environ.get("PIPER_LOG_FILE")
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(logging.NOTSET)
            fh.setFormatter(_Formatter(False))  # plain text for the shipper
            logger.addHandler(fh)
        except Exception as exc:  # noqa: BLE001 - logging must never break startup
            print(f"[logging] could not open log file {log_file!r}: {exc}")

    logger.setLevel(_norm_level(level) or "DEBUG")
    logger.propagate = False  # print exactly once via our own handler(s)
    # propagate=False cuts this component off from any handler on the parent, so
    # in-process viewers have to be attached here directly (see add_log_tap).
    for tap in _TAPS:
        logger.addHandler(tap)
    setattr(logger, "_piper_console_pinned", True)
    return logger
