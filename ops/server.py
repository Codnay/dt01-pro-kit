"""HTTP control plane for the ops dashboard.

Stdlib ``http.server`` on purpose. This process must start on a bare
``/usr/bin/python3`` with nothing installed, because its whole reason to exist
is being reachable when the robot venv is not.

Every mutating route runs the corresponding :class:`~supervisor.Supervisor`
call and returns immediately; slow work (a 150-second recorder drain) happens on
the supervisor's own threads and is polled through ``/api/status``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):  # allow `python ops/server.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "ops"

import machine_config
from machine_config import MachineConfigError

from .config import CONFIG_PATH, ConfigError, load_raw, save_raw, update_job
from .supervisor import OPS_DIR, Supervisor, SupervisorError

INDEX_PATH = OPS_DIR / "index.html"


class Handler(BaseHTTPRequestHandler):
    """One request. The ``supervisor`` is set on the server object."""

    server_version = "dt01-panel"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    @property
    def sup(self) -> Supervisor:
        """The shared supervisor, hung off the server object at startup."""
        return self.server.supervisor  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quieter than the default: one line, no client hostname lookups."""
        sys.stderr.write(
            f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n"
        )

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-poll

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    @staticmethod
    def _one(query: dict[str, list[str]], key: str) -> str | None:
        """First value for ``key``, or None when absent or empty."""
        values = query.get(key) or []
        return values[0] if values and values[0] != "" else None

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise SupervisorError(f"bad JSON body: {exc}") from exc

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Serve the panel and every read-only API route."""
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"

        if path == "/":
            try:
                self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._error("index.html is missing", 500)
            return
        if path == "/api/ping":  # unauthenticated: for uptime checks
            self._json({"ok": True, "service": "dt01-panel"})
            return
        try:
            if path == "/api/status":
                self._json({"ok": True, **self.sup.status()})
            elif path == "/api/sweep":
                self._json({"ok": True, "found": self.sup.sweep(kill=False)})
            elif path == "/api/can":
                self._json(
                    {
                        "ok": True,
                        **self.sup.can_state(),
                        "actions": self.sup.action_list(),
                    }
                )
            elif path == "/api/config":
                self._json({"ok": True, "text": CONFIG_PATH.read_text()})
            elif path == "/api/arm-config":
                text = machine_config.read_text()
                validation_error = None
                try:
                    machine_config.parse(text)
                except MachineConfigError as exc:
                    validation_error = str(exc)
                self._json(
                    {
                        "ok": True,
                        "text": text,
                        "path": str(machine_config.config_path()),
                        "revision": machine_config.revision(text),
                        "validation_error": validation_error,
                        "applies_on": "stack start",
                    }
                )
            elif path.startswith("/api/job/") and path.endswith("/logs"):
                job_id = path.split("/")[3]
                run = self._one(query, "run")
                raw_offset = self._one(query, "offset")
                offset = int(raw_offset) if raw_offset is not None else None
                payload = self.sup.log_tail(job_id, run=run, offset=offset)
                self._json({"ok": True, **payload})
            else:
                self._error("not found", 404)
        except (SupervisorError, ConfigError, MachineConfigError) as exc:
            self._error(str(exc))
        except Exception as exc:  # noqa: BLE001 - never take the panel down
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Handle the mutating routes: start, stop, adopt, config, sweep."""
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"
        try:
            parts = path.strip("/").split("/")
            if path == "/api/sweep":
                self._json({"ok": True, "found": self.sup.sweep(kill=True)})
            elif path == "/api/config":
                body = self._body()
                text = str(body.get("text") or "")
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"not valid JSON: {exc}") from exc
                save_raw(raw)
                self.sup.reload_config()
                self._json({"ok": True})
            elif path == "/api/arm-config":
                body = self._body()
                if not isinstance(body.get("text"), str) or not isinstance(
                    body.get("revision"), str
                ):
                    raise MachineConfigError(
                        "body must carry 'text' and 'revision' strings"
                    )
                revision = machine_config.save_text(body["text"], body["revision"])
                self._json(
                    {"ok": True, "revision": revision, "applies_on": "stack start"}
                )
            elif len(parts) == 3 and parts[:2] == ["api", "action"]:
                self._json({"ok": True, **self.sup.run_action(parts[2])})
            elif path == "/api/can/apply":
                body = self._body()
                assign = body.get("assign") or {}
                if not isinstance(assign, dict) or not assign:
                    raise SupervisorError("no adapter assignments given")
                result = self.sup.can_apply(
                    {str(k): str(v) for k, v in assign.items()},
                    bitrate=body.get("bitrate"),
                )
                self._json({"ok": True, **result})
            elif path == "/api/spool/clear":
                self._json({"ok": True, **self._clear_spool()})
            elif len(parts) == 4 and parts[:2] == ["api", "job"]:
                job_id, action = parts[2], parts[3]
                if action == "start":
                    run = self.sup.start(job_id)
                    self._json(
                        {
                            "ok": True,
                            "pid": run.pid,
                            "log": run.log_path,
                            "cleaned": self.sup.last_cleanup.get(job_id) or [],
                        }
                    )
                elif action == "stop":
                    force = self._one(query, "force") == "1"
                    self.sup.stop(job_id, force=force)
                    self._json({"ok": True})
                elif action == "cancel-stop":
                    self.sup.cancel_stop(job_id)
                    self._json({"ok": True})
                elif action == "adopt":
                    pid = self._one(query, "pid")
                    run = self.sup.adopt(job_id, int(pid) if pid else None)
                    self._json({"ok": True, "pid": run.pid})
                elif action == "command":
                    body = self._body()
                    env = body.get("env")
                    update_job(
                        job_id,
                        command_line=body.get("command"),
                        env={str(k): str(v) for k, v in env.items()} if env else None,
                    )
                    self.sup.reload_config()
                    self._json({"ok": True})
                else:
                    self._error("not found", 404)
            else:
                self._error("not found", 404)
        except (SupervisorError, ConfigError, MachineConfigError) as exc:
            self._error(str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(f"{type(exc).__name__}: {exc}", 500)

    # -- housekeeping data ------------------------------------------------

    def _spool_path(self) -> Path:
        return Path(self.sup.system_info()["spool_path"])

    def _clear_spool(self) -> dict[str, Any]:
        """Delete the spool, but only when nothing can still be uploading it.

        The old ``kill_gcs_recorder.sh`` ran ``rm -rf`` on this directory as part
        of every stop. Chunks live here until the uploader has pushed them to
        GCS, so any stop that happened during an upload silently destroyed
        footage. Clearing is now a separate, guarded action.
        """
        status = self.sup.status()
        busy = [j["name"] for j in status["jobs"] if j["state"] != "stopped"]
        if busy:
            raise SupervisorError(
                f"refusing to clear the spool while {', '.join(busy)} is running — "
                "those files may still be uploading"
            )
        spool = self._spool_path()
        if not spool.is_dir():
            return {"removed": 0, "note": "spool directory does not exist"}
        blobs = [f for f in spool.rglob("*") if f.is_file()]
        freed = sum(f.stat().st_size for f in blobs)
        shutil.rmtree(spool)
        return {"removed": len(blobs), "freed_mb": round(freed / 1e6, 1)}


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    # This process must never compete with the teleop control loop or the
    # recorder's encoders for CPU. It serves a handful of small requests a
    # second, so it can afford to always lose that contest.
    try:
        os.nice(10)
    except OSError:
        pass

    try:
        load_raw(args.config)
        supervisor = Supervisor(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.supervisor = supervisor  # type: ignore[attr-defined]

    running = [
        j["name"] for j in supervisor.status()["jobs"] if j["state"] != "stopped"
    ]
    print(f"dt01-panel on http://{args.host}:{args.port}")
    print(f"  project: {supervisor.subs['project_dir']}")
    if running:
        print(f"  adopted already-running job(s): {', '.join(running)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping dashboard (supervised jobs keep running)")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
