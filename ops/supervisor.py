"""Process lifecycle for the ops dashboard: start, stop, adopt, sweep.

Three ideas carry the whole file.

**Every job gets its own session.** Children are spawned with
``start_new_session=True``, so the job becomes both a session leader and a
process-group leader and everything it forks -- the ``multiprocessing`` recorder
child, its ``ffmpeg`` encoders, the ``--robot-process`` worker -- inherits that
group. Signalling the *group* is what makes "stop" actually stop everything;
the old ``pgrep -f gcs_recorder.py`` approach matched only the parent and left
the encoders running, which is why they kept piling up.

**A stop is a negotiation, not a kill.** Both supervised scripts unwind on
``KeyboardInterrupt`` and neither installs a SIGTERM handler that reaches its
teardown, so the stop signal is SIGINT -- exactly what Ctrl+C delivers to a
foreground process group. Escalation past that is bounded by the job's
:class:`~config.StopPolicy`, because for the teleop stack a SIGKILL landing
inside ``graceful_stop()`` strands the arm in a latched E-stop that only an arm
power cycle clears (see ``arms/perf/piper_process.py``).

**PIDs are never trusted alone.** Everything stored on disk carries the boot id
and the process start time, so a stale state file after a reboot or a PID reuse
can never make the supervisor signal an unrelated process.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import can as can_mod
from . import proc
from .config import (
    CONFIG_PATH,
    Action,
    ConfigError,
    Job,
    build_common_env,
    load_raw,
    parse,
    parse_actions,
    parse_can,
    parse_links,
    robot_id,
)

OPS_DIR = Path(__file__).resolve().parent
STATE_DIR = OPS_DIR / "state"
LOG_DIR = OPS_DIR / "logs"

# Log retention. A run log is deleted once ANY of these is exceeded, oldest
# first. The age limit is the binding one and is enforced on a timer as well as
# at start, so "nothing older than KEEP_DAYS" holds even during a stretch where
# no job is ever started. The log of a *running* job is never a candidate.
# Where the recorder parks video chunks on their way to GCS. Non-zero while
# nothing is running means an upload was interrupted and those files are still
# recoverable, which is why clearing the spool is a guarded action.
SPOOL_DIRNAME = "gcs_video_spool"

KEEP_DAYS = 3  # hard age limit
KEEP_RUNS = 20  # most recent runs kept per job
KEEP_MB = 512  # total per-job budget, oldest runs dropped first

# How often the age limit is re-applied while the dashboard just sits there.
PRUNE_INTERVAL_S = 1800.0

# How long spool figures may be stale. They inform a human decision, not a
# control loop, so seconds of staleness cost nothing and a walk per poll does.
_SPOOL_TTL = 20.0

_SIGNALS = {
    "SIGINT": signal.SIGINT,
    "SIGTERM": signal.SIGTERM,
    "SIGHUP": signal.SIGHUP,
    "SIGQUIT": signal.SIGQUIT,
}


class SupervisorError(RuntimeError):
    """Raised for operator-visible refusals (already running, unknown job...)."""


@dataclass
class Running:
    """What we know about a job we believe is alive."""

    pid: int
    pgid: int
    starttime: int
    boot_id: str
    started_at: float
    log_path: str
    argv: list[str]
    adopted: bool = False
    popen: subprocess.Popen | None = field(default=None, repr=False)

    def alive(self) -> bool:
        """Identity-checked liveness -- immune to PID reuse."""
        return proc.alive(self.pid, self.starttime)


@dataclass
class StopProgress:
    """Live state of an in-flight stop, surfaced verbatim in the UI."""

    phase: str = ""
    detail: str = ""
    started_at: float = 0.0
    deadline: float = 0.0
    needs_force: bool = False


class Supervisor:
    """Owns every supervised process. All public methods are thread-safe."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        """Load the config, re-adopt any running jobs, and start log pruning."""
        self.config_path = config_path
        self._lock = threading.RLock()
        self._running: dict[str, Running] = {}
        self._stopping: dict[str, StopProgress] = {}
        self._last_exit: dict[str, dict[str, Any]] = {}
        self._spool_cache: tuple[float, int, int] = (0.0, 0, 0)
        self.last_cleanup: dict[str, list[dict[str, Any]]] = {}
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.links: list[dict[str, Any]] = []
        self.can: dict[str, Any] = {}
        self.actions: list[Action] = []
        self._action_running: set[str] = set()
        self.last_action: dict[str, dict[str, Any]] = {}
        self.reload_config()
        self._adopt_all()
        self.prune_all()
        threading.Thread(target=self._prune_loop, name="log-prune", daemon=True).start()

    # -- config -----------------------------------------------------------

    def reload_config(self) -> None:
        """Re-read jobs.json. Running jobs keep the argv they started with."""
        with self._lock:
            raw = load_raw(self.config_path)
            self.jobs, self.subs = parse(raw)
            self.links = parse_links(raw, self.subs)
            self.can = parse_can(raw)
            self.actions = parse_actions(
                raw, self.subs, build_common_env(raw, self.subs)
            )

    def job(self, job_id: str) -> Job:
        """Look up a job by id, refreshing from disk so UI edits take effect."""
        self.reload_config()
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise SupervisorError(f"no such job: {job_id}")

    # -- persistence ------------------------------------------------------

    def _state_path(self, job_id: str) -> Path:
        return STATE_DIR / f"{job_id}.json"

    def _write_state(self, job_id: str, run: Running | None) -> None:
        """Persist (or clear) the record that lets a restart re-adopt this job.

        The directory is re-created on every write, not just at construction:
        it lives inside the repo, and a ``git clean`` between two starts would
        otherwise leave the supervisor unable to record a running job -- which
        surfaces much later, as a restart that cannot find the arm session it
        was supposed to still be managing.
        """
        path = self._state_path(job_id)
        if run is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pid": run.pid,
                    "pgid": run.pgid,
                    "starttime": run.starttime,
                    "boot_id": run.boot_id,
                    "started_at": run.started_at,
                    "log_path": run.log_path,
                    "argv": run.argv,
                },
                indent=2,
            )
        )

    def _adopt_all(self) -> None:
        """Re-attach to jobs that outlived a dashboard restart.

        The dashboard is restartable *because* of this: a service restart or a
        crash must never orphan a running teleop stack, and must never leave the
        operator without a stop button for it.
        """
        current_boot = proc.boot_id()
        for job in self.jobs:
            path = self._state_path(job.id)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                continue
            # A reboot invalidates every PID on the host; do not even look.
            if data.get("boot_id") != current_boot or not proc.alive(
                int(data.get("pid", -1)), int(data.get("starttime", -1))
            ):
                self._last_exit.setdefault(
                    job.id,
                    {
                        "exit_code": None,
                        "exited_at": path.stat().st_mtime,
                        "reason": "not running when the dashboard started",
                    },
                )
                path.unlink(missing_ok=True)
                continue
            self._running[job.id] = Running(
                pid=int(data["pid"]),
                pgid=int(data["pgid"]),
                starttime=int(data["starttime"]),
                boot_id=str(data["boot_id"]),
                started_at=float(data.get("started_at", 0.0)),
                log_path=str(data.get("log_path", "")),
                argv=list(data.get("argv", [])),
                adopted=True,
            )

    # -- start ------------------------------------------------------------

    # -- one-shot actions ---------------------------------------------------

    def action_list(self) -> list[dict[str, Any]]:
        """Actions and whether each can run right now."""
        with self._lock:
            self.reload_config()
            actions = list(self.actions)
            running = set(self._action_running)
        blockers = self._can_blockers()
        return [
            {
                "id": a.id,
                "name": a.name,
                "blurb": a.blurb,
                "command": a.command_line,
                "confirm": a.confirm,
                "running": a.id in running,
                "blocked_by": blockers if a.requires_arms_stopped else [],
                "last": self.last_action.get(a.id),
            }
            for a in actions
        ]

    def run_action(self, action_id: str) -> dict[str, Any]:
        """Run a one-shot command to completion and return its output.

        Synchronous on purpose: these finish in seconds and the output is the
        reason to run them, so there is nothing useful to show in the meantime.
        The server is threaded, so a slow action does not block the panel.
        """
        with self._lock:
            self.reload_config()
            action = next((a for a in self.actions if a.id == action_id), None)
            if action is None:
                raise SupervisorError(f"no such action: {action_id}")
            if action_id in self._action_running:
                raise SupervisorError(f"{action.name} is already running")
            self._action_running.add(action_id)

        try:
            if action.requires_arms_stopped:
                blockers = self._can_blockers()
                if blockers:
                    raise SupervisorError(
                        f"stop {', '.join(blockers)} first — this talks to the "
                        "arm over CAN, and the running stack owns that socket"
                    )
            started = time.time()
            try:
                out = subprocess.run(  # noqa: S603 - argv comes from our config
                    action.argv,
                    cwd=action.cwd,
                    env=action.env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=action.timeout_sec,
                    check=False,
                )
                output = (out.stdout or "") + (out.stderr or "")
                code: int | None = out.returncode
            except subprocess.TimeoutExpired as exc:
                # The child is killed by TimeoutExpired, but say so plainly:
                # a silent timeout on a recovery step reads as "it worked".
                partial = ""
                for stream in (exc.stdout, exc.stderr):
                    if stream:
                        partial += (
                            stream.decode("utf-8", "replace")
                            if isinstance(stream, bytes)
                            else stream
                        )
                output = partial + (
                    f"\n[ops] timed out after {int(action.timeout_sec)}s "
                    "and was killed"
                )
                code = None
            except OSError as exc:
                raise SupervisorError(f"could not run {action.name}: {exc}") from exc

            result = {
                "id": action.id,
                "name": action.name,
                "exit_code": code,
                "ok": code == 0,
                "output": output.strip(),
                "ran_for": round(time.time() - started, 1),
                "finished_at": time.time(),
            }
            self.last_action[action.id] = result
            return result
        finally:
            with self._lock:
                self._action_running.discard(action_id)

    # -- CAN ---------------------------------------------------------------

    @staticmethod
    def _roles_needed(job: Job, roles: dict[str, str]) -> list[str]:
        """Which CAN roles this job's argv actually needs.

        Read off ``--arm``, whose values are both/right/left and whose default
        in the stack is both. A job running one arm must not be blocked because
        the other arm's adapter is unplugged, so the check has to be this
        specific rather than "every configured role".
        """
        arm = "both"
        for i, token in enumerate(job.argv):
            if token == "--arm" and i + 1 < len(job.argv):
                arm = job.argv[i + 1]
            elif token.startswith("--arm="):
                arm = token.split("=", 1)[1]
        # "both", absent, or anything unrecognised: require everything, because
        # guessing narrower than the truth is how you let a start through that
        # then dies on a missing socket.
        return [arm] if arm in roles else sorted(roles)

    def _can_preflight(self, job: Job) -> None:
        """Refuse to start an arm job whose CAN interfaces are not up.

        Without it the stack starts, fails to open a socket it needs and exits
        with a traceback naming an interface, which tells the operator nothing
        about what to do. This is a MESSAGE, not an interlock -- the arm cannot
        move without CAN either way -- which is why an unreadable CAN state
        lets the start through rather than blocking it: refusing on a broken
        ``ip`` would leave no way to run the robot at all.
        """
        if not job.uses_can:
            return
        roles = dict(self.can.get("roles") or {})
        if not roles:
            return
        needed = self._roles_needed(job, roles)
        try:
            adapters = can_mod.discover(roles)
        except Exception as exc:  # noqa: BLE001 - see docstring: fail open
            print(f"[ops] CAN preflight skipped ({type(exc).__name__}: {exc})")
            return
        live = {a.role for a in adapters if a.role and a.state == "UP"}
        missing = [r for r in needed if r not in live]
        if missing:
            named = ", ".join(f"{r} ({roles[r]})" for r in missing)
            raise SupervisorError(
                f"cannot start {job.name}: no CAN interface up for {named}. "
                "Assign the adapters on the CAN & arms tab first, then start."
            )

    def _can_blockers(self) -> list[str]:
        """Jobs currently holding the CAN bus, tracked or not.

        Strays count. An arm session started by hand owns the socket just as
        firmly as one the dashboard launched, and renaming its interface out
        from under it is exactly as destructive.
        """
        with self._lock:
            self.reload_config()
            snap = proc.snapshot()
            blockers = []
            for job in self.jobs:
                if not job.uses_can:
                    continue
                self._reap(job.id)
                if job.id in self._running or self._find_strays(job, snap):
                    blockers.append(job.name)
            return blockers

    def can_state(self) -> dict[str, Any]:
        """Adapters present, which role each fills, and whether it is safe now."""
        with self._lock:
            self.reload_config()
            settings = dict(self.can)
        adapters = can_mod.discover(settings.get("roles") or {})
        blockers = self._can_blockers()
        return {
            "adapters": [a.as_dict() for a in adapters],
            "bitrate": settings.get("bitrate", 1_000_000),
            "roles": settings.get("roles") or {},
            "blocked_by": blockers,
            "helper_installed": Path(can_mod.HELPER).exists(),
        }

    def can_apply(
        self, wanted: dict[str, str], bitrate: int | None = None
    ) -> dict[str, Any]:
        """Assign adapters to interface names, refusing while an arm is live.

        Reconfiguring a CAN link means taking it down and renaming it. Doing
        that under a running control loop drops the socket mid-command, so this
        is refused outright rather than merely warned about.
        """
        blockers = self._can_blockers()
        if blockers:
            raise SupervisorError(
                f"stop {', '.join(blockers)} first — reconfiguring CAN takes the "
                "interface down, which would cut the control loop's socket "
                "mid-command"
            )
        with self._lock:
            settings = dict(self.can)
            repo = Path(self.subs["project_dir"])
        try:
            return can_mod.apply(
                wanted,
                int(bitrate or settings.get("bitrate", 1_000_000)),
                repo=repo,
            )
        except can_mod.CanError as exc:
            raise SupervisorError(str(exc)) from exc

    def cleanup_leftovers(self, job: Job) -> list[dict[str, Any]]:
        """Kill debris from a previous run of ``job`` before it starts again.

        "Debris" means specifically *orphans*: encoder processes and worker
        children whose parent is gone (``ppid == 1``). Those cannot shut
        themselves down in an orderly way any more -- there is nothing left to
        orchestrate them -- and they still hold the cameras, the CAN socket and
        the spool, so a fresh start would fail on top of them.

        A *live* instance of the job is explicitly not debris and is never
        touched here; :meth:`start` refuses instead, so a running session is
        never killed by someone reaching for the Start button.
        """
        cleaned: list[dict[str, Any]] = []
        snap = proc.snapshot()
        tracked = self._tracked_pgids()
        project_dir = self.subs["project_dir"]
        host_uptime = proc.uptime_sec()

        for info in snap:
            if info.pid in (os.getpid(), 1) or info.pgid in (os.getpgrp(), *tracked):
                continue
            if info.ppid != 1:  # still parented => not an orphan => not ours to kill
                continue
            argv = proc.cmdline_of(info)
            if not argv:
                continue
            joined = " ".join(argv)
            work_dir = proc.cwd(info.pid)
            if work_dir and not work_dir.startswith(project_dir):
                continue

            is_encoder = any(c in Path(argv[0]).name for c in job.collateral)
            is_worker = "multiprocessing" in joined
            if not (is_encoder or is_worker or SPOOL_DIRNAME in joined):
                continue
            cleaned.append(
                {
                    "pid": info.pid,
                    "kind": "orphaned encoder" if is_encoder else "orphaned worker",
                    "age_sec": round(proc.age_sec(info, host_uptime), 1),
                    "cmd": joined[:200],
                    "worker": is_worker,
                }
            )

        if not cleaned:
            return []

        # An orphaned worker self-exits when it notices the parent pipe is
        # closed, so give it a moment to do that on its own before forcing it.
        pids = [c["pid"] for c in cleaned]
        deadline = time.time() + 8.0
        while time.time() < deadline and any(proc.alive(p) for p in pids):
            time.sleep(0.25)
        for pid in pids:
            if proc.alive(pid):
                _quiet_kill(pid, signal.SIGTERM)
        time.sleep(2.0)
        for pid in pids:
            if proc.alive(pid):
                _quiet_kill(pid, signal.SIGKILL)
        for item in cleaned:
            item["killed"] = not proc.alive(item["pid"])
        return cleaned

    def start(self, job_id: str, *, cleanup: bool = True) -> Running:
        """Launch a job in its own session, logging to a fresh run file.

        With ``cleanup`` (the default) any orphaned debris from a previous run
        is cleared first, so a start never has to fight leftovers for the
        cameras or the CAN socket.
        """
        job = self.job(job_id)
        self._can_preflight(job)
        if cleanup:
            self.last_cleanup[job_id] = self.cleanup_leftovers(job)
        with self._lock:
            self._reap(job_id)
            if job_id in self._stopping:
                raise SupervisorError(f"{job.name} is still stopping")
            if job_id in self._running:
                raise SupervisorError(f"{job.name} is already running")

            # A stray instance from before the dashboard existed still owns the
            # cameras or the CAN socket, so starting a second one would fail in
            # a confusing way. Say so instead, and point at the sweep.
            strays = self._find_strays(job)
            if strays:
                pids = ", ".join(str(p.pid) for p in strays)
                raise SupervisorError(
                    f"an untracked {job.name} is already running (pid {pids}). "
                    "Use 'Clean up strays' first."
                )

            log_path = self._new_log_path(job)
            self._prune_logs(job)
            log_fh = open(log_path, "ab", buffering=0)
            try:
                header = (
                    f"=== ops start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"$ {job.command_line}\n"
                    f"cwd: {job.cwd}\n\n"
                ).encode()
                log_fh.write(header)
                popen = subprocess.Popen(  # noqa: S603 - argv comes from our config
                    job.argv,
                    cwd=job.cwd,
                    env=job.env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    # New session => new process group. Everything the job forks
                    # lands in it, which is what makes the group signal complete.
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log_fh.close()  # the child holds its own dup of the fd

            info = proc.read(popen.pid)
            run = Running(
                pid=popen.pid,
                pgid=info.pgid if info else popen.pid,
                starttime=info.starttime if info else 0,
                boot_id=proc.boot_id(),
                started_at=time.time(),
                log_path=str(log_path),
                argv=list(job.argv),
                popen=popen,
            )
            self._running[job_id] = run
            self._last_exit.pop(job_id, None)
            self._write_state(job_id, run)
            proc.invalidate()  # a new tree exists; no poll may miss it

        threading.Thread(
            target=self._watch, args=(job_id, run), name=f"watch-{job_id}", daemon=True
        ).start()
        return run

    def _watch(self, job_id: str, run: Running) -> None:
        """Reap the child and record why it went away."""
        if run.popen is None:
            return
        code = run.popen.wait()
        proc.invalidate()  # the tree just went away
        with self._lock:
            if self._running.get(job_id) is run:
                del self._running[job_id]
            self._write_state(job_id, None)
            requested = self._stopping.pop(job_id, None) is not None
            self._last_exit[job_id] = {
                "exit_code": code,
                "exited_at": time.time(),
                "reason": "stopped as requested" if requested else _exit_reason(code),
                "log_path": run.log_path,
                "ran_for": time.time() - run.started_at,
            }

    # -- stop -------------------------------------------------------------

    def stop(self, job_id: str, *, force: bool = False) -> None:
        """Begin an asynchronous stop. Returns as soon as the signal is sent.

        Stops are slow by design -- the recorder finishes uploading its open
        chunk, the teleop stack lowers the arm under gravity before cutting
        torque -- so this never blocks the HTTP handler. Progress shows up in
        :meth:`status`.
        """
        job = self.job(job_id)
        with self._lock:
            self._reap(job_id)
            run = self._running.get(job_id)
            if run is None:
                raise SupervisorError(f"{job.name} is not running")
            if job_id in self._stopping and not force:
                raise SupervisorError(f"{job.name} is already stopping")

            # Safety interlock, enforced here rather than in the UI so the API,
            # the CLI and a stray curl are all bound by it. For a job whose
            # policy forbids automatic escalation, SIGKILL is reachable only
            # after a graceful stop has actually been tried and has visibly run
            # out its grace. On the teleop stack that ordering is the
            # difference between the arm settling under gravity and the arm
            # latching in E-stop with its drivers cut.
            if force and not job.stop.auto_force:
                progress = self._stopping.get(job_id)
                if progress is None or not progress.needs_force:
                    raise SupervisorError(
                        f"refusing to force-kill {job.name}: a graceful stop has "
                        f"not been tried yet. Stop it normally first and wait "
                        f"out the {int(job.stop.grace_sec)}s grace period — "
                        f"force only becomes available if that fails."
                    )
            progress = self._stopping.setdefault(job_id, StopProgress())
            progress.started_at = progress.started_at or time.time()
            progress.needs_force = False

        threading.Thread(
            target=self._stop_worker,
            args=(job, run, force),
            name=f"stop-{job_id}",
            daemon=True,
        ).start()

    def cancel_stop(self, job_id: str) -> None:
        """Give up waiting on a stalled stop, without signalling anything.

        The stop signal has already been delivered and the job may still be
        unwinding; this only stops the dashboard from holding the job in a
        "stopping" state. It exists so that a job which outlives its grace
        leaves the operator with a choice other than SIGKILL, which for the
        teleop stack means a latched E-stop and an arm power cycle.
        """
        job = self.job(job_id)
        with self._lock:
            progress = self._stopping.get(job_id)
            if progress is None:
                raise SupervisorError(f"{job.name} is not stopping")
            if not progress.needs_force:
                raise SupervisorError(
                    f"{job.name} is still shutting down; wait out the "
                    f"{int(job.stop.grace_sec)}s grace before cancelling"
                )
            self._stopping.pop(job_id, None)
            self._reap(job_id)

    def _stop_worker(self, job: Job, run: Running, force: bool) -> None:
        policy = job.stop
        progress = self._stopping[job.id]

        def note(phase: str, detail: str, deadline: float = 0.0) -> None:
            progress.phase = phase
            progress.detail = detail
            progress.deadline = deadline

        try:
            if force:
                note("force", "SIGKILL to the process group")
                self._signal_group(run, signal.SIGKILL)
                self._wait_gone(run, 10.0)
            else:
                sig = _SIGNALS[policy.signal]
                note(
                    policy.signal,
                    f"sent {policy.signal} to process group {run.pgid} "
                    f"(same as Ctrl+C); waiting up to {int(policy.grace_sec)}s",
                    time.time() + policy.grace_sec,
                )
                self._signal_group(run, sig)

                if policy.repeat_signal:
                    # The recorder treats a second SIGINT as "teardown is
                    # wedged, abandon the upload and exit now". The teleop stack
                    # does NOT: a second KeyboardInterrupt lands inside its
                    # finally block and can skip the robot worker's shutdown
                    # RPC, so repeat_signal is false for it.
                    if not self._wait_gone(run, policy.repeat_after_sec):
                        note(
                            f"{policy.signal} x2",
                            "still running — sending a second "
                            f"{policy.signal} to force its fast-exit path",
                            time.time() + (policy.grace_sec - policy.repeat_after_sec),
                        )
                        self._signal_pid(run, sig)

                remaining = max(
                    0.0, policy.grace_sec - (time.time() - progress.started_at)
                )
                if not self._wait_gone(run, remaining):
                    if not policy.auto_force:
                        # Deliberate dead end. Escalating here is a hardware
                        # decision, so hand it back to the operator with the
                        # consequence spelled out.
                        note(
                            "needs-force",
                            f"still running after {int(policy.grace_sec)}s. "
                            + (policy.force_warning or "Force stop to escalate."),
                        )
                        progress.needs_force = True
                        return
                    note("SIGTERM", "grace expired — SIGTERM to the process group")
                    self._signal_group(run, signal.SIGTERM)
                    if not self._wait_gone(run, 15.0):
                        note("SIGKILL", "SIGTERM ignored — SIGKILL to the group")
                        self._signal_group(run, signal.SIGKILL)
                        self._wait_gone(run, 10.0)

            # The leader is gone; anything left in its group is collateral it
            # failed to reap (ffmpeg mid-write, a wedged multiprocessing child).
            leftovers = self._reap_group(run)
            if leftovers:
                note("cleanup", f"cleaned up {leftovers} leftover child process(es)")
            else:
                note("done", "stopped cleanly")
        finally:
            with self._lock:
                if not run.alive():
                    self._stopping.pop(job.id, None)
                    if job.id in self._running and not self._running[job.id].alive():
                        del self._running[job.id]
                        self._write_state(job.id, None)
                        self._last_exit.setdefault(
                            job.id,
                            {
                                "exit_code": None,
                                "exited_at": time.time(),
                                "reason": "stopped as requested",
                                "log_path": run.log_path,
                                "ran_for": time.time() - run.started_at,
                            },
                        )

    def _wait_gone(self, run: Running, timeout: float, step: float = 0.25) -> bool:
        """Poll until the leader exits or ``timeout`` elapses."""
        end = time.time() + timeout
        while time.time() < end:
            if not run.alive():
                proc.invalidate()
                return True
            time.sleep(step)
        return not run.alive()

    # -- signalling -------------------------------------------------------

    def _guard(self, pgid: int) -> None:
        """Refuse to signal a group that would take the dashboard down with it."""
        if pgid <= 1:
            raise SupervisorError(f"refusing to signal process group {pgid}")
        if pgid == os.getpgrp():
            raise SupervisorError("refusing to signal the dashboard's own group")

    def _signal_group(self, run: Running, sig: int) -> None:
        """Signal the whole process group, but only if the leader is still ours.

        The identity check is the point: without it, a job that exited seconds
        ago could have had its PID (and therefore its PGID) reused, and this
        would signal a stranger's process group.
        """
        self._guard(run.pgid)
        if not run.alive():
            return
        try:
            os.killpg(run.pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise SupervisorError(f"not permitted to signal group {run.pgid}") from exc

    def _signal_pid(self, run: Running, sig: int) -> None:
        """Signal only the leader (used for the recorder's second-SIGINT path)."""
        if not run.alive():
            return
        try:
            os.kill(run.pid, sig)
        except ProcessLookupError:
            pass

    def _reap_group(self, run: Running) -> int:
        """Clear out whatever survived the leader, group first then descendants."""
        victims = {p.pid: p for p in proc.group_members(run.pgid)}
        for child in proc.descendants(run.pid):  # caught setsid() escapees
            victims.setdefault(child.pid, child)
        victims.pop(os.getpid(), None)
        if not victims:
            return 0
        for pid in victims:
            _quiet_kill(pid, signal.SIGTERM)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if not any(proc.alive(pid) for pid in victims):
                return len(victims)
            time.sleep(0.25)
        for pid in victims:
            if proc.alive(pid):
                _quiet_kill(pid, signal.SIGKILL)
        return len(victims)

    # -- strays / sweep ---------------------------------------------------

    def _tracked_pgids(self) -> set[int]:
        return {r.pgid for r in self._running.values()}

    def adopt(self, job_id: str, pid: int | None = None) -> Running:
        """Take ownership of a matching process this supervisor did not start.

        Covers the everyday case of a stack launched by hand over SSH before
        the dashboard existed: rather than telling the operator to go find it,
        the panel picks it up and its Stop button starts working. Adoption is
        signature-checked against the job's ``match`` patterns, so it can never
        latch onto an unrelated process.
        """
        job = self.job(job_id)
        with self._lock:
            self._reap(job_id)
            if job_id in self._running:
                raise SupervisorError(f"{job.name} is already tracked")
            candidates = self._find_strays(job)
            if pid is not None:
                candidates = [c for c in candidates if c.pid == pid]
            if not candidates:
                raise SupervisorError(f"no untracked {job.name} process found")
            if len(candidates) > 1 and pid is None:
                pids = ", ".join(str(c.pid) for c in candidates)
                raise SupervisorError(
                    f"{len(candidates)} candidates ({pids}) — adopt one by pid"
                )
            info = candidates[0]

            # An adopted process was started from a shell, so it may share that
            # shell's process group. Signalling that group would hit the shell
            # (and anything else in it) too, so fall back to signalling just the
            # leader plus its descendants when the group is not exclusively ours.
            exclusive = (
                all(
                    m.pid == info.pid or m.ppid != 1
                    for m in proc.group_members(info.pgid)
                )
                and info.pgid == info.pid
            )

            run = Running(
                pid=info.pid,
                pgid=info.pgid if exclusive else info.pid,
                starttime=info.starttime,
                boot_id=proc.boot_id(),
                started_at=time.time() - proc.age_sec(info),
                log_path="",  # started outside the dashboard: no captured log
                argv=proc.cmdline(info.pid),
                adopted=True,
            )
            self._running[job_id] = run
            self._last_exit.pop(job_id, None)
            self._write_state(job_id, run)
            proc.invalidate()
            return run

    def system_info(self, *, fresh: bool = False) -> dict[str, Any]:
        """Spool state: the one piece of host state the panel acts on.

        Deliberately no host load or disk figures: Grafana already owns host
        metrics, and duplicating them here would mean walking the disk on a
        timer for numbers that are better somewhere else.

        The spool walk is cached because while a recording is running that
        directory is written to continuously, and stat-ing it every couple of
        seconds would put the dashboard in the recorder's I/O path.
        """
        spool = Path(self.subs["project_dir"]) / SPOOL_DIRNAME
        now = time.monotonic()
        if fresh or now - self._spool_cache[0] > _SPOOL_TTL:
            count = total = 0
            if spool.is_dir():
                for path in spool.rglob("*"):
                    try:
                        if path.is_file():
                            count += 1
                            total += path.stat().st_size
                    except OSError:
                        continue
                    if count > 20_000:  # pathological spool; stop walking
                        break
            self._spool_cache = (now, count, total)
        _, count, total = self._spool_cache
        return {
            "host": socket.gethostname(),
            "spool_files": count,
            "spool_mb": round(total / 1e6, 1),
            "spool_path": str(spool),
        }

    def _find_strays(
        self, job: Job, snap: list[proc.ProcInfo] | None = None
    ) -> list[proc.ProcInfo]:
        """Instances of ``job`` that this supervisor is not tracking."""
        tracked = self._tracked_pgids()
        out = []
        for info in snap if snap is not None else proc.snapshot():
            if info.pgid in tracked or info.pid == os.getpid():
                continue
            if _matches(proc.cmdline_of(info), job.match):
                out.append(info)
        return out

    def sweep(self, *, kill: bool = False) -> list[dict[str, Any]]:
        """Find (and optionally kill) processes no tracked job accounts for.

        This is the answer to "something is still running and I can't find it".
        Detection is signature-based and scoped to ``project_dir``, so an
        unrelated ``ffmpeg`` elsewhere on the host is never a candidate.
        """
        with self._lock:
            self.reload_config()
            tracked = self._tracked_pgids()
            jobs = list(self.jobs)

        project_dir = self.subs["project_dir"]
        self_pgid = os.getpgrp()
        found: list[dict[str, Any]] = []

        snap = proc.snapshot()
        host_uptime = proc.uptime_sec()
        for info in snap:
            if info.pid in (os.getpid(), 1) or info.pgid in (self_pgid, *tracked):
                continue
            argv = proc.cmdline_of(info)
            if not argv:
                continue
            joined = " ".join(argv)
            work_dir = proc.cwd(info.pid)
            if work_dir and not work_dir.startswith(project_dir):
                continue

            kind = risk = None
            for job in jobs:
                if _matches(argv, job.match):
                    kind, risk = f"{job.name} (untracked)", "high"
                    break
                if any(c in argv[0] for c in job.collateral) and project_dir in joined:
                    kind, risk = f"orphaned {Path(argv[0]).name}", "low"
                    break
            if kind is None and "multiprocessing" in joined and info.ppid == 1:
                # A worker whose parent died. The recorder's child and the
                # teleop robot worker both self-exit on parent-pipe EOF, so one
                # lingering here means that path wedged.
                kind, risk = "orphaned worker process", "high"
            if kind is None and SPOOL_DIRNAME in joined:
                kind, risk = "orphaned encoder", "low"
            if kind is None:
                continue

            found.append(
                {
                    "pid": info.pid,
                    "ppid": info.ppid,
                    "pgid": info.pgid,
                    "kind": kind,
                    "risk": risk,
                    "age_sec": round(proc.age_sec(info, host_uptime), 1),
                    "rss_mb": round(info.rss_kb / 1024, 1),
                    "cmd": joined[:400],
                    "killed": False,
                }
            )

        if kill:
            for item in found:
                _quiet_kill(item["pid"], signal.SIGTERM)
            time.sleep(3.0)
            for item in found:
                if proc.alive(item["pid"]):
                    _quiet_kill(item["pid"], signal.SIGKILL)
                item["killed"] = not proc.alive(item["pid"])
        return found

    # -- status -----------------------------------------------------------

    def _reap(self, job_id: str) -> None:
        """Drop a tracked job that has already exited (adopted ones especially)."""
        run = self._running.get(job_id)
        if run is not None and not run.alive():
            del self._running[job_id]
            # Clear any in-flight stop too. When a graceful stop runs out its
            # grace on a job that forbids auto-escalation, _stop_worker parks
            # the request in _stopping and exits its thread. If the job then
            # goes away on its own, nothing else was clearing that entry, so
            # the job stayed "stopping" forever and start() kept refusing --
            # leaving no way to launch it again short of restarting the panel.
            self._stopping.pop(job_id, None)
            self._write_state(job_id, None)
            self._last_exit.setdefault(
                job_id,
                {
                    "exit_code": None,
                    "exited_at": time.time(),
                    "reason": "exited (adopted process — exit code unavailable)",
                    "log_path": run.log_path,
                    "ran_for": time.time() - run.started_at,
                },
            )

    def status(self) -> dict[str, Any]:
        """Full snapshot for the dashboard: one poll, everything it renders."""
        with self._lock:
            try:
                self.reload_config()
                config_error = None
            except ConfigError as exc:
                config_error = str(exc)
            jobs = list(self.jobs)
            raw_common = (load_raw(self.config_path).get("common_env") or {})
            common_keys = set(raw_common)
            # Exactly one fresh /proc walk per status call, shared by every
            # job below. One walk per poll is the intended cost; one walk per
            # job per poll was not, and a stale walk is worse than either.
            snap = proc.snapshot(max_age=proc.SNAPSHOT_TTL)
            for job in jobs:
                self._reap(job.id)

            out = []
            for job in jobs:
                run = self._running.get(job.id)
                progress = self._stopping.get(job.id)
                entry: dict[str, Any] = {
                    "id": job.id,
                    "name": job.name,
                    "command": job.command_line,
                    "env": {k: v for k, v in job.env.items() if k not in ("PATH",)},
                    "links": job.links,
                    "force_warning": job.stop.force_warning,
                    "auto_force": job.stop.auto_force,
                    "grace_sec": job.stop.grace_sec,
                    "state": "stopped",
                    "last_exit": self._last_exit.get(job.id),
                    "last_cleanup": self.last_cleanup.get(job.id) or [],
                }
                if run is not None:
                    members = proc.group_members(run.pgid, snap)
                    entry.update(
                        state="stopping" if progress else "running",
                        pid=run.pid,
                        pgid=run.pgid,
                        adopted=run.adopted,
                        uptime_sec=round(time.time() - run.started_at, 1),
                        children=max(0, len(members) - 1),
                        rss_mb=round(sum(m.rss_kb for m in members) / 1024, 1),
                        log_path=run.log_path,
                        started_command=" ".join(shlex.quote(a) for a in run.argv),
                    )
                if progress is not None:
                    # A stop is not finished when the leader exits: the worker
                    # still reaps leftover children, and start() refuses until
                    # that completes. Reporting "stopped" here would enable the
                    # dashboard's Start button a moment before a start can
                    # actually succeed.
                    entry["state"] = "stopping"
                    entry["stop_progress"] = {
                        "phase": progress.phase,
                        "detail": progress.detail,
                        "elapsed_sec": round(time.time() - progress.started_at, 1),
                        "remaining_sec": (
                            round(max(0.0, progress.deadline - time.time()), 1)
                            if progress.deadline
                            else None
                        ),
                        "needs_force": progress.needs_force,
                    }
                if run is None and not progress:
                    strays = self._find_strays(job, snap)
                    if strays:
                        entry["strays"] = [p.pid for p in strays]
                out.append(entry)

        return {
            "jobs": out,
            "links": self.links,
            # The UI hides inherited variables in its per-job env editor. It
            # used to carry its own copy of this list, which drifted: a variable
            # added to common_env showed up as if it were job-specific, and
            # saving the form copied it into every job.
            "common_env_keys": sorted(common_keys),
            "system": self.system_info(),
            "config_error": config_error,
            "project_dir": self.subs["project_dir"],
            # Which robot cell this is, straight from the repo .env. Shown in
            # the header so a browser tab left open on one deployment cannot be
            # mistaken for another.
            "robot_id": robot_id(self.subs["project_dir"]),
            "now": time.time(),
        }

    # -- logs -------------------------------------------------------------

    def _log_dir(self, job: Job) -> Path:
        path = LOG_DIR / job.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _new_log_path(self, job: Job) -> Path:
        return self._log_dir(job) / f"{time.strftime('%Y%m%dT%H%M%S')}.log"

    def _live_log_name(self, job_id: str) -> str | None:
        """Basename of the log a running job is writing, if any."""
        run = self._running.get(job_id)
        return Path(run.log_path).name if run and run.log_path else None

    def _prune_logs(self, job: Job) -> None:
        """Apply the retention policy, oldest first, sparing any live log."""
        keep_live = self._live_log_name(job.id)
        runs = sorted(self._log_dir(job).glob("*.log"))
        cutoff = time.time() - KEEP_DAYS * 86400
        budget = KEEP_MB * 1_000_000

        # Walk newest -> oldest so the size budget keeps the most recent runs.
        kept = 0
        doomed: list[Path] = []
        for idx, path in enumerate(reversed(runs)):
            try:
                stat = path.stat()
            except OSError:
                continue
            kept += stat.st_size
            if idx >= KEEP_RUNS - 1 or stat.st_mtime < cutoff or kept > budget:
                doomed.append(path)
        for path in doomed:
            if keep_live and path.name == keep_live:
                continue
            path.unlink(missing_ok=True)

    def prune_all(self) -> None:
        """Re-apply retention across every job. Safe to call at any time."""
        with self._lock:
            jobs = list(self.jobs)
        for job in jobs:
            try:
                self._prune_logs(job)
            except OSError:
                continue

    def _prune_loop(self) -> None:
        """Keep the age limit true even when nothing is ever started."""
        while True:
            time.sleep(PRUNE_INTERVAL_S)
            try:
                self.prune_all()
            except Exception:  # noqa: BLE001 - a prune failure must not kill this
                continue

    def log_tail(
        self,
        job_id: str,
        *,
        run: str | None = None,
        offset: int | None = None,
        max_bytes: int = 200_000,
    ) -> dict[str, Any]:
        """Read a run log, incrementally when ``offset`` is given.

        Passing back the previous ``next_offset`` returns only the bytes written
        since -- typically nothing, or a few lines. That is what lets the log
        pane update every second without re-sending the whole file, without
        repainting the pane (so scrolling back stays put), and without making
        the dashboard re-read megabytes off the Jetson's disk on a timer.
        """
        job = self.job(job_id)
        with self._lock:
            current = self._running.get(job_id)
        if run:
            if "/" in run or not run.endswith(".log"):
                raise SupervisorError("bad log name")
            path = self._log_dir(job) / run
        elif current is not None and current.log_path:
            path = Path(current.log_path)
        else:
            candidates = sorted(self._log_dir(job).glob("*.log"), reverse=True)
            if not candidates:
                return {
                    "name": None,
                    "text": "",
                    "size": 0,
                    "offset": 0,
                    "next_offset": 0,
                    "reset": True,
                }
            path = candidates[0]

        try:
            size = path.stat().st_size
            with path.open("rb") as fh:
                reset = False
                if offset is None or offset > size:
                    # First read, or the file was rotated/truncated under us:
                    # send the tail and tell the client to replace what it has.
                    reset = True
                    start = max(0, size - max_bytes)
                    fh.seek(start)
                    if start:
                        fh.readline()  # drop the partial first line
                else:
                    start = offset
                    if size - start > max_bytes:
                        # Fell too far behind (a burst of output): skip ahead
                        # rather than ship the backlog.
                        reset = True
                        start = size - max_bytes
                    fh.seek(start)
                    if reset and start:
                        fh.readline()
                data = fh.read()
        except OSError as exc:
            return {
                "name": path.name,
                "text": f"(cannot read log: {exc})",
                "size": 0,
                "offset": 0,
                "next_offset": 0,
                "reset": True,
            }

        return {
            "name": path.name,
            "text": data.decode("utf-8", "replace"),
            "size": size,
            "offset": start,
            "next_offset": size,
            "reset": reset,
        }


def _matches(argv: list[str], patterns: list[str]) -> bool:
    """True when ``argv`` *runs* one of ``patterns``, not merely mentions it.

    The test is "some argument IS this script", not "this string appears
    somewhere in the command line". A shell running ``pgrep -af gcs_recorder.py``
    carries the whole pipeline in a single ``-c`` argument, so a substring test
    would flag the shell as a stray copy of the recorder and offer to kill it.
    """
    for pattern in patterns:
        for arg in argv:
            if arg == pattern or arg.endswith("/" + pattern):
                return True
    return False


def _exit_reason(code: int) -> str:
    """Human-readable cause for an exit we did not ask for."""
    if code == 0:
        return "exited normally"
    if code < 0:
        return f"killed by signal {-code} ({signal.Signals(-code).name})"
    if code == 75:
        return "exited 75 — camera/native failure, safe to restart"
    if code > 128:
        return f"killed by signal {code - 128}"
    return f"crashed with exit code {code}"


def _quiet_kill(pid: int, sig: int) -> None:
    """Signal a PID, tolerating the race where it exits first."""
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
