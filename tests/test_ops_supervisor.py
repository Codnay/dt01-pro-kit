"""Tests for the ops supervisor's process lifecycle.

These exercise the parts that are dangerous to get wrong on a robot host:
that a stop reaches the *whole* process tree (not just the parent, which is how
``ffmpeg`` encoders kept surviving the old kill script), that a job which
refuses to die is escalated only as far as its policy allows, and that a stale
PID can never be signalled after the process it named has gone.

Every process spawned here is a synthetic stand-in built from ``sleep`` and a
tiny Python worker -- nothing imports the robot stack or touches hardware.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops import proc  # noqa: E402
from ops.config import (  # noqa: E402
    ConfigError,
    parse,
    parse_actions,
    parse_links,
)
from ops.supervisor import Supervisor, SupervisorError  # noqa: E402

# A stand-in for the supervised stacks: forks two children (like the recorder's
# multiprocessing child and its ffmpeg encoders) and, by default, exits on
# SIGINT the way both real scripts do. With ignore_sigint it models a wedged
# stack that only escalation can clear.
WORKER = r"""
import os, signal, subprocess, sys, time
ignore = "ignore_sigint" in sys.argv
if ignore:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
kids = [subprocess.Popen([sys.executable, "-c",
        "import time\nwhile True: time.sleep(0.2)"]) for _ in range(2)]
print("ready", os.getpid(), flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    time.sleep(0.3)          # a short, realistic teardown
    print("clean exit", flush=True)
    sys.exit(0)
"""


def _wait(
    predicate: Callable[[], object], timeout: float = 20.0, step: float = 0.1
) -> bool:
    """Poll until ``predicate`` holds; returns whether it ever did."""
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


class OpsSupervisorTestCase(unittest.TestCase):
    """Base: a throwaway ops dir with a config pointing at synthetic jobs."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # Unique per test: stray detection is host-wide by design, so a worker
        # leaked by an earlier test would otherwise look like this test's job.
        self.tag = f"opsworker_{uuid.uuid4().hex[:10]}.py"
        self.worker = root / self.tag
        self.worker.write_text(WORKER)

        self.config_path = root / "jobs.json"
        self._write_config()

        # Redirect the module-level state/log dirs at the temp tree so tests
        # never touch a real deployment's records.
        from ops import supervisor as sup_mod

        self._saved = (sup_mod.STATE_DIR, sup_mod.LOG_DIR)
        sup_mod.STATE_DIR = root / "state"
        sup_mod.LOG_DIR = root / "logs"
        self.sup = Supervisor(self.config_path)
        self.addCleanup(self._teardown)

    def _teardown(self) -> None:
        # SIGKILL by name, not by tracked state: the escalation tests leave a
        # deliberately unkillable-by-signal worker behind, and a leaked one
        # would poison every later test on this host.
        for info in list(proc.iter_procs()):
            argv = proc.cmdline(info.pid)
            if any(a.endswith(self.tag) for a in argv):
                try:
                    os.killpg(os.getpgid(info.pid), signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(info.pid, signal.SIGKILL)
                    except OSError:
                        pass
        _wait(lambda: not self._leftovers(), 10)
        from ops import supervisor as sup_mod

        sup_mod.STATE_DIR, sup_mod.LOG_DIR = self._saved
        self.tmp.cleanup()

    def _start_ready(self, job_id: str) -> Any:
        """Start a job and wait until it is genuinely up.

        The log file is non-empty the instant it is created (the supervisor
        writes a header), so file size proves nothing. The worker prints
        ``ready`` only after installing its signal handlers and spawning its
        children, which is exactly the state every test below assumes.
        """
        run = self.sup.start(job_id)
        self.assertTrue(
            _wait(lambda: "ready" in Path(run.log_path).read_text(), 30),
            f"{job_id} never reported ready; log: "
            f"{Path(run.log_path).read_text()!r}",
        )
        return run

    def _wait_stopped(self, job_id: str, timeout: float = 30.0) -> None:
        """Wait until the supervisor reports the job fully stopped.

        The leader exiting is not the end of a stop: the worker still reaps any
        leftover children before releasing the job. Start is refused until that
        finishes, which is exactly what the dashboard's disabled Start button
        reflects, so tests must wait for the reported state rather than for the
        process to disappear.
        """
        ok = _wait(
            lambda: next(
                j for j in self.sup.status()["jobs"] if j["id"] == job_id
            )["state"]
            == "stopped",
            timeout,
        )
        self.assertTrue(ok, f"{job_id} never reached the stopped state")

    def _leftovers(self) -> list[int]:
        """PIDs of this test's synthetic processes that are still alive."""
        return [
            info.pid
            for info in proc.iter_procs()
            if any(a.endswith(self.tag) for a in proc.cmdline(info.pid))
        ]

    def _write_config(self, **overrides: Any) -> None:
        job = {
            "id": "good",
            "name": "Good job",
            "argv": [sys.executable, "-u", str(self.worker)],
            "stop": {"signal": "SIGINT", "grace_sec": 10, "auto_force": True},
            "match": [self.tag],
        }
        job.update(overrides)
        wedged = dict(job)
        wedged.update(
            id="wedged",
            name="Wedged job",
            argv=[sys.executable, "-u", str(self.worker), "ignore_sigint"],
            stop={"signal": "SIGINT", "grace_sec": 2, "auto_force": False},
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "project_dir": str(Path(self.tmp.name)),
                    "python": sys.executable,
                    "jobs": [job, wedged],
                    "actions": [
                        {
                            "id": "ok",
                            "name": "Says hello",
                            "argv": [sys.executable, "-c", "print('hello')"],
                            "requires_arms_stopped": False,
                        },
                        {
                            "id": "fails",
                            "name": "Fails",
                            "argv": [
                                sys.executable, "-c",
                                "import sys; sys.stderr.write('nope\\n');"
                                " sys.exit(3)",
                            ],
                            "requires_arms_stopped": False,
                        },
                        {
                            "id": "slow",
                            "name": "Hangs",
                            "argv": [
                                sys.executable, "-c",
                                "import time; print('start', flush=True);"
                                " time.sleep(60)",
                            ],
                            "requires_arms_stopped": False,
                            "timeout_sec": 5,
                        },
                        {
                            "id": "guarded",
                            "name": "Needs the arms stopped",
                            "argv": [sys.executable, "-c", "print('ran')"],
                        },
                    ],
                },
                indent=2,
            )
        )


class TestStartAndStop(OpsSupervisorTestCase):
    """The happy path, which is also the safety-critical path."""

    def test_start_puts_the_job_in_its_own_process_group(self) -> None:
        run = self._start_ready("good")
        self.addCleanup(lambda: None)
        info = proc.read(run.pid)
        assert info is not None
        # Its own session leader => children inherit the group => a group
        # signal reaches all of them. This is what the old pgrep approach missed.
        self.assertEqual(info.pgid, run.pid)
        self.assertEqual(info.sid, run.pid)
        self.assertTrue(_wait(lambda: len(proc.group_members(run.pgid)) >= 3))

    def test_stop_reaps_the_entire_process_tree(self) -> None:
        run = self._start_ready("good")
        self.assertTrue(_wait(lambda: len(proc.group_members(run.pgid)) >= 3))
        members = [p.pid for p in proc.group_members(run.pgid)]

        self.sup.stop("good")
        self.assertTrue(_wait(lambda: not run.alive(), 20))
        # Not one survivor: the children are the whole point of the exercise.
        self.assertTrue(
            _wait(lambda: not any(proc.alive(pid) for pid in members), 20),
            f"leftover processes: {[p for p in members if proc.alive(p)]}",
        )

    def test_stop_uses_sigint_so_teardown_actually_runs(self) -> None:
        run = self._start_ready("good")
        self.sup.stop("good")
        self.assertTrue(_wait(lambda: not run.alive(), 20))
        # The worker only prints this from its KeyboardInterrupt handler, so
        # seeing it proves the job unwound rather than being killed outright.
        self.assertIn("clean exit", Path(run.log_path).read_text())

    def test_status_reports_running_then_stopped(self) -> None:
        self._start_ready("good")
        entry = next(j for j in self.sup.status()["jobs"] if j["id"] == "good")
        self.assertEqual(entry["state"], "running")
        self.assertGreaterEqual(entry["children"], 2)

        self.sup.stop("good")
        self.assertTrue(
            _wait(
                lambda: next(j for j in self.sup.status()["jobs"] if j["id"] == "good")[
                    "state"
                ]
                == "stopped",
                20,
            )
        )

    def test_double_start_is_refused(self) -> None:
        self._start_ready("good")
        with self.assertRaises(SupervisorError):
            self.sup.start("good")


class TestEscalation(OpsSupervisorTestCase):
    """What happens when a job will not go quietly."""

    def test_job_without_auto_force_stalls_instead_of_escalating(self) -> None:
        run = self._start_ready("wedged")
        self.sup.stop("wedged")
        # auto_force is false, so the supervisor must give up and ask rather
        # than SIGKILL. On the real teleop stack that restraint is what keeps
        # the arm out of a latched E-stop.
        self.assertTrue(
            _wait(
                lambda: self.sup._stopping.get("wedged")
                and self.sup._stopping["wedged"].needs_force,
                20,
            )
        )
        self.assertTrue(run.alive(), "must NOT have been killed automatically")

        entry = next(j for j in self.sup.status()["jobs"] if j["id"] == "wedged")
        self.assertTrue(entry["stop_progress"]["needs_force"])

    def test_force_is_refused_before_a_graceful_stop_is_tried(self) -> None:
        run = self._start_ready("wedged")
        # The interlock lives in the supervisor, not the UI, so this holds for
        # the HTTP API and the CLI too. On the real teleop job it is what stops
        # a reflexive "force" click from stranding the arm in E-stop.
        with self.assertRaises(SupervisorError) as ctx:
            self.sup.stop("wedged", force=True)
        self.assertIn("graceful stop has not been tried", str(ctx.exception))
        self.assertTrue(run.alive())

    def test_force_stop_kills_the_wedged_tree(self) -> None:
        run = self._start_ready("wedged")
        self.assertTrue(_wait(lambda: len(proc.group_members(run.pgid)) >= 3))
        members = [p.pid for p in proc.group_members(run.pgid)]
        self.sup.stop("wedged")
        self.assertTrue(
            _wait(
                lambda: (self.sup._stopping.get("wedged") or None)
                and self.sup._stopping["wedged"].needs_force,
                20,
            )
        )
        self.sup.stop("wedged", force=True)
        self.assertTrue(_wait(lambda: not run.alive(), 20))
        self.assertTrue(_wait(lambda: not any(proc.alive(pid) for pid in members), 20))


class TestStalledStopRecovery(OpsSupervisorTestCase):
    """A stop that outlives its grace must never trap the job."""

    def _stall(self) -> object:
        """Start the wedged job and drive it into the needs-force state."""
        run = self._start_ready("wedged")
        self.sup.stop("wedged")
        self.assertTrue(
            _wait(
                lambda: (self.sup._stopping.get("wedged") or None)
                and self.sup._stopping["wedged"].needs_force,
                20,
            )
        )
        return run

    def test_job_can_be_started_again_once_it_finally_exits(self) -> None:
        run = self._stall()
        # The exact trap: a stop parks in needs-force, then the job goes away on
        # its own. Nothing was clearing that parked request, so the job stayed
        # "stopping" forever and could never be started again.
        os.killpg(run.pgid, signal.SIGKILL)
        self.assertTrue(_wait(lambda: not run.alive(), 20))
        self._wait_stopped("wedged")
        self.assertNotIn("wedged", self.sup._stopping)
        run2 = self._start_ready("wedged")   # must not raise "still stopping"
        self.assertTrue(run2.alive())
        os.killpg(run2.pgid, signal.SIGKILL)

    def test_cancel_stop_releases_without_signalling(self) -> None:
        run = self._stall()
        self.sup.cancel_stop("wedged")
        # Nothing was signalled: the job is still up, and the panel is free.
        self.assertTrue(run.alive(), "cancel_stop must not kill anything")
        self.assertNotIn("wedged", self.sup._stopping)
        entry = next(j for j in self.sup.status()["jobs"] if j["id"] == "wedged")
        self.assertEqual(entry["state"], "running")

    def test_cancel_stop_is_refused_while_the_grace_is_still_running(self) -> None:
        run = self._start_ready("wedged")
        self.sup.stop("wedged")
        with self.assertRaises(SupervisorError):
            self.sup.cancel_stop("wedged")   # still inside its grace window
        os.killpg(run.pgid, signal.SIGKILL)


class TestActions(OpsSupervisorTestCase):
    """One-shot commands: output, failure, timeout, and the arm interlock."""

    def test_output_and_exit_code_are_reported(self) -> None:
        res = self.sup.run_action("ok")
        self.assertTrue(res["ok"])
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("hello", res["output"])

    def test_failure_is_surfaced_not_swallowed(self) -> None:
        res = self.sup.run_action("fails")
        self.assertFalse(res["ok"])
        self.assertEqual(res["exit_code"], 3)
        # stderr is where these scripts explain themselves, so it must survive.
        self.assertIn("nope", res["output"])

    def test_a_hanging_action_is_killed_and_says_so(self) -> None:
        res = self.sup.run_action("slow")
        self.assertIsNone(res["exit_code"])
        self.assertFalse(res["ok"])
        # A silent timeout on a recovery step reads as success; it must not.
        self.assertIn("timed out", res["output"])
        self.assertIn("start", res["output"], "partial output should survive")

    def test_action_is_refused_while_an_arm_job_runs(self) -> None:
        run = self._start_ready("wedged")   # uses_can, since auto_force is off
        with self.assertRaises(SupervisorError) as ctx:
            self.sup.run_action("guarded")
        self.assertIn("stop", str(ctx.exception).lower())
        # An action that does not touch the arms is unaffected.
        self.assertTrue(self.sup.run_action("ok")["ok"])
        os.killpg(run.pgid, signal.SIGKILL)

    def test_blocked_state_is_visible_before_clicking(self) -> None:
        run = self._start_ready("wedged")
        listed = {a["id"]: a for a in self.sup.action_list()}
        self.assertTrue(listed["guarded"]["blocked_by"])
        self.assertFalse(listed["ok"]["blocked_by"])
        os.killpg(run.pgid, signal.SIGKILL)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(SupervisorError):
            self.sup.run_action("nope")

    def test_duplicate_action_ids_are_rejected(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["actions"].append(dict(raw["actions"][0]))
        with self.assertRaises(ConfigError):
            parse_actions(raw, parse(raw)[1], {})


class TestAdoptAndSweep(OpsSupervisorTestCase):
    """Recovering processes the dashboard did not start."""

    def test_untracked_instance_is_reported_and_adoptable(self) -> None:
        import subprocess

        stray = subprocess.Popen(
            [sys.executable, "-u", str(self.worker)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.addCleanup(stray.wait)
        self.addCleanup(lambda: _kill_tree(stray.pid))
        self.assertTrue(_wait(lambda: len(proc.descendants(stray.pid)) >= 2, 30))

        entry = next(j for j in self.sup.status()["jobs"] if j["id"] == "good")
        self.assertIn(stray.pid, entry.get("strays", []))

        run = self.sup.adopt("good", stray.pid)
        self.assertEqual(run.pid, stray.pid)
        self.assertTrue(run.adopted)
        entry = next(j for j in self.sup.status()["jobs"] if j["id"] == "good")
        self.assertEqual(entry["state"], "running")

        # And once adopted it is stoppable, which is the whole point.
        self.sup.stop("good")
        self.assertTrue(_wait(lambda: not run.alive(), 20))

    def test_state_survives_its_directory_being_deleted(self) -> None:
        from ops import supervisor as sup_mod

        run = self._start_ready("good")
        self.assertTrue(self.sup._state_path("good").exists())
        # Mimic a `git clean` wiping the repo-local state dir between starts.
        shutil.rmtree(sup_mod.STATE_DIR)
        self.sup.stop("good")
        self._wait_stopped("good")
        run2 = self._start_ready("good")
        self.assertTrue(
            self.sup._state_path("good").exists(),
            "state was not re-created, so a restart could not re-adopt this job",
        )
        self.sup.stop("good")
        self.assertTrue(_wait(lambda: not run2.alive(), 20))

    def test_adopt_refuses_when_nothing_matches(self) -> None:
        with self.assertRaises(SupervisorError):
            self.sup.adopt("good")

    def test_sweep_lists_untracked_and_skips_tracked(self) -> None:
        run = self._start_ready("good")
        pids = {p["pid"] for p in self.sup.sweep(kill=False)}
        # A job the supervisor is tracking is accounted for, so it is not a stray.
        self.assertNotIn(run.pid, pids)


class TestShippedConfig(unittest.TestCase):
    """Guards on the shipped ops/jobs.example.json, not the test config.

    The live ``jobs.json`` is per-host and gitignored, so the template is the
    only copy this repo can make promises about -- and it is what every new
    deployment starts from.
    """

    def setUp(self) -> None:
        from ops import config as config_mod

        self.raw = config_mod.load_raw(config_mod.EXAMPLE_PATH)

    def test_template_is_valid_config(self) -> None:
        # A fresh robot runs whatever this file says. A typo here breaks the
        # dashboard on a machine nobody has SSH'd into yet.
        from ops import config as config_mod

        project = os.path.expanduser(self.raw["project_dir"])
        if not os.path.isdir(project):
            # parse() insists project_dir exists; on a dev box that has the
            # checkout somewhere else, that says nothing about the template.
            self.skipTest(f"template project_dir not present here: {project}")
        jobs, _ = config_mod.parse(self.raw)
        self.assertTrue(jobs)

    def test_template_leaves_robot_id_to_dotenv(self) -> None:
        # Two cells copying this file must not both claim one robot_id; the id
        # comes from .env, which is per-host and not in git. A ROBOT_ID pinned
        # in the job env would silently win over it for that job alone, so the
        # arm and its telemetry would disagree about which robot they are.
        envs = [self.raw.get("common_env") or {}]
        envs += [job.get("env") or {} for job in self.raw["jobs"]]
        for env in envs:
            self.assertNotIn("ROBOT_ID", env)

    def test_captured_logs_keep_their_ansi_colour(self) -> None:
        # configure_logging() decides colour from sys.stdout.isatty(), and the
        # supervisor redirects stdout to a file, so without this the captured
        # log is plain and the dashboard has nothing to colourise.
        self.assertEqual(self.raw["common_env"].get("PIPER_LOG_COLOR"), "1")

    def test_logs_are_unbuffered_and_utf8(self) -> None:
        # Buffered output would arrive in 8 KB bursts instead of streaming, and
        # a systemd unit does not inherit a login shell's UTF-8 locale.
        self.assertEqual(self.raw["common_env"].get("PYTHONUNBUFFERED"), "1")
        self.assertEqual(self.raw["common_env"].get("PYTHONIOENCODING"), "utf-8")

    def test_arm_driving_jobs_never_auto_escalate(self) -> None:
        # The interlock is only as good as the config behind it: any job that
        # commands an arm must refuse automatic SIGKILL.
        for job in self.raw["jobs"]:
            argv = " ".join(job["argv"])
            if "arms.cameras" in argv:
                continue
            self.assertFalse(
                job["stop"].get("auto_force", False),
                f"job {job['id']!r} drives an arm but allows automatic escalation",
            )
            self.assertEqual(job["stop"].get("signal", "SIGINT"), "SIGINT")


class TestLogRetention(OpsSupervisorTestCase):
    """Old run logs must actually go away, and live ones must not."""

    def test_logs_past_the_age_limit_are_deleted(self) -> None:
        from ops import supervisor as sup_mod

        log_dir = sup_mod.LOG_DIR / "good"
        log_dir.mkdir(parents=True, exist_ok=True)
        old_log = log_dir / "20200101T000000.log"
        old_log.write_text("ancient\n")
        stale = time.time() - (sup_mod.KEEP_DAYS + 1) * 86400
        os.utime(old_log, (stale, stale))
        recent = log_dir / "20991231T000000.log"
        recent.write_text("recent\n")

        self.sup.prune_all()
        self.assertFalse(old_log.exists(), "log past the age limit survived")
        self.assertTrue(recent.exists(), "a recent log was deleted")

    def test_a_running_jobs_log_is_never_pruned(self) -> None:
        from ops import supervisor as sup_mod

        run = self._start_ready("good")
        live = Path(run.log_path)
        # Backdate the live log well past the limit: pruning it would truncate
        # the record of a session that is still going.
        stale = time.time() - (sup_mod.KEEP_DAYS + 5) * 86400
        os.utime(live, (stale, stale))
        self.sup.prune_all()
        self.assertTrue(live.exists(), "pruned the log of a running job")


class TestIdentityAndConfig(OpsSupervisorTestCase):
    """Guards against signalling the wrong process, and bad config."""

    def test_stale_state_from_a_previous_boot_is_discarded(self) -> None:
        run = self._start_ready("good")
        state = self.sup._state_path("good")
        data = json.loads(state.read_text())
        data["boot_id"] = "00000000-0000-0000-0000-000000000000"
        state.write_text(json.dumps(data))

        fresh = Supervisor(self.config_path)
        # A PID recorded before a reboot names a different process now, so the
        # replacement supervisor must not adopt (and later signal) it.
        self.assertNotIn("good", fresh._running)
        _kill_tree(run.pid)

    def test_pid_reuse_cannot_be_signalled(self) -> None:
        run = self._start_ready("good")
        info = proc.read(run.pid)
        assert info is not None
        real_start = info.starttime
        run.starttime = real_start + 1  # simulate a different process, same pid
        self.assertFalse(run.alive())
        # alive() is false, so _signal_group is a no-op and the live tree stays up.
        self.sup._signal_group(run, signal.SIGKILL)
        self.assertTrue(proc.alive(run.pid))
        _kill_tree(run.pid)

    def test_supervisor_refuses_to_signal_its_own_group(self) -> None:
        run = self._start_ready("good")
        run.pgid = os.getpgrp()
        with self.assertRaises(SupervisorError):
            self.sup._signal_group(run, signal.SIGTERM)

    def test_config_rejects_sigkill_as_a_stop_signal(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["jobs"][0]["stop"]["signal"] = "SIGKILL"
        with self.assertRaises(ConfigError):
            parse(raw)

    def test_config_rejects_an_empty_argv(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["jobs"][0]["argv"] = []
        with self.assertRaises(ConfigError):
            parse(raw)

    def test_header_links_accept_a_url_or_a_port(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["links"] = [
            {"label": "Camera dashboard", "url": "https://example.invalid/"},
            {"label": "Preset", "port": 8082},
        ]
        links = parse_links(raw, parse(raw)[1])
        self.assertEqual(links[0]["url"], "https://example.invalid/")
        # A port stays a port: the browser resolves it against whatever address
        # the panel is open on, so one entry works on Tailscale and the LAN.
        self.assertEqual(links[1]["port"], 8082)
        self.assertNotIn("url", links[1])

    def test_a_link_without_url_or_port_is_rejected(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["links"] = [{"label": "Nowhere"}]
        with self.assertRaises(ConfigError):
            parse_links(raw, parse(raw)[1])

    def test_config_expands_placeholders(self) -> None:
        raw = json.loads(self.config_path.read_text())
        raw["jobs"][0]["argv"] = ["{python}", "{project_dir}/worker.py"]
        jobs, subs = parse(raw)
        self.assertEqual(jobs[0].argv[0], sys.executable)
        self.assertTrue(jobs[0].argv[1].endswith("worker.py"))
        self.assertNotIn("{", jobs[0].argv[1])


def _kill_tree(pid: int) -> None:
    """Best-effort cleanup for processes a test deliberately left running."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        pass


class TestPerDeploymentConfig(unittest.TestCase):
    """jobs.json is per-robot: seeded from the template, never shared."""

    def setUp(self) -> None:
        from ops import config as config_mod

        self.config_mod = config_mod
        self.root = Path(tempfile.mkdtemp(prefix="ops-seed-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.example = self.root / "jobs.example.json"
        self.live = self.root / "jobs.json"
        self.example.write_text(json.dumps({
            "project_dir": str(self.root),
            "python": sys.executable,
            "jobs": [{
                "id": "demo",
                "name": "Demo",
                "argv": ["{python}", "-c", "pass"],
            }],
        }))
        self._patch("EXAMPLE_PATH", self.example)
        self._patch("CONFIG_PATH", self.live)

    def _patch(self, name: str, value: Any) -> None:
        old = getattr(self.config_mod, name)
        setattr(self.config_mod, name, value)
        self.addCleanup(setattr, self.config_mod, name, old)

    def test_missing_config_is_seeded_from_the_template(self) -> None:
        # A fresh clone has no jobs.json (it is gitignored). Without seeding the
        # dashboard would refuse to start on a robot nobody has logged into.
        raw = self.config_mod.load_raw(self.live)
        self.assertTrue(self.live.is_file())
        self.assertEqual(raw["jobs"][0]["id"], "demo")

    def test_seeding_never_overwrites_a_local_config(self) -> None:
        # The whole point: this host's camera serials survive everything,
        # including a template that changed upstream.
        self.live.write_text(json.dumps({
            "project_dir": str(self.root),
            "python": sys.executable,
            "jobs": [{"id": "mine", "name": "Mine", "argv": ["{python}"]}],
        }))
        raw = self.config_mod.load_raw(self.live)
        self.assertEqual(raw["jobs"][0]["id"], "mine")

    def test_saving_from_the_ui_touches_only_the_local_copy(self) -> None:
        # If a UI edit rewrote the tracked template, every pull would conflict
        # again -- which is the bug this split exists to remove.
        before = self.example.read_text()
        raw = self.config_mod.load_raw(self.live)
        raw["jobs"][0]["argv"] = [sys.executable, "-c", "pass  # edited"]
        self.config_mod.save_raw(raw, self.live)
        self.assertEqual(self.example.read_text(), before)
        self.assertIn("edited", self.live.read_text())


class TestRobotIdentity(unittest.TestCase):
    """The header's robot id: read from .env, never guessed."""

    def setUp(self) -> None:
        from ops import config as config_mod

        self.config_mod = config_mod
        self.root = Path(tempfile.mkdtemp(prefix="ops-rid-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        config_mod._dotenv_cache = None
        self.addCleanup(setattr, config_mod, "_dotenv_cache", None)
        self.env_before = os.environ.get("ROBOT_ID")
        self.addCleanup(self._restore_env)
        os.environ.pop("ROBOT_ID", None)

    def _restore_env(self) -> None:
        if self.env_before is None:
            os.environ.pop("ROBOT_ID", None)
        else:
            os.environ["ROBOT_ID"] = self.env_before

    def _dotenv(self, text: str) -> None:
        (self.root / ".env").write_text(text)
        self.config_mod._dotenv_cache = None

    def test_reads_robot_id_from_dotenv(self) -> None:
        self._dotenv("GCP_PROJECT=x\nROBOT_ID=dt01-pro-02\nCOLLECTED_BY=y\n")
        self.assertEqual(self.config_mod.robot_id(str(self.root)), "dt01-pro-02")

    def test_environment_overrides_dotenv(self) -> None:
        self._dotenv("ROBOT_ID=from-file\n")
        os.environ["ROBOT_ID"] = "from-env"
        self.assertEqual(self.config_mod.robot_id(str(self.root)), "from-env")

    def test_unset_is_reported_as_unset_not_guessed(self) -> None:
        # An unset id means this host's data lands under the robot0 default.
        # The panel has to say so; inventing a name here would hide it.
        self._dotenv("COLLECTED_BY=y\n")
        self.assertIsNone(self.config_mod.robot_id(str(self.root)))
        self.assertIsNone(self.config_mod.robot_id(str(self.root / "nope")))

    def test_junk_id_is_rejected(self) -> None:
        # The value is interpolated into an unauthenticated page and, upstream,
        # into channel names and GCS paths. Same charset rule as robot_identity.
        self._dotenv("ROBOT_ID=oops id/../x\n")
        self.assertIsNone(self.config_mod.robot_id(str(self.root)))

    def test_only_robot_id_is_read_from_dotenv(self) -> None:
        # .env also holds API keys and a service-account path, and this value is
        # served to a dashboard with no auth. Nothing else may leak out of it.
        self._dotenv("ANTHROPIC_API_KEY=sk-ant-secret\nROBOT_ID=dt01-pro-03\n")
        self.assertEqual(self.config_mod.robot_id(str(self.root)), "dt01-pro-03")
        self.assertNotIn(
            "sk-ant-secret", json.dumps(self.config_mod._dotenv_cache)
        )

    def test_a_changed_dotenv_is_picked_up(self) -> None:
        # Cached against mtime/size; editing .env must not need a restart.
        self._dotenv("ROBOT_ID=dt01-pro-01\n")
        self.assertEqual(self.config_mod.robot_id(str(self.root)), "dt01-pro-01")
        time.sleep(0.01)
        (self.root / ".env").write_text("ROBOT_ID=dt01-pro-09\n")
        self.assertEqual(self.config_mod.robot_id(str(self.root)), "dt01-pro-09")

    def test_channel_prefers_an_explicit_agora_channel(self) -> None:
        # The camera dashboard views the Agora channel, and a host pinned to a
        # legacy channel name publishes there, not under its robot id. Building
        # the link from robot_id alone would open an empty feed.
        self._dotenv("ROBOT_ID=dt01-pro-04\nAGORA_CHANNEL=legacy-cam\n")
        self.assertEqual(self.config_mod.channel(str(self.root)), "legacy-cam")

    def test_channel_falls_back_to_the_robot_id(self) -> None:
        self._dotenv("ROBOT_ID=dt01-pro-04\n")
        self.assertEqual(self.config_mod.channel(str(self.root)), "dt01-pro-04")

    def test_links_expand_the_channel_placeholder(self) -> None:
        # How the header's camera-dashboard link gets ?channel=<this cell>
        # without anyone editing a url per deployment.
        self._dotenv("ROBOT_ID=dt01-pro-05\n")
        subs = {
            "home": str(self.root),
            "project_dir": str(self.root),
            "robot_id": self.config_mod.robot_id(str(self.root)) or "",
            "channel": self.config_mod.channel(str(self.root)) or "",
        }
        links = parse_links(
            {"links": [{"label": "Camera dashboard",
                        "url": "https://cam.example/?channel={channel}"}]},
            subs,
        )
        self.assertEqual(links[0]["url"], "https://cam.example/?channel=dt01-pro-05")


class TestCanPreflight(OpsSupervisorTestCase):
    """An arm job must not start with its CAN interfaces down.

    The value is the message: without this the stack starts, fails to open a
    socket and dies with a traceback naming an interface. So the tests that
    matter most are the ones asserting it does NOT block a legitimate start.
    """

    def setUp(self) -> None:
        super().setUp()
        from ops import can as can_mod

        self.can_mod = can_mod
        self._real_discover = can_mod.discover
        self.addCleanup(setattr, can_mod, "discover", self._real_discover)

    def _adapters(self, **state: str):
        """Fake discovery: role -> "UP"/"DOWN"."""
        from ops.can import Adapter

        return [
            Adapter(iface=f"can_{role}", bus=f"3-{i}:1.0", bus_kind="usb",
                    state=st, bitrate=1_000_000, restart_ms=100, driver="gs_usb",
                    product="", serial="", role=role, errors={})
            for i, (role, st) in enumerate(state.items(), start=1)
        ]

    def _configure(self, argv_extra: list[str], roles: dict[str, str]) -> None:
        self._write_config(
            id="good",
            name="Arm job",
            argv=[sys.executable, "-u", str(self.worker), *argv_extra],
            # auto_force False is what marks a job as driving an arm.
            stop={"signal": "SIGINT", "grace_sec": 10, "auto_force": False},
            match=[self.tag],
        )
        raw = json.loads(self.config_path.read_text())
        raw["can"] = {"bitrate": 1_000_000, "roles": roles}
        self.config_path.write_text(json.dumps(raw, indent=2))
        self.sup.reload_config()

    ROLES = {"right": "can_right", "left": "can_left"}

    def test_refuses_when_one_arm_has_no_interface(self) -> None:
        self._configure(["--arm", "both"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(right="UP")
        with self.assertRaises(SupervisorError) as ctx:
            self.sup.start("good")
        msg = str(ctx.exception)
        self.assertIn("can_left", msg)          # names the interface at fault
        self.assertIn("CAN & arms", msg)        # and where to fix it
        self.assertNotIn("good", [j["id"] for j in self.sup.status()["jobs"]
                                  if j["state"] != "stopped"])

    def test_refuses_when_an_interface_exists_but_is_down(self) -> None:
        # Present but not brought up is exactly the state can_activate.sh
        # leaves behind on a half-finished assignment.
        self._configure(["--arm", "both"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(
            right="UP", left="DOWN")
        with self.assertRaises(SupervisorError):
            self.sup.start("good")

    def test_starts_when_both_arms_are_up(self) -> None:
        self._configure(["--arm", "both"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(
            right="UP", left="UP")
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_one_armed_job_does_not_need_the_other_arm(self) -> None:
        # The check that would be most annoying if wrong: running --arm right
        # must not be blocked because the left adapter is unplugged.
        self._configure(["--arm", "right"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(right="UP")
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_one_armed_job_still_needs_its_own_arm(self) -> None:
        # The other half of the single-arm case: --arm right is exempt from
        # needing can_left, not from needing can_right.
        self._configure(["--arm", "right"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(
            right="DOWN", left="UP")
        with self.assertRaises(SupervisorError) as ctx:
            self.sup.start("good")
        self.assertIn("can_right", str(ctx.exception))
        self.assertNotIn("can_left", str(ctx.exception))

    def test_a_lookalike_flag_is_not_read_as_arm(self) -> None:
        # --arm-speed=5 must not be parsed as an --arm value, or a job would be
        # preflighted against an arm that does not exist.
        self._configure(["--arm-speed=5", "--arm", "right"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(right="UP")
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_a_value_less_arm_flag_requires_every_role(self) -> None:
        # A trailing "--arm" with nothing after it tells us nothing, so fall
        # back to requiring everything rather than guessing one arm.
        self._configure(["--arm"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(right="UP")
        with self.assertRaises(SupervisorError):
            self.sup.start("good")

    def test_arm_equals_form_is_understood(self) -> None:
        self._configure(["--arm=left"], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(left="UP")
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_missing_arm_flag_requires_every_role(self) -> None:
        # The stack defaults --arm to both, so an argv without the flag needs
        # both interfaces. Guessing narrower would let a doomed start through.
        self._configure([], self.ROLES)
        self.can_mod.discover = lambda roles=None: self._adapters(right="UP")
        with self.assertRaises(SupervisorError):
            self.sup.start("good")

    def test_unreadable_can_state_lets_the_start_through(self) -> None:
        # Fail open, deliberately. Refusing when `ip` is broken would leave no
        # way to run the robot at all, and the stack still fails loudly if CAN
        # really is missing.
        self._configure(["--arm", "both"], self.ROLES)

        def boom(roles=None):
            raise OSError("ip: command not found")

        self.can_mod.discover = boom
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_non_arm_jobs_are_never_checked(self) -> None:
        # The recorder owns cameras, not the bus; a CAN fault must not stop a
        # recording session.
        self._configure(["--arm", "both"], self.ROLES)
        raw = json.loads(self.config_path.read_text())
        raw["jobs"][0]["stop"]["auto_force"] = True     # -> uses_can False
        self.config_path.write_text(json.dumps(raw, indent=2))
        self.sup.reload_config()

        def boom(roles=None):
            raise AssertionError("discovery must not run for a non-arm job")

        self.can_mod.discover = boom
        run = self._start_ready("good")
        self.assertTrue(run.alive)

    def test_no_roles_configured_means_no_check(self) -> None:
        # A deployment that has not named its interfaces cannot be preflighted.
        self._configure(["--arm", "both"], {})

        def boom(roles=None):
            raise AssertionError("discovery must not run without roles")

        self.can_mod.discover = boom
        run = self._start_ready("good")
        self.assertTrue(run.alive)


if __name__ == "__main__":
    unittest.main()
