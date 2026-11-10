"""Recovery, stale-input fencing and bounded subprocess failure tests; no hardware."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arms.livekit_rt import (  # noqa: E402
    livekit_quest_reader as reader_module,
)
from arms.livekit_rt.transport_supervisor import (  # noqa: E402
    WorkerTransport,
)
from arms.xr_control import encode_xr_frame  # noqa: E402


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.make_reader()

    def make_reader(self, input_timeout_s=reader_module.DEFAULT_INPUT_TIMEOUT_S):
        self.transport = WorkerTransport("unused", "test", "follower", lambda: "test")
        self.transport._update(state="connected", generation=1, connection_epoch=1)
        with patch.object(
            reader_module, "WorkerTransport", return_value=self.transport
        ), patch.object(self.transport, "start"), patch.object(
            reader_module.LiveKitQuestReader, "_run_loop"
        ):
            self.reader = reader_module.LiveKitQuestReader(
                url="wss://test.invalid",
                token="test",
                room="test",
                identity="follower",
                stats_period_s=0,
                input_timeout_s=input_timeout_s,
            )
        self.addCleanup(self.reader.close)
        # Keep the configured lease timeout while disabling handoff suppression.
        self.reader._lease._suppress_sec = 0
        self.reader._refresh_transport()
        self.claim(1)

    def claim(self, epoch, cid="op"):
        self.reader._on_control(
            json.dumps(dict(kind="claim", cid=cid, ep=epoch)).encode()
        )

    def pose(self, epoch=1, seq=1, trigger=1, cid="op", pose=None):
        payload = encode_xr_frame(
            {"right": (np.eye(4) if pose is None else pose, 0.5, trigger, 0.8)},
            cid=cid,
            epoch=epoch,
            seq=seq,
        )
        self.reader._on_xr(payload.encode())

    def test_outage_neutralizes_immediately_without_reader_tick(self):
        self.pose()
        self.assertEqual(self.reader.get_trigger_value("right"), 1)
        self.transport._lost(time.monotonic())
        self.assertEqual(self.reader.get_trigger_value("right"), 0)
        self.assertEqual(self.reader.get_joystick_value("right"), (0, 0))
        self.assertIsNone(self.reader.get_hand_controller_transform_ros("right"))

    def test_resume_fences_old_epoch_and_requires_new_pose(self):
        self.pose()
        self.transport._lost(time.monotonic())
        self.transport._message(
            dict(kind="state", state="connected", sid="new"), time.monotonic()
        )
        # The full loss and resume occurred between reader polls.
        self.assertFalse(self.reader.input_available("right"))
        self.reader._refresh_transport()
        self.reader._announce_lease_state()
        state = json.loads(self.transport._outbound[-1]["data"])
        self.assertEqual(state["resume_cid"], "op")
        self.assertEqual(state["ep"], 1)
        self.claim(1)
        self.pose(seq=2)
        self.assertFalse(self.reader.input_available("right"))
        self.claim(2)
        self.assertFalse(self.reader.input_available("right"))
        self.pose(epoch=2)
        self.assertTrue(self.reader.input_available("right"))

    def test_expired_lease_during_loss_still_fences_the_old_epoch(self):
        self.reader._lease._last_seen -= 10
        self.transport._lost(time.monotonic())
        self.reader._refresh_transport()
        self.claim(1)
        self.assertIsNone(self.reader.active_controller()[0])
        self.assertEqual(self.reader._epoch_floor, 1)

    def test_cached_and_duplicate_frames_expire_even_while_heartbeating(self):
        self.pose()
        self.reader._pose_at["right"] -= 0.3
        self.claim(1)
        self.pose(seq=1)
        self.assertFalse(self.reader.input_available("right"))
        self.assertEqual(self.reader.get_trigger_value("right"), 0)
        old = self.reader.input_revision()
        self.pose(seq=2)
        self.assertNotEqual(self.reader.input_revision(), old)

    def test_new_operator_cannot_reuse_previous_operators_pose(self):
        self.pose()
        self.claim(2, "other")
        self.assertFalse(self.reader.input_available("right"))
        self.pose(epoch=1, seq=2)
        self.assertFalse(self.reader.input_available("right"))
        self.pose(epoch=2, cid="other")
        self.assertTrue(self.reader.input_available("right"))

    def test_input_cannot_acquire_a_lease_and_old_buttons_do_not_fire(self):
        seen = []
        self.reader.on("button_a", lambda: seen.append(True))
        self.transport._lost(time.monotonic())
        self.reader._refresh_transport()
        self.reader._on_button(b'{"cid":"op","ep":1,"name":"a"}')
        self.pose(epoch=3)
        self.assertEqual(seen, [])
        self.assertIsNone(self.reader.active_controller()[0])

    def test_nonfinite_input_is_neutral(self):
        pose = np.eye(4)
        pose[0, 3] = np.nan
        self.pose(pose=pose)
        self.assertFalse(self.reader.input_available("right"))

    def test_ten_second_timeout_retains_input_and_expires_at_deadline(self):
        self.make_reader(input_timeout_s=10)
        self.pose()
        self.reader._pose_at["right"] -= 9.9
        self.assertTrue(self.reader.input_available("right"))
        self.assertEqual(self.reader.get_trigger_value("right"), 1)
        self.assertEqual(self.reader.get_joystick_value("right"), (0, 0.8))
        self.reader._pose_at["right"] -= 0.2
        self.assertFalse(self.reader.input_available("right"))
        self.assertEqual(self.reader.get_trigger_value("right"), 0)
        self.assertEqual(self.reader.get_joystick_value("right"), (0, 0))
        self.assertEqual(self.reader.get_connection_status()["input_timeout_s"], 10)

    def test_short_gap_keeps_clutch_revision_but_long_gap_invalidates_it(self):
        self.make_reader(input_timeout_s=10)
        self.pose()
        revision = self.reader.input_revision()
        self.reader._pose_at["right"] -= 2
        self.pose(seq=2)
        self.assertEqual(self.reader.input_revision(), revision)
        self.reader._pose_at["right"] -= 10.1
        self.pose(seq=3)
        self.assertNotEqual(self.reader.input_revision(), revision)

    def test_ten_second_timeout_still_neutralizes_transport_loss_immediately(self):
        self.make_reader(input_timeout_s=10)
        self.pose()
        self.transport._lost(time.monotonic())
        self.assertFalse(self.reader.input_available("right"))
        self.assertEqual(self.reader.get_trigger_value("right"), 0)
        self.assertEqual(self.reader.get_joystick_value("right"), (0, 0))

    def test_ten_second_timeout_does_not_keep_invalid_tracking(self):
        self.make_reader(input_timeout_s=10)
        self.pose()
        invalid = np.eye(4)
        invalid[0, 3] = np.nan
        self.pose(seq=2, pose=invalid)
        self.assertFalse(self.reader.input_available("right"))
        self.assertEqual(self.reader.get_trigger_value("right"), 0)

    def test_lease_does_not_expire_before_configured_input_timeout(self):
        self.make_reader(input_timeout_s=10)
        self.reader._lease._last_seen -= 9.9
        self.reader._lease.check_expiry()
        self.assertEqual(self.reader.active_controller()[0], "op")
        self.reader._lease._last_seen -= 0.2
        self.reader._lease.check_expiry()
        self.assertIsNone(self.reader.active_controller()[0])

    def test_input_timeout_must_be_positive_and_finite(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                reader_module.LiveKitQuestReader(input_timeout_s=value)

    def test_input_timeout_flag_parses_ten_seconds_and_keeps_default(self):
        parser = argparse.ArgumentParser()
        reader_module.add_livekit_args(parser)
        self.assertEqual(parser.parse_args([]).livekit_input_timeout, 0.25)
        self.assertEqual(
            parser.parse_args(["--livekit-input-timeout", "10"]).livekit_input_timeout,
            10,
        )

    def test_ping_and_rtt_still_work_before_taking_control(self):
        self.reader._on_control(b'{"kind":"ping","pid":"xyz"}')
        self.assertEqual(json.loads(self.transport._outbound[-1]["data"])["pid"], "xyz")
        for i in range(100):
            self.reader._on_control(json.dumps(dict(kind="ping", pid=str(i))).encode())
        self.assertGreater(self.reader._pongs_dropped, 0)
        with patch.object(reader_module, "_meta_record_rtt") as record:
            self.reader._on_control(b'{"kind":"heartbeat","cid":"op","ep":1,"rtt":238}')
            record.assert_called_once_with(0.238)

    def test_old_ipc_data_and_data_during_resume_are_dropped(self):
        self.transport._message(
            dict(kind="data", at=time.monotonic() - 0.3, data="{}"), time.monotonic()
        )
        self.assertEqual(self.transport.events(), [])
        self.transport._lost(time.monotonic())
        self.transport._message(
            dict(kind="data", at=time.monotonic(), data="{}"), time.monotonic()
        )
        self.assertEqual(self.transport.events(), [])


class WorkerTests(unittest.TestCase):
    def make(self, modes, **kwargs):
        self.mints = []

        def token():
            i = len(self.mints)
            self.mints.append(i)
            return modes[min(i, len(modes) - 1)]

        self.env = patch.dict(
            os.environ, {"PYTHONPATH": str(ROOT / "tests") + os.pathsep + str(ROOT)}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        transport = WorkerTransport(
            "unused",
            "test",
            "follower",
            token,
            worker_module="livekit_worker_fixture",
            connect_timeout=0.25,
            heartbeat_timeout=0.12,
            shutdown_timeout=0.06,
            backoff_max=0.03,
            probe_interval=0.04,
            **kwargs,
        )
        transport.start()
        self.addCleanup(transport.close)
        return transport

    def test_hung_connect_offline_start_crash_and_stuck_resume_recover(self):
        for mode in (
            "hang_connect",
            "crash",
            "resume",
            "lost_callback",
            "no_heartbeat",
        ):
            with self.subTest(mode=mode):
                t = self.make([mode, "healthy"], probe=lambda *a: True)
                self.assertTrue(
                    wait_for(
                        lambda: t.snapshot()["generation"] >= 2
                        and t.snapshot()["state"] == "connected"
                    )
                )
                self.assertGreaterEqual(len(self.mints), 2)  # fresh token every attempt
                self.assertEqual(t.snapshot()["restarts"], 1)
                t.close()

    def test_two_explicit_cloud_absences_replace_a_locally_connected_worker(self):
        t = self.make(["healthy"], probe=lambda *a: False)
        self.assertTrue(wait_for(lambda: t.snapshot()["generation"] >= 2))

    def test_api_errors_are_unknown_and_idle_room_does_not_churn(self):
        t = self.make(["healthy"], probe=lambda *a: None)
        self.assertTrue(wait_for(lambda: t.snapshot()["state"] == "connected"))
        time.sleep(0.25)
        self.assertEqual(t.snapshot()["generation"], 1)
        self.assertIsNone(t.snapshot()["last_verified_at"])

    def test_hung_membership_probe_cannot_block_worker_recovery(self):
        import threading

        blocked = threading.Event()
        self.addCleanup(blocked.set)
        t = self.make(["resume", "healthy"], probe=lambda *a: blocked.wait(5))
        self.assertTrue(
            wait_for(
                lambda: t.snapshot()["generation"] >= 2
                and t.snapshot()["state"] == "connected"
            )
        )

    def test_stop_reaps_only_owned_worker_even_when_disconnect_hangs(self):
        t = self.make(["hang_shutdown"], probe=lambda *a: True)
        self.assertTrue(wait_for(lambda: t.snapshot()["state"] == "connected"))
        proc = t._process
        start = time.monotonic()
        t.close()
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(proc.poll(), -9)
        generation = t.snapshot()["generation"]
        time.sleep(0.1)
        self.assertEqual(t.snapshot()["state"], "stopped")
        self.assertEqual(t.snapshot()["generation"], generation)

    def test_real_parent_reader_loop_receives_fresh_input_after_worker_replacement(
        self,
    ):
        t = self.make(["controller1", "controller2"], probe=lambda *a: True)
        # The fixture may send the first claim before the parent starts; keep it
        # queued, exactly as a fast real SDK connection can do at startup.
        with patch.object(
            reader_module, "WorkerTransport", return_value=t
        ), patch.object(t, "start"):
            reader = reader_module.LiveKitQuestReader(
                url="wss://test.invalid",
                token="test",
                room="test",
                identity="follower",
                stats_period_s=0,
            )
        self.addCleanup(reader.close)
        self.assertTrue(wait_for(lambda: reader.input_available("right")))
        self.assertEqual(reader.get_trigger_value("right"), 1)
        t._process.kill()
        self.assertTrue(wait_for(lambda: not reader.input_available("right")))
        self.assertTrue(
            wait_for(
                lambda: t.snapshot()["generation"] >= 2
                and reader.input_available("right")
            )
        )
        self.assertEqual(reader.active_controller()[:2], ("op", 2))

    def test_ipc_100hz_latency(self):
        t = self.make(["data"], probe=lambda *a: True)
        self.assertTrue(wait_for(lambda: t.snapshot()["state"] == "connected"))
        ages = []
        until = time.monotonic() + 2
        while time.monotonic() < until:
            t.changed.wait(0.02)
            ages.extend(
                (time.monotonic() - e["at"]) * 1000
                for e in t.events()
                if e["kind"] == "data"
            )
        self.assertGreater(len(ages), 150)
        p99 = sorted(ages)[int(0.99 * len(ages))]
        print(f"100 Hz IPC: {len(ages)} frames, p99={p99:.3f} ms")
        # Scheduler jitter is host-dependent; enforce the deployment budget in
        # the opt-in Jetson acceptance run, not on an arbitrary busy CI host.
        self.assertLess(max(ages), 250)
        if os.environ.get("LIVEKIT_REQUIRE_IPC_BUDGET") == "1":
            self.assertLess(p99, 5)


if __name__ == "__main__":
    unittest.main()
