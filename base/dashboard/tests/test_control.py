"""Tests for the one-driver-at-a-time control lease.

    python3 -m unittest discover -s dashboard/tests -t dashboard

Three properties carry the safety of this feature, and each has tests here:

* only the holder may command, and a route added later is protected by default;
* stopping is never gated, whoever asks;
* losing the lease stops the robot rather than leaving it coasting.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import control                                            # noqa: E402
import drive                                              # noqa: E402
import mock_robot                                         # noqa: E402
import server                                             # noqa: E402


class Clock:
    """Injectable time, so expiry is tested without sleeping through it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestLease(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.lease = control.ControlLease(timeout_s=8.0, clock=self.clock)

    def test_first_taker_gets_it(self) -> None:
        self.assertEqual(self.lease.take("alice", "Alice"), (True, ""))
        self.assertTrue(self.lease.is_holder("alice"))
        self.assertEqual(self.lease.status()["name"], "Alice")

    def test_a_second_person_is_refused_and_told_who_has_it(self) -> None:
        self.lease.take("alice", "Alice")
        took, reason = self.lease.take("bob", "Bob")
        self.assertFalse(took)
        self.assertIn("Alice", reason)
        self.assertTrue(self.lease.is_holder("alice"))
        self.assertFalse(self.lease.is_holder("bob"))

    def test_force_steals_it(self) -> None:
        self.lease.take("alice", "Alice")
        self.assertEqual(self.lease.take("bob", "Bob", force=True), (True, ""))
        self.assertTrue(self.lease.is_holder("bob"))
        self.assertFalse(self.lease.is_holder("alice"))

    def test_retaking_your_own_lease_renews_rather_than_fails(self) -> None:
        self.lease.take("alice", "Alice")
        self.clock.advance(5)
        self.assertEqual(self.lease.take("alice", "Alice"), (True, ""))
        self.clock.advance(5)                     # 10s total, only 5s since renew
        self.assertTrue(self.lease.is_holder("alice"))

    def test_it_expires_without_a_heartbeat(self) -> None:
        self.lease.take("alice", "Alice")
        self.clock.advance(8.1)
        self.assertFalse(self.lease.is_holder("alice"))
        self.assertFalse(self.lease.status()["held"])

    def test_heartbeats_keep_it_alive_indefinitely(self) -> None:
        self.lease.take("alice", "Alice")
        for _ in range(20):
            self.clock.advance(2.5)
            self.assertTrue(self.lease.heartbeat("alice"))
        self.assertTrue(self.lease.is_holder("alice"))

    def test_heartbeat_from_a_non_holder_is_refused_and_changes_nothing(self) -> None:
        self.lease.take("alice", "Alice")
        self.assertFalse(self.lease.heartbeat("bob"))
        self.assertTrue(self.lease.is_holder("alice"))

    def test_an_expired_holder_cannot_heartbeat_its_way_back(self) -> None:
        # Otherwise a laptop waking from sleep silently reclaims a robot that
        # someone else has since taken.
        self.lease.take("alice", "Alice")
        self.clock.advance(8.1)
        self.lease.take("bob", "Bob")
        self.assertFalse(self.lease.heartbeat("alice"))
        self.assertTrue(self.lease.is_holder("bob"))

    def test_release_frees_it_for_the_next_person(self) -> None:
        self.lease.take("alice", "Alice")
        self.assertTrue(self.lease.release("alice"))
        self.assertTrue(self.lease.take("bob", "Bob")[0])

    def test_only_the_holder_can_release(self) -> None:
        self.lease.take("alice", "Alice")
        self.assertFalse(self.lease.release("bob"))
        self.assertTrue(self.lease.is_holder("alice"))

    def test_a_blank_client_is_never_the_holder(self) -> None:
        self.assertFalse(self.lease.is_holder(""))
        self.assertFalse(self.lease.is_holder(None))
        self.assertEqual(self.lease.take("")[0], False)

    def test_holder_changes_are_reported_exactly_once(self) -> None:
        # The server drains this to stop the robot on every handover.
        self.lease.take("alice", "Alice")
        self.assertTrue(self.lease.take_changed())
        self.assertFalse(self.lease.take_changed())
        self.clock.advance(8.1)
        self.lease.status()                       # expiry noticed here
        self.assertTrue(self.lease.take_changed())

    def test_names_are_trimmed_and_never_blank(self) -> None:
        self.lease.take("alice", "   ")
        self.assertEqual(self.lease.status()["name"], "someone")
        self.lease.release("alice")
        self.lease.take("bob", "x" * 200)
        self.assertLessEqual(len(self.lease.status()["name"]), control.MAX_NAME)


class TestGateRequiresControl(unittest.TestCase):
    """An expired lease must stop the robot through the normal gate."""

    CLEAR = {
        "obstacles": {"protection": {"enabled": True, "ready": True}},
        "reachable": True,
        "control": {"held": True},
        "brake": {"engaged": False},
        "bumper": {"estop": False, "front": False, "rear": False},
        "safety_stale": False,
        "drive": {"armed": False, "linear": 0.0, "angular": 0.0},
        "nav": {"nav_status_type": "idle"},
    }

    def setUp(self) -> None:
        self._was = drive.TELEOP_DISABLED
        drive.TELEOP_DISABLED = False
        self.addCleanup(lambda: setattr(drive, "TELEOP_DISABLED", self._was))

    def test_motion_allowed_while_someone_holds_control(self) -> None:
        self.assertIsNone(drive.motion_gate(self.CLEAR))

    def test_motion_refused_when_nobody_holds_control(self) -> None:
        loose = {**self.CLEAR, "control": {"held": False}}
        self.assertIn("nobody has taken control", drive.motion_gate(loose))

    def test_lift_refused_when_nobody_holds_control(self) -> None:
        loose = {**self.CLEAR, "control": {"held": False}}
        self.assertIn("nobody has taken control", drive.lift_gate(loose))

    def test_a_missing_control_block_is_treated_as_nobody(self) -> None:
        # Fail closed: an older snapshot shape must not read as "all clear".
        bare = {key: value for key, value in self.CLEAR.items() if key != "control"}
        self.assertIn("nobody has taken control", drive.motion_gate(bare))


class TestControlOverHttp(unittest.TestCase):
    """The lease as the browser actually meets it."""

    def setUp(self) -> None:
        self.mock = mock_robot.serve(port=0)
        self.addCleanup(self.mock.shutdown)
        self.web = server.DashboardServer(("127.0.0.1", 0))
        threading.Thread(target=self.web.serve_forever, daemon=True).start()
        self.addCleanup(self.web.shutdown)
        self.base = f"http://127.0.0.1:{self.web.server_address[1]}"
        server.SESSION.lease = control.ControlLease()
        self.addCleanup(server.SESSION.disconnect)
        # Control first, then connect. The lease does not depend on the link, so
        # this is the real flow too: connecting tears down the drive loop, which
        # is not something a bystander may do to whoever is driving.
        self.take("alice", "Alice")
        self.post("/api/connect", {"host": "127.0.0.1",
                                   "port": self.mock.server_address[1]},
                  client="alice")
        self._was = drive.TELEOP_DISABLED
        drive.TELEOP_DISABLED = False
        self.addCleanup(lambda: setattr(drive, "TELEOP_DISABLED", self._was))

    def post(self, path: str, body: dict, client: str | None = None) -> dict:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"X-Control-Token": client} if client else {})},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read())

    def take(self, client: str, name: str, force: bool = False) -> dict:
        return self.post("/api/control/take",
                         {"client": client, "name": name, "force": force})

    # --- the guard ---------------------------------------------------------
    def test_driving_without_the_lease_is_refused(self) -> None:
        result = self.post("/api/drive", {"linear": 0.2, "angular": 0.0})
        self.assertFalse(result.get("ok"), result)
        self.assertIn("Take control", result["error"])

    def test_driving_with_the_lease_is_allowed(self) -> None:
        # This test exercises control ownership/disconnect with usable sensors.
        from unittest.mock import patch
        self.mock.sim.params["/navigation/nav/avoid_obstacle"]["value"] = "true"
        self.mock.sim.params["/devices/fallprevention/switch_fallprevention"]["value"] = ["true"] * mock_robot.IRSENSOR_COUNT
        self.mock.sim.params["/devices/fallprevention/min_range_fallprevention"]["value"] = ["0.05"] * mock_robot.IRSENSOR_COUNT
        self.mock.sim.params["/devices/camera/switch_camera"]["value"] = "true"
        for method, count, distance in (("ultrasonic", mock_robot.ULTRASOUND_COUNT, .5), ("irsensor", mock_robot.IRSENSOR_COUNT, .15)):
            patched = patch.object(self.mock.sim, method, return_value=[{"range": distance} for _ in range(count)])
            patched.start(); self.addCleanup(patched.stop)
        server.SESSION.link.params(refresh=True)
        deadline = time.monotonic() + 6
        while not server.SESSION.link.snapshot()["obstacles"]["protection"]["ready"] and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertTrue(server.SESSION.link.snapshot()["obstacles"]["protection"]["ready"])
        self.take("alice", "Alice")
        result = self.post("/api/drive", {"linear": 0.2, "angular": 0.0},
                           client="alice")
        self.assertTrue(result.get("ok"), result)
        # `ok` only means the request was well formed — /api/drive reports a
        # refusal in `accepted`, so asserting `ok` alone would pass even with
        # the gate rejecting every command.
        self.assertTrue(result["data"]["accepted"], result["data"])

    def test_a_second_browser_cannot_command(self) -> None:
        self.take("alice", "Alice")
        result = self.post("/api/drive", {"linear": 0.2}, client="bob")
        self.assertFalse(result.get("ok"))
        self.assertIn("Alice", result["error"])

    def test_every_protected_post_refuses_without_the_lease(self) -> None:
        # The point of the fail-closed list: this asserts the DEFAULT, so a
        # route added later is covered without anyone remembering to add it.
        protected = [(method, path) for (method, path) in server.ROUTES
                     if method == "POST" and path not in server.OPEN_POSTS]
        self.assertGreater(len(protected), 5, protected)
        for _method, path in protected:
            with self.subTest(path=path):
                result = self.post(path, {})
                self.assertFalse(result.get("ok"), f"{path} ran without the lease")
                self.assertIn("control", result["error"].lower())

    # --- what must stay open ----------------------------------------------
    def test_emergency_stop_works_without_the_lease(self) -> None:
        # The whole point: a bystander watching a robot head for a wall must be
        # able to stop it without negotiating for control first.
        self.take("alice", "Alice")
        result = self.post("/api/estop", {"engaged": True})
        self.assertTrue(result.get("ok"), result)

    def test_explicit_stops_work_without_the_lease(self) -> None:
        self.take("alice", "Alice")
        for path in ("/api/drive/stop", "/api/nav/stop"):
            with self.subTest(path=path):
                self.assertTrue(self.post(path, {}).get("ok"))

    # --- handover ----------------------------------------------------------
    def test_taking_over_requires_force_and_then_works(self) -> None:
        self.take("alice", "Alice")
        refused = self.take("bob", "Bob")
        self.assertFalse(refused.get("ok"))
        self.assertTrue(self.take("bob", "Bob", force=True).get("ok"))
        self.assertTrue(self.post("/api/drive", {"linear": 0.1},
                                  client="bob").get("ok"))

    def test_a_handover_stops_the_robot(self) -> None:
        # Control must never change hands with the base still carrying the
        # previous operator's command.
        self.take("alice", "Alice")
        self.post("/api/drive", {"linear": 0.3, "angular": 0.0}, client="alice")
        self.take("bob", "Bob", force=True)
        self.assertEqual(server.SESSION.drive.status()["linear"], 0.0)

    def test_release_frees_the_robot(self) -> None:
        self.take("alice", "Alice")
        self.assertTrue(self.post("/api/control/release",
                                  {"client": "alice"}).get("ok"))
        self.assertTrue(self.take("bob", "Bob").get("ok"))

    # --- shared, persistent state -----------------------------------------
    def test_speed_is_server_side_so_every_viewer_sees_one_value(self) -> None:
        self.take("alice", "Alice")
        self.post("/api/settings", {"linear": 0.35}, client="alice")
        state = json.loads(urllib.request.urlopen(self.base + "/api/state",
                                                  timeout=10).read())["data"]
        self.assertEqual(state["settings"]["linear"], 0.35)

    def test_speed_cannot_be_pushed_past_the_drive_ceiling(self) -> None:
        self.take("alice", "Alice")
        self.post("/api/settings", {"linear": 99.0, "angular": 99.0}, client="alice")
        self.assertEqual(server.SESSION.settings["linear"], drive.MAX_LINEAR)
        self.assertEqual(server.SESSION.settings["angular"], drive.MAX_ANGULAR)

    def test_control_state_is_visible_to_everyone(self) -> None:
        self.take("alice", "Alice")
        state = json.loads(urllib.request.urlopen(self.base + "/api/state",
                                                  timeout=10).read())["data"]
        self.assertTrue(state["control"]["held"])
        self.assertEqual(state["control"]["name"], "Alice")


if __name__ == "__main__":
    unittest.main()
