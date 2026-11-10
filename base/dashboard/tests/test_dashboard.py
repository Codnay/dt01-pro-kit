"""Dashboard tests, run against the bundled mock controller.

    python3 -m unittest discover -s dashboard/tests -t dashboard

These check that the dashboard handles what the vendor documentation says the
controller does — the response shapes, the ``successed`` spelling, the state
chain, the disabled auto-charge endpoints, and the failure paths. They do not
and cannot prove the real robot behaves that way.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import drive                                              # noqa: E402
import drobot_api                                         # noqa: E402
import mock_robot                                         # noqa: E402
import server as dashboard_server                         # noqa: E402
from drobot_api import DashboardClient, DRobotError       # noqa: E402
from telemetry import RobotLink                           # noqa: E402


def wait_for(predicate, timeout: float = 6.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class MockBacked(unittest.TestCase):
    """Base class: one mock controller per test class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock = mock_robot.serve(port=0)
        cls.port = cls.mock.server_address[1]
        cls.client = DashboardClient("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock.stopping.set()
        cls.mock.shutdown()

    def setUp(self) -> None:
        self.mock.sim.__init__()                          # a clean robot per test


class TestClient(MockBacked):
    def test_state_and_brake_round_trip(self) -> None:
        self.assertEqual(self.client.state()["state_operation"], "Idle")
        self.assertFalse(self.client.brake_engaged())
        self.client.set_brake(True)
        self.assertTrue(self.client.brake_engaged())

    def test_failed_call_raises_despite_http_200(self) -> None:
        """The vendor answers 200 with successed:false; that must not look like success."""
        with self.assertRaises(DRobotError):
            self.client.move_to("NOWHERE")               # also wrong state

    def test_state_chain_is_single_hop(self) -> None:
        with self.assertRaises(DRobotError):
            self.client.set_state("RunningTask")         # Idle -> RunningTask is two hops
        steps: list[str] = []
        drobot_api.walk_to_state(self.client, "RunningTask", steps.append)
        self.assertEqual(self.client.state()["state_operation"], "RunningTask")
        self.assertEqual(len(steps), 3)                  # two hops plus the confirmation

    def test_sensor_shapes(self) -> None:
        readings = self.client.ultrasonic()
        self.assertEqual(len(readings), mock_robot.ULTRASOUND_COUNT)
        self.assertIn("range", readings[0])
        self.assertIn("frameid", readings[0])

    def test_bumper_bits_decode_right_to_left(self) -> None:
        self.assertEqual(drobot_api.parse_bumper("001"),
                         {"estop": True, "front": False, "rear": False,
                          "left": False, "right": False})
        self.assertTrue(drobot_api.parse_bumper("010")["front"])
        self.assertTrue(drobot_api.parse_bumper("100")["rear"])

    def test_chassis_bitmask_is_split_per_sensor(self) -> None:
        parsed = drobot_api.parse_chassis({"Laser": 0, "Ultrasonic": 6, "ChargingState": 1})
        self.assertTrue(parsed["devices"]["Laser"]["healthy"])
        self.assertFalse(parsed["devices"]["Ultrasonic"]["healthy"])
        self.assertEqual(parsed["devices"]["Ultrasonic"]["faulty_indices"], [1, 2])
        self.assertTrue(parsed["charging"])

    def test_param_table_keeps_limits_and_arrays(self) -> None:
        table = self.client.param_table()
        level = table["/navigation/nav/speed_level"]
        self.assertEqual((level.lower, level.upper), ("0", "2"))
        switches = table["/devices/ultrasonic/switch_ultrasound"]
        self.assertTrue(switches.is_array)
        self.assertIsInstance(switches.value, list)

    def test_map_png_is_a_png(self) -> None:
        self.assertTrue(self.client.map_png("mock_floor").startswith(b"\x89PNG"))

    def test_error_codes_are_explained(self) -> None:
        try:
            self.client.set_state("RunningTask")
        except DRobotError as exc:
            self.assertIn("state does not match", drobot_api.explain(exc))
        else:
            self.fail("expected the two-hop jump to be refused")


class TestPortability(unittest.TestCase):
    """The dashboard has to run on whatever laptop is next to the robot."""

    def test_imports_without_posix_terminal_modules(self) -> None:
        """A colleague's laptop may be Windows; termios/tty do not exist there.

        Run in a subprocess with those modules poisoned, because the real ones
        are already imported in this one.
        """
        import subprocess
        script = (
            "import sys;"
            "sys.modules['termios'] = None; sys.modules['tty'] = None;"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r});"
            "import drobot_api, telemetry, drive, server;"
            "print(len(server.ROUTES))"
        )
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(int(result.stdout.strip()) > 0)

    def test_local_subnets_are_plausible_prefixes(self) -> None:
        subnets = drobot_api.local_subnets()
        self.assertTrue(subnets, "no subnet detected and no fallback returned")
        for subnet in subnets:
            octets = subnet.split(".")
            self.assertEqual(len(octets), 3, subnet)
            self.assertTrue(all(part.isdigit() for part in octets), subnet)
            self.assertFalse(subnet.startswith("127."), "loopback is not a scan target")

    def test_handoff_package_lists_only_what_it_needs(self) -> None:
        """drobot_api imports ../tools/drobot.py, so a copy needs both trees."""
        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root.parent / "tools" / "drobot.py").is_file())
        self.assertTrue((root / "make_handoff.sh").is_file())


class TeleopEnabled:
    """Lift the build-level teleop lockout for tests that exercise driving.

    ``drive.TELEOP_DISABLED`` short-circuits the gate before every other check,
    which is the point of it — but it would mask the behaviour these tests are
    here to pin down. Lifted per-test and always restored, so the lockout's own
    tests still see it on.
    """

    def setUp(self) -> None:
        self._teleop_was = drive.TELEOP_DISABLED
        drive.TELEOP_DISABLED = False
        super().setUp()

    def tearDown(self) -> None:
        try:
            super().tearDown()
        finally:
            drive.TELEOP_DISABLED = self._teleop_was


class TestTeleopLockout(unittest.TestCase):
    """The lockout must beat every other consideration, including a clean robot."""

    def setUp(self) -> None:
        self._was = drive.TELEOP_DISABLED
        drive.TELEOP_DISABLED = True

    def tearDown(self) -> None:
        drive.TELEOP_DISABLED = self._was

    CLEAN = {"reachable": True, "brake": {"engaged": False},
             "bumper": {"estop": False, "front": False, "rear": False},
             "safety_stale": False, "nav": {"nav_status_type": "idle"}}

    def test_gate_refuses_even_a_perfectly_clean_robot(self) -> None:
        self.assertIn("teleop is disabled", drive.motion_gate(self.CLEAN))

    def test_wasd_intents_are_rejected_and_send_no_motion(self) -> None:
        sent: list[tuple[float, float]] = []

        class Client:
            def move(self, velocity): sent.append((velocity.linear, velocity.angular))
            def stop(self): sent.append((0.0, 0.0))

        loop = drive.DriveLoop(Client(), lambda: drive.motion_gate(self.CLEAN))
        for linear, angular in ((0.4, 0.0), (0.0, 0.7), (-0.4, 0.0), (0.0, -0.7)):
            result = loop.request(linear, angular)
            self.assertFalse(result["accepted"])
            self.assertIn("teleop is disabled", result["reason"])
        self.assertEqual([v for v in sent if v != (0.0, 0.0)], [],
                         "a nonzero velocity reached the base while locked out")


class TestMotionGate(TeleopEnabled, unittest.TestCase):
    """The gate is a pure function, so every refusal is testable without a robot."""

    def test_allows_a_clean_robot(self) -> None:
        self.assertIsNone(drive.motion_gate(
            {"brake": {"engaged": False}, "bumper": {}, "nav": {},
             "control": {"held": True},
             "obstacles": {"protection": {"enabled": True, "ready": True}}}))

    def test_refuses_when_the_controller_is_not_answering(self) -> None:
        self.assertIn("not answering", drive.motion_gate({"reachable": False}))

    def test_refuses_on_software_brake(self) -> None:
        self.assertIn("software emergency stop",
                      drive.motion_gate({"brake": {"engaged": True}}))

    def test_refuses_on_chassis_estop_and_bumpers(self) -> None:
        self.assertIn("chassis emergency stop", drive.motion_gate(
            {"brake": {}, "bumper": {"estop": True}}))
        self.assertIn("bumper", drive.motion_gate(
            {"brake": {}, "bumper": {"front": True}}))

    def test_refuses_while_the_navigator_owns_the_base(self) -> None:
        reason = drive.motion_gate(
            {"brake": {}, "bumper": {}, "nav": {"nav_status_type": "navigation"}})
        self.assertIn("navigator is running a task", reason)


class TestDriveLoop(TeleopEnabled, MockBacked):
    def setUp(self) -> None:
        super().setUp()
        self.link = RobotLink("127.0.0.1", self.port)
        self.gate_reason: str | None = None
        self.loop = drive.DriveLoop(self.link.client, lambda: self.gate_reason)
        self.loop.start()

    def tearDown(self) -> None:
        self.loop.shutdown()
        self.link.stop()

    def test_intent_drives_and_the_deadman_stops_it(self) -> None:
        self.loop.request(0.2, 0.0)
        self.assertTrue(wait_for(lambda: abs(self.mock.sim.linear - 0.2) < 1e-6),
                        "the base never saw the commanded velocity")
        # Stop refreshing the intent: the server-side deadman must zero the base
        # without the browser doing anything at all.
        self.assertTrue(wait_for(lambda: self.mock.sim.linear == 0.0, timeout=4.0),
                        "the deadman did not zero the base")

    def test_speed_is_clamped_to_the_dashboard_ceiling(self) -> None:
        result = self.loop.request(99.0, 99.0)
        self.assertEqual(result["linear"], drive.MAX_LINEAR)
        self.assertEqual(result["angular"], drive.MAX_ANGULAR)

    def test_a_closed_gate_refuses_motion(self) -> None:
        self.gate_reason = "software emergency stop is engaged"
        result = self.loop.request(0.2, 0.0)
        self.assertFalse(result["accepted"])
        self.assertEqual(self.mock.sim.linear, 0.0)

    def test_emergency_stop_brakes_before_zeroing(self) -> None:
        self.loop.request(0.3, 0.0)
        wait_for(lambda: self.mock.sim.linear > 0)
        result = self.loop.emergency_stop()
        self.assertTrue(result["braked"])
        self.assertTrue(self.mock.sim.brake)
        self.assertEqual(self.mock.sim.linear, 0.0)

    def test_stop_is_repeated(self) -> None:
        """One zero packet on a lossy link is not a stop."""
        sent: list[str] = []
        original = self.link.client.move

        def counting(velocity):
            sent.append("zero" if velocity.is_zero() else "move")
            return original(velocity)

        self.link.client.move = counting                  # type: ignore[method-assign]
        self.loop.stop()
        self.assertGreaterEqual(sent.count("zero"), drive.STOP_REPEAT)


class TestTelemetry(MockBacked):
    def setUp(self) -> None:
        super().setUp()
        self.link = RobotLink("127.0.0.1", self.port)
        self.link.start()

    def tearDown(self) -> None:
        self.link.stop()

    def test_websocket_push_populates_the_snapshot(self) -> None:
        self.assertTrue(wait_for(lambda: self.link.notice_feed.connected),
                        "the notice websocket never connected")
        self.assertTrue(wait_for(lambda: self.link.snapshot()["localization"].get("status")
                                 == "normal"))
        snapshot = self.link.snapshot()
        self.assertIn("x", snapshot["pose"]["grid"])
        self.assertIsNotNone(snapshot["battery"]["battery_capacity_percentage"])

    def test_laser_push_arrives_and_is_shaped_for_the_map(self) -> None:
        self.assertTrue(wait_for(lambda: self.link.snapshot().get("scan"), timeout=8.0),
                        "no laser frame arrived")
        scan = self.link.snapshot()["scan"]
        self.assertTrue(scan["gridpoint"])
        self.assertIn("x", scan["gridpoint"][0])

    def test_obstacle_alarm_trips_when_the_wall_is_close(self) -> None:
        self.link.params(refresh=True)
        self.assertTrue(wait_for(lambda: self.link.snapshot()["ultrasonic"]))
        self.assertFalse(self.link.snapshot()["obstacles"]["alarm"], "wall starts 2.5 m away")
        self.mock.sim.obstacle_x = 0.25                   # move the wall onto the robot
        self.assertTrue(wait_for(lambda: self.link.snapshot()["obstacles"]["alarm"], timeout=6.0),
                        "the proximity alarm never tripped")
        self.assertTrue(self.link.snapshot()["obstacles"]["reasons"])

    def test_a_zero_range_reading_is_not_an_obstacle(self) -> None:
        """The controller reports 0 for a sensor that sees nothing."""
        obstacles = self.link._obstacles(
            {"ultrasonic": [{"range": 0.0, "frameid": "ultrasound1"}], "irsensor": [],
             "bumper": {}, "chassis": {}})
        self.assertFalse(obstacles["alarm"])

    def test_a_failing_endpoint_backs_off_without_killing_the_rest(self) -> None:
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mock/config", method="POST",
            data=json.dumps({"fail": ["/sensor_data/raw_ultrasonic"]}).encode()))
        self.assertTrue(wait_for(
            lambda: "ultrasonic" in self.link.snapshot()["link"]["poll_errors"], timeout=6.0))
        # Battery still updates, so one dead endpoint has not starved the loop.
        self.assertTrue(wait_for(
            lambda: self.link.snapshot()["battery"].get("battery_capacity_percentage")
            is not None))

    def test_preflight_reports_the_brake(self) -> None:
        self.assertTrue(wait_for(
            lambda: self.link.snapshot()["brake"].get("engaged") is False))
        self.assertTrue(self.link.preflight()["ok"])
        self.link.client.set_brake(True)
        self.assertTrue(wait_for(lambda: not self.link.preflight()["ok"], timeout=5.0))


class TestObservedFirmware(TeleopEnabled, MockBacked):
    """Behaviour recorded from a live DT-01 on 2026-08-31.

    That robot serves neither /robot/ws/notice (404) nor the bumper endpoint
    (error 103), returns an empty chassis_state, reports battery voltage in
    decivolts, and never populates current or temperature. Each of those broke
    something, so each gets a test.
    """

    def setUp(self) -> None:
        super().setUp()
        self.mock.sim.legacy_firmware = True
        self.link = RobotLink("127.0.0.1", self.port)
        self.link.start()

    def tearDown(self) -> None:
        try:
            self.link.stop()
        finally:
            super().tearDown()

    def test_push_feed_fails_over_to_the_path_this_firmware_serves(self) -> None:
        self.assertTrue(wait_for(lambda: self.link.notice_feed.connected, timeout=20.0),
                        "never connected to either push path")
        self.assertTrue(self.link.notice_feed.url.endswith("/robot/notification"))

    def test_flat_schema_still_fills_the_safety_fields(self) -> None:
        self.assertTrue(wait_for(
            lambda: "estop" in (self.link.snapshot().get("bumper") or {}), timeout=20.0),
            "bumper/e-stop never arrived from the flat push schema")
        snapshot = self.link.snapshot()
        self.assertFalse(snapshot["safety_stale"])
        self.assertEqual(snapshot["localization"]["status"], "normal")

    def test_decivolt_battery_is_normalised(self) -> None:
        self.assertTrue(wait_for(
            lambda: self.link.snapshot()["battery"].get("battery_voltage"), timeout=10.0))
        raw = self.link.snapshot()["battery"]["battery_voltage"]
        self.assertGreater(raw, 200)                       # firmware reports decivolts
        self.assertLess(drobot_api.normalize_voltage(raw), 40)

    def test_dead_current_sensor_means_unknown_not_discharging(self) -> None:
        self.assertIsNone(drobot_api.charging_state({"battery_current": 0.0}, {}))

    def test_blind_interlocks_refuse_motion(self) -> None:
        """The whole point: missing bumper data must not read as all-clear."""
        blind = {"brake": {"engaged": False}, "bumper": {}, "safety_stale": True}
        self.assertIn("interlocks are blind", drive.motion_gate(blind))


class TestHttpApi(TeleopEnabled, MockBacked):
    """Exercise the dashboard's own HTTP surface end to end."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.web = dashboard_server.DashboardServer(("127.0.0.1", 0))
        import threading
        threading.Thread(target=cls.web.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.web.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.web.shutting_down.set()
        dashboard_server.SESSION.disconnect()
        cls.web.shutdown()
        super().tearDownClass()

    CLIENT = "test-http-client"

    def setUp(self) -> None:
        super().setUp()
        # Commanding needs the control lease. Taken before connect because
        # /api/connect tears down the drive loop, which is not something a
        # bystander may do to whoever is driving.
        import control
        dashboard_server.SESSION.lease = control.ControlLease()
        self.post("/api/control/take", {"client": self.CLIENT, "name": "Tester"})
        self.post("/api/connect", {"host": "127.0.0.1", "port": self.port})

    def request(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        if data:
            headers["X-Control-Token"] = self.CLIENT
        request = urllib.request.Request(self.base + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read())

    def post(self, path: str, body: dict) -> dict:
        return self.request(path, body)

    def get(self, path: str) -> dict:
        return self.request(path)

    def test_state_snapshot_is_complete(self) -> None:
        payload = self.get("/api/state")["data"]
        self.assertTrue(payload["connected"])
        for key in ("robot", "brake", "battery", "obstacles", "drive", "link", "speed_presets"):
            self.assertIn(key, payload)

    def test_serves_the_page_and_its_assets(self) -> None:
        for path in ("/", "/app.css", "/app.js", "/map.js"):
            with urllib.request.urlopen(self.base + path, timeout=5) as response:
                self.assertEqual(response.status, 200, path)
                self.assertTrue(response.read())

    def test_static_traversal_is_refused(self) -> None:
        request = urllib.request.Request(self.base + "/../server.py")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read()
            self.assertNotIn(b"DashboardServer", body)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)

    def test_map_png_is_proxied(self) -> None:
        with urllib.request.urlopen(self.base + "/api/map.png?name=mock_floor",
                                    timeout=5) as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertTrue(response.read().startswith(b"\x89PNG"))

    def test_goto_requires_localisation(self) -> None:
        self.mock.sim.state_initialized = False
        # Localisation is now read from the push feed, which the controller sends
        # on its own schedule, so wait for it rather than racing it.
        self.assertTrue(wait_for(
            lambda: self.get("/api/state")["data"]["localized"] is False, timeout=8.0))
        payload = self.post("/api/nav/goto", {"name": "BENCH"})
        self.assertFalse(payload["ok"])
        self.assertIn("not localised", payload["error"])

    def test_localisation_prefers_the_notice_over_state_initialized(self) -> None:
        """Hardware reports state_initialized: false while the push feed says
        initialized: true, score 1.0. Gating on the flag refuses to move a
        perfectly localised robot."""
        link = dashboard_server.SESSION.link
        self.assertTrue(wait_for(lambda: link.snapshot()["localization"].get("status")
                                 == "normal", timeout=8.0))
        self.mock.sim.state_initialized = False           # the red herring
        with link._lock:                                  # notice still says converged
            link._state["localization"] = {"status": "normal", "initialized": True,
                                           "score": 1.0}
            link._state["robot"]["state_initialized"] = False
        self.assertTrue(link.localized())

    def test_goto_walks_to_running_task_and_starts_the_task(self) -> None:
        payload = self.post("/api/nav/goto", {"name": "BENCH"})
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(self.mock.sim.state_operation, "RunningTask")
        self.assertEqual(self.mock.sim.nav_goal, "BENCH")

    def test_navigation_reaches_the_point(self) -> None:
        self.post("/api/nav/goto", {"name": "BENCH"})
        self.assertTrue(wait_for(lambda: self.mock.sim.nav_goal is None, timeout=45.0),
                        "the mock robot never arrived")
        self.assertAlmostEqual(self.mock.sim.x, 1.6, delta=0.2)

    def test_realtime_path_is_returned_in_grid_coordinates(self) -> None:
        self.post("/api/nav/goto", {"name": "DOOR"})
        path = self.get("/api/nav/path")["data"]
        self.assertTrue(path["grid_points"])
        self.assertIn("x", path["grid_points"][0])

    def test_nav_stop_is_repeated_and_clears_the_task(self) -> None:
        self.post("/api/nav/goto", {"name": "DOOR"})
        self.assertTrue(self.post("/api/nav/stop", {})["data"]["stopped"])
        self.assertIsNone(self.mock.sim.nav_goal)

    def test_saving_a_charge_point_then_arming_auto_charge(self) -> None:
        self.post("/api/mode", {"state": "RunningTask"})
        self.assertTrue(self.post("/api/points/save",
                                  {"name": "DOCK2", "type": "charge"})["ok"])
        # Endpoint disabled by default, exactly as the API reference marks it.
        refused = self.post("/api/charge/auto", {"enabled": True, "point": "DOCK2"})
        self.assertFalse(refused["ok"])
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mock/config", method="POST",
            data=json.dumps({"autocharge_enabled_endpoint": True}).encode()))
        accepted = self.post("/api/charge/auto", {"enabled": True, "point": "DOCK2"})
        self.assertTrue(accepted["ok"], accepted)
        self.assertTrue(self.mock.sim.autocharge_on)

    def test_auto_charge_refuses_a_non_charge_point(self) -> None:
        payload = self.post("/api/charge/auto", {"enabled": True, "point": "BENCH"})
        self.assertFalse(payload["ok"])
        self.assertIn("not a charge point", payload["error"])

    def test_charging_pile_id_failure_is_surfaced_not_swallowed(self) -> None:
        """The vendor documents this endpoint only by its error case. Whatever
        it does on real hardware, the panel must show it rather than hide it."""
        payload = self.get("/api/charge")["data"]
        self.assertIn("charging_pile", payload)
        self.assertIsNone(payload["charging_pile"])
        self.assertIn("FAILED CALL SERVICE", payload["charging_pile_error"])

    def test_charge_panel_reports_the_disabled_endpoint_rather_than_lying(self) -> None:
        payload = self.get("/api/charge")["data"]
        self.assertFalse(payload["autocharge_available"])
        self.assertIsNotNone(payload["autocharge_error"])
        self.assertIn("battery_capacity_percentage", payload["battery"])

    def test_dock_now_picks_the_only_charge_point(self) -> None:
        payload = self.post("/api/charge/dock", {})
        self.assertEqual(payload["data"]["navigating_to"], "DOCK")

    def test_protection_enable_requires_stopped_base(self) -> None:
        result = self.post("/api/protection/enable", {})
        self.assertFalse(result["ok"])
        self.assertIn("emergency stop", result["error"])
        self.assertEqual(self.mock.sim.params["/navigation/nav/avoid_obstacle"]["value"], "false")

    def test_protection_enable_verifies_readback_and_keeps_brake(self) -> None:
        self.post("/api/estop", {"engaged": True})
        result = self.post("/api/protection/enable", {})
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["data"]["configured"])
        self.assertTrue(self.mock.sim.brake)
        self.assertEqual(self.mock.sim.params["/devices/fallprevention/switch_fallprevention"]["value"], ["true"] * mock_robot.IRSENSOR_COUNT)
        self.assertEqual(self.mock.sim.params["/navigation/nav/avoid_obstacle"]["value"], "true")
        # This mock returns zero cliff ranges: configured does not mean ready.
        self.assertFalse(result["data"]["obstacles"]["protection"]["ready"])

    def test_protection_cannot_be_disabled_through_parameter_editor(self) -> None:
        result = self.post("/api/params", {"updates": [{"namespace": "/navigation/nav/avoid_obstacle", "value": False}]})
        self.assertFalse(result["ok"])
        self.assertIn("remain enabled", result["error"])

    def test_protection_write_failure_preserves_stop(self) -> None:
        from unittest.mock import patch
        self.post("/api/estop", {"engaged": True})
        with patch.object(dashboard_server.SESSION.link.client, "write_params", side_effect=DRobotError("rejected")):
            result = self.post("/api/protection/enable", {})
        self.assertFalse(result["ok"])
        self.assertTrue(self.mock.sim.brake)
        self.assertIn("could not be verified", result["error"])

    def test_protection_silent_write_failure_is_not_success(self) -> None:
        from unittest.mock import patch
        self.post("/api/estop", {"engaged": True})
        with patch.object(dashboard_server.SESSION.link.client, "write_params", return_value=None):
            result = self.post("/api/protection/enable", {})
        self.assertFalse(result["ok"])
        self.assertIn("readback failed", result["error"])
        self.assertTrue(self.mock.sim.brake)

    def test_nonfinite_protection_thresholds_are_rejected(self) -> None:
        for field in ("alarm_distance", "sensor_stale_s"):
            self.assertFalse(self.post("/api/settings", {field: float("nan")})["ok"])

    def test_speed_level_write_is_clamped_and_typed(self) -> None:
        payload = self.post("/api/params", {"updates": [
            {"namespace": "/navigation/nav/speed_level", "value": "2"}]})
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(
            self.mock.sim.params["/navigation/nav/speed_level"]["value"], "2")

    def test_array_parameters_round_trip_as_lists(self) -> None:
        values = ["0.30"] * mock_robot.ULTRASOUND_COUNT
        payload = self.post("/api/params", {"updates": [
            {"namespace": "/devices/ultrasonic/max_range_ultrasound", "value": values}]})
        self.assertTrue(payload["ok"], payload)
        written = self.mock.sim.params["/devices/ultrasonic/max_range_ultrasound"]["value"]
        self.assertEqual(written, ["0.300000"] * mock_robot.ULTRASOUND_COUNT)

    def test_unknown_parameter_is_refused(self) -> None:
        payload = self.post("/api/params", {"updates": [
            {"namespace": "/etc/passwd", "value": "1"}]})
        self.assertFalse(payload["ok"])

    def test_estop_engages_and_releases(self) -> None:
        self.assertTrue(self.post("/api/estop", {"engaged": True})["data"]["braked"])
        self.assertTrue(self.mock.sim.brake)
        self.assertTrue(self.post("/api/estop", {"engaged": False})["data"]["released"])
        self.assertFalse(self.mock.sim.brake)

    def test_drive_is_refused_while_braked(self) -> None:
        self.post("/api/estop", {"engaged": True})
        payload = self.post("/api/drive", {"linear": 0.2, "angular": 0})["data"]
        self.assertFalse(payload["accepted"])
        self.post("/api/estop", {"engaged": False})

    def test_estop_takes_effect_in_the_snapshot_immediately(self) -> None:
        """No waiting on the 1 Hz brake poll: the operator must see it at once."""
        self.post("/api/estop", {"engaged": True})
        state = self.get("/api/state")["data"]
        self.assertTrue(state["brake"]["engaged"])
        self.assertIn("software emergency stop", state["motion_blocked_by"])
        self.post("/api/estop", {"engaged": False})
        self.assertFalse(self.get("/api/state")["data"]["brake"]["engaged"])

    def test_drive_then_disconnect_stops_the_robot(self) -> None:
        # This test exercises control ownership/disconnect with usable sensors.
        from unittest.mock import patch
        self.mock.sim.params["/navigation/nav/avoid_obstacle"]["value"] = "true"
        self.mock.sim.params["/devices/fallprevention/switch_fallprevention"]["value"] = ["true"] * mock_robot.IRSENSOR_COUNT
        self.mock.sim.params["/devices/fallprevention/min_range_fallprevention"]["value"] = ["0.05"] * mock_robot.IRSENSOR_COUNT
        self.mock.sim.params["/devices/camera/switch_camera"]["value"] = "true"
        for method, count, distance in (("ultrasonic", mock_robot.ULTRASOUND_COUNT, .5), ("irsensor", mock_robot.IRSENSOR_COUNT, .15)):
            patched = patch.object(self.mock.sim, method, return_value=[{"range": distance} for _ in range(count)])
            patched.start(); self.addCleanup(patched.stop)
        dashboard_server.SESSION.link.params(refresh=True)
        self.assertTrue(wait_for(lambda: dashboard_server.SESSION.link.snapshot()["obstacles"]["protection"]["ready"]))
        self.post("/api/drive", {"linear": 0.25, "angular": 0.0})
        self.assertTrue(wait_for(lambda: self.mock.sim.linear > 0))
        self.post("/api/disconnect", {})
        self.assertEqual(self.mock.sim.linear, 0.0)
        self.assertFalse(self.get("/api/state")["data"]["connected"])

    def test_drive_with_unavailable_obstacle_sensors_warns_and_remains_stoppable(self) -> None:
        from unittest.mock import patch
        for method in ("ultrasonic", "irsensor"):
            patched = patch.object(self.mock.sim, method, return_value=[{"range": 0}])
            patched.start(); self.addCleanup(patched.stop)
        self.assertTrue(wait_for(lambda: not self.get("/api/state")["data"]["safety_stale"]))
        state = self.get("/api/state")["data"]
        protection = state["obstacles"]["protection"]
        self.assertFalse(protection["ready"])
        self.assertIsNone(protection["blocked_by"])
        self.assertIsNotNone(protection["warning"])
        self.assertIsNone(state["motion_blocked_by"])
        result = self.post("/api/drive", {"linear": 0, "angular": .25})
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["data"]["accepted"], result)
        self.assertTrue(wait_for(lambda: self.mock.sim.angular != 0))
        self.assertFalse(self.mock.sim.brake)
        self.post("/api/estop", {"engaged": True})
        self.assertTrue(self.mock.sim.brake)
        self.assertEqual(self.mock.sim.angular, 0)
        self.assertIn("software emergency stop", self.get("/api/state")["data"]["motion_blocked_by"])

    def test_commands_without_a_connection_are_refused_cleanly(self) -> None:
        self.post("/api/disconnect", {})
        payload = self.post("/api/drive", {"linear": 0.2, "angular": 0})
        self.assertFalse(payload["ok"])
        self.assertIn("not connected", payload["error"])

    def test_settings_change_does_not_drop_the_link(self) -> None:
        before = self.get("/api/state")["data"]["link"]
        self.post("/api/settings", {"alarm_distance": 0.8})
        after = self.get("/api/state")["data"]
        self.assertTrue(after["connected"])
        self.assertEqual(after["obstacles"]["alarm_distance"], 0.8)
        self.assertEqual(after["link"]["host"], before["host"])

    def test_mode_switch_reports_each_hop(self) -> None:
        payload = self.post("/api/mode", {"state": "RunningTask"})["data"]
        self.assertEqual(payload["state"], "RunningTask")
        self.assertTrue(any("->" in step for step in payload["steps"]))

    def test_auto_mode_rejects_an_invalid_value(self) -> None:
        self.assertTrue(self.post("/api/auto_mode", {"mode": 0})["ok"])
        self.assertFalse(self.post("/api/auto_mode", {"mode": 7})["ok"])

    def test_preflight_endpoint(self) -> None:
        checks = self.get("/api/preflight")["data"]["checks"]
        self.assertTrue(any(check["name"] == "Software e-stop released" for check in checks))

    def test_unknown_route_is_a_clean_404(self) -> None:
        self.assertIn("no route", self.get("/api/nope")["error"])

    def test_an_address_with_nothing_on_it_is_not_reported_as_connected(self) -> None:
        """Starting pollers against a dead address succeeds quietly. Saying
        'connected' there sends an operator hunting for a software fault when
        the real problem is an unplugged cable."""
        # Port 1 on loopback: nothing listens, and it refuses fast.
        result = dashboard_server.SESSION.connect("127.0.0.1", 1)
        try:
            self.assertFalse(result["reachable"])
            self.assertTrue(result["error"])
            state = self.get("/api/state")["data"]
            self.assertFalse(state["reachable"])
            self.assertIn("not answering", state["motion_blocked_by"])
        finally:
            dashboard_server.SESSION.connect("127.0.0.1", self.port)

    def test_a_live_controller_is_reported_reachable(self) -> None:
        result = dashboard_server.SESSION.connect("127.0.0.1", self.port)
        self.assertTrue(result["reachable"], result)
        self.assertIsNone(result["error"])
        self.assertTrue(wait_for(lambda: self.get("/api/state")["data"]["reachable"]))

    def test_wire_frame_drops_what_the_browser_never_reads(self) -> None:
        """The raw arrays are already folded into obstacles.sensors, and the
        map draws scan.gridpoint, never scan.points. Both are pure weight."""
        raw = dashboard_server.SESSION.snapshot()
        frame = dashboard_server.wire_frame(raw)
        self.assertNotIn("ultrasonic", frame)
        self.assertNotIn("irsensor", frame)
        self.assertIn("obstacles", frame)
        self.assertTrue(frame["obstacles"]["sensors"], "sensor data must survive the trim")

    def test_wire_frame_thins_a_dense_laser_return(self) -> None:
        dense = {"scan": {"gridpoint": [{"x": i, "y": i} for i in range(5000)],
                          "points": [{"x": 1.0, "y": 2.0}] * 5000,
                          "mapinfo": {"gridWidth": 10}, "at": 1.0}}
        scan = dashboard_server.wire_frame(dense)["scan"]
        self.assertLessEqual(len(scan["gridpoint"]), dashboard_server.SCAN_MAX_POINTS)
        self.assertEqual(scan["decimated_from"], 5000)
        self.assertNotIn("points", scan)

    def test_wire_frame_rounds_floats(self) -> None:
        frame = dashboard_server.wire_frame({"a": {"b": [0.20000000298023224]}})
        self.assertEqual(frame["a"]["b"][0], 0.2)

    def test_scan_can_be_kept_off_the_wire_entirely(self) -> None:
        self.assertNotIn("scan", dashboard_server.wire_frame(
            dashboard_server.SESSION.snapshot(), want_scan=False))
        self.assertNotIn("scan", self.get("/api/state?scan=0")["data"])

    def test_stream_sends_a_full_frame_then_deltas_that_merge_back(self) -> None:
        """A late or reconnecting client must converge without special handling."""
        merged: dict = {}
        fulls = 0
        deadline = time.monotonic() + 8
        with urllib.request.urlopen(self.base + "/api/stream?hz=5", timeout=15) as stream:
            while time.monotonic() < deadline:
                line = stream.readline()
                if not line.startswith(b"data: "):
                    continue
                message = json.loads(line[6:])
                if message["full"]:
                    fulls += 1
                    merged = message["d"]
                else:
                    self.assertTrue(merged, "a delta arrived before any full frame")
                    merged.update(message["d"])
                if fulls and len(merged) > 5 and not message["full"]:
                    break
        self.assertGreaterEqual(fulls, 1)
        # The merged view must be a usable snapshot, not a pile of fragments.
        for key in ("robot", "brake", "obstacles", "drive", "link", "connected"):
            self.assertIn(key, merged)

    def test_stream_deltas_are_smaller_than_full_frames(self) -> None:
        sizes = {"full": [], "delta": []}
        deadline = time.monotonic() + 6
        with urllib.request.urlopen(self.base + "/api/stream?hz=5", timeout=15) as stream:
            while time.monotonic() < deadline:
                line = stream.readline()
                if not line.startswith(b"data: "):
                    continue
                sizes["full" if json.loads(line[6:])["full"] else "delta"].append(len(line))
        self.assertTrue(sizes["full"] and sizes["delta"], "expected both frame kinds")
        average = lambda values: sum(values) / len(values)          # noqa: E731
        self.assertLess(average(sizes["delta"]), average(sizes["full"]))

    def test_subnets_endpoint_reports_where_a_scan_would_look(self) -> None:
        subnets = self.get("/api/subnets")["data"]["subnets"]
        self.assertTrue(subnets)
        self.assertNotIn("127.0.0", subnets)

    def test_discover_reports_which_subnets_it_scanned(self) -> None:
        """Scanning a subnet the laptop is not on is the usual reason for
        'nothing answering', so the answer has to say where it looked."""
        payload = self.get("/api/discover?subnet=127.0.0")["data"]
        self.assertEqual(payload["scanned"], ["127.0.0"])
        self.assertIn("hosts", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
