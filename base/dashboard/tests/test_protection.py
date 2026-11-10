"""Unavailable sensing warns; detected hazards stop translation and turns."""

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import drive
import protection as p
from telemetry import Poller
from drobot_api import DashboardClient


def settings():
    return {p.ULTRASOUND: ["true"] * 8, p.CLIFF: ["true"] * 4,
            p.NAV: "true", p.FOLLOW: "true",
            "/devices/ultrasonic/min_range_ultrasound": ["0.15"] * 8,
            "/devices/ultrasonic/max_range_ultrasound": ["0.6"] * 8,
            "/devices/fallprevention/min_range_fallprevention": ["0.05"] * 4,
            "/devices/fallprevention/max_range_fallprevention": ["0.3"] * 4}


def readings():
    return {"ultrasonic": [{"range": 0.5} for _ in range(8)],
            "irsensor": [{"range": 0.15} for _ in range(4)]}


class ProtectionTests(unittest.TestCase):
    def evaluate(self, state=None, params=None, ages=None, errors=None, param_age=0):
        return p.obstacles(state if state is not None else readings(), params or settings(),
                           ages or {"ultrasonic": 0.1, "irsensor": 0.1}, errors or {},
                           0.35, 2, param_age)

    def test_good_floor_is_not_a_nearby_obstacle(self):
        result = self.evaluate()
        self.assertTrue(result["protection"]["ready"])
        self.assertFalse(result["alarm"])

    def test_nearby_obstacle_blocks(self):
        state = readings(); state["ultrasonic"][2]["range"] = 0.2
        result = self.evaluate(state)
        self.assertTrue(result["alarm"])
        self.assertFalse(result["protection"]["ready"])

    def test_missing_or_out_of_range_floor_is_not_ready(self):
        for value in (0, 0.01, 0.8, None, float("nan")):
            state = readings(); state["irsensor"][1]["range"] = value
            self.assertFalse(self.evaluate(state)["protection"]["ready"], value)

    def test_unknown_sensor_returns_never_claim_clear(self):
        for value in (None, 0, -1, float("nan"), float("inf"), "bad", 0.1):
            state = readings(); state["ultrasonic"][0]["range"] = value
            self.assertFalse(self.evaluate(state)["protection"]["ready"], value)

    def test_disabled_config_and_missing_readings_are_not_ready(self):
        for namespace in (p.ULTRASOUND, p.CLIFF, p.NAV, p.FOLLOW):
            params = settings(); params[namespace] = ["false"] if isinstance(params[namespace], list) else "false"
            self.assertFalse(self.evaluate(params=params)["protection"]["ready"])
        state = readings(); state["ultrasonic"].pop()
        self.assertFalse(self.evaluate(state)["protection"]["ready"])

    def test_stale_failed_or_never_read_is_not_healthy(self):
        for age in (None, 2.1):
            self.assertFalse(self.evaluate(ages={"ultrasonic": age, "irsensor": 0})["protection"]["ready"])
        self.assertFalse(self.evaluate(errors={"ultrasonic": "failed"})["protection"]["ready"])
        self.assertFalse(self.evaluate(param_age=11)["protection"]["ready"])
        self.assertFalse(self.evaluate(param_age=None)["protection"]["ready"])

    def test_extra_sensors_not_fabricated_as_enabled_and_cliff_switches_used(self):
        state = readings(); state["ultrasonic"] += [{"range": 0}, {"range": 0}]
        params = settings(); params[p.CLIFF] = ["false"] * 4
        sensors = self.evaluate(state, params)["sensors"]
        self.assertIsNone(sensors[8]["enabled"])
        self.assertIsNone(sensors[9]["enabled"])
        self.assertTrue(all(s["enabled"] is False for s in sensors[10:]))

    def test_invalid_limits_block(self):
        params = settings(); params["/devices/fallprevention/min_range_fallprevention"] = ["0.5"] * 4
        self.assertFalse(self.evaluate(params=params)["protection"]["ready"])

    def test_sensor_array_write_preserves_firmware_array_metadata(self):
        client = DashboardClient("unused")
        client._call = Mock()
        client.write_params([{"namespace": p.ULTRASOUND, "type": "bool", "value": ["true"] * 8},
                             {"namespace": p.NAV, "type": "bool", "value": "true"}])
        payload = client._call.call_args.args[1]["params"]
        self.assertIs(payload[0]["isArray"], True)
        self.assertEqual(payload[0]["value"], ["true"] * 8)
        self.assertIs(payload[1]["isArray"], False)

    def test_enable_preserves_reported_lengths_and_optional_absence(self):
        table = {name: SimpleNamespace(value=value, type="bool", is_array=isinstance(value, list))
                 for name, value in settings().items() if name in p.SWITCHES}
        updates = p.enable_updates(table)
        self.assertEqual(len(updates), 4)
        self.assertEqual(len(updates[0]["value"]), 8)
        p.verify_updates(table, updates)
        table[p.ULTRASOUND].value = ["false"] * 8
        with self.assertRaisesRegex(ValueError, "readback"):
            p.verify_updates(table, updates)

    def test_missing_required_setting_fails_before_any_write(self):
        with self.assertRaises(ValueError):
            p.enable_updates({})

    def test_gate_missing_protection_allows_manual_driving(self):
        self.assertIsNone(drive.motion_gate({"control": {"held": True}}))

    def test_unavailable_sensors_warn_and_allow_every_manual_direction(self):
        state = readings()
        for group in ("ultrasonic", "irsensor"):
            for sensor in state[group]:
                sensor["range"] = 0
        result = self.evaluate(state)
        self.assertFalse(result["protection"]["ready"])
        self.assertIsNone(result["protection"]["blocked_by"])
        self.assertIn("unavailable", result["protection"]["warning"])
        snapshot = {"control": {"held": True}, "obstacles": result}
        for linear, angular in ((.2, 0), (-.2, 0), (0, .4), (0, -.4)):
            with self.subTest(linear=linear, angular=angular):
                client = Mock()
                loop = drive.DriveLoop(client, lambda: drive.motion_gate(snapshot))
                self.assertTrue(loop.request(linear, angular)["accepted"])
                loop._tick(time.monotonic())
                sent = client.move.call_args.args[0]
                self.assertEqual((sent.linear, sent.angular), (linear, angular))
                client.set_brake.assert_not_called()
                self.assertIsNone(loop.status()["protection_latched"])

    def test_detection_loss_during_turn_warns_without_latching(self):
        snapshot = {"control": {"held": True}, "obstacles": self.evaluate()}
        client = Mock()
        loop = drive.DriveLoop(client, lambda: drive.motion_gate(snapshot))
        loop.request(0, .4)
        loop._tick(time.monotonic())
        snapshot["obstacles"] = self.evaluate(ages={"ultrasonic": 3, "irsensor": 3})
        loop._tick(time.monotonic())
        self.assertEqual(client.move.call_args.args[0].angular, .4)
        client.set_brake.assert_not_called()
        self.assertIsNone(loop.status()["protection_latched"])
        self.assertFalse(snapshot["obstacles"]["protection"]["ready"])

    def test_fresh_hazard_still_blocks_with_other_sensor_unavailable(self):
        for kind, value in (("ultrasonic", .2), ("irsensor", .8)):
            with self.subTest(kind=kind):
                state = readings()
                state["ultrasonic"][0]["range"] = 0
                state[kind][1]["range"] = value
                result = self.evaluate(state)
                self.assertIsNotNone(result["protection"]["warning"])
                self.assertIn("obstacle protection", drive.motion_gate(
                    {"control": {"held": True}, "obstacles": result}))

    def test_detection_warnings_do_not_bypass_other_interlocks(self):
        result = self.evaluate(errors={"ultrasonic": "no sensor", "irsensor": "no sensor"})
        for change in ({"brake": {"engaged": True}}, {"bumper": {"estop": True}},
                       {"bumper": {"front": True}}, {"bumper": {"rear": True}},
                       {"safety_stale": True}, {"reachable": False},
                       {"control": {"held": False}}):
            with self.subTest(change=change):
                self.assertIsNotNone(drive.motion_gate(
                    {"control": {"held": True}, "obstacles": result, **change}))

    def test_missing_configuration_or_malformed_data_warn_without_blocking(self):
        cases = [self.evaluate(params={p.NAV: "false"}),
                 self.evaluate(param_age=11),
                 self.evaluate(errors={"params": "failed"}),
                 self.evaluate({"ultrasonic": "bad", "irsensor": []})]
        for result in cases:
            with self.subTest(result=result):
                self.assertFalse(result["protection"]["ready"])
                self.assertIsNotNone(result["protection"]["warning"])
                self.assertIsNone(drive.motion_gate(
                    {"control": {"held": True}, "obstacles": result}))

    def test_sensor_fault_warns_but_motor_fault_blocks(self):
        for name, blocks in (("Laser", False), ("Ultrasonic", False),
                             ("Infrared", False), ("Right_Motor", True)):
            state = readings()
            state["chassis"] = {"devices": {name: {"healthy": False}}}
            result = self.evaluate(state)
            reason = drive.motion_gate({"control": {"held": True}, "obstacles": result})
            self.assertEqual(reason is not None, blocks, name)
            self.assertFalse(result["protection"]["ready"])

    def test_unavailable_detection_still_obeys_deadman(self):
        client = Mock()
        snapshot = {"control": {"held": True}, "obstacles": self.evaluate(
            errors={"ultrasonic": "failed", "irsensor": "failed"})}
        loop = drive.DriveLoop(client, lambda: drive.motion_gate(snapshot))
        self.assertTrue(loop.request(.2, 0)["accepted"])
        loop._tick(time.monotonic() + drive.DEADMAN_S + .1)
        self.assertTrue(client.move.call_args.args[0].is_zero())
        self.assertIn("deadman", loop.status()["blocked_by"])

    def test_each_manual_direction_rejected_on_obstacle(self):
        for linear, angular in ((0.2, 0), (-0.2, 0), (0, 0.4), (0, -0.4)):
            client = Mock()
            loop = drive.DriveLoop(client, lambda: "obstacle protection: nearby object")
            self.assertFalse(loop.request(linear, angular)["accepted"])
            client.move.assert_not_called()
            client.set_brake.assert_called_with(True)

    def test_obstacle_during_turn_brakes_latches_and_never_auto_resumes(self):
        client = Mock(); reason = [None]
        loop = drive.DriveLoop(client, lambda: reason[0])
        loop.request(0, 0.4); loop._tick(time.monotonic())
        self.assertEqual(client.move.call_args.args[0].angular, 0.4)
        reason[0] = "obstacle protection: nearby object"
        loop._tick(time.monotonic())
        client.set_brake.assert_called_with(True)
        self.assertGreaterEqual(client.stop.call_count, 3)
        reason[0] = None
        self.assertFalse(loop.request(0, 0.4)["accepted"])
        self.assertTrue(loop.status()["protection_latched"])
        loop.release_brake()
        self.assertTrue(loop.request(0, 0.4)["accepted"])

    def test_failed_brake_still_zeroes_and_latches(self):
        from drobot_api import DRobotError
        client = Mock(); client.set_brake.side_effect = DRobotError("HTTP 500")
        loop = drive.DriveLoop(client, lambda: "obstacle protection: nearby object")
        self.assertFalse(loop.request(.2, 0)["accepted"])
        self.assertGreaterEqual(client.stop.call_count, 3)
        self.assertTrue(loop.status()["protection_latched"])

    def test_malformed_sensor_payload_warns_instead_of_crashing(self):
        for value in (7, "invalid", {"bad": True}):
            state = readings(); state["ultrasonic"] = value
            self.assertFalse(self.evaluate(state)["protection"]["ready"])

    def test_unexpected_gate_exception_stops_instead_of_killing_drive_thread(self):
        client = Mock(); reason = Mock(return_value=None)
        loop = drive.DriveLoop(client, reason)
        loop.request(.2, 0); loop._tick(time.monotonic())
        reason.side_effect = ValueError("malformed feedback")
        loop._tick(time.monotonic())
        client.set_brake.assert_called_with(True)
        self.assertTrue(loop.status()["protection_latched"])

    def test_sensor_network_stall_does_not_hold_snapshot_lock(self):
        lock = threading.RLock(); started = threading.Event(); finish = threading.Event()
        def slow_read():
            started.set(); finish.wait(2); return 1
        poller = Poller("sensor", .5, slow_read, lambda state, value: state.update(value=value))
        thread = threading.Thread(target=poller.run, args=({}, time.monotonic(), lock))
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            acquired = lock.acquire(timeout=.1)
            self.assertTrue(acquired, "stale-data stop cannot run while network holds snapshot lock")
            if acquired: lock.release()
        finally:
            finish.set(); thread.join(2)


if __name__ == "__main__":
    unittest.main()
