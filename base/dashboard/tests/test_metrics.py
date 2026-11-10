"""Tests for the Prometheus exporter.

    python3 -m unittest discover -s dashboard/tests -t dashboard

The property worth defending here is that a value the robot does not report is
*absent* from the scrape rather than zero. Prometheus has no way to express "I
asked and got nothing" other than a gap, and a zero-filled gap draws a
confident flat line that reads as a real measurement.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import metrics                                            # noqa: E402

# A snapshot from the live DT-01 on 2026-09-01, trimmed to the fields the
# exporter reads. battery_current/temperature really are a hard 0.0 on this
# firmware whether or not the robot knows them.
LIVE = {
    "reachable": True,
    "localized": True,
    "safety_stale": False,
    "safety_data_age": 0.3,
    "battery": {"battery_voltage": 218.0, "battery_current": 0.0,
                "battery_capacity_percentage": 52.0, "battery_temperature": 0.0},
    "chassis": {},
    "brake": {"engaged": False},
    "bumper": {"front": False, "rear": False, "estop": False},
    "robot": {"state_operation": "RunningTask"},
    "localization": {"status": "normal", "initialized": True, "score": 0.96},
    "nav": {"nav_status_type": "idle"},
    "velocity": {"linear": 0.0, "angular": 0.0},
    "drive": {"armed": False, "commands_sent": 28},
    "obstacles": {"alarm": False, "alarm_distance": 0.35, "sensors": [
        {"kind": "ultrasound", "range": 0.0, "enabled": False, "triggered": False},
        {"kind": "ultrasound", "range": 0.9, "enabled": True, "triggered": False},
        {"kind": "irsensor", "range": 0.2, "enabled": True, "triggered": True},
    ]},
    "link": {
        "notice_feed": {"connected": True, "messages": 2375, "age_s": 0.34},
        "scan_feed": {"connected": True, "messages": 1310, "age_s": 0.47},
        "poll_errors": {"bumper": "rejected", "chassis": "rejected"},
    },
}


def sample(text: str, name: str) -> list[str]:
    """Every sample line for one metric family."""
    return [line for line in text.splitlines()
            if line.startswith(name + "{") or line == name]


class TestExposition(unittest.TestCase):
    def setUp(self) -> None:
        self.text = metrics.render(LIVE, "dt01-test")

    def test_every_sample_carries_robot_id(self) -> None:
        # Without it two platforms merge into one history. It is emitted here
        # rather than attached by the scraper on purpose — a target label wins
        # and would rename ours to exported_robot_id.
        for line in self.text.splitlines():
            if not line.startswith("#") and line.strip():
                self.assertIn('robot_id="dt01-test"', line)

    def test_help_and_type_precede_every_family(self) -> None:
        seen: set[str] = set()
        for line in self.text.splitlines():
            if line.startswith("# TYPE "):
                seen.add(line.split()[2])
            elif line and not line.startswith("#"):
                family = line.split("{")[0]
                self.assertIn(family, seen, f"{family} emitted with no # TYPE")

    def test_voltage_is_normalised_out_of_decivolts(self) -> None:
        self.assertIn("dt01_battery_volts{robot_id=\"dt01-test\"} 21.8", self.text)

    def test_percentage_and_score_survive(self) -> None:
        self.assertEqual(sample(self.text, "dt01_battery_percent"),
                         ['dt01_battery_percent{robot_id="dt01-test"} 52'])
        self.assertEqual(sample(self.text, "dt01_localization_score"),
                         ['dt01_localization_score{robot_id="dt01-test"} 0.96'])


class TestUnknownIsAbsent(unittest.TestCase):
    """A value the robot does not report must not be exported as zero."""

    def test_charging_is_omitted_when_the_robot_cannot_say(self) -> None:
        text = metrics.render(LIVE, "dt01-test")
        self.assertEqual(sample(text, "dt01_charging"), [],
                         "a dead current sensor was exported as 'not charging'")

    def test_charging_is_exported_when_the_robot_does_say(self) -> None:
        snapshot = {**LIVE, "battery": {**LIVE["battery"], "battery_current": 2.4}}
        self.assertEqual(sample(metrics.render(snapshot, "dt01-test"), "dt01_charging"),
                         ['dt01_charging{robot_id="dt01-test"} 1'])

    def test_missing_battery_block_emits_no_battery_series(self) -> None:
        text = metrics.render({**LIVE, "battery": {}}, "dt01-test")
        self.assertEqual(sample(text, "dt01_battery_volts"), [])
        self.assertEqual(sample(text, "dt01_battery_percent"), [])

    def test_a_sensor_seeing_nothing_is_not_an_obstacle_at_zero_metres(self) -> None:
        # The controller reports 0 for an idle sensor. Exported literally that
        # is "something touching the robot", which is the opposite of the truth.
        text = metrics.render(LIVE, "dt01-test")
        ultrasound = sample(text, "dt01_obstacle_nearest_meters")
        self.assertTrue(any('kind="ultrasound"' in line and line.endswith(" 0.9")
                            for line in ultrasound), ultrasound)
        self.assertFalse(any(line.endswith(" 0") for line in ultrasound), ultrasound)


class TestScrapeAlwaysSucceeds(unittest.TestCase):
    """A scrape must distinguish 'robot down' from 'exporter down'."""

    def test_no_snapshot_still_renders_up_zero(self) -> None:
        text = metrics.render(None, "dt01-test")
        self.assertEqual(sample(text, "dt01_up"), ['dt01_up{robot_id="dt01-test"} 0'])

    def test_unreachable_robot_is_up_zero(self) -> None:
        text = metrics.render({**LIVE, "reachable": False}, "dt01-test")
        self.assertEqual(sample(text, "dt01_up"), ['dt01_up{robot_id="dt01-test"} 0'])

    def test_a_snapshot_of_junk_does_not_raise(self) -> None:
        junk = {"battery": "not a dict", "obstacles": {"sensors": [None, 7]},
                "link": {"notice_feed": []}, "nav": None, "drive": 5}
        self.assertIn("dt01_up", metrics.render(junk, "dt01-test"))


class TestSafetySignals(unittest.TestCase):
    def test_blind_interlocks_are_visible_as_a_series(self) -> None:
        text = metrics.render({**LIVE, "safety_stale": True}, "dt01-test")
        self.assertEqual(sample(text, "dt01_safety_stale"),
                         ['dt01_safety_stale{robot_id="dt01-test"} 1'])

    def test_teleop_lockout_is_exported(self) -> None:
        text = metrics.render(LIVE, "dt01-test", teleop_disabled=True)
        self.assertEqual(sample(text, "dt01_teleop_disabled"),
                         ['dt01_teleop_disabled{robot_id="dt01-test"} 1'])

    def test_nav_active_tracks_a_running_goal(self) -> None:
        for state, expected in (("idle", "0"), ("navigation", "1"),
                                ("follow_path", "1")):
            snapshot = {**LIVE, "nav": {"nav_status_type": state}}
            self.assertEqual(sample(metrics.render(snapshot, "dt01-test"),
                                    "dt01_nav_active"),
                             [f'dt01_nav_active{{robot_id="dt01-test"}} {expected}'])

    def test_label_values_are_escaped(self) -> None:
        snapshot = {**LIVE, "robot": {"state_operation": 'we"ird\\'}}
        text = metrics.render(snapshot, "dt01-test")
        self.assertIn(r'state="we\"ird\\"', text)


if __name__ == "__main__":
    unittest.main()
