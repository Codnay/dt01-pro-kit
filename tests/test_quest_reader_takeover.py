from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

try:  # the real scipy when installed; the stub below only stands in for a missing one
    import scipy.spatial.transform  # noqa: F401
except Exception:  # noqa: BLE001
    pass
if "scipy.spatial.transform" not in sys.modules:
    scipy_mod = types.ModuleType("scipy")
    spatial_mod = types.ModuleType("scipy.spatial")
    transform_mod = types.ModuleType("scipy.spatial.transform")

    class _DummyRotation:
        @staticmethod
        def from_matrix(_matrix: Any) -> "_DummyRotation":
            return _DummyRotation()

        @staticmethod
        def from_quat(_quat: Any) -> "_DummyRotation":
            return _DummyRotation()

        def as_quat(self) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    transform_mod.Rotation = _DummyRotation
    spatial_mod.transform = transform_mod
    scipy_mod.spatial = spatial_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.spatial", spatial_mod)
    sys.modules.setdefault("scipy.spatial.transform", transform_mod)

from arms.data_manager import DataManager, RobotActivityState  # noqa: E402
from arms.threads.quest_reader import quest_reader_thread  # noqa: E402


class _FakeQuest:
    def __init__(self) -> None:
        self.trigger = 0.0

    def get_grip_value(self, hand: str) -> float:
        return 0.0

    def get_trigger_value(self, hand: str) -> float:
        return self.trigger

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray:
        return np.eye(4)

    def get_joystick_value(self, hand: str) -> tuple[float, float]:
        return (0.0, 0.0)


def _wait(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class QuestReaderTakeoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dm = DataManager()
        self.dm.set_current_end_effector_pose(np.eye(4), "right")
        self.quest = _FakeQuest()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=quest_reader_thread,
            args=(self.dm, self.quest, ("right",)),
            kwargs={"stop_event": self.stop},
            daemon=True,
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)

    def _settle_released(self) -> None:
        self.quest.trigger = 0.0
        time.sleep(0.3)  # > TELEOP_DEADMAN_RELEASE_OBSERVE_SAMPLES at 50 Hz

    def test_squeeze_under_policy_control_files_a_takeover_then_arms_on_enabled(
        self,
    ) -> None:
        self.dm.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED, "right")
        self._settle_released()
        self.quest.trigger = 1.0
        self.assertTrue(_wait(lambda: self.dm.policy_takeover_pending() == ("right",)))
        time.sleep(0.2)
        self.assertFalse(
            self.dm.get_teleop_active("right"),
            "teleop must not arm while the policy holds the arm",
        )
        # The policy runner consumes the request, holds the arms and flips them to ENABLED.
        self.assertEqual(self.dm.consume_policy_takeover(), ("right",))
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        self.assertTrue(
            _wait(lambda: self.dm.get_teleop_active("right")),
            "teleop should arm on the same squeeze",
        )
        self.quest.trigger = 0.0
        self.assertTrue(_wait(lambda: not self.dm.get_teleop_active("right")))

    def test_release_before_handoff_abandons_the_takeover(self) -> None:
        self.dm.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED, "right")
        self._settle_released()
        self.quest.trigger = 1.0
        self.assertTrue(_wait(lambda: self.dm.policy_takeover_pending() == ("right",)))
        self.quest.trigger = 0.0
        time.sleep(0.4)
        self.dm.consume_policy_takeover()
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        time.sleep(0.3)
        self.assertFalse(self.dm.get_teleop_active("right"))

    def test_squeeze_when_enabled_arms_teleop_directly(self) -> None:
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        self._settle_released()
        self.quest.trigger = 1.0
        self.assertTrue(_wait(lambda: self.dm.get_teleop_active("right")))
        self.assertEqual(self.dm.policy_takeover_pending(), ())

    def test_squeeze_when_paused_does_nothing(self) -> None:
        self.dm.set_robot_activity_state(RobotActivityState.PAUSED, "right")
        self._settle_released()
        self.quest.trigger = 1.0
        time.sleep(0.3)
        self.assertFalse(self.dm.get_teleop_active("right"))
        self.assertEqual(self.dm.policy_takeover_pending(), ())


class _FreshQuest(_FakeQuest):
    def __init__(self):
        super().__init__()
        self.available = True
        self.revision = 0
        self.frozen_seq = False

    def input_available(self, hand):
        return self.available

    def input_revision(self):
        return self.revision

    def get_seq(self, hand):
        return 1 if self.frozen_seq else int(time.monotonic() * 100)


class QuestRecoverySafetyTests(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager()
        self.dm.set_current_end_effector_pose(np.eye(4), "right")
        self.dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        self.quest = _FreshQuest()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=quest_reader_thread,
            args=(self.dm, self.quest, ("right",)),
            kwargs={"stop_event": self.stop},
        )
        self.thread.start()

    def tearDown(self):
        self.stop.set()
        self.thread.join(2)

    def test_loss_holds_immediately_and_held_trigger_cannot_resume(self):
        time.sleep(0.35)
        self.quest.trigger = 1
        self.assertTrue(_wait(lambda: self.dm.get_teleop_active("right")))
        self.quest.available = False
        self.assertTrue(_wait(lambda: not self.dm.get_teleop_active("right"), 0.12))
        self.quest.available = True
        time.sleep(0.4)
        self.assertFalse(self.dm.get_teleop_active("right"))
        self.quest.trigger = 0
        time.sleep(0.35)
        self.quest.trigger = 1
        self.assertTrue(_wait(lambda: self.dm.get_teleop_active("right")))

    def test_recovery_between_polls_still_invalidates_previously_observed_release(self):
        time.sleep(0.35)
        self.quest.revision += 1
        self.quest.trigger = 1
        time.sleep(0.4)
        self.assertFalse(self.dm.get_teleop_active("right"))
        self.assertEqual(
            self.dm.get_robot_activity_state("right"), RobotActivityState.ENABLED
        )

    def test_repeated_cached_release_does_not_arm(self):
        self.quest.revision += 1
        self.quest.frozen_seq = True
        time.sleep(0.4)
        self.quest.trigger = 1
        self.quest.frozen_seq = False
        time.sleep(0.3)
        self.assertFalse(self.dm.get_teleop_active("right"))

    def test_reconnect_does_not_resume_policy_or_clear_pause(self):
        self.dm.set_robot_activity_state(RobotActivityState.PAUSED, "right")
        self.quest.available = False
        self.quest.revision += 1
        time.sleep(0.1)
        self.quest.available = True
        self.quest.trigger = 1
        time.sleep(0.3)
        self.assertEqual(
            self.dm.get_robot_activity_state("right"), RobotActivityState.PAUSED
        )
        self.assertEqual(self.dm.policy_takeover_pending(), ())
        self.assertFalse(self.dm.get_teleop_active("right"))


if __name__ == "__main__":
    unittest.main()
