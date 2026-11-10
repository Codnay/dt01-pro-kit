"""The J6 thumbstick must reach the solver without a held dead-man.

J6 is not solved by IK. It is jogged directly from the controller's Y axis
(``ik_solver`` applies ``joystick_y * J6_JOYSTICK_SPEED * dt``), and that gate
asks only for ``RobotActivityState.ENABLED``.

Commit 10dea9ea added a second, undocumented requirement in the polling thread:
the stick was forced to 0.0 unless the grip had been active on the PREVIOUS
iteration. It was aimed at a real incident -- a Quest thumbstick froze at 0.9228
after a transport drop and drove left J6 to its +100 limit -- but it made the
wrist silently immovable in normal use on both arms, and the rule lived in a
different file from the logic it constrained.

Staleness is handled where it belongs instead: the transport's
``input_available`` fails on a pose older than 250 ms, a dropped lease or a
changed connection epoch, and ``get_joystick_value`` returns neutral then. These
tests hold both halves of that contract.
"""

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

from arms.data_manager import DataManager  # noqa: E402
from arms.threads.quest_reader import quest_reader_thread  # noqa: E402

JOG = 0.8  # well past J6_JOYSTICK_DEADZONE (0.1)


class _FakeQuest:
    """A LiveKit-shaped reader: it exposes ``input_available``.

    That attribute is what arms the polling thread's transport-aware paths, so
    a fake without it would exercise the wrong branch entirely.
    """

    def __init__(self) -> None:
        self.trigger = 0.0
        self.grip = 0.0
        self.joystick = 0.0
        self.available = True
        self.revision = 0

    def get_grip_value(self, hand: str) -> float:
        return self.grip

    def get_trigger_value(self, hand: str) -> float:
        return self.trigger

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray:
        return np.eye(4)

    def get_joystick_value(self, hand: str) -> tuple[float, float]:
        # Mirrors the real reader: neutral whenever input is not available.
        if not self.available:
            return (0.0, 0.0)
        return (0.0, self.joystick)

    def input_available(self, hand: str) -> bool:
        return self.available

    def input_revision(self) -> tuple:
        return (self.revision,)


def _wait(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class QuestReaderJoystickTests(unittest.TestCase):
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

    def test_stick_reaches_the_solver_with_no_grip_and_no_trigger(self) -> None:
        """The regression this file exists for.

        Dead-man released, never pressed since startup -- exactly the state in
        the 2026-09-26 report, where both arms' J6 would not move.
        """
        self.quest.grip = 0.0
        self.quest.trigger = 0.0
        self.quest.joystick = JOG

        self.assertTrue(
            _wait(lambda: self.dm.get_joystick_y("right") == JOG),
            f"stick never reached DataManager (saw {self.dm.get_joystick_y('right')})",
        )

    def test_stick_follows_the_axis_both_ways(self) -> None:
        """Not merely non-zero: the value is passed through, sign and all."""
        self.quest.joystick = JOG
        self.assertTrue(_wait(lambda: self.dm.get_joystick_y("right") == JOG))

        self.quest.joystick = -JOG
        self.assertTrue(
            _wait(lambda: self.dm.get_joystick_y("right") == -JOG),
            "negative deflection did not pass through",
        )

        self.quest.joystick = 0.0
        self.assertTrue(_wait(lambda: self.dm.get_joystick_y("right") == 0.0))

    def test_unavailable_input_still_neutralises_a_stuck_axis(self) -> None:
        """The frozen-thumbstick protection must survive removing the grip gate.

        This is the 2026-09-15 incident: the axis stays pinned at a large value
        while the transport stops delivering. Nothing may reach J6.
        """
        self.quest.joystick = 0.9228  # the value the real stick froze at
        self.assertTrue(_wait(lambda: self.dm.get_joystick_y("right") == 0.9228))

        self.quest.available = False  # transport drops; the axis never changes

        self.assertTrue(
            _wait(lambda: self.dm.get_joystick_y("right") == 0.0),
            "a stuck axis survived an unavailable transport",
        )

    def test_recovered_transport_restores_the_stick_without_a_grip_press(self) -> None:
        """Recovery must not need a dead-man cycle the operator cannot know about."""
        self.quest.joystick = JOG
        self.quest.available = False
        self.assertTrue(_wait(lambda: self.dm.get_joystick_y("right") == 0.0))

        self.quest.available = True

        self.assertTrue(
            _wait(lambda: self.dm.get_joystick_y("right") == JOG),
            "stick stayed dead after the transport came back",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
