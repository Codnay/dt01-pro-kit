"""Per-hand control-source codes (activity_mode) used to label dataset frames."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import numpy as np  # noqa: E402

from arms.data_manager import (  # noqa: E402
    ACTIVITY_MODE_CODES,
    ACTIVITY_MODE_NAMES,
    DataManager,
    RobotActivityState,
    activity_mode_code,
)


class ActivityModeTests(unittest.TestCase):
    def test_policy_control_is_never_teleop(self):
        for active in (False, True):
            for driven in (False, True):
                self.assertEqual(
                    activity_mode_code(RobotActivityState.POLICY_CONTROLLED, active, driven),
                    ACTIVITY_MODE_CODES["policy"],
                )

    def test_operator_states(self):
        self.assertEqual(activity_mode_code(RobotActivityState.ENABLED, True), ACTIVITY_MODE_CODES["teleop"])
        self.assertEqual(activity_mode_code(RobotActivityState.ENABLED, False, driven=True), ACTIVITY_MODE_CODES["teleop"])
        self.assertEqual(activity_mode_code(RobotActivityState.ENABLED, False), ACTIVITY_MODE_CODES["idle"])
        self.assertEqual(activity_mode_code(RobotActivityState.PRESET_CONTROLLED, True), ACTIVITY_MODE_CODES["preset"])
        self.assertEqual(activity_mode_code(RobotActivityState.HOMING, True), ACTIVITY_MODE_CODES["homing"])
        self.assertEqual(activity_mode_code(RobotActivityState.PAUSED, True), ACTIVITY_MODE_CODES["paused"])
        self.assertEqual(activity_mode_code(RobotActivityState.DISABLED, False), ACTIVITY_MODE_CODES["disabled"])

    def test_codes_are_unique_and_named(self):
        self.assertEqual(len(set(ACTIVITY_MODE_CODES.values())), len(ACTIVITY_MODE_CODES))
        self.assertEqual(ACTIVITY_MODE_NAMES[ACTIVITY_MODE_CODES["policy"]], "policy")
        self.assertEqual(ACTIVITY_MODE_CODES["policy"], 7)


if __name__ == "__main__":
    unittest.main()


# ``joint_state`` imports ``piper_controller``, which needs the robot-host-only
# ``piper_sdk`` and ``scipy``. Neither is installable here, and neither is used by
# the code under test, so both are stubbed before the import (same approach as
# tests/test_ik_solver_thread.py).
for _name, _attrs in (
    ("piper_sdk", {"C_PiperInterface_V2": type("C_PiperInterface_V2", (), {})}),
    ("scipy", {}),
    ("scipy.spatial", {}),
    ("scipy.spatial.transform", {"Rotation": type("Rotation", (), {})}),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

from arms.threads.joint_state import joint_state_thread  # noqa: E402


class _FakeArm:
    """The handful of PiperController calls a non-ENABLED tick makes."""

    def get_current_joint_angles(self) -> np.ndarray:
        """Measured joints (degrees): a still arm."""
        return np.zeros(6, dtype=float)

    def get_current_gripper_open_value(self) -> float:
        """Measured gripper open amount."""
        return 1.0


class _BoundedDataManager(DataManager):
    """A DataManager that shuts the loop down after ``ticks`` iterations."""

    def __init__(self, ticks: int = 3) -> None:
        """Allow ``ticks`` loop iterations, then report shutdown."""
        super().__init__()
        self._remaining = int(ticks)

    def is_shutdown_requested(self) -> bool:
        """False for the first ``ticks`` calls, then True."""
        self._remaining -= 1
        return self._remaining < 0


class ActivityModeReachesTelemetryTests(unittest.TestCase):
    """The label must be EMITTED in every state -- not merely computed correctly.

    2026-09-07: ``activity_mode_code`` was right, but its only call site sat inside
    the joint-state loop's ``elif robot_activity_state in (ENABLED, PAUSED)`` arm,
    so the one state the signal exists to record -- POLICY_CONTROLLED -- emitted no
    row at all. 14 policy runs reached BigQuery unlabelled, the episode exporter's
    ``_policy_session()`` found no code 7, and its fail-closed drop of
    policy-commanded frames never fired. The unit tests above passed throughout:
    they exercised the function, never the call site.
    """

    def _codes_for(self, state: RobotActivityState) -> list[float]:
        dm = _BoundedDataManager(ticks=3)
        seen: list[float] = []
        dm.set_on_change_callback(
            lambda name, value, ts: (
                seen.append(float(value))
                if name == "log_activity_mode/right"
                else None
            )
        )
        dm.set_robot_activity_state(state, "right")
        joint_state_thread(dm, _FakeArm(), hand="right")
        return seen

    def test_policy_controlled_ticks_emit_the_policy_code(self) -> None:
        codes = self._codes_for(RobotActivityState.POLICY_CONTROLLED)
        self.assertTrue(codes, "POLICY_CONTROLLED emitted no activity_mode row")
        self.assertEqual(set(codes), {float(ACTIVITY_MODE_CODES["policy"])})

    def test_preset_controlled_ticks_emit_too(self) -> None:
        codes = self._codes_for(RobotActivityState.PRESET_CONTROLLED)
        self.assertTrue(codes, "PRESET_CONTROLLED emitted no activity_mode row")
        self.assertEqual(set(codes), {float(ACTIVITY_MODE_CODES["preset"])})

    def test_enabled_ticks_still_emit(self) -> None:
        codes = self._codes_for(RobotActivityState.ENABLED)
        self.assertTrue(codes, "ENABLED emitted no activity_mode row")
        self.assertEqual(set(codes), {float(ACTIVITY_MODE_CODES["idle"])})
