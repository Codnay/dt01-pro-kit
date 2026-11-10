"""Teach ownership reaches the operator loop through both controller paths."""

from __future__ import annotations

import importlib
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from arms.data_manager import DataManager, RobotActivityState
from arms.perf import piper_process as ipc


class TeachOperatorStateTests(unittest.TestCase):
    """A physical Teach retires the clutch without implicitly rearming it."""

    def test_stop_latch_retires_teleop_and_brain_then_requires_enable(self):
        with patch.dict(sys.modules, {"arms.piper_controller": SimpleNamespace(PiperController=object)}):
            module = importlib.import_module("arms.threads.joint_state")
        data = DataManager()
        data.set_robot_activity_state(RobotActivityState.ENABLED, "left")
        data.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        data.set_teleop_state("left", True, np.eye(4), np.eye(4))
        brain = data.get_brain_control("left")
        data.set_brain_control_params(
            armed=True, diameter=brain.diameter, slider_norm=brain.slider_norm,
            clamp_mode=brain.clamp_mode, active_group=brain.active_group,
            selected_alpha=brain.selected_alpha, unselected_alpha=brain.unselected_alpha,
            hand="left",
        )
        controller = SimpleNamespace(is_stop_hold_latched=lambda: True)
        module._apply_controller_stop_latch(data, controller, "left")
        self.assertFalse(data.get_teleop_active("left"))
        self.assertFalse(data.get_brain_control_armed("left"))
        self.assertEqual(data.get_robot_activity_state("left"), RobotActivityState.PAUSED)
        self.assertEqual(data.get_robot_activity_state("right"), RobotActivityState.ENABLED)
        controller.is_stop_hold_latched = lambda: False
        module._apply_controller_stop_latch(data, controller, "left")
        self.assertEqual(data.get_robot_activity_state("left"), RobotActivityState.PAUSED)

    def test_proxy_publishes_latch_without_rpc_and_ignores_pre_enable_state(self):
        state = ipc._SeqBlock(ipc._STATE_SLOTS, create=True)
        try:
            proxy = object.__new__(ipc.PiperControllerProxy)
            proxy._state = state
            proxy._state_dirty_since = 0
            proxy._rpc = Mock(side_effect=AssertionError("hot getter used RPC"))
            now = time.time()
            state.write({ipc._S_STAMP: now, ipc._S_STOP_LATCHED: 1})
            self.assertTrue(proxy.is_stop_hold_latched())
            proxy._state_dirty_since = now + 0.001
            self.assertFalse(proxy.is_stop_hold_latched())
            state.write({ipc._S_STAMP: now + 0.002, ipc._S_STOP_LATCHED: 0})
            self.assertFalse(proxy.is_stop_hold_latched())
            proxy._rpc.assert_not_called()
        finally:
            state.close(unlink=True)


if __name__ == "__main__":
    unittest.main()
