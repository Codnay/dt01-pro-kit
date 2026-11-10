"""Offline regressions for soft-limit recovery, handoff, and final containment."""

import multiprocessing as mp
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from arms.joint_limits import clamp_recovery_step, recovery_joint_bounds  # noqa: E402

HARD = np.array(
    [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120.0]]
)
SOFT = HARD + np.array([[3, -3]] * 5 + [[20, -20]])


class RecoveryBoundsTests(unittest.TestCase):
    def test_hold_in_multiple_margins_is_unchanged(self):
        held = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 119.0])
        np.testing.assert_array_equal(clamp_recovery_step(held, held, SOFT, HARD), held)
        outward = held + [0, -1, 1, 0, 0, 1]
        np.testing.assert_array_equal(
            clamp_recovery_step(held, outward, SOFT, HARD), held
        )

    def test_progress_cannot_be_undone_even_before_reaching_soft_limit(self):
        held = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        inward = held + [0, 0.1, -0.1, 0, 0, 0]
        accepted = clamp_recovery_step(held, inward, SOFT, HARD)
        np.testing.assert_array_equal(
            clamp_recovery_step(accepted, held, SOFT, HARD), accepted
        )
        inside = held + [0, 4, -4, 0, 0, 0]
        bounds = recovery_joint_bounds(inside, SOFT, HARD)
        np.testing.assert_array_equal(bounds, SOFT)

    def test_nonfinite_or_hard_invalid_hold_is_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            held = np.zeros(6)
            held[1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                recovery_joint_bounds(held, SOFT, HARD)

    def test_random_steps_never_increase_any_limit_violation(self):
        rng = np.random.default_rng(19)
        for _ in range(500):
            held = rng.uniform(HARD[:, 0], HARD[:, 1])
            request = held + rng.normal(0, 30, 6)
            result = clamp_recovery_step(held, request, SOFT, HARD)
            before = np.abs(held - np.clip(held, SOFT[:, 0], SOFT[:, 1]))
            after = np.abs(result - np.clip(result, SOFT[:, 0], SOFT[:, 1]))
            self.assertTrue(np.all(after <= before))
            self.assertTrue(np.all(result >= HARD[:, 0]))
            self.assertTrue(np.all(result <= HARD[:, 1]))


class RecoveryThreadTests(unittest.TestCase):
    def run_reference(self, held, candidate=None, loops=5, before=None):
        from test_teleop_reference import (
            FeedbackSequence,
            IncrementSolver,
            ik_module,
            _capture_piper_logs,
        )

        held = np.asarray(held, dtype=float)
        events = []
        clock = [0.0]

        def prepare(dm, index):
            clock[0] += 0.01
            dm.set_current_joint_angles(held, "right")
            if before:
                before(dm, index)
            generation = dm.get_teleop_snapshot("right")[1]
            dm.prepare_teleop_baseline("right", generation, held)

        dm = FeedbackSequence([0.0] * loops, before_tick=prepare)
        dm.set_on_change_callback(
            lambda name, value, stamp: events.append((name, value))
        )
        backend = Mock()
        backend.solve.side_effect = lambda pose, reference, **kwargs: (
            reference.copy()
            if candidate is None
            else np.asarray(candidate, dtype=float)
        )
        module = types.ModuleType("arms.agilex_teleop")
        module.AgilexTeleopIK = Mock(return_value=backend)
        solver = IncrementSolver(joint_limits_deg=SOFT, hard_joint_limits_deg=HARD)
        with patch.dict(sys.modules, {"arms.agilex_teleop": module}), patch.object(
            ik_module, "USE_AGILEX_TELEOP", True
        ), patch.object(ik_module.time, "sleep"), patch.object(
            ik_module.time, "monotonic", side_effect=lambda: clock[0]
        ), _capture_piper_logs() as logs:
            ik_module.ik_solver_thread(dm, solver)
        self.assertFalse(dm._shutdown_event.is_set())
        return dm, backend, events, logs.getvalue()

    def test_original_warning_cases_now_preserve_stationary_hold(self):
        for held in (
            [0, 0, 0, 0, 0, 0],
            [0, 3.592, -2.941, 0, 0, 0],
            [0, 4, -2.999, 0, 0, 0],
        ):
            with self.subTest(held=held):
                dm, _, events, logs = self.run_reference(held)
                self.assertTrue(dm.get_teleop_active("right"))
                np.testing.assert_array_equal(dm.published, np.tile(held, (5, 1)))
                event = [
                    v for n, v in events if n == "log_teleop_reengagement_event/right"
                ]
                self.assertEqual(len(event), 1)
                self.assertEqual(event[0][1], 1)
                np.testing.assert_array_equal(event[0][2:8], held)
                self.assertIn("clutch=", logs)
                self.assertIn("excess=", logs)

    def test_inward_motion_is_shaped_without_initial_margin_jump(self):
        dm, backend, _, _ = self.run_reference([0] * 6, [0, 20, -20, 0, 0, 0], loops=20)
        samples = np.array(dm.published)
        np.testing.assert_array_equal(samples[0], np.zeros(6))
        self.assertGreater(samples[-1, 1], 0)
        self.assertLess(samples[1, 1], 3)
        self.assertTrue(np.all(np.diff(samples[:, 1]) >= 0))
        self.assertTrue(np.all(np.diff(samples[:, 2]) <= 0))
        velocity = np.diff(samples, axis=0) / 0.01
        acceleration = np.diff(np.vstack([np.zeros(6), velocity]), axis=0) / 0.01
        self.assertLessEqual(np.max(np.abs(acceleration)), 360 + 1e-6)
        self.assertLessEqual(np.max(np.abs(velocity)), 90 + 1e-6)
        for call in backend.solve.call_args_list:
            reference = call.args[1]
            bounds = call.kwargs["bounds_deg"]
            self.assertGreaterEqual(bounds[1, 0], min(3, reference[1]))
            self.assertLessEqual(bounds[2, 1], max(-3, reference[2]))

    def test_outward_request_does_not_escape_the_hold(self):
        dm, _, _, _ = self.run_reference([0] * 6, [0, -2, 2, 0, 0, 0])
        np.testing.assert_array_equal(dm.published, np.zeros((5, 6)))

    def test_hard_limit_refusal_is_diagnostic_and_sends_nothing(self):
        dm, backend, events, logs = self.run_reference([0, -1, 0, 0, 0, 0], loops=1)
        self.assertFalse(dm.get_teleop_active("right"))
        self.assertEqual(dm.published, [])
        backend.solve.assert_not_called()
        self.assertIn("J2 held=-1.000000", logs)
        self.assertIn("allowed=[0.000000, 180.000000]", logs)
        self.assertEqual([v[1] for n, v in events if "reengagement" in n], [2])

    def test_repeated_clutches_do_not_ratchet_the_hold(self):
        def reengage(dm, index):
            if index in (2, 4):
                dm.set_teleop_state("right", False, None, None)
                dm.set_teleop_state("right", True, np.eye(4), np.eye(4))

        dm, _, _, _ = self.run_reference([0] * 6, loops=6, before=reengage)
        np.testing.assert_array_equal(dm.published, np.zeros((6, 6)))


class RecoveryControllerTests(unittest.TestCase):
    def setUp(self):
        from test_piper_safe_startup import PiperSafeStartupTests, _JointMsg

        self.fixture = PiperSafeStartupTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        cls = self.fixture.piper_controller.PiperController
        self.controller = c = cls.__new__(cls)
        c.cleanup = lambda: None
        c.position_lock = threading.Lock()
        c._command_handoff_lock = threading.RLock()
        c._stop_hold_latched = threading.Event()
        c._stop_hold_lock = threading.RLock()
        c._stop_hold_generation = 0
        c._send_stop_hold = Mock(return_value=True)
        c._startup_feedback_max_age_s = 0.5
        c._sdk_error_sentinel = object()
        c._sdk_call = lambda label, fn: fn()
        c._log = Mock()
        c.piper = Mock()
        c.piper.GetArmJointMsgs.return_value = _JointMsg([0] * 6, timestamp=time.time())
        c.is_robot_enabled = lambda: True
        c.get_control_mode = lambda: c.ControlMode.JOINT_SPACE
        c.get_current_gripper_open_value = lambda: None
        c.JOINT_LIMITS = HARD.copy()
        c.EFFECTIVE_JOINT_LIMITS = SOFT.copy()
        c._target_joint_angles = np.zeros(6)
        c._last_sent_joint_angles = np.zeros(6)
        c._last_sent_joint_mit_mode = 0
        c._last_sent_joint_speed_cap = 10
        c._agilex_teleop_until = 0
        c._agilex_teleop_holding = False
        c._motionctrl_speed_percent = 30
        c._hold_current_position_speed_percent = 10
        c._active_motion_speed_cap_percent = 10
        c.debug_mode = False

    def send(self, q, generation=1):
        self.controller.set_target_joint_angles(
            q, use_agilex_teleop=True, teleop_generation=generation
        )

    def test_process_command_carries_generation_atomically(self):
        from arms.perf import piper_process

        cls = piper_process.PiperControllerProxy
        proxy = cls.__new__(cls)
        proxy._joint_seq = 0.0
        proxy._cmd = Mock()
        proxy.set_target_joint_angles(
            np.zeros(6), use_agilex_teleop=True, teleop_generation=17
        )
        payload = proxy._cmd.write.call_args.args[0]
        self.assertEqual(payload[piper_process._C_TELEOP_GENERATION], 17)
        self.assertEqual(payload[piper_process._C_AGILEX_TELEOP], 1)
        proxy.set_target_joint_angles(np.zeros(6))
        self.assertEqual(
            proxy._cmd.write.call_args.args[0][piper_process._C_TELEOP_GENERATION], -1
        )


if __name__ == "__main__":
    unittest.main()
