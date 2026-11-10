"""Regression tests for feedback-independent teleop reference integration."""

import unittest
from unittest.mock import patch

import numpy as np

from test_ik_solver_thread import (
    FakeIKSolver,
    _LoopLimitedDataManager,
    _capture_piper_logs,
    _pose,
)
from arms.data_manager import DataManager, RobotActivityState
from arms.threads import ik_solver as ik_module


class FeedbackSequence(_LoopLimitedDataManager):
    def __init__(self, wrist_values, before_tick=None):
        super().__init__(len(wrist_values))
        self.wrist_values = wrist_values
        self.before_tick = before_tick
        self.published = []
        self.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        self.set_controller_data("right", np.eye(4), 1.0, 1.0)
        self.set_teleop_state("right", True, np.eye(4), np.eye(4))

    def is_shutdown_requested(self):
        stop = super().is_shutdown_requested()
        if not stop:
            index = self._loop_count - 1
            joints = np.zeros(6)
            joints[3] = self.wrist_values[index]
            self.set_current_joint_angles(joints, "right")
            if self.before_tick is not None:
                self.before_tick(self, index)
        return stop

    def set_target_joint_angles(self, values, hand=None):
        super().set_target_joint_angles(values, hand)
        self.published.append(np.asarray(values).copy())


class IncrementSolver(FakeIKSolver):
    """Integrate a deterministic small desired wrist velocity."""

    def __init__(self, step=.02, after_solve=None, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.inputs = []
        self.after_solve = after_solve

    def solve_ik(self):
        self.inputs.append(np.degrees(self._config).copy())
        self._config[3] += np.radians(self.step)
        self.solve_ik_calls += 1
        if self.after_solve is not None:
            self.after_solve()
        return True

    def get_current_end_effector_pose(self):
        # Lets the test distinguish measured FK from commanded FK.
        return _pose([np.degrees(self._config[3]), 0, 0])


class TeleopReferenceTests(unittest.TestCase):
    def run_loop(self, dm, solver):
        with _capture_piper_logs(), patch.object(ik_module.time, "sleep"), patch.object(
            ik_module, "TELEOP_ACCEL_LIMIT_ENABLED", False
        ):
            ik_module.ik_solver_thread(dm, solver)
        self.assertFalse(DataManager.is_shutdown_requested(dm))

    def test_feedback_zero_toggle_does_not_reverse_the_desired_trajectory(self):
        dm = FeedbackSequence([0, .3, 0, .3])
        solver = IncrementSolver()
        self.run_loop(dm, solver)
        np.testing.assert_allclose(np.array(solver.inputs)[:, 3], [0, .02, .04, .06])
        np.testing.assert_allclose(np.array(dm.published)[:, 3], [.02, .04, .06, .08])
        self.assertAlmostEqual(dm.get_current_end_effector_pose("right")[0, 3], .3)

    def test_slew_limited_result_is_the_next_integration_reference(self):
        dm = FeedbackSequence([0, 0, 0])
        solver = IncrementSolver(step=1)
        with patch.object(ik_module, "MAX_JOINT_SLEW_DEG_S", [1.] * 6):
            self.run_loop(dm, solver)
        for i in (1, 2):
            np.testing.assert_allclose(solver.inputs[i], dm.published[i - 1])
        self.assertLess(dm.published[0][3], .02)

    def test_joint_limited_result_is_the_next_integration_reference(self):
        limits = np.tile([-180., 180.], (6, 1))
        limits[3, 1] = .03
        dm = FeedbackSequence([0, 0, 0])
        solver = IncrementSolver(step=.1, joint_limits_deg=limits)
        self.run_loop(dm, solver)
        np.testing.assert_allclose(np.array(solver.inputs)[:, 3], [0, .03, .03])
        np.testing.assert_allclose(np.array(dm.published)[:, 3], [.03, .03, .03])

    def test_identical_anchor_reengagement_starts_from_new_measured_pose(self):
        def reengage(dm, index):
            if index == 2:
                dm.set_teleop_state("right", False, None, None)
                dm.set_teleop_state("right", True, np.eye(4), np.eye(4))

        dm = FeedbackSequence([0, 0, 2], reengage)
        solver = IncrementSolver()
        self.run_loop(dm, solver)
        self.assertAlmostEqual(solver.inputs[2][3], 2.)
        self.assertAlmostEqual(dm.published[-1][3], 2.02)

    def test_stalled_arm_cannot_accumulate_an_unbounded_target(self):
        limit = ik_module.TELEOP_MAX_TRACKING_ERROR_DEG[3]
        dm = FeedbackSequence([0] * (int(limit) + 7))
        solver = IncrementSolver(step=1.)
        # Use nominal elapsed ticks so the existing slew guard permits the
        # requested increment and this specifically exercises tracking error.
        clock = iter(np.arange(0., 100., .008))
        with patch.object(ik_module.time, "monotonic", side_effect=lambda: next(clock)):
            self.run_loop(dm, solver)
        self.assertTrue(dm.get_teleop_active("right"))
        self.assertLessEqual(max(q[3] for q in dm.published), limit)
        self.assertAlmostEqual(dm.get_target_joint_angles("right")[3], limit)

    def test_large_feedback_disagreement_deactivates_before_another_solve(self):
        dm = FeedbackSequence([0, 40])
        solver = IncrementSolver()
        self.run_loop(dm, solver)
        self.assertEqual(solver.solve_ik_calls, 1)
        self.assertFalse(dm.get_teleop_active("right"))
        self.assertAlmostEqual(dm.get_target_joint_angles("right")[3], 40.)

    def test_each_joint_uses_its_own_tracking_limit(self):
        limits = np.array([2., 4., 6., 8., 10., 12.])
        for joint in range(6):
            with self.subTest(joint=joint):
                measured = np.arange(6, dtype=float)
                target = measured.copy()
                target[joint] += limits[joint]
                self.assertIsNone(ik_module._tracking_error_violation(target, measured, limits))
                target[joint] += .25
                self.assertEqual(
                    ik_module._tracking_error_violation(target, measured, limits),
                    (joint, limits[joint] + .25, limits[joint]),
                )
        # The binding joint is relative to its own limit, not the largest angle.
        self.assertEqual(
            ik_module._tracking_error_violation(
                np.array([3., 0., 0., 0., 0., 13.]), np.zeros(6), limits
            ),
            (0, 3., 2.),
        )

    def test_configured_tracking_limit_does_not_end_teleop(self):
        limits = [15., 15., 15., 1., 15., 15.]
        # Moderate feedback movement outside the envelope does not snap or exit.
        dm = FeedbackSequence([0, 2])
        solver = IncrementSolver()
        with patch.object(ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", limits):
            self.run_loop(dm, solver)
        self.assertEqual(solver.solve_ik_calls, 2)
        self.assertTrue(dm.get_teleop_active("right"))
        self.assertLess(dm.get_target_joint_angles("right")[3], .1)
        # A proposed target is limited before it can be published.
        dm = FeedbackSequence([0])
        clock = iter(np.arange(0., 100., .008))
        with patch.object(ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", limits), patch.object(
            ik_module.time, "monotonic", side_effect=lambda: next(clock)
        ):
            self.run_loop(dm, IncrementSolver(step=1.1))
        self.assertTrue(dm.get_teleop_active("right"))
        self.assertAlmostEqual(dm.get_target_joint_angles("right")[3], 1.)

    def test_tracking_limit_resumes_as_feedback_catches_up(self):
        dm = FeedbackSequence([0.] * 5 + [1., 2., 3.])
        generation = dm.get_teleop_snapshot("right")[1]
        clock = iter(np.arange(0., 100., .008))
        with patch.object(ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", [2.] * 6), patch.object(
            ik_module.time, "monotonic", side_effect=lambda: next(clock)
        ):
            self.run_loop(dm, IncrementSolver(step=1.))
        np.testing.assert_allclose(np.array(dm.published)[:, 3], [1., 2., 2., 2., 2., 3., 4., 5.])
        self.assertEqual(dm.get_teleop_snapshot("right")[1], generation)
        self.assertTrue(dm.get_teleop_active("right"))

    def test_acceleration_brakes_toward_tracking_limit(self):
        dm = FeedbackSequence([0.] * 400)
        clock = iter(np.arange(0., 100., .008))
        with _capture_piper_logs(), patch.object(ik_module.time, "sleep"), patch.object(
            ik_module, "TELEOP_ACCEL_LIMIT_ENABLED", True
        ), patch.object(ik_module.time, "monotonic", side_effect=lambda: next(clock)), patch.object(
            ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", [2.] * 6
        ):
            ik_module.ik_solver_thread(dm, IncrementSolver(step=10.))
        trajectory = np.array(dm.published)[:, 3]
        self.assertTrue(np.all(trajectory <= 2. + 1e-12))
        self.assertAlmostEqual(trajectory[-1], 2.)
        velocities = np.diff(np.r_[0., trajectory]) / .008
        acceleration = np.diff(np.r_[0., velocities]) / .008
        self.assertLessEqual(np.max(np.abs(acceleration)), ik_module.TELEOP_MAX_JOINT_ACCEL_DEG_S2[3] + 1e-7)
        self.assertTrue(dm.get_teleop_active("right"))

    def test_tracking_boundary_contains_filter_momentum_and_resumes(self):
        def move_feedback_boundary(dm, index):
            if index == 12:
                measured = np.zeros(6)
                measured[3] = dm.get_target_joint_angles("right")[3] - 2.1
                dm.set_current_joint_angles(measured, "right")

        dm = FeedbackSequence([0.] * 30, before_tick=move_feedback_boundary)
        clock = iter(np.arange(0., 100., .008))
        with _capture_piper_logs(), patch.object(ik_module.time, "sleep"), patch.object(
            ik_module, "TELEOP_ACCEL_LIMIT_ENABLED", True
        ), patch.object(ik_module.time, "monotonic", side_effect=lambda: next(clock)), patch.object(
            ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", [2.] * 6
        ):
            ik_module.ik_solver_thread(dm, IncrementSolver(step=10.))
        q = np.array(dm.published)[:, 3]
        self.assertGreater(q[11] - q[10], 0.)
        self.assertEqual(q[12], q[11])
        self.assertAlmostEqual(q[13] - q[12], ik_module.TELEOP_MAX_JOINT_ACCEL_DEG_S2[3] * .008**2)
        self.assertTrue(dm.get_teleop_active("right"))

    def test_preset_takeover_during_solve_preserves_the_preset_target(self):
        dm = FeedbackSequence([0])

        def takeover():
            dm.set_robot_activity_state(RobotActivityState.PRESET_CONTROLLED, "right")
            dm.set_teleop_state("right", False, None, None)
            dm.set_target_joint_angles(np.ones(6), "right")

        self.run_loop(dm, IncrementSolver(after_solve=takeover))
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), np.ones(6))
        self.assertEqual(len(dm.published), 1)

    def test_release_and_reengagement_during_solve_discards_old_result(self):
        dm = FeedbackSequence([0])

        def reengage():
            dm.set_teleop_state("right", False, None, None)
            dm.set_teleop_state("right", True, np.eye(4), np.eye(4))

        self.run_loop(dm, IncrementSolver(after_solve=reengage))
        self.assertEqual(dm.published, [])

    def test_paused_arm_does_not_integrate_a_lingering_active_flag(self):
        dm = FeedbackSequence([0])
        dm.set_robot_activity_state(RobotActivityState.PAUSED, "right")
        solver = IncrementSolver()
        self.run_loop(dm, solver)
        self.assertEqual(solver.solve_ik_calls, 0)

    def test_nonfinite_solution_is_not_published(self):
        dm = FeedbackSequence([0])
        self.run_loop(dm, IncrementSolver(step=float("nan")))
        self.assertFalse(dm.get_teleop_active("right"))
        self.assertTrue(all(np.all(np.isfinite(q)) for q in dm.published))

    def test_rejected_controller_jump_keeps_the_accepted_reference(self):
        dm = FeedbackSequence([0, .3])
        solver = IncrementSolver()
        # Inject at the filtered-pose boundary consumed by IK, so the input
        # filter's wall-clock timing cannot mask the jump in this unit test.
        with patch.object(dm, "get_controller_data", side_effect=[
            (np.eye(4), 1., 1.), (_pose([10, 0, 0]), 1., 1.),
        ]):
            self.run_loop(dm, solver)
        self.assertEqual(solver.solve_ik_calls, 1)
        self.assertEqual(len(dm.published), 1)
        self.assertAlmostEqual(dm.published[0][3], .02)


class AgilexSelectionTests(unittest.TestCase):
    def test_agilex_tracking_limit_keeps_clutch_and_ready_target(self):
        import sys
        import types
        from unittest.mock import Mock

        backend = Mock()
        backend.solve.return_value = np.full(6, 100.)
        module = types.ModuleType("arms.agilex_teleop")
        module.AgilexTeleopIK = Mock(return_value=backend)
        dm = FeedbackSequence([0.] * 200)
        generation = dm.get_teleop_snapshot("right")[1]
        dm.prepare_teleop_baseline("right", generation, np.zeros(6))
        clock = iter(np.arange(0., 100., .002))
        with patch.dict(sys.modules, {"arms.agilex_teleop": module}), patch.object(
            ik_module, "USE_AGILEX_TELEOP", True
        ), patch.object(ik_module, "TELEOP_MAX_TRACKING_ERROR_DEG", [.2] * 6), patch.object(
            ik_module.time, "monotonic", side_effect=lambda: next(clock)
        ), patch.object(ik_module.time, "sleep"), _capture_piper_logs():
            ik_module.ik_solver_thread(dm, IncrementSolver())
        self.assertEqual(len(dm.published), 200)
        self.assertTrue(np.all(np.array(dm.published) <= .2 + 1e-12))
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), .2, atol=1e-6)
        self.assertTrue(dm.get_teleop_active("right"))
        self.assertEqual(dm.get_teleop_snapshot("right")[1], generation)
        self.assertIsNotNone(dm.get_ready_teleop_target("right"))

    def test_alternate_solver_uses_acknowledged_hold_and_keeps_output_guards(self):
        import sys
        import types
        from unittest.mock import Mock

        backend = Mock()
        backend.solve.return_value = np.full(6, 100.)
        module = types.ModuleType('arms.agilex_teleop')
        module.AgilexTeleopIK = Mock(return_value=backend)
        dm = FeedbackSequence([0., 0.])
        generation = dm.get_teleop_snapshot('right')[1]
        dm.prepare_teleop_baseline('right', generation, np.zeros(6))
        solver = IncrementSolver()
        with patch.dict(sys.modules, {'arms.agilex_teleop': module}), patch.object(
            ik_module, 'USE_AGILEX_TELEOP', True
        ), patch.object(ik_module, 'TELEOP_ACCEL_LIMIT_ENABLED', False), patch.object(
            ik_module, 'TELEOP_JOINT_SLEW_CLAMP_ENABLED', False
        ), patch.object(ik_module.time, 'sleep'), _capture_piper_logs():
            ik_module.ik_solver_thread(dm, solver)
        self.assertEqual(solver.solve_ik_calls, 0)
        np.testing.assert_array_equal(backend.solve.call_args.args[1], np.zeros(6))
        self.assertEqual(len(dm.published), 2)
        np.testing.assert_array_equal(dm.published[0], np.zeros(6))
        self.assertLessEqual(np.max(dm.published[1]), 15.)

    def test_old_target_is_hidden_until_current_clutch_is_seeded(self):
        dm = DataManager()
        dm.set_target_joint_angles(np.ones(6) * 12, 'right')
        dm.set_teleop_state('right', True, np.eye(4), np.eye(4))
        _, generation, _, _ = dm.get_teleop_snapshot('right')
        self.assertIsNone(dm.get_ready_teleop_target('right'))
        self.assertTrue(dm.commit_teleop_target('right', generation, np.ones(6)))
        np.testing.assert_array_equal(dm.get_ready_teleop_target('right'), np.ones(6))
        dm.set_teleop_state('right', False, None, None)
        dm.set_teleop_state('right', True, np.eye(4), np.eye(4))
        self.assertIsNone(dm.get_ready_teleop_target('right'))
        self.assertFalse(dm.commit_teleop_target('right', generation, np.ones(6) * 20))

    def test_tracking_offset_does_not_change_reengagement_target(self):
        import sys
        import types
        from unittest.mock import Mock
        backend = Mock()
        backend.solve.return_value = np.zeros(6)
        module = types.ModuleType('arms.agilex_teleop')
        module.AgilexTeleopIK = Mock(return_value=backend)
        def prepare(dm, index):
            if index == 2:
                dm.set_teleop_state('right', True, np.eye(4), np.eye(4))
            generation = dm.get_teleop_snapshot('right')[1]
            dm.prepare_teleop_baseline('right', generation, np.zeros(6))
        dm = FeedbackSequence([1.5] * 4, before_tick=prepare)
        solver = IncrementSolver()
        with patch.dict(sys.modules, {'arms.agilex_teleop': module}), patch.object(
            ik_module, 'USE_AGILEX_TELEOP', True
        ), patch.object(ik_module.time, 'sleep'), _capture_piper_logs():
            ik_module.ik_solver_thread(dm, solver)
        self.assertEqual(len(dm.published), 4)
        np.testing.assert_array_equal(np.array(dm.published), np.zeros((4, 6)))
        for call in backend.solve.call_args_list:
            np.testing.assert_array_equal(call.args[1], np.zeros(6))

    def test_clutch_baseline_ack_cannot_cross_generations(self):
        dm = DataManager()
        dm.set_teleop_state('right', True, np.eye(4), np.eye(4))
        generation = dm.get_teleop_snapshot('right')[1]
        self.assertIsNone(dm.get_teleop_baseline('right', generation))
        self.assertTrue(dm.prepare_teleop_baseline('right', generation, np.zeros(6)))
        dm.set_teleop_state('right', True, np.eye(4), np.eye(4))
        self.assertFalse(dm.prepare_teleop_baseline('right', generation, np.ones(6)))
        self.assertIsNone(dm.get_teleop_baseline('right', generation + 1))

    def test_joint_sender_holds_and_waits_instead_of_sending_old_target(self):
        from arms.threads import joint_state as module
        from test_ik_solver_thread import JointStateClampTests
        class Controller(JointStateClampTests.FakeController):
            holds = 0
            sends = 0
            def hold_teleop_position(self):
                self.holds += 1
                return np.ones(6) * 4

            def prepare_teleop(self, generation):
                return self.hold_teleop_position()

            def set_target_joint_angles(self, *args, **kwargs):
                self.sends += 1
        controller = Controller()
        def release(dm, index):
            if index == 1:
                dm.set_teleop_state('right', False, None, None)
        dm = FeedbackSequence([0., 0.], before_tick=release)
        dm.set_target_joint_angles(np.ones(6) * 99, 'right')
        with patch.object(module, 'USE_AGILEX_TELEOP', True), patch.object(module.time, 'sleep'), _capture_piper_logs():
            module.joint_state_thread(dm, controller)
        self.assertEqual(controller.holds, 2)
        self.assertEqual(controller.sends, 0)
        self.assertFalse(DataManager.is_shutdown_requested(dm))

    def test_process_target_flag_resets_with_next_normal_target(self):
        from arms.perf import piper_process
        from unittest.mock import Mock

        # Exercise shared-memory payload construction without creating a process.
        cls = next(value for value in vars(piper_process).values()
                   if isinstance(value, type) and value.__module__ == piper_process.__name__
                   and hasattr(value, 'set_target_joint_angles'))
        proxy = cls.__new__(cls)
        proxy._joint_seq = 0.
        proxy._cmd = Mock()
        proxy.set_target_joint_angles(np.zeros(6), use_agilex_teleop=True)
        self.assertEqual(proxy._cmd.write.call_args.args[0][piper_process._C_AGILEX_TELEOP], 1.)
        proxy.set_target_joint_angles(np.zeros(6))
        self.assertEqual(proxy._cmd.write.call_args.args[0][piper_process._C_AGILEX_TELEOP], 0.)


if __name__ == "__main__":
    unittest.main()
