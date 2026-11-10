"""Offline motion-shaping and teleop integration regressions."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from arms.joint_motion import JointAccelerationLimiter, limit_tracking_step


class JointMotionTests(unittest.TestCase):
    def test_tracking_limit_scales_the_whole_step(self):
        limits = np.arange(1., 7.)
        for joint in range(6):
            for sign in (-1., 1.):
                with self.subTest(joint=joint, sign=sign):
                    reference = np.arange(6, dtype=float)
                    step = np.full(6, .1)
                    step[joint] = 2 * sign * limits[joint]
                    target, scale, binding = limit_tracking_step(reference, reference + step, reference, limits)
                    self.assertEqual(scale, .5)
                    self.assertEqual(binding, joint)
                    np.testing.assert_allclose(target, reference + .5 * step)

    def test_tracking_limit_moves_with_actual_joint(self):
        reference = np.full(6, 40.)
        target, _, _ = limit_tracking_step(reference, np.full(6, 70.), reference, np.full(6, 15.))
        np.testing.assert_allclose(target, 55.)
        target, _, _ = limit_tracking_step(target, np.full(6, 70.), np.full(6, 45.), np.full(6, 15.))
        np.testing.assert_allclose(target, 60.)
        inward, scale, _ = limit_tracking_step(target, target - 1., np.full(6, 45.), np.full(6, 15.))
        np.testing.assert_allclose(inward, 59.)
        self.assertEqual(scale, 1.)

    def test_feedback_drift_allows_only_inward_recovery_without_a_jump(self):
        reference = np.full(6, 16.)
        actual = np.zeros(6)
        limits = np.full(6, 15.)
        outward, scale, _ = limit_tracking_step(reference, reference + 1., actual, limits)
        np.testing.assert_array_equal(outward, reference)
        self.assertEqual(scale, 0.)
        inward, scale, _ = limit_tracking_step(reference, reference - .1, actual, limits)
        np.testing.assert_array_equal(inward, reference - .1)
        self.assertEqual(scale, 1.)

    def test_tracking_limit_rejects_invalid_inputs(self):
        for bad in (float("nan"), float("inf")):
            for index in range(4):
                args = [np.zeros(6), np.ones(6), np.zeros(6), np.full(6, 15.)]
                args[index][0] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    limit_tracking_step(*args)
        with self.assertRaises(ValueError):
            limit_tracking_step(np.zeros(6), np.ones(6), np.zeros(6), np.zeros(6))

    def test_tracking_steps_cannot_worsen_an_outside_reference(self):
        rng = np.random.default_rng(2026)
        for _ in range(1000):
            actual = rng.uniform(-90., 90., 6)
            limits = rng.uniform(1., 30., 6)
            reference = actual + rng.uniform(-1.8, 1.8, 6) * limits
            requested = rng.uniform(-180., 180., 6)
            accepted, scale, _ = limit_tracking_step(reference, requested, actual, limits)
            self.assertGreaterEqual(scale, 0.)
            self.assertLessEqual(scale, 1.)
            allowed = np.maximum(limits, np.abs(reference - actual))
            self.assertTrue(np.all(np.abs(accepted - actual) <= allowed + 1e-12))

    def test_velocity_and_acceleration_bounds_with_reversals_and_variable_dt(self):
        limiter = JointAccelerationLimiter([45] * 6, [180] * 6)
        limiter.reset(np.zeros(6))
        previous_v = np.zeros(6)
        for i in range(1200):
            dt = [.004, .008, .012][i % 3]
            target = np.full(6, 20. if i < 300 else -10.)
            previous_q = limiter.position.copy()
            q = limiter.step(target, dt)
            v = (q - previous_q) / dt
            self.assertLessEqual(np.max(np.abs(v)), 45 + 1e-9)
            self.assertLessEqual(np.max(np.abs(v - previous_v)) / dt, 180 + 1e-8)
            previous_v = v
        np.testing.assert_allclose(q, -10., atol=1e-6)

    def test_fixed_target_settles_without_overshoot_from_rest(self):
        limiter = JointAccelerationLimiter([180] * 6, [360] * 6)
        limiter.reset(np.zeros(6))
        trajectory = np.array([limiter.step(np.ones(6), .008) for _ in range(300)])
        self.assertTrue(np.all(trajectory <= 1. + 1e-12))
        self.assertTrue(np.all(np.diff(trajectory, axis=0) >= -1e-12))
        np.testing.assert_allclose(trajectory[-1], 1., atol=1e-8)

    def test_reset_removes_old_momentum(self):
        limiter = JointAccelerationLimiter([180] * 6, [360] * 6)
        limiter.reset(np.zeros(6))
        for _ in range(10):
            limiter.step(np.ones(6) * 20, .008)
        limiter.reset(np.ones(6) * 5)
        np.testing.assert_array_equal(limiter.step(np.ones(6) * 5, .008), 5.)

    def test_invalid_inputs_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                JointAccelerationLimiter([180] * 6, [value] * 6)

    def test_ik_integration_ramps_and_reanchors(self):
        from test_teleop_reference import FeedbackSequence, IncrementSolver
        from test_ik_solver_thread import _capture_piper_logs
        from arms.threads import ik_solver as module

        def reanchor(dm, index):
            if index == 10:
                dm.set_teleop_state('right', True, np.eye(4), np.eye(4))

        dm = FeedbackSequence([0.] * 20, before_tick=reanchor)
        solver = IncrementSolver(step=1.)
        with _capture_piper_logs(), patch.object(module.time, 'sleep'), patch.object(
            module.time, 'monotonic', side_effect=np.arange(20) * .008
        ), patch.object(module, 'TELEOP_ACCEL_LIMIT_ENABLED', True):
            module.ik_solver_thread(dm, solver)
        q = np.array(dm.published)[:, 3]
        self.assertEqual(len(q), 20)
        self.assertAlmostEqual(q[0], 360 * .008**2)
        self.assertAlmostEqual(q[10], q[0])
        v = np.diff(np.r_[0., q[:10]]) / .008
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0., v]))) / .008, 360 + 1e-8)
