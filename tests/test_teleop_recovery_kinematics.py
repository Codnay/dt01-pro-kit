"""Real Pinocchio/SciPy tests of held-pose recovery; never import CAN hardware."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

try:
    from arms.pink_ik_solver import PinkIKSolver
    from arms.agilex_teleop import AgilexTeleopIK
except ImportError:
    PinkIKSolver = None

from arms.joint_limits import recovery_joint_bounds  # noqa: E402


@unittest.skipIf(PinkIKSolver is None, "Requires Pinocchio, Pink and SciPy")
class RecoveryKinematicsTests(unittest.TestCase):
    def setUp(self):
        hard = np.degrees(
            np.array(
                [
                    [-2.6179, 2.6179],
                    [0, 3.14],
                    [-2.967, 0],
                    [-1.745, 1.745],
                    [-1.22, 1.22],
                    [-2.09439, 2.09439],
                ]
            )
        )
        requested = hard + np.array([[3.0, -3.0]] * 5 + [[20.0, -20.0]])
        self.solver = PinkIKSolver(
            urdf_path=str(ROOT / "piper_description/urdf/piper_description.urdf"),
            end_effector_frame="gripper_center",
            joint_position_limits_deg=requested,
        )
        self.soft = self.solver.get_joint_position_limits_deg()
        self.hard = self.solver.get_joint_position_limits_deg(hard=True)
        self.ik = AgilexTeleopIK(self.solver)

    def test_stationary_pose_at_soft_margin_has_no_solver_drift(self):
        for held in (
            np.zeros(6),
            np.array([0, 3.592, -2.941, 0, 0, 0]),
            np.array([0, 4, -2.999, 0, 0, 119]),
        ):
            with self.subTest(held=held):
                pose = self.solver.get_pose_for_configuration_deg(held)
                reference = held.copy()
                for _ in range(50):
                    result = self.ik.solve(
                        pose,
                        reference,
                        bounds_deg=recovery_joint_bounds(
                            reference, self.soft, self.hard
                        ),
                    )
                    np.testing.assert_allclose(result, held, atol=1e-12, rtol=0)
                    reference = result

    def test_recovery_solver_can_move_inward_from_home(self):
        held = np.zeros(6)
        desired = np.array([0, 1, -1, 0, 0, 0.0])
        pose = self.solver.get_pose_for_configuration_deg(desired)
        result = self.ik.solve(
            pose, held, bounds_deg=recovery_joint_bounds(held, self.soft, self.hard)
        )
        self.assertIsNotNone(result)
        self.assertGreater(result[1], 0)
        self.assertLess(result[2], 0)
        before = np.linalg.norm(
            self.solver.get_pose_for_configuration_deg(held)[:3, 3] - pose[:3, 3]
        )
        after = np.linalg.norm(
            self.solver.get_pose_for_configuration_deg(result)[:3, 3] - pose[:3, 3]
        )
        self.assertLess(after, before * 0.8)

    def test_held_fk_is_exact_and_normal_solver_limits_remain_unchanged(self):
        held = np.zeros(6)
        actual_pose = self.solver.get_pose_for_configuration_deg(held)
        self.solver.set_configuration_no_task_update(np.radians(held))
        clipped_pose = self.solver.get_current_end_effector_pose()
        self.assertGreater(np.linalg.norm(actual_pose - clipped_pose), 0.001)
        np.testing.assert_array_equal(
            self.solver.get_joint_position_limits_deg(), self.soft
        )
        np.testing.assert_array_equal(
            np.degrees(self.solver.urdf_model.lowerPositionLimit), self.soft[:, 0]
        )


if __name__ == "__main__":
    unittest.main()
