"""Unit tests for the IK solver thread (arms/threads/ik_solver.py).

The thread runs an infinite control loop, so the tests drive it deterministically
by subclassing ``DataManager`` to allow a bounded number of loop iterations.

``scipy`` and ``pink_ik_solver`` are not installable in the test environment
(they pull in heavy robotics/native dependencies), so both are stubbed in
``sys.modules`` before importing the module under test. The stub ``Rotation``
implements just enough real math (Euler/axis-angle conversions) to keep the
frame-correction and target-transform assertions meaningful.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

# Stub out scipy (Rotation) and pink_ik_solver before importing the module.
class _StubRotation:
    """Minimal but faithful drop-in for ``scipy.spatial.transform.Rotation``."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "_StubRotation":
        return cls(matrix)

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False) -> "_StubRotation":
        values = np.asarray(angles, dtype=np.float64).reshape(-1)
        if degrees:
            values = np.radians(values)

        def _axis(axis: str, angle: float) -> np.ndarray:
            c, s = np.cos(angle), np.sin(angle)
            if axis == "x":
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
            if axis == "y":
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        matrix = np.eye(3, dtype=np.float64)
        for axis, angle in zip(seq.lower(), values):
            matrix = matrix @ _axis(axis, angle)
        return cls(matrix)

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "_StubRotation":
        vec = np.asarray(rotvec, dtype=np.float64).reshape(3)
        theta = float(np.linalg.norm(vec))
        if theta < 1e-12:
            return cls(np.eye(3))
        k = vec / theta
        skew = np.array(
            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
            dtype=np.float64,
        )
        matrix = (
            np.eye(3)
            + np.sin(theta) * skew
            + (1.0 - np.cos(theta)) * (skew @ skew)
        )
        return cls(matrix)

    @classmethod
    def from_quat(cls, _quat) -> "_StubRotation":
        return cls(np.eye(3))

    def as_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def as_rotvec(self) -> np.ndarray:
        matrix = self._matrix
        angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
        if angle < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axis = np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        )
        return axis / (2.0 * np.sin(angle)) * angle

    def as_quat(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _rotation_is_adequate() -> bool:
    """True only if a real (or sufficiently rich) Rotation is importable.

    Another test module may have already installed a *minimal* scipy stub whose
    Rotation lacks ``from_euler``/``from_rotvec``; that is not good enough here.
    """
    try:
        from scipy.spatial.transform import Rotation as _R  # noqa: F401
    except Exception:
        return False
    return all(hasattr(_R, attr) for attr in ("from_euler", "from_rotvec", "as_rotvec"))


if not _rotation_is_adequate():
    scipy_mod = sys.modules.get("scipy") or types.ModuleType("scipy")
    spatial_mod = sys.modules.get("scipy.spatial") or types.ModuleType("scipy.spatial")
    transform_mod = sys.modules.get("scipy.spatial.transform") or types.ModuleType(
        "scipy.spatial.transform"
    )
    # Force the richer Rotation even if another test installed a minimal stub.
    transform_mod.Rotation = _StubRotation
    spatial_mod.transform = transform_mod
    scipy_mod.spatial = spatial_mod
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.spatial"] = spatial_mod
    sys.modules["scipy.spatial.transform"] = transform_mod

if "arms.pink_ik_solver" not in sys.modules:
    pink_mod = types.ModuleType("arms.pink_ik_solver")

    class _PinkIKSolverStub:  # noqa: D401 - placeholder for type annotation only
        """Placeholder so the module-level import/annotation resolves."""

    pink_mod.PinkIKSolver = _PinkIKSolverStub
    sys.modules["arms.pink_ik_solver"] = pink_mod


from arms.configs import (  # noqa: E402
    IK_SOLVER_RATE,
    J6_JOYSTICK_SPEED,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from arms.data_manager import DataManager, RobotActivityState  # noqa: E402
from arms.threads.ik_solver import (  # noqa: E402
    _select_continuous_rotvec,
    ik_solver_thread,
)

# If another test imported common.* first, those modules may have bound a
# minimal stub ``Rotation`` (via ``from scipy... import Rotation``). Rebind the
# richer stub in every already-loaded module that needs the extra methods.
if not _rotation_is_adequate():
    for _mod_name, _mod in list(sys.modules.items()):
        if not _mod_name.startswith("common"):
            continue
        _rot = getattr(_mod, "Rotation", None)
        if _rot is not None and not hasattr(_rot, "from_rotvec"):
            _mod.Rotation = _StubRotation


_DT = 1.0 / IK_SOLVER_RATE


@contextlib.contextmanager
def _capture_piper_logs(level: int = logging.DEBUG):
    """Capture ``piper`` logger output as text.

    Thread status/diagnostic messages were migrated from ``print`` to structured
    logging, so these tests assert on the logger stream instead of stdout.
    Temporarily swaps the ``piper`` logger's handlers for a buffer handler
    (children propagate up to it) and lowers the level so DEBUG dumps are
    captured too. Also isolates the test from the process-wide stdout handler.
    """
    logger = logging.getLogger("piper")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(logging.Formatter("%(message)s"))
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    for h in saved_handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield buf
    finally:
        logger.removeHandler(handler)
        for h in saved_handlers:
            logger.addHandler(h)
        logger.setLevel(saved_level)


class _LoopLimitedDataManager(DataManager):
    """DataManager that stops the IK loop after a fixed number of iterations.

    ``ik_solver_thread`` calls ``is_shutdown_requested`` once per loop check, so
    returning ``True`` after ``max_iterations`` checks bounds the run.
    """

    def __init__(self, max_iterations: int = 1) -> None:
        super().__init__()
        self._max_iterations = int(max_iterations)
        self._loop_count = 0

    def is_shutdown_requested(self) -> bool:
        self._loop_count += 1
        if self._loop_count > self._max_iterations:
            return True
        return super().is_shutdown_requested()


class _ControllerSequenceDataManager(_LoopLimitedDataManager):
    """DataManager that returns a deterministic controller transform sequence."""

    def __init__(self, controller_transforms: list[np.ndarray]) -> None:
        super().__init__(max_iterations=len(controller_transforms))
        self._controller_transforms = [
            np.asarray(transform, dtype=np.float64).copy()
            for transform in controller_transforms
        ]
        self._controller_data_calls = 0

    def get_controller_data(
        self, hand: str | None = None
    ) -> tuple[np.ndarray | None, float, float]:
        index = min(self._controller_data_calls, len(self._controller_transforms) - 1)
        self._controller_data_calls += 1
        return self._controller_transforms[index].copy(), 1.0, 1.0


class _GripSequenceDataManager(_LoopLimitedDataManager):
    """DataManager that returns deterministic Quest grip values."""

    def __init__(self, grip_values: list[float]) -> None:
        super().__init__(max_iterations=len(grip_values))
        self._grip_values = [float(value) for value in grip_values]
        self._controller_data_calls = 0

    def get_controller_data(
        self, hand: str | None = None
    ) -> tuple[np.ndarray | None, float, float]:
        index = min(self._controller_data_calls, len(self._grip_values) - 1)
        self._controller_data_calls += 1
        return _pose([0.0, 0.0, 0.0]), self._grip_values[index], 1.0


class FakeIKSolver:
    """Duck-typed stand-in for ``PinkIKSolver`` recording all interactions."""

    def __init__(
        self,
        *,
        solve_success: bool = True,
        config_deg: np.ndarray | None = None,
        solved_config_deg: np.ndarray | None = None,
        last_solve_time_ms: float = 1.23,
        ee_pose: np.ndarray | None = None,
        raise_on_ee_pose: bool = False,
        joint_limits_deg: np.ndarray | None = None,
        hard_joint_limits_deg: np.ndarray | None = None,
    ) -> None:
        start = np.zeros(6, dtype=np.float64) if config_deg is None else np.asarray(
            config_deg, dtype=np.float64
        )
        self._config = np.radians(start)
        self.hard_joint_limits_deg = hard_joint_limits_deg
        self.solve_success = solve_success
        self.solved_config_deg = solved_config_deg
        self.last_solve_time_ms = last_solve_time_ms
        self.ee_pose = np.eye(4) if ee_pose is None else np.asarray(ee_pose, np.float64)
        self.raise_on_ee_pose = raise_on_ee_pose
        self.joint_limits_deg = (
            None
            if joint_limits_deg is None
            else np.asarray(joint_limits_deg, dtype=np.float64)
        )

        self.set_config_calls: list[np.ndarray] = []
        self.set_config_no_task_calls: list[np.ndarray] = []
        self.set_target_pose_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.solve_ik_calls = 0

    def get_joint_position_limits_deg(self, *, hard=False):
        limits = self.hard_joint_limits_deg if hard else self.joint_limits_deg
        return np.tile([-180.0, 180.0], (6, 1)) if limits is None else limits.copy()

    def get_pose_for_configuration_deg(self, joints):
        previous = self._config.copy()
        self._config = np.radians(joints)
        try:
            return self.get_current_end_effector_pose()
        finally:
            self._config = previous

    def clamp_joint_angles_deg(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        angles = np.asarray(joint_angles_deg, dtype=np.float64)
        if self.joint_limits_deg is None:
            return angles.copy()
        return np.clip(angles, self.joint_limits_deg[:, 0], self.joint_limits_deg[:, 1])

    def _clamp_config_rad(self, joint_config: np.ndarray) -> np.ndarray:
        return np.radians(self.clamp_joint_angles_deg(np.degrees(joint_config)))

    def set_configuration(self, joint_config: np.ndarray) -> None:
        cfg = self._clamp_config_rad(np.asarray(joint_config, dtype=np.float64))
        self.set_config_calls.append(cfg)
        self._config = cfg

    def set_configuration_no_task_update(self, joint_config: np.ndarray) -> None:
        cfg = self._clamp_config_rad(np.asarray(joint_config, dtype=np.float64))
        self.set_config_no_task_calls.append(cfg)
        self._config = cfg

    def get_current_configuration(self) -> np.ndarray:
        return self._config.copy()

    def get_current_end_effector_pose(self) -> np.ndarray:
        if self.raise_on_ee_pose:
            raise RuntimeError("boom")
        return self.ee_pose.copy()

    def set_target_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        self.set_target_pose_calls.append(
            (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(orientation, dtype=np.float64).copy(),
            )
        )

    def solve_ik(self) -> bool:
        self.solve_ik_calls += 1
        if self.solve_success and self.solved_config_deg is not None:
            self._config = np.radians(np.asarray(self.solved_config_deg, np.float64))
        return self.solve_success

    def get_statistics(self) -> dict[str, float]:
        return {"last_solve_time_ms": self.last_solve_time_ms}


def _pose(translation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def _pose_with_z_rotation(angle_deg: float) -> np.ndarray:
    pose = _pose([0.0, 0.0, 0.0])
    pose[:3, :3] = _StubRotation.from_rotvec(
        np.array([0.0, 0.0, np.radians(angle_deg)], dtype=np.float64)
    ).as_matrix()
    return pose


def _z_angle_deg(rotation_matrix: np.ndarray) -> float:
    angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return float(np.degrees(angle_rad))


def _joint_limits_with_j6_max(j6_max_deg: float) -> np.ndarray:
    limits = np.tile(np.array([-180.0, 180.0], dtype=np.float64), (6, 1))
    limits[5, 1] = float(j6_max_deg)
    return limits


def _load_joint_state_thread():
    old_controller_module = sys.modules.get("arms.piper_controller")
    old_joint_state_module = sys.modules.pop("arms.threads.joint_state", None)
    stub_controller_module = types.ModuleType("arms.piper_controller")
    stub_controller_module.PiperController = object
    sys.modules["arms.piper_controller"] = stub_controller_module
    try:
        module = importlib.import_module("arms.threads.joint_state")
        return module.joint_state_thread
    finally:
        sys.modules.pop("arms.piper_controller", None)
        if old_controller_module is not None:
            sys.modules["arms.piper_controller"] = old_controller_module
        if old_joint_state_module is not None:
            sys.modules["arms.threads.joint_state"] = old_joint_state_module


class FeedbackSyncTests(unittest.TestCase):
    def test_feedback_is_authoritative_and_ee_pose_is_published(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.DISABLED, "right")
        current = np.array([10.0, 20.0, -30.0, 40.0, 50.0, -60.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        solver = FakeIKSolver(ee_pose=_pose([0.4, 0.5, 0.6]))

        ik_solver_thread(dm, solver, hand="right")

        # First top-of-loop sync pushes measured feedback (radians) into the solver.
        self.assertGreaterEqual(len(solver.set_config_no_task_calls), 1)
        np.testing.assert_allclose(
            solver.set_config_no_task_calls[0], np.radians(current)
        )
        # The solver's EE pose is mirrored into shared state for visualization.
        np.testing.assert_allclose(
            dm.get_current_end_effector_pose("right"), solver.ee_pose
        )


class PolicyControlledTests(unittest.TestCase):
    def test_policy_controlled_holds_pose_and_marks_success(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        solver = FakeIKSolver(ee_pose=_pose([1.0, 1.0, 1.0]))

        ik_solver_thread(dm, solver, hand="right")

        # Targets are not overridden; the current EE pose is held as the target.
        np.testing.assert_allclose(dm.get_target_pose("right"), solver.ee_pose)
        self.assertTrue(dm.get_ik_success("right"))
        self.assertEqual(dm.get_ik_solve_time_ms("right"), 0.0)
        # No teleop IK target was computed.
        self.assertIsNone(dm.get_target_joint_angles("right"))
        self.assertEqual(solver.solve_ik_calls, 0)
        # The solver configuration was synced from feedback in this branch too.
        self.assertTrue(
            any(np.allclose(c, np.radians(current)) for c in solver.set_config_calls)
        )

    def test_preset_controlled_behaves_like_policy(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.PRESET_CONTROLLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        solver = FakeIKSolver(ee_pose=_pose([2.0, 0.0, 0.0]))

        ik_solver_thread(dm, solver, hand="right")

        np.testing.assert_allclose(dm.get_target_pose("right"), solver.ee_pose)
        self.assertTrue(dm.get_ik_success("right"))
        self.assertEqual(solver.solve_ik_calls, 0)


class HomingDisabledTests(unittest.TestCase):
    def test_disabled_holds_current_joint_angles(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.DISABLED, "right")
        current = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        solver = FakeIKSolver(ee_pose=_pose([0.1, 0.2, 0.3]))

        ik_solver_thread(dm, solver, hand="right")

        np.testing.assert_allclose(dm.get_target_joint_angles("right"), current)
        np.testing.assert_allclose(dm.get_target_pose("right"), solver.ee_pose)
        self.assertTrue(dm.get_ik_success("right"))
        self.assertEqual(dm.get_ik_solve_time_ms("right"), 0.0)
        # J6 joystick offset only applies when ENABLED.
        self.assertEqual(dm.get_joystick_y("right"), 0.0)

    def test_disabled_without_feedback_is_a_noop(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.DISABLED, "right")
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        self.assertIsNone(dm.get_target_joint_angles("right"))


class BrainControlTests(unittest.TestCase):
    def _arm_brain(
        self,
        dm: DataManager,
        *,
        diameter: float,
        slider,
        active_group: str,
        clamp_mode: str,
        selected_alpha: float,
        unselected_alpha: float,
        hand: str = "right",
    ) -> None:
        dm.set_brain_control_params(
            armed=True,
            diameter=diameter,
            slider_norm=np.asarray(slider, dtype=np.float64),
            clamp_mode=clamp_mode,
            active_group=active_group,
            selected_alpha=selected_alpha,
            unselected_alpha=unselected_alpha,
            hand=hand,
        )

    def test_brain_control_all_joints_offsets_from_baseline(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.PAUSED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        self._arm_brain(
            dm,
            diameter=10.0,
            slider=np.ones(6),
            active_group="all",
            clamp_mode="off",
            selected_alpha=1.0,
            unselected_alpha=0.0,
        )
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        # offset = slider * diameter * BRAIN_GAINS([1,1,1,.35,.35,.35]); alpha=1 everywhere.
        expected = np.array([10.0, 10.0, 10.0, 3.5, 3.5, 3.5], dtype=np.float64)
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)
        self.assertTrue(dm.get_ik_success("right"))
        self.assertEqual(dm.get_ik_solve_time_ms("right"), 0.0)
        # Baseline was captured from measured feedback (all zeros).
        np.testing.assert_allclose(
            dm.get_brain_control("right").baseline_deg, np.zeros(6)
        )
        # Smoothing continuity is recorded for the next iteration.
        np.testing.assert_allclose(
            dm.get_brain_control("right").last_commanded_deg, expected
        )

    def test_brain_control_soft_damp_only_moves_active_group(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        self._arm_brain(
            dm,
            diameter=10.0,
            slider=np.ones(6),
            active_group="J4-6",
            clamp_mode="soft_damp",
            selected_alpha=1.0,
            unselected_alpha=0.0,
        )
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        # Only joints 4-6 are active; unselected alpha is 0 so 1-3 stay at baseline.
        expected = np.array([0.0, 0.0, 0.0, 3.5, 3.5, 3.5], dtype=np.float64)
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)

    def test_brain_control_clamps_to_solver_joint_limits(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        self._arm_brain(
            dm,
            diameter=400.0,
            slider=np.ones(6),
            active_group="J6",
            clamp_mode="soft_damp",
            selected_alpha=1.0,
            unselected_alpha=0.0,
        )
        solver = FakeIKSolver(joint_limits_deg=_joint_limits_with_j6_max(100.0))

        ik_solver_thread(dm, solver, hand="right")

        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=np.float64)
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)
        np.testing.assert_allclose(
            dm.get_brain_control("right").last_commanded_deg, expected
        )

    def test_brain_control_skipped_when_disabled(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.DISABLED, "right")
        current = np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        self._arm_brain(
            dm,
            diameter=10.0,
            slider=np.ones(6),
            active_group="all",
            clamp_mode="off",
            selected_alpha=1.0,
            unselected_alpha=0.0,
        )
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        # DISABLED falls through to the hold branch, not brain control.
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), current)


class TeleopTests(unittest.TestCase):
    def _setup_teleop(
        self,
        dm: DataManager,
        *,
        controller_translation,
        controller_initial_translation,
        robot_initial_translation,
        rpy_offset_deg,
        hand: str = "right",
    ) -> None:
        dm.set_robot_activity_state(RobotActivityState.ENABLED, hand)
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), hand)
        dm.set_teleop_delta_rpy_offset_deg(rpy_offset_deg, hand)
        dm.set_teleop_state(
            hand,
            True,
            _pose(controller_initial_translation),
            _pose(robot_initial_translation),
        )
        dm.set_controller_data(hand, _pose(controller_translation), 1.0, 1.0)

    def test_successful_solve_publishes_joint_target(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        solved = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float64)
        solver = FakeIKSolver(
            solve_success=True, solved_config_deg=solved, last_solve_time_ms=4.5
        )
        self._setup_teleop(
            dm,
            controller_translation=[0.1, 0.2, 0.3],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[1.0, 2.0, 3.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],  # frame correction disabled
        )

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 1)
        expected_pos = np.array([1.0, 2.0, 3.0]) + np.array(
            [0.1, 0.2, 0.3]
        ) * TRANSLATION_SCALE
        target_pos, _ = solver.set_target_pose_calls[-1]
        np.testing.assert_allclose(target_pos, expected_pos)
        # The shared target pose mirrors the same translation.
        np.testing.assert_allclose(dm.get_target_pose("right")[:3, 3], expected_pos)
        # The target advances toward IK while ramping up from rest.
        actual = dm.get_target_joint_angles("right")
        self.assertTrue(np.all(actual > 0))
        self.assertTrue(np.all(actual <= solved))
        self.assertLessEqual(float(np.max(actual)), 360.0 / IK_SOLVER_RATE**2)
        self.assertTrue(dm.get_ik_success("right"))
        self.assertEqual(dm.get_ik_solve_time_ms("right"), 4.5)

    def test_successful_solve_clamps_to_solver_joint_limits(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        solved = np.array([11.0, 22.0, 33.0, 44.0, 55.0, 125.0], dtype=np.float64)
        solver = FakeIKSolver(
            solve_success=True,
            solved_config_deg=solved,
            joint_limits_deg=_joint_limits_with_j6_max(100.0),
        )
        self._setup_teleop(
            dm,
            controller_translation=[0.1, 0.2, 0.3],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[1.0, 2.0, 3.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],
        )
        expected = solved.copy()
        expected[5] = 100.0
        dm.set_current_joint_angles(expected, "right")

        ik_solver_thread(dm, solver, hand="right")

        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)
        np.testing.assert_allclose(
            np.degrees(solver.set_config_no_task_calls[-1]), expected
        )

    def test_frame_correction_rotates_controller_delta(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        solver = FakeIKSolver(solve_success=True, solved_config_deg=np.zeros(6))
        self._setup_teleop(
            dm,
            controller_translation=[0.1, 0.2, 0.3],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[1.0, 2.0, 3.0],
            rpy_offset_deg=[0.0, 0.0, 90.0],  # yaw +90deg => Rz(90)
        )

        ik_solver_thread(dm, solver, hand="right")

        rotated_delta = np.array([-0.2, 0.1, 0.3])
        expected_pos = np.array([1.0, 2.0, 3.0]) + rotated_delta * TRANSLATION_SCALE
        target_pos, _ = solver.set_target_pose_calls[-1]
        np.testing.assert_allclose(target_pos, expected_pos, atol=1e-9)

    def test_rotation_vector_unwraps_across_180_degrees(self) -> None:
        dm = _ControllerSequenceDataManager(
            [
                _pose_with_z_rotation(170.0),
                _pose_with_z_rotation(190.0),
            ]
        )
        solver = FakeIKSolver(solve_success=True, solved_config_deg=np.zeros(6))
        self._setup_teleop(
            dm,
            controller_translation=[0.0, 0.0, 0.0],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[0.0, 0.0, 0.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],
        )

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 2)
        _, target_orientation = solver.set_target_pose_calls[-1]
        self.assertAlmostEqual(
            _z_angle_deg(target_orientation), 190.0 * ROTATION_SCALE, places=6
        )

    def test_rotation_vector_unwrap_preserves_later_windings(self) -> None:
        dm = _ControllerSequenceDataManager(
            [
                _pose_with_z_rotation(170.0),
                _pose_with_z_rotation(190.0),
                _pose_with_z_rotation(350.0),
                _pose_with_z_rotation(370.0),
                _pose_with_z_rotation(530.0),
                _pose_with_z_rotation(550.0),
            ]
        )
        solver = FakeIKSolver(solve_success=True, solved_config_deg=np.zeros(6))
        self._setup_teleop(
            dm,
            controller_translation=[0.0, 0.0, 0.0],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[0.0, 0.0, 0.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],
        )

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 6)
        _, target_orientation = solver.set_target_pose_calls[-1]
        np.testing.assert_allclose(
            target_orientation,
            _pose_with_z_rotation(550.0 * ROTATION_SCALE)[:3, :3],
            atol=1e-9,
        )

    def test_rotation_vector_reset_seeds_from_current_principal_value(self) -> None:
        raw_rotvec = _StubRotation.from_rotvec(
            np.array([0.0, 0.0, np.radians(190.0)], dtype=np.float64)
        ).as_rotvec()

        unwrapped = _select_continuous_rotvec(
            raw_rotvec, np.array([0.0, 0.0, np.radians(170.0)], dtype=np.float64)
        )
        reset = _select_continuous_rotvec(raw_rotvec, None)

        np.testing.assert_allclose(unwrapped, [0.0, 0.0, np.radians(190.0)])
        np.testing.assert_allclose(reset, [0.0, 0.0, np.radians(-170.0)])

    def test_missing_initial_transforms_skips_iteration(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        # Active teleop but no captured initial transforms.
        dm.set_teleop_state("right", True, None, None)
        dm.set_controller_data("right", _pose([0.1, 0.2, 0.3]), 1.0, 1.0)
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 0)
        self.assertEqual(solver.set_target_pose_calls, [])
        self.assertIsNone(dm.get_target_joint_angles("right"))

    def test_repeated_failures_deactivate_teleop(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=10)
        solver = FakeIKSolver(solve_success=False)
        self._setup_teleop(
            dm,
            controller_translation=[0.1, 0.2, 0.3],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[1.0, 2.0, 3.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],
        )

        ik_solver_thread(dm, solver, hand="right")

        # 10 consecutive failures (MAX_CONSECUTIVE_IK_FAILURES) disarm teleop.
        self.assertEqual(solver.solve_ik_calls, 10)
        self.assertFalse(dm.get_teleop_active("right"))
        self.assertFalse(dm.get_ik_success("right"))

    def test_single_failure_keeps_teleop_active(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        solver = FakeIKSolver(solve_success=False)
        self._setup_teleop(
            dm,
            controller_translation=[0.1, 0.2, 0.3],
            controller_initial_translation=[0.0, 0.0, 0.0],
            robot_initial_translation=[1.0, 2.0, 3.0],
            rpy_offset_deg=[0.0, 0.0, 0.0],
        )

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 1)
        self.assertTrue(dm.get_teleop_active("right"))
        self.assertFalse(dm.get_ik_success("right"))


class JoystickJ6Tests(unittest.TestCase):
    def _activate_teleop(self, dm: DataManager) -> np.ndarray:
        controller_pose = _pose([0.3, 0.2, 0.1])
        dm.set_teleop_state(
            "right",
            True,
            _pose([0.0, 0.0, 0.0]),
            _pose([1.0, 1.0, 1.0]),
        )
        dm.set_controller_data("right", controller_pose, 1.0, 1.0)
        return controller_pose

    def test_joystick_applies_j6_offset_when_enabled(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        dm.set_joystick_y(0.5, "right")
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        # delta = joystick_y * J6_JOYSTICK_SPEED * dt, added to joint 6 only.
        expected = current.copy()
        expected[5] += 0.5 * J6_JOYSTICK_SPEED * _DT
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)

    def test_joystick_clamps_to_solver_joint_limits(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 99.9], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        dm.set_joystick_y(0.5, "right")
        solver = FakeIKSolver(joint_limits_deg=_joint_limits_with_j6_max(100.0))

        ik_solver_thread(dm, solver, hand="right")

        expected = current.copy()
        expected[5] = 100.0
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)
        np.testing.assert_allclose(
            np.degrees(solver.set_config_no_task_calls[-1]), expected
        )

    def test_joystick_skips_teleop_ik_while_active(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        self._activate_teleop(dm)
        dm.set_joystick_y(0.5, "right")
        solver = FakeIKSolver(
            solve_success=True,
            solved_config_deg=np.array([90.0, 90.0, 90.0, 90.0, 90.0, 90.0]),
        )

        ik_solver_thread(dm, solver, hand="right")

        expected = current.copy()
        expected[5] += 0.5 * J6_JOYSTICK_SPEED * _DT
        self.assertEqual(solver.solve_ik_calls, 0)
        self.assertEqual(solver.set_target_pose_calls, [])
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)

    def test_joystick_uses_previous_target_as_baseline(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        previous_target = np.array(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64
        )
        dm.set_current_joint_angles(current, "right")
        dm.set_target_joint_angles(previous_target, "right")
        dm.set_joystick_y(0.5, "right")
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        expected = previous_target.copy()
        expected[5] += 0.5 * J6_JOYSTICK_SPEED * _DT
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)

    def test_joystick_reanchors_active_teleop_to_accepted_fk_pose(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        controller_pose = self._activate_teleop(dm)
        dm.set_joystick_y(0.5, "right")
        accepted_pose = _pose([9.0, 8.0, 7.0])
        solver = FakeIKSolver(ee_pose=accepted_pose)

        ik_solver_thread(dm, solver, hand="right")

        controller_initial, robot_initial = dm.get_initial_robot_controller_transforms(
            "right"
        )
        np.testing.assert_allclose(controller_initial, controller_pose)
        np.testing.assert_allclose(robot_initial, accepted_pose)
        np.testing.assert_allclose(dm.get_target_pose("right"), accepted_pose)

    def test_joystick_inside_deadzone_preserves_teleop_ik(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        self._activate_teleop(dm)
        dm.set_joystick_y(0.05, "right")
        solved = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float64)
        solver = FakeIKSolver(solve_success=True, solved_config_deg=solved)

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(solver.solve_ik_calls, 1)
        actual = dm.get_target_joint_angles("right")
        self.assertTrue(np.all(actual > 0))
        self.assertTrue(np.all(actual <= solved))

    def test_joystick_inside_deadzone_is_ignored(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        dm.set_joystick_y(0.05, "right")  # below J6_JOYSTICK_DEADZONE (0.1)
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        # ENABLED + inactive teleop falls to the hold branch -> target == current.
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), current)

    def test_suppressed_joystick_y_is_ignored(self) -> None:
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        dm.set_current_joint_angles(current, "right")
        dm.suppress_joystick_y("right", duration_s=10.0)
        dm.set_joystick_y(0.9, "right")
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right")

        self.assertEqual(dm.get_joystick_y("right"), 0.0)
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), current)


class JointStateClampTests(unittest.TestCase):
    class FakeController:
        def __init__(self) -> None:
            self.current = np.zeros(6, dtype=np.float64)
            self.target = np.zeros(6, dtype=np.float64)
            self.limits = np.tile(np.array([-100.0, 100.0]), (6, 1))
            self.current_gripper = 0.0
            self.gripper_history: list[float] = []
            self.diagnostics: list[str] = []

        def get_current_joint_angles(self) -> np.ndarray:
            return self.current.copy()

        def get_current_gripper_open_value(self) -> float:
            return self.current_gripper

        def set_gripper_open_value(self, value: float) -> None:
            self.current_gripper = float(value)
            self.gripper_history.append(self.current_gripper)

        def is_robot_homed(self) -> bool:
            return False

        def hold_current_position(self) -> bool:
            return True

        def is_control_loop_running(self) -> bool:
            return True

        def get_last_command_tx_age_s(self) -> float:
            return 0.0

        def is_robot_enabled(self) -> bool:
            return True

        def get_motionctrl_age_s(self) -> float:
            return 0.0

        def get_motionctrl_move_mode(self) -> int:
            return 1

        def set_target_joint_angles(self, joint_angles: np.ndarray) -> None:
            requested = np.asarray(joint_angles, dtype=np.float64)
            self.target = np.clip(requested, self.limits[:, 0], self.limits[:, 1])

        def get_target_joint_angles(self) -> np.ndarray:
            return self.target.copy()

        def get_effective_joint_limits_deg(self) -> np.ndarray:
            return self.limits.copy()

        def log_hardware_diagnostics(self, reason: str, force: bool = False) -> None:
            self.diagnostics.append(reason)

    def _teleop_gripper_loop(
        self,
        dm: DataManager,
        controller: FakeController,
    ) -> None:
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_teleop_state(
            "right",
            True,
            _pose([0.0, 0.0, 0.0]),
            _pose([0.0, 0.0, 0.0]),
        )
        dm.set_target_joint_angles(np.zeros(6, dtype=np.float64), "right")
        joint_state_thread = _load_joint_state_thread()
        with _capture_piper_logs():
            joint_state_thread(dm, controller, hand="right")

    def test_clamp_warning_identifies_joint_and_syncs_data_manager_target(self) -> None:
        joint_state_thread = _load_joint_state_thread()
        dm = _LoopLimitedDataManager(max_iterations=1)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_teleop_state(
            "right",
            True,
            _pose([0.0, 0.0, 0.0]),
            _pose([0.0, 0.0, 0.0]),
        )
        requested = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 125.0], dtype=np.float64)
        dm.set_target_joint_angles(requested, "right")
        controller = self.FakeController()

        with _capture_piper_logs() as output:
            joint_state_thread(dm, controller, hand="right")

        expected = requested.copy()
        expected[5] = 100.0
        np.testing.assert_allclose(dm.get_target_joint_angles("right"), expected)
        self.assertIn("joint6: requested 125.0° -> 100.0°", output.getvalue())
        self.assertIn("limit -100.0°..100.0°", output.getvalue())
        self.assertIn("joint target clamp (right)", controller.diagnostics)

    def test_hold_gripper_mode_scales_open_limit(self) -> None:
        dm = _GripSequenceDataManager([0.5])
        dm.set_teleop_gripper_settings(open_limit=0.2, mode="hold")
        controller = self.FakeController()

        self._teleop_gripper_loop(dm, controller)

        self.assertEqual(controller.gripper_history, [0.1])
        self.assertEqual(dm.get_target_gripper_open_value("right"), 0.1)

    def test_toggle_gripper_mode_latches_until_next_press(self) -> None:
        dm = _GripSequenceDataManager([0.0, 1.0, 1.0, 0.0, 1.0])
        dm.set_teleop_gripper_settings(open_limit=0.2, mode="toggle")
        controller = self.FakeController()

        self._teleop_gripper_loop(dm, controller)

        self.assertEqual(controller.gripper_history, [0.0, 0.2, 0.2, 0.2, 0.0])
        self.assertEqual(dm.get_target_gripper_open_value("right"), 0.0)


class ErrorHandlingTests(unittest.TestCase):
    def test_exception_triggers_callback_and_sets_stop_event(self) -> None:
        import threading

        dm = _LoopLimitedDataManager(max_iterations=5)
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        solver = FakeIKSolver(raise_on_ee_pose=True)
        stop_event = threading.Event()
        seen: list[Exception] = []

        ik_solver_thread(
            dm,
            solver,
            hand="right",
            stop_event=stop_event,
            error_callback=seen.append,
        )

        self.assertTrue(stop_event.is_set())
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], RuntimeError)
        # With a stop_event provided, shutdown is not requested globally.
        self.assertFalse(dm.is_shutdown_requested())

    def test_exception_without_stop_event_requests_shutdown(self) -> None:
        dm = DataManager()
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        solver = FakeIKSolver(raise_on_ee_pose=True)

        ik_solver_thread(dm, solver, hand="right")

        self.assertTrue(dm.is_shutdown_requested())


class StopEventTests(unittest.TestCase):
    def test_preset_stop_event_prevents_any_iteration(self) -> None:
        import threading

        dm = DataManager()
        dm.set_robot_activity_state(RobotActivityState.ENABLED, "right")
        dm.set_current_joint_angles(np.zeros(6, dtype=np.float64), "right")
        stop_event = threading.Event()
        stop_event.set()
        solver = FakeIKSolver()

        ik_solver_thread(dm, solver, hand="right", stop_event=stop_event)

        self.assertEqual(solver.set_config_no_task_calls, [])
        self.assertEqual(solver.solve_ik_calls, 0)


if __name__ == "__main__":
    unittest.main()
