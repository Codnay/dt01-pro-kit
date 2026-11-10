"""Pink URDF Inverse Kinematics Solver.

A generic class for performing inverse kinematics using Pink (Python inverse
kinematics based on Pinocchio) with any URDF robot model.
"""

import time
from typing import Any

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import DampingTask, FrameTask
from scipy.spatial.transform import Rotation

from arms.vectorised_posture_task import VectorisedPostureTask

try:
    from arms.logging_setup import get_logger
except ModuleNotFoundError:
    from arms.logging_setup import get_logger

log = get_logger("ik")


class PinkIKSolver:
    """A generic Pink-based inverse kinematics solver for URDF robots.

    This class provides a clean interface for setting up and solving inverse
    kinematics problems using Pink with configurable tasks, limits, and
    parameters.
    """

    def __init__(
        self,
        urdf_path: str,
        end_effector_frame: str,
        solver_name: str = "quadprog",
        position_cost: float = 1.0,
        orientation_cost: float = 0.75,
        frame_task_gain: float = 1.0,
        lm_damping: float = 0.25,
        damping_cost: float = 0.0,
        solver_damping_value: float = 1e-12,
        integration_time_step: float = 0.001,
        initial_configuration: np.ndarray | None = None,
        posture_cost_vector: np.ndarray | None = None,
        joint_position_limits_deg: np.ndarray | None = None,
        joint_velocity_limits_deg_s: np.ndarray | None = None,
    ) -> None:
        """Initialize the Pink IK solver.

        Args:
            urdf_path: Path to the URDF file
            end_effector_frame: Name of the end effector frame
            solver_name: Name of the QP solver to use
            position_cost: Cost weight for position tracking
            orientation_cost: Cost weight for orientation tracking
            frame_task_gain: Gain for the frame task
            lm_damping: Marquardt damping factor
            damping_cost: Cost weight for velocity damping
            solver_damping_value: Value for solver  Tikhonov regularization parameter
            integration_time_step: Time step for integration
            initial_configuration:Initial joint configuration (if None, uses neutral)
            posture_cost_vector: Cost weights for posture task per joint (if None, uses zeros)
            joint_position_limits_deg: Optional effective joint limits as a
                (nq, 2) array in degrees. When provided, these are intersected
                with the URDF limits before Pink is configured.
            joint_velocity_limits_deg_s: Optional per-joint velocity limits as an
                (nv,) array in degrees/second. When provided, these are
                intersected with the URDF velocity limits before Pink is
                configured, so the QP itself never commands a joint faster than
                this (enforcing the slew limit at the source).
        """
        self.urdf_path: str = urdf_path
        self.end_effector_frame: str = end_effector_frame
        self.solver_name: str = solver_name

        # Task parameters
        self.position_cost: float = position_cost
        self.orientation_cost: float = orientation_cost
        self.frame_task_gain: float = frame_task_gain
        self.lm_damping: float = lm_damping
        self.damping_cost: float = damping_cost
        self.solver_damping_value: float = solver_damping_value
        self.integration_time_step: float = integration_time_step

        # Statistics
        self.solve_times: list[float] = []
        self.last_solve_time: float = 0.0

        # Initialize the robot
        self._build_robot_model()
        self._apply_joint_position_limits(joint_position_limits_deg)
        self._apply_joint_velocity_limits(joint_velocity_limits_deg_s)

        # Validate end effector frame
        self._validate_end_effector_frame()

        # Set up tasks
        self._setup_tasks(initial_configuration, posture_cost_vector)

    def _build_robot_model(self) -> None:
        """Build the robot model from URDF."""
        log.info(f"Loading URDF: {self.urdf_path}")
        self.urdf_model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self.urdf_model_data: pin.ModelData = self.urdf_model.createData()
        log.info("Robot loaded successfully!")

    def _apply_joint_position_limits(
        self, joint_position_limits_deg: np.ndarray | None
    ) -> None:
        """Apply optional effective joint limits to the Pinocchio model."""
        assert self.urdf_model is not None, "Robot model must be initialized"
        self._position_limit_lower_rad = self.urdf_model.lowerPositionLimit.copy()
        self._position_limit_upper_rad = self.urdf_model.upperPositionLimit.copy()
        self._hard_position_limit_lower_rad = self._position_limit_lower_rad.copy()
        self._hard_position_limit_upper_rad = self._position_limit_upper_rad.copy()

        if joint_position_limits_deg is None:
            return

        limits_deg = np.asarray(joint_position_limits_deg, dtype=np.float64)
        expected_shape = (self.urdf_model.nq, 2)
        if limits_deg.shape != expected_shape:
            raise ValueError(
                "joint_position_limits_deg must have shape "
                f"{expected_shape}, got {limits_deg.shape}"
            )
        if np.any(~np.isfinite(limits_deg)):
            raise ValueError("joint_position_limits_deg contains non-finite values")
        if np.any(limits_deg[:, 0] > limits_deg[:, 1]):
            raise ValueError("joint_position_limits_deg lower limit exceeds upper limit")

        requested_lower = np.radians(limits_deg[:, 0])
        requested_upper = np.radians(limits_deg[:, 1])
        lower = np.maximum(self.urdf_model.lowerPositionLimit, requested_lower)
        upper = np.minimum(self.urdf_model.upperPositionLimit, requested_upper)
        if np.any(lower > upper):
            raise ValueError(
                "joint_position_limits_deg do not overlap with the URDF joint limits"
            )

        self.urdf_model.lowerPositionLimit[:] = lower
        self.urdf_model.upperPositionLimit[:] = upper
        self._position_limit_lower_rad = lower.copy()
        self._position_limit_upper_rad = upper.copy()
        effective_limits_deg = np.column_stack(
            (np.degrees(lower), np.degrees(upper))
        )
        log.debug(f"IK effective joint limits (deg): {effective_limits_deg}")

    def _apply_joint_velocity_limits(
        self, joint_velocity_limits_deg_s: np.ndarray | None
    ) -> None:
        """Apply optional per-joint velocity limits to the Pinocchio model.

        Pink's QP enforces ``model.velocityLimit`` as a hard constraint, so
        lowering it here bounds the joint velocity the IK will ever solve for --
        the limit is enforced at the source instead of relying on a downstream
        clamp. Requested limits are intersected with (min of) the URDF limits, so
        this can only lower them, never raise above what the model declares.
        """
        assert self.urdf_model is not None, "Robot model must be initialized"
        self._velocity_limit_rad_s = self.urdf_model.velocityLimit.copy()

        if joint_velocity_limits_deg_s is None:
            return

        limits_deg_s = np.asarray(joint_velocity_limits_deg_s, dtype=np.float64).reshape(-1)
        expected_shape = (self.urdf_model.nv,)
        if limits_deg_s.shape != expected_shape:
            raise ValueError(
                "joint_velocity_limits_deg_s must have shape "
                f"{expected_shape}, got {limits_deg_s.shape}"
            )
        if np.any(~np.isfinite(limits_deg_s)) or np.any(limits_deg_s <= 0.0):
            raise ValueError("joint_velocity_limits_deg_s must be finite and > 0")

        requested_rad_s = np.radians(limits_deg_s)
        effective = np.minimum(self.urdf_model.velocityLimit, requested_rad_s)
        self.urdf_model.velocityLimit[:] = effective
        self._velocity_limit_rad_s = effective.copy()
        log.debug(
            "IK effective joint velocity limits (deg/s): "
            f"{np.degrees(effective)}"
        )

    def _validate_end_effector_frame(self) -> None:
        """Validate that the end effector frame exists.

        Raises:
            ValueError: If the end effector frame is not found in the URDF.
        """
        assert self.urdf_model is not None, "Robot model must be initialized"
        try:
            frame_id = self.urdf_model.getFrameId(self.end_effector_frame)
            if frame_id >= self.urdf_model.nframes:
                raise ValueError(f"Frame {self.end_effector_frame} not found in URDF")
        except Exception:
            available_frames = [
                self.urdf_model.frames[i].name for i in range(self.urdf_model.nframes)
            ]
            raise ValueError(
                f"Frame {self.end_effector_frame} not found in URDF. Available frames: {available_frames}"
            )

    def _setup_tasks(
        self,
        initial_configuration: np.ndarray | None = None,
        posture_cost_vector: np.ndarray | None = None,
    ) -> None:
        """Set up Pink tasks and configuration.

        Args:
            initial_configuration: Initial joint configuration (if None, uses neutral)
            posture_cost_vector: Cost weights for posture task per joint (if None, uses zeros)

        Raises:
            ValueError: If the initial configuration is not valid (initial
                configuration must have the same number of joints as the robot
                model).
        """
        assert self.urdf_model is not None, "Robot model must be initialized"

        # Initial configuration
        self.initial_configuration = initial_configuration
        self.posture_cost_vector = posture_cost_vector

        if (
            self.initial_configuration is not None
            and len(self.initial_configuration) != self.urdf_model.nq
        ):
            raise ValueError(
                f"Initial configuration must have {self.urdf_model.nq} values, got {len(self.initial_configuration)}"
            )

        log.info("Setting up Pink tasks and limits...")

        assert self.urdf_model is not None, "Robot model must be initialized"
        assert self.urdf_model_data is not None, "Robot model data must be initialized"
        self.initial_configuration = (
            self.initial_configuration
            if self.initial_configuration is not None
            else pin.neutral(self.urdf_model)
        )
        self.initial_configuration = np.clip(
            self.initial_configuration,
            self._position_limit_lower_rad,
            self._position_limit_upper_rad,
        )
        self.configuration = pink.Configuration(
            self.urdf_model, self.urdf_model_data, self.initial_configuration
        )

        # Set up end effector task
        self.ee_task = FrameTask(
            self.end_effector_frame,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
            gain=self.frame_task_gain,
        )
        self.ee_task.set_target_from_configuration(self.configuration)

        # Set up damping task
        self.damping_task = DampingTask(cost=self.damping_cost)

        # Set up posture task with vectorized cost
        # Default to zeros if not provided, matching the number of joints
        if self.posture_cost_vector is None:
            self.posture_cost_vector = np.zeros(self.urdf_model.nq)
        elif len(self.posture_cost_vector) != self.urdf_model.nq:
            raise ValueError(
                f"Posture cost vector must have {self.urdf_model.nq} values, got {len(self.posture_cost_vector)}"
            )
        else:
            self.posture_cost_vector = np.array(self.posture_cost_vector).copy()

        self.posture_task = VectorisedPostureTask(cost=self.posture_cost_vector)
        self.posture_task.set_target_from_configuration(self.configuration)

        log.info("Tasks configured!")

    def update_task_parameters(
        self,
        position_cost: float | None = None,
        orientation_cost: float | None = None,
        frame_task_gain: float | None = None,
        lm_damping: float | None = None,
        damping_cost: float | None = None,
        solver_damping_value: float | None = None,
        integration_time_step: float | None = None,
        posture_cost_vector: np.ndarray | None = None,
    ) -> None:
        """Update task parameters dynamically.

        Args:
            position_cost: Cost weight for position tracking
            orientation_cost: Cost weight for orientation tracking
            frame_task_gain: Gain for the frame task
            lm_damping: Marquardt damping factor
            damping_cost: Cost weight for velocity damping
            solver_damping_value: Value for solver damping - Tikhonov regularization parameter
            integration_time_step: Time step for integration
            posture_cost_vector: Cost weights for posture task per joint
        """
        assert self.ee_task is not None, "End effector task must be initialized"
        assert self.damping_task is not None, "Damping task must be initialized"
        if position_cost is not None:
            self.position_cost = position_cost
            if hasattr(self.ee_task, "set_position_cost"):
                self.ee_task.set_position_cost(position_cost)
            else:
                self.ee_task.position_cost = position_cost

        if orientation_cost is not None:
            self.orientation_cost = orientation_cost
            if hasattr(self.ee_task, "set_orientation_cost"):
                self.ee_task.set_orientation_cost(orientation_cost)
            else:
                self.ee_task.orientation_cost = orientation_cost

        if frame_task_gain is not None:
            self.frame_task_gain = frame_task_gain
            self.ee_task.gain = frame_task_gain

        if lm_damping is not None:
            self.lm_damping = lm_damping
            self.ee_task.lm_damping = lm_damping

        if damping_cost is not None:
            self.damping_cost = damping_cost
            self.damping_task.cost = damping_cost

        if solver_damping_value is not None:
            self.solver_damping_value = solver_damping_value

        if integration_time_step is not None:
            self.integration_time_step = integration_time_step

        assert self.urdf_model is not None, "Robot model must be initialized"
        if posture_cost_vector is not None:
            if len(posture_cost_vector) != self.urdf_model.nq:
                raise ValueError(
                    f"Posture cost vector must have {self.urdf_model.nq} values, got {len(posture_cost_vector)}"
                )
            self.posture_cost_vector = np.array(posture_cost_vector).copy()
            assert self.posture_task is not None, "Posture task must be initialized"
            self.posture_task.cost = self.posture_cost_vector

    def set_target_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set target pose from position and orientation.

        Args:
            position: 3D position vector
            orientation: 3x3 rotation matrix or 4-element
                quaternion (wxyz)

        Raises:
            ValueError: If the orientation is not a 3x3 matrix or 4-element
                quaternion.
        """
        if orientation.shape == (4,):
            # Quaternion (wxyz) to rotation matrix
            target_rotation = Rotation.from_quat(
                [orientation[1], orientation[2], orientation[3], orientation[0]]
            )
            rotation_matrix = target_rotation.as_matrix()
        elif orientation.shape == (3, 3):
            # Already a rotation matrix
            rotation_matrix = orientation
        else:
            raise ValueError("Orientation must be a 3x3 matrix or 4-element quaternion")

        target_transform = pin.SE3(rotation_matrix, position)
        assert self.ee_task is not None, "End effector task must be initialized"
        self.ee_task.set_target(target_transform)

    def solve_ik(self, dt: float | None = None) -> bool:
        """Solve inverse kinematics for current target.

        Args:
            dt: Integration time step (uses instance default if None).

        Returns:
            True if successful, False otherwise.

        Raises:
            Exception: If an error occurs during the IK solve.
        """
        if dt is None:
            dt = self.integration_time_step

        start_time = time.time()

        assert self.configuration is not None, "Configuration must be initialized"
        assert self.ee_task is not None, "End effector task must be initialized"
        assert self.damping_task is not None, "Damping task must be initialized"
        assert self.posture_task is not None, "Posture task must be initialized"
        try:
            # Prepare tasks and limits
            tasks = (
                self.ee_task,
                self.damping_task,
                self.posture_task,
            )
            limits = (
                self.configuration.model.configuration_limit,
                self.configuration.model.velocity_limit,
            )

            # Solve differential IK
            velocity = pink.solve_ik(
                self.configuration,
                tasks,
                dt,
                solver=self.solver_name,
                damping=self.solver_damping_value,
                limits=limits,
            )

            # Integrate configuration
            new_q = self.configuration.integrate(velocity, dt)

            # update configuration
            self.set_configuration_no_task_update(new_q)

            # Update timing statistics
            elapsed_time = time.time() - start_time
            self.last_solve_time = elapsed_time * 1000  # Convert to ms
            self.solve_times.append(self.last_solve_time)

            # Keep only last 100 solve times for statistics
            if len(self.solve_times) > 100:
                self.solve_times = self.solve_times[-100:]

            return True

        except Exception as e:
            log.error(f"IK solve failed: {e}")
            return False

    def get_current_configuration(self) -> np.ndarray:
        """Get current joint configuration (in radians)."""
        assert self.configuration is not None, "Configuration must be initialized"
        return self.configuration.q.copy()

    def get_joint_position_limits_deg(self, *, hard: bool = False) -> np.ndarray:
        """Return original URDF limits or the narrower effective IK limits."""
        lower = (
            self._hard_position_limit_lower_rad
            if hard
            else self._position_limit_lower_rad
        )
        upper = (
            self._hard_position_limit_upper_rad
            if hard
            else self._position_limit_upper_rad
        )
        return np.degrees(np.column_stack((lower, upper)))

    def get_pose_for_configuration_deg(self, joints_deg: np.ndarray) -> np.ndarray:
        """Compute FK without clipping a valid held pose to the soft margin."""
        joints = np.asarray(joints_deg, dtype=np.float64)
        limits = self.get_joint_position_limits_deg(hard=True)
        if (
            joints.shape != (6,)
            or not np.all(np.isfinite(joints))
            or np.any(joints < limits[:, 0])
            or np.any(joints > limits[:, 1])
        ):
            raise ValueError("FK reference is outside finite URDF joint limits")
        data = self.urdf_model.createData()
        pin.framesForwardKinematics(self.urdf_model, data, np.radians(joints))
        return data.oMf[
            self.urdf_model.getFrameId(self.end_effector_frame)
        ].homogeneous.copy()

    def clamp_joint_angles_deg(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """Clamp joint angles in degrees to the IK model's effective limits."""
        assert self.urdf_model is not None, "Robot model must be initialized"
        angles_deg = np.asarray(joint_angles_deg, dtype=np.float64)
        if angles_deg.shape != (self.urdf_model.nq,):
            raise ValueError(
                "joint_angles_deg must have shape "
                f"({self.urdf_model.nq},), got {angles_deg.shape}"
            )
        lower_deg = np.degrees(self._position_limit_lower_rad)
        upper_deg = np.degrees(self._position_limit_upper_rad)
        return np.clip(angles_deg, lower_deg, upper_deg)

    def get_current_end_effector_pose(self) -> np.ndarray:
        """Get current end effector pose.

        Returns:
            4x4 transform matrix.
        """
        assert self.configuration is not None, "Configuration must be initialized"
        transform: pin.SE3 = self.configuration.get_transform_frame_to_world(
            self.end_effector_frame
        )
        return transform.np.copy()

    def get_statistics(self) -> dict[str, float | int]:
        """Get solver timing statistics (in milliseconds)."""
        if not self.solve_times:
            return {
                "last_solve_time_ms": 0.0,
                "avg_solve_time_ms": 0.0,
                "max_solve_time_ms": 0.0,
            }

        return {
            "last_solve_time_ms": self.last_solve_time,
            "avg_solve_time_ms": np.mean(self.solve_times),
            "max_solve_time_ms": np.max(self.solve_times),
            "solve_count": len(self.solve_times),
        }

    def set_configuration_no_task_update(self, joint_config: np.ndarray) -> None:
        """Set the robot to a specific joint configuration without updating the task.

        Args:
            joint_config: Array of joint angles to set

        Raises:
            ValueError: If the joint configuration is not valid (joint
                configuration must have the same number of joints as the robot
                model).
        """
        assert self.urdf_model is not None, "Robot model must be initialized"
        assert self.configuration is not None, "Configuration must be initialized"
        if len(joint_config) != self.urdf_model.nq:
            raise ValueError(
                f"Joint configuration must have {self.urdf_model.nq} values, got {len(joint_config)}"
            )

        # Clamp configuration to limits
        q_max = self.urdf_model.upperPositionLimit
        q_min = self.urdf_model.lowerPositionLimit
        joint_config = np.clip(joint_config, q_min, q_max)

        # Update configuration
        self.configuration.update(joint_config)

    def set_configuration(self, joint_config: np.ndarray) -> None:
        """Set the robot to a specific joint configuration.

        Args:
            joint_config: Array of joint angles to set

        Raises:
            ValueError: If the joint configuration is not valid (joint
                configuration must have the same number of joints as the robot
                model).
        """
        self.set_configuration_no_task_update(joint_config)
        self.ee_task.set_target_from_configuration(self.configuration)

    def reset_to_neutral(self) -> None:
        """Reset robot to neutral configuration."""
        assert self.urdf_model is not None, "Robot model must be initialized"
        self.set_configuration(pin.neutral(self.urdf_model))

    def _get_available_frames(self) -> list[str]:
        """Get list of available frame names."""
        assert self.urdf_model is not None, "Robot model must be initialized"
        return [self.urdf_model.frames[i].name for i in range(self.urdf_model.nframes)]

    def get_robot_info(self) -> dict[str, Any]:
        """Get robot information."""
        assert self.urdf_model is not None, "Robot model must be initialized"
        return {
            "urdf_path": self.urdf_path,
            "end_effector_frame": self.end_effector_frame,
            "num_joints": self.urdf_model.nq,
            "num_dof": self.urdf_model.nv,
            "num_frames": self.urdf_model.nframes,
            "solver_name": self.solver_name,
            "available_frames": self._get_available_frames(),
        }
