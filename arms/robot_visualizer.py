#!/usr/bin/env python3
"""Shared robot visualizer for Piper robot demos.

This module provides a clean interface for visualizing robot state using Viser,
encapsulating all the repeated setup, GUI controls, and update logic.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf


class RobotVisualizer:
    """Shared visualizer for robot demos.

    Encapsulates viser server setup, GUI controls, and update logic.
    """

    def __init__(self, urdf_path: str) -> None:
        """Initialize the visualizer.

        Args:
            urdf_path: Path to URDF file for robot visualization
        """
        # Initialize viser server
        self._server = viser.ViserServer()
        self._server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Load URDF for visualization
        urdf = yourdfpy.URDF.load(urdf_path)
        self._urdf_vis = ViserUrdf(self._server, urdf, root_node_name="/robot_actual")

        ghost_urdf = yourdfpy.URDF.load(urdf_path)
        # Ghost robot with semi-transparent blue color to make it visually distinct
        self._ghost_robot_urdf = ViserUrdf(
            self._server,
            ghost_urdf,
            root_node_name="/robot_ghost",
            mesh_color_override=(0.2, 0.4, 1.0, 0.25),  # Blue with 60% opacity
        )

        # GUI handles (initialized as None, created on demand) - all private
        self._timing_handle = None
        self._joint_angles_handle = None
        self._robot_status_handle = None
        self._teleop_status_handle = None
        self._controller_status_handle = None
        self._gripper_status_handle = None

        # Pink parameter handles
        self._position_weight_handle = None
        self._orientation_weight_handle = None
        self._frame_task_gain_handle = None
        self._lm_damping_handle = None
        self._damping_weight_handle = None
        self._solver_damping_value_handle = None
        self._posture_cost_handles: list[Any] = []

        # Robot control handles
        self._enable_robot_handle = None
        self._disable_robot_handle = None
        self._emergency_stop_handle = None
        self._go_home_button = None
        self._toggle_robot_enabled_status_button = None

        # Teleop-specific handles
        self._grip_value_handle = None
        self._trigger_value_handle = None

        # Rate control handles
        self._controller_min_cutoff_handle = None
        self._controller_beta_handle = None
        self._controller_d_cutoff_handle = None

        self._translation_scale_handle = None
        self._rotation_scale_handle = None

        # Visualization handles
        self._controller_handle = None
        self._target_frame_handle = None

        # Policy-related handles
        self._policy_status_handle = None
        self._prediction_ratio_handle = None
        self._policy_execution_rate_handle = None
        self._robot_rate_handle = None
        self._execution_mode_dropdown = None
        self._run_policy_button = None
        self._start_policy_execution_button = None
        self._run_and_start_policy_execution_button = None
        self._play_policy_button = None

        # Brain control handles (support multiple independent control groups, e.g. left/right).
        self._brain_controls_by_key: dict[str, dict[str, Any]] = {}
        self._teleop_frame_controls_by_key: dict[str, dict[str, Any]] = {}

        # Internal state
        self._ema_timing = 0.001

    @property
    def server(self) -> viser.ViserServer:
        """Expose the underlying Viser server for advanced/custom UI."""
        return self._server

    def add_basic_controls(self) -> None:
        """Add basic GUI controls (timing, joint angles)."""
        self._timing_handle = self._server.gui.add_number(
            "IK Solve Time (ms)", 0.001, disabled=True
        )
        self._joint_angles_handle = self._server.gui.add_text(
            "Joint Angles", "Waiting for IK solution..."
        )

    def add_robot_status_controls(self) -> None:
        """Add robot status display controls."""
        self._robot_status_handle = self._server.gui.add_text(
            "Robot Status", "Initializing..."
        )

    def add_teleop_controls(self) -> None:
        """Add teleoperation-specific controls."""
        self._grip_value_handle = self._server.gui.add_number(
            "Grip Value", 0.0, disabled=True
        )
        self._trigger_value_handle = self._server.gui.add_number(
            "Trigger Value (Gripper)", 0.0, disabled=True
        )
        self._teleop_status_handle = self._server.gui.add_text(
            "Teleop Status", "Inactive"
        )
        self._controller_status_handle = self._server.gui.add_text(
            "Controller Status", "Waiting..."
        )

    def add_gripper_status_controls(self) -> None:
        """Add gripper status display controls."""
        self._gripper_status_handle = self._server.gui.add_text(
            "Gripper Status", "Open (0%)"
        )

    def add_homing_controls(self) -> None:
        """Add homing controls."""
        self._go_home_button = self._server.gui.add_button("Go Home")

    def add_robot_control_buttons(self) -> None:
        """Add robot control buttons (enable, disable, emergency stop).

        Note: For homing functionality, use add_homing_controls() instead.
        """
        self._enable_robot_handle = self._server.gui.add_button("Enable Robot")
        self._disable_robot_handle = self._server.gui.add_button("Disable Robot")
        self._emergency_stop_handle = self._server.gui.add_button("Emergency Stop")

    def add_toggle_robot_enabled_status_button(self) -> None:
        """Add toggle robot enabled status button.

        This creates a single button that toggles between "Enable Robot" and "Disable Robot"
        based on the current robot state.
        """
        self._toggle_robot_enabled_status_button = self._server.gui.add_button(
            "Enable Robot"
        )

    def update_toggle_robot_enabled_status(self, enabled: bool) -> None:
        """Update toggle robot enabled status button label based on robot state.

        Args:
            enabled: Whether robot is currently enabled
        """
        if self._toggle_robot_enabled_status_button is not None:
            self._toggle_robot_enabled_status_button.label = (
                "Enable Robot" if not enabled else "Disable Robot"
            )

    def set_toggle_robot_enabled_status_callback(
        self, callback: Callable[[], Any]
    ) -> None:
        """Set callback for toggle robot enabled status button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._toggle_robot_enabled_status_button is not None:
            self._toggle_robot_enabled_status_button.on_click(lambda _: callback())

    def add_pink_parameter_controls(
        self,
        position_cost: float,
        orientation_cost: float,
        frame_task_gain: float,
        lm_damping: float,
        damping_cost: float,
        solver_damping_value: float,
        posture_cost_vector: list[float],
    ) -> None:
        """Add Pink IK parameter controls.

        Args:
            position_cost: Initial position cost value
            orientation_cost: Initial orientation cost value
            frame_task_gain: Initial frame task gain value
            lm_damping: Initial LM damping value
            damping_cost: Initial damping cost value
            solver_damping_value: Initial solver damping value
            posture_cost_vector: Initial posture cost vector (one value per joint)
        """
        self._position_weight_handle = self._server.gui.add_number(
            "Position Weight", position_cost, min=0.0, max=10.0, step=0.1
        )
        self._orientation_weight_handle = self._server.gui.add_number(
            "Orientation Weight", orientation_cost, min=0.0, max=1.0, step=0.01
        )
        self._frame_task_gain_handle = self._server.gui.add_number(
            "Frame Task Gain", frame_task_gain, min=0.0, max=10.0, step=0.1
        )
        self._lm_damping_handle = self._server.gui.add_number(
            "LM Damping", lm_damping, min=0.0, max=5.0, step=0.01
        )
        self._damping_weight_handle = self._server.gui.add_number(
            "Damping Weight", damping_cost, min=0.0, max=1.0, step=0.01
        )
        self._solver_damping_value_handle = self._server.gui.add_number(
            "Solver Damping Value", solver_damping_value, min=0.0, max=1.0, step=0.0001
        )

        # Posture cost controls (one per joint)
        self._posture_cost_handles = []
        for i in range(len(posture_cost_vector)):
            handle = self._server.gui.add_number(
                f"Posture Cost J{i+1}",
                posture_cost_vector[i],
                min=0.0,
                max=1.0,
                step=0.01,
            )
            self._posture_cost_handles.append(handle)

    def add_controller_filter_controls(
        self,
        initial_min_cutoff: float,
        initial_beta: float,
        initial_d_cutoff: float,
    ) -> None:
        """Add 1€ Filter parameter controls for controller.

        Args:
            initial_min_cutoff: Initial minimum cutoff frequency
            initial_beta: Initial speed coefficient
            initial_d_cutoff: Initial derivative cutoff frequency
        """
        self._controller_min_cutoff_handle = self._server.gui.add_number(
            "Controller Min Cutoff",
            initial_min_cutoff,
            min=0.01,
            max=10.0,
            step=0.01,
        )
        self._controller_beta_handle = self._server.gui.add_number(
            "Controller Beta", initial_beta, min=0.0, max=10.0, step=0.01
        )
        self._controller_d_cutoff_handle = self._server.gui.add_number(
            "Controller D Cutoff",
            initial_d_cutoff,
            min=0.01,
            max=10.0,
            step=0.01,
        )

    def add_scaling_controls(
        self, initial_translation_scale: float, initial_rotation_scale: float
    ) -> None:
        """Add scaling factor controls for translation and rotation.

        Args:
            initial_translation_scale: Initial translation scale factor
            initial_rotation_scale: Initial rotation scale factor
        """
        self._translation_scale_handle = self._server.gui.add_number(
            "Translation Scale",
            initial_translation_scale,
            min=0.1,
            max=10.0,
            step=0.001,
        )
        self._rotation_scale_handle = self._server.gui.add_number(
            "Rotation Scale", initial_rotation_scale, min=0.1, max=10.0, step=0.001
        )

    def add_controller_visualization(self) -> None:
        """Add controller transform visualization."""
        self._controller_handle = self._server.scene.add_transform_controls(
            "/controller",
            scale=0.15,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
        )

    def add_target_frame_visualization(self) -> None:
        """Add target/goal frame visualization."""
        self._target_frame_handle = self._server.scene.add_frame(
            "/target_goal", axes_length=0.1, axes_radius=0.003
        )

    def add_brain_control_controls(self, *, key: str = "right", label_prefix: str = "") -> None:
        """Add 'Final Brain Control' GUI controls for joint-level nudges.

        Args:
            key: Identifier for this control group (e.g. "right", "left").
            label_prefix: Optional prefix to disambiguate labels in the GUI.
        """
        if key in self._brain_controls_by_key:
            return

        prefix = (label_prefix.strip() + ": ") if label_prefix.strip() else ""
        arm_dropdown = self._server.gui.add_dropdown(
            f"{prefix}Final Brain Control",
            options=["DISARMED", "ARMED"],
            initial_value="DISARMED",
        )

        diameter_handle = self._server.gui.add_number(
            f"{prefix}Diameter",
            initial_value=0.0,
            min=0.0,
            max=10.0,
            step=0.1,
        )

        joint_offset_handles: list[Any] = []
        for i in range(6):
            handle = self._server.gui.add_slider(
                f"{prefix}J{i+1} Offset",
                min=-1.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
            )
            joint_offset_handles.append(handle)

        zero_offsets_button = self._server.gui.add_button(f"{prefix}Zero Offsets")

        clamp_mode_dropdown = self._server.gui.add_dropdown(
            f"{prefix}Clamp Mode",
            options=["off", "soft_damp"],
            initial_value="off",
        )

        active_joints_dropdown = self._server.gui.add_dropdown(
            f"{prefix}Active Joints",
            options=["all", "J1", "J2", "J3", "J4", "J5", "J6", "J1-3", "J4-6"],
            initial_value="all",
        )

        selected_alpha_handle = self._server.gui.add_number(
            f"{prefix}Selected Alpha",
            initial_value=0.2,
            min=0.05,
            max=0.5,
            step=0.01,
        )
        unselected_alpha_handle = self._server.gui.add_number(
            f"{prefix}Unselected Alpha",
            initial_value=0.01,
            min=0.001,
            max=0.05,
            step=0.001,
        )

        capture_baseline_button = self._server.gui.add_button(
            f"{prefix}Capture Baseline Now"
        )

        self._brain_controls_by_key[key] = {
            "arm_dropdown": arm_dropdown,
            "diameter_handle": diameter_handle,
            "joint_offset_handles": joint_offset_handles,
            "zero_offsets_button": zero_offsets_button,
            "clamp_mode_dropdown": clamp_mode_dropdown,
            "active_joints_dropdown": active_joints_dropdown,
            "selected_alpha_handle": selected_alpha_handle,
            "unselected_alpha_handle": unselected_alpha_handle,
            "capture_baseline_button": capture_baseline_button,
        }

    def add_teleop_frame_controls(
        self,
        *,
        key: str = "right",
        label_prefix: str = "",
        initial_rpy_deg: np.ndarray | list[float] | tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        """Add per-arm teleop frame correction controls (roll/pitch/yaw in degrees)."""
        if key in self._teleop_frame_controls_by_key:
            return

        init = np.asarray(initial_rpy_deg, dtype=np.float64).reshape(-1)
        if init.size != 3:
            init = np.zeros(3, dtype=np.float64)

        prefix = (label_prefix.strip() + ": ") if label_prefix.strip() else ""
        roll_handle = self._server.gui.add_number(
            f"{prefix}Teleop Roll Offset (deg)",
            float(init[0]),
            min=-180.0,
            max=180.0,
            step=0.1,
        )
        pitch_handle = self._server.gui.add_number(
            f"{prefix}Teleop Pitch Offset (deg)",
            float(init[1]),
            min=-180.0,
            max=180.0,
            step=0.1,
        )
        yaw_handle = self._server.gui.add_number(
            f"{prefix}Teleop Yaw Offset (deg)",
            float(init[2]),
            min=-180.0,
            max=180.0,
            step=0.1,
        )
        zero_button = self._server.gui.add_button(f"{prefix}Zero Teleop RPY Offsets")
        zero_button.on_click(
            lambda _: (
                setattr(roll_handle, "value", 0.0),
                setattr(pitch_handle, "value", 0.0),
                setattr(yaw_handle, "value", 0.0),
            )
        )

        self._teleop_frame_controls_by_key[key] = {
            "roll_handle": roll_handle,
            "pitch_handle": pitch_handle,
            "yaw_handle": yaw_handle,
            "zero_button": zero_button,
        }

    def get_teleop_frame_offset_rpy_deg(self, *, key: str = "right") -> np.ndarray:
        """Get per-arm teleop frame correction [roll, pitch, yaw] from GUI."""
        controls = self._teleop_frame_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Teleop frame controls not initialized")
        return np.array(
            [
                float(controls["roll_handle"].value),
                float(controls["pitch_handle"].value),
                float(controls["yaw_handle"].value),
            ],
            dtype=np.float64,
        )

    def update_robot_pose(self, joint_config: np.ndarray) -> None:
        """Update robot visualization from joint configuration.

        Args:
            joint_config: Joint angles in radians
        """
        self._urdf_vis.update_cfg(joint_config)

    def update_joint_angles_display(
        self, joint_config: np.ndarray, show_gripper: bool = False
    ) -> None:
        """Update joint angles display.

        Args:
            joint_config: Joint angles in radians
            show_gripper: Whether to show gripper joints (joints 7&8)
        """
        if self._joint_angles_handle is None:
            return

        joint_angles_str = "Joint Angles (rad):\n"
        joint_angles_deg = np.degrees(joint_config)
        num_joints = len(joint_config) if show_gripper else 6

        for i in range(num_joints):
            angle_rad = joint_config[i]
            angle_deg = joint_angles_deg[i]
            joint_type = "Robot" if i < 6 else "Gripper"
            label = f"Joint {i+1} ({joint_type})" if show_gripper else f"J{i+1}"
            joint_angles_str += f"  {label}: {angle_rad:.3f} rad ({angle_deg:.1f}°)\n"

        self._joint_angles_handle.value = joint_angles_str

    def update_timing(self, solve_time_ms: float) -> None:
        """Update timing display with exponential moving average.

        Args:
            solve_time_ms: IK solve time in milliseconds
        """
        if self._timing_handle is None:
            return

        self._ema_timing = 0.99 * self._ema_timing + 0.01 * solve_time_ms
        self._timing_handle.value = self._ema_timing

    def update_robot_status(self, status: str) -> None:
        """Update robot status display.

        Args:
            status: Status string to display
        """
        if self._robot_status_handle is not None:
            self._robot_status_handle.value = status

    def update_teleop_status(self, active: bool) -> None:
        """Update teleop status display.

        Args:
            active: Whether teleop is active
        """
        if self._teleop_status_handle is not None:
            self._teleop_status_handle.value = (
                "Teleop Status: Active" if active else "Teleop Status: Inactive"
            )

    def update_controller_status_display(
        self, position: np.ndarray | None, connected: bool = True
    ) -> None:
        """Update controller status display.

        Args:
            position: Controller position (3D array) or None
            connected: Whether controller is connected
        """
        if self._controller_status_handle is None:
            return

        if connected and position is not None:
            controller_status_str = "Controller Status:\n"
            controller_status_str += f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]\n"
            controller_status_str += "  Connected: ✓\n"
            self._controller_status_handle.value = controller_status_str
        else:
            self._controller_status_handle.value = "Controller Status:\n  Connected: ✗"

    def update_gripper_status(
        self, trigger_value: float, robot_enabled: bool = True
    ) -> None:
        """Update gripper status display.

        Args:
            trigger_value: Trigger value (0.0 = open, 1.0 = closed)
            robot_enabled: Whether robot is enabled
        """
        if self._gripper_status_handle is None:
            return

        gripper_closed_percent = trigger_value * 100.0
        if trigger_value > 0.9:
            gripper_state = "Closed"
        elif trigger_value > 0.1:
            gripper_state = "Closing"
        else:
            gripper_state = "Open"

        status = f"Gripper: {gripper_state} ({gripper_closed_percent:.0f}% closed)"
        if not robot_enabled:
            status += " [Disabled]"

        self._gripper_status_handle.value = status

    def update_controller_visualization(self, transform: np.ndarray | None) -> None:
        """Update controller transform visualization.

        Args:
            transform: 4x4 transformation matrix or None
        """
        if self._controller_handle is None or transform is None:
            return

        controller_pos = transform[:3, 3]
        controller_rot = Rotation.from_matrix(transform[:3, :3])
        controller_quat_xyzw = controller_rot.as_quat()
        controller_quat_wxyz = [
            controller_quat_xyzw[3],
            controller_quat_xyzw[0],
            controller_quat_xyzw[1],
            controller_quat_xyzw[2],
        ]

        self._controller_handle.position = tuple(controller_pos)
        self._controller_handle.wxyz = tuple(controller_quat_wxyz)

    def update_target_visualization(self, transform: np.ndarray | None) -> None:
        """Update target/goal frame visualization.

        Args:
            transform: 4x4 transformation matrix or None
        """
        if self._target_frame_handle is None or transform is None:
            return

        target_pos = transform[:3, 3]
        target_rot = Rotation.from_matrix(transform[:3, :3])
        target_quat_xyzw = target_rot.as_quat()
        target_quat_wxyz = [
            target_quat_xyzw[3],
            target_quat_xyzw[0],
            target_quat_xyzw[1],
            target_quat_xyzw[2],
        ]

        self._target_frame_handle.position = tuple(target_pos)
        self._target_frame_handle.wxyz = tuple(target_quat_wxyz)

    def get_pink_parameters(self) -> dict:
        """Get Pink IK parameters from GUI controls.

        Returns:
            Dictionary with parameter values

        Raises:
            ValueError: If Pink parameter controls not initialized
        """
        if not self._posture_cost_handles:
            raise ValueError("Pink parameter controls not initialized")

        if (
            self._position_weight_handle is None
            or self._orientation_weight_handle is None
            or self._frame_task_gain_handle is None
            or self._lm_damping_handle is None
            or self._damping_weight_handle is None
            or self._solver_damping_value_handle is None
        ):
            raise ValueError("Pink parameter controls not initialized")

        posture_cost_vector = np.array(
            [handle.value for handle in self._posture_cost_handles]
        )

        params = {
            "position_cost": self._position_weight_handle.value,
            "orientation_cost": self._orientation_weight_handle.value,
            "frame_task_gain": self._frame_task_gain_handle.value,
            "lm_damping": self._lm_damping_handle.value,
            "damping_cost": self._damping_weight_handle.value,
            "solver_damping_value": self._solver_damping_value_handle.value,
            "posture_cost_vector": posture_cost_vector,
        }
        return params

    def get_controller_filter_params(self) -> tuple[float, float, float]:
        """Get 1€ Filter parameters from GUI.

        Returns:
            Tuple of (min_cutoff, beta, d_cutoff)

        Raises:
            ValueError: If controller filter controls not initialized
        """
        if (
            self._controller_min_cutoff_handle is None
            or self._controller_beta_handle is None
            or self._controller_d_cutoff_handle is None
        ):
            raise ValueError("Controller filter controls not initialized")
        return (
            self._controller_min_cutoff_handle.value,
            self._controller_beta_handle.value,
            self._controller_d_cutoff_handle.value,
        )

    def get_translation_scale(self) -> float:
        """Get translation scale value from GUI.

        Returns:
            Translation scale value

        Raises:
            ValueError: If scaling controls not initialized
        """
        if self._translation_scale_handle is None:
            raise ValueError("Scaling controls not initialized")
        return self._translation_scale_handle.value

    def get_rotation_scale(self) -> float:
        """Get rotation scale value from GUI.

        Returns:
            Rotation scale value

        Raises:
            ValueError: If scaling controls not initialized
        """
        if self._rotation_scale_handle is None:
            raise ValueError("Scaling controls not initialized")
        return self._rotation_scale_handle.value

    def get_brain_control_armed(self, *, key: str = "right") -> bool:
        """Get whether brain control is ARMED from GUI."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return controls["arm_dropdown"].value == "ARMED"

    def get_brain_diameter(self, *, key: str = "right") -> float:
        """Get brain control diameter from GUI."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return float(controls["diameter_handle"].value)

    def get_brain_joint_offset_norms(self, *, key: str = "right") -> np.ndarray:
        """Get per-joint normalized offset sliders (shape (6,))."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        handles = controls["joint_offset_handles"]
        return np.array([h.value for h in handles], dtype=np.float64)

    def zero_brain_joint_offsets(self, *, key: str = "right") -> None:
        """Zero all joint offset sliders in the GUI."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            return
        for h in controls["joint_offset_handles"]:
            h.value = 0.0

    def get_brain_clamp_mode(self, *, key: str = "right") -> str:
        """Get clamp mode ("off" or "soft_damp")."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return str(controls["clamp_mode_dropdown"].value)

    def get_brain_active_group(self, *, key: str = "right") -> str:
        """Get active joint group selection."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return str(controls["active_joints_dropdown"].value)

    def get_brain_selected_alpha(self, *, key: str = "right") -> float:
        """Get alpha for selected joints."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return float(controls["selected_alpha_handle"].value)

    def get_brain_unselected_alpha(self, *, key: str = "right") -> float:
        """Get alpha for unselected joints."""
        controls = self._brain_controls_by_key.get(key)
        if controls is None:
            raise ValueError("Brain control controls not initialized")
        return float(controls["unselected_alpha_handle"].value)

    def set_brain_zero_offsets_callback(
        self, callback: Callable[[], Any], *, key: str = "right"
    ) -> None:
        """Set callback for 'Zero Offsets' button."""
        controls = self._brain_controls_by_key.get(key)
        if controls is not None:
            controls["zero_offsets_button"].on_click(lambda _: callback())

    def set_brain_capture_baseline_callback(
        self, callback: Callable[[], Any], *, key: str = "right"
    ) -> None:
        """Set callback for 'Capture Baseline Now' button."""
        controls = self._brain_controls_by_key.get(key)
        if controls is not None:
            controls["capture_baseline_button"].on_click(lambda _: callback())

    def set_grip_value(self, value: float) -> None:
        """Set grip value display.

        Args:
            value: Grip value (0.0-1.0)

        Raises:
            ValueError: If grip value control not initialized
        """
        if self._grip_value_handle is None:
            raise ValueError("Grip value control not initialized")
        self._grip_value_handle.value = value

    def set_trigger_value(self, value: float) -> None:
        """Set trigger value display.

        Args:
            value: Trigger value (0.0-1.0)

        Raises:
            ValueError: If trigger value control not initialized
        """
        if self._trigger_value_handle is None:
            raise ValueError("Trigger value control not initialized")
        self._trigger_value_handle.value = value

    def set_joint_angles_text(self, text: str) -> None:
        """Set joint angles text display.

        Args:
            text: Text to display

        Raises:
            ValueError: If joint angles control not initialized
        """
        if self._joint_angles_handle is None:
            raise ValueError("Joint angles control not initialized")
        self._joint_angles_handle.value = text

    def update_ghost_robot_pose(self, joint_config: np.ndarray) -> None:
        """Update ghost robot visualization from joint configuration.

        Args:
            joint_config: Joint angles in radians
        """
        if self._ghost_robot_urdf is not None:
            self._ghost_robot_urdf.update_cfg(joint_config)

    def update_ghost_robot_visibility(self, flag: bool) -> None:
        """Update ghost robot visibility.

        Args:
            flag: Whether to show the ghost robot
        """
        if self._ghost_robot_urdf is not None:
            self._ghost_robot_urdf.show_visual = flag

    def add_policy_controls(
        self,
        initial_prediction_ratio: float = 0.8,
        initial_policy_rate: float = 200.0,
        initial_robot_rate: float = 200.0,
        initial_execution_mode: str = "targeting_time",
    ) -> None:
        """Add policy-related GUI controls.

        Args:
            initial_prediction_ratio: Initial prediction horizon execution ratio (0.0-1.0)
            initial_policy_rate: Initial policy execution rate in Hz
            initial_robot_rate: Initial robot rate in Hz
            initial_execution_mode: Initial execution mode ("targeting_time" or "targeting_pose")
        """
        self._policy_status_handle = self._server.gui.add_text(
            "Policy Status", "Ready - Press Right Joystick to get prediction"
        )

        self._prediction_ratio_handle = self._server.gui.add_number(
            "Prediction Horizon Execution Ratio",
            initial_prediction_ratio,
            min=0.0,
            max=1.0,
            step=0.01,
        )

        self._policy_execution_rate_handle = self._server.gui.add_number(
            "Policy Execution Rate",
            initial_policy_rate,
            min=1.0,
            max=200.0,
            step=1.0,
        )

        self._robot_rate_handle = self._server.gui.add_number(
            "Robot Rate",
            initial_robot_rate,
            min=1.0,
            max=200.0,
            step=1.0,
        )

        self._execution_mode_dropdown = self._server.gui.add_dropdown(
            "Execution Mode",
            options=["targeting_time", "targeting_pose"],
            initial_value=initial_execution_mode,
        )

    def add_policy_buttons(self) -> None:
        """Add policy control buttons."""
        self._run_policy_button = self._server.gui.add_button("Run Policy")
        self._start_policy_execution_button = self._server.gui.add_button(
            "Start Policy Execution"
        )
        self._run_and_start_policy_execution_button = self._server.gui.add_button(
            "Run and Execute Policy"
        )
        self._play_policy_button = self._server.gui.add_button("Play Policy")

    def update_policy_status(self, status: str) -> None:
        """Update policy status display.

        Args:
            status: Status string to display
        """
        if self._policy_status_handle is not None:
            self._policy_status_handle.value = status

    def get_prediction_ratio(self) -> float:
        """Get prediction horizon execution ratio from GUI.

        Returns:
            Prediction ratio (0.0-1.0)

        Raises:
            ValueError: If policy controls not initialized
        """
        if self._prediction_ratio_handle is None:
            raise ValueError("Policy controls not initialized")
        return self._prediction_ratio_handle.value

    def get_policy_execution_rate(self) -> float:
        """Get policy execution rate from GUI.

        Returns:
            Policy execution rate in Hz

        Raises:
            ValueError: If policy controls not initialized
        """
        if self._policy_execution_rate_handle is None:
            raise ValueError("Policy controls not initialized")
        return self._policy_execution_rate_handle.value

    def get_robot_rate(self) -> float:
        """Get robot rate from GUI.

        Returns:
            Robot rate in Hz

        Raises:
            ValueError: If policy controls not initialized
        """
        if self._robot_rate_handle is None:
            raise ValueError("Policy controls not initialized")
        return self._robot_rate_handle.value

    def get_execution_mode(self) -> str:
        """Get execution mode from GUI.

        Returns:
            Execution mode string ("targeting_time" or "targeting_pose")

        Raises:
            ValueError: If policy controls not initialized
        """
        if self._execution_mode_dropdown is None:
            raise ValueError("Policy controls not initialized")
        return self._execution_mode_dropdown.value

    def get_ghost_robot_visibility(self) -> bool:
        """Get ghost robot visibility.

        Returns:
            Whether the ghost robot is visible
        """
        if self._ghost_robot_urdf is not None:
            return self._ghost_robot_urdf.show_visual
        return False

    def set_run_policy_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Run Policy button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._run_policy_button is not None:
            self._run_policy_button.on_click(lambda _: callback())

    def set_start_policy_execution_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Execute Policy button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._start_policy_execution_button is not None:
            self._start_policy_execution_button.on_click(lambda _: callback())

    def set_run_and_start_policy_execution_callback(
        self, callback: Callable[[], Any]
    ) -> None:
        """Set callback for Run and Execute Policy button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._run_and_start_policy_execution_button is not None:
            self._run_and_start_policy_execution_button.on_click(lambda _: callback())

    def set_play_policy_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Play Policy button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._play_policy_button is not None:
            self._play_policy_button.on_click(lambda _: callback())

    def set_execution_mode_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for execution mode dropdown.

        Args:
            callback: Callback function to call when dropdown value changes
        """
        if self._execution_mode_dropdown is not None:
            self._execution_mode_dropdown.on_update(lambda _: callback())

    def set_enable_robot_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Enable Robot button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._enable_robot_handle is not None:
            self._enable_robot_handle.on_click(lambda _: callback())

    def set_disable_robot_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Disable Robot button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._disable_robot_handle is not None:
            self._disable_robot_handle.on_click(lambda _: callback())

    def set_emergency_stop_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Emergency Stop button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._emergency_stop_handle is not None:
            self._emergency_stop_handle.on_click(lambda _: callback())

    def set_go_home_callback(self, callback: Callable[[], Any]) -> None:
        """Set callback for Go Home button.

        Args:
            callback: Callback function to call when button is clicked
        """
        if self._go_home_button is not None:
            self._go_home_button.on_click(lambda _: callback())

    def set_run_policy_button_disabled(self, disabled: bool) -> None:
        """Set Run Policy button disabled state.

        Args:
            disabled: Whether button should be disabled
        """
        if self._run_policy_button is not None:
            self._run_policy_button.disabled = disabled

    def set_start_policy_execution_button_disabled(self, disabled: bool) -> None:
        """Set Execute Policy button disabled state.

        Args:
            disabled: Whether button should be disabled
        """
        if self._start_policy_execution_button is not None:
            self._start_policy_execution_button.disabled = disabled

    def set_run_and_start_policy_execution_button_disabled(
        self, disabled: bool
    ) -> None:
        """Set Run and Execute Policy button disabled state.

        Args:
            disabled: Whether button should be disabled
        """
        if self._run_and_start_policy_execution_button is not None:
            self._run_and_start_policy_execution_button.disabled = disabled

    def set_play_policy_button_disabled(self, disabled: bool) -> None:
        """Set Play Policy button disabled state.

        Args:
            disabled: Whether button should be disabled
        """
        if self._play_policy_button is not None:
            self._play_policy_button.disabled = disabled

    def update_play_policy_button_status(self, active: bool) -> None:
        """Update play policy button label based on active state.

        Args:
            active: Whether continuous play is currently active
        """
        if self._play_policy_button is not None:
            self._play_policy_button.label = "Stop Policy" if active else "Play Policy"

    def stop(self) -> None:
        """Stop the visualizer server."""
        self._server.stop()
