"""Helpers to integrate 'Final Brain Control' GUI with DataManager.

These functions keep entrypoints (example scripts) thin and consistent.
"""

from __future__ import annotations

from typing import Callable

from arms.data_manager import DataManager
from arms.robot_visualizer import RobotVisualizer


def register_brain_control_callbacks(
    *,
    data_manager: DataManager,
    visualizer: RobotVisualizer,
    hand: str = "right",
    key: str = "right",
) -> None:
    """Wire GUI buttons to DataManager and GUI state."""

    def _on_zero_offsets() -> None:
        visualizer.zero_brain_joint_offsets(key=key)

    def _on_capture_baseline() -> None:
        ok = data_manager.capture_brain_baseline(hand)
        if not ok:
            print(f"Cannot capture {hand} baseline yet (no joint feedback available)")

    visualizer.set_brain_zero_offsets_callback(_on_zero_offsets, key=key)
    visualizer.set_brain_capture_baseline_callback(_on_capture_baseline, key=key)


def sync_brain_control_from_gui(
    *,
    data_manager: DataManager,
    visualizer: RobotVisualizer,
    hand: str = "right",
    key: str = "right",
    prev_armed: bool,
    on_arm_warn: Callable[[str], None] | None = None,
) -> bool:
    """Sync brain-control GUI values into DataManager for a given hand.

    Returns:
        The current armed state (to store as next prev_armed).
    """
    brain_armed = visualizer.get_brain_control_armed(key=key)
    clamp_mode = visualizer.get_brain_clamp_mode(key=key)
    if clamp_mode not in ("off", "soft_damp"):
        clamp_mode = "off"

    data_manager.set_brain_control_params(
        armed=brain_armed,
        diameter=visualizer.get_brain_diameter(key=key),
        slider_norm=visualizer.get_brain_joint_offset_norms(key=key),
        clamp_mode=clamp_mode,  # type: ignore[arg-type]
        active_group=visualizer.get_brain_active_group(key=key),
        selected_alpha=visualizer.get_brain_selected_alpha(key=key),
        unselected_alpha=visualizer.get_brain_unselected_alpha(key=key),
        hand=hand,
    )

    if brain_armed and (not prev_armed):
        if not data_manager.capture_brain_baseline(hand):
            if on_arm_warn is not None:
                on_arm_warn(hand)
            else:
                print(
                    f"Brain control armed ({hand}) but baseline not captured yet (waiting for feedback)"
                )

    return brain_armed

