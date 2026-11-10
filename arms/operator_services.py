"""Shared state for the teleop process.

The teleop runtime expects a small "services" object that owns the shared
``DataManager`` and a stop signal. In this kit nothing else lives in the teleop
process: the cameras run in their own process (``python -m arms.cameras``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from arms.data_manager import DataManager


@dataclass
class OperatorServicesConfig:
    """Which arms this process drives."""

    active_hands: tuple[str, ...] = ("right", "left")


class OperatorServices:
    """Own the shared ``DataManager`` and the process-wide stop signal."""

    def __init__(
        self,
        config: OperatorServicesConfig,
        *,
        data_manager: DataManager | None = None,
    ) -> None:
        self.config = config
        self.data_manager = data_manager or DataManager()
        self.stop_event = threading.Event()
        self.service_errors: list[str] = []

    def start(self) -> None:
        """Clear the stop signal before the runtime starts."""
        self.stop_event.clear()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Raise the stop signal."""
        del timeout_s  # nothing here blocks
        self.stop_event.set()

    def attach_robot_controllers(
        self,
        robot_controllers: dict[str, Any],
        *,
        active_hands: tuple[str, ...] | None = None,
        dashboard_ik_solvers: dict[str, Any] | None = None,
    ) -> None:
        """Called by the runtime once the arms are up. Nothing to attach here."""

    def detach_robot_controllers(self) -> None:
        """Called by the runtime before the arms go away. Nothing to detach."""


__all__ = ["OperatorServices", "OperatorServicesConfig"]
