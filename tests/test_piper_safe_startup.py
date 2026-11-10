#!/usr/bin/env python3
"""Safe startup/resume tests for PiperController."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextlib.contextmanager
def _capture_piper_logs(level: int = logging.DEBUG):
    """Capture ``piper`` logger output as text.

    Startup/diagnostic messages were migrated from ``print`` to structured
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


class _JointState:
    def __init__(self, joints_deg: list[float]) -> None:
        for idx, value in enumerate(joints_deg, start=1):
            setattr(self, f"joint_{idx}", float(value) * 1000.0)


class _JointMsg:
    Hz = 100.0

    def __init__(
        self, joints_deg: list[float], *, timestamp: float | None = None
    ) -> None:
        self.joint_state = _JointState(joints_deg)
        self.time_stamp = time.time() if timestamp is None else timestamp


class _GripperState:
    grippers_angle = 53800.0


class _GripperMsg:
    Hz = 100.0
    gripper_state = _GripperState()


class _EndPose:
    X_axis = 0.0
    Y_axis = 0.0
    Z_axis = 0.0
    RX_axis = 0.0
    RY_axis = 0.0
    RZ_axis = 0.0


class _EndPoseMsg:
    Hz = 100.0
    end_pose = _EndPose()


class _FakeCanBus:
    def is_can_bus_ok(self) -> int:
        return 100019


class _FakeRotation:
    @staticmethod
    def from_euler(
        _seq: str, _angles: np.ndarray | list[float], *, degrees: bool = False
    ) -> "_FakeRotation":
        return _FakeRotation()

    @staticmethod
    def from_matrix(_matrix: np.ndarray) -> "_FakeRotation":
        return _FakeRotation()

    def as_matrix(self) -> np.ndarray:
        return np.eye(3)

    def as_euler(self, _seq: str, *, degrees: bool = False) -> np.ndarray:
        return np.zeros(3)


class _FakePiper:
    instances: list["_FakePiper"] = []
    joint_messages: list[_JointMsg] = []
    joint_limits_deg: list[tuple[float, float]] = [(-180.0, 180.0)] * 6

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.joint_messages = []
        cls.joint_limits_deg = [(-180.0, 180.0)] * 6

    def __init__(
        self,
        can_interface: str,
        *,
        start_sdk_joint_limit: bool = True,
        start_sdk_gripper_limit: bool = True,
    ) -> None:
        self.can_interface = can_interface
        self.connected = False
        self.enable_calls = 0
        self.disable_calls = 0
        self.disconnect_calls = 0
        self.joint_read_index = 0
        self.motionctrl2_calls: list[tuple[int, int, int, int]] = []
        self.motionctrl1_calls: list[tuple[int, int, int]] = []
        self.jointctrl_calls: list[tuple[int, int, int, int, int, int]] = []
        self.gripperctrl_calls: list[tuple[int, int, int, int]] = []
        self.endposectrl_calls: list[tuple[int, int, int, int, int, int]] = []
        _FakePiper.instances.append(self)

    def ConnectPort(self) -> bool:
        self.connected = True
        return True

    def DisconnectPort(self) -> bool:
        self.disconnect_calls += 1
        self.connected = False
        return True

    def get_connect_status(self) -> bool:
        return self.connected

    def SetSDKGripperRangeParam(self, _min: float, _max: float) -> bool:
        return True

    def GetSDKJointLimitParam(self, joint_name: str) -> list[float]:
        idx = int(joint_name[1:]) - 1
        lower_deg, upper_deg = self.joint_limits_deg[idx]
        return [float(np.radians(lower_deg)), float(np.radians(upper_deg))]

    def GetArmJointMsgs(self) -> _JointMsg:
        if not self.joint_messages:
            return _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        idx = min(self.joint_read_index, len(self.joint_messages) - 1)
        self.joint_read_index += 1
        return self.joint_messages[idx]

    def GetArmGripperMsgs(self) -> _GripperMsg:
        return _GripperMsg()

    def GetArmEndPoseMsgs(self) -> _EndPoseMsg:
        return _EndPoseMsg()

    def GetArmStatus(self) -> dict[str, str]:
        return {"status": "ok"}

    def GetArmEnableStatus(self) -> dict[str, bool]:
        return {"enabled": self.enable_calls > self.disable_calls}

    def GetDriverStates(self) -> dict[str, str]:
        return {"drivers": "ok"}

    def GetArmLowSpdInfoMsgs(self) -> dict[str, str]:
        return {"low_speed": "ok"}

    def GetCanBus(self) -> _FakeCanBus:
        return _FakeCanBus()

    def EnablePiper(self) -> bool:
        self.enable_calls += 1
        return True

    def DisablePiper(self) -> bool:
        self.disable_calls += 1
        return True

    def MotionCtrl_2(self, *args: int) -> bool:
        self.motionctrl2_calls.append(tuple(args))
        return True

    def MotionCtrl_1(self, *args: int) -> bool:
        self.motionctrl1_calls.append(tuple(args))
        return True

    def JointCtrl(self, *args: int) -> bool:
        self.jointctrl_calls.append(tuple(args))
        return True

    def EndPoseCtrl(self, *args: int) -> bool:
        self.endposectrl_calls.append(tuple(args))
        return True

    def GripperCtrl(self, *args: int) -> bool:
        self.gripperctrl_calls.append(tuple(args))
        return True


class PiperSafeStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakePiper.reset()
        self._old_piper_sdk = sys.modules.get("piper_sdk")
        self._old_piper_controller = sys.modules.pop("arms.piper_controller", None)
        self._old_scipy_modules = {
            name: sys.modules.get(name)
            for name in (
                "scipy",
                "scipy.spatial",
                "scipy.spatial.transform",
            )
        }

        fake_sdk = types.ModuleType("piper_sdk")
        fake_sdk.C_PiperInterface_V2 = _FakePiper
        sys.modules["piper_sdk"] = fake_sdk

        fake_scipy = types.ModuleType("scipy")
        fake_spatial = types.ModuleType("scipy.spatial")
        fake_transform = types.ModuleType("scipy.spatial.transform")
        fake_transform.Rotation = _FakeRotation
        fake_spatial.transform = fake_transform
        fake_scipy.spatial = fake_spatial
        sys.modules["scipy"] = fake_scipy
        sys.modules["scipy.spatial"] = fake_spatial
        sys.modules["scipy.spatial.transform"] = fake_transform

        self._env_patch = patch.dict(
            os.environ,
            {
                "PIPER_STARTUP_FEEDBACK_SAMPLE_INTERVAL_S": "0",
                "PIPER_CAN_RECOVER_ON_INIT": "0",
                "PIPER_CAN_RECOVER_ON_RESUME": "0",
            },
        )
        self._env_patch.start()
        self.piper_controller = importlib.import_module("arms.piper_controller")

    def tearDown(self) -> None:
        self._env_patch.stop()
        sys.modules.pop("arms.piper_controller", None)
        if self._old_piper_controller is not None:
            sys.modules["arms.piper_controller"] = self._old_piper_controller

        sys.modules.pop("piper_sdk", None)
        if self._old_piper_sdk is not None:
            sys.modules["piper_sdk"] = self._old_piper_sdk

        for name in ("scipy.spatial.transform", "scipy.spatial", "scipy"):
            sys.modules.pop(name, None)
        for name, old_module in self._old_scipy_modules.items():
            if old_module is not None:
                sys.modules[name] = old_module

    def _make_controller(self, **kwargs):
        with _capture_piper_logs() as output:
            controller = self.piper_controller.PiperController(**kwargs)
        return controller, _FakePiper.instances[-1], output.getvalue()

    def test_agilex_mode_change_is_sent_even_at_same_speed(self):
        controller, sdk, _ = self._make_controller()
        self.addCleanup(self._cleanup_controller, controller)
        controller._ensure_motionctrl(1, 50, mit_mode=0xAD)
        self.assertEqual(sdk.motionctrl2_calls[-1], (1, 1, 50, 0xAD))
        controller._ensure_motionctrl(1, 50)
        self.assertEqual(sdk.motionctrl2_calls[-1], (1, 1, 50, 0))

    def test_agilex_target_flag_clears_for_normal_commands_and_hold(self):
        controller, sdk, _ = self._make_controller()
        self.addCleanup(self._cleanup_controller, controller)
        q = np.zeros(6)
        controller.set_target_joint_angles(q, use_agilex_teleop=True)
        self.assertGreater(controller._agilex_teleop_until, time.monotonic())
        controller.set_target_joint_angles(q)
        self.assertEqual(controller._agilex_teleop_until, 0)
        controller.set_target_joint_angles(q, use_agilex_teleop=True)
        controller._set_hold_joint_target_angles(q)
        self.assertEqual(controller._agilex_teleop_until, 0)

    def test_teleop_release_preserves_sent_target_and_mode(self):
        import threading
        controller = self.piper_controller.PiperController.__new__(self.piper_controller.PiperController)
        controller.cleanup = lambda: None
        controller.position_lock = threading.Lock()
        controller._command_handoff_lock = threading.RLock()
        controller.is_robot_enabled = lambda: True
        controller.get_control_mode = lambda: controller.ControlMode.JOINT_SPACE
        controller._target_joint_angles = np.ones(6) * 8  # queued, not transmitted
        controller._last_sent_joint_angles = np.ones(6) * 4
        controller._last_sent_joint_mit_mode = 0xAD
        controller._agilex_teleop_until = time.monotonic() + .1
        controller._active_motion_speed_cap_percent = None
        np.testing.assert_array_equal(controller.hold_teleop_position(), np.ones(6) * 4)
        self.assertEqual(controller._teleop_joint_mode_locked(), 0xAD)
        controller._agilex_teleop_until = time.monotonic() - 1.
        controller._agilex_teleop_holding = False
        controller._target_joint_angles = np.ones(6) * 9
        self.assertEqual(controller._teleop_joint_mode_locked(), 0xAD)
        np.testing.assert_array_equal(controller._target_joint_angles, np.ones(6) * 4)
        controller.is_robot_enabled = lambda: False
        self.assertIsNone(controller.hold_teleop_position())

    def _cleanup_controller(self, controller) -> None:
        with _capture_piper_logs():
            if controller.is_robot_enabled():
                controller._disable_robot(timeout_s=0.01)
            controller._set_robot_status_enabled(False)
            controller.cleanup()

    def test_constructor_does_not_enable_or_sync_targets(self) -> None:
        _FakePiper.joint_messages = [
            _JointMsg([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        ]

        controller, fake, startup_output = self._make_controller()
        self.addCleanup(self._cleanup_controller, controller)

        self.assertEqual(fake.enable_calls, 0)
        self.assertFalse(controller.is_robot_enabled())
        np.testing.assert_allclose(controller.get_target_joint_angles(), np.zeros(6))
        self.assertIn("Initial hardware feedback (startup)", startup_output)
        self.assertIn("joint_angles_deg", startup_output)
        self.assertEqual(fake.motionctrl2_calls, [])
        self.assertEqual(fake.jointctrl_calls, [])
        self.assertEqual(fake.gripperctrl_calls, [])

    def test_effective_joint_limits_apply_j6_margin_and_return_copy(self) -> None:
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 5 + [(-120.0, 120.0)]

        controller, _fake, _startup_output = self._make_controller(
            joint_limit_margin_deg={"joint6": 20.0}
        )
        self.addCleanup(self._cleanup_controller, controller)

        expected = np.array(
            [
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-100.0, 100.0],
            ],
            dtype=np.float64,
        )
        limits = controller.get_effective_joint_limits_deg()
        np.testing.assert_allclose(limits, expected)

        limits[5, 1] = 0.0
        np.testing.assert_allclose(controller.get_effective_joint_limits_deg(), expected)

    def test_disabled_control_loop_sends_no_motion_commands(self) -> None:
        controller, fake, _startup_output = self._make_controller(robot_rate=500.0)
        self.addCleanup(self._cleanup_controller, controller)

        controller.start_control_loop()
        time.sleep(0.03)
        controller.stop_control_loop()

        self.assertFalse(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 0)
        self.assertEqual(fake.motionctrl2_calls, [])
        self.assertEqual(fake.jointctrl_calls, [])
        self.assertEqual(fake.gripperctrl_calls, [])

    def test_bad_resume_feedback_blocks_enable(self) -> None:
        _FakePiper.joint_messages = [
            _JointMsg([500.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([500.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([500.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([500.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ]
        controller, fake, _startup_output = self._make_controller()
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs() as output:
            ok = controller.resume_robot(force=True)

        self.assertFalse(ok)
        self.assertFalse(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 0)
        self.assertIn("Refusing to enable robot", output.getvalue())

    def test_resume_allows_feedback_at_software_limit(self) -> None:
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        _FakePiper.joint_messages = [
            _JointMsg([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        controller, fake, _startup_output = self._make_controller(
            joint_limit_margin_deg=3.0
        )
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs():
            ok = controller.resume_robot(force=True)

        self.assertTrue(ok)
        self.assertTrue(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 1)
        np.testing.assert_allclose(
            controller.get_target_joint_angles(),
            np.array([-7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_resume_allows_feedback_at_hardware_limit(self) -> None:
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        _FakePiper.joint_messages = [
            _JointMsg([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        controller, fake, _startup_output = self._make_controller(
            joint_limit_margin_deg=3.0
        )
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs() as output:
            ok = controller.resume_robot(force=True)

        self.assertTrue(ok)
        self.assertTrue(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 1)

    def test_resume_blocks_feedback_outside_hardware_limit(self) -> None:
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        _FakePiper.joint_messages = [
            _JointMsg([-10.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _JointMsg([-10.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        controller, fake, _startup_output = self._make_controller(
            joint_limit_margin_deg=3.0
        )
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs() as output:
            ok = controller.resume_robot(force=True)

        self.assertFalse(ok)
        self.assertFalse(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 0)
        self.assertIn("outside hardware limit", output.getvalue())

    def test_stale_resume_feedback_blocks_enable(self) -> None:
        stale_ts = time.time() - 5.0
        _FakePiper.joint_messages = [
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], timestamp=stale_ts),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], timestamp=stale_ts),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], timestamp=stale_ts),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], timestamp=stale_ts),
        ]
        controller, fake, _startup_output = self._make_controller()
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs() as output:
            ok = controller.resume_robot(force=True)

        self.assertFalse(ok)
        self.assertFalse(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 0)
        self.assertIn("stale feedback age", output.getvalue())

    def test_valid_resume_locks_target_enables_and_streams(self) -> None:
        _FakePiper.joint_messages = [
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ]
        controller, fake, _startup_output = self._make_controller(robot_rate=500.0)
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs():
            ok = controller.resume_robot(force=True)
            controller.start_control_loop()
            time.sleep(0.03)
            controller.stop_control_loop()

        self.assertTrue(ok)
        self.assertTrue(controller.is_robot_enabled())
        self.assertEqual(fake.enable_calls, 1)
        np.testing.assert_allclose(
            controller.get_target_joint_angles(),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        self.assertGreater(len(fake.motionctrl2_calls), 0)
        self.assertGreater(len(fake.jointctrl_calls), 0)
        self.assertEqual(
            fake.jointctrl_calls[0],
            (1000, 2000, 3000, 4000, 5000, 6000),
        )

    def _stop_test_controller(self, **kwargs):
        controller, fake, _ = self._make_controller(**kwargs)
        # Test fake teardown without the hardware's five-second descent delay.
        controller._uninterruptible_sleep = lambda _: False
        self.addCleanup(self._cleanup_controller, controller)
        return controller, fake

    @staticmethod
    def _teach_status(mode, *, timestamp=None):
        return types.SimpleNamespace(
            arm_status=types.SimpleNamespace(ctrl_mode=mode, teach_status=1),
            time_stamp=time.time() if timestamp is None else timestamp,
        )

    def _control_tick(self, controller):
        controller.running.set()
        with patch.object(self.piper_controller.time, "sleep",
                          side_effect=lambda _: controller.running.clear()):
            controller.control_loop()

    def test_teach_exit_holds_physical_pose_and_rejects_stale_software_targets(self):
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        controller, sdk = self._stop_test_controller(joint_limit_margin_deg=3)
        controller._set_robot_status_enabled(True)
        controller.set_target_joint_angles(np.ones(6))
        mode = [2]
        actual = [9.0] * 6  # Outside the soft margin but physically valid.
        with patch.object(sdk, "GetArmStatus", side_effect=lambda: self._teach_status(mode[0])), \
                patch.object(sdk, "GetArmJointMsgs", side_effect=lambda: _JointMsg(actual)):
            self._control_tick(controller)
            self.assertTrue(controller.is_stop_hold_latched())
            self.assertEqual(sdk.jointctrl_calls, [])
            self.assertEqual(sdk.motionctrl2_calls, [])
            controller.set_target_joint_angles(np.full(6, -5.0))
            mode[0] = 1
            self._control_tick(controller)
            self.assertEqual(sdk.jointctrl_calls[-1], (9000,) * 6)
            self.assertEqual(sdk.motionctrl1_calls, [(0, 1, 0), (0, 3, 0)])
            # Repeated late targets, gripper writes and cleanup cannot replace
            # the fixed target, even if the measured arm subsequently drifts.
            actual[:] = [8.0] * 6
            controller.set_target_joint_angles(np.full(6, -5.0))
            controller._send_joint_command(np.full(6, -5.0))
            controller._send_end_effector_command(np.eye(4))
            controller._send_gripper_command(80)
            self._control_tick(controller)
            self.assertTrue(all(q == (9000,) * 6 for q in sdk.jointctrl_calls))
            self.assertEqual(sdk.gripperctrl_calls, [])
            self.assertEqual(sdk.endposectrl_calls, [])
            self.assertFalse(controller.resume_robot(force=True))
        self.assertEqual(sdk.enable_calls, 0)
        self.assertEqual(sdk.disable_calls, 0)

    def test_teach_is_detected_at_final_io_before_next_control_tick(self):
        for operation in ("JointCtrl", "MotionCtrl_2", "GripperCtrl", "EnablePiper",
                          "EndPoseCtrl", "JointConfig(clear_err)"):
            with self.subTest(operation=operation):
                controller, sdk = self._stop_test_controller()
                with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
                    write = unittest.mock.Mock()
                    self.assertIs(controller._sdk_call(operation, write),
                                  controller._sdk_error_sentinel)
                    write.assert_not_called()
                    self.assertTrue(controller.is_stop_hold_latched())

    def test_teach_reentry_discards_previous_stop_destination(self):
        controller, sdk = self._stop_test_controller()
        self.assertTrue(controller.stop_and_hold())
        sdk.jointctrl_calls.clear()
        mode = [2]
        with patch.object(sdk, "GetArmStatus", side_effect=lambda: self._teach_status(mode[0])):
            self._control_tick(controller)
            self.assertIsNone(controller._stop_hold_target)
            self.assertFalse(controller.stop_and_hold())
            self.assertEqual(sdk.jointctrl_calls, [])
            mode[0] = 1
            with patch.object(sdk, "GetArmJointMsgs", side_effect=lambda: _JointMsg([12.0] * 6)):
                self._control_tick(controller)
            self.assertEqual(sdk.jointctrl_calls, [(12000,) * 6])

    def test_teach_missing_or_stale_status_cannot_exit_or_enable(self):
        controller, sdk = self._stop_test_controller()
        with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
            self._control_tick(controller)
        for status in (None, {}, self._teach_status(float("nan")),
                       self._teach_status(1, timestamp=time.time() - 5),
                       types.SimpleNamespace(arm_status=types.SimpleNamespace(ctrl_mode=1)),
                       self._teach_status(99)):
            with self.subTest(status=status), patch.object(sdk, "GetArmStatus", return_value=status):
                self._control_tick(controller)
                self.assertTrue(controller._teach_mode_active)
                self.assertFalse(controller.resume_robot(force=True, clear_stop_hold=True))
        self.assertEqual(sdk.motionctrl2_calls, [])
        self.assertEqual(sdk.jointctrl_calls, [])
        self.assertEqual(sdk.enable_calls, 0)

    def test_teach_bad_exit_feedback_never_reuses_or_retries_a_target(self):
        for feedback in (None, _JointMsg([float("nan")] * 6),
                         _JointMsg([10000.0] * 6),
                         _JointMsg([12.0] * 6, timestamp=time.time() - 5)):
            with self.subTest(feedback=feedback):
                controller, sdk = self._stop_test_controller()
                with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
                    self._control_tick(controller)
                with patch.object(sdk, "GetArmStatus", side_effect=lambda: self._teach_status(1)):
                    with patch.object(sdk, "GetArmJointMsgs", return_value=feedback):
                        self._control_tick(controller)
                    # Feedback recovers, but failure must not turn into an
                    # automatic hold at an arbitrary later angle.
                    self._control_tick(controller)
                    self.assertTrue(controller.is_stop_hold_latched())
                    self.assertIsNone(controller._stop_hold_target)
                    self.assertEqual(sdk.jointctrl_calls, [])
                    self.assertEqual(sdk.enable_calls, 0)
                    self.assertEqual(sdk.disable_calls, 0)

    def test_teach_only_fresh_explicit_enable_can_resume(self):
        controller, sdk = self._stop_test_controller()
        token = controller.get_stop_hold_generation()
        with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
            self._control_tick(controller)
            self.assertFalse(controller.resume_robot(force=True, clear_stop_hold=True))
        with patch.object(sdk, "GetArmStatus", side_effect=lambda: self._teach_status(1)):
            self._control_tick(controller)
            self.assertFalse(controller.resume_robot(force=True, clear_stop_hold=True,
                                                     expected_stop_generation=token))
            self.assertTrue(controller.resume_robot(force=True, clear_stop_hold=True,
                expected_stop_generation=controller.get_stop_hold_generation()))
        self.assertFalse(controller.is_stop_hold_latched())
        np.testing.assert_allclose(controller.get_target_joint_angles(), np.arange(1, 7))

    def test_teach_is_detected_while_disabled_and_breaker_open(self):
        controller, sdk = self._stop_test_controller()
        controller._cb_state = "OPEN"
        with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
            self._control_tick(controller)
        self.assertTrue(controller.is_stop_hold_latched())
        self.assertEqual(sdk.jointctrl_calls, [])

    def test_teach_blocks_previous_stop_target_at_final_io(self):
        controller, sdk = self._stop_test_controller()
        controller.stop_and_hold()
        sdk.jointctrl_calls.clear()
        with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
            write = unittest.mock.Mock()
            controller._sdk_call("JointCtrl(stop_hold)", write)
            write.assert_not_called()
            self.assertIsNone(controller._stop_hold_target)

    def test_teach_detected_after_waiting_for_sdk_lock(self):
        controller, sdk = self._stop_test_controller()
        entered = threading.Event()
        write = unittest.mock.Mock()

        def send():
            entered.set()
            controller._sdk_call("JointCtrl", write)

        controller._sdk_lock.acquire()
        sender = threading.Thread(target=send)
        with patch.object(sdk, "GetArmStatus", return_value=self._teach_status(2)):
            sender.start()
            try:
                self.assertTrue(entered.wait(1))
            finally:
                controller._sdk_lock.release()
                sender.join(1)
        self.assertFalse(sender.is_alive())
        write.assert_not_called()
        self.assertTrue(controller.is_stop_hold_latched())

    def test_teach_cancels_enable_already_validating_feedback(self):
        controller, sdk = self._stop_test_controller()
        entered, release = threading.Event(), threading.Event()
        mode = [1]
        result = []

        def feedback(_reason):
            entered.set()
            self.assertTrue(release.wait(2))
            return np.arange(1.0, 7.0)

        with patch.object(sdk, "GetArmStatus", side_effect=lambda: self._teach_status(mode[0])), \
                patch.object(controller, "_read_validated_joint_feedback", side_effect=feedback):
            sender = threading.Thread(target=lambda: result.append(controller.resume_robot()))
            sender.start()
            try:
                self.assertTrue(entered.wait(1))
                mode[0] = 2
                self._control_tick(controller)
            finally:
                release.set()
                sender.join(1)
        self.assertEqual(result, [False])
        self.assertEqual(sdk.enable_calls, 0)
        self.assertTrue(controller.is_stop_hold_latched())

    def test_stop_attempts_pause_with_bad_encoder_and_fences_all_motion(self):
        for feedback in (None, _JointMsg([float("nan")] * 6),
                         _JointMsg([10000.0] * 6),
                         _JointMsg([12.0] * 6, timestamp=time.time() - 60)):
            with self.subTest(feedback=feedback):
                controller, fake = self._stop_test_controller(robot_rate=500)
                controller._set_robot_status_enabled(True)
                with patch.object(fake, "GetArmJointMsgs", return_value=feedback) as read:
                    self.assertFalse(controller.stop_and_hold())
                    read.assert_called_once()
                # Commands already copied by a control iteration are fenced too.
                controller._ensure_motionctrl(1, 100)
                controller._send_joint_command(np.ones(6) * 50)
                controller._send_end_effector_command(np.eye(4))
                controller._send_gripper_command(80)
                self.assertFalse(controller.move_to_home())
                self.assertFalse(controller.resume_robot(force=True))
        self.assertTrue(all(call == (0, 1, 0) for call in fake.motionctrl1_calls))
        self.assertEqual(fake.motionctrl2_calls, [])
        self.assertEqual(fake.jointctrl_calls, [])
        self.assertEqual(fake.endposectrl_calls, [])
        self.assertEqual(fake.gripperctrl_calls, [])
        self.assertEqual(fake.enable_calls, 0)
        self.assertEqual(fake.disable_calls, 0)

    def test_failed_stop_tx_stays_latched_and_retries_with_breaker_open(self):
        controller, fake = self._stop_test_controller()
        with patch.object(fake, "JointCtrl", return_value=False):
            self.assertFalse(controller.stop_and_hold())
        controller._cb_state = "OPEN"
        controller._last_stop_hold_tx = 0
        # One deterministic loop iteration, with the software-enabled flag False.
        with patch.object(self.piper_controller.time, "sleep",
                          side_effect=lambda _: controller.running.clear()):
            controller.control_loop()
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.motionctrl1_calls, [(0, 1, 0), (0, 3, 0)])
        self.assertEqual(fake.jointctrl_calls, [(1000, 2000, 3000, 4000, 5000, 6000)])

    def test_stop_replaces_preset_even_when_firmware_ignores_pause(self):
        controller, fake = self._stop_test_controller()
        controller._set_robot_status_enabled(True)
        actual = np.arange(1.0, 7.0)
        firmware_target = np.full(6, 50.0)

        def command(*raw):
            firmware_target[:] = np.asarray(raw) / 1000.0
            return None  # Installed SDK's normal return convention.

        with patch.object(fake, "GetArmJointMsgs", side_effect=lambda: _JointMsg(actual)), \
                patch.object(fake, "JointCtrl", side_effect=command), \
                patch.object(fake, "MotionCtrl_1", return_value=None):
            self.assertTrue(controller.stop_and_hold())
            np.testing.assert_array_equal(firmware_target, actual)
            # Late preset/teleop writes and cleanup holds cannot change it.
            controller.set_target_joint_angles(np.full(6, 40.0))
            controller._send_joint_command(np.full(6, 40.0))
            controller._send_end_effector_command(np.eye(4))
            self.assertTrue(controller.hold_current_position(allow_resume=False))
            self.assertTrue(controller.verify_stop_hold())
            np.testing.assert_array_equal(firmware_target, actual)
        self.assertEqual(fake.enable_calls, 0)
        self.assertEqual(fake.disable_calls, 0)

    def test_repeated_stop_never_chases_drifting_feedback(self):
        controller, fake = self._stop_test_controller()
        self.assertTrue(controller.stop_and_hold())
        captured = controller.get_target_joint_angles()
        with patch.object(fake, "GetArmJointMsgs", return_value=_JointMsg([10.0] * 6)):
            self.assertTrue(controller.stop_and_hold())
            self.assertTrue(controller.hold_current_position(allow_resume=False))
        np.testing.assert_array_equal(controller.get_target_joint_angles(), captured)
        self.assertTrue(all(call == tuple((captured * 1000).astype(int))
                            for call in fake.jointctrl_calls))

    def test_stop_preserves_mit_and_does_not_apply_soft_margin(self):
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        controller, fake = self._stop_test_controller(joint_limit_margin_deg=3)
        controller._ensure_motionctrl(1, 50, mit_mode=0xAD)

        read = controller._read_stop_hold_feedback

        def feedback():
            # A normal command copied before STOP finishes during capture.
            controller._ensure_motionctrl(1, 100, mit_mode=0)
            return read()

        with patch.object(controller, "_read_stop_hold_feedback", side_effect=feedback), \
                patch.object(fake, "GetArmJointMsgs", return_value=_JointMsg([9.0] * 6)):
            self.assertTrue(controller.stop_and_hold())
        self.assertEqual(fake.motionctrl2_calls[-1], (1, 1, 10, 0xAD))
        self.assertEqual(fake.jointctrl_calls[-1], (9000,) * 6)

    def test_stop_does_not_report_hold_for_ignored_commands_or_stale_feedback(self):
        controller, fake = self._stop_test_controller()
        self.assertTrue(controller.stop_and_hold())
        for read in (
            lambda: _JointMsg([10.0] * 6),  # Still travelling to the old preset.
            lambda: _JointMsg([1, 2, 3, 4, 5, 6], timestamp=time.time() - 1),
            lambda sample=_JointMsg([1, 2, 3, 4, 5, 6]): sample,  # Frozen cache.
        ):
            with patch.object(fake, "GetArmJointMsgs", side_effect=read):
                self.assertFalse(controller.verify_stop_hold(timeout_s=0.2))
        self.assertTrue(controller._stop_hold_latched.is_set())

    def test_stop_without_timestamp_never_commands_an_angle(self):
        controller, fake = self._stop_test_controller()
        raw = _JointMsg([1, 2, 3, 4, 5, 6])
        del raw.time_stamp
        with patch.object(fake, "GetArmJointMsgs", return_value=raw):
            self.assertFalse(controller.stop_and_hold())
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.jointctrl_calls, [])
        self.assertTrue(fake.motionctrl1_calls)

    def test_stop_cancels_inflight_enable_before_it_can_power_arm(self):
        controller, fake = self._stop_test_controller()
        entered, release = threading.Event(), threading.Event()
        result = []

        def feedback(_reason):
            entered.set()
            self.assertTrue(release.wait(2))
            return np.ones(6)

        with patch.object(controller, "_read_validated_joint_feedback", feedback):
            thread = threading.Thread(target=lambda: result.append(controller.resume_robot()))
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(controller.stop_and_hold())
            finally:
                release.set()
                thread.join(2)
        self.assertEqual(result, [False])
        self.assertEqual(fake.enable_calls, 0)
        self.assertEqual(fake.disable_calls, 0)

    def test_explicit_enable_discards_stopped_plan_and_old_targets(self):
        controller, fake = self._stop_test_controller()
        self.assertTrue(controller.stop_and_hold())
        controller.set_target_joint_angles(np.ones(6) * 50)
        controller.set_control_mode(controller.ControlMode.END_EFFECTOR)
        self.assertTrue(controller.resume_robot(force=True, clear_stop_hold=True))
        self.assertFalse(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.motionctrl1_calls, [(0, 1, 0), (0, 3, 0), (0, 3, 0)])
        self.assertEqual(controller.get_control_mode(), controller.ControlMode.JOINT_SPACE)
        np.testing.assert_allclose(controller.get_target_joint_angles(), np.arange(1, 7))

    def test_queued_enable_token_cannot_clear_newer_stop(self):
        controller, fake = self._stop_test_controller()
        token = controller.get_stop_hold_generation()
        controller.stop_and_hold()
        self.assertFalse(controller.resume_robot(
            force=True, clear_stop_hold=True, expected_stop_generation=token,
        ))
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.enable_calls, 0)

    def test_bad_feedback_cannot_release_stop_or_disable_powered_arm(self):
        controller, fake = self._stop_test_controller()
        controller._set_robot_status_enabled(True)
        controller.stop_and_hold()
        with patch.object(controller, "_read_validated_joint_feedback", return_value=None):
            self.assertFalse(controller.resume_robot(force=True, clear_stop_hold=True))
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.motionctrl1_calls, [(0, 1, 0), (0, 3, 0)])
        self.assertEqual(fake.enable_calls, 0)
        self.assertEqual(fake.disable_calls, 0)

    def test_bad_feedback_during_forced_resume_pauses_without_power_cut(self):
        controller, fake = self._stop_test_controller()
        controller._set_robot_status_enabled(True)
        with patch.object(controller, "_read_validated_joint_feedback", return_value=None):
            self.assertFalse(controller.resume_robot(force=True))
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertEqual(fake.motionctrl1_calls, [(0, 1, 0), (0, 3, 0)])
        self.assertEqual(fake.disable_calls, 0)

    def test_measured_hold_preserves_pose_outside_soft_margin(self):
        _FakePiper.joint_limits_deg = [(-10.0, 10.0)] * 6
        _FakePiper.joint_messages = [_JointMsg([9.0] * 6)]
        controller, fake = self._stop_test_controller(joint_limit_margin_deg=3)
        controller._set_robot_status_enabled(True)
        self.assertTrue(controller.hold_current_position(allow_resume=False))
        np.testing.assert_allclose(controller.get_target_joint_angles(), [9.0] * 6)

    def test_unusable_measured_hold_pauses_instead_of_leaving_home_active(self):
        controller, fake = self._stop_test_controller()
        controller._set_robot_status_enabled(True)
        with patch.object(fake, "GetArmJointMsgs", return_value=_JointMsg([float("nan")] * 6)):
            self.assertFalse(controller.hold_current_position(allow_resume=False))
        self.assertTrue(controller._stop_hold_latched.is_set())
        self.assertTrue(all(call == (0, 1, 0) for call in fake.motionctrl1_calls))

    def test_hold_current_position_caps_motion_speed_to_10_percent(self) -> None:
        _FakePiper.joint_messages = [
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ]
        with patch.dict(
            os.environ, {"PIPER_HOLD_CURRENT_POSITION_SPEED_PERCENT": "10"}
        ):
            controller, fake, _startup_output = self._make_controller(robot_rate=500.0)
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs():
            self.assertTrue(controller.resume_robot(force=True))
            controller.set_motion_speed_percent(100)
            self.assertTrue(controller.hold_current_position())
            fake.motionctrl2_calls.clear()
            controller.start_control_loop()
            time.sleep(0.03)
            controller.stop_control_loop()

        self.assertGreater(len(fake.motionctrl2_calls), 0)
        self.assertEqual(fake.motionctrl2_calls[0], (1, 1, 10, 0))

    def test_normal_joint_target_clears_hold_motion_speed_cap(self) -> None:
        _FakePiper.joint_messages = [
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _JointMsg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ]
        controller, fake, _startup_output = self._make_controller(robot_rate=500.0)
        self.addCleanup(self._cleanup_controller, controller)

        with _capture_piper_logs():
            self.assertTrue(controller.resume_robot(force=True))
            controller.set_motion_speed_percent(100)
            self.assertTrue(controller.hold_current_position())
            controller.start_control_loop()
            time.sleep(0.03)
            controller.stop_control_loop()

            fake.motionctrl2_calls.clear()
            controller.set_target_joint_angles(
                np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            )
            controller.start_control_loop()
            time.sleep(0.03)
            controller.stop_control_loop()

        self.assertGreater(len(fake.motionctrl2_calls), 0)
        self.assertEqual(fake.motionctrl2_calls[0], (1, 1, 100, 0))


if __name__ == "__main__":
    unittest.main()
