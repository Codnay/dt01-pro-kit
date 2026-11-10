from __future__ import annotations

import threading
import unittest

from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

# DataManager imports one_euro_filter, which imports scipy Rotation.
# Provide a tiny stub so tests can run in environments without scipy.
if "scipy.spatial.transform" not in sys.modules:
    scipy_mod = types.ModuleType("scipy")
    spatial_mod = types.ModuleType("scipy.spatial")
    transform_mod = types.ModuleType("scipy.spatial.transform")

    class _DummyRotation:
        @staticmethod
        def from_matrix(_matrix):
            return _DummyRotation()

        @staticmethod
        def from_quat(_quat):
            return _DummyRotation()

        def as_quat(self):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    transform_mod.Rotation = _DummyRotation
    spatial_mod.transform = transform_mod
    scipy_mod.spatial = spatial_mod

    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.spatial", spatial_mod)
    sys.modules.setdefault("scipy.spatial.transform", transform_mod)

from arms.data_manager import DataManager


class DataManagerCameraTests(unittest.TestCase):
    def test_named_camera_set_get(self) -> None:
        dm = DataManager()
        rgb = np.full((4, 5, 3), 7, dtype=np.uint8)
        depth = np.full((4, 5), 320, dtype=np.uint16)

        dm.set_camera_rgb("front", rgb, timestamp_ns=10)
        dm.set_camera_depth("front", depth, timestamp_ns=20)
        frame = dm.get_camera_frame("front")

        self.assertEqual(frame.rgb_timestamp_ns, 10)
        self.assertEqual(frame.depth_timestamp_ns, 20)
        self.assertTrue(np.array_equal(frame.rgb_image, rgb))
        self.assertTrue(np.array_equal(frame.depth_image_mm, depth))
        self.assertEqual(dm.get_camera_names(), ["front"])

    def test_backward_compatible_rgb_path(self) -> None:
        dm = DataManager()
        rgb = np.full((3, 3, 3), 11, dtype=np.uint8)
        dm.set_rgb_image(rgb)
        fetched = dm.get_rgb_image()
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertTrue(np.array_equal(fetched, rgb))
        self.assertEqual(dm.get_camera_names(), ["rgb"])

    def test_joint_telemetry_stream_rate_config(self) -> None:
        dm = DataManager()

        self.assertFalse(dm.joint_telemetry_streams_enabled())
        dm.set_joint_telemetry_streams_enabled(True, max_hz=10.0)
        self.assertTrue(dm.joint_telemetry_streams_enabled())
        self.assertAlmostEqual(dm.joint_telemetry_min_period_s(), 0.1)

        dm.set_joint_telemetry_streams_enabled(False)
        self.assertFalse(dm.joint_telemetry_streams_enabled())
        self.assertAlmostEqual(dm.joint_telemetry_min_period_s(), 0.1)

    def test_thread_safety_for_multi_camera_updates(self) -> None:
        dm = DataManager()
        stop = threading.Event()

        def writer(name: str, value: int) -> None:
            rgb = np.full((8, 8, 3), value, dtype=np.uint8)
            depth = np.full((8, 8), value * 10, dtype=np.uint16)
            while not stop.is_set():
                dm.set_camera_rgb(name, rgb)
                dm.set_camera_depth(name, depth)

        t1 = threading.Thread(target=writer, args=("front", 1), daemon=True)
        t2 = threading.Thread(target=writer, args=("wrist", 2), daemon=True)
        t1.start()
        t2.start()

        for _ in range(100):
            frames = dm.get_all_camera_frames()
            for cam in ("front", "wrist"):
                self.assertIn(cam, frames)
                rgb = frames[cam].rgb_image
                depth = frames[cam].depth_image_mm
                self.assertIsNotNone(rgb)
                self.assertIsNotNone(depth)

        stop.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
