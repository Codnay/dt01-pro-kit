from __future__ import annotations

import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from arms.camera_config import parse_camera_specs


class CameraConfigTests(unittest.TestCase):
    def test_parse_valid_specs(self) -> None:
        specs = parse_camera_specs(["front:123", "wrist:456"])
        self.assertEqual([(s.name, s.serial) for s in specs], [("front", "123"), ("wrist", "456")])

    def test_parse_rejects_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            parse_camera_specs(["front123"])

    def test_parse_rejects_duplicate_name(self) -> None:
        with self.assertRaises(ValueError):
            parse_camera_specs(["front:123", "front:456"])

    def test_parse_rejects_duplicate_serial(self) -> None:
        with self.assertRaises(ValueError):
            parse_camera_specs(["front:123", "wrist:123"])


if __name__ == "__main__":
    unittest.main()
