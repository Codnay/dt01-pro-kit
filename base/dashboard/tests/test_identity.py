"""Tests for fleet-identity resolution.

    python3 -m unittest discover -s dashboard/tests -t dashboard

Identity decides which robot a metric belongs to, so the property worth
defending is that the *stated* source is the real one. A value that quietly
came from somewhere other than where the operator thinks is how two platforms
end up sharing one history in Grafana.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import identity                                           # noqa: E402


class TestPrecedence(unittest.TestCase):
    """Ordering only, isolated from disk.

    Without this patch these tests pass or fail depending on whether the machine
    running them happens to have a .env — which is exactly the kind of ambient
    dependency the resolver exists to make explicit. Disk lookup is covered by
    TestFileLookup below.
    """

    def setUp(self) -> None:
        patcher = mock.patch.object(identity, "_candidates", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_flag_beats_environment(self) -> None:
        self.assertEqual(identity.resolve("dt01-pro-01", {"ROBOT_ID": "from-env"}),
                         ("dt01-pro-01", "--robot-id"))

    def test_environment_used_when_no_flag(self) -> None:
        self.assertEqual(identity.resolve(None, {"ROBOT_ID": "dt01-pro-01"}),
                         ("dt01-pro-01", "$ROBOT_ID"))

    def test_blank_values_do_not_count_as_set(self) -> None:
        # ROBOT_ID="" in a shell profile must fall through, not name the robot
        # the empty string.
        value, source = identity.resolve("   ", {"ROBOT_ID": "  "})
        self.assertNotEqual(source, "--robot-id")
        self.assertNotEqual(source, "$ROBOT_ID")

    def test_falls_back_visibly_rather_than_plausibly(self) -> None:
        # A plausible default is worse than none: two platforms would share it
        # and the merged history would look healthy.
        value, source = identity.resolve(None, {})
        self.assertEqual(source, "fallback")
        self.assertEqual(value, identity.FALLBACK)
        self.assertNotIn("01", value)


class TestEnvFile(unittest.TestCase):
    def parse(self, text: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text(text, encoding="utf-8")
            return identity.read_env_file(path)

    def test_plain_assignment(self) -> None:
        self.assertEqual(self.parse("ROBOT_ID=dt01-pro-01"),
                         {"ROBOT_ID": "dt01-pro-01"})

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        self.assertEqual(self.parse("# a note\n\nROBOT_ID=dt01-pro-01\n"),
                         {"ROBOT_ID": "dt01-pro-01"})

    def test_trailing_comment_is_stripped(self) -> None:
        self.assertEqual(self.parse("ROBOT_ID=dt01-pro-01  # the US platform"),
                         {"ROBOT_ID": "dt01-pro-01"})

    def test_quotes_and_export_prefix(self) -> None:
        self.assertEqual(self.parse('export ROBOT_ID="dt01-pro-01"'),
                         {"ROBOT_ID": "dt01-pro-01"})
        self.assertEqual(self.parse("ROBOT_ID='dt01-pro-01'"),
                         {"ROBOT_ID": "dt01-pro-01"})

    def test_a_hash_inside_a_value_survives(self) -> None:
        # Only " #" starts a comment, so a value may contain one.
        self.assertEqual(self.parse("ROBOT_ID=piper#us#01"),
                         {"ROBOT_ID": "piper#us#01"})

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(identity.read_env_file(Path("/nonexistent/.env")), {})

    def test_a_directory_where_a_file_was_expected(self) -> None:
        self.assertEqual(identity.read_env_file(Path(tempfile.gettempdir())), {})


class TestFileLookup(unittest.TestCase):
    """Reading a .env off disk, and where the resolver looks for one."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.path = Path(self._folder.name) / ".env"
        patcher = mock.patch.object(identity, "_candidates",
                                    return_value=[self.path])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_candidate_env_file_is_used_and_named(self) -> None:
        # A value no other .env on this machine could supply, so the test
        # cannot pass by coincidence.
        self.path.write_text("ROBOT_ID=only-from-this-file\n")
        self.assertEqual(identity.resolve(None, {}),
                         ("only-from-this-file", str(self.path)))

    def test_environment_still_beats_a_file(self) -> None:
        self.path.write_text("ROBOT_ID=from-file\n")
        self.assertEqual(identity.resolve(None, {"ROBOT_ID": "from-env"}),
                         ("from-env", "$ROBOT_ID"))

    def test_a_file_without_robot_id_falls_through(self) -> None:
        self.path.write_text("SOMETHING_ELSE=1\n")
        self.assertEqual(identity.resolve(None, {})[1], "fallback")


class TestSearchPath(unittest.TestCase):
    """Where the resolver looks, asserted directly rather than by side effect."""

    def test_repository_root_and_package_dir_are_both_searched(self) -> None:
        # The dashboard also ships as a standalone zip with no repo root above
        # it, so both must be candidates — and the user's .env lives at the root.
        package = Path(identity.__file__).resolve().parent
        candidates = [Path(p) for p in identity._candidates()]
        self.assertIn(package / ".env", candidates)
        self.assertIn(package.parent / ".env", candidates)

    def test_nearest_wins(self) -> None:
        candidates = [Path(p) for p in identity._candidates()]
        package = Path(identity.__file__).resolve().parent
        self.assertLess(candidates.index(package / ".env"),
                        candidates.index(package.parent / ".env"))


if __name__ == "__main__":
    unittest.main()
