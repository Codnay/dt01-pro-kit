"""Unit tests for the clock-free link measurement and the pose de-duplication.

Two independent fixes, both about the operator->arm path being honest:

1. Round-trip latency. Every earlier attempt subtracted a Quest browser
   timestamp from a robot timestamp across two unsynchronised clocks, which
   measured skew and reported it as latency -- usually NEGATIVE. The follower
   now echoes an opaque id and the leader does all the arithmetic on one clock.
2. Filter de-duplication. The robot polls the reader at a fixed rate; when the
   link delivers fewer frames than that, the 1 euro filter used to be re-fed
   poses it had already seen, which flattened its speed estimate and smoothed
   hardest exactly when the link was worst.

    python -m unittest discover -s tests -p 'test_xr_link_quality.py'
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms.data_manager import DataManager  # noqa: E402
from arms.xr_control import (  # noqa: E402
    CONTROL_PING,
    CONTROL_PONG,
    decode_control,
    encode_pong,
    ping_id,
)


class PongWireTest(unittest.TestCase):
    """The echo carries the id and no follower clock reading."""

    def test_pong_echoes_the_id_verbatim(self) -> None:
        msg = encode_pong("abc-123")
        self.assertEqual(ping_id(msg), "abc-123")
        self.assertEqual(decode_control(msg)["kind"], CONTROL_PONG)

    def test_pong_carries_no_timestamp(self) -> None:
        """A follower clock reading would reintroduce the skew being avoided."""
        obj = json.loads(encode_pong("1"))
        self.assertNotIn("stamp", obj)
        self.assertEqual(set(obj) - {"kind", "cid", "ep", "name", "pid"}, set())

    def test_lease_decoder_contract_is_unchanged(self) -> None:
        """decode_control stays {kind,cid,ep,name} — both transports rely on it."""
        self.assertEqual(
            set(decode_control(encode_pong("9"))), {"kind", "cid", "ep", "name"}
        )

    def test_ping_id_is_forgiving(self) -> None:
        self.assertEqual(ping_id(b"not json"), "")
        self.assertEqual(ping_id(b'{"kind":"ping"}'), "")
        self.assertEqual(
            ping_id(json.dumps({"kind": CONTROL_PING, "pid": "7"}).encode()), "7"
        )


class PoseDeduplicationTest(unittest.TestCase):
    """``pose_is_new=False`` must move the dead-man but not the pose filter."""

    def setUp(self) -> None:
        self.dm = DataManager()
        self.pose_a = np.eye(4)
        self.pose_b = np.eye(4)
        self.pose_b[0, 3] = 1.0  # 1 m away, so any filter motion is obvious

    def test_duplicate_pose_leaves_the_filtered_output_alone(self) -> None:
        self.dm.set_controller_data("right", self.pose_a, 0.0, 0.0)
        # A genuinely new pose moves the filtered output toward it...
        self.dm.set_controller_data("right", self.pose_b, 0.0, 0.0)
        moved, _, _ = self.dm.get_controller_data("right")
        self.assertGreater(moved[0, 3], 0.0)

        # ...but re-feeding the SAME pose must not move it again.
        for _ in range(20):
            self.dm.set_controller_data(
                "right", self.pose_b, 0.0, 0.0, pose_is_new=False
            )
        held, _, _ = self.dm.get_controller_data("right")
        np.testing.assert_allclose(held, moved)

    def test_dead_man_still_updates_on_a_duplicate_pose(self) -> None:
        """Trigger and grip are safety inputs — they must never go stale."""
        self.dm.set_controller_data("right", self.pose_a, 0.0, 0.0)
        self.dm.set_controller_data("right", self.pose_a, 0.4, 0.9, pose_is_new=False)
        _, grip, trigger = self.dm.get_controller_data("right")
        self.assertAlmostEqual(grip, 0.4)
        self.assertAlmostEqual(trigger, 0.9)

    def test_raw_pose_still_refreshes(self) -> None:
        """Debug views read raw; only the filter is skipped."""
        self.dm.set_controller_data("right", self.pose_a, 0.0, 0.0)
        self.dm.set_controller_data("right", self.pose_b, 0.0, 0.0, pose_is_new=False)
        state = self.dm._controller_state_by_hand["right"]
        np.testing.assert_allclose(state.transform_raw, self.pose_b)

    def test_new_pose_after_duplicates_is_differentiated_over_the_real_gap(
        self,
    ) -> None:
        """The filter's clock must not advance across skipped samples.

        If it did, the next genuine frame would be differentiated over a tiny
        interval and read as a huge velocity.
        """
        self.dm.set_controller_data("right", self.pose_a, 0.0, 0.0)
        state = self.dm._controller_state_by_hand["right"]
        self.dm.set_controller_data("right", self.pose_b, 0.0, 0.0)
        t_after_real = state._filter.t_prev
        for _ in range(5):
            self.dm.set_controller_data(
                "right", self.pose_b, 0.0, 0.0, pose_is_new=False
            )
        self.assertEqual(state._filter.t_prev, t_after_real)


if __name__ == "__main__":
    unittest.main()
