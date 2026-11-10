"""Unit tests for the take-control lease (single active operator).

Covers the follower-side arbitration that lets exactly one Meta Quest operator
drive the arms at a time, with preemptive "take control" handoff. Pure logic —
no Agora SDK required.

    python -m unittest discover -s tests -p 'test_control_lease.py'
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms.xr_control import (  # noqa: E402
    CONTROL_STATE,
    ControlLease,
    decode_control,
    decode_xr_frame,
    encode_control_state,
    encode_xr_frame,
    frame_control_id,
)


class _Clock:
    """Manually-advanced clock for deterministic suppression-window tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ControlLeaseTest(unittest.TestCase):
    def test_first_operator_becomes_holder(self) -> None:
        lease = ControlLease()
        self.assertTrue(lease.is_holder("alice", 0, "Alice"))
        self.assertEqual(lease.snapshot(), ("alice", 0, "Alice"))
        # Same holder, same epoch: still allowed, no re-handoff.
        self.assertTrue(lease.is_holder("alice", 0, "Alice"))

    def test_second_operator_rejected_without_higher_epoch(self) -> None:
        lease = ControlLease()
        self.assertTrue(lease.is_holder("zed", 5, "Zed"))
        # A challenger at the same epoch with a lexicographically smaller cid loses.
        self.assertFalse(lease.is_holder("amy", 5, "Amy"))
        # A challenger at a LOWER epoch loses regardless of cid (stale publisher).
        self.assertFalse(lease.is_holder("zzz", 4, "Zzz"))
        self.assertEqual(lease.snapshot()[0], "zed")

    def test_higher_epoch_preempts(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 5, "Alice")
        self.assertTrue(lease.is_holder("bob", 6, "Bob"))
        self.assertEqual(lease.snapshot(), ("bob", 6, "Bob"))
        # Alice's now-stale frames at epoch 5 are rejected.
        self.assertFalse(lease.is_holder("alice", 5, "Alice"))

    def test_equal_epoch_tie_break_is_deterministic(self) -> None:
        # Two operators race to claim the same epoch; the greater cid wins,
        # independent of arrival order.
        order_a = ControlLease()
        order_a.is_holder("aaa", 1)
        self.assertTrue(order_a.is_holder("zzz", 1))  # zzz > aaa preempts
        self.assertEqual(order_a.snapshot()[0], "zzz")

        order_b = ControlLease()
        order_b.is_holder("zzz", 1)
        self.assertFalse(order_b.is_holder("aaa", 1))  # aaa < zzz loses
        self.assertEqual(order_b.snapshot()[0], "zzz")

    def test_handoff_opens_suppression_window(self) -> None:
        clock = _Clock()
        lease = ControlLease(suppress_sec=0.75, clock=clock)
        lease.is_holder("alice", 0)
        self.assertTrue(lease.suppressed())  # first claim is a handoff too
        clock.advance(0.8)
        self.assertFalse(lease.suppressed())
        # A preemption re-opens the window.
        lease.is_holder("bob", 1)
        self.assertTrue(lease.suppressed())
        clock.advance(0.5)
        self.assertTrue(lease.suppressed())
        clock.advance(0.3)
        self.assertFalse(lease.suppressed())

    def test_non_handoff_frame_does_not_reopen_suppression(self) -> None:
        clock = _Clock()
        lease = ControlLease(suppress_sec=0.75, clock=clock)
        lease.is_holder("alice", 0)
        clock.advance(1.0)
        # Holder's ongoing frames must NOT keep re-suppressing teleop.
        self.assertTrue(lease.is_holder("alice", 0))
        self.assertFalse(lease.suppressed())

    def test_release_drops_lease_and_keeps_epoch_monotonic(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 3, "Alice")
        # A non-holder cannot release the lease.
        self.assertFalse(lease.release("bob", 3))
        self.assertTrue(lease.release("alice", 3))
        self.assertEqual(lease.snapshot()[0], None)
        # After release, a claim must still out-epoch the released holder.
        self.assertFalse(lease.is_holder("carol", 3))  # ties the old epoch -> loses
        self.assertTrue(lease.is_holder("carol", 4))

    def test_stale_holder_expires_and_frees_control(self) -> None:
        # The lockout bug: a holder disconnects without releasing, and a fresh
        # browser session (epoch counter reset LOW) must still be able to take
        # over once the dead holder times out.
        clock = _Clock()
        lease = ControlLease(holder_timeout_sec=3.0, clock=clock)
        self.assertTrue(lease.is_holder("dead-holder", 5, "Old"))
        # A newcomer at a LOWER epoch loses while the holder is still fresh...
        self.assertFalse(lease.is_holder("fresh-browser", 1, "New"))
        # ...but after the holder goes silent past the timeout, it expires and
        # the lower-epoch newcomer wins.
        clock.advance(3.5)
        self.assertTrue(lease.is_holder("fresh-browser", 1, "New"))
        self.assertEqual(lease.snapshot()[0], "fresh-browser")

    def test_check_expiry_surfaces_silent_holder(self) -> None:
        # A holder goes silent with NO successor; check_expiry() must vacate it
        # and take_expired() must report it exactly once (for logging).
        clock = _Clock()
        lease = ControlLease(holder_timeout_sec=3.0, clock=clock)
        lease.is_holder("gone", 4, "Gone")
        lease.take_expired()  # drain the initial-claim state, if any
        clock.advance(3.5)
        lease.check_expiry()
        self.assertEqual(lease.snapshot()[0], None)
        self.assertEqual(lease.take_expired(), ("gone", 4, "Gone"))
        self.assertIsNone(lease.take_expired())  # cleared after one read

    def test_live_holder_does_not_expire(self) -> None:
        clock = _Clock()
        lease = ControlLease(holder_timeout_sec=3.0, clock=clock)
        lease.is_holder("holder", 2, "H")
        # Heartbeats/frames within the timeout keep the holder alive indefinitely.
        for _ in range(10):
            clock.advance(1.0)
            self.assertTrue(lease.is_holder("holder", 2, "H"))
        # A lower-epoch challenger still loses to the live holder.
        self.assertFalse(lease.is_holder("other", 1, "O"))
        self.assertEqual(lease.snapshot()[0], "holder")

    def test_legacy_untagged_operator_is_single_holder(self) -> None:
        # Frames from a pre-take-control leader decode to ("", 0).
        lease = ControlLease()
        cid, ep = frame_control_id('{"left":{},"right":{}}')
        self.assertEqual((cid, ep), ("", 0))
        self.assertTrue(lease.is_holder(cid, ep))
        self.assertTrue(lease.is_holder(cid, ep))  # stays the holder
        # A tagged WebXR operator can still take over with a higher epoch.
        self.assertTrue(lease.is_holder("webxr-abc", 1, "Operator"))


class FrameTaggingTest(unittest.TestCase):
    def test_encode_decode_roundtrip_carries_cid_epoch(self) -> None:
        per_hand = {
            "left": (None, 0.0, 0.0, 0.0),
            "right": (None, 0.5, 1.0, 0.0),
        }
        msg = encode_xr_frame(per_hand, cid="webxr-xyz", epoch=7)
        self.assertEqual(frame_control_id(msg), ("webxr-xyz", 7))
        # The reserved keys must not leak into the decoded hand set.
        decoded = decode_xr_frame(msg)
        self.assertEqual(set(decoded.keys()), {"left", "right"})

    def test_untagged_frame_defaults(self) -> None:
        per_hand = {"left": (None, 0.0, 0.0, 0.0), "right": (None, 0.0, 0.0, 0.0)}
        msg = encode_xr_frame(per_hand)
        self.assertEqual(frame_control_id(msg), ("", 0))


class AnnouncedLeaseStateTest(unittest.TestCase):
    """The follower's ``state`` broadcast — the leader's only source of the floor.

    A leader page keeps its epoch counter in memory and restarts it at 0 on every
    reload, while the follower's floor SURVIVES a clean release. Without the
    broadcast the two disagree and every reloaded browser has to click Take
    Control once per retained epoch before anything happens.
    """

    def test_epoch_floor_is_retained_across_a_clean_release(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 3, "Alice")
        self.assertTrue(lease.release("alice", 3))
        # Vacant, but still refusing anything at or below 3.
        self.assertIsNone(lease.snapshot()[0])
        self.assertEqual(lease.epoch_floor(), 3)

    def test_reloaded_leader_loses_without_the_floor_and_wins_with_it(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 3, "Alice")
        lease.release("alice", 3)

        # What a reloaded page used to send: its own counter, starting at 0.
        self.assertFalse(lease.is_holder("bob", 0 + 1, "Bob"))
        # What it sends once the follower has announced the floor.
        floor = lease.epoch_floor()
        self.assertTrue(lease.is_holder("bob", floor + 1, "Bob"))

    def test_floor_climbs_with_each_clean_release(self) -> None:
        """Why the symptom got worse the longer a stack stayed up: every clean
        release raises the bar for the NEXT operator by one."""
        lease = ControlLease()
        floors = []
        for who in ("alice", "bob", "carol"):
            ep = lease.epoch_floor() + 1
            self.assertTrue(lease.is_holder(who, ep, who))
            lease.release(who, ep)
            floors.append(lease.epoch_floor())
        self.assertEqual(floors, [0, 1, 2])

    def test_announced_state_names_the_holder(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 5, "Alice")
        cid, epoch, name = lease.snapshot()
        msg = decode_control(
            encode_control_state(holder_cid=cid or "", epoch=epoch, name=name)
        )
        self.assertEqual(msg["kind"], CONTROL_STATE)
        self.assertEqual(msg["cid"], "alice")
        self.assertEqual(msg["ep"], 5)
        self.assertEqual(msg["name"], "Alice")

    def test_announced_state_is_empty_cid_when_vacant(self) -> None:
        lease = ControlLease()
        lease.is_holder("alice", 5, "Alice")
        lease.release("alice", 5)
        cid, epoch, name = lease.snapshot()
        msg = decode_control(
            encode_control_state(holder_cid=cid or "", epoch=epoch, name=name)
        )
        self.assertEqual(msg["cid"], "")  # nobody is driving
        self.assertEqual(msg["ep"], 5)  # ...but 5 is still the bar to beat

    def test_expired_holder_resets_the_floor(self) -> None:
        """A holder that vanishes WITHOUT releasing frees the epoch entirely — the
        reason the bug looked intermittent: a crash left the next operator fine,
        a tidy exit did not."""
        clock = _Clock()
        lease = ControlLease(clock=clock)
        lease.is_holder("alice", 9, "Alice")
        clock.advance(10.0)
        lease.check_expiry()
        self.assertEqual(lease.epoch_floor(), -1)
        self.assertTrue(lease.is_holder("bob", 1, "Bob"))


if __name__ == "__main__":
    unittest.main()
