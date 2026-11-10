#!/usr/bin/env python3
"""Clear a latched PiPER emergency stop / teach mode and reclaim CAN control.

Two states latch in the ARM and therefore survive restarting teleop:

* **Emergency stop** -- ``arm_status == 0x01``. The firmware drops to standby
  (``ctrl_mode == 0x00``) and discards every joint command. No fault flag is
  set anywhere (E-stop is a *status*, not a fault), joint currents fall to
  ~0.05 A, and because PiPER joints have no brakes the arm sags onto its
  limits. Released with ``MotionCtrl_1(0x02)`` (Resume).
* **Teach/drag mode** -- ``ctrl_mode == 0x02``, described below.

After a physical drag/teach the firmware reports ``arm_status.ctrl_mode == 0x02``
(TEACH) on CAN 0x2A1 and **keeps reporting it** -- it does not clear when the
operator lets go. Only a ``MotionCtrl_2`` with ``ctrl_mode=0x01`` (CAN command
mode) takes control back.

That creates a startup deadlock for teleop: ``PiperController``'s teach guard
pauses command TX as soon as it sees TEACH, which means the control loop never
gets to send the ``MotionCtrl_2`` that would clear it. Re-running the teleop
script does not help, because the guard trips again before the reclaim. Symptom::

    teach-mode hold active - command TX paused (ctrl_mode==TEACH).
    Teleop will not resume on its own; re-run the script to resume.

Run this once, with teleop STOPPED, to get out of that state.

SAFETY
------
This sends only control-mode messages. It does **not** enable motors and does
**not** command any motion, so the arm cannot move as a result of running it. It is still worth keeping hands clear: the arm becomes command-followable
again afterwards, so whatever you start next can move it.

Usage::

    python scripts/piper/reclaim_can_control.py --arm right
    python scripts/piper/reclaim_can_control.py --can can_right --verify
"""

from __future__ import annotations

import argparse
import sys
import time

ARM_TO_CAN = {"left": "can_left", "right": "can_right"}

CTRL_MODE_STANDBY = 0x00
CTRL_MODE_CAN = 0x01
CTRL_MODE_TEACH = 0x02

# MotionCtrl_1 emergency_stop field, per the piper_sdk docstring:
#   0x00 Invalid   0x01 Emergency stop   0x02 Resume
# An E-stop latches in the ARM, so it survives restarting teleop -- the arm
# reports arm_status == 0x01 and ctrl_mode == 0x00 (standby) and silently
# discards every joint command, with no fault flag set anywhere.
ESTOP_ENGAGE = 0x01
ESTOP_RESUME = 0x02

ARM_STATUS_NORMAL = 0x00
ARM_STATUS_ESTOP = 0x01


def _read_ctrl_mode(piper) -> int | None:  # noqa: ANN001 - SDK object
    """Return the arm's reported ctrl_mode, or None if unreadable."""
    try:
        status = piper.GetArmStatus()
        mode = status.arm_status.ctrl_mode
        return int(getattr(mode, "value", mode))
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


def _read_arm_status(piper) -> int | None:  # noqa: ANN001 - SDK object
    """Return the arm's reported arm_status, or None if unreadable."""
    try:
        status = piper.GetArmStatus()
        value = status.arm_status.arm_status
        return int(getattr(value, "value", value))
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


def _clear_estop(piper, retries: int) -> None:  # noqa: ANN001 - SDK object
    """Send MotionCtrl_1 Resume a few times to release a latched E-stop."""
    for _ in range(retries):
        piper.MotionCtrl_1(ESTOP_RESUME, 0, 0)
        time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    """Reclaim CAN command mode on one arm."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--arm", choices=sorted(ARM_TO_CAN), help="Arm to reclaim.")
    group.add_argument("--can", help="CAN interface name (e.g. can_right).")
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="How many MotionCtrl_2 reclaim frames to send (default: 5).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Poll ctrl_mode before and after and report whether it cleared.",
    )
    args = parser.parse_args(argv)

    channel = args.can or ARM_TO_CAN[args.arm or "right"]

    try:
        from piper_sdk import C_PiperInterface_V2
    except Exception as exc:  # noqa: BLE001
        print(f"piper_sdk unavailable: {exc}", file=sys.stderr)
        return 2

    piper = C_PiperInterface_V2(channel)
    # NOTE: this SDK returns falsy from ConnectPort()/EnablePiper() even when the
    # call succeeded (piper_controller logs "returned False (no exception)" for
    # the same reason), so the return value is NOT a connection test. The real
    # test is whether arm-status feedback starts arriving.
    piper.ConnectPort()

    before = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        time.sleep(0.2)
        before = _read_ctrl_mode(piper)
        if before is not None:
            break
    if before is None:
        print(
            f"no arm-status feedback on {channel} after 3s -- is the interface up "
            f"(`ip -br link show {channel}`) and the arm powered?",
            file=sys.stderr,
        )
        return 2

    print(f"{channel}: ctrl_mode before = {_label(before)}")

    # An E-stop must be released BEFORE anything else: while it is latched the
    # firmware sits in standby and refuses the MotionCtrl_2 mode change below,
    # so a reclaim that skips this silently does nothing.
    arm_status = _read_arm_status(piper)
    if arm_status == ARM_STATUS_ESTOP:
        print(f"{channel}: arm_status = 0x01 EMERGENCY STOP -- releasing")
        _clear_estop(piper, args.retries)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.2)
            arm_status = _read_arm_status(piper)
            if arm_status == ARM_STATUS_NORMAL:
                break
        if arm_status == ARM_STATUS_NORMAL:
            print(f"{channel}: emergency stop cleared")
            before = _read_ctrl_mode(piper)
        else:
            print(
                f"{channel}: emergency stop did NOT clear (arm_status={arm_status}). "
                "A latched software E-stop clears with this command, so a failure "
                "here points at the PHYSICAL E-stop button still being engaged -- "
                "release it and re-run.",
                file=sys.stderr,
            )
            return 1
    elif arm_status is not None and arm_status != ARM_STATUS_NORMAL:
        print(f"{channel}: note - arm_status = 0x{arm_status:02X} (not normal)")

    if before == CTRL_MODE_CAN:
        print("already in CAN command mode; nothing to do.")
        return 0

    # ctrl_mode=0x01 CAN command, move_mode=0x01 MOVE J, 50% speed, MIT off.
    # Repeated a few times: a single frame can be missed, and the firmware only
    # leaves TEACH when it actually sees one.
    for _ in range(args.retries):
        piper.MotionCtrl_2(CTRL_MODE_CAN, 0x01, 50, 0x00)
        time.sleep(0.1)
    print(f"sent MotionCtrl_2(ctrl_mode=0x01) x{args.retries} to reclaim CAN control")

    if not args.verify:
        return 0

    deadline = time.time() + 3.0
    after = before
    while time.time() < deadline:
        time.sleep(0.2)
        after = _read_ctrl_mode(piper)
        if after == CTRL_MODE_CAN:
            break
    print(f"{channel}: ctrl_mode after  = {_label(after)}")
    if after == CTRL_MODE_CAN:
        print("OK - teleop can be started now.")
        return 0
    print(
        "STILL NOT in CAN mode. Most likely the physical teach/drag button on the "
        "arm is still held or latched -- release it and re-run. Stop teleop first "
        "so nothing else is driving the bus.",
        file=sys.stderr,
    )
    return 1


def _label(mode: int | None) -> str:
    """Human-readable ctrl_mode."""
    return {
        None: "unreadable",
        CTRL_MODE_STANDBY: "0x00 standby",
        CTRL_MODE_CAN: "0x01 CAN command",
        CTRL_MODE_TEACH: "0x02 TEACH (drag)",
    }.get(mode, f"0x{mode:02X} unknown")


if __name__ == "__main__":
    raise SystemExit(main())
