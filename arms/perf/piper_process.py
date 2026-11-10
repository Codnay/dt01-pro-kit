"""Run ``PiperController`` (and the SDK's CAN threads) in a separate process.

WHY
---
The teleop process is one CPython interpreter, so every hot thread serialises on
one GIL. Measured against a known-good reference (the x86 laptop, which runs the
same code with *no* jitter):

===========================  ==================  ==================
                             laptop (smooth)     Jetson (jitter)
===========================  ==================  ==================
single-thread Python           2887 k decode/s     1149 k decode/s
``ReadCan`` thread                 0.275 cores         0.504 cores
all Python threads                 0.495 cores         1.04 cores
GIL utilisation                          ~50%               ~100%
control-loop ticks > 12 ms               0.0%                9.3%
control-loop period max                11.61 ms            27.24 ms
===========================  ==================  ==================

Same code, same 8 physical cores, same ~3.4k CAN frames/s. The Jetson's CPU is
2.5x slower per thread (2.0 GHz Cortex-A78AE + CPython 3.10 vs 4.5 GHz Zen4 +
CPython 3.12), so identical work costs ~2x the GIL and crosses saturation. Below
saturation the control loop takes the lock immediately; at saturation it queues,
and the arm receives setpoints late-then-in-a-burst. The joint firmware
interpolates assuming even 10 ms spacing, so uneven arrival becomes velocity
discontinuity -- the buzzing the operator feels.

Adding cores does not help: the GIL admits one thread at a time. **Adding an
interpreter does.** ``ReadCan`` (~0.50 cores) and ``control_loop`` (~0.15) move
into a child process with almost nothing else in it, while IK / quest / services
keep the parent. Two GILs, each well under saturation.

DESIGN
------
Two channels, because the traffic has two very different shapes:

* **Hot state -- lock-free shared memory.** Joint angles, gripper, liveness ages
  and flags, at 100+ Hz in both directions. A seqlock (even/odd counter around
  the payload) gives wait-free reads with no syscall, no pickling and no chance
  of a hot thread blocking on the child. A torn read is *detected* and retried,
  never returned.
* **Everything else -- a request/response pipe.** ``resume_robot``,
  ``move_to_home``, ``recover_from_fault``, preset moves, diagnostics: dozens of
  methods, all low-rate. A generic ``__getattr__`` RPC forwards them by name, so
  the proxy does not have to hand-mirror the whole 2,900-line API surface.

The child is started with the **spawn** context deliberately: forking a process
that already owns CAN sockets and live threads would duplicate them.

SAFETY
------
* Every RPC has a timeout; a wedged child can never block a control thread.
* The hot setters are fire-and-forget writes to shared memory, so the IK path
  never waits on IPC.
* If the child dies, :meth:`PiperControllerProxy.is_alive` goes False and the
  hot getters return ``None`` -- the same contract callers already handle for
  "no feedback yet", which is what the existing watchdogs key off.
* This is **opt-in** (``--robot-process``). The default keeps everything
  in-process, so nothing changes until the flag is passed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from arms.feedback_timing import FeedbackTiming
from arms.logging_setup import get_logger

log = get_logger("robot-proc")

# -- shared-memory layout ----------------------------------------------------
# One float64 per slot. Slot 0 of each block is the seqlock counter: the writer
# makes it odd before touching the payload and even after, so a reader that sees
# the same even value before and after its copy knows the copy is coherent.

_SEQ = 0

# child -> parent
_S_JOINTS = 1  # 6
_S_GRIPPER = 7
_S_CTRL_LOOP_AGE = 8
_S_CMD_TX_AGE = 9
_S_MOTIONCTRL_AGE = 10
_S_MOTIONCTRL_MODE = 11
_S_ENABLED = 12
_S_HOMED = 13
_S_LOOP_RUNNING = 14
_S_HAS_JOINTS = 15
_S_HAS_GRIPPER = 16
_S_TARGET = 17  # 6 -- the worker's applied (post-clamp) joint target
_S_HAS_TARGET = 23
_S_CLAMP = 24  # 6 -- |applied - requested| for the last target the worker took
_S_HAS_CLAMP = 30
_S_STAMP = 31
_S_STOP_GENERATION = 32
_S_JOINT_TIMING = 33  # event, poll, raw SDK receive time, source code
_S_GRIPPER_TIMING = 37  # same four fields, paired with measured gripper
_S_STOP_LATCHED = 41
_STATE_SLOTS = 42

# parent -> child
_C_JOINTS = 1  # 6
_C_JOINT_SEQ = 7  # bumped on every new joint target
_C_GRIPPER = 8
_C_GRIPPER_SEQ = 9
_C_AGILEX_TELEOP = 10
_C_TELEOP_GENERATION = 11  # -1 for ordinary/dashboard commands
_CMD_SLOTS = 12

# How often the child republishes measured state and picks up new targets.
# Both consumers (joint_state, control_loop) run at 100 Hz, so this only needs
# to sit comfortably above that -- every extra Hz is GIL the child's control
# loop has to compete for. Tunable because the child's GIL headroom is the
# thing that decides control-loop jitter on a slow-per-core host.
_DEFAULT_PUBLISH_HZ = float(os.environ.get("PIPER_ROBOT_PUBLISH_HZ", "150") or 150)

# How stale a published snapshot may be before the hot getters report "no data".
# The child republishes at _DEFAULT_PUBLISH_HZ, so anything beyond a few periods
# means it died or wedged. Without this the parent would keep serving the last
# values out of shared memory forever -- silently feeding stale joint angles into
# telemetry and starving the watchdogs that key off missing feedback.
_STATE_STALE_AFTER_S = 0.5

# RPCs that change whether the motors are enabled. The snapshot these values are
# normally read from is published on the child's OWN loop, so for up to one
# publish period after such a call it still carries the PRE-call value -- and
# resume_robot() deliberately holds the flag False for the tens of milliseconds
# it spends validating feedback, so the last snapshot before it returns almost
# always says "disabled". A caller that asks right after therefore gets the
# opposite of the truth: the preset dashboard read "not enabled" 1 ms after the
# worker logged "robot resumed successfully", and refused to enable the arms.
# (In-process there is no window at all, which is why the same button worked
# without --robot-process.) After one of these, `is_robot_enabled` asks the
# worker directly until a newer snapshot lands.
_STATE_CHANGING_RPCS = frozenset(
    {
        "resume_robot",
        "emergency_stop",
        "graceful_stop",
        "recover_from_fault",
        "cleanup",
    }
)

_RPC_TIMEOUT_S = 5.0

# How long the parent waits for the worker to finish tearing down before it
# resorts to SIGTERM. Must stay above the worker's worst-case teardown --
# publisher join (1 s) + graceful_stop (0.1 s + 5 s settle + up to 5 s disable)
# + DisconnectPort, ~11 s -- because a SIGTERM lands inside graceful_stop()'s
# settle window, skips its finally block, and strands the arm in a latched
# E-stop that only an arm power cycle clears. Raise this if the settle window
# in piper_controller.graceful_stop() ever grows.
_WORKER_TEARDOWN_TIMEOUT_S = 20.0
_SLOW_RPC_TIMEOUT_S = 30.0  # homing / preset moves legitimately take a while
_SLOW_METHODS = frozenset(
    {"move_to_home", "recover_from_fault", "graceful_stop", "cleanup", "wait_for_idle"}
)


def _timing_payload(timing: FeedbackTiming | None, offset: int) -> dict[int, float]:
    """Copy original feedback time alongside its values in the same seqlock."""
    values = timing.as_values() if timing is not None else [0.0] * 4
    return {offset + i: value for i, value in enumerate(values)}


def _snapshot_timing(snap: np.ndarray, offset: int) -> FeedbackTiming | None:
    """Decode timestamp provenance without replacing it with parent poll time."""
    event, polled, sdk, source = map(float, snap[offset : offset + 4])
    if polled <= 0:
        return None
    return FeedbackTiming(
        event, polled, sdk or None, "sdk_receive" if source == 1.0 else "poll_fallback"
    )


class _SeqBlock:
    """Seqlock-protected float64 array over shared memory."""

    def __init__(self, slots: int, name: str | None = None, create: bool = False):
        """Attach to (or create) a shared float64 block of ``slots`` entries."""
        self._slots = slots
        size = slots * 8
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.buf = np.ndarray((slots,), dtype=np.float64, buffer=self.shm.buf)
        if create:
            self.buf[:] = 0.0

    @property
    def name(self) -> str:
        """Shared-memory segment name, for handing to the child process."""
        return str(self.shm.name)

    def write(self, values: dict[int, float]) -> None:
        """Publish ``{slot: value}`` atomically with respect to readers."""
        self.buf[_SEQ] += 1.0  # odd -> write in progress
        for slot, value in values.items():
            self.buf[slot] = value
        self.buf[_SEQ] += 1.0  # even -> consistent

    def read(self, retries: int = 8) -> np.ndarray | None:
        """Return a coherent copy, or None if the writer kept winning."""
        for _ in range(retries):
            before = float(self.buf[_SEQ])
            if before % 2.0 != 0.0:  # write in progress
                continue
            snapshot = self.buf.copy()
            if float(self.buf[_SEQ]) == before:
                return snapshot
        return None

    def close(self, unlink: bool = False) -> None:
        try:
            self.shm.close()
            if unlink:
                self.shm.unlink()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass


# -- child process -----------------------------------------------------------


def _worker_main(
    conn: Any,
    state_name: str,
    cmd_name: str,
    ctor_kwargs: dict[str, Any],
    publish_hz: float,
    stop_conn: Any,
) -> None:
    """Own the real PiperController; publish state, apply commands, serve RPC."""
    import signal
    import sys

    # Ctrl-C reaches the whole process group. The child must NOT die on it: the
    # parent runs an ordered teardown (stop teleop -> controller.cleanup() ->
    # shutdown) and this process owns the CAN socket and the arm, so tearing it
    # down first turns a graceful stop into a traceback and skips the SDK's own
    # cleanup. Ignoring SIGINT here leaves shutdown driven solely by the parent,
    # via the __shutdown__ RPC or pipe EOF if the parent dies.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # This interpreter runs only the SDK CAN threads and the control loop, so a
    # short GIL handoff interval costs almost nothing and keeps the control loop
    # from ever waiting behind a burst of CAN decoding.
    sys.setswitchinterval(0.0005)

    from arms.piper_controller import PiperController  # noqa: PLC0415 - child-only

    state = _SeqBlock(_STATE_SLOTS, name=state_name)
    cmd = _SeqBlock(_CMD_SLOTS, name=cmd_name)

    controller = PiperController(**ctor_kwargs)
    stop = threading.Event()
    command_apply_lock = threading.Lock()
    discard_joint_through = [0.0]
    discard_gripper_through = [0.0]

    def _serve_stop() -> None:
        # Dedicated channel/thread: never queue a stop behind an enable,
        # recovery, or telemetry RPC. The controller fences CAN writes itself.
        while not stop.is_set():
            try:
                if not stop_conn.poll(0.1):
                    continue
                req_id = stop_conn.recv()
                result = bool(controller.stop_and_hold())
                stop_conn.send((req_id, result))
            except (EOFError, OSError):
                break
            except Exception as exc:  # noqa: BLE001
                log.error("priority stop failed: %s", exc)
                stop_conn.send((req_id, False))

    stop_thread = threading.Thread(target=_serve_stop, name="priority-stop", daemon=True)
    stop_thread.start()

    def _publish_loop() -> None:
        period = 1.0 / max(publish_hz, 1.0)
        last_joint_seq = 0.0
        last_gripper_seq = 0.0
        # |applied - requested| for the most recent target we took from the
        # parent. Measured HERE, where set-then-read is synchronous, because
        # the parent cannot measure it: its write lands in shared memory and
        # its read-back is a pipe RPC, so a read-back that predates its own
        # write reads as a clamp. Latched only when we actually apply a
        # target, so a hold/home moving _target_joint_angles on its own does
        # not masquerade as clamping.
        last_clamp: np.ndarray | None = None
        while not stop.is_set():
            tick = time.time()
            # ---- apply parent commands -------------------------------------
            with command_apply_lock:
                snap = cmd.read()
                if snap is not None:
                    js = float(snap[_C_JOINT_SEQ])
                    if js != last_joint_seq and js > discard_joint_through[0]:
                        last_joint_seq = js
                        requested = np.asarray(
                            snap[_C_JOINTS : _C_JOINTS + 6], dtype=np.float64
                        )
                        try:
                            if snap[_C_AGILEX_TELEOP] > 0.5:
                                generation = int(snap[_C_TELEOP_GENERATION])
                                controller.set_target_joint_angles(
                                    requested,
                                    use_agilex_teleop=True,
                                    teleop_generation=(
                                        generation if generation >= 0 else None
                                    ),
                                )
                            else:
                                controller.set_target_joint_angles(requested)
                            applied = controller.get_target_joint_angles()
                            last_clamp = (
                                None
                                if applied is None
                                else np.abs(
                                    np.asarray(applied, dtype=np.float64).reshape(-1)[:6]
                                    - requested
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            last_clamp = None
                            log.warning("set_target_joint_angles failed: %s", exc)
                    gs = float(snap[_C_GRIPPER_SEQ])
                    if gs != last_gripper_seq and gs > discard_gripper_through[0]:
                        last_gripper_seq = gs
                        try:
                            controller.set_gripper_open_value(float(snap[_C_GRIPPER]))
                        except Exception as exc:  # noqa: BLE001
                            log.warning("set_gripper_open_value failed: %s", exc)

            # ---- publish measured state ------------------------------------
            payload: dict[int, float] = {}
            joint_timing = None
            try:
                joints, joint_timing = controller.get_current_joint_feedback()
            except Exception:  # noqa: BLE001
                joints = None
            if joints is not None:
                arr = np.asarray(joints, dtype=np.float64).reshape(-1)[:6]
                for i in range(6):
                    payload[_S_JOINTS + i] = float(arr[i])
                payload[_S_HAS_JOINTS] = 1.0
            else:
                payload[_S_HAS_JOINTS] = 0.0
            payload.update(_timing_payload(joint_timing, _S_JOINT_TIMING))
            gripper_timing = None
            try:
                grip, gripper_timing = controller.get_current_gripper_feedback()
            except Exception:  # noqa: BLE001
                grip = None
            payload[_S_GRIPPER] = float(grip) if grip is not None else 0.0
            payload[_S_HAS_GRIPPER] = 1.0 if grip is not None else 0.0
            payload.update(_timing_payload(gripper_timing, _S_GRIPPER_TIMING))

            for slot, fn, default in (
                (_S_CTRL_LOOP_AGE, "get_control_loop_age_s", 0.0),
                (_S_CMD_TX_AGE, "get_last_command_tx_age_s", 0.0),
                (_S_MOTIONCTRL_AGE, "get_motionctrl_age_s", 0.0),
                (_S_MOTIONCTRL_MODE, "get_motionctrl_move_mode", 0.0),
            ):
                try:
                    payload[slot] = float(getattr(controller, fn)())
                except Exception:  # noqa: BLE001
                    payload[slot] = default
            for slot, fn in (
                (_S_ENABLED, "is_robot_enabled"),
                (_S_HOMED, "is_robot_homed"),
                (_S_LOOP_RUNNING, "is_control_loop_running"),
                (_S_STOP_LATCHED, "is_stop_hold_latched"),
            ):
                try:
                    payload[slot] = 1.0 if getattr(controller, fn)() else 0.0
                except Exception:  # noqa: BLE001
                    payload[slot] = 0.0

            try:
                target = controller.get_target_joint_angles()
            except Exception:  # noqa: BLE001
                target = None
            if target is None:
                payload[_S_HAS_TARGET] = 0.0
            else:
                arr = np.asarray(target, dtype=np.float64).reshape(-1)[:6]
                for i in range(6):
                    payload[_S_TARGET + i] = float(arr[i])
                payload[_S_HAS_TARGET] = 1.0
            if last_clamp is None:
                payload[_S_HAS_CLAMP] = 0.0
            else:
                for i in range(6):
                    payload[_S_CLAMP + i] = float(last_clamp[i])
                payload[_S_HAS_CLAMP] = 1.0

            payload[_S_STAMP] = time.time()
            payload[_S_STOP_GENERATION] = controller.get_stop_hold_generation()
            state.write(payload)

            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)

    publisher = threading.Thread(target=_publish_loop, name="state-pub", daemon=True)
    publisher.start()
    conn.send(("ready", None))

    # ---- RPC loop ----------------------------------------------------------
    while True:
        try:
            if not conn.poll(0.2):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        req_id, method, args, kwargs = msg
        if method == "__shutdown__":
            conn.send((req_id, True, None))
            break
        try:
            if method in {"hold_teleop_position", "prepare_teleop", "resume_robot"}:
                with command_apply_lock:
                    pending = cmd.read()
                    if pending is None:
                        raise RuntimeError("Could not fence pending teleop commands")
                    discard_joint_through[0] = float(pending[_C_JOINT_SEQ])
                    discard_gripper_through[0] = float(pending[_C_GRIPPER_SEQ])
                    result = getattr(controller, method)(*args, **kwargs)
            else:
                result = getattr(controller, method)(*args, **kwargs)
            conn.send((req_id, True, result))
        except Exception as exc:  # noqa: BLE001 - report, never die
            conn.send((req_id, False, f"{type(exc).__name__}: {exc}"))

    stop.set()
    stop_thread.join(timeout=1.0)
    stop_conn.close()
    publisher.join(timeout=1.0)
    try:
        controller.cleanup()
    except Exception:  # noqa: BLE001
        pass
    state.close()
    cmd.close()


# -- parent proxy ------------------------------------------------------------


class PiperControllerProxy:
    """Stand-in for ``PiperController`` backed by a worker process.

    Hot getters/setters go through shared memory (no syscall, no blocking);
    everything else is forwarded by name over a pipe with a timeout.
    """

    def __init__(
        self, publish_hz: float = _DEFAULT_PUBLISH_HZ, **ctor_kwargs: Any
    ) -> None:
        """Spawn the worker and block until its controller is constructed.

        ``ctor_kwargs`` are forwarded verbatim to ``PiperController`` in the
        child, so callers build the proxy exactly as they built the controller.
        """
        self._state = _SeqBlock(_STATE_SLOTS, create=True)
        self._cmd = _SeqBlock(_CMD_SLOTS, create=True)
        self._joint_seq = 0.0
        self._gripper_seq = 0.0
        self._rpc_lock = threading.Lock()
        self._stop_rpc_lock = threading.Lock()
        self._stop_req_id = 0
        self._req_id = 0
        # time.time() of the last _STATE_CHANGING_RPCS call; a snapshot stamped
        # before this cannot be trusted for the enabled flag (see that constant).
        self._state_dirty_since = 0.0

        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._stop_conn, child_stop_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(
                child_conn,
                self._state.name,
                self._cmd.name,
                ctor_kwargs,
                publish_hz,
                child_stop_conn,
            ),
            name="piper-can",
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        child_stop_conn.close()

        if not self._conn.poll(60.0):
            raise RuntimeError("robot process did not come up within 60s")
        tag, _ = self._conn.recv()
        if tag != "ready":
            raise RuntimeError(f"robot process reported {tag!r} instead of ready")
        log.info("robot process up (pid=%s)", self._proc.pid)

    # -- lifecycle ----------------------------------------------------------
    def is_alive(self) -> bool:
        """True while the worker process is up."""
        return bool(self._proc.is_alive())

    def cleanup(self) -> None:
        """Teardown entry point used by the runtime.

        Deliberately shadows the generic RPC forward: the worker already calls
        ``controller.cleanup()`` when its loop exits, so shutting the process
        down does the real teardown *and* reaps the child. Forwarding
        ``cleanup`` over RPC instead would tear the SDK down but leave the
        process alive holding the CAN socket.
        """
        self.shutdown()

    def shutdown(self) -> None:
        """Stop the worker, then release the shared-memory segments.

        The join timeout is SAFETY-CRITICAL, not politeness. The worker acks
        ``__shutdown__`` immediately and only then runs its teardown, so the ack
        says nothing about progress. That teardown is
        ``controller.cleanup() -> graceful_stop()``, which deliberately E-STOPS
        the arm to cut torque so it settles under gravity, and clears the E-stop
        again in its ``finally`` block.

        ``terminate()`` is SIGTERM: Python does not unwind the stack, so that
        ``finally`` never runs. Killing the child mid-teardown therefore leaves
        the arm latched in emergency stop with its drivers cut -- reporting
        ``arm_status=0``/``ctrl_mode=1`` yet drawing exactly 0.00 A, recoverable
        only by physically re-plugging arm power. This is what made every Ctrl+C
        on the Jetson brick both arms while the laptop (no child process,
        ``cleanup()`` inline, nothing to SIGTERM) was unaffected.

        So the budget must exceed the worker's worst case, which is
        ``publisher.join`` 1 s + ``graceful_stop`` (0.1 s + 5 s settle + up to
        5 s disable) + DisconnectPort ~= 11 s.
        """
        try:
            if self._proc.is_alive():
                self._rpc("__shutdown__", timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.join(timeout=_WORKER_TEARDOWN_TIMEOUT_S)
            if self._proc.is_alive():
                # Never silent: a SIGTERM here means the arm may be stranded in
                # E-stop, and the operator needs to know why it will not come
                # back without a power cycle.
                log.error(
                    "robot worker did not finish teardown in %.0fs — SIGTERM may "
                    "have interrupted graceful_stop(), leaving the arm latched in "
                    "emergency stop with drivers cut. If it will not enable, "
                    "power-cycle the arm.",
                    _WORKER_TEARDOWN_TIMEOUT_S,
                )
                self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self._state.close(unlink=True)
        self._cmd.close(unlink=True)
        self._stop_conn.close()

    # -- hot path (shared memory) -------------------------------------------
    def _snapshot(self) -> np.ndarray | None:
        snap = self._state.read()
        if snap is None:
            return None
        stamp = float(snap[_S_STAMP])
        if not stamp or (time.time() - stamp) > _STATE_STALE_AFTER_S:
            return None  # child died or wedged -- do NOT serve stale state
        return snap

    def get_current_joint_angles(self) -> np.ndarray | None:
        """Latest measured joint angles (deg), or None if none published yet."""
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_JOINTS] < 0.5:
            return None
        return np.asarray(snap[_S_JOINTS : _S_JOINTS + 6], dtype=np.float64).copy()

    def get_current_gripper_open_value(self) -> float | None:
        """Latest measured gripper opening, or None if none published yet."""
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_GRIPPER] < 0.5:
            return None
        return float(snap[_S_GRIPPER])

    def get_current_joint_feedback(
        self,
    ) -> tuple[np.ndarray | None, FeedbackTiming | None]:
        """Joint values and CAN receive time from one coherent worker snapshot."""
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_JOINTS] < 0.5:
            return None, None
        return snap[_S_JOINTS : _S_JOINTS + 6].copy(), _snapshot_timing(
            snap, _S_JOINT_TIMING
        )

    def get_current_gripper_feedback(
        self,
    ) -> tuple[float | None, FeedbackTiming | None]:
        """Gripper value and CAN receive time from one coherent worker snapshot."""
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_GRIPPER] < 0.5:
            return None, None
        return float(snap[_S_GRIPPER]), _snapshot_timing(snap, _S_GRIPPER_TIMING)

    def set_target_joint_angles(
        self,
        joint_angles: Any,
        *,
        use_agilex_teleop: bool = False,
        teleop_generation: int | None = None,
    ) -> None:
        """Publish a joint target (deg). Fire-and-forget; never blocks."""
        arr = np.asarray(joint_angles, dtype=np.float64).reshape(-1)[:6]
        self._joint_seq += 1.0
        payload = {_C_JOINTS + i: float(arr[i]) for i in range(6)}
        payload[_C_JOINT_SEQ] = self._joint_seq
        payload[_C_AGILEX_TELEOP] = float(use_agilex_teleop)
        payload[_C_TELEOP_GENERATION] = (
            float(teleop_generation) if teleop_generation is not None else -1.0
        )
        self._cmd.write(payload)

    def set_gripper_open_value(self, value: float) -> None:
        """Publish a gripper opening. Fire-and-forget; never blocks."""
        self._gripper_seq += 1.0
        self._cmd.write({_C_GRIPPER: float(value), _C_GRIPPER_SEQ: self._gripper_seq})

    def get_control_loop_age_s(self) -> float:
        """Seconds since the worker's last control-loop tick."""
        snap = self._snapshot()
        return float(snap[_S_CTRL_LOOP_AGE]) if snap is not None else float("inf")

    def get_last_command_tx_age_s(self) -> float:
        """Seconds since the worker last transmitted a CAN command."""
        snap = self._snapshot()
        return float(snap[_S_CMD_TX_AGE]) if snap is not None else float("inf")

    def get_motionctrl_age_s(self) -> float:
        """Seconds since the last successful MotionCtrl exchange."""
        snap = self._snapshot()
        return float(snap[_S_MOTIONCTRL_AGE]) if snap is not None else float("inf")

    def get_motionctrl_move_mode(self) -> int:
        """Last MotionCtrl move mode reported by the arm."""
        snap = self._snapshot()
        return int(snap[_S_MOTIONCTRL_MODE]) if snap is not None else 0

    def is_robot_enabled(self) -> bool:
        """True when the worker reports the motors enabled.

        Reads the published snapshot, EXCEPT while that snapshot is older than
        the last enable/disable-changing RPC (see _STATE_CHANGING_RPCS) -- there
        it would report the state from before that call. The direct read costs an
        RPC, but only for the one publish period (~7 ms) it takes a fresh
        snapshot to land, so the callers that poll this every tick keep hitting
        shared memory.
        """
        snap = self._snapshot()
        if snap is not None and float(snap[_S_STAMP]) >= self._state_dirty_since:
            return bool(snap[_S_ENABLED] >= 0.5)
        return bool(self._rpc_soft("is_robot_enabled", False))

    def is_robot_homed(self) -> bool:
        """True when the worker reports the arm homed."""
        snap = self._snapshot()
        return bool(snap is not None and snap[_S_HOMED] >= 0.5)

    def is_control_loop_running(self) -> bool:
        """True while the worker's control loop thread is alive."""
        snap = self._snapshot()
        return bool(snap is not None and snap[_S_LOOP_RUNNING] >= 0.5)

    # -- everything else (pipe RPC) -----------------------------------------
    def _rpc(
        self, method: str, *args: Any, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        if timeout is None:
            timeout = _SLOW_RPC_TIMEOUT_S if method in _SLOW_METHODS else _RPC_TIMEOUT_S
        with self._rpc_lock:
            if not self._proc.is_alive():
                raise RuntimeError("robot process is not running")
            self._req_id += 1
            req_id = self._req_id
            self._conn.send((req_id, method, args, kwargs))
            if not self._conn.poll(timeout):
                raise TimeoutError(
                    f"robot process did not answer {method} in {timeout}s"
                )
            got_id, ok, payload = self._conn.recv()
            if got_id != req_id:
                raise RuntimeError(f"robot process RPC out of sync on {method}")
            if method in _STATE_CHANGING_RPCS:
                self._state_dirty_since = time.time()
        if not ok:
            raise RuntimeError(f"{method} failed in robot process: {payload}")
        return payload

    def _rpc_soft(self, method: str, default: Any, *args: Any, **kwargs: Any) -> Any:
        """RPC that degrades to ``default`` instead of raising.

        The joint-state loop calls the telemetry getters below without a guard,
        because in-process they could only ever return None. Over IPC they can
        also time out or race a restarting child, and an exception there would
        kill the thread that feeds every telemetry sink. Returning the same
        "no data this tick" value those callers already handle keeps collection
        degrading gracefully instead of stopping.
        """
        try:
            return self._rpc(method, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - must not break the caller
            log.debug("%s over IPC failed (%s); reporting no data", method, exc)
            return default

    def get_current_joint_motor_telemetry(self, **kwargs: Any) -> Any:
        """Per-joint (currents, velocities); (None, None) when unavailable."""
        return self._rpc_soft(
            "get_current_joint_motor_telemetry", (None, None), **kwargs
        )

    def get_piper_feedback_status(self, **kwargs: Any) -> Any:
        """Arm/joint status dict, or None when unavailable."""
        return self._rpc_soft("get_piper_feedback_status", None, **kwargs)

    def get_motor_fault_flags(self, **kwargs: Any) -> Any:
        """Per-joint fault flags, or None when unavailable."""
        return self._rpc_soft("get_motor_fault_flags", None, **kwargs)

    def get_target_joint_angles(self) -> Any:
        """Worker's applied (post-clamp) joint target, or None when unavailable.

        Served from shared memory rather than the pipe so it shares a transport
        with :meth:`set_target_joint_angles`. Over RPC this raced the write and
        returned the *previous* target, which callers read as a clamp.
        """
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_TARGET] < 0.5:
            return None
        return np.asarray(snap[_S_TARGET : _S_TARGET + 6], dtype=np.float64)

    def get_target_clamp_deltas(self) -> Any:
        """Per-joint ``|applied - requested|`` for the last applied target.

        Measured inside the worker, where set-then-read is synchronous. None
        when the worker has not applied a target yet. Callers must prefer this
        over differencing their own request against
        :meth:`get_target_joint_angles`, which is only eventually consistent.
        """
        snap = self._snapshot()
        if snap is None or snap[_S_HAS_CLAMP] < 0.5:
            return None
        return np.asarray(snap[_S_CLAMP : _S_CLAMP + 6], dtype=np.float64)

    def is_state_fresh(self) -> bool:
        """False when the worker's snapshot is too stale to conclude anything.

        :meth:`is_robot_enabled` and :meth:`is_control_loop_running` return
        False on a stale snapshot because a bool has no third value. Callers
        that would *warn* on a False must gate on this first, so a publish
        hiccup is not reported as "the robot is disabled".
        """
        return self._snapshot() is not None

    def get_health_snapshot(self) -> dict[str, Any] | None:
        """Coherent health view from ONE snapshot read, or None if stale.

        Every other accessor takes its own snapshot, so a caller that checks
        freshness and then reads two more values can have the snapshot go
        stale in between -- which surfaced as "control_loop thread is not
        running" and "No command TX for infs" while the loop was demonstrably
        streaming at 100 Hz. Warning paths must read this once instead.
        """
        snap = self._snapshot()
        if snap is None:
            return None
        return {
            "enabled": bool(snap[_S_ENABLED] >= 0.5),
            "loop_running": bool(snap[_S_LOOP_RUNNING] >= 0.5),
            "control_loop_age_s": float(snap[_S_CTRL_LOOP_AGE]),
            "cmd_tx_age_s": float(snap[_S_CMD_TX_AGE]),
            "motionctrl_age_s": float(snap[_S_MOTIONCTRL_AGE]),
        }

    def get_stop_hold_generation(self) -> int:
        """Capture an Enable token without waiting behind another RPC."""
        snap = self._snapshot()
        return int(snap[_S_STOP_GENERATION]) if snap is not None else -1

    def is_stop_hold_latched(self) -> bool:
        """Expose the worker's Teach/Stop fence without a blocking RPC."""
        snap = self._snapshot()
        if snap is None or snap[_S_STAMP] <= self._state_dirty_since:
            return False  # Do not re-pause from a pre-Enable snapshot.
        return bool(snap[_S_STOP_LATCHED] >= 0.5)

    def stop_and_hold(self) -> bool:
        """Deliver STOP independently of a slow/in-flight normal RPC."""
        with self._stop_rpc_lock:
            try:
                if not self._proc.is_alive():
                    return False
                self._stop_req_id += 1
                req_id = self._stop_req_id
                self._stop_conn.send(req_id)
                deadline = time.monotonic() + _RPC_TIMEOUT_S
                while self._stop_conn.poll(max(0.0, deadline - time.monotonic())):
                    got_id, result = self._stop_conn.recv()
                    if got_id == req_id:
                        return bool(result)
                    # A late reply to a timed-out stop cannot acknowledge this one.
                    if time.monotonic() >= deadline:
                        break
            except (EOFError, OSError) as exc:
                log.warning("Could not deliver priority stop: %s", exc)
        return False

    def hold_current_position(self, *, allow_resume: bool = True) -> bool:
        """Ask the arm to hold. False when the request could not be delivered.

        ``allow_resume`` MUST be forwarded: this signature used to take no
        arguments, so a caller passing ``allow_resume=False`` got a TypeError and
        (where one existed) fell back to the no-argument call -- silently running
        with the worker's ``allow_resume=True`` default. Shutdown and detach
        paths pass False precisely so that stopping a service can never
        implicitly re-enable hardware, and over IPC that guarantee was lost.
        """
        return bool(
            self._rpc_soft("hold_current_position", False, allow_resume=allow_resume)
        )

    def log_hardware_diagnostics(self, *args: Any, **kwargs: Any) -> None:
        """Diagnostics logging; never worth raising over."""
        self._rpc_soft("log_hardware_diagnostics", None, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward any other controller method to the worker over the pipe."""
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return self._rpc(name, *args, **kwargs)

        _call.__name__ = name
        return _call


__all__ = ["PiperControllerProxy"]
