"""Exercise the real worker's independent STOP channel without CAN hardware."""

from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from arms.perf import piper_process as process


class PriorityStopTests(unittest.TestCase):
    def test_worker_delivers_stop_while_normal_rpc_is_blocked(self):
        entered, release, stopped = (threading.Event() for _ in range(3))

        class Controller:
            def __init__(self, **kwargs):
                self.generation = 0

            def resume_robot(self):
                entered.set()
                release.wait(3)
                return not stopped.is_set()

            def stop_and_hold(self):
                self.generation += 1
                stopped.set()
                return True

            def get_stop_hold_generation(self):
                return self.generation

            def cleanup(self):
                pass

        state = process._SeqBlock(process._STATE_SLOTS, create=True)
        cmd = process._SeqBlock(process._CMD_SLOTS, create=True)
        parent, child = mp.Pipe()
        priority, child_priority = mp.Pipe()
        fake_module = types.ModuleType("arms.piper_controller")
        fake_module.PiperController = Controller
        old_interval = sys.getswitchinterval()
        worker = threading.Thread(
            target=process._worker_main,
            args=(child, state.name, cmd.name, {}, 150, child_priority),
        )
        try:
            with patch.dict(sys.modules, {"arms.piper_controller": fake_module}), patch(
                "signal.signal"
            ):
                worker.start()
                self.assertTrue(parent.poll(2))
                self.assertEqual(parent.recv(), ("ready", None))
                parent.send((1, "resume_robot", (), {}))
                self.assertTrue(entered.wait(1))
                priority.send(1)
                self.assertTrue(priority.poll(1), "Stop queued behind resume")
                self.assertEqual(priority.recv(), (1, True))
                self.assertTrue(stopped.is_set())
                self.assertFalse(parent.poll(0))
                release.set()
                self.assertTrue(parent.poll(1))
                self.assertEqual(parent.recv(), (1, True, False))
                parent.send(None)
                worker.join(2)
                self.assertFalse(worker.is_alive())
        finally:
            release.set()
            if worker.is_alive():
                parent.send(None)
                worker.join(2)
            sys.setswitchinterval(old_interval)
            for pipe in (parent, child, priority, child_priority):
                pipe.close()
            state.close(unlink=True)
            cmd.close(unlink=True)

    def test_proxy_stop_ignores_normal_rpc_lock_and_discards_late_ack(self):
        parent, child = mp.Pipe()
        proxy = process.PiperControllerProxy.__new__(process.PiperControllerProxy)
        proxy._stop_conn = parent
        proxy._stop_rpc_lock = threading.Lock()
        proxy._rpc_lock = threading.Lock()
        proxy._stop_req_id = 1
        proxy._proc = types.SimpleNamespace(is_alive=lambda: True)
        proxy._rpc_lock.acquire()

        def answer():
            self.assertEqual(child.recv(), 2)
            child.send((1, False))  # late answer from an earlier timed-out stop
            child.send((2, True))

        responder = threading.Thread(target=answer)
        responder.start()
        try:
            self.assertTrue(proxy.stop_and_hold())
        finally:
            responder.join(1)
            proxy._rpc_lock.release()
            parent.close()
            child.close()


if __name__ == "__main__":
    unittest.main()
