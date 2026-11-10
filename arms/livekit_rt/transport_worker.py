"""Disposable LiveKit network worker. Never imports robot controllers or CAN.

The parent owns safety and leases. This process owns only the SDK and a private,
bounded Unix datagram socket. Killing a stuck native SDK cannot kill an arm worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time

MAX_PACKET = 65536


def encode_message(message: dict) -> bytes:
    """Encode an atomic IPC message, including its local monotonic receive time."""
    return json.dumps(message, separators=(",", ":")).encode()


async def run(sock: socket.socket, config: dict) -> None:
    """Connect once; the parent replaces us if connection or teardown stalls."""
    from livekit import rtc

    loop = asyncio.get_running_loop()
    room = rtc.Room()
    stopping = asyncio.Event()
    parent_pid = os.getppid()
    publish_tasks: set[asyncio.Task] = set()

    def send(kind: str, **fields) -> None:
        message = encode_message(dict(kind=kind, at=time.monotonic(), **fields))
        if len(message) > MAX_PACKET:
            if fields.get("topic") != "xr":
                stopping.set()
            return
        try:
            sock.send(message)
        except BlockingIOError:
            if fields.get("topic") != "xr":
                stopping.set()  # never silently lose reliable controls
        except OSError:
            stopping.set()

    def state(value: str) -> None:
        send("state", state=value, sid=getattr(room.local_participant, "sid", ""))

    @room.on("data_received")
    def data_received(packet, *args):
        if hasattr(packet, "data"):
            payload, topic = packet.data, packet.topic
        else:
            payload = packet
            topic = next((a for a in reversed(args) if isinstance(a, str)), "")
        if topic in ("xr", "btn", "ctl"):
            try:
                send("data", topic=topic, data=bytes(payload).decode("utf-8"))
            except (UnicodeError, TypeError, ValueError):
                pass

    @room.on("reconnecting")
    def reconnecting(*args):
        state("reconnecting")

    @room.on("reconnected")
    def reconnected(*args):
        state("connected")

    @room.on("disconnected")
    def disconnected(*args):
        state("disconnected")
        stopping.set()

    @room.on("participant_connected")
    def joined(participant):
        send("joined", identity=participant.identity)

    @room.on("participant_disconnected")
    def left(participant):
        send("left", identity=participant.identity)

    async def publish(payload: str) -> None:
        try:
            await asyncio.wait_for(
                room.local_participant.publish_data(
                    payload.encode(), reliable=True, topic="ctl"
                ),
                timeout=1.0,
            )
            send("publish_ok")
        except Exception:
            # Signal failures independently of the SDK's disconnect callback.
            send("publish_failed")

    def commands() -> None:
        try:
            for _ in range(64):
                raw = sock.recv(MAX_PACKET)
                if not raw:
                    stopping.set()
                    return
                message = json.loads(raw)
                if message.get("kind") == "stop":
                    stopping.set()
                elif message.get("kind") == "publish" and not stopping.is_set():
                    if len(publish_tasks) >= 16:
                        stopping.set()
                        return
                    task = asyncio.create_task(publish(message["data"]))
                    publish_tasks.add(task)
                    task.add_done_callback(publish_tasks.discard)
        except BlockingIOError:
            pass
        except (OSError, ValueError, KeyError):
            stopping.set()

    async def heartbeat() -> None:
        while not stopping.is_set():
            if os.getppid() != parent_pid:
                stopping.set()
                break
            send("heartbeat", state=int(room.connection_state))
            await asyncio.sleep(0.25)

    loop.add_reader(sock.fileno(), commands)
    heart = asyncio.create_task(heartbeat())
    try:
        await room.connect(
            config["url"], config["token"], rtc.RoomOptions(auto_subscribe=False)
        )
        state("connected")
        await stopping.wait()
    except Exception as exc:
        # Native errors can include token-bearing URLs. Never forward them.
        detail = str(exc).lower()
        auth = any(
            word in detail
            for word in ("401", "unauthorized", "token expired", "invalid token")
        )
        send(
            "error",
            error=(
                "authentication rejected; check token/credentials"
                if auth
                else type(exc).__name__
            ),
        )
    finally:
        heart.cancel()
        for task in publish_tasks:
            task.cancel()
        loop.remove_reader(sock.fileno())
        await room.disconnect()  # parent enforces a hard shutdown deadline


def main() -> None:
    """Read credentials over the inherited socket, never from argv or a file."""
    sock = socket.socket(fileno=int(sys.argv[1]))
    sock.settimeout(15)
    config = json.loads(sock.recv(MAX_PACKET))
    sock.setblocking(False)
    try:
        asyncio.run(run(sock, config))
    finally:
        sock.close()


if __name__ == "__main__":
    # main() has already disconnected the room and closed private IPC. Like a
    # multiprocessing worker, exit this disposable child without interpreter
    # finalization: the SDK's atexit disposer can call Rust/ctypes callbacks after
    # Python has begun tearing them down (observed on Jetson livekit 1.1.16).
    # The OS reclaims only this child's native runtime; it owns no robot objects.
    try:
        main()
    except KeyboardInterrupt:
        os._exit(130)
    except BaseException:
        os._exit(1)
    os._exit(0)
