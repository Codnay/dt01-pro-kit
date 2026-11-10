"""Hardware-free subprocess used by transport failure and IPC tests."""

import json
import os
import select
import socket
import sys
import time

sock = socket.socket(fileno=int(sys.argv[1]))
config = json.loads(sock.recv(65536))
mode = config["token"]
started = time.monotonic()
last_heart = last_data = 0.0
resuming = False
seq = 0
epoch = int(mode[-1]) if mode.startswith("controller") else 0


def send(kind, **fields):
    try:
        sock.send(json.dumps(dict(kind=kind, at=time.monotonic(), **fields)).encode())
    except OSError:
        if mode == "hang_shutdown":
            time.sleep(10)
        sys.exit(0)


if mode != "hang_connect":
    send("state", state="connected", sid="fixture-sid")
    if epoch:
        send(
            "data", topic="ctl", data=json.dumps(dict(kind="claim", cid="op", ep=epoch))
        )
while True:
    now = time.monotonic()
    if mode == "crash" and now - started > 0.08:
        os._exit(7)
    if mode in ("resume", "lost_callback") and now - started > 0.08 and not resuming:
        resuming = True
        if mode == "resume":
            send("state", state="reconnecting")
    if mode not in ("hang_connect", "no_heartbeat") and now - last_heart >= 0.02:
        send("heartbeat", state=2 if resuming else 1)
        last_heart = now
    if mode == "data" and now - last_data >= 0.01:
        send("data", topic="xr", data="{}")
        last_data = now
    if epoch and now - last_data >= 0.01:
        seq += 1
        hand = dict(
            transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            trigger=1,
            grip=0.5,
            joystick_y=0,
            valid=True,
            seq=seq,
        )
        send("data", topic="xr", data=json.dumps(dict(cid="op", ep=epoch, right=hand)))
        last_data = now
    if select.select([sock], [], [], 0.001)[0]:
        try:
            message = json.loads(sock.recv(65536))
        except OSError:
            if mode == "hang_shutdown":
                time.sleep(10)
            break
        if message["kind"] == "stop" and mode != "hang_shutdown":
            break
        if message["kind"] == "publish":
            send("publish_ok")
            send("data", topic="ctl", data=message["data"])
sock.close()
