# How it works

```
 Quest browser ──▶ LiveKit ──▶ arms.teleop ──▶ CAN ──▶ the two arms
 (VR page)                     on the robot computer

 cameras ──▶ arms.cameras ──▶ LiveKit ──▶ Cameras page

 laptop browser ──▶ Base + lift page ──▶ the base's own controller (192.168.1.15)
                                    └──▶ the lift (USB serial)
```

Three separate parts share the robot computer. None depends on the others.

## The arms (`arms/`)

- The **VR page** reads the position of the two controllers about 50 times a
  second and sends it through LiveKit.
- **`arms.teleop`** receives it, works out the joint angles that put each
  gripper where your hand is, and streams them to the arms over CAN.
- Only one headset can hold control at a time. **Take control** hands it over.
- The arm follows only while a grip button is held (a dead-man switch). If the
  link drops, the arm holds its position.

| File | What it does |
|---|---|
| `arms/teleop.py` | Entry point: `python -m arms.teleop` |
| `arms/bimanual_teleop_runtime.py` | Starts and stops the arms; button handling |
| `arms/threads/` | The loops: controller input, inverse kinematics, joint feedback, cameras |
| `arms/piper_controller.py` | The last step before the hardware; speed limits and safe stops live here |
| `arms/pink_ik_solver.py` | Inverse kinematics |
| `arms/livekit_rt/` | LiveKit link, the VR page and the camera page |
| `arms/cameras.py` | Entry point: `python -m arms.cameras` |
| `arms/configs.py` | Shipped settings. Your edits are saved in `config/configs.py` |
| `piper_description/` | The arm's 3D model and joint limits |

Units: the inverse kinematics works in radians and metres; the arm interface
takes degrees.

## The base and the lift (`base/`)

A web page for the DT-01 base. The base has its own controller with its own
navigation software; this page is a client of it. The lift is separate
hardware on a USB serial cable.

It needs nothing but Python: no installs. `python3 base/dashboard/server.py
--mock` runs it against a pretend robot, on any laptop.

The safety timer runs on the robot computer, not in the browser: if the page
closes or the network drops, the base stops.

## The control panel (`ops/`)

Starts and stops the two long-running jobs, shows their logs, sets up the CAN
adapters, and edits the settings file. Jobs are listed in `ops/jobs.json`.

It stops jobs the way Ctrl+C does and waits. It never force-stops the arms on
its own.

## Files that are yours

These are created on your robot and never uploaded:

| File | What is in it |
|---|---|
| `.env` | Robot name and LiveKit keys |
| `ops/jobs.json` | Your jobs: camera serials, which arms |
| `config/configs.py` | Your settings |
| `config/certs/` | The VR page's certificate |

## Tests

```bash
scripts/run_tests.sh
```

No robot is needed and nothing moves.
