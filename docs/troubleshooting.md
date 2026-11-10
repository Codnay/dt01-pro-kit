# Troubleshooting

Start with the control panel (`http://<robot>:8090`). It shows each job's log,
and the last lines usually say what went wrong.

The four pages are services. To see one's log from a terminal on the robot
computer:

```bash
journalctl -u dt01-panel -n 50         # control panel
journalctl -u dt01-base -n 50          # base + lift page
journalctl -u dt01-vr-page -n 50       # VR page
journalctl -u dt01-camera-page -n 50   # camera page
```

`sudo systemctl restart <name>` restarts one.

## The robot computer

**None of the pages open.** Check the robot computer has power (its light is
on) and is on your network. If you use a router, the cable goes in a **LAN**
port. The base's own network cable does not provide internet: the robot
computer needs a second cable, or Wi-Fi router, for that.

**Everything worked, then went offline after the base's cable was plugged in.**
The robot computer can pick the base's cable as its route to the internet, and
the base has none. Fix it once:

```bash
nmcli -t -f NAME,DEVICE con show --active     # find the connection on the base's port
sudo nmcli con mod "<that name>" ipv4.never-default yes ipv6.never-default yes ipv4.ignore-auto-dns yes
sudo nmcli con up "<that name>"
```

## The base and the lift

**"Not connected".** The base's controller is at `192.168.1.15`. From the
robot computer, `ping 192.168.1.15` should answer. If not: the base is off, or
its network cable is not in the robot computer.

**Commands are accepted but the base does not move.** In this order:

1. Push the red e-stop on the base in, and pull it out. This is needed after
   every power-up, and the page cannot see it.
2. On the remote control: **SWB in the middle**, **SWD down**.
3. Press **Run preflight** on the Drive tab.

**"Lift serial port not opened".** The lift talks over a USB cable, and the
page looks for it once, when it starts. Plug the lift's USB adapter into the
robot computer, then `sudo systemctl restart dt01-base`.

## The cameras

**The Cameras job stops straight away.** Open its log. "No camera produced a
frame" usually means another program is using the cameras, or none is plugged
in. `.venv/bin/python -m arms.cameras --list` shows what the computer can see.

**One tile is black.** That camera dropped off USB. Re-plug it, then Stop and
Start the **Cameras** job. If the same one keeps dropping, move it to another
USB port and use a shorter cable.

**Wrong names, or a camera was replaced.** Edit the *Cameras* job in the
control panel: each `--camera` entry is `name:serial`.

**The camera page is empty but the job is running.** Check the LiveKit keys in
`.env`, then `sudo systemctl restart dt01-camera-page`.

## The arms

**Arm teleop will not start: "CAN port ... is not UP".** Control panel → CAN & arms.
Both adapters need a role (*left* and *right*). Press **Apply**. Check the
arms have power.

**An arm will not turn on, or stays limp.** Control panel → CAN & arms →
**Reclaim CAN control** for that arm, then start Arm teleop again. If that
does not help, unplug the arm's power, wait ten seconds, and plug it back in.

**"Enter VR" does nothing.** The address must start with `https://`, and you
must accept the certificate warning once (**Advanced → Proceed**). Use the
Quest's own browser.

**In VR, but the arms do not move.** Go down this list:

1. The VR page shows your robot's name, and **Control** says you hold it.
   If not, press **Take control**.
2. Press **A** once. The Arm teleop log says the arms are enabled.
3. Hold a **grip** button. The arm follows only while it is held.

**The arm keeps letting go while I hold the grip.** The link between the
headset and the robot is dropping movements. Use a stronger Wi-Fi signal for
the headset. If it still happens, add `--livekit-input-timeout 2` to the
*Arm teleop* job (the default is 0.25 seconds): the arm then waits that long
for the link to come back before it lets go.

**The arms paused by themselves.** A motor got too hot (60 °C). Let them cool,
then press **A**.

**Arm teleop will not stop.** Wait the full minute. If it is truly stuck, the
panel offers **Force stop**: read its warning first. A forced stop can leave
an arm locked until its power is unplugged and plugged back in.

## Starting again from scratch

`./setup.sh` is safe to run again. It keeps `.env` and your jobs, and
reinstalls the services. To change the robot's name or the LiveKit keys, edit
`.env` and restart the VR page, the camera page and the jobs.
