#!/bin/sh
# Build the file set someone else needs to run the dashboard on their own laptop.
#
# The result is self-contained: no pip install, no network access, no build step.
# They need python3 (3.9+) and nothing else. Runs from anywhere it is unzipped.
#
#   ./dashboard/make_handoff.sh          -> dt01-dashboard.zip
#   ./dashboard/make_handoff.sh /tmp/out -> /tmp/out/dt01-dashboard.zip

set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$repo}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/dt01-dashboard"
# dashboard/ plus the CLI client it extends — drobot_api.py imports ../tools/drobot.py.
# Options before the operand: BSD tar (macOS) ignores --exclude after it.
for item in dashboard tools; do
    (cd "$repo" && tar cf - \
        --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
        "$item") \
        | (cd "$stage/dt01-dashboard" && tar xf -)
done

cat > "$stage/dt01-dashboard/START-HERE.txt" <<'TXT'
DT-01 dashboard
===============

You need python3 (3.9 or newer) and nothing else. No installing, no internet.
  macOS / Linux : already there. Check with:  python3 --version
  Windows       : install from python.org, then use "python" instead of "python3"

1. TRY IT WITH NO ROBOT
   Open a terminal in this folder and run:

       python3 dashboard/server.py --mock

   Open http://127.0.0.1:8800 in a browser. A simulated DT-01 is running behind
   it. Drive it, send it to a point, watch the battery. Nothing can be damaged.

2. USE THE REAL ROBOT
   Join the robot's WiFi (SSID drobot_5G, password 12345678), or put this laptop
   on whatever LAN the robot is on. Then:

       python3 dashboard/server.py

   Open http://127.0.0.1:8800, go to the Setup tab, press "Scan local network",
   and press Connect on the address it finds. If you already know the
   controller's IP you can skip the scan:

       python3 dashboard/server.py --robot 192.168.1.15

3. BEFORE THE ROBOT MOVES
   Three switches on the hardware will silently swallow every command if they
   are wrong. Check them first:
     - RC transmitter SWB to the MIDDLE position (CAN/RS-232 mode).
       In remote mode the transmitter wins and the dashboard is ignored.
     - RC transmitter SWD DOWN (releases the remote emergency stop).
     - The physical red e-stop on the chassis pulled OUT.
   Then use "Run preflight" on the Drive tab for the software-side checks.

4. STOPPING IT
   - The red EMERGENCY STOP button locks the brakes. Press it any time; Esc does
     the same thing.
   - Closing the browser tab also stops the robot. The safety timer runs in the
     server, not the page, so a crashed browser or dropped WiFi stops it too.
   - Ctrl-C in the terminal stops the robot and then the server.

Full documentation, including the safety model: dashboard/README.md
TXT

(cd "$stage" && zip -qr "$out/dt01-dashboard.zip" dt01-dashboard)
echo "wrote $out/dt01-dashboard.zip"
echo "send it over; they unzip and run:  python3 dashboard/server.py --mock"
