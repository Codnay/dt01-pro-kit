#!/usr/bin/env bash
# One-time setup of the DT-01 Pro kit on the robot computer.
#
#   ./setup.sh                 everything (asks for your sudo password once)
#   ./setup.sh --no-services   Python environment and .env only, no sudo
#
# If the account that will run the robot is not an administrator, split it:
#
#   ./setup.sh --no-services                           as that account
#   sudo KIT_USER=<account> ./setup.sh --services-only  as an administrator
#
# Safe to run again: it keeps your .env and ops/jobs.json, and only reinstalls
# the services.
#
# Settings you can override, e.g. `PANEL_PORT=8091 ./setup.sh`:
#   BASE_IP      address of the base's own controller   (default 192.168.1.15)
#   PANEL_PORT   control panel                          (default 8090)
#   BASE_PORT    base + lift dashboard                  (default 8800)
#   VR_PORT      VR page, https                         (default 8443)
#   CAMERA_PORT  camera page                            (default 5175)
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IP="${BASE_IP:-192.168.1.15}"
PANEL_PORT="${PANEL_PORT:-8090}"
BASE_PORT="${BASE_PORT:-8800}"
VR_PORT="${VR_PORT:-8443}"
CAMERA_PORT="${CAMERA_PORT:-5175}"
KIT_USER="${KIT_USER:-$(id -un)}"
MODE="${1:-all}"

PY="$KIT/.venv/bin/python"
CAN_HELPER=/usr/local/sbin/dt01-can-apply
SUDOERS=/etc/sudoers.d/dt01-kit
CERT_DIR="$KIT/config/certs"
UNITS=(dt01-panel dt01-base dt01-vr-page dt01-camera-page)

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\n!! %s\n' "$*" >&2; exit 1; }

case "$MODE" in
  all|--no-services|--services-only) ;;
  *) fail "Unknown option: $MODE (see the top of this file)" ;;
esac

[[ "$(uname -s)" == "Linux" ]] || fail "Run this on the robot computer (Linux)."
command -v python3 >/dev/null || fail "python3 is not installed."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || fail "Python 3.10 or newer is needed (found $(python3 -V))."

# --- 1. Python environment, .env and jobs (no sudo) ------------------------

ask() {  # ask <prompt> <default>
  local reply
  read -r -p "    $1 [${2}]: " reply
  printf '%s' "${reply:-$2}"
}

install_python() {
  say "Python environment in $KIT/.venv"
  if [[ ! -x "$PY" ]]; then
    python3 -m venv "$KIT/.venv" \
      || fail "Could not create the environment. Try: sudo apt install python3-venv"
  fi
  "$PY" -m pip install --quiet --upgrade pip
  # --no-deps: requirements.txt is the complete, proven set (see its header).
  "$PY" -m pip install --quiet --no-deps -r "$KIT/requirements.txt"
  (cd "$KIT" && "$PY" -c 'import arms.teleop, arms.cameras') \
    || fail "The arm software did not import. See the error above."
  echo "    ok"
}

write_env() {
  if [[ -f "$KIT/.env" ]]; then
    say "Keeping your existing .env"
  elif [[ -t 0 ]]; then
    say "Robot name and LiveKit keys (from https://cloud.livekit.io -> Settings -> Keys)"
    local robot_id url key secret
    robot_id="$(ask 'Robot name' 'dt01-pro-01')"
    url="$(ask 'LiveKit URL (wss://...)' '')"
    key="$(ask 'LiveKit API key' '')"
    read -r -s -p "    LiveKit API secret (hidden): " secret; echo
    (umask 077; printf 'ROBOT_ID=%s\nLIVEKIT_URL=%s\nLIVEKIT_API_KEY=%s\nLIVEKIT_API_SECRET=%s\n' \
      "$robot_id" "$url" "$key" "$secret" > "$KIT/.env")
    echo "    wrote $KIT/.env"
  else
    cp "$KIT/.env.example" "$KIT/.env"
    chmod 600 "$KIT/.env"
    say "Created .env from the template. Fill in the LiveKit keys before using VR."
  fi
}

write_jobs() {
  say "Control panel jobs"
  local serials
  serials="$(cd "$KIT" && "$PY" -m arms.cameras --list 2>/dev/null | awk '/^[0-9]+ /{print $1}' || true)"
  # shellcheck disable=SC2086
  python3 "$KIT/scripts/make_jobs.py" "$KIT" $serials | sed 's/^/    /'
}

if [[ "$MODE" != "--services-only" ]]; then
  install_python
  write_env
  write_jobs
fi

if [[ "$MODE" == "--no-services" ]]; then
  say "Done (no services installed). Try: $PY -m arms.teleop --help"
  exit 0
fi

# --- 2. services (needs sudo) ----------------------------------------------

id "$KIT_USER" >/dev/null 2>&1 || fail "No such account: $KIT_USER"
[[ -x "$PY" ]] || fail "Run './setup.sh --no-services' as $KIT_USER first."

as_kit_user() {  # run a command as the account that owns the kit
  if [[ "$(id -un)" == "$KIT_USER" ]]; then "$@"; else sudo -u "$KIT_USER" "$@"; fi
}

port_free() {  # port_free <port> <our unit> <setting name>
  if ss -ltn 2>/dev/null | grep -q ":$1 " && ! systemctl is-active --quiet "$2"; then
    fail "Port $1 is already used by another program. Stop it, or pick another
   port, e.g.  ${3}=$(( $1 + 1 )) ./setup.sh"
  fi
}
port_free "$PANEL_PORT" dt01-panel PANEL_PORT
port_free "$BASE_PORT" dt01-base BASE_PORT
port_free "$VR_PORT" dt01-vr-page VR_PORT
port_free "$CAMERA_PORT" dt01-camera-page CAMERA_PORT

say "Installing services for the account '$KIT_USER' (sudo)"
sudo -v || fail "sudo is needed to install the services."

# Device access: the lift is a USB serial device, the cameras are video devices.
sudo usermod -aG dialout,video,plugdev "$KIT_USER"

# One root-owned helper may rename and bring up the arms' CAN adapters. It is
# installed outside this folder so the kit's account cannot rewrite it, and
# sudo is granted for that one path, plus restarting the kit's own services.
SYSTEMCTL="$(command -v systemctl)"
sudo install -o root -g root -m 0755 "$KIT/scripts/piper/can_apply.sh" "$CAN_HELPER"
sudo sed -i "s#^REPO=.*#REPO=\"$KIT\"#" "$CAN_HELPER"
{
  printf '%s ALL=(root) NOPASSWD: %s\n' "$KIT_USER" "$CAN_HELPER"
  for name in "${UNITS[@]}"; do
    printf '%s ALL=(root) NOPASSWD: %s restart %s\n' "$KIT_USER" "$SYSTEMCTL" "$name"
  done
} | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
if ! sudo visudo -c -f "$SUDOERS" >/dev/null; then
  sudo rm -f "$SUDOERS"
  fail "The sudo rule was rejected and has been removed."
fi

# A certificate for the VR page. Browsers only allow VR on https; this one is
# self-signed, so the headset shows a warning once and then remembers it.
if [[ ! -f "$CERT_DIR/vr.crt" ]]; then
  as_kit_user mkdir -p "$CERT_DIR"
  as_kit_user openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -subj "/CN=dt01-pro" \
    -keyout "$CERT_DIR/vr.key" -out "$CERT_DIR/vr.crt" >/dev/null 2>&1 \
    || fail "Could not create the VR page certificate (is openssl installed?)."
  as_kit_user chmod 600 "$CERT_DIR/vr.key"
fi

unit() {  # unit <name> <description> <working dir> <extra [Service] line> <command...>
  local name="$1" description="$2" workdir="$3" extra="$4"; shift 4
  sudo tee "/etc/systemd/system/$name.service" >/dev/null <<UNIT
[Unit]
Description=$description
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$KIT_USER
WorkingDirectory=$workdir
ExecStart=$*
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$KIT
SupplementaryGroups=dialout video plugdev
Restart=always
RestartSec=5
TimeoutStopSec=20
$extra

[Install]
WantedBy=multi-user.target
UNIT
}

SYSPY="$(command -v python3)"
# KillMode=process: stopping or restarting the panel must never signal the jobs
# it supervises. The teleop job only parks the arms cleanly on Ctrl+C (SIGINT),
# which is how the panel itself stops it.
unit dt01-panel "DT-01 Pro control panel" "$KIT" "KillMode=process" \
  "$SYSPY" -u "$KIT/ops/server.py" --host 0.0.0.0 --port "$PANEL_PORT"
unit dt01-base "DT-01 Pro base and lift dashboard" "$KIT/base" "" \
  "$SYSPY" dashboard/server.py --robot "$BASE_IP" --bind 0.0.0.0 --port "$BASE_PORT" --stream-hz 1
unit dt01-vr-page "DT-01 Pro VR page" "$KIT" "" \
  "$PY" -u -m arms.livekit_rt.livekit_xr_server --port "$VR_PORT" \
  --cert-file "$CERT_DIR/vr.crt" --key-file "$CERT_DIR/vr.key"
unit dt01-camera-page "DT-01 Pro camera page" "$KIT" "" \
  /usr/bin/env "LIVEKIT_VIEWER_PORT=$CAMERA_PORT" "$PY" -u -m arms.livekit_rt.livekit_camera_dashboard.server

sudo systemctl daemon-reload
for name in "${UNITS[@]}"; do
  sudo systemctl enable --quiet "$name"
  sudo systemctl restart "$name"
done
sleep 2

IP="$(hostname -I | awk '{print $1}')"
say "Done. Open these from a laptop on the same network:"
echo "    Control panel   http://$IP:$PANEL_PORT"
echo "    Base + lift     http://$IP:$BASE_PORT"
echo "    Cameras         http://$IP:$CAMERA_PORT"
echo "    VR page         https://$IP:$VR_PORT   (open this one in the headset)"
for name in "${UNITS[@]}"; do
  systemctl is-active --quiet "$name" \
    || echo "    !! $name is not running yet: journalctl -u $name -n 20"
done
echo
echo "    Next: README.md, 'Drive the base and the lift'."
