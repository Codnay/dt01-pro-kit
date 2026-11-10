#!/bin/bash
# Assign CAN adapters to interface names, as root, in one shot.
#
#   can_apply.sh <bitrate> <name>=<bus> [<name>=<bus> ...]
#   can_apply.sh 1000000 can_right=3-2:1.0 can_left=3-1:1.0
#
# Why this exists rather than calling can_activate.sh directly:
#
#   * Swapping the two arms is not a sequence of independent renames.
#     `ip link set X name can_right` fails with EEXIST while another interface
#     still holds that name, so every name that is in the way is parked on a
#     temporary name FIRST, and only then are the real names handed out.
#
#   * It is the single command the ops dashboard may run under sudo. Install it
#     with setup.sh, which copies it to /usr/local/sbin/dt01-can-apply
#     root-owned: a helper the dashboard user could rewrite would turn one
#     NOPASSWD entry into a root shell.
#
# Configuration of each interface (bitrate, restart-ms, bring-up) is still done
# by scripts/piper/can_activate.sh, which stays the single source of truth for
# what a correctly configured PiPER CAN link looks like.
set -euo pipefail

REPO="${PIPER_REPO:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
ACTIVATE="$REPO/scripts/piper/can_activate.sh"

if [[ $EUID -ne 0 ]]; then
  echo "can_apply: must run as root" >&2
  exit 1
fi
if [[ ! -f "$ACTIVATE" ]]; then
  echo "can_apply: cannot find $ACTIVATE (set PIPER_REPO)" >&2
  exit 1
fi
if [[ $# -lt 2 ]]; then
  echo "usage: can_apply.sh <bitrate> <name>=<bus> [...]" >&2
  exit 2
fi

BITRATE="$1"; shift
[[ "$BITRATE" =~ ^[0-9]+$ ]] || { echo "can_apply: bad bitrate $BITRATE" >&2; exit 2; }

NAMES=(); BUSES=()
for spec in "$@"; do
  name="${spec%%=*}"; bus="${spec#*=}"
  # Validated again here, not only in the caller: this runs as root and must
  # not depend on the dashboard for its safety.
  [[ "$name" =~ ^[a-zA-Z][a-zA-Z0-9_]{0,14}$ ]] || { echo "can_apply: bad name $name" >&2; exit 2; }
  [[ "$bus"  =~ ^[0-9]+-[0-9.]+:[0-9]+\.[0-9]+$ ]] || { echo "can_apply: bad bus $bus" >&2; exit 2; }
  NAMES+=("$name"); BUSES+=("$bus")
done

bus_of() {  # current USB path of an interface, straight from sysfs
  local iface="$1" link
  link="$(readlink -f "/sys/class/net/$iface/device" 2>/dev/null || true)"
  [[ -n "$link" ]] && basename "$link" || true
}

iface_for_bus() {
  local want="$1" iface
  for iface in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
    [[ "$(bus_of "$iface")" == "$want" ]] && { echo "$iface"; return; }
  done
}

echo "=== can_apply: bitrate $BITRATE ==="
for i in "${!NAMES[@]}"; do
  echo "  want ${NAMES[$i]} on USB ${BUSES[$i]}"
done

# Every adapter we are about to rename, so we never park an unrelated link.
declare -A WANTED_FOR_IFACE=()
for i in "${!NAMES[@]}"; do
  iface="$(iface_for_bus "${BUSES[$i]}")"
  if [[ -z "$iface" ]]; then
    echo "Error: no CAN adapter found on USB port ${BUSES[$i]}" >&2
    exit 1
  fi
  WANTED_FOR_IFACE["$iface"]="${NAMES[$i]}"
done

# Pass 1: park any interface that is squatting on a name we need.
PARKED=0
for iface in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
  for want in "${NAMES[@]}"; do
    if [[ "$iface" == "$want" && "${WANTED_FOR_IFACE[$iface]:-}" != "$iface" ]]; then
      tmp="canpark$PARKED"
      while ip link show "$tmp" &>/dev/null; do PARKED=$((PARKED+1)); tmp="canpark$PARKED"; done
      echo "  parking $iface -> $tmp (it holds a name we need)"
      ip link set "$iface" down || true
      ip link set "$iface" name "$tmp"
      PARKED=$((PARKED+1))
      break
    fi
  done
done

# Pass 2: hand out the real names, re-resolving each adapter by USB path since
# pass 1 may have renamed it.
STATUS=0
for i in "${!NAMES[@]}"; do
  echo "--- ${NAMES[$i]} <- USB ${BUSES[$i]} ---"
  if ! bash "$ACTIVATE" "${NAMES[$i]}" "$BITRATE" "${BUSES[$i]}"; then
    echo "Error: failed to configure ${NAMES[$i]} on ${BUSES[$i]}" >&2
    STATUS=1
  fi
done

echo "=== can_apply: done (exit $STATUS) ==="
exit $STATUS
