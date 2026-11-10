#!/usr/bin/env python3
"""DT-01 dashboard server: one origin in front of the robot's HTTP + WS API.

    python3 dashboard/server.py --robot 192.168.1.15
    python3 dashboard/server.py --mock            # no robot needed

The browser never talks to the controller directly. Everything goes through
here, which buys three things worth the hop: one place that owns the motion
deadman, one cached picture of robot state instead of every panel polling the
same endpoint, and no dependence on whether the vendor's server sends CORS
headers.

Stdlib only — on the robot's own access point there is no internet to install
anything from, and that is exactly where this has to run.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import mimetypes
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import control
import protection
import drive
import drobot_api
import lift_serial
import identity
import metrics
from drive import SPEED_PRESETS, DriveLoop, lift_gate, motion_gate
from drobot_api import DRobotError, explain
from telemetry import RobotLink

STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAM_HZ = 5.0
STREAM_KEEPALIVE_S = 15.0

# The dashboard is often reached over a phone hotspot from the other side of the
# world, where every kilobyte is metered and the round trip is over a second.
# The stream is therefore trimmed rather than shipping the raw snapshot:
#
#   - fields the browser never reads are dropped (the raw ultrasonic/infrared
#     arrays are already folded into `obstacles.sensors`; the laser's
#     robot-frame `points` are unused because the map draws `gridpoint`),
#   - floats are rounded, because 0.20000000298023224 costs 19 bytes to say 0.2,
#   - a dense laser return is decimated to something a screen can resolve,
#   - and only changed top-level keys go out, with a periodic full resync.
STREAM_DROP = ("ultrasonic", "irsensor")
SCAN_WIRE_KEYS = ("gridpoint", "mapinfo", "at")
SCAN_MAX_POINTS = 360
STREAM_RESYNC_S = 10.0
FLOAT_DIGITS = 4


def _compact(value: Any) -> Any:
    """Round floats recursively. Display precision, not storage precision."""
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _thin_scan(scan: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only what the map draws, at a density a screen can actually show."""
    if not scan:
        return scan
    trimmed = {key: scan[key] for key in SCAN_WIRE_KEYS if key in scan}
    points = trimmed.get("gridpoint") or []
    if len(points) > SCAN_MAX_POINTS:
        stride = len(points) // SCAN_MAX_POINTS + 1
        trimmed["gridpoint"] = points[::stride]
        trimmed["decimated_from"] = len(points)
    return trimmed


def wire_frame(snapshot: dict[str, Any], want_scan: bool = True) -> dict[str, Any]:
    """The snapshot as the browser needs it: trimmed, thinned and rounded."""
    frame = {key: value for key, value in snapshot.items() if key not in STREAM_DROP}
    if want_scan:
        frame["scan"] = _thin_scan(frame.get("scan"))
    else:
        frame.pop("scan", None)
    return _compact(frame)


class ApiError(Exception):
    """A request the dashboard itself rejects, with an HTTP status."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Session:
    """The dashboard's connection to one robot: telemetry plus the drive loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.link: RobotLink | None = None
        self.drive: DriveLoop | None = None
        self.lease = control.ControlLease()
        # Its own device on its own cable — deliberately not tied to the robot
        # link, so a controller outage does not take the column with it.
        self.lift = lift_serial.LiftLink()
        self.lift_ready = False
        # Server-side so every viewer sees the same numbers and a reload does
        # not silently reset them. They were browser-local, which meant two
        # people could be driving to different limits while reading the same
        # screen.
        self.settings: dict[str, float] = dict(SPEED_PRESETS["normal"])

    def _gate_view(self, link: RobotLink) -> dict[str, Any]:
        """A snapshot shaped for the gates, cheap enough for the 10 Hz loop.

        Not Session.snapshot(): that one drains the lease's change flag, which
        belongs to the stream path, and does more work than a gate needs.
        Reading the lease here is also what expires it — so a holder who goes
        silent stops the robot through the loop that is already running.
        """
        return {**link.snapshot(), "control": self.lease.status()}

    def connect(self, host: str, port: int) -> dict[str, Any]:
        with self._lock:
            self._teardown()
            link = RobotLink(host, port)
            link.start()
            drive = DriveLoop(link.client, lambda: motion_gate(self._gate_view(link)))
            drive.start()
            self.link, self.drive = link, drive
        # Probe before claiming anything. Starting pollers against a dead
        # address succeeds quietly, so "connected" has to mean the controller
        # answered, not that we wrote its IP down.
        reachable, error = True, None
        try:
            link.client.state()
        except DRobotError as exc:
            reachable, error = False, explain(exc)
        # Pull the parameter table once so speed levels and sensor thresholds
        # are populated before the operator opens a panel.
        if reachable:
            try:
                link.params(refresh=True)
            except DRobotError:
                pass
        return {"host": host, "port": port, "reachable": reachable, "error": error}

    def disconnect(self) -> None:
        with self._lock:
            self._teardown()

    def _teardown(self) -> None:
        if self.drive is not None:
            # Stop the robot before dropping the link, never after.
            try:
                self.drive.shutdown()
            except Exception:                             # noqa: BLE001
                traceback.print_exc()
        if self.link is not None:
            self.link.stop()
        self.link, self.drive = None, None

    # --- accessors that refuse to half-work --------------------------------
    def require_link(self) -> RobotLink:
        link = self.link
        if link is None:
            raise ApiError("not connected to a robot", 409)
        return link

    def require_drive(self) -> DriveLoop:
        drive = self.drive
        if drive is None:
            raise ApiError("not connected to a robot", 409)
        return drive

    def snapshot(self) -> dict[str, Any]:
        link, drive = self.link, self.drive
        if link is None or drive is None:
            return {"connected": False}
        state = link.snapshot()
        state["connected"] = True
        state["drive"] = drive.status()
        state["lift"] = self.lift.status() if self.lift_ready else {
            "connected": False, "error": "lift serial port not opened",
            "min_mm": lift_serial.MIN_HEIGHT_MM, "max_mm": lift_serial.MAX_HEIGHT_MM}
        state["control"] = self.lease.status()
        state["settings"] = dict(self.settings)
        state["motion_blocked_by"] = motion_gate(state)
        state["lift_blocked_by"] = lift_gate(state)
        state["speed_presets"] = SPEED_PRESETS
        # A handover must not inherit the previous operator's motion, and an
        # expired lease must not leave the base coasting on its last command.
        if self.lease.take_changed():
            drive.stop()
        return state


SESSION = Session()


# ---------------------------------------------------------------------------
# Route handlers. Each takes (body, query) and returns a JSON-able object.
# ---------------------------------------------------------------------------

def _text(body: dict[str, Any], key: str, required: bool = True) -> str:
    value = body.get(key)
    if value is None or value == "":
        if required:
            raise ApiError(f"{key!r} is required")
        return ""
    return str(value)


def _number(body: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(body.get(key, default))
    except (TypeError, ValueError):
        raise ApiError(f"{key!r} must be a number") from None


def api_connect(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    host = _text(body, "host")
    port = int(body.get("port") or drobot_api.DEFAULT_PORT)
    return SESSION.connect(host, port)


def api_disconnect(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    SESSION.disconnect()
    return {"connected": False}


def api_discover(_body: dict[str, Any], query: dict[str, list[str]]) -> Any:
    """Scan for the controller on whatever LAN this machine is actually on."""
    subnets = query.get("subnet") or drobot_api.local_subnets()
    found: list[dict[str, Any]] = []
    for subnet in subnets[:4]:                # a /24 sweep each; four is plenty
        for host in drobot_api.discover(subnet):
            found.append({"host": host, "subnet": subnet})
    return {"hosts": [entry["host"] for entry in found], "scanned": subnets[:4]}


def api_subnets(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return {"subnets": drobot_api.local_subnets()}


def api_state(_body: dict[str, Any], query: dict[str, list[str]]) -> Any:
    """The same frame the stream sends, for a first paint or a one-off check."""
    want_scan = (query.get("scan") or ["1"])[0] not in ("0", "false", "")
    return wire_frame(SESSION.snapshot(), want_scan)


def api_preflight(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return SESSION.require_link().preflight()


def api_settings(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Adjust dashboard-side settings in place.

    Deliberately not part of /api/connect: changing a display threshold must
    never tear down the link and the drive loop underneath a moving robot.
    """
    link = SESSION.require_link()
    if "alarm_distance" in body:
        value = _number(body, "alarm_distance", 0.35)
        if not math.isfinite(value) or not 0.1 <= value <= 1.5:
            raise ApiError("obstacle stop distance must be finite and between 0.1 and 1.5 m")
        link.alarm_distance = value
    if "sensor_stale_s" in body:
        value = _number(body, "sensor_stale_s")
        if not math.isfinite(value) or not 0.5 <= value <= 5:
            raise ApiError("sensor freshness limit must be between 0.5 and 5 seconds")
        link.sensor_stale_s = value
    # Speed lives here rather than in the page so every viewer reads the same
    # limits and a reload does not reset them. Clamped to the same ceilings the
    # drive loop enforces, so a hand-written request cannot raise them.
    if "linear" in body:
        SESSION.settings["linear"] = max(0.05, min(drive.MAX_LINEAR,
                                                   _number(body, "linear", 0.2)))
    if "angular" in body:
        SESSION.settings["angular"] = max(0.05, min(drive.MAX_ANGULAR,
                                                    _number(body, "angular", 0.45)))
    return {"alarm_distance": link.alarm_distance, "sensor_stale_s": link.sensor_stale_s, **SESSION.settings}


# --- movement --------------------------------------------------------------

def api_drive(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return SESSION.require_drive().request(_number(body, "linear"), _number(body, "angular"))


def api_drive_stop(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return SESSION.require_drive().stop()


def api_estop(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    drive, link = SESSION.require_drive(), SESSION.require_link()
    if body.get("engaged", True):
        result = drive.emergency_stop()
        if result.get("braked"):
            link.note_brake(True)
        return result
    result = drive.release_brake()
    if result.get("released"):
        link.note_brake(False)
    return result


def api_control_take(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Claim control. ``force`` displaces a live holder, deliberately."""
    client = _text(body, "client")
    took, reason = SESSION.lease.take(client, body.get("name"),
                                      force=bool(body.get("force")))
    if not took:
        raise ApiError(reason, 409)
    # Never inherit the previous operator's motion across a handover.
    if SESSION.drive is not None:
        SESSION.drive.stop()
    return SESSION.lease.status()


def api_control_release(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    released = SESSION.lease.release(_text(body, "client"))
    if released and SESSION.drive is not None:
        SESSION.drive.stop()
    return {**SESSION.lease.status(), "released": released}


def api_control_heartbeat(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Renew the lease. Reports loss rather than erroring, so the page can react.

    A browser that has been asleep needs to *learn* it lost control, not get an
    exception it might swallow — the UI turns read-only off the back of this.
    """
    alive = SESSION.lease.heartbeat(_text(body, "client"))
    return {**SESSION.lease.status(), "still_holding": alive}


def _lift() -> Any:
    if not SESSION.lift_ready:
        raise ApiError("the lifting column's serial port is not open — check the "
                       "cable and that the dashboard's user is in 'dialout'", 503)
    return SESSION.lift


def _lift_blocked(direction: int) -> str | None:
    """Both gates: the robot's condition, then the column's own."""
    link = SESSION.link
    if link is not None:
        snapshot = dict(link.snapshot())
        snapshot["drive"] = SESSION.require_drive().status()
        snapshot["control"] = SESSION.lease.status()
        robot = drive.lift_gate(snapshot)
        if robot:
            return robot
    return lift_serial.gate(SESSION.lift.status(), direction)


def api_lift_move(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Move the column a distance in mm, or to an absolute height in mm.

    Millimetres throughout. The motor percentage is a duty cycle with no
    documented relationship to speed, so it is an optional refinement rather
    than the unit an operator has to think in.
    """
    lift = _lift()
    percent = int(_number(body, "percent", 30))
    if "to_mm" in body:
        target = int(_number(body, "to_mm"))
        direction = 1 if target > (lift.status().get("height_mm") or 0) else -1
        blocked = _lift_blocked(direction)
        if blocked:
            raise ApiError(blocked, 409)
        result = lift.seek(target, percent)
    elif "by_mm" in body:
        delta = int(_number(body, "by_mm"))
        blocked = _lift_blocked(1 if delta > 0 else -1)
        if blocked:
            raise ApiError(blocked, 409)
        result = lift.nudge(delta, percent)
    else:
        raise ApiError('body needs "by_mm" (relative) or "to_mm" (absolute)')
    if not result.get("accepted"):
        raise ApiError(result.get("reason", "the column refused the move"), 409)
    return result


def api_lift_hold(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Hold-to-move. Expires on its own — the column latches its last velocity."""
    lift = _lift()
    percent = int(_number(body, "percent"))
    if percent:
        blocked = _lift_blocked(1 if percent > 0 else -1)
        if blocked:
            raise ApiError(blocked, 409)
    return lift.hold(percent)


def api_lift_stop(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return _lift().stop()


def api_mode(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Walk the Idle <-> Localization <-> RunningTask chain to a target state."""
    link = SESSION.require_link()
    target = _text(body, "state")
    if target not in drobot_api.STATE_CHAIN:
        raise ApiError(f"{target!r} is not one of {drobot_api.STATE_CHAIN}")
    # Never change modes with the base moving.
    SESSION.require_drive().stop()
    steps: list[str] = []
    try:
        drobot_api.walk_to_state(link.client, target, steps.append)
    except DRobotError as exc:
        raise ApiError(explain(exc), 409) from exc
    return {"state": target, "steps": [step.strip() for step in steps]}


def api_auto_mode(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    mode = int(_number(body, "mode"))
    if mode not in (0, 1):
        raise ApiError("mode must be 0 (autonomous) or 1 (remote driving)")
    SESSION.require_link().client.set_auto_mode(mode)
    return {"mode": mode}


# --- points and navigation -------------------------------------------------

def _live_state(link: RobotLink) -> dict[str, Any]:
    """Read state and brake straight from the controller, not from the cache.

    The polled snapshot can be a second stale. That is fine for a display and
    for the drive loop, whose deadman bounds the exposure — but a command that
    hands the base to the navigator gets a fresh answer.
    """
    return {"state": link.client.state(), "braked": link.client.brake_engaged()}


def api_points(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    poses = link.client.poses()
    for pose in poses:
        pose["type_name"] = drobot_api.POSE_TYPE_NAMES.get(pose.get("type"), "unknown")
    return {"points": poses, "types": drobot_api.POSE_TYPES}


def api_point_save(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    name = _text(body, "name")
    kind = _text(body, "type", required=False) or "nav"
    if kind not in drobot_api.POSE_TYPES:
        raise ApiError(f"type must be one of {sorted(drobot_api.POSE_TYPES)}")
    if not link.localized():
        raise ApiError("the robot is not localised — initialise its position first", 409)
    if link.snapshot()["robot"].get("state_operation") != "RunningTask":
        drobot_api.walk_to_state(link.client, "RunningTask", lambda _message: None)
    link.client.add_current_pose(name, drobot_api.POSE_TYPES[kind])
    return {"name": name, "type": kind}


def api_goto(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Start a navigate-to-point task. Returns as soon as the task is accepted."""
    link = SESSION.require_link()
    name = _text(body, "name")
    live = _live_state(link)
    if live["braked"]:
        raise ApiError("software emergency stop is engaged — release it first", 409)
    if not link.localized():
        raise ApiError("the robot is not localised — it cannot navigate", 409)
    # Teleop and the navigator must never drive at the same time.
    SESSION.require_drive().stop()
    if live["state"].get("state_operation") != "RunningTask":
        drobot_api.walk_to_state(link.client, "RunningTask", lambda _message: None)
    link.client.move_to(name)
    return {"navigating_to": name}


def api_nav_stop(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    errors = []
    for _ in range(3):                     # one packet on a lossy link is not a stop
        try:
            link.client.stop_task()
        except DRobotError as exc:
            errors.append(explain(exc))
    return {"stopped": not errors, "errors": errors}


def api_nav_path(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    return SESSION.require_link().client.realtime_path()


def api_localize(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    turning = bool(body.get("turning"))
    SESSION.require_drive().stop()
    if link.client.state().get("state_operation") != "RunningTask":
        drobot_api.walk_to_state(link.client, "RunningTask", lambda _message: None)
    point = body.get("point")
    if point:
        link.client.initialize_at_point(str(point), turning)
        return {"initializing_at": point, "turning": turning}
    link.client.initialize_at_pose(_number(body, "x"), _number(body, "y"),
                                   _number(body, "angle"), turning)
    return {"initializing_at": {"x": body.get("x"), "y": body.get("y")}, "turning": turning}


def api_maps(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    current = ""
    try:
        current = link.client.current_map().get("name", "")
    except DRobotError:
        pass
    return {"maps": link.client.maps(), "current": current}


# --- charging --------------------------------------------------------------

def api_charge(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    state = link.snapshot()
    battery = dict(state.get("battery") or {})
    chassis = state.get("chassis") or {}
    battery["volts"] = drobot_api.normalize_voltage(battery.get("battery_voltage"))
    result: dict[str, Any] = {
        "battery": battery,
        "chassis": chassis,
        # True / False / null. Null means the robot reports no usable charge
        # signal at all, which is the truth on firmware whose current sensor
        # reads a constant zero — not the same as "not charging".
        "charging": drobot_api.charging_state(battery, chassis),
        "autocharge": None,
        # The auto-charge endpoints are marked 未启用 (not enabled) in the API
        # reference. If they answer, use them; if not, say so rather than
        # showing a switch that quietly does nothing.
        "autocharge_available": False,
        "autocharge_error": None,
    }
    try:
        result["charging_pile"] = link.client.charging_pile()
    except DRobotError as exc:
        result["charging_pile"] = None
        result["charging_pile_error"] = explain(exc)
    try:
        status = link.client.autocharge()
        result["autocharge"] = {
            "enabled": status.get("AutoChargeSwitch"),
            "point": status.get("PointName"),
            "state": status.get("ChargeState"),
            "state_text": drobot_api.CHARGE_STATES.get(str(status.get("ChargeState")),
                                                       "unknown"),
        }
        result["autocharge_available"] = True
    except DRobotError as exc:
        result["autocharge_error"] = explain(exc)
    return result


def api_charge_auto(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    enabled = bool(body.get("enabled"))
    point = _text(body, "point", required=enabled)
    if enabled:
        poses = {pose.get("name"): pose for pose in link.client.poses()}
        if point not in poses:
            raise ApiError(f"no point named {point!r} on this map")
        if poses[point].get("type") != drobot_api.POSE_TYPES["charge"]:
            raise ApiError(f"{point!r} is not a charge point — the dock pose must be "
                           f"saved with type 'charge'")
    try:
        link.client.set_autocharge(enabled, point)
    except DRobotError as exc:
        raise ApiError(explain(exc), 502) from exc
    return {"enabled": enabled, "point": point}


def api_dock_now(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Send the robot to a charge point right now, as a plain navigation task."""
    link = SESSION.require_link()
    point = body.get("point")
    if not point:
        charge = [pose for pose in link.client.poses()
                  if pose.get("type") == drobot_api.POSE_TYPES["charge"]]
        if len(charge) != 1:
            raise ApiError("name the charge point: this map has "
                           f"{len(charge)} of them", 400)
        point = charge[0]["name"]
    return api_goto({"name": point}, {})


# --- parameters ------------------------------------------------------------

def api_params(_body: dict[str, Any], query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    refresh = (query.get("refresh") or ["0"])[0] not in ("0", "", "false")
    return {
        "params": link.params(refresh=refresh),
        "error": link.params_error(),
        "groups": ["speed", "obstacle", "charge", "battery"],
    }


def api_protection_enable(_body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    """Enable supported safety switches, with the base already immobilized.

    No automatic brake release or movement, including on a partial failure.
    """
    link = SESSION.require_link()
    state = link.snapshot()
    physical_stop = not state.get("safety_stale") and (state.get("bumper") or {}).get("estop") is True
    if not physical_stop and not link.client.brake_engaged():
        raise ApiError("engage the physical emergency stop or software brake before enabling protection", 409)
    if (state.get("nav") or {}).get("nav_status_type") not in ("idle", None):
        raise ApiError("stop the navigation task before configuring protection", 409)
    SESSION.require_drive().stop()
    try:
        table = link.client.param_table()
        updates = protection.enable_updates(table)
        before = {u["namespace"]: table[u["namespace"]].value for u in updates}
        link.client.write_params(updates)
        table = link.client.param_table()
        protection.verify_updates(table, updates)
        with link._lock:
            link._apply_params(table)
    except (ValueError, DRobotError) as exc:
        # Keep the cache invalid until a successful controller refresh. Never
        # undo successful safety enables merely because another write failed.
        with link._lock:
            link._params_error = str(exc)
            link._params_seen_at = None
        raise ApiError(f"protection could not be verified; keep the robot stopped: {exc}", 409) from exc
    print("obstacle protection enabled: " + json.dumps({"before": before, "after": updates}), flush=True)
    return {"configured": True, "updated": updates,
            "unavailable": [name for name in protection.OPTIONAL if name not in table],
            "obstacles": link.snapshot()["obstacles"], "brake_released": False}


def api_params_update(body: dict[str, Any], _query: dict[str, list[str]]) -> Any:
    link = SESSION.require_link()
    updates = body.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ApiError("'updates' must be a non-empty list")
    payload = []
    for update in updates:
        namespace = str(update.get("namespace") or "")
        known = drobot_api.PARAMS_BY_NAMESPACE.get(namespace)
        if known is None:
            raise ApiError(f"{namespace!r} is not a parameter this dashboard exposes")
        value = update.get("value")
        if namespace in protection.SWITCHES:
            values = value if isinstance(value, list) else [value]
            if not values or not all(protection.is_on(v) for v in values):
                raise ApiError("obstacle protection must remain enabled, including during manual driving", 409)
            raise ApiError("use Enable obstacle protection to configure and verify safety switches", 409)
        # Values go over the wire as strings; array parameters as lists of them.
        if known.is_array:
            if not isinstance(value, list):
                raise ApiError(f"{namespace} is an array parameter — send a list")
            encoded: Any = [_encode_param(known.type, item) for item in value]
        else:
            encoded = _encode_param(known.type, value)
        payload.append({"type": known.type, "namespace": namespace, "value": encoded})
    link.client.write_params(payload)
    return {"updated": [entry["namespace"] for entry in payload],
            "params": link.params(refresh=True)}


def _encode_param(kind: str, value: Any) -> str:
    text = str(value).strip()
    try:
        if kind == "int":
            return str(int(float(text)))
        if kind == "double":
            return f"{float(text):.6f}"
        if kind == "bool":
            return "true" if text.lower() in ("true", "1", "on", "yes") else "false"
    except ValueError:
        raise ApiError(f"{text!r} is not a valid {kind}") from None
    return text


ROUTES: dict[tuple[str, str], Callable[..., Any]] = {
    ("GET", "/api/state"): api_state,
    ("GET", "/api/preflight"): api_preflight,
    ("GET", "/api/discover"): api_discover,
    ("GET", "/api/subnets"): api_subnets,
    ("GET", "/api/points"): api_points,
    ("GET", "/api/nav/path"): api_nav_path,
    ("GET", "/api/maps"): api_maps,
    ("GET", "/api/charge"): api_charge,
    ("GET", "/api/params"): api_params,
    ("POST", "/api/connect"): api_connect,
    ("POST", "/api/settings"): api_settings,
    ("POST", "/api/disconnect"): api_disconnect,
    ("POST", "/api/drive"): api_drive,
    ("POST", "/api/drive/stop"): api_drive_stop,
    ("POST", "/api/estop"): api_estop,
    ("POST", "/api/control/take"): api_control_take,
    ("POST", "/api/control/release"): api_control_release,
    ("POST", "/api/control/heartbeat"): api_control_heartbeat,
    ("POST", "/api/lift/move"): api_lift_move,
    ("POST", "/api/lift/hold"): api_lift_hold,
    ("POST", "/api/lift/stop"): api_lift_stop,
    ("POST", "/api/mode"): api_mode,
    ("POST", "/api/auto_mode"): api_auto_mode,
    ("POST", "/api/points/save"): api_point_save,
    ("POST", "/api/nav/goto"): api_goto,
    ("POST", "/api/nav/stop"): api_nav_stop,
    ("POST", "/api/localize"): api_localize,
    ("POST", "/api/charge/auto"): api_charge_auto,
    ("POST", "/api/charge/dock"): api_dock_now,
    ("POST", "/api/params"): api_params_update,
    ("POST", "/api/protection/enable"): api_protection_enable,
}


# POSTs that anyone may call, lease or no lease. The list is deliberately short
# and every entry either STOPS the robot or manages the lease itself: someone
# watching a robot about to hit something must be able to stop it without first
# negotiating for control. Everything else is protected, and a route added later
# is protected by default — the safe direction to be wrong in.
OPEN_POSTS = frozenset({
    "/api/estop",            # emergency stop, always
    "/api/drive/stop",       # explicit stop
    "/api/nav/stop",         # cancel a navigation goal
    "/api/lift/stop",        # stop the lifting column
    "/api/control/take",     # how you get the lease in the first place
    "/api/control/release",
    "/api/control/heartbeat",
})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DT01Dashboard/1.0"

    # --- plumbing ----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:                           # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _metrics(self) -> None:
        """Serve a Prometheus scrape.

        Never fails on a disconnected robot: a scrape that errors would leave
        Grafana unable to tell "the robot is down" from "the exporter is down",
        and the first of those is the one worth alerting on.
        """
        link = SESSION.link
        try:
            snapshot = link.snapshot() if link is not None else None
        except Exception:                                 # noqa: BLE001
            snapshot = None
        # getattr, not an attribute: the lockout postdates some deployed
        # builds and a missing flag means "teleop was never locked out".
        body = metrics.render(snapshot, self.server.robot_id,  # type: ignore[attr-defined]
                              getattr(drive, "TELEOP_DISABLED", False)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ApiError(f"body is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ApiError("body must be a JSON object")
        return parsed

    # --- dispatch ----------------------------------------------------------
    def do_GET(self) -> None:                             # noqa: N802
        parts = urlsplit(self.path)
        if parts.path == "/api/stream":
            self._stream(parse_qs(parts.query))
            return
        if parts.path == "/api/map.png":
            self._map_png(parse_qs(parts.query))
            return
        if parts.path == "/metrics":
            self._metrics()
            return
        if parts.path.startswith("/api/"):
            self._dispatch("GET", parts.path, {}, parse_qs(parts.query))
            return
        self._static(parts.path)

    def do_POST(self) -> None:                            # noqa: N802
        parts = urlsplit(self.path)
        try:
            body = self._read_body()
        except ApiError as exc:
            self._send_json({"error": str(exc)}, exc.status)
            return
        self._dispatch("POST", parts.path, body, parse_qs(parts.query))

    def _dispatch(self, method: str, path: str, body: dict[str, Any],
                  query: dict[str, list[str]]) -> None:
        handler = ROUTES.get((method, path))
        if handler is None:
            self._send_json({"error": f"no route for {method} {path}"}, 404)
            return
        if method == "POST" and path not in OPEN_POSTS:
            client = self.headers.get("X-Control-Token") or body.get("client") or ""
            if not SESSION.lease.is_holder(client):
                holder = SESSION.lease.status()
                who = f"{holder['name']} has control" if holder["held"] \
                    else "nobody has taken control"
                self._send_json({"ok": False, "error": f"{who} — press Take control"},
                                409)
                return
        try:
            self._send_json({"ok": True, "data": handler(body, query)})
        except ApiError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
        except DRobotError as exc:
            self._send_json({"ok": False, "error": explain(exc)}, 502)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    # --- streaming ---------------------------------------------------------
    def _stream(self, query: dict[str, list[str]]) -> None:
        """Server-sent events: the UI's only source of live state.

        Each message is `{"full": bool, "d": {...}}`. A full frame carries the
        whole trimmed snapshot; a delta carries only the top-level keys that
        changed. A full frame is sent on connect and every STREAM_RESYNC_S after,
        so a client that joins late — or reconnects, which EventSource does by
        itself — always converges without special handling.
        """
        want_scan = (query.get("scan") or ["1"])[0] not in ("0", "false", "")
        try:
            rate = max(0.2, min(20.0, float((query.get("hz") or [""])[0])))
        except ValueError:
            rate = self.server.stream_hz                  # type: ignore[attr-defined]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        period = 1.0 / rate
        previous: dict[str, Any] = {}
        last_full = 0.0
        last_keepalive = time.monotonic()
        try:
            while not self.server.shutting_down.is_set():  # type: ignore[attr-defined]
                now = time.monotonic()
                frame = wire_frame(SESSION.snapshot(), want_scan)
                full = not previous or (now - last_full) > STREAM_RESYNC_S
                if full:
                    body, last_full = {"full": True, "d": frame}, now
                else:
                    changed = {key: value for key, value in frame.items()
                               if previous.get(key) != value}
                    body = {"full": False, "d": changed}
                previous = frame
                self.wfile.write(f"data: {json.dumps(body)}\n\n".encode())
                self.wfile.flush()
                if now - last_keepalive > STREAM_KEEPALIVE_S:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_keepalive = now
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass          # the operator closed the tab; the deadman covers the robot

    def _map_png(self, query: dict[str, list[str]]) -> None:
        try:
            link = SESSION.require_link()
            name = (query.get("name") or [""])[0]
            if not name:
                name = link.client.current_map().get("name", "")
            if not name:
                raise ApiError("no map is running and no name was given", 404)
            image = link.client.map_png(name)
        except (ApiError, DRobotError) as exc:
            status = getattr(exc, "status", 502)
            self._send_json({"ok": False, "error": explain(exc)
                             if isinstance(exc, DRobotError) else str(exc)}, status)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(image)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(image)

    # --- static ------------------------------------------------------------
    def _static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send_json({"error": f"not found: {path}"}, 404)
            return
        body = target.read_bytes()
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], verbose: bool = False,
                 stream_hz: float = STREAM_HZ, robot_id: str = "dt01-01") -> None:
        super().__init__(address, Handler)
        self.verbose = verbose
        self.stream_hz = stream_hz
        self.robot_id = robot_id
        self.shutting_down = threading.Event()


def _shutdown() -> None:
    """Stop the robot and the column, then let the process go."""
    SESSION.disconnect()
    try:
        SESSION.lift.close()
    except Exception:                                     # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", help="controller IP to connect to at startup")
    parser.add_argument("--robot-port", type=int, default=drobot_api.DEFAULT_PORT)
    parser.add_argument("--bind", default="127.0.0.1",
                        help="dashboard listen address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8800, help="dashboard port")
    parser.add_argument("--mock", action="store_true",
                        help="start the bundled mock robot and connect to it")
    parser.add_argument("--stream-hz", type=float, default=STREAM_HZ,
                        help="live-update rate (default: %(default)s). Lower it on a "
                             "metered or high-latency link; 1 is fine for monitoring")
    parser.add_argument("--robot-id", default=None,
                        help="fleet identity stamped on every Prometheus metric. "
                             "Defaults to $ROBOT_ID, then ROBOT_ID= in a .env "
                             "beside the dashboard or at the repository root. "
                             "Give each platform its own, or their histories merge")
    parser.add_argument("--lift-device", default=lift_serial.DEVICE,
                        help="serial port of the lifting column "
                             "(default: %(default)s)")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    robot_id, id_source = identity.resolve(args.robot_id)
    SESSION.lift.device = args.lift_device

    robot_host, robot_port = args.robot, args.robot_port
    if args.mock:
        import mock_robot
        mock = mock_robot.serve(port=0)
        robot_host, robot_port = "127.0.0.1", mock.server_address[1]
        print(f"  mock robot on {robot_host}:{robot_port}", flush=True)

    atexit.register(_shutdown)
    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, lambda *_: sys.exit(0))

    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        # Binding off loopback puts a drive-the-robot control on the network with
        # no password in front of it. That is sometimes what you want; it should
        # never be something you did without noticing.
        print(f"\n  !! listening on {args.bind} — anyone who can reach "
              f"{args.bind}:{args.port} can drive this robot.")
        print("  !! there is no login. Use a trusted LAN or a private overlay "
              "(Tailscale/WireGuard),")
        print("  !! never a public interface or a port-forward.\n", flush=True)

    server = DashboardServer((args.bind, args.port), verbose=args.verbose,
                             stream_hz=args.stream_hz, robot_id=robot_id)
    if robot_host:
        try:
            result = SESSION.connect(robot_host, robot_port)
        except Exception as exc:                          # noqa: BLE001
            result = {"reachable": False, "error": str(exc)}
        if result.get("reachable"):
            print(f"  connected to robot {robot_host}:{robot_port}", flush=True)
        else:
            print(f"\n  !! NO REPLY from {robot_host}:{robot_port}", flush=True)
            print(f"  !! {result.get('error')}", flush=True)
            print("  !! the robot is addressed but not answering. Check, in order:",
                  flush=True)
            print("  !!   ip -br link          the wired port must say UP, not DOWN",
                  flush=True)
            print("  !!   ip -br addr          this host needs an address on the "
                  "robot's subnet", flush=True)
            print(f"  !!   curl -s --max-time 5 http://{robot_host}:{robot_port}"
                  "/robot/cmd/get_robot_state", flush=True)
            print("  !! the dashboard still starts, and reconnects on its own once "
                  "the robot replies.\n", flush=True)

    # The column is a separate device on a separate cable: it opens whether or
    # not the robot answered, and a missing one is reported, never fatal.
    SESSION.lift_ready = SESSION.lift.open()
    if SESSION.lift_ready:
        print(f"  lifting column on {SESSION.lift.device} "
              f"({lift_serial.MIN_HEIGHT_MM}-{lift_serial.MAX_HEIGHT_MM} mm)",
              flush=True)
    else:
        print(f"  !! no lifting column: {SESSION.lift.status().get('error')}",
              flush=True)

    print(f"  dashboard on http://{args.bind}:{args.port}", flush=True)
    if args.bind in ("0.0.0.0", "::"):
        # "listening on 0.0.0.0" tells an operator nothing they can type.
        for address, kind in drobot_api.local_addresses():
            print(f"    http://{address}:{args.port}   ({kind})", flush=True)
    print(f"  metrics at http://{args.bind}:{args.port}/metrics "
          f"(robot_id={robot_id} from {id_source})", flush=True)
    if robot_id == identity.FALLBACK:
        # Not fatal — but two unidentified platforms silently share one history,
        # and the chart looks perfectly healthy while doing it.
        print(f"  !! no {identity.VARIABLE} set, so metrics are labelled "
              f"{identity.FALLBACK}. Put ROBOT_ID=<name> in a .env at the "
              "repository root", flush=True)
    print(f"  live updates at {args.stream_hz} Hz "
          f"(lower with --stream-hz on a metered link)", flush=True)
    print("  Ctrl-C stops the server and the robot", flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.shutting_down.set()
        _shutdown()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
