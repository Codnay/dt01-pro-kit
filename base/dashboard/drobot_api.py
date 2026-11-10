"""Wider DRobot HTTP client for the dashboard.

Extends the CLI client in ``tools/drobot.py`` rather than restating the wire
protocol: the base class already owns the request plumbing, the ``successed``
check, the state chain, and the point/charge-parameter vocabulary. This module
adds the endpoints a live dashboard needs and the CLI does not — raw sensors,
the map PNG, the realtime plan, and the whole parameter table with its
``isArray`` / ``limit`` metadata.

Stdlib only, like its parent. When you are on the robot's own access point you
have no internet and cannot pip install, which is exactly when this has to run.
"""

from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import drobot  # noqa: E402  (path juggling above is deliberate)

DRobotError = drobot.DRobotError
Velocity = drobot.Velocity
POSE_TYPES = drobot.POSE_TYPES
POSE_TYPE_NAMES = drobot.POSE_TYPE_NAMES
STATE_CHAIN = drobot.STATE_CHAIN
DEFAULT_PORT = drobot.DEFAULT_PORT
discover = drobot.discover
DEFAULT_SUBNET = drobot.DEFAULT_SUBNET
walk_to_state = drobot.walk_to_state

# Off-chain states the controller can also report (API doc p.21). walk_to_state
# only steps along STATE_CHAIN; these are shown to the operator as "get out of
# this in the app first".
OFF_CHAIN_STATES = ("Mapping", "FollowTarget", "Sleep", "WakingUp")

# Error codes worth translating for an operator (API doc pp.8-9).
ERROR_CODES = {
    100: "state does not match — the robot is not in a state this call allows",
    102: "cannot open file",
    103: "no message received from that sensor yet",
    200: "input parameter empty",
    201: "input parameter format error",
    202: "illegal character in parameter",
    203: "POST body empty",
    204: "parameter is invalid",
    210: "cannot call service",
    211: "service called but returned an error",
    212: "timed out waiting for service",
    801: "rotation failed",
    1002: "system error",
    1003: "no map loaded",
    1107: "path is not safe",
    1200: "requested point does not exist",
    1201: "requested point already exists",
    1300: "requested task does not exist",
    1302: "no task is running",
    1303: "a task is already running",
    1500: "map does not exist",
    1510: "no map is running",
}

# ---------------------------------------------------------------------------
# Parameter registry
#
# The controller exposes one flat namespace table (/param/get_param). These are
# the entries the dashboard surfaces, grouped by the feature they belong to.
# Namespaces are transcribed from the Chinese original (参数解释, pp.363-366),
# which is authoritative; the English translation garbles some of them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One tunable exposed in the dashboard's settings panels."""

    namespace: str
    type: str                      # int | double | bool
    label: str
    group: str                     # speed | obstacle | charge | battery
    help: str = ""
    is_array: bool = False
    unit: str = ""
    warn: str = ""                 # shown in red next to the field


PARAMS: tuple[Param, ...] = (
    # --- speed -------------------------------------------------------------
    Param("/navigation/nav/speed_level", "int", "Autonomous speed level", "speed",
          "Speed used when navigating to a point. 0/1/2 = low/medium/high."),
    Param("/navigation/follow/speed_level", "int", "Path-follow speed level", "speed",
          "Speed used when following a recorded path. 0/1/2 = low/medium/high."),
    # --- obstacle ----------------------------------------------------------
    Param("/navigation/nav/avoid_obstacle", "bool", "Avoid obstacles (navigate)", "obstacle",
          "Whether navigate-to-point plans around obstacles or only stops for them.",
          warn="Turning this off means the robot will not route around anything."),
    Param("/navigation/follow/avoid_obstacle", "bool", "Avoid obstacles (path follow)",
          "obstacle", "Same switch for the follow-path task."),
    Param("/navigation/follow/front_safe_check_dis", "double", "Front safety distance",
          "obstacle", "How far ahead the robot looks for an obstacle while following a path.",
          unit="m"),
    Param("/navigation/nav/find_next_goal_duration", "double", "Replan timeout", "obstacle",
          "Longest time spent replanning around an obstacle before giving up.", unit="s"),
    Param("/navigation/nav/goal_unsafe_quit_duration", "double", "Blocked-goal timeout",
          "obstacle", "How long to wait for a blocked goal to clear before quitting.",
          unit="s"),
    Param("/devices/ultrasonic/switch_ultrasound", "bool", "Ultrasonic enabled", "obstacle",
          "One switch per ultrasonic sensor, in sensor order.", is_array=True),
    Param("/devices/ultrasonic/max_range_ultrasound", "double", "Ultrasonic max range",
          "obstacle",
          "Detection range per sensor. The vendor manual recommends 0.20-0.40 m when the "
          "robot stalls from ultrasonic cross-talk.", is_array=True, unit="m"),
    Param("/devices/ultrasonic/min_range_ultrasound", "double", "Ultrasonic min range",
          "obstacle", "Readings below this are ignored, per sensor.", is_array=True, unit="m"),
    Param("/devices/fallprevention/switch_fallprevention", "bool", "Fall prevention enabled",
          "obstacle", "Downward-facing cliff sensors, one switch each.", is_array=True),
    Param("/devices/fallprevention/max_range_fallprevention", "double",
          "Fall prevention max range", "obstacle", "", is_array=True, unit="m"),
    Param("/devices/fallprevention/min_range_fallprevention", "double",
          "Fall prevention min range", "obstacle", "", is_array=True, unit="m"),
    Param("/devices/camera/switch_camera", "bool", "Stereo camera avoidance", "obstacle",
          "Depth-camera obstacle avoidance, if the camera is fitted."),
    Param("/navigation/task/check_to_start", "bool", "Pre-start safety check", "obstacle",
          "Run the start-of-task check before a task begins."),
    # --- charging ----------------------------------------------------------
    Param("/navigation/charger/goal_offset_dis", "double", "Dock standoff distance", "charge",
          "Where the robot stops in front of the dock before backing in.", unit="m"),
    Param("/navigation/charger/rotate_yaw_tolerance", "double", "Dock heading tolerance",
          "charge", "How well aligned it must be before backing in.", unit="rad"),
    Param("/navigation/charger/backward_dis", "double", "Dock reverse distance", "charge",
          "How far it reverses into the dock.", unit="m",
          warn="This is a BLIND reverse. Read the real value off the robot and shorten it "
               "before the first dock attempt — the vendor default is 3.5 m."),
    Param("/navigation/charger/backward_velocity", "double", "Dock reverse speed", "charge",
          "Speed of the blind reverse.", unit="m/s"),
    Param("/navigation/charger/going_forward_offset_dis", "double", "Pull-forward distance",
          "charge", "How far it drives out of the dock once charged.", unit="m"),
    # --- battery -----------------------------------------------------------
    Param("/devices/battery/auto_charge_percentage", "int", "Auto-charge trigger", "battery",
          "Battery percentage that sends the robot to the dock.", unit="%"),
    Param("/devices/battery/restart_task_fullbattery", "bool", "Resume task when full",
          "battery", "Go back to the interrupted task once charging finishes."),
    Param("/devices/battery/empty_voltage", "double", "Empty voltage", "battery", "", unit="V"),
    Param("/devices/battery/full_voltage", "double", "Full voltage", "battery", "", unit="V"),
)

PARAMS_BY_NAMESPACE = {param.namespace: param for param in PARAMS}

# Auto-charge state strings (API doc p.191). The vendor's spellings, kept
# verbatim so a lookup miss cannot be mistaken for a match.
CHARGE_STATES = {
    "WAITTING": "waiting to be enabled",
    "WATHCING": "monitoring battery level",
    "RUNNING": "driving to the dock",
    "CHARIGING": "charging",
    "FINISHCHARIGING": "charging finished",
    "FAILED": "auto-charge task failed",
    "UNDEFINE": "unknown",
}

# Bumper bit string, right to left (API doc p.319). The last two are wired but
# unused on this chassis.
BUMPER_BITS = ("estop", "front", "rear", "left", "right")

# chassis_state device fields: 0 is healthy, anything else is a fault. Infrared
# and Ultrasonic are bit masks, one bit per sensor.
CHASSIS_DEVICES = ("Laser", "Infrared", "Ultrasonic", "Right_Motor", "Left_Motor",
                   "Engine_Motor", "Steering_Motor", "E_stop")
CHASSIS_BITMASKS = ("Infrared", "Ultrasonic")


@dataclass
class ParamValue:
    """A live parameter reading joined with its registry entry."""

    namespace: str
    type: str
    value: Any
    is_array: bool = False
    lower: str | None = None
    upper: str | None = None
    known: Param | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        known = self.known
        return {
            "namespace": self.namespace,
            "type": self.type,
            "value": self.value,
            "isArray": self.is_array,
            "lower": self.lower,
            "upper": self.upper,
            "label": known.label if known else self.namespace,
            "group": known.group if known else "other",
            "help": known.help if known else "",
            "unit": known.unit if known else "",
            "warn": known.warn if known else "",
        }


# Addresses used only to ask the OS which local interface would route to them.
# A UDP connect() sends nothing, so this works with no network and no internet.
_ROUTE_PROBES = ("192.168.1.1", "10.255.255.255", "172.16.0.1", "8.8.8.8")

# Tailscale's own resolver, inside the 100.64/10 range it assigns from. Probing
# it reveals this machine's tailnet address. Kept out of _ROUTE_PROBES because a
# tailnet /24 is not somewhere to go looking for a robot.
_OVERLAY_PROBE = "100.100.100.100"


def local_subnets() -> list[str]:
    """The /24 prefixes this machine is actually on, most likely first.

    The robot's own access point hands out 192.168.1.x, but the base is often
    bridged onto whatever LAN is in the room instead. Scanning a hard-coded
    subnet finds nothing there, so ask the OS what it is connected to.
    """
    found: list[str] = []
    for probe in _ROUTE_PROBES:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((probe, 9))
                address = sock.getsockname()[0]
            except OSError:
                continue
        prefix = address.rsplit(".", 1)[0]
        if address.startswith("127.") or prefix in found:
            continue
        found.append(prefix)
    # The robot's own AP wins when we are on it, and is a sane last resort.
    if DEFAULT_SUBNET in found:
        found.remove(DEFAULT_SUBNET)
        found.insert(0, DEFAULT_SUBNET)
    return found or [DEFAULT_SUBNET]


def local_addresses() -> list[tuple[str, str]]:
    """This machine's own IPv4 addresses, labelled by what they are good for.

    A robot host usually has several: a wired leg to the robot, a hotspot leg to
    the internet, and often a private overlay. Which one to hand to a colleague
    is the whole question, so name them rather than printing a bare 0.0.0.0.
    """
    seen: dict[str, str] = {}
    for probe in (*_ROUTE_PROBES, _OVERLAY_PROBE):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((probe, 9))
                address = sock.getsockname()[0]
            except OSError:
                continue
        if address.startswith("127.") or address in seen:
            continue
        first, second = (int(part) for part in address.split(".")[:2])
        if first == 100 and 64 <= second <= 127:
            # 100.64/10 is carrier-grade NAT space, which Tailscale uses for
            # tailnet addresses. Reachable from anywhere on the tailnet.
            seen[address] = "private overlay (Tailscale) — reachable remotely"
        else:
            seen[address] = "local network"
    return sorted(seen.items())


def explain(exc: DRobotError) -> str:
    """Append the vendor error-code meaning to a client error, when we know it."""
    text = str(exc)
    for code, meaning in ERROR_CODES.items():
        if f"errorCode={code}" in text:
            return f"{text} ({meaning})"
    return text


# No mobile robot in this class runs a pack above ~70 V. Firmware observed on
# hardware reports decivolts (236 -> 23.6 V on a 24 V pack), while the reference
# shows plain volts. Normalise on magnitude rather than trusting either.
IMPLAUSIBLE_PACK_VOLTAGE = 70.0


def normalize_voltage(raw: Any) -> float | None:
    """Battery voltage in volts, or None if not reported."""
    try:
        volts = float(raw)
    except (TypeError, ValueError):
        return None
    if volts <= 0:
        return None
    return volts / 10.0 if volts > IMPLAUSIBLE_PACK_VOLTAGE else volts


def charging_state(battery: dict[str, Any], chassis: dict[str, Any]) -> bool | None:
    """True / False / None-for-unknown.

    Documented answer is "battery current goes positive while charging". On
    hardware the current sensor reads a constant 0.0 and chassis_state comes
    back empty, so that test can never fire. Returning None rather than False
    keeps the panel from stating "not charging" about a robot that is, in fact,
    sitting on its dock charging.
    """
    if "charging" in chassis:
        return bool(chassis["charging"])
    try:
        current = float(battery.get("battery_current"))
    except (TypeError, ValueError):
        return None
    return current > 0 if current != 0 else None


def parse_bumper(raw: Any) -> dict[str, bool]:
    """Decode the bumper bit string. Rightmost character is the e-stop switch."""
    bits = str(raw or "")[::-1]
    return {name: (bits[index] == "1" if index < len(bits) else False)
            for index, name in enumerate(BUMPER_BITS)}


def parse_chassis(data: dict[str, Any]) -> dict[str, Any]:
    """Split chassis_state into per-device health. 0 is healthy; masks are per sensor."""
    devices: dict[str, Any] = {}
    for name in CHASSIS_DEVICES:
        if name not in data:
            continue
        try:
            code = int(data[name])
        except (TypeError, ValueError):
            continue
        entry: dict[str, Any] = {"code": code, "healthy": code == 0}
        if name in CHASSIS_BITMASKS and code:
            # Bit set == that sensor is faulty, counted from the right.
            entry["faulty_indices"] = [i for i in range(16) if code >> i & 1]
        devices[name] = entry
    return {
        "devices": devices,
        "charging": bool(int(data.get("ChargingState") or 0)),
        # ChargedState is the vendor's "charge complete" flag. Its own comment
        # reads "if 0, charging is complete", which contradicts every other
        # 0-is-inactive field on this struct, so both readings are passed up
        # and the UI labels it as unverified rather than guessing.
        "charged_raw": data.get("ChargedState"),
    }


class DashboardClient(drobot.DRobotClient):
    """Everything ``DRobotClient`` does, plus the endpoints a dashboard needs."""

    # --- raw sensors -------------------------------------------------------
    def ultrasonic(self) -> list[dict[str, Any]]:
        return self._call("/sensor_data/raw_ultrasonic").get("data") or []

    def irsensor(self) -> list[dict[str, Any]]:
        return self._call("/sensor_data/raw_irsensor").get("data") or []

    def bumper(self) -> dict[str, bool]:
        return parse_bumper(self._call("/sensor_data/bumper").get("data"))

    def chassis_state(self) -> dict[str, Any]:
        return parse_chassis(self._call("/sensor_data/chassis_state").get("data") or {})

    def velocity(self) -> dict[str, float]:
        data = (self._call("/sensor_data/raw_velocity").get("data") or {})
        device = data.get("device_data") or {}
        return {"linear": float(device.get("linear_vel") or 0.0),
                "angular": float(device.get("angular_rotate") or 0.0)}

    def pose_grid(self) -> dict[str, Any]:
        """Robot pose in both world metres and map pixels, with the map header."""
        data = self._call("/sensor_data/robot_pose_grid").get("data") or {}
        world = ((data.get("worldPose") or {}).get("position")) or {}
        grid = data.get("nav_status_data") or {}
        return {
            "world": {"x": world.get("x"), "y": world.get("y"), "theta": world.get("theta")},
            "grid": grid.get("gridPosition") or {},
            "angle": grid.get("angle"),
            "mapinfo": grid.get("mapinfo") or {},
        }

    def raw_scan(self) -> dict[str, Any]:
        """Deprecated since 2022-05-28 but still the only pull-based laser read."""
        return self._call("/sensor_data/raw_scan").get("data") or {}

    # --- maps --------------------------------------------------------------
    def map_png(self, name: str) -> bytes:
        """The map background image. Bypasses _call: the body is PNG, not JSON."""
        query = urllib.parse.urlencode({"map_name": name})
        url = f"{self.base}/map_manage/maps_pngs?{query}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.URLError, socket.timeout) as exc:
            raise DRobotError(f"{url} unreachable: {exc}") from exc

    # --- navigation --------------------------------------------------------
    def realtime_path(self) -> dict[str, Any]:
        """The plan the navigator is currently following, in map pixels."""
        return self._call("/navigate/get_realtime_path").get("data") or {}

    def virtual_obstacles(self) -> dict[str, Any]:
        """Virtual walls on the running map.

        Do NOT poll this while a task is running: the vendor warns that map and
        virtual-wall calls steal compute from the navigator (API doc p.18).
        """
        return (self._call("/navigate/get_virtual_obstacles").get("data") or {}).get(
            "obstacles", {})

    def set_auto_mode(self, mode: int) -> None:
        """0 = autonomous driving, 1 = remote driving (API doc p.226)."""
        self._get("/navigate/set_auto_mode", mode=int(mode))

    def task_status(self) -> dict[str, Any]:
        return self._call("/task_manager/get_task_status").get("data") or {}

    def pause_task(self) -> None:
        self._call("/task_manager/pause_task_queue")

    def resume_task(self) -> None:
        self._call("/task_manager/resume_task_queue")

    def charging_pile(self) -> Any:
        """The charging pile (dock unit) id the controller is paired with.

        API doc pp.395-397 (获取充电桩 / 修改充电桩). The success response body is
        an image in the PDF, so its shape is unknown and it is passed through
        raw rather than guessed at. Read-only here: writing the id is a pairing
        action, and writing a shape nobody has seen is not something to do to a
        robot from a dashboard.
        """
        return self._call("/param/get_devices_param").get("data")

    # --- parameters --------------------------------------------------------
    def param_table(self) -> dict[str, ParamValue]:
        """The full parameter table, keyed by namespace, with limits preserved."""
        table: dict[str, ParamValue] = {}
        for entry in self._call("/param/get_param").get("data") or []:
            namespace = entry.get("namespace")
            if not namespace:
                continue
            limit = entry.get("limit") or {}
            table[namespace] = ParamValue(
                namespace=namespace,
                type=entry.get("type", "string"),
                value=entry.get("value"),
                is_array=bool(entry.get("isArray")),
                lower=limit.get("lower_bound"),
                upper=limit.get("upper_bound"),
                known=PARAMS_BY_NAMESPACE.get(namespace),
            )
        return table

    def write_params(self, updates: list[dict[str, Any]]) -> None:
        """Write parameters. Values go over the wire as strings, arrays as lists.

        Only the entries being changed need to be sent (API doc p.361).
        """
        # Observed on the US controller: omitting isArray for sensor arrays
        # returns error 210 even though scalar updates succeed. Mirror the
        # parameter-table metadata instead of silently treating arrays as scalars.
        encoded = [{**entry, "isArray": isinstance(entry.get("value"), list)}
                   for entry in updates]
        self._call("/param/update_param", {"params": encoded})
