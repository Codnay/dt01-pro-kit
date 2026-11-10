#!/usr/bin/env python3
"""Minimal client for the DSLAM/DRobot HTTP API on the DT-01 mobile base.

Stdlib only, so it runs on a stock macOS python3 with no pip install.

  python3 tools/drobot.py discover
  python3 tools/drobot.py --host 192.168.1.15 preflight
  python3 tools/drobot.py --host 192.168.1.15 teleop
  python3 tools/drobot.py --host 192.168.1.15 point-save DOCK --type charge
  python3 tools/drobot.py --host 192.168.1.15 goto DOCK
  python3 tools/drobot.py --host 192.168.1.15 charge-status
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_PORT = 1819
DEFAULT_SUBNET = "192.168.1"
HTTP_TIMEOUT_S = 3.0

TELEOP_RATE_HZ = 10.0
DEADMAN_S = 0.8          # zero the base if no key seen for this long
DEFAULT_LINEAR = 0.15    # m/s
DEFAULT_ANGULAR = 0.4    # rad/s
MAX_LINEAR = 0.6
MAX_ANGULAR = 1.0

NAV_POLL_S = 1.0
NAV_TIMEOUT_S = 300.0

LOC_POLL_S = 1.0
LOC_TIMEOUT_S = 60.0
LOC_READY = "normal"     # other states: uninitialized, initializing, out_of_map

# Point types, per the API doc (add_cur_pose, p.100). 3 is deprecated.
POSE_TYPES = {"init": 0, "charge": 1, "nav": 2, "inspect": 4}
POSE_TYPE_NAMES = {value: name for name, value in POSE_TYPES.items()}

AUTOCHARGE_DISABLED_HINT = (
    "  note: the API reference marks the autocharge endpoints 未启用 (not enabled).\n"
    "  If they stay unavailable, configure auto charge in the Robot APP and use\n"
    "  charge-params to set the trigger level."
)


class DRobotError(RuntimeError):
    """Any failure talking to the robot controller."""


@dataclass
class Velocity:
    linear: float = 0.0   # m/s, forward positive
    angular: float = 0.0  # rad/s, right turn positive

    def payload(self) -> dict[str, Any]:
        return {"speed": {"linearSpeed": round(self.linear, 3),
                          "angularSpeed": round(self.angular, 3)}}

    def is_zero(self) -> bool:
        return self.linear == 0.0 and self.angular == 0.0


@dataclass(frozen=True)
class ParamSpec:
    """One tunable controller parameter (API doc, 参数解释 pp.364-365)."""
    namespace: str
    type: str
    help: str


# Short aliases for the docking and battery knobs, so the CLI never has to
# spell a ROS namespace. Values go over the wire as strings.
CHARGE_PARAMS: dict[str, ParamSpec] = {
    "backward_dis": ParamSpec("/navigation/charger/backward_dis", "double",
                              "reverse distance into the dock (m)"),
    "backward_speed": ParamSpec("/navigation/charger/backward_velocity", "double",
                                "reverse speed into the dock (m/s)"),
    "goal_offset": ParamSpec("/navigation/charger/goal_offset_dis", "double",
                             "standoff distance before the final back-in (m)"),
    "yaw_tolerance": ParamSpec("/navigation/charger/rotate_yaw_tolerance", "double",
                               "heading tolerance on approach (rad)"),
    "forward_offset": ParamSpec("/navigation/charger/going_forward_offset_dis", "double",
                                "pull-forward distance once charged (m)"),
    "nav_speed": ParamSpec("/navigation/nav/speed_level", "int",
                           "autonomous navigation speed level"),
    "auto_charge_at": ParamSpec("/devices/battery/auto_charge_percentage", "int",
                                "battery % that triggers auto charge"),
    "resume_when_full": ParamSpec("/devices/battery/restart_task_fullbattery", "bool",
                                  "resume the interrupted task once charged"),
}

BOOL_WORDS = {"true": "true", "on": "true", "1": "true",
              "false": "false", "off": "false", "0": "false"}

PARAM_ENCODERS: dict[str, Callable[[str], str]] = {
    "int": lambda raw: str(int(raw)),
    "double": lambda raw: f"{float(raw):.6f}",
    "bool": lambda raw: BOOL_WORDS[raw.lower()],
}


class DRobotClient:
    """Thin wrapper over the /robot HTTP API."""

    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 timeout: float = HTTP_TIMEOUT_S) -> None:
        self.base = f"http://{host}:{port}/robot"
        self.timeout = timeout

    def _call(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base}{path}"
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if body else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, socket.timeout) as exc:
            raise DRobotError(f"{url} unreachable: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DRobotError(f"{url} returned non-JSON: {raw[:120]!r}") from exc

        if parsed.get("successed") is False:
            raise DRobotError(f"{path} rejected: errorCode={parsed.get('errorCode')} "
                              f"msg={parsed.get('msg')!r}")
        return parsed

    def _get(self, path: str, **query: Any) -> dict[str, Any]:
        """GET `path` with URL-encoded query parameters."""
        if query:
            path = f"{path}?{urllib.parse.urlencode(query)}"
        return self._call(path)

    # --- reads ---
    def state(self) -> dict[str, Any]:
        return self._call("/cmd/get_robot_state").get("data", {})

    def brake_engaged(self) -> bool:
        data = self._call("/cmd/get_move_brake").get("data", {})
        return bool(data.get("brake_state", True))

    def battery(self) -> dict[str, Any]:
        return self._call("/sensor_data/battery").get("data", {})

    # --- writes ---
    def set_state(self, name: str) -> None:
        self._get("/cmd/change_robot_state", state=name)

    def set_brake(self, engaged: bool) -> None:
        self._call("/cmd/move_brake", {"brake": engaged})

    def move(self, vel: Velocity) -> None:
        self._call("/cmd/move", vel.payload())

    def stop(self) -> None:
        self.move(Velocity())

    # --- mapping (building a new map) ---
    def start_mapping(self) -> None:
        """Idle -> Mapping. The robot records a new map while you drive it."""
        self._call("/mapping/start_scan_map")

    def stop_mapping(self, name: str, rotation: float | None = None) -> None:
        """Mapping -> Idle, saving the scan under `name`."""
        path = f"/mapping/stop_scan_map?{urllib.parse.urlencode({'map_name': name})}"
        if rotation is not None:
            path += f"&?rotation_angle={rotation}"   # vendor's literal '&?' (doc p.45)
        self._call(path)

    def cancel_mapping(self) -> None:
        self._call("/mapping/cancel_scan_map")

    # --- maps ---
    def maps(self) -> list[dict[str, Any]]:
        data = self._get("/map_manage/get_map_info", map_name="").get("data", {})
        return data.get("maps", [])

    def current_map(self) -> dict[str, Any]:
        return self._call("/map_manage/current_map_info").get("data", {})

    def load_map(self, name: str, auto_init: bool = True) -> None:
        """Switch the running map. The controller auto-initializes localization on
        switch unless that is explicitly cancelled (API doc p.63)."""
        path = f"/map_manage/load_map?{urllib.parse.urlencode({'map_name': name})}"
        if not auto_init:
            path += "?delete_init_pose"     # vendor's second '?' is intentional
        self._call(path)

    # --- points ---
    def poses(self, name: str | None = None) -> list[dict[str, Any]]:
        """Recorded points on the current map, or just `name` if given."""
        resp = self._get("/pose_manage/get_pose", pose_name=name or "")
        return resp.get("PoseList") or resp.get("data") or []

    def add_current_pose(self, name: str, pose_type: int) -> None:
        self._get("/pose_manage/add_cur_pose", pose_name=name, pose_type=pose_type)

    # --- navigation ---
    def move_to(self, name: str) -> None:
        """Start an anonymous navigate-to-point task. Runs until it finishes
        or stop_task() is called."""
        self._get("/navigate/move_to", pose_name=name)

    def nav_status(self) -> dict[str, Any]:
        return self._call("/navigate/get_navigator_status").get("data", {})

    def stop_task(self) -> None:
        self._call("/task_manager/stop_task_queue")

    # --- localization ---
    def localization(self) -> dict[str, Any]:
        """Localization status and score. Marked deprecated in the API doc; the
        same fields also stream over ws://<host>:<port>/robot/notification."""
        return self._call("/notice/get_localization_notice").get("data", {})

    def initialize_at_point(self, name: str, turning: bool = False) -> None:
        self._get("/pose_manage/initialize_customized",
                  point_name=name, turning=str(turning).lower())

    def initialize_at_pose(self, x: float, y: float, angle: float,
                           turning: bool = False) -> None:
        """Initialize at raw map coordinates (metres from the map origin, radians)."""
        query = urllib.parse.urlencode({"turning": str(turning).lower()})
        self._call(f"/pose_manage/initialize_customized?{query}",
                   {"point": {"angle": angle, "position": {"x": x, "y": y}}})

    # --- charging ---
    def autocharge(self) -> dict[str, Any]:
        return self._call("/task_manager/get_autocharge_status").get("data", {})

    def set_autocharge(self, enabled: bool, point: str) -> None:
        self._call("/task_manager/set_autocharge_status",
                   {"AutoChargeSwitch": enabled, "PositionName": point})

    # --- parameters ---
    def params(self) -> dict[str, str]:
        entries = self._call("/param/get_param").get("data", [])
        return {entry["namespace"]: entry.get("value")
                for entry in entries if "namespace" in entry}

    def update_params(self, params: list[dict[str, str]]) -> None:
        self._call("/param/update_param", {"params": params})


def port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def discover(subnet: str = DEFAULT_SUBNET, port: int = DEFAULT_PORT) -> list[str]:
    """Return every host on the /24 answering on the API port."""
    hosts = [f"{subnet}.{octet}" for octet in range(1, 255)]
    with ThreadPoolExecutor(max_workers=64) as pool:
        hits = pool.map(lambda h: port_open(h, port), hosts)
        return [host for host, is_open in zip(hosts, hits) if is_open]


# The controller walks this chain one step at a time (API doc, state diagram p.19):
# Idle <-> Localization <-> RunningTask. Mapping/Sleep are off-chain and not handled here.
STATE_CHAIN = ["Idle", "Localization", "RunningTask"]
STATE_SETTLE_S = 10.0


def walk_to_state(client: DRobotClient, target: str,
                  report: Callable[[str], None] = print) -> None:
    """Step the robot along the state chain until it reaches `target`."""
    if target not in STATE_CHAIN:
        raise DRobotError(f"{target!r} not one of {STATE_CHAIN}")
    for _ in range(len(STATE_CHAIN)):
        current = client.state().get("state_operation")
        if current == target:
            report(f"  state is {current}")
            return
        if current not in STATE_CHAIN:
            raise DRobotError(f"robot is in {current!r}, which is off the "
                              f"Idle/Localization/RunningTask chain — fix it in the app")
        step = STATE_CHAIN.index(current) + (1 if STATE_CHAIN.index(target) >
                                             STATE_CHAIN.index(current) else -1)
        nxt = STATE_CHAIN[step]
        report(f"  {current} -> {nxt}")
        client.set_state(nxt)
        _await_state(client, nxt)
    raise DRobotError(f"gave up before reaching {target}")


def _await_state(client: DRobotClient, expected: str) -> None:
    deadline = time.monotonic() + STATE_SETTLE_S
    while time.monotonic() < deadline:
        if client.state().get("state_operation") == expected:
            return
        time.sleep(0.5)
    raise DRobotError(f"timed out waiting for state {expected}")


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


TELEOP_HELP = """
  w / s : forward / reverse      a / d : turn left / right
  space : stop                   q     : quit (sends stop)
  + / - : linear speed           [ / ] : angular speed
Hold a key to keep moving; the base zeroes if no key arrives for %.1fs.
""" % DEADMAN_S


def _read_key(timeout: float) -> str | None:
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def _apply_key(key: str, vel: Velocity, lin: float, ang: float) -> tuple[Velocity, float, float]:
    """Map one keypress to a new (velocity, linear step, angular step)."""
    moves: dict[str, Velocity] = {
        "w": Velocity(lin, 0.0),
        "s": Velocity(-lin, 0.0),
        "a": Velocity(0.0, -ang),
        "d": Velocity(0.0, ang),
        " ": Velocity(),
    }
    if key in moves:
        return moves[key], lin, ang
    if key == "+":
        return vel, clamp(lin + 0.05, MAX_LINEAR), ang
    if key == "-":
        return vel, max(0.05, lin - 0.05), ang
    if key == "]":
        return vel, lin, clamp(ang + 0.1, MAX_ANGULAR)
    if key == "[":
        return vel, lin, max(0.1, ang - 0.1)
    return vel, lin, ang


def teleop(client: DRobotClient, linear: float, angular: float) -> None:
    # termios/tty are POSIX-only and used nowhere else. Importing them here
    # rather than at module scope keeps the rest of this module — the API client
    # the dashboard builds on — importable on Windows.
    import termios
    import tty

    if not sys.stdin.isatty():
        raise DRobotError("teleop needs an interactive terminal (stdin is not a tty)")
    print(TELEOP_HELP)
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    vel, last_key_at = Velocity(), 0.0
    period = 1.0 / TELEOP_RATE_HZ
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key(period)
            if key in ("q", "\x03"):
                break
            if key:
                vel, linear, angular = _apply_key(key, vel, linear, angular)
                last_key_at = time.monotonic()
            if not vel.is_zero() and time.monotonic() - last_key_at > DEADMAN_S:
                vel = Velocity()
            client.move(vel)
            sys.stdout.write(f"\r  lin={vel.linear:+.2f} m/s  ang={vel.angular:+.2f} rad/s "
                             f" [step {linear:.2f}/{angular:.2f}]   ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print("\nstopping...")
        for _ in range(3):   # stop is worth repeating on a lossy link
            try:
                client.stop()
            except DRobotError as exc:
                print(f"  stop failed: {exc}")


def describe_battery(battery: dict[str, Any]) -> str:
    """One-line battery summary. Current is positive while charging (API doc p.318)."""
    current = float(battery.get("battery_current") or 0.0)
    return (f"{battery.get('battery_capacity_percentage')}% "
            f"current={current:+.2f}A "
            f"({'CHARGING' if current > 0 else 'not charging'})")


def describe_localization(loc: dict[str, Any]) -> str:
    return (f"status={loc.get('status')} "
            f"score={loc.get('localization_score')} "
            f"map_avg={loc.get('avg_localization_score_on_map')}")


def watch_localization(client: DRobotClient, timeout: float = LOC_TIMEOUT_S) -> bool:
    """Poll until the localizer converges. Falls back to the state flag if the
    deprecated notice endpoint is gone."""
    deadline = time.monotonic() + timeout
    warned = False
    while time.monotonic() < deadline:
        try:
            loc = client.localization()
        except DRobotError as exc:
            if not warned:
                print(f"  localization notice unavailable ({exc}); "
                      f"falling back to state_initialized")
                warned = True
            if client.state().get("state_initialized"):
                return True
            time.sleep(LOC_POLL_S)
            continue
        sys.stdout.write(f"\r  {describe_localization(loc)}   ")
        sys.stdout.flush()
        if loc.get("status") == LOC_READY:
            print("\n  localized")
            return True
        time.sleep(LOC_POLL_S)
    print(f"\n  still not localized after {timeout:.0f}s")
    return False


def describe_pose(pose: dict[str, Any]) -> str:
    position = (pose.get("world") or {}).get("position") or {}
    kind = POSE_TYPE_NAMES.get(pose.get("type"), f"type {pose.get('type')}")
    return (f"  {pose.get('name') or '?':<20} {kind:<8} "
            f"x={float(position.get('x') or 0.0):+.2f} "
            f"y={float(position.get('y') or 0.0):+.2f} "
            f"angle={float(pose.get('angle') or 0.0):+.3f}rad")


def navigation_finished(status: dict[str, Any]) -> bool:
    """The controller reports `finished` as the string "true"/"false"."""
    return str((status.get("nav_status_data") or {}).get("finished")).lower() == "true"


def watch_navigation(client: DRobotClient, timeout: float = NAV_TIMEOUT_S) -> bool:
    """Poll until navigation reports finished. True if the goal was reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.nav_status()
        progress = status.get("process") or {}
        sys.stdout.write(f"\r  {status.get('nav_status_type', '?'):<11} "
                         f"remaining={progress.get('remaining_milleage', '?')}m  "
                         f"battery {describe_battery(client.battery())}   ")
        sys.stdout.flush()
        if navigation_finished(status):
            print("\n  goal reached")
            return True
        time.sleep(NAV_POLL_S)
    print(f"\n  gave up waiting after {timeout:.0f}s")
    return False


def abort_navigation(client: DRobotClient) -> None:
    """Stop the running task. Repeated — one packet on a flaky link is not a stop."""
    for _ in range(3):
        try:
            client.stop_task()
        except DRobotError as exc:
            print(f"  stop_task_queue failed: {exc}")


def encode_param(assignment: str) -> dict[str, str]:
    """Turn 'backward_speed=0.1' into the {type, namespace, value} wire form."""
    alias, _, raw = assignment.partition("=")
    spec = CHARGE_PARAMS.get(alias.strip())
    if spec is None:
        raise DRobotError(f"unknown parameter {alias.strip()!r}; "
                          f"expected one of: {', '.join(CHARGE_PARAMS)}")
    if not raw.strip():
        raise DRobotError(f"missing value in {assignment!r}; expected KEY=VALUE")
    try:
        value = PARAM_ENCODERS[spec.type](raw.strip())
    except (ValueError, KeyError) as exc:
        raise DRobotError(f"{alias.strip()} expects a {spec.type}, got {raw.strip()!r}") from exc
    return {"type": spec.type, "namespace": spec.namespace, "value": value}


def cmd_discover(args: argparse.Namespace) -> int:
    print(f"scanning {args.subnet}.1-254 on port {args.port} ...")
    found = discover(args.subnet, args.port)
    if not found:
        print("nothing answering — are you on the drobot_5G network?")
        return 1
    for host in found:
        print(f"  found {host}:{args.port}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    state = client.state()
    print(f"  state       : {state.get('state_operation')} "
          f"(initialized={state.get('state_initialized')})")
    battery = client.battery()
    print(f"  battery     : {battery.get('battery_capacity_percentage')}% "
          f"current={battery.get('battery_current')}")
    engaged = client.brake_engaged()
    print(f"  soft e-stop : {'ENGAGED — release before driving' if engaged else 'released'}")
    return 1 if engaged else 0


def cmd_state(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if args.target is None:
        print(json.dumps(client.state(), indent=2))
        return 0
    walk_to_state(client, args.target)
    return 0


def cmd_brake(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    client.set_brake(args.engaged == "on")
    print(f"  soft e-stop set to {args.engaged}")
    return 0


def cmd_move(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    client.move(Velocity(args.linear, args.angular))
    print(f"  sent lin={args.linear} ang={args.angular} "
          f"(no deadman — send 'stop' to halt)")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    DRobotClient(args.host, args.port).stop()
    print("  stop sent")
    return 0


def cmd_teleop(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if client.brake_engaged():
        print("soft e-stop is engaged — run: drobot.py brake off")
        return 1
    teleop(client, args.linear, args.angular)
    return 0


MAPPING_STATE = "Mapping"

MAPPING_ADVICE = """  now drive the whole area with:  drobot.py teleop
  - go slowly and keep the laser's view unobstructed
  - close loops: return along paths you have already driven
  - when done:   drobot.py map-save <name>
  - to discard:  drobot.py map-cancel"""


def cmd_map_start(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    current = client.state().get("state_operation")
    if current == MAPPING_STATE:
        print("  already mapping")
        print(MAPPING_ADVICE)
        return 0
    walk_to_state(client, "Idle")
    client.start_mapping()
    print("  mapping started")
    print(MAPPING_ADVICE)
    return 0


def cmd_map_save(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    current = client.state().get("state_operation")
    if current != MAPPING_STATE:
        print(f"robot is in {current!r}, not {MAPPING_STATE} — "
              f"start with: drobot.py map-start", file=sys.stderr)
        return 1
    if args.name in {entry.get("mapname") for entry in client.maps()}:
        print(f"a map named {args.name!r} already exists — pick another name",
              file=sys.stderr)
        return 1
    client.stop_mapping(args.name, args.rotation)
    print(f"  saved map {args.name!r}; robot returns to Idle")
    print(f"  next: drobot.py map-use {args.name} && drobot.py loc-status")
    return 0


def cmd_map_cancel(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    client.cancel_mapping()
    print("  mapping cancelled — nothing saved")
    return 0


def cmd_maps(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    try:
        current = client.current_map().get("name", "")
    except DRobotError as exc:
        current = ""
        print(f"  running map unavailable ({exc})")
    maps = client.maps()
    if not maps:
        print("  no maps on the controller")
        return 1
    for entry in maps:
        mark = "*" if entry.get("mapname") == current else " "
        print(f"  {mark} {entry.get('mapname') or '?':<18} "
              f"{entry.get('gridWidth')}x{entry.get('gridHeight')} "
              f"res={entry.get('resolution')} "
              f"origin=({float(entry.get('originX') or 0.0):+.2f}, "
              f"{float(entry.get('originY') or 0.0):+.2f})")
    return 0


def cmd_map_use(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if args.name not in {entry.get("mapname") for entry in client.maps()}:
        print(f"no map named {args.name!r} (see: drobot.py maps)", file=sys.stderr)
        return 1
    client.load_map(args.name, auto_init=not args.no_auto_init)
    print(f"  loaded {args.name!r}")
    if not args.no_auto_init:
        print("  the controller auto-initializes localization on switch — "
              "confirm with: drobot.py loc-status")
    return 0


def cmd_loc_status(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    print(f"  state_initialized : {client.state().get('state_initialized')}")
    try:
        print(f"  localization      : {describe_localization(client.localization())}")
    except DRobotError as exc:
        print(f"  localization      : unavailable ({exc})")
        return 1
    return 0


def cmd_localize(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if args.point is None and (args.x is None or args.y is None):
        print("give either --point NAME or --x X --y Y [--angle A]", file=sys.stderr)
        return 2
    walk_to_state(client, "RunningTask")
    if args.turning:
        print("  --turning: the robot will ROTATE IN PLACE to converge — clear the area")
    try:
        if args.point:
            client.initialize_at_point(args.point, args.turning)
            print(f"  initializing at point {args.point!r}")
        else:
            client.initialize_at_pose(args.x, args.y, args.angle, args.turning)
            print(f"  initializing at x={args.x} y={args.y} angle={args.angle}")
        settled = watch_localization(client, args.timeout)
    except KeyboardInterrupt:
        # --turning spins the base; the brake is the documented immediate stop.
        print("\n  interrupted — engaging the software brake")
        try:
            client.set_brake(True)
            print("  brake ENGAGED; release it with: drobot.py brake off")
        except DRobotError as exc:
            print(f"  brake failed: {exc} — use the physical e-stop")
        raise
    if not settled:
        print("  the pose guess was probably too far off; reposition and retry, "
              "or add --turning")
        return 1
    return 0


def cmd_points(args: argparse.Namespace) -> int:
    poses = DRobotClient(args.host, args.port).poses(args.name)
    if not poses:
        print("  no points recorded on the current map")
        return 1
    for pose in poses:
        print(describe_pose(pose))
    return 0


def cmd_point_save(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if not client.state().get("state_initialized"):
        print("robot is not localized — run: drobot.py localize --point NAME "
              "(or --x X --y Y)", file=sys.stderr)
        return 1
    walk_to_state(client, "RunningTask")
    client.add_current_pose(args.name, POSE_TYPES[args.type])
    print(f"  saved {args.name!r} as a {args.type} point at the current pose")
    return 0


def cmd_goto(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if client.brake_engaged():
        print("soft e-stop is engaged — run: drobot.py brake off", file=sys.stderr)
        return 1
    if not sys.stdin.isatty():
        print("  warning: stdin is not a tty — Ctrl-C will not abort this run")
    walk_to_state(client, "RunningTask")
    print(f"  navigating to {args.name!r} (Ctrl-C aborts)")
    client.move_to(args.name)
    reached = False
    try:
        reached = watch_navigation(client, args.timeout)
    finally:
        # The navigation module keeps driving until the task is stopped, so every
        # exit path — arrival, timeout, Ctrl-C, exception — stops it.
        abort_navigation(client)
        try:
            print(f"  battery {describe_battery(client.battery())}")
        except DRobotError as exc:
            print(f"  battery read failed: {exc}")
    return 0 if reached else 1


def cmd_nav_stop(args: argparse.Namespace) -> int:
    abort_navigation(DRobotClient(args.host, args.port))
    print("  stop_task_queue sent")
    return 0


def cmd_charge_status(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    print(f"  battery     : {describe_battery(client.battery())}")
    try:
        status = client.autocharge()
    except DRobotError as exc:
        print(f"  auto charge : unavailable ({exc})")
        print(AUTOCHARGE_DISABLED_HINT)
        return 1
    print(f"  auto charge : switch={status.get('AutoChargeSwitch')} "
          f"state={status.get('ChargeState')} point={status.get('PointName')!r}")
    return 0


def cmd_charge_auto(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    enabled = args.switch == "on"
    if enabled and not args.point:
        print("--point is required when enabling auto charge", file=sys.stderr)
        return 2
    if args.point and not _check_charge_point(client, args.point):
        return 1
    try:
        client.set_autocharge(enabled, args.point or "")
    except DRobotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(AUTOCHARGE_DISABLED_HINT)
        return 1
    print(f"  auto charge {'enabled at ' + args.point if enabled else 'disabled'}")
    return 0


def _check_charge_point(client: DRobotClient, name: str) -> bool:
    """False if `name` is not a point on this map; warn if it is not a charge point."""
    poses = {pose.get("name"): pose for pose in client.poses()}
    if name not in poses:
        print(f"no point named {name!r} on this map (see: drobot.py points)",
              file=sys.stderr)
        return False
    if poses[name].get("type") != POSE_TYPES["charge"]:
        print(f"  warning: {name!r} is not a charge point — the dock pose must be "
              f"saved with --type charge")
    return True


def cmd_charge_params(args: argparse.Namespace) -> int:
    client = DRobotClient(args.host, args.port)
    if args.set:
        updates = [encode_param(item) for item in args.set]
        client.update_params(updates)
        print(f"  updated {len(updates)} parameter(s)")
    values = client.params()
    for alias, spec in CHARGE_PARAMS.items():
        print(f"  {alias:<17} {values.get(spec.namespace, '<absent>'):<12} {spec.help}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="robot controller IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    subs = parser.add_subparsers(dest="command", required=True)

    scan = subs.add_parser("discover", help="scan the /24 for the controller")
    scan.add_argument("--subnet", default=DEFAULT_SUBNET)
    scan.set_defaults(func=cmd_discover)

    subs.add_parser("preflight", help="state, battery, e-stop").set_defaults(func=cmd_preflight)
    subs.add_parser("stop", help="send zero velocity").set_defaults(func=cmd_stop)

    state = subs.add_parser("state", help="show state, or walk to Idle/Localization/RunningTask")
    state.add_argument("target", nargs="?", choices=STATE_CHAIN)
    state.set_defaults(func=cmd_state)

    brake = subs.add_parser("brake", help="engage/release the software e-stop")
    brake.add_argument("engaged", choices=["on", "off"])
    brake.set_defaults(func=cmd_brake)

    move = subs.add_parser("move", help="send one velocity command")
    move.add_argument("--linear", type=float, default=0.0, help="m/s, forward positive")
    move.add_argument("--angular", type=float, default=0.0, help="rad/s, right positive")
    move.set_defaults(func=cmd_move)

    drive = subs.add_parser("teleop", help="keyboard teleop at 10 Hz with a deadman")
    drive.add_argument("--linear", type=float, default=DEFAULT_LINEAR)
    drive.add_argument("--angular", type=float, default=DEFAULT_ANGULAR)
    drive.set_defaults(func=cmd_teleop)

    subs.add_parser("map-start", help="start building a new map (Idle -> Mapping)") \
        .set_defaults(func=cmd_map_start)

    save_map = subs.add_parser("map-save", help="save the map being built and return to Idle")
    save_map.add_argument("name")
    save_map.add_argument("--rotation", type=float,
                          help="rotation_angle to store with the map")
    save_map.set_defaults(func=cmd_map_save)

    subs.add_parser("map-cancel", help="discard the map being built") \
        .set_defaults(func=cmd_map_cancel)

    subs.add_parser("maps", help="list maps, marking the running one") \
        .set_defaults(func=cmd_maps)

    use = subs.add_parser("map-use", help="switch the running map")
    use.add_argument("name")
    use.add_argument("--no-auto-init", action="store_true",
                     help="cancel the automatic localization init on switch")
    use.set_defaults(func=cmd_map_use)

    subs.add_parser("loc-status", help="localization status and score") \
        .set_defaults(func=cmd_loc_status)

    loc = subs.add_parser("localize", help="initialize localization at a point or pose")
    loc.add_argument("--point", help="name of a recorded point the robot is sitting on")
    loc.add_argument("--x", type=float, help="map x in metres (with --y)")
    loc.add_argument("--y", type=float, help="map y in metres (with --x)")
    loc.add_argument("--angle", type=float, default=0.0, help="heading in radians")
    loc.add_argument("--turning", action="store_true",
                     help="let the robot rotate in place to converge (it MOVES)")
    loc.add_argument("--timeout", type=float, default=LOC_TIMEOUT_S)
    loc.set_defaults(func=cmd_localize)

    points = subs.add_parser("points", help="list recorded points on the current map")
    points.add_argument("name", nargs="?", help="show one point instead of all")
    points.set_defaults(func=cmd_points)

    save = subs.add_parser("point-save", help="record the current pose as a named point")
    save.add_argument("name")
    save.add_argument("--type", choices=sorted(POSE_TYPES), default="nav",
                      help="point type (default: nav; the dock needs charge)")
    save.set_defaults(func=cmd_point_save)

    goto = subs.add_parser("goto", help="navigate to a recorded point and watch progress")
    goto.add_argument("name")
    goto.add_argument("--timeout", type=float, default=NAV_TIMEOUT_S,
                      help="seconds to wait before aborting (default: %(default)s)")
    goto.set_defaults(func=cmd_goto)

    subs.add_parser("nav-stop", help="stop the running navigation task") \
        .set_defaults(func=cmd_nav_stop)

    subs.add_parser("charge-status", help="battery and auto-charge state") \
        .set_defaults(func=cmd_charge_status)

    auto = subs.add_parser("charge-auto", help="turn auto charging on or off")
    auto.add_argument("switch", choices=["on", "off"])
    auto.add_argument("--point", help="name of the charge point to dock at")
    auto.set_defaults(func=cmd_charge_auto)

    params = subs.add_parser("charge-params",
                             help="show or set docking speed/distance parameters")
    params.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="e.g. --set backward_speed=0.08 --set backward_dis=1.0")
    params.set_defaults(func=cmd_charge_params)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    if handler is not cmd_discover and not args.host:
        print("--host is required (find it with: drobot.py discover)", file=sys.stderr)
        return 2
    try:
        return handler(args)
    except DRobotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
