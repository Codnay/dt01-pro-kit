"""Live robot state for the dashboard: push feeds, pollers, and one snapshot.

Everything the UI renders comes from a single ``RobotLink.snapshot()`` dict, so
the browser never talks to the robot directly and there is exactly one place
where "what is the robot doing right now" is decided.

Two sources fill it:

* **push** — ``ws://host:port/robot/ws/notice`` carries state, battery, bumper
  and localisation at the controller's own rate, and
  ``/robot/notification/sensor_scan`` carries the laser.
* **poll** — everything with no push channel (ultrasonic, infrared, chassis
  health, navigation progress). Each poller has its own interval and backs off
  when it fails, so one dead endpoint cannot starve the rest.

The pollers deliberately leave the map and virtual-wall endpoints alone while a
task is running: the vendor warns those calls steal compute from the navigator
and can make navigation fail (API doc p.18).
"""

from __future__ import annotations

import contextlib
import protection
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import drobot_api
from drobot_api import DashboardClient, DRobotError, explain
from wsclient import WebSocketFeed

# How close something may come before the dashboard calls it an obstacle. This
# is the dashboard's own alarm threshold for display and for gating teleop; it
# is not the controller's avoidance distance, which lives in the parameter table.
DEFAULT_ALARM_DISTANCE_M = 0.35

# A push feed this stale is treated as dead and the REST fallbacks take over.
PUSH_STALE_S = 5.0

# Bumper / e-stop readings older than this are not evidence of anything. Some
# firmware serves neither the bumper endpoint nor the documented push path.
SAFETY_STALE_S = 10.0

LOC_READY = "normal"          # localisation status meaning "converged"

# Poll intervals, seconds. Sensor reads are the safety-relevant ones and run
# fastest; the rest are background colour.
INTERVAL_SENSORS = 0.5
INTERVAL_STATE = 1.0
INTERVAL_NAV = 1.0
INTERVAL_SLOW = 4.0

BACKOFF_MAX_S = 30.0


@dataclass
class Poller:
    """One periodic read, with its own failure backoff."""

    name: str
    interval: float
    call: Callable[[], Any]
    apply: Callable[[dict[str, Any], Any], None]
    when: Callable[[dict[str, Any]], bool] | None = None
    next_at: float = 0.0
    error: str | None = None
    _backoff: float = 0.0
    last_ok: float | None = None

    def due(self, now: float, state: dict[str, Any]) -> bool:
        if self.when is not None and not self.when(state):
            return False
        return now >= self.next_at

    def run(self, state: dict[str, Any], now: float, lock: Any = None) -> None:
        # Never hold the snapshot lock over network I/O: the drive loop must
        # remain able to notice stale sensors and stop during a blocked read.
        try:
            value = self.call()
            with lock if lock is not None else contextlib.nullcontext():
                self.apply(state, value)
                self.last_ok = time.monotonic()
                self.error = None
                self._backoff = 0.0
                self.next_at = self.last_ok + self.interval
        except Exception as exc:
            with lock if lock is not None else contextlib.nullcontext():
                self.error = explain(exc) if isinstance(exc, DRobotError) else f"{type(exc).__name__}: {exc}"
                self._backoff = min(BACKOFF_MAX_S, max(self.interval, self._backoff * 2 or 1.0))
                self.next_at = time.monotonic() + self._backoff


def _blank_state() -> dict[str, Any]:
    return {
        "robot": {"state_operation": None, "state_initialized": None},
        "brake": {"engaged": None},
        "battery": {},
        "chassis": {},
        "bumper": {},
        "ultrasonic": [],
        "irsensor": [],
        "velocity": {"linear": 0.0, "angular": 0.0},
        "pose": {},
        "localization": {},
        "nav": {},
        "scan": None,
        "notices": [],
        "map": {},
        "task": {},
    }


class RobotLink:
    """Owns the connection to one robot and the live picture of it."""

    def __init__(self, host: str, port: int = drobot_api.DEFAULT_PORT,
                 alarm_distance: float = DEFAULT_ALARM_DISTANCE_M) -> None:
        self.host = host
        self.port = port
        self.alarm_distance = alarm_distance
        self.sensor_stale_s = protection.SENSOR_STALE_S
        self.client = DashboardClient(host, port)
        self.started_at = time.time()

        self._lock = threading.RLock()
        self._state = _blank_state()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._push_seen_at: dict[str, float] = {}
        self._params: dict[str, drobot_api.ParamValue] = {}
        self._params_error: str | None = None
        self._params_seen_at: float | None = None
        # When the bumper / e-stop interlocks were last actually observed. Some
        # firmware serves neither the bumper endpoint nor the documented push
        # path, and an absent reading must not read as "all clear".
        self._safety_seen_at: float = 0.0

        self.notice_feed = WebSocketFeed(
            [f"ws://{host}:{port}/robot/ws/notice",
             f"ws://{host}:{port}/robot/notification"],
            self._on_notice, name="notice")
        self.scan_feed = WebSocketFeed(
            f"ws://{host}:{port}/robot/notification/sensor_scan", self._on_scan,
            on_open=self._configure_scan, name="scan")

        self._pollers = self._build_pollers()

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.notice_feed.start()
        self.scan_feed.start()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.notice_feed.stop()
        self.scan_feed.stop()

    # --- push handlers -----------------------------------------------------
    @staticmethod
    def _flatten_notice(data: dict[str, Any]) -> tuple[dict, dict, list]:
        """Return (device, localization, notices) from either push schema.

        Two firmwares, same facts, different shape. /robot/ws/notice nests them
        under device_notice / loc_notice / nav_notice; /robot/notification puts
        the identical fields at the top level. Detect by shape rather than by
        which URL answered, so a firmware that moves the path again still works.
        """
        if "device_notice" in data or "loc_notice" in data:
            return (data.get("device_notice") or {},
                    data.get("loc_notice") or {},
                    data.get("nav_notice") or [])
        device = {key: data[key] for key in
                  ("angular_rotate", "emergency_stop", "battery", "battery_voltage",
                   "bumper_front", "battery_current", "bumper_back", "current_speed",
                   "battery_temperature") if key in data}
        localization = {key: data[key] for key in
                        ("status", "initialized", "localization_score",
                         "avg_localization_score_on_map", "gridPosition", "worldPose")
                        if key in data}
        return device, localization, (data.get("nav_status") or [])

    def _on_notice(self, message: dict[str, Any]) -> None:
        data = message.get("data") or {}
        device, localization, notices = self._flatten_notice(data)
        with self._lock:
            state = self._state
            if device:
                battery = dict(state["battery"])
                if "battery" in device:
                    battery["battery_capacity_percentage"] = device.get("battery")
                for key in ("battery_voltage", "battery_current", "battery_temperature"):
                    if key in device:
                        battery[key] = device[key]
                state["battery"] = battery
                if {"bumper_front", "bumper_back", "emergency_stop"} & set(device):
                    state["bumper"] = {
                        **state["bumper"],
                        "front": bool(device.get("bumper_front")),
                        "rear": bool(device.get("bumper_back")),
                        "estop": bool(device.get("emergency_stop")),
                    }
                    self._safety_seen_at = time.monotonic()
                state["velocity"] = {
                    "linear": float(device.get("current_speed") or 0.0),
                    "angular": float(device.get("angular_rotate") or 0.0),
                }
                self._push_seen_at["device"] = time.monotonic()
            if localization:
                state["localization"] = {
                    "status": localization.get("status"),
                    "initialized": localization.get("initialized"),
                    "score": localization.get("localization_score"),
                    "map_average": localization.get("avg_localization_score_on_map"),
                }
                world = (localization.get("worldPose") or {}).get("position") or {}
                grid = localization.get("gridPosition") or {}
                if world or grid:
                    state["pose"] = {
                        **state.get("pose", {}),
                        "world": {"x": world.get("x"), "y": world.get("y"),
                                  "theta": world.get("theta")},
                        "grid": {"x": grid.get("x"), "y": grid.get("y")},
                    }
                self._push_seen_at["pose"] = time.monotonic()
            if notices:
                stamped = [{**entry, "at": message.get("data", {}).get("date")}
                           for entry in notices if isinstance(entry, dict)]
                state["notices"] = (stamped + state["notices"])[:25]

    def _configure_scan(self, socket_: Any) -> None:
        """Ask for the rasterised 2D scan once a second (API doc p.390)."""
        try:
            socket_.send_json({"sensor_interval": 1, "req_scan_id": ["scan_grid"]})
        except OSError:
            pass

    def _on_scan(self, message: dict[str, Any]) -> None:
        data = message.get("data") or {}
        payload = None
        for key in ("scan_grid_3d", "scan_grid", "scan_grid00", "scan_grid01"):
            entry = data.get(key)
            if isinstance(entry, dict) and entry.get("sensor_data"):
                payload = entry["sensor_data"]
                break
        if payload is None:
            return
        with self._lock:
            self._state["scan"] = {
                "points": payload.get("points") or [],
                "gridpoint": payload.get("gridpoint") or [],
                "mapinfo": payload.get("mapinfo") or {},
                "range_max": payload.get("range_max"),
                "at": time.time(),
            }
            self._push_seen_at["scan"] = time.monotonic()

    def _push_fresh(self, key: str) -> bool:
        seen = self._push_seen_at.get(key)
        return seen is not None and (time.monotonic() - seen) < PUSH_STALE_S

    # --- pollers -----------------------------------------------------------
    def _build_pollers(self) -> list[Poller]:
        def set_key(key: str) -> Callable[[dict[str, Any], Any], None]:
            def apply(state: dict[str, Any], value: Any) -> None:
                state[key] = value
            return apply

        def navigating(state: dict[str, Any]) -> bool:
            return state["robot"].get("state_operation") == "RunningTask"

        def no_pose_push(_state: dict[str, Any]) -> bool:
            return not self._push_fresh("pose")

        def no_device_push(_state: dict[str, Any]) -> bool:
            return not self._push_fresh("device")

        return [
            Poller("params", INTERVAL_SLOW, self.client.param_table,
                   lambda _state, table: self._apply_params(table)),
            Poller("state", INTERVAL_STATE, self.client.state, set_key("robot")),
            Poller("brake", INTERVAL_STATE,
                   lambda: {"engaged": self.client.brake_engaged()}, set_key("brake")),
            Poller("ultrasonic", INTERVAL_SENSORS, self.client.ultrasonic,
                   set_key("ultrasonic")),
            Poller("irsensor", INTERVAL_SENSORS, self.client.irsensor, set_key("irsensor")),
            # The push feed carries front/rear bumpers and the e-stop; this fills
            # in the rest of the bit string and covers a dead socket.
            Poller("bumper", INTERVAL_SENSORS, self.client.bumper,
                   lambda state, value: (state["bumper"].update(value),
                                         setattr(self, "_safety_seen_at",
                                                 time.monotonic()))[0]),
            Poller("velocity", INTERVAL_SENSORS, self.client.velocity, set_key("velocity"),
                   when=no_device_push),
            Poller("battery", INTERVAL_SLOW, self.client.battery,
                   lambda state, value: state["battery"].update(value)),
            Poller("chassis", INTERVAL_SLOW, self.client.chassis_state, set_key("chassis")),
            Poller("pose", INTERVAL_STATE, self.client.pose_grid,
                   lambda state, value: state.__setitem__(
                       "pose", {**state.get("pose", {}), **value}),
                   when=no_pose_push),
            Poller("map", INTERVAL_SLOW, self.client.current_map, set_key("map")),
            Poller("nav", INTERVAL_NAV, self.client.nav_status, set_key("nav"),
                   when=navigating),
        ]

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                state = self._state
                due = [poller for poller in self._pollers if poller.due(now, state)]
            for poller in due:
                if self._stop.is_set():
                    return
                poller.run(self._state, time.monotonic(), self._lock)
            self._stop.wait(0.1)

    # --- parameters (read on demand, cached) -------------------------------
    def _apply_params(self, table: dict) -> None:
        self._params = table
        self._params_error = None
        self._params_seen_at = time.monotonic()

    def params(self, refresh: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            cached, error = self._params, self._params_error
        if cached and not refresh:
            return [value.as_dict() for value in cached.values()]
        try:
            table = self.client.param_table()
            error = None
        except DRobotError as exc:
            error = explain(exc)
            table = cached
        with self._lock:
            self._params, self._params_error = table, error
            if error is None:
                self._params_seen_at = time.monotonic()
        return [value.as_dict() for value in table.values()]

    def params_error(self) -> str | None:
        with self._lock:
            return self._params_error

    def param_value(self, namespace: str) -> Any:
        with self._lock:
            entry = self._params.get(namespace)
        return entry.value if entry else None

    # --- derived safety view ----------------------------------------------
    def _obstacles(self, state: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            ages = {p.name: None if p.last_ok is None else now - p.last_ok for p in self._pollers}
            errors = {p.name: p.error for p in self._pollers if p.error}
            if self._params_error:
                errors["params"] = self._params_error
            params = {name: entry.value for name, entry in self._params.items()}
            params_age = None if self._params_seen_at is None else now - self._params_seen_at
            return protection.obstacles(state, params, ages, errors, self.alarm_distance,
                                        self.sensor_stale_s, params_age)

    # --- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = {key: value for key, value in self._state.items()}
            state["bumper"] = dict(state.get("bumper") or {})
            poll_errors = {poller.name: poller.error
                           for poller in self._pollers if poller.error}
            state["obstacles"] = self._obstacles(state)
        state["reachable"] = self.reachable()
        state["localized"] = self.localized()
        age = self.safety_data_age()
        state["safety_data_age"] = None if age is None else round(age, 1)
        state["safety_stale"] = age is None or age > SAFETY_STALE_S
        state["link"] = {
            "host": self.host,
            "port": self.port,
            "notice_feed": self.notice_feed.status(),
            "scan_feed": self.scan_feed.status(),
            "poll_errors": poll_errors,
            "params_error": self._params_error,
        }
        state["ts"] = time.time()
        return state

    def localized(self) -> bool:
        """Whether the robot actually knows where it is.

        `/cmd/get_robot_state` reports state_initialized: false on hardware whose
        push feed simultaneously reports initialized: true, status "normal" and a
        localisation score of 1.0. Gating navigation on that flag alone refuses
        to move a perfectly localised robot, so the notice is preferred and the
        flag is only a fallback — the same order drobot.py already uses.
        """
        with self._lock:
            localization = self._state.get("localization") or {}
            if localization.get("status") == LOC_READY or localization.get("initialized"):
                return True
            if localization:
                return False
            return bool(self._state["robot"].get("state_initialized"))

    def safety_data_age(self) -> float | None:
        """Seconds since the bumper / e-stop interlocks were last observed."""
        with self._lock:
            if not self._safety_seen_at:
                return None
            return time.monotonic() - self._safety_seen_at

    def reachable(self) -> bool:
        """Is the controller actually answering, as opposed to merely addressed?

        Pointing this at an IP with nothing on it must not look like success.
        The state poller is the cheapest continuous proof of life there is, so
        it is the one that decides.
        """
        with self._lock:
            if self._state["robot"].get("state_operation") is None:
                return False
            poller = next((entry for entry in self._pollers if entry.name == "state"), None)
            return poller is None or poller.error is None

    def note_brake(self, engaged: bool) -> None:
        """Record a brake change we just made, without waiting for the poller.

        The brake poll runs at 1 Hz. For a display that is fine; for the
        emergency stop it is not — the operator must see it take effect at once,
        and the motion gate must refuse on the right reason immediately.
        """
        with self._lock:
            self._state["brake"] = {"engaged": engaged}

    # --- preflight ---------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        """The checks that decide whether it is safe to command motion.

        Read from the live cache, not by issuing fresh requests, so this stays
        cheap enough to run before every command.
        """
        state = self.snapshot()
        battery = state.get("battery") or {}
        percent = battery.get("battery_capacity_percentage")
        checks = [
            {"name": "Controller reachable",
             "ok": not state["link"]["poll_errors"].get("state"),
             "detail": state["link"]["poll_errors"].get("state") or f"{self.host}:{self.port}"},
            {"name": "Software e-stop released",
             "ok": state["brake"].get("engaged") is False,
             "detail": "engaged — release it before driving"
                       if state["brake"].get("engaged") else "released"},
            {"name": "Chassis e-stop released",
             "ok": not (state.get("bumper") or {}).get("estop"),
             "detail": "hardware emergency stop is active"
                       if (state.get("bumper") or {}).get("estop")
                       else "no hardware e-stop reported"},
            {"name": "Bumpers clear",
             "ok": not ((state.get("bumper") or {}).get("front")
                        or (state.get("bumper") or {}).get("rear")),
             "detail": "a bumper is pressed" if ((state.get("bumper") or {}).get("front")
                                                 or (state.get("bumper") or {}).get("rear"))
                       else "clear"},
            {"name": "Battery",
             "ok": percent is None or float(percent) > 15,
             "detail": f"{percent}%" if percent is not None else "unknown"},
        ]
        return {"ok": all(check["ok"] for check in checks), "checks": checks}
