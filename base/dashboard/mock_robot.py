#!/usr/bin/env python3
"""A stand-in for the DT-01 controller, shaped like the vendor's API.

    python3 dashboard/mock_robot.py --port 1819

There is no robot on the desk, so every claim about this dashboard has to be
checked against something. This reproduces the documented response shapes —
including the vendor's ``successed`` spelling, HTTP 200 on failure, the state
chain, and the auto-charge endpoints being disabled — plus a small simulation
so the map, the sensors and the battery actually move.

**A mock pass is not a hardware pass.** Nothing here proves the real controller
behaves this way; it proves the dashboard handles what the documentation says.

Failure injection, so the unhappy paths are testable too:

    curl -X POST localhost:1819/mock/config -d '{"fail":["/cmd/move"]}'
    curl -X POST localhost:1819/mock/config -d '{"obstacle_x":1.0}'
    curl -X POST localhost:1819/mock/config -d '{"autocharge_enabled_endpoint":true}'
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

TICK_S = 0.05
# The controller stops the base if /cmd/move is not re-sent at ~3 Hz
# (API doc p.257). Modelled here so the dashboard's deadman is observable.
MOVE_WATCHDOG_S = 0.5

GRID_W, GRID_H = 400, 300
RESOLUTION = 0.05
ORIGIN_X, ORIGIN_Y = -10.0, -7.5

ULTRASOUND_COUNT = 10
IRSENSOR_COUNT = 4


class Sim:
    """A differential base on a map with one wall in front of it."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state_operation = "Idle"
        self.state_initialized = True
        self.brake = False
        self.linear = self.angular = 0.0
        self.last_move_at = 0.0
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.battery = 78.0
        self.charging = False
        self.nav_goal: str | None = None
        self.nav_type = "idle"
        self.nav_finished = True
        self.total_mileage = 0.0
        self.auto_mode = 1
        self.autocharge_on = False
        self.autocharge_point = ""
        self.bumper_front = self.bumper_rear = False
        # Mock controls
        self.fail_paths: set[str] = set()
        # Reproduce the firmware seen on the real DT-01 (2026-08-31): the
        # documented /robot/ws/notice 404s, the bumper endpoint returns 103,
        # chassis_state is empty, battery voltage is decivolts, and current and
        # temperature are never populated. Off by default so the mock still
        # matches the reference; switch on to test against reality.
        self.legacy_firmware = False
        self.obstacle_x = 2.5
        self.autocharge_endpoint_enabled = False
        self.params = _default_params()
        self.points = _default_points()
        self.notices: list[dict[str, Any]] = []

    # --- simulation --------------------------------------------------------
    def tick(self, now: float) -> None:
        with self.lock:
            if self.brake or (now - self.last_move_at) > MOVE_WATCHDOG_S:
                self.linear = self.angular = 0.0
            if self.nav_goal:
                self._drive_to_goal()
            self.theta = (self.theta + self.angular * TICK_S + math.pi) % (2 * math.pi) - math.pi
            step = self.linear * TICK_S
            self.x += step * math.cos(self.theta)
            self.y += step * math.sin(self.theta)
            self.total_mileage += abs(step)
            self.bumper_front = self._front_gap() < 0.05
            dock = self._point("DOCK")
            docked = dock is not None and math.hypot(
                self.x - dock["world"]["position"]["x"],
                self.y - dock["world"]["position"]["y"]) < 0.4
            self.charging = docked
            self.battery = min(100.0, self.battery + 0.05) if docked \
                else max(0.0, self.battery - (0.004 if abs(self.linear) > 0 else 0.001))

    def _point(self, name: str) -> dict[str, Any] | None:
        return next((point for point in self.points if point["name"] == name), None)

    def _drive_to_goal(self) -> None:
        goal = self._point(self.nav_goal or "")
        if goal is None:
            self.nav_goal, self.nav_type, self.nav_finished = None, "idle", True
            return
        target_x = goal["world"]["position"]["x"]
        target_y = goal["world"]["position"]["y"]
        dx, dy = target_x - self.x, target_y - self.y
        distance = math.hypot(dx, dy)
        if distance < 0.12:
            self.linear = self.angular = 0.0
            self.nav_finished, self.nav_type = True, "idle"
            self._notice("Finish target point", self.nav_goal or "")
            self.nav_goal = None
            return
        speed_level = int(float(self.params["/navigation/nav/speed_level"]["value"]))
        speed = (0.15, 0.3, 0.5)[max(0, min(2, speed_level))]
        heading = math.atan2(dy, dx)
        error = (heading - self.theta + math.pi) % (2 * math.pi) - math.pi
        self.angular = max(-0.8, min(0.8, error * 1.5))
        self.linear = 0.0 if abs(error) > 0.6 else speed
        self.remaining = distance

    def _notice(self, message: str, target: str) -> None:
        self.notices.insert(0, {
            "message": message, "target_name": target, "notice_level": "INFO",
            "nav_type": "NAV_POSE", "task_name": "NAVIGATION",
            "target_pose": {"position": {"x": self.x, "y": self.y, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0,
                                            "z": math.sin(self.theta / 2),
                                            "w": math.cos(self.theta / 2)}},
        })
        del self.notices[8:]

    def _front_gap(self) -> float:
        """Distance to the wall ahead, or a large number when facing away."""
        if abs(self.theta) > math.pi / 2:
            return 9.9
        gap = (self.obstacle_x - self.x) / max(0.15, math.cos(self.theta))
        return max(0.0, gap)

    # --- sensor views ------------------------------------------------------
    def ultrasonic(self) -> list[dict[str, Any]]:
        gap = self._front_gap()
        readings = []
        for index in range(ULTRASOUND_COUNT):
            angle = round(-0.6 + index * 0.13, 3)
            spread = gap + abs(angle) * 0.8
            # 0 means "nothing in range", exactly as the controller reports it.
            distance = round(spread, 3) if spread < 1.1 else 0.0
            readings.append({"angle": angle, "range": distance, "originX": 0.33,
                             "originY": round(0.1 - index * 0.02, 3),
                             "frameid": f"ultrasound{index + 1}"})
        return readings

    def irsensor(self) -> list[dict[str, Any]]:
        return [{"angle": round(-0.52 + index * 0.35, 3), "range": 0.0, "originX": 0.33,
                 "originY": round(0.1 - index * 0.07, 3), "frameid": f"irsensor{index + 1}"}
                for index in range(IRSENSOR_COUNT)]

    def bumper_bits(self) -> str:
        # Right to left: e-stop, front, rear, left, right.
        bits = ["0", "1" if self.bumper_front else "0", "1" if self.bumper_rear else "0"]
        return "".join(reversed(bits))

    def grid(self) -> dict[str, int]:
        return {"x": int((self.x - ORIGIN_X) / RESOLUTION),
                "y": int((self.y - ORIGIN_Y) / RESOLUTION)}

    def mapinfo(self) -> dict[str, Any]:
        return {"mapname": "mock_floor", "gridWidth": GRID_W, "gridHeight": GRID_H,
                "originX": ORIGIN_X, "originY": ORIGIN_Y, "resolution": RESOLUTION}

    def notice_push(self) -> dict[str, Any]:
        with self.lock:
            return {
                "errorCode": "", "msg": "successed", "successed": True,
                "data": {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device_notice": {
                        "angular_rotate": round(self.angular, 4),
                        "emergency_stop": self.brake,
                        "battery": round(self.battery, 1),
                        "battery_voltage": round(24.0 + self.battery * 0.05, 2),
                        "bumper_front": self.bumper_front,
                        "battery_current": 3.2 if self.charging else -1.4,
                        "bumper_back": self.bumper_rear,
                        "current_speed": round(self.linear, 3),
                    },
                    "nav_notice": self.notices[:3],
                    "loc_notice": {
                        "status": "normal" if self.state_initialized else "uninitialized",
                        "initialized": self.state_initialized,
                        "localization_score": 0.94,
                        "avg_localization_score_on_map": 0.86,
                        "gridPosition": self.grid(),
                        "worldPose": {"position": {"x": round(self.x, 4),
                                                   "y": round(self.y, 4),
                                                   "theta": round(self.theta, 4)}},
                    },
                },
            }

    def flat_notice_push(self) -> dict[str, Any]:
        """The /robot/notification schema: same facts, no nesting."""
        nested = self.notice_push()["data"]
        device, localization = nested["device_notice"], nested["loc_notice"]
        return {"errorCode": "", "msg": "successed", "successed": True, "data": {
            **device, **localization, "nav_status": nested["nav_notice"],
            "date": nested["date"], "rain_mode": False}}

    def scan_push(self) -> dict[str, Any]:
        with self.lock:
            points, gridpoints = [], []
            for index in range(60):
                angle = -2.35 + index * 0.078
                distance = min(6.0, self._front_gap() + abs(angle) * 1.2)
                world_x = self.x + distance * math.cos(self.theta + angle)
                world_y = self.y + distance * math.sin(self.theta + angle)
                points.append({"x": round(distance * math.cos(angle), 3),
                               "y": round(distance * math.sin(angle), 3), "z": 0.0})
                gridpoints.append({"x": int((world_x - ORIGIN_X) / RESOLUTION),
                                   "y": int((world_y - ORIGIN_Y) / RESOLUTION)})
            return {"errorCode": "", "msg": "successed", "successed": True,
                    "data": {"scan_grid_3d": {
                        "sensor_type": "laser_scan_grid", "sensor_id": "1",
                        "sensor_data": {"angle_min": -2.35, "angle_max": 2.35,
                                        "angle_increment": 0.078, "range_min": 0.05,
                                        "range_max": 20.0, "mapinfo": self.mapinfo(),
                                        "header": {"stamp": time.time(), "frame_id": "laser",
                                                   "seq": 1},
                                        "points": points, "gridpoint": gridpoints,
                                        "intensities": [1.0] * len(points)}}}}


def _default_points() -> list[dict[str, Any]]:
    def point(name: str, x: float, y: float, kind: int, angle: float = 0.0) -> dict[str, Any]:
        return {"name": name, "angle": angle, "type": kind,
                "poseId": f"mock-{name}", "image": {"name": "", "width": 0, "height": 0},
                "gridX": int((x - ORIGIN_X) / RESOLUTION),
                "gridY": int((y - ORIGIN_Y) / RESOLUTION),
                "world": {"position": {"x": x, "y": y, "z": 0.0},
                          "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(angle / 2),
                                          "w": math.cos(angle / 2)}}}
    return [point("HOME", 0.0, 0.0, 0), point("DOCK", -2.2, 0.6, 1, math.pi),
            point("BENCH", 1.6, -1.4, 2, -0.4), point("DOOR", 0.4, 2.2, 2, 1.2),
            point("SHELF", -1.4, -2.0, 4, 2.6)]


def _default_params() -> dict[str, dict[str, Any]]:
    def entry(namespace: str, kind: str, value: Any, is_array: bool = False,
              limit: dict[str, str] | None = None) -> tuple[str, dict[str, Any]]:
        record = {"namespace": namespace, "type": kind, "value": value,
                  "isArray": is_array}
        if limit:
            record["limit"] = limit
        return namespace, record

    level = {"lower_bound": "0", "upper_bound": "2"}
    return dict([
        entry("/navigation/nav/speed_level", "int", "1", limit=level),
        entry("/navigation/follow/speed_level", "int", "1", limit=level),
        entry("/navigation/nav/avoid_obstacle", "bool", "false"),
        entry("/navigation/follow/avoid_obstacle", "bool", "true"),
        entry("/navigation/follow/front_safe_check_dis", "double", "1.800000"),
        entry("/navigation/nav/find_next_goal_duration", "double", "2.000000"),
        entry("/navigation/nav/goal_unsafe_quit_duration", "double", "8.000000"),
        entry("/navigation/task/check_to_start", "bool", "true"),
        entry("/devices/ultrasonic/switch_ultrasound", "bool",
              ["true"] * ULTRASOUND_COUNT, True),
        entry("/devices/ultrasonic/max_range_ultrasound", "double",
              ["1.100000"] + ["0.600000"] * (ULTRASOUND_COUNT - 1), True),
        entry("/devices/ultrasonic/min_range_ultrasound", "double",
              ["0.100000"] + ["0.050000"] * (ULTRASOUND_COUNT - 1), True),
        entry("/devices/fallprevention/switch_fallprevention", "bool",
              ["false"] * IRSENSOR_COUNT, True),
        entry("/devices/fallprevention/max_range_fallprevention", "double",
              ["0.300000"] * IRSENSOR_COUNT, True),
        entry("/devices/fallprevention/min_range_fallprevention", "double",
              ["0.500000"] * IRSENSOR_COUNT, True),
        entry("/devices/camera/switch_camera", "bool", "false"),
        # 3.5 m is the vendor's own documented default for the blind reverse.
        entry("/navigation/charger/backward_dis", "double", "3.500000"),
        entry("/navigation/charger/backward_velocity", "double", "0.100000"),
        entry("/navigation/charger/goal_offset_dis", "double", "0.400000"),
        entry("/navigation/charger/going_forward_offset_dis", "double", "0.500000"),
        entry("/navigation/charger/rotate_yaw_tolerance", "double", "0.050000"),
        entry("/devices/battery/auto_charge_percentage", "int", "20"),
        entry("/devices/battery/restart_task_fullbattery", "bool", "true"),
        entry("/devices/battery/empty_voltage", "double", "22.000000"),
        entry("/devices/battery/full_voltage", "double", "39.300000"),
    ])


def make_png(sim: Sim) -> bytes:
    """A map image: free space, a border, and the wall the sensors see.

    Written by hand because a PNG is a handful of zlib-deflated scanlines and
    Pillow is not installable on the robot's own access point.
    """
    obstacle_col = int((sim.obstacle_x - ORIGIN_X) / RESOLUTION)
    rows = bytearray()
    for row in range(GRID_H):
        rows.append(0)                                # filter type: none
        grid_y = GRID_H - 1 - row                     # PNG rows run top-down
        for col in range(GRID_W):
            edge = col < 2 or col >= GRID_W - 2 or grid_y < 2 or grid_y >= GRID_H - 2
            wall = abs(col - obstacle_col) <= 1 and 60 <= grid_y <= 240
            rows.extend((30, 40, 60) if (edge or wall) else (236, 240, 244))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", GRID_W, GRID_H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))


def ok(data: Any = "") -> dict[str, Any]:
    return {"errorCode": "", "msg": "successed", "data": data, "successed": True}


def fail(code: int, message: str = "") -> dict[str, Any]:
    return {"errorCode": code, "msg": message, "data": "", "successed": False}


# --- WebSocket server side -------------------------------------------------

def ws_frame(text: str) -> bytes:
    """Server-to-client text frame. Server frames are never masked."""
    payload = text.encode()
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 1 << 16:
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockDRobot/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    @property
    def sim(self) -> Sim:
        return self.server.sim                            # type: ignore[attr-defined]

    def do_GET(self) -> None:                             # noqa: N802
        self._handle()

    def do_POST(self) -> None:                            # noqa: N802
        self._handle()

    def _handle(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/robot/ws/notice" and self.sim.legacy_firmware:
            self._json(fail(100, "not found"))            # this firmware has no such path
            return
        if path in ("/robot/ws/notice", "/robot/notification",
                    "/robot/notification/sensor_scan"):
            self._websocket(path)
            return
        query = {key: value[0] for key, value in parse_qs(parts.query).items()}
        body: dict[str, Any] = {}
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                body = {}
        if path.startswith("/mock/"):
            self._json(self._mock_control(body))
            return
        endpoint = path[len("/robot"):] if path.startswith("/robot") else path
        if endpoint in self.sim.fail_paths:
            self._json(fail(211, "injected failure"))
            return
        if endpoint == "/map_manage/maps_pngs":
            image = make_png(self.sim)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return
        self._json(self._route(endpoint, query, body))

    def _json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)                # the vendor returns 200 even on failure
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mock_control(self, body: dict[str, Any]) -> dict[str, Any]:
        sim = self.sim
        with sim.lock:
            if "fail" in body:
                sim.fail_paths = set(body["fail"] or [])
            for field in ("obstacle_x", "battery"):
                if field in body:
                    setattr(sim, field, float(body[field]))
            if "autocharge_enabled_endpoint" in body:
                sim.autocharge_endpoint_enabled = bool(body["autocharge_enabled_endpoint"])
            if "bumper_rear" in body:
                sim.bumper_rear = bool(body["bumper_rear"])
            if "legacy_firmware" in body:
                sim.legacy_firmware = bool(body["legacy_firmware"])
            return ok({"fail": sorted(sim.fail_paths), "obstacle_x": sim.obstacle_x,
                       "battery": round(sim.battery, 1),
                       "autocharge_enabled_endpoint": sim.autocharge_endpoint_enabled})

    # --- routing -----------------------------------------------------------
    def _route(self, path: str, query: dict[str, str],           # noqa: C901
               body: dict[str, Any]) -> dict[str, Any]:
        sim = self.sim
        with sim.lock:
            if path == "/cmd/get_robot_state":
                return ok({"state_initialized": sim.state_initialized,
                           "state_operation": sim.state_operation})
            if path == "/cmd/change_robot_state":
                chain = ["Idle", "Localization", "RunningTask"]
                target = query.get("state", "")
                if target not in chain or sim.state_operation not in chain:
                    return fail(100, "state does not match")
                if abs(chain.index(target) - chain.index(sim.state_operation)) > 1:
                    return fail(100, "state does not match")   # single-hop only
                sim.state_operation = target
                return ok()
            if path == "/cmd/get_move_brake":
                return ok({"brake_state": sim.brake})
            if path == "/cmd/move_brake":
                sim.brake = bool(body.get("brake"))
                if sim.brake:
                    sim.linear = sim.angular = 0.0
                    sim.nav_goal = None
                return ok()
            if path == "/cmd/move":
                speed = body.get("speed") or {}
                # Accepted but ignored while braked, per the API note that
                # control is ineffective once the software e-stop is on.
                if not sim.brake:
                    sim.linear = float(speed.get("linearSpeed") or 0.0)
                    sim.angular = float(speed.get("angularSpeed") or 0.0)
                sim.last_move_at = time.monotonic()
                return ok()
            if path == "/sensor_data/battery":
                if sim.legacy_firmware:
                    return ok({"battery_capacity_percentage": round(sim.battery, 1),
                               "battery_current": 0.0, "battery_temperature": 0.0,
                               "battery_voltage": round((24.0 + sim.battery * 0.05) * 10, 1),
                               "battery_power": 0, "battery_24voltage": 0,
                               "battery_24current": 0, "battery_12voltage": 0,
                               "battery_12current": 0})
                return ok({"battery_capacity_percentage": round(sim.battery, 1),
                           "battery_current": 3.2 if sim.charging else -1.4,
                           "battery_voltage": round(24.0 + sim.battery * 0.05, 2),
                           "battery_temperature": 28, "battery_power": 0,
                           "battery_12voltage": 0, "battery_24voltage": 0,
                           "battery_12current": 0, "battery_24current": 0})
            if path == "/sensor_data/raw_ultrasonic":
                return ok(sim.ultrasonic())
            if path == "/sensor_data/raw_irsensor":
                return ok(sim.irsensor())
            if path == "/sensor_data/bumper":
                if sim.legacy_firmware:
                    return fail(103, "not recived msg")     # vendor's spelling
                return ok(sim.bumper_bits())
            if path == "/sensor_data/chassis_state":
                if sim.legacy_firmware:
                    return ok("")                            # empty, as observed
                return ok({"Laser": 0, "Infrared": 0, "Ultrasonic": 0, "Right_Motor": 0,
                           "Left_Motor": 0, "Engine_Motor": 0, "Steering_Motor": 0,
                           "E_stop": 1 if sim.brake else 0,
                           "ChargingState": 1 if sim.charging else 0,
                           "ChargedState": 1 if sim.battery >= 99.5 else 0})
            if path == "/sensor_data/raw_velocity":
                return ok({"device_type": "motor_vel", "device_id": 1,
                           "device_data": {"linear_vel": round(sim.linear, 3),
                                           "linear_y": 0,
                                           "angular_rotate": round(sim.angular, 3)}})
            if path == "/sensor_data/robot_pose_grid":
                return ok({"nav_status_type": "robot_pose", "robot_id": "1",
                           "worldPose": {"position": {"x": round(sim.x, 4),
                                                      "y": round(sim.y, 4),
                                                      "theta": round(sim.theta, 4)}},
                           "nav_status_data": {"angle": round(sim.theta, 4),
                                               "gridPosition": sim.grid(),
                                               "mapinfo": sim.mapinfo()}})
            if path == "/map_manage/current_map_info":
                return ok({"name": "mock_floor", "mapId": "mock", **sim.mapinfo()})
            if path == "/map_manage/get_map_info":
                return ok({"maps": [{"mapname": "mock_floor", "mapId": "mock",
                                     "gridWidth": GRID_W, "gridHeight": GRID_H,
                                     "originX": ORIGIN_X, "originY": ORIGIN_Y,
                                     "resolution": RESOLUTION, "pngMD5": "mock",
                                     "maxZoom": 0, "minBoxSize": GRID_W}]})
            if path == "/pose_manage/get_pose":
                wanted = query.get("pose_name") or ""
                poses = [point for point in sim.points
                         if not wanted or point["name"] == wanted]
                return {"errorCode": 0, "msg": "successed", "successed": True,
                        "PoseList": poses,
                        "mapInfo": {"gridWidth": GRID_W, "gridHeight": GRID_H}}
            if path == "/pose_manage/add_cur_pose":
                name = query.get("pose_name") or ""
                if not name:
                    return fail(200, "input parameter empty")
                if any(point["name"] == name for point in sim.points):
                    return fail(1201, "point already exists")
                kind = int(query.get("pose_type") or 2)
                sim.points.append({
                    "name": name, "angle": round(sim.theta, 4), "type": kind,
                    "poseId": f"mock-{name}", "image": {"name": "", "width": 0, "height": 0},
                    **sim.grid_point_fields()})
                return ok()
            if path == "/pose_manage/initialize_customized":
                sim.state_initialized = True
                return ok()
            if path == "/navigate/move_to":
                name = query.get("pose_name") or ""
                if sim.state_operation != "RunningTask":
                    return fail(100, "state does not match")
                if not any(point["name"] == name for point in sim.points):
                    return fail(1200, "requested point does not exist")
                sim.nav_goal, sim.nav_type, sim.nav_finished = name, "navigation", False
                sim._notice("Running to target point", name)
                return ok()
            if path == "/navigate/get_navigator_status":
                goal = sim._point(sim.nav_goal or "")
                remaining = 0.0
                if goal:
                    remaining = math.hypot(goal["world"]["position"]["x"] - sim.x,
                                           goal["world"]["position"]["y"] - sim.y)
                return ok({"nav_status_type": sim.nav_type,
                           "nav_status_data": {"finished": str(sim.nav_finished).lower(),
                                               "information": "" if sim.nav_goal
                                               else "NO_PATH"},
                           "process": {"path_name": sim.nav_goal or "DEFAULT", "index": "0",
                                       "total_milleage": f"{sim.total_mileage:.4f}",
                                       "total_time": "0", "remaining_time": f"{remaining*4:.2f}",
                                       "remaining_milleage": f"{remaining:.4f}",
                                       "passed_points": "[]",
                                       "next_point": sim.nav_goal or ""}})
            if path == "/navigate/get_realtime_path":
                goal = sim._point(sim.nav_goal or "")
                if not goal:
                    return ok({"mapInfo": sim.mapinfo(), "grid_points": []})
                start, end = sim.grid(), {"x": goal["gridX"], "y": goal["gridY"]}
                steps = 24
                return ok({"mapInfo": sim.mapinfo(), "grid_points": [
                    {"x": int(start["x"] + (end["x"] - start["x"]) * i / steps),
                     "y": int(start["y"] + (end["y"] - start["y"]) * i / steps)}
                    for i in range(steps + 1)]})
            if path == "/navigate/set_auto_mode":
                mode = query.get("mode")
                if mode not in ("0", "1"):
                    return fail(204, "param is invalid")
                sim.auto_mode = int(mode)
                return ok()
            if path == "/task_manager/stop_task_queue":
                sim.nav_goal, sim.nav_type, sim.nav_finished = None, "idle", True
                sim.linear = sim.angular = 0.0
                return ok()
            if path == "/task_manager/get_autocharge_status":
                if not sim.autocharge_endpoint_enabled:
                    return fail(210, "cannot call service")     # 未启用, as documented
                state = "CHARIGING" if sim.charging else (
                    "WATHCING" if sim.autocharge_on else "WAITTING")
                return ok({"ChargeState": state, "AutoChargeSwitch": sim.autocharge_on,
                           "PointName": sim.autocharge_point})
            if path == "/task_manager/set_autocharge_status":
                if not sim.autocharge_endpoint_enabled:
                    return fail(210, "cannot call service")
                sim.autocharge_on = bool(body.get("AutoChargeSwitch"))
                sim.autocharge_point = str(body.get("PositionName") or "")
                return ok()
            if path == "/param/get_devices_param":
                # Documented only by its error case (API doc p.395); the success
                # body is an image in the PDF. The mock returns the failure the
                # doc actually shows rather than inventing a success shape.
                return fail(210, "FAILED CALL SERVICE")
            if path == "/param/get_param":
                return ok(list(sim.params.values()))
            if path == "/param/update_param":
                # Real firmware rejects array writes that omit this metadata.
                for update in body.get("params") or []:
                    if isinstance(update.get("value"), list) and update.get("isArray") is not True:
                        return fail(210, "FAILED CALL SERVICE")
                for update in body.get("params") or []:
                    namespace = update.get("namespace")
                    if namespace in sim.params:
                        sim.params[namespace]["value"] = update.get("value")
                return ok()
            if path == "/notice/get_localization_notice":
                return ok({"status": "normal" if sim.state_initialized else "uninitialized",
                           "localization_score": 0.94,
                           "avg_localization_score_on_map": 0.86})
        return fail(100, f"no mock route for {path}")

    # --- websocket ---------------------------------------------------------
    def _websocket(self, path: str) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._json(fail(100, "not a websocket request"))
            return
        import base64
        import hashlib
        digest = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", digest)
        self.end_headers()
        self.close_connection = True
        if path == "/robot/notification":
            push, period = self.sim.flat_notice_push, 0.5
        elif path.endswith("notice"):
            push, period = self.sim.notice_push, 0.2
        else:
            push, period = self.sim.scan_push, 1.0
        try:
            while not self.server.stopping.is_set():      # type: ignore[attr-defined]
                self.wfile.write(ws_frame(json.dumps(push())))
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def _grid_point_fields(self: Sim) -> dict[str, Any]:
    grid = self.grid()
    return {"gridX": grid["x"], "gridY": grid["y"],
            "world": {"position": {"x": round(self.x, 4), "y": round(self.y, 4), "z": 0.0},
                      "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(self.theta / 2),
                                      "w": math.cos(self.theta / 2)}}}


Sim.grid_point_fields = _grid_point_fields                # type: ignore[attr-defined]


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], verbose: bool = False) -> None:
        super().__init__(address, Handler)
        self.sim = Sim()
        self.verbose = verbose
        self.stopping = threading.Event()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client hanging up mid-push is normal, not a fault worth a traceback."""
        if self.verbose:
            super().handle_error(request, client_address)


def serve(port: int = 1819, bind: str = "127.0.0.1", verbose: bool = False) -> MockServer:
    """Start the mock in a background thread and return the server."""
    server = MockServer((bind, port), verbose=verbose)

    def tick() -> None:
        while not server.stopping.is_set():
            server.sim.tick(time.monotonic())
            time.sleep(TICK_S)

    threading.Thread(target=server.serve_forever, daemon=True, name="mock-http").start()
    threading.Thread(target=tick, daemon=True, name="mock-sim").start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=1819)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    server = serve(args.port, args.bind, args.verbose)
    print(f"  mock DRobot controller on http://{args.bind}:{server.server_address[1]}/robot/")
    print("  Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stopping.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
