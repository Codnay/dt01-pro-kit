"""Always-on manual proximity/cliff interlock; no motion or network I/O here.

Controller configuration is not proof of working sensors. Unknown, disabled,
invalid and stale readings warn without preventing manual driving. Detected
hazards still stop motion. Laser coverage is not inferred from an uncalibrated scan.
"""

from __future__ import annotations

import math
from typing import Any

ULTRASOUND = "/devices/ultrasonic/switch_ultrasound"
CLIFF = "/devices/fallprevention/switch_fallprevention"
NAV = "/navigation/nav/avoid_obstacle"
FOLLOW = "/navigation/follow/avoid_obstacle"
OPTIONAL = ("/devices/camera/switch_camera", "/navigation/task/check_to_start")
SWITCHES = (ULTRASOUND, CLIFF, NAV, FOLLOW, *OPTIONAL)
SENSOR_STALE_S = 2.0  # four nominal 0.5 s polls; exposed in the UI/settings
PARAM_STALE_S = 10.0  # configuration is refreshed every 4 s


def is_on(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def enable_updates(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Only write known, present switches; never invent sensor array lengths."""
    updates = []
    for name in SWITCHES:
        entry = table.get(name)
        if entry is None:
            if name not in OPTIONAL:
                raise ValueError(f"required protection setting unavailable: {name}")
            continue
        if entry.type != "bool":
            raise ValueError(f"unexpected protection setting type: {name}")
        if name in (ULTRASOUND, CLIFF):
            if not entry.is_array or not isinstance(entry.value, list) or not entry.value:
                raise ValueError(f"missing sensor configuration: {name}")
            value: Any = ["true"] * len(entry.value)
        else:
            if entry.is_array:
                raise ValueError(f"unexpected array setting: {name}")
            value = "true"
        updates.append({"namespace": name, "type": "bool", "value": value})
    return updates


def verify_updates(table: dict[str, Any], updates: list[dict[str, Any]]) -> None:
    for update in updates:
        actual = table.get(update["namespace"])
        expected = update["value"]
        value = actual.value if actual else None
        if isinstance(expected, list):
            verified = isinstance(value, list) and len(value) == len(expected) and all(map(is_on, value))
        else:
            verified = is_on(value)
        if not verified:
            raise ValueError(f"protection readback failed: {update['namespace']}")


def obstacles(state: dict, params: dict, ages: dict, errors: dict,
              distance: float, stale_s: float, params_age: float | None) -> dict:
    reasons, warnings, sensors = [], [], []
    if params_age is None or params_age > PARAM_STALE_S or errors.get("params"):
        warnings.append("protection settings are unavailable or stale")
    for namespace in (NAV, FOLLOW, *(name for name in OPTIONAL if name in params)):
        if not is_on(params.get(namespace)):
            warnings.append(f"{namespace.rsplit('/', 2)[1]} obstacle avoidance is not enabled")
    groups = (
        ("ultrasonic", "ultrasound", ULTRASOUND,
         "/devices/ultrasonic/min_range_ultrasound", "/devices/ultrasonic/max_range_ultrasound"),
        ("irsensor", "irsensor", CLIFF,
         "/devices/fallprevention/min_range_fallprevention", "/devices/fallprevention/max_range_fallprevention"),
    )
    for key, kind, switches_name, min_name, max_name in groups:
        switches, mins, maxs = params.get(switches_name), params.get(min_name), params.get(max_name)
        readings = state.get(key) or []
        if not isinstance(readings, list):
            readings = []
        configured = isinstance(switches, list) and bool(switches)
        if not configured:
            warnings.append(f"{kind} configuration unavailable")
            switches = []
        age = ages.get(key)
        fresh = age is not None and age <= stale_s and not errors.get(key)
        if not fresh:
            warnings.append(f"{kind} readings unavailable or stale")
        for index in range(max(len(readings), len(switches))):
            reading = readings[index] if index < len(readings) and isinstance(readings[index], dict) else {}
            known = index < len(switches)
            enabled = is_on(switches[index]) if known else None
            low = number(mins[index]) if isinstance(mins, list) and index < len(mins) else None
            high = number(maxs[index]) if isinstance(maxs, list) and index < len(maxs) else None
            value = number(reading.get("range"))
            limits_ok = low is not None and high is not None and 0 <= low < high
            if kind == "ultrasound" and limits_ok and high < distance:
                limits_ok = False
            valid = fresh and limits_ok and value is not None and value > 0
            triggered = False
            status = "configuration unknown"
            if known:
                if not enabled:
                    status = "disabled"
                elif not valid:
                    status = "reading unavailable"
                elif kind == "irsensor":
                    triggered = not low <= value <= high
                    status = "floor outside safe range" if triggered else "floor detected"
                elif value <= low:
                    status = "below sensor minimum; unverified"
                else:
                    triggered = value <= distance
                    status = "obstacle" if triggered else "clear"
                if not triggered and status not in ("clear", "floor detected"):
                    warnings.append(f"{kind}{index + 1}: {status}")
            if triggered:
                reasons.append(f"{kind}{index + 1}: {status} ({value:.2f} m)")
            sensors.append({"kind": kind, "index": index,
                            "id": reading.get("frameid") or f"{kind}{index + 1}",
                            "range": value, "enabled": enabled, "triggered": triggered,
                            "status": status, "age_s": age,
                            "angle": reading.get("angle"), "originX": reading.get("originX"),
                            "originY": reading.get("originY")})
    for side in ("front", "rear"):
        if (state.get("bumper") or {}).get(side):
            reasons.append(f"{side} bumper pressed")
    devices = (state.get("chassis") or {}).get("devices") or {}
    faults = [name for name, entry in devices.items() if not entry.get("healthy") and name != "E_stop"]
    # Missing obstacle sensing is advisory; motor/other chassis faults remain
    # motion interlocks. A sensor fault must not suppress another sensor's alarm.
    sensor_faults = {"Laser", "Infrared", "Ultrasonic"}
    warnings.extend(f"chassis sensor fault: {name}" for name in faults if name in sensor_faults)
    blocked = reasons + [f"chassis fault: {name}" for name in faults if name not in sensor_faults]
    return {"alarm": bool(reasons), "reasons": reasons, "sensors": sensors,
            "alarm_distance": distance, "faults": faults,
            "protection": {"enabled": True, "ready": not blocked and not warnings,
                           "blocked_by": blocked[0] if blocked else None,
                           "warning": warnings[0] if warnings else None,
                           "unavailable_policy": "warn",
                           "issues": blocked + warnings, "sensor_stale_s": stale_s,
                           "params_stale_s": PARAM_STALE_S,
                           "laser": "manual stopping not verified",
                           "optional": {name: is_on(params[name]) if name in params else None
                                        for name in OPTIONAL}}}
