"""Prometheus text-format exporter for the dashboard's snapshot.

The dashboard answers "what is the robot doing *right now*". This module exists
for the other question — "what happened, and is something degrading" — by
handing the same snapshot to Prometheus so Grafana can keep the history.

Three rules shaped what is in here:

* **Only what is worth a time series.** Battery, localisation quality,
  navigation outcomes and link health drift over days and are invisible on a
  live gauge. Laser scans and 5 Hz pose do not: they are high cardinality and
  tell you nothing a week later, so they are deliberately absent.
* **A dead sensor must not look like a reading.** This firmware reports
  ``battery_current`` and ``battery_temperature`` as a hard 0.0 whether or not
  it knows them (see ``docs/jetson-deployment.md`` §8.3). Emitting 0 would put a
  confident flat line on a chart that means "no data", so anything the robot
  does not actually report is *omitted* rather than zero-filled. A gap in
  Grafana is honest; a zero is not.
* **``robot_id`` is emitted here, not attached by the scraper.** A target label
  in ``prometheus.yml`` silently *wins* over a label the exporter sets and
  renames ours to ``exported_robot_id`` — which is how two robots quietly merge
  into one series. Set it with ``--robot-id`` and leave the scrape job's
  ``labels:`` block empty.

Stdlib only, like the rest of the dashboard: the robot's own network has no
internet, so ``prometheus_client`` is not installable where this has to run.
The text format is simple enough that this is a feature, not a hardship.
"""

from __future__ import annotations

from typing import Any, Iterable

import drive
import drobot_api

PREFIX = "dt01"


def _escape(value: str) -> str:
    """Escape a label *value* per the Prometheus exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Registry:
    """Collects samples and renders them in the Prometheus text format.

    Deliberately tiny: ``gauge``/``counter`` record a value, ``render`` emits
    the ``# HELP``/``# TYPE`` header once per metric family. Passing ``None`` as
    a value is how a caller says "the robot did not report this", and the sample
    is dropped — see the module docstring.
    """

    def __init__(self, robot_id: str) -> None:
        self._robot_id = robot_id
        self._families: dict[str, dict[str, Any]] = {}

    def _add(self, name: str, kind: str, help_text: str, value: Any,
             labels: dict[str, str] | None) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            value = 1 if value else 0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if number != number:                       # NaN — same as not reported
            return
        family = self._families.setdefault(
            f"{PREFIX}_{name}", {"kind": kind, "help": help_text, "samples": []})
        tags = {"robot_id": self._robot_id, **(labels or {})}
        family["samples"].append((tags, number))

    def gauge(self, name: str, help_text: str, value: Any,
              labels: dict[str, str] | None = None) -> None:
        self._add(name, "gauge", help_text, value, labels)

    def counter(self, name: str, help_text: str, value: Any,
                labels: dict[str, str] | None = None) -> None:
        self._add(name, "counter", help_text, value, labels)

    def render(self) -> str:
        lines: list[str] = []
        for name in sorted(self._families):
            family = self._families[name]
            lines.append(f"# HELP {name} {family['help']}")
            lines.append(f"# TYPE {name} {family['kind']}")
            for tags, number in family["samples"]:
                rendered = ",".join(f'{key}="{_escape(str(tags[key]))}"'
                                    for key in sorted(tags))
                lines.append(f"{name}{{{rendered}}} {number:g}")
        return "\n".join(lines) + "\n"


def _mapping(node: Any, key: str) -> dict[str, Any]:
    """``node[key]`` when it is a dict, else ``{}``.

    Not paranoia: this controller already returns ``""`` where the reference
    documents an object (``chassis_state`` on the observed firmware), and a
    scrape that raises is indistinguishable from an exporter that is down.
    """
    if not isinstance(node, dict):
        return {}
    value = node.get(key)
    return value if isinstance(value, dict) else {}


def _dig(snapshot: dict[str, Any], *path: str) -> Any:
    """Walk nested dicts, returning None the moment a level is missing."""
    node: Any = snapshot
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _feeds(registry: Registry, link: dict[str, Any]) -> None:
    """Push-feed health: a silent socket is the first sign of a sick link."""
    for feed in ("notice", "scan"):
        status = _mapping(link, f"{feed}_feed")
        if not status:
            continue
        tags = {"feed": feed}
        registry.gauge("feed_connected",
                       "1 when the robot's push socket is connected.",
                       bool(status.get("connected")), tags)
        registry.gauge("feed_message_age_seconds",
                       "Seconds since the last message on this push feed.",
                       status.get("age_s"), tags)
        registry.counter("feed_messages_total",
                         "Messages received on this push feed since connect.",
                         status.get("messages"), tags)


def _obstacles(registry: Registry, obstacles: dict[str, Any]) -> None:
    raw = obstacles.get("sensors")
    sensors: Iterable[dict[str, Any]] = [entry for entry in (raw or [])
                                         if isinstance(entry, dict)] \
        if isinstance(raw, list) else []
    registry.gauge("obstacle_alarm",
                   "1 while any enabled proximity sensor sees something inside "
                   "the alarm distance.", bool(obstacles.get("alarm")))
    registry.gauge("obstacle_alarm_distance_meters",
                   "The configured alarm threshold, for context on the alarm.",
                   obstacles.get("alarm_distance"))
    # Nearest *enabled* reading per sensor kind. Disabled sensors and readings
    # at or below the sensor's own floor are "nothing seen", not "zero metres".
    nearest: dict[str, float] = {}
    triggered = 0
    for sensor in sensors:
        if sensor.get("triggered"):
            triggered += 1
        if not sensor.get("enabled"):
            continue
        try:
            distance = float(sensor.get("range"))
        except (TypeError, ValueError):
            continue
        if distance <= 0:
            continue
        kind = str(sensor.get("kind") or "unknown")
        nearest[kind] = min(nearest.get(kind, distance), distance)
    for kind, distance in nearest.items():
        registry.gauge("obstacle_nearest_meters",
                       "Closest live reading from any enabled sensor of this kind.",
                       distance, {"kind": kind})
    registry.gauge("obstacle_sensors_triggered",
                   "How many sensors are inside the alarm distance at once.",
                   triggered)
    registry.gauge("obstacle_sensors_enabled",
                   "Enabled proximity sensors — a drop means one was switched "
                   "off or stopped reporting.",
                   sum(1 for sensor in sensors if sensor.get("enabled")))


def render(snapshot: dict[str, Any] | None, robot_id: str,
           teleop_disabled: bool = False) -> str:
    """Render one scrape from a dashboard snapshot.

    ``snapshot`` is None when nothing is connected, which still produces a
    useful scrape: ``dt01_up 0``. A scrape that fails outright would leave
    Grafana unable to tell "robot is down" from "exporter is down".
    """
    registry = Registry(robot_id)
    registry.gauge("teleop_disabled",
                   "1 when this build refuses manual driving "
                   "(drive.TELEOP_DISABLED).", teleop_disabled)
    if not snapshot:
        registry.gauge("up", "1 when the dashboard is talking to a controller.", False)
        return registry.render()

    registry.gauge("up", "1 when the dashboard is talking to a controller.",
                   snapshot.get("reachable") is not False)

    # --- battery. The single highest-value history: it shows real runtime per
    # charge and pack decay over months, neither of which a live gauge reveals.
    battery = _mapping(snapshot, "battery")
    registry.gauge("battery_volts", "Pack voltage, normalised to volts.",
                   drobot_api.normalize_voltage(battery.get("battery_voltage")))
    registry.gauge("battery_percent", "Reported state of charge.",
                   battery.get("battery_capacity_percentage"))
    # Omitted entirely when unknown — on this firmware it usually is, so
    # `charging` is better inferred from the slope of battery_percent.
    registry.gauge("charging", "1 while the pack is charging, absent when the "
                   "robot does not report it.",
                   drobot_api.charging_state(battery, _mapping(snapshot, "chassis")))

    # --- localisation. Score degrades *before* navigation starts failing, which
    # makes it the closest thing here to an early warning.
    localization = _mapping(snapshot, "localization")
    registry.gauge("localized", "1 when the robot knows where it is.",
                   bool(snapshot.get("localized")))
    registry.gauge("localization_score", "Match quality against the map, 0-1.",
                   localization.get("score"))

    # --- safety interlocks.
    bumper = _mapping(snapshot, "bumper")
    registry.gauge("brake_engaged", "1 while the software brake is locked.",
                   bool(_dig(snapshot, "brake", "engaged")))
    registry.gauge("estop_active", "1 while the chassis emergency stop is pressed.",
                   bumper.get("estop"))
    for position in ("front", "rear"):
        registry.gauge("bumper_pressed", "1 while this bumper is pressed.",
                       bumper.get(position), {"position": position})
    registry.gauge("safety_stale",
                   "1 when bumper/e-stop readings are too old to trust — the "
                   "interlocks are blind and motion is refused.",
                   bool(snapshot.get("safety_stale")))
    registry.gauge("safety_data_age_seconds",
                   "Age of the newest bumper/e-stop reading.",
                   snapshot.get("safety_data_age"))

    _obstacles(registry, _mapping(snapshot, "obstacles"))
    guard = _mapping(_mapping(snapshot, "obstacles"), "protection")
    registry.gauge("manual_obstacle_protection_enabled", "1 when detected obstacle hazards are enforced during manual driving.", guard.get("enabled"))
    registry.gauge("manual_obstacle_protection_ready", "1 when configured proximity and cliff readings are ready and clear; unavailable readings warn without blocking manual driving.", guard.get("ready"))
    registry.gauge("manual_obstacle_unavailable_warn_only", "1 when unavailable obstacle readings warn instead of blocking manual driving.", guard.get("unavailable_policy") == "warn")
    registry.gauge("manual_obstacle_sensor_stale_seconds", "Configured maximum sensor age for manual driving.", guard.get("sensor_stale_s"))
    registry.gauge("manual_obstacle_stop_latched", "1 after an obstacle stop awaiting explicit acknowledgement.", bool(_dig(snapshot, "drive", "protection_latched")))

    # --- motion and navigation.
    velocity = _mapping(snapshot, "velocity")
    registry.gauge("velocity_linear_mps", "Reported forward velocity.",
                   velocity.get("linear"))
    registry.gauge("velocity_angular_radps", "Reported yaw rate.",
                   velocity.get("angular"))
    nav_type = str(_dig(snapshot, "nav", "nav_status_type") or "idle")
    # drive.BUSY_NAV_TYPES, not a guessed list: these are the exact strings the
    # controller reports, and the motion gate already refuses teleop on them.
    registry.gauge("nav_active",
                   "1 while the controller is executing a navigation goal.",
                   nav_type in drive.BUSY_NAV_TYPES)
    registry.gauge("nav_state", "Navigation state, carried as a label.",
                   1, {"state": nav_type})
    registry.gauge("robot_state", "Controller operating state, as a label.",
                   1, {"state": str((_mapping(snapshot, "robot"))
                                    .get("state_operation") or "unknown")})

    drive_status = _mapping(snapshot, "drive")
    registry.gauge("drive_armed", "1 while a teleop intent is live.",
                   bool(drive_status.get("armed")))
    registry.counter("drive_commands_total",
                     "Motion commands sent to the base since connect.",
                     drive_status.get("commands_sent"))

    # --- link health. Poll errors are how a half-working controller shows up:
    # the socket is fine, individual endpoints are not.
    link = _mapping(snapshot, "link")
    _feeds(registry, link)
    registry.gauge("poll_errors",
                   "Endpoints currently failing — nonzero means the controller "
                   "is answering some requests but not others.",
                   len(_mapping(link, "poll_errors")))
    return registry.render()
