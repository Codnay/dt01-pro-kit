"""Load, validate and rewrite ``config/arm_config.json``.

Per-deployment arm settings: mount orientation, homing pose, motor speed
ceilings and the teleop motion scales. The dashboard edits this file; the teleop
stack reads it at startup (:mod:`arms.arm_config`). The two sides
share the file format, not code -- this module imports nothing from the robot
stack, for the same reason the rest of ``ops/`` does not: the panel has to keep
working, and keep its Stop button, when the robot venv is broken.

Same custody rules as ``jobs.json``: the tracked artifact is
``arm_config.example.json``, the live file is gitignored and local to one robot
cell, and every write validates first and keeps a ``.bak``. A dashboard that
corrupts its own config is a dashboard that hands the next operator a bad arm
pose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

OPS_DIR = Path(__file__).resolve().parent
REPO_DIR = OPS_DIR.parent
CONFIG_PATH = REPO_DIR / "config" / "arm_config.json"
EXAMPLE_PATH = REPO_DIR / "config" / "arm_config.example.json"

HANDS = ("right", "left")

# Upper bound on the firmware per-motor speed ceiling (MotorMaxSpdSet, in
# 0.001 rad/s units). Mirrors MOTOR_SPEED_CONFIG_MAX_RAW in
# arms/preset_dashboard/models.py -- duplicated rather than imported
# because this module must load without the robot stack. 1500 raw = 86 deg/s.
MAX_MOTOR_SPEED_RAW = 1500

# Hard ceiling on the joint slew clamp, in deg/s. This is a SAFETY bound, not a
# preference: the joint driver disarms when commanded well past its rated speed,
# and because the joints have no brakes, a disarm drops the arm. The bound exists
# so a config file cannot remove the guard entirely -- the URDF alone would allow
# 573 deg/s.
#
# Raised from 180 to 270 on request (2026-09-16). Note what this does NOT do:
# the arm cannot execute faster than the firmware per-motor ceiling
# (MotorMaxSpdSet, capped at MAX_MOTOR_SPEED_RAW = 1500 raw = 86 deg/s), so a
# slew clamp above ~86 is slack in the rope rather than extra speed. Lowering
# this is always safe; raising it only widens how far a VR tracking glitch may
# slew the setpoint before the guard catches it.
MAX_JOINT_SLEW_CEILING_DEG_S = 270.0

# Generous static bound on a homing joint angle. The real authority is the
# controller's EFFECTIVE_JOINT_LIMITS (SDK limits minus a soft margin), which
# clamps the target anyway; this only catches a fat-fingered entry (a stray
# digit, radians typed where degrees belong) before it reaches the arm.
MAX_JOINT_ANGLE_DEG = 200.0


class ArmConfigError(ValueError):
    """Raised for any arm config the dashboard refuses to write."""


def _seed_from_example(path: Path) -> None:
    """Create a missing ``arm_config.json`` from the tracked template.

    A fresh clone has no local file (it is gitignored), so without this the
    Arms tab would be empty on a new cell until someone copied a file by hand.
    Copying is one-shot: once the local file exists it is authoritative and the
    template is never consulted again.
    """
    if path != CONFIG_PATH or not EXAMPLE_PATH.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.seed")
    tmp.write_text(EXAMPLE_PATH.read_text())
    try:
        # Lose the race rather than clobber: another process may have seeded or
        # written the real config between the exists() check and here.
        os.link(tmp, path)
    except FileExistsError:
        return
    finally:
        tmp.unlink(missing_ok=True)
    print(
        f"[ops] seeded {path.name} from {EXAMPLE_PATH.name} -- edit it for this "
        f"robot cell (mount orientation, homing pose, motor speed). It is local "
        f"to this host and is not tracked by git."
    )


def load_raw(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the arm config as plain JSON, seeding it from the template once."""
    if not path.exists():
        _seed_from_example(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ArmConfigError(f"arm config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArmConfigError(f"{path.name} is not valid JSON: {exc}") from exc


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArmConfigError(f"{where} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ArmConfigError(f"{where} must be finite")
    return number


def _vector(value: Any, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ArmConfigError(f"{where} must be a list of {length} numbers")
    return [_number(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _bounded(value: float, low: float, high: float, where: str) -> float:
    if not low <= value <= high:
        raise ArmConfigError(f"{where} must be between {low:g} and {high:g}")
    return value


def parse(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the raw config and return it normalised.

    Every field is optional: a cell that omits a key keeps the shipped default
    from ``arms/configs.py``. Only what is present is checked, so a
    partial file is valid and an operator can pin one arm's mount angle without
    restating the whole document.
    """
    if not isinstance(raw, dict):
        raise ArmConfigError("top level of arm_config.json must be an object")

    out: dict[str, Any] = {}

    arms_raw = raw.get("arms") or {}
    if not isinstance(arms_raw, dict):
        raise ArmConfigError("'arms' must be an object")
    arms: dict[str, Any] = {}
    for hand, entry in arms_raw.items():
        if hand not in HANDS:
            raise ArmConfigError(f"unknown arm {hand!r}: expected right or left")
        if not isinstance(entry, dict):
            raise ArmConfigError(f"arms.{hand} must be an object")
        clean: dict[str, Any] = {}

        if "home_joint_angles_deg" in entry:
            angles = _vector(
                entry["home_joint_angles_deg"], 6, f"arms.{hand}.home_joint_angles_deg"
            )
            clean["home_joint_angles_deg"] = [
                _bounded(
                    a,
                    -MAX_JOINT_ANGLE_DEG,
                    MAX_JOINT_ANGLE_DEG,
                    f"arms.{hand}.home_joint_angles_deg[{i}]",
                )
                for i, a in enumerate(angles)
            ]

        if "teleop_delta_rpy_offset_deg" in entry:
            rpy = _vector(
                entry["teleop_delta_rpy_offset_deg"],
                3,
                f"arms.{hand}.teleop_delta_rpy_offset_deg",
            )
            clean["teleop_delta_rpy_offset_deg"] = [
                _bounded(
                    v, -360.0, 360.0, f"arms.{hand}.teleop_delta_rpy_offset_deg[{i}]"
                )
                for i, v in enumerate(rpy)
            ]

        if "motor_speed_raw" in entry:
            speeds = _vector(
                entry["motor_speed_raw"], 6, f"arms.{hand}.motor_speed_raw"
            )
            clean["motor_speed_raw"] = [
                int(
                    _bounded(
                        s, 1, MAX_MOTOR_SPEED_RAW, f"arms.{hand}.motor_speed_raw[{i}]"
                    )
                )
                for i, s in enumerate(speeds)
            ]

        arms[hand] = clean
    if arms:
        out["arms"] = arms

    motion_raw = raw.get("motion") or {}
    if not isinstance(motion_raw, dict):
        raise ArmConfigError("'motion' must be an object")
    motion: dict[str, Any] = {}

    if "translation_scale" in motion_raw:
        motion["translation_scale"] = _bounded(
            _number(motion_raw["translation_scale"], "motion.translation_scale"),
            0.05,
            2.0,
            "motion.translation_scale",
        )
    if "rotation_scale" in motion_raw:
        motion["rotation_scale"] = _bounded(
            _number(motion_raw["rotation_scale"], "motion.rotation_scale"),
            0.05,
            2.0,
            "motion.rotation_scale",
        )
    if "max_joint_slew_deg_s" in motion_raw:
        slew = _vector(
            motion_raw["max_joint_slew_deg_s"], 6, "motion.max_joint_slew_deg_s"
        )
        motion["max_joint_slew_deg_s"] = [
            _bounded(
                v,
                1.0,
                MAX_JOINT_SLEW_CEILING_DEG_S,
                f"motion.max_joint_slew_deg_s[{i}]",
            )
            for i, v in enumerate(slew)
        ]
    if "home_speed_percent" in motion_raw:
        motion["home_speed_percent"] = int(
            _bounded(
                _number(motion_raw["home_speed_percent"], "motion.home_speed_percent"),
                1,
                100,
                "motion.home_speed_percent",
            )
        )
    if motion:
        out["motion"] = motion

    return out


def _write(body: str, path: Path) -> None:
    """Back up the current file, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Derived from ``path``, not the module constant, so a caller writing
        # somewhere else (a test) cannot overwrite the real host's backup.
        path.with_name(path.name + ".bak").write_text(path.read_text())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(body)
    os.replace(tmp, path)  # atomic: readers see old or new, never a partial file


def save_raw(raw: dict[str, Any], path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Validate ``raw``, back up the current file, then write it atomically.

    Nothing is written unless :func:`parse` accepts it, so a bad value typed in
    the browser cannot leave the next teleop start with an unreachable homing
    pose or a motor speed the firmware will refuse.
    """
    parsed = parse(raw)  # raises ArmConfigError before anything is touched
    _write(json.dumps(raw, indent=2) + "\n", path)
    return parsed


def save_text(text: str, path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Validate ``text`` as the whole file, then write it back VERBATIM.

    The difference from :func:`save_raw` matters for the file editor: round
    tripping through ``json.dumps`` would reflow whoever's hand-written layout
    and reorder nothing but still churn the file on every save. Someone editing
    the document directly expects their own formatting, spacing and comment
    block to come back unchanged, so the text is validated and then stored as
    typed. Values are never rewritten either way.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArmConfigError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArmConfigError("top level of arm_config.json must be an object")
    parsed = parse(raw)  # raises before anything is touched
    _write(text if text.endswith("\n") else text + "\n", path)
    return parsed


def read_text(path: Path = CONFIG_PATH) -> str:
    """The file exactly as it sits on disk, seeding it from the template once."""
    load_raw(path)  # seeds a missing file, and rejects invalid JSON early
    return path.read_text()


def limits() -> dict[str, Any]:
    """Validation bounds, so the UI can constrain inputs instead of guessing."""
    return {
        "hands": list(HANDS),
        "max_motor_speed_raw": MAX_MOTOR_SPEED_RAW,
        "max_joint_slew_deg_s": MAX_JOINT_SLEW_CEILING_DEG_S,
        "max_joint_angle_deg": MAX_JOINT_ANGLE_DEG,
        # raw = deg/s * (pi/180) * 1000. Given to the UI so it can show deg/s
        # next to the raw box without hardcoding the conversion in two places.
        "raw_per_deg_s": 1000.0 * 3.141592653589793 / 180.0,
    }
