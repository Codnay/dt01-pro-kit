"""Full Python configuration shared by the stdlib ops panel and the runtime.

The local file is data expressed in Python: comments, constant assignments,
references to earlier constants, arithmetic and the shipped Path expression.
Evaluate that subset without executing user code in the dashboard or robot.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from ops.arm_config import ArmConfigError
from ops.arm_config import parse as parse_legacy

REPO_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = REPO_DIR / "arms" / "configs.py"
CONFIG_PATH = REPO_DIR / "config" / "configs.py"
LOADER_MARKER = "# --- Machine config loader (not part of the editable file) ---"
_LOCK = threading.RLock()


class MachineConfigError(ValueError):
    """An invalid or stale machine configuration."""


def config_path() -> Path:
    """Resolve the same persistent file for the dashboard and the runtime."""
    override = os.environ.get("PIPER_CONFIG_PATH", "").strip()
    return Path(override).expanduser().resolve() if override else CONFIG_PATH


def defaults_text() -> str:
    """Read all shipped constants and comments, excluding the import hook."""
    return DEFAULTS_PATH.read_text(encoding="utf-8").split(LOADER_MARKER)[0]


def _expression(node: ast.AST | None, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_expression(item, values) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Dict):
        return {
            _expression(key, values): _expression(value, values)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _expression(node.operand, values)
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _expression(node.left, values), _expression(node.right, values)
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        operation = operations.get(type(node.op))
        if operation:
            # Bound repetitions before allocating a list/string in an HTTP request.
            if isinstance(node.op, ast.Mult):
                for sequence, count in ((left, right), (right, left)):
                    if isinstance(sequence, (list, tuple, str)) and isinstance(
                        count, int
                    ):
                        if len(sequence) * count > 10000:
                            raise MachineConfigError("repeated value is too large")
            return operation(left, right)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        value = _expression(node.value, values)
        if isinstance(value, Path):
            return value.parent
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if len(node.args) == 1 and not node.keywords:
            value = _expression(node.args[0], values)
            if node.func.id == "Path" and isinstance(value, (str, Path)):
                return Path(value)
            if node.func.id == "str":
                return str(value)
    raise MachineConfigError(
        f"line {getattr(node, 'lineno', '?')}: use constant values, references, "
        "arithmetic or Path expressions; executable code is not supported"
    )


def _assignments(text: str) -> dict[str, Any]:
    if len(text.encode("utf-8")) > 256_000:
        raise MachineConfigError("configs.py must be smaller than 256 KB")
    try:
        tree = ast.parse(text, filename="configs.py")
        # __file__ retains the shipped module's location, so copying the config
        # into config/ never shifts URDF_PATH outside this checkout.
        values: dict[str, Any] = {"__file__": str(DEFAULTS_PATH)}
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    continue  # module docstrings / section notes
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and (
                    (
                        node.module == "pathlib"
                        and [(n.name, n.asname) for n in node.names] == [("Path", None)]
                    )
                    or (
                        node.module == "__future__"
                        and [(n.name, n.asname) for n in node.names]
                        == [("annotations", None)]
                    )
                ):
                    continue
            target = None
            expression = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                expression = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target = node.target
                expression = node.value
            if (
                not isinstance(target, ast.Name)
                or not target.id.isupper()
                or target.id.startswith("_")
            ):
                raise MachineConfigError(
                    f"line {node.lineno}: expected an UPPERCASE config assignment"
                )
            if target.id in values:
                raise MachineConfigError(
                    f"line {node.lineno}: duplicate setting {target.id}"
                )
            values[target.id] = _expression(expression, values)
        del values["__file__"]
        return values
    except MachineConfigError:
        raise
    except SyntaxError as exc:
        raise MachineConfigError(f"line {exc.lineno}: {exc.msg}") from exc
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError) as exc:
        raise MachineConfigError(f"invalid config expression: {exc}") from exc


def _check_type(value: Any, default: Any, name: str) -> None:
    if default is None:
        if value is None or isinstance(value, str):
            return
    elif isinstance(default, bool):
        if isinstance(value, bool):
            return
    elif isinstance(default, (int, float)):
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            integer_setting = (
                isinstance(default, int) and "[" not in name and "." not in name
            )
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if finite and (not integer_setting or isinstance(value, int)):
                return
    elif isinstance(default, str):
        if isinstance(value, str):
            return
    elif isinstance(default, (tuple, list)):
        if isinstance(value, (tuple, list)) and len(value) == len(default):
            for i, (item, expected) in enumerate(zip(value, default)):
                _check_type(item, expected, f"{name}[{i}]")
            return
    elif isinstance(default, dict) and isinstance(value, dict):
        if value.keys() == default.keys():
            for key in default:
                _check_type(value[key], default[key], f"{name}.{key}")
            return
    raise MachineConfigError(
        f"{name}: expected {type(default).__name__} with the original shape and finite numbers"
    )


def parse(text: str) -> dict[str, Any]:
    """Validate the full document and retain the existing arm safety bounds."""
    values = _assignments(text)
    defaults = _assignments(defaults_text())
    # Older complete machine files predate these optional settings. Preserve
    # their existing tuning rather than rejecting the entire file on upgrade.
    for name in (
        "TELEOP_ACCEL_LIMIT_ENABLED",
        "TELEOP_MAX_JOINT_ACCEL_DEG_S2",
        "USE_AGILEX_TELEOP",
        "TELEOP_MAX_TRACKING_ERROR_DEG",
    ):
        values.setdefault(name, defaults[name])
    missing = defaults.keys() - values.keys()
    if missing:
        raise MachineConfigError("missing settings: " + ", ".join(sorted(missing)))
    for name, default in defaults.items():
        if name != "MOTOR_SPEED_RAW_BY_HAND":
            _check_type(values[name], default, name)
    if any(v <= 0 for v in values["TELEOP_MAX_JOINT_ACCEL_DEG_S2"]):
        raise MachineConfigError("TELEOP_MAX_JOINT_ACCEL_DEG_S2 must be positive")
    if any(v <= 0 for v in values["TELEOP_MAX_TRACKING_ERROR_DEG"]):
        raise MachineConfigError("TELEOP_MAX_TRACKING_ERROR_DEG must be positive")
    motors = values["MOTOR_SPEED_RAW_BY_HAND"]
    if not isinstance(motors, dict) or motors.keys() != {"right", "left"}:
        raise MachineConfigError("MOTOR_SPEED_RAW_BY_HAND must contain right and left")
    arms = {}
    for hand in ("right", "left"):
        angles = values[f"NEUTRAL_JOINT_ANGLES_{hand.upper()}"]
        if list(values["NEUTRAL_JOINT_ANGLES_BY_HAND"][hand]) != list(angles):
            raise MachineConfigError(
                f"NEUTRAL_JOINT_ANGLES_BY_HAND.{hand} must match NEUTRAL_JOINT_ANGLES_{hand.upper()}"
            )
        arms[hand] = {
            "home_joint_angles_deg": list(angles),
            "teleop_delta_rpy_offset_deg": list(
                values["TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND"][hand]
            ),
        }
        if motors[hand] is not None:
            speeds = motors[hand]
            if not isinstance(speeds, (list, tuple)) or any(
                type(v) is not int for v in speeds
            ):
                raise MachineConfigError(
                    f"MOTOR_SPEED_RAW_BY_HAND.{hand}: expected six integers or None"
                )
            arms[hand]["motor_speed_raw"] = list(speeds)
    try:
        parse_legacy(
            {
                "arms": arms,
                "motion": {
                    "translation_scale": values["TRANSLATION_SCALE"],
                    "rotation_scale": values["ROTATION_SCALE"],
                    "home_speed_percent": values["HOME_SPEED_PERCENT"],
                    "max_joint_slew_deg_s": list(values["MAX_JOINT_SLEW_DEG_S"]),
                },
            }
        )
    except ArmConfigError as exc:
        raise MachineConfigError(str(exc)) from exc
    return values


def _seed_text() -> str:
    text = defaults_text()
    legacy_path = Path(
        os.environ.get("ARM_CONFIG_PATH") or REPO_DIR / "config" / "arm_config.json"
    ).expanduser()
    if not legacy_path.exists():
        return text
    import json

    try:
        legacy = parse_legacy(json.loads(legacy_path.read_text(encoding="utf-8")))
    except (ValueError, OSError) as exc:
        raise MachineConfigError(f"cannot migrate {legacy_path}: {exc}") from exc
    replacements = {}
    for old, new in {
        "translation_scale": "TRANSLATION_SCALE",
        "rotation_scale": "ROTATION_SCALE",
        "home_speed_percent": "HOME_SPEED_PERCENT",
        "max_joint_slew_deg_s": "MAX_JOINT_SLEW_DEG_S",
    }.items():
        if old in legacy.get("motion", {}):
            replacements[new] = legacy["motion"][old]
    defaults = _assignments(text)
    for hand, entry in legacy.get("arms", {}).items():
        if "home_joint_angles_deg" in entry:
            replacements[f"NEUTRAL_JOINT_ANGLES_{hand.upper()}"] = entry[
                "home_joint_angles_deg"
            ]
        for old, new in {
            "motor_speed_raw": "MOTOR_SPEED_RAW_BY_HAND",
            "teleop_delta_rpy_offset_deg": "TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND",
        }.items():
            if old in entry:
                replacements.setdefault(new, defaults[new])[hand] = entry[old]
    # Replace only expression spans: comments, annotations and references (e.g.
    # the neutral-pose dict) remain part of the full editable Python document.
    source = text.encode("utf-8")
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign)
            else getattr(node, "target", None)
        )
        if isinstance(target, ast.Name) and target.id in replacements:
            value = node.value
            edits.append(
                (
                    offsets[value.lineno - 1] + value.col_offset,
                    offsets[(value.end_lineno or value.lineno) - 1]
                    + (value.end_col_offset or 0),
                    repr(replacements[target.id]).encode("utf-8"),
                )
            )
    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source.decode("utf-8")


def _write(text: str, path: Path, *, seed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(text.encode("utf-8"))
            tmp.flush()
            os.fsync(tmp.fileno())
            os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
            if seed:
                try:
                    os.link(
                        temporary, path
                    )  # never overwrite a concurrently seeded file
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_text() -> str:
    """Return exact disk text, even when invalid so the editor can repair it."""
    with _LOCK:
        path = config_path()
        if not path.exists():
            text = _seed_text()
            parse(text)
            _write(text, path, seed=True)
        return path.read_text(encoding="utf-8")


def revision(text: str) -> str:
    """Identify a loaded revision to reject stale browser saves."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_text(text: str, expected_revision: str) -> str:
    """Validate, reject stale edits, back up and atomically persist the file."""
    parse(text)
    with _LOCK:
        previous = read_text()
        if revision(previous) != expected_revision:
            raise MachineConfigError(
                "configs.py changed on disk. Reload before saving your changes."
            )
        path = config_path()
        _write(previous, path.with_name(path.name + ".bak"))
        _write(text, path)
    return revision(text)


def apply(namespace: dict[str, Any]) -> None:
    """Load machine values at import, preserving shipped defaults on failure."""
    namespace["ARM_CONFIG_PATH"] = str(config_path())
    namespace["ARM_CONFIG_LOADED"] = False
    try:
        values = parse(read_text())
    except (MachineConfigError, OSError, UnicodeError) as exc:
        print(f"[configs] ignoring {config_path()}: {exc}; using shipped defaults")
        return
    namespace.update(values)
    namespace["ARM_CONFIG_PATH"] = str(config_path())
    namespace["ARM_CONFIG_LOADED"] = True
