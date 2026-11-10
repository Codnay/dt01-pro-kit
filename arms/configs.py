"""Configuration parameters for Piper robot demos."""

from pathlib import Path

URDF_PATH = str(
    Path(__file__).parent.parent
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)

GRIPPER_FRAME_NAME = "gripper_center"


# Pink IK parameters
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.25
FRAME_TASK_GAIN = 0.5
LM_DAMPING = 0.25
DAMPING_COST = 0.4
SOLVER_DAMPING_VALUE = 1e-12

# Controller 1€ Filter parameters
CONTROLLER_MIN_CUTOFF = 0.8  # Minimum cutoff frequency (stabilizes when holding still)
CONTROLLER_BETA = 2.0  # Speed coefficient (reduces lag when moving)
CONTROLLER_D_CUTOFF = 0.9  # Derivative cutoff frequency

# Controller parameters
GRIP_THRESHOLD = 0.75  # Trigger threshold to activate teleop control
TELEOP_TRIGGER_RELEASE_THRESHOLD = 0.55  # Lower threshold to release teleop
TELEOP_DEADMAN_RELEASE_CONSECUTIVE_SAMPLES = 2  # ~0.24s at 50 Hz
# Press must also be debounced: require N consecutive samples with the trigger
# held AND valid controller tracking before activating teleop. Without this a
# single bouncy high-trigger frame on tracking recovery spuriously activates and
# re-anchors the teleop origin, lurching the arm while controllers are at rest.
TELEOP_DEADMAN_PRESS_CONSECUTIVE_SAMPLES = 2  # ~0.06s at 50 Hz
# Re-arming requires the trigger to be observed RELEASED (below the release
# threshold) on valid frames for this many consecutive samples. A real hand
# release lasts well over 100ms; a flaky/intermittently-stuck trigger that only
# dips low for a frame or two will not qualify, so it cannot re-arm teleop.
TELEOP_DEADMAN_RELEASE_OBSERVE_SAMPLES = 2  # ~0.10s at 50 Hz

# Scaling factors for translation and rotation
TRANSLATION_SCALE = 0.4
ROTATION_SCALE = 0.3

# Per-hand teleop frame correction in roll/pitch/yaw (degrees), applied to
# controller deltas before IK target generation (ik_solver: delta = Rz(yaw) @ delta).
# Format: [roll, pitch, yaw]
#
# This is a MOUNT correction: it must match how each arm is physically bolted
# down. Both arms currently face the same way, so both take the same +90.
#
# Symptom -> fix, since this is easy to get wrong:
# - left/right AND forward/back both reversed, up/down fine -> yaw is 180 deg out.
# - only ONE horizontal axis reversed -> NOT fixable here. That is a mirror, and
#   a rotation cannot flip a single axis; the mount does not match this model.
# - up/down wrong -> not this setting at all (roll/pitch, or the leader axis map).
#
# Example for MIRRORED side-by-side mounts (arms angled 90 deg inward, facing
# each other) -- not the current layout:
# TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND = {
#     "right": [-45.0, 0.0, 90.0],
#     "left": [45.0, 0.0, -90.0],
# }

TELEOP_DELTA_RPY_OFFSET_DEG_BY_HAND = {'right': [-45.0, 0.0, 90.0], 'left': [45.0, 0.0, -90.0]}


# Thread rates (Hz)
CONTROLLER_DATA_RATE = 50.0  # Controller input reading
# IK solving. Every solve past ROBOT_RATE (100 Hz) is computed and discarded --
# the arm cannot accept commands faster -- and the input it solves against only
# refreshes at CONTROLLER_DATA_RATE (50 Hz). At 250 Hz two solvers (one per arm,
# ~0.95 ms each on the Jetson) took ~45% of the parent process's single GIL,
# which starved the teleop threads: setpoints went out late and bursty, and the
# stalls made shared-memory snapshots look stale enough to log phantom
# "control_loop not running" warnings. 125 Hz keeps 2.5x headroom over the input
# rate and bounds setpoint staleness at 8 ms, a fixed cost far cheaper than the
# unbounded lateness GIL saturation produces.
IK_SOLVER_RATE = 125.0
VISUALIZATION_RATE = 60.0  # GUI updates
ROBOT_RATE = 100.0


# ---------------------------------------------------------------------------
# Teleop safety guards (VR-glitch -> arm disarm/drop protection)
# ---------------------------------------------------------------------------
# A VR controller tracking glitch can teleport the commanded pose, producing an
# IK target far outside the reachable workspace. The URDF declares joint velocity
# limits of 10 rad/s (~573 deg/s), so Pink's QP authorizes joint slews well above
# the real Piper firmware joint-speed limit (~170 deg/s). When commanded past that
# limit the joint driver disarms (a clean torque cut, no fault flag) and -- since
# Piper joints have no brakes -- the arm drops. The two guards below prevent this.

# (1) Controller pose jump-rejection. If the raw controller position moves faster
# than a human plausibly can between two controller samples, treat it as a
# tracking glitch and hold the previous target instead of feeding the teleported
# pose into IK. 3 m/s is well above normal teleop hand speed.
TELEOP_CONTROLLER_JUMP_REJECT_ENABLED = False
TELEOP_CONTROLLER_MAX_SPEED_M_S = 6.0

# (2) Joint slew clamp. Rate-limit the joint setpoint so it never advances faster
# than the hardware can safely follow, regardless of the target. Per joint, in
# deg/s, kept below the firmware joint-speed trip with margin. Raise toward the
# SDK `max_joint_spd` per joint if the arms feel rate-limited; lowering is always
# safe.
#
# Single source of truth for BOTH flows that rate-limit joint motion:
#   - teleop IK guard (arms/threads/ik_solver.py) uses this directly;
#   - the preset/trajectory path (ArmPresetDashboardController) uses it as the
#     offline fallback cap when live SDK limits (get_joint_velocity_limits_deg_s,
#     the real per-joint firmware ceilings) are unavailable.
TELEOP_JOINT_SLEW_CLAMP_ENABLED = False
MAX_JOINT_SLEW_DEG_S = (220.0, 220.0, 220.0, 220.0, 220.0, 220.0)

# Teleop reference acceleration shaping (startup-loaded, deg/s^2).
TELEOP_ACCEL_LIMIT_ENABLED = True
USE_AGILEX_TELEOP = True  # Teleop + dashboard Cartesian jog; restart required.
TELEOP_MAX_JOINT_ACCEL_DEG_S2 = (180.0, 180.0, 180.0, 180.0, 180.0, 180.0)

# Teleop tracking-error guard: maximum desired-minus-measured angle, in degrees.
# J1, J2, J3, J4, J5, J6; applies to both arms. Six finite positive values.
# Constrains target advancement without ending teleop; resumes as the arm follows.
# Existing-reference error above 2x a limit remains a fault stop. Restart to apply.
TELEOP_MAX_TRACKING_ERROR_DEG = (20.0, 20.0, 20.0, 40.0, 40.0, 40.0)

# Neuracore data collection rates
JOINT_STATE_STREAMING_RATE = 100.0  # Data collection rate for neuracore
CAMERA_FRAME_STREAMING_RATE = 30.0  # Data collection rate for camera frame
# Depth-to-color alignment + the depth numpy pipeline are CPU-heavy. Process
# depth at this (lower) rate instead of every color frame so it does not starve
# the real-time IK/robot control loops. Color/RGB stays at the full rate above.
DEPTH_PROCESS_RATE = 0.0
CAMERA_ENABLE_AUTO_EXPOSURE = True
# When enabled, auto-exposure may reduce FPS in low light to brighten frames.
# Mirrors RealSense "Auto Exposure Priority" behavior.
CAMERA_ENABLE_AUTO_EXPOSURE_PRIORITY = False
# Warmup frames dropped after stream start so auto-exposure can settle.
CAMERA_AUTO_EXPOSURE_WARMUP_FRAMES = 30
# Optional path to a RealSense JSON settings file exported from realsense-viewer.
# When set, it is loaded on stream start to match viewer controls more closely.
# Example: "/absolute/path/to/my_realsense_settings.json"
CAMERA_REALSENSE_VIEWER_JSON_PRESET_PATH: str | None = None

# Initial neutral pose for robot (degrees)
# Single-arm examples use this default value.
NEUTRAL_JOINT_ANGLES = [-1.003, 80.167, -51.064, -4.127, 16.548, 2.619]

# Bimanual neutral poses (degrees), can be tuned per arm.
# Defaults are aligned to single-arm neutral for backward compatibility.
NEUTRAL_JOINT_ANGLES_RIGHT = [120.0, 28.0, -12.0, -56.0, 49.0, 96.0]
NEUTRAL_JOINT_ANGLES_LEFT = [-94.0, 39.0, -4.0, -90.0, -8.0, 99.0]
NEUTRAL_JOINT_ANGLES_BY_HAND = {
    "right": NEUTRAL_JOINT_ANGLES_RIGHT,
    "left": NEUTRAL_JOINT_ANGLES_LEFT,
}

# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


POLICY_EXECUTION_RATE = 100.0  # Hz
PREDICTION_HORIZON_EXECUTION_RATIO = (
    0.8  # percentage of the prediction horizon that is executed
)
MAX_SAFETY_THRESHOLD = 20.0  # degrees
MAX_ACTION_ERROR_THRESHOLD = 3.0  # degrees
TARGETING_POSE_TIME_THRESHOLD = 1.0  # seconds

# Motor over-temperature auto-pause watchdog. When any active arm's motor or
# driver (FOC) temperature crosses the PAUSE limit, the runtime auto-pauses +
# holds the arms (same behaviour as Button A) and latches so it will not re-arm
# on its own — an operator must re-enable. This is a soft pause that keeps
# motors powered (no torque cut / power drop).
TEMPERATURE_WATCHDOG_ENABLED = True
MOTOR_TEMP_PAUSE_THRESHOLD_C = 60.0  # motor coil temp (SDK motor_temp)
DRIVER_TEMP_PAUSE_THRESHOLD_C = 60.0  # motor driver / FOC temp (SDK foc_temp)
# Re-arm (hysteresis) thresholds: after an auto-pause the latch only clears once
# EVERY joint has cooled below these, so a manual re-enable will stick instead of
# instantly re-pausing. They MUST sit below the pause thresholds AND above the
# arm's normal resting temps (idle motor ~40-46C, driver ~44-48C) — otherwise the
# arm can never re-arm. Re-enabling while still above the pause threshold
# re-pauses by design (safety). For a bench test, set the PAUSE value just above
# idle and the REARM value ~1C below PAUSE (both still above resting).
MOTOR_TEMP_REARM_THRESHOLD_C = 62.0
DRIVER_TEMP_REARM_THRESHOLD_C = 46.0
TEMPERATURE_WATCHDOG_POLL_PERIOD_S = 1.0  # how often to poll per-joint temps

GRIPPER_LOGGING_NAME = "gripper"
CAMERA_LOGGING_NAME = "rgb"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Bimanual joint/gripper names for dual-arm Neuracore logging
BIMANUAL_JOINT_NAMES = {
    "right": [
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
    ],
    "left": [
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
    ],
}
BIMANUAL_GRIPPER_NAMES = {"right": "right_gripper", "left": "left_gripper"}

# Speed scaling (0.1 to 1.0) — interpolates between previous and new targets
SPEED_SCALE = 1.0

# Soft per-joint margins (degrees) inward from SDK joint limits to avoid firmware
# torque protection near mechanical extremes. Keep keys explicit to avoid unit/name mixups.
JOINT_LIMIT_MARGIN_DEG_BY_JOINT = {'joint1': 2.0, 'joint2': 2.0, 'joint3': 2.0, 'joint4': 2.0, 'joint5': 2.0, 'joint6': 10.0}

# Joystick J6 rotation control
J6_JOYSTICK_SPEED = 90.0  # degrees/second at full joystick deflection
J6_JOYSTICK_DEADZONE = 0.1  # ignore joystick values below this (0.0-1.0)

# LeRobot dataset recording frame rate
DATASET_FPS = 20


# SDK MotionCtrl_2 speed percentage for the X-button homing move.
# PIPER_HOME_SPEED_PERCENT still overrides this per process.
HOME_SPEED_PERCENT = 2

# Firmware per-motor speed ceilings (0.001 rad/s), asserted at teleop start.
# None leaves the ceiling already stored in the motor firmware unchanged.
MOTOR_SPEED_RAW_BY_HAND: dict[str, list[int] | None] = {'right': [900, 900, 900, 900, 900, 900], 'left': [900, 900, 900, 900, 900, 900]}


# --- Machine config loader (not part of the editable file) ---
# The complete constants above seed config/configs.py once on each machine.
# That gitignored file is then authoritative across stack/dashboard restarts.
from machine_config import apply as _apply_machine_config  # noqa: E402  # isort: skip

# Runtime provenance, kept outside the editable constants.
ARM_CONFIG_PATH = ""
ARM_CONFIG_LOADED = False
_apply_machine_config(globals())
