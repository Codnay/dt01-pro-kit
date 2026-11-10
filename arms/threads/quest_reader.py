"""Quest reader thread - reads controller data and manages teleop state."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from arms.configs import (
    CONTROLLER_DATA_RATE,
    GRIP_THRESHOLD,
    ROTATION_SCALE,
    TELEOP_DEADMAN_PRESS_CONSECUTIVE_SAMPLES,
    TELEOP_DEADMAN_RELEASE_CONSECUTIVE_SAMPLES,
    TELEOP_DEADMAN_RELEASE_OBSERVE_SAMPLES,
    TELEOP_TRIGGER_RELEASE_THRESHOLD,
    TRANSLATION_SCALE,
)
from arms.data_manager import DataManager, RobotActivityState
from arms.logging_setup import get_logger

log = get_logger("teleop")

if TYPE_CHECKING:
    # Type-hint only. Importing at runtime would force meta_quest_teleop onto
    # the follower host, which uses a remote reader instead (--quest-source
    # agora/livekit). With `from __future__ import annotations`, the hints are strings
    # so the real symbol is never needed at runtime.
    from meta_quest_teleop.reader import MetaQuestReader


def _call_first_successful(
    obj: object,
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]],
) -> object | None:
    for name, args, kwargs in calls:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    return None


def _get_grip_value(quest_reader: MetaQuestReader, hand: str) -> float:
    value = _call_first_successful(
        quest_reader,
        [
            ("get_grip_value", (hand,), {}),
            ("get_grip_value", (), {"hand": hand}),
            ("getGripValue", (hand,), {}),
            ("getGripValue", (), {"hand": hand}),
        ],
    )
    try:
        if value is None:
            return 0.0
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) == 0:
                return 0.0
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_frame_seq(quest_reader: MetaQuestReader, hand: str) -> int | None:
    """Return the leader's frame sequence for ``hand``, or None if unsupported.

    Only the remote transports stamp a sequence; a locally-attached Quest is
    read synchronously and has nothing to de-duplicate, so None here means
    "assume every sample is new" and the old behaviour stands.
    """
    value = _call_first_successful(
        quest_reader,
        [
            ("get_seq", (hand,), {}),
            ("get_seq", (), {"hand": hand}),
        ],
    )
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _get_trigger_value(quest_reader: MetaQuestReader, hand: str) -> float:
    value = _call_first_successful(
        quest_reader,
        [
            ("get_trigger_value", (hand,), {}),
            ("get_trigger_value", (), {"hand": hand}),
            ("getTriggerValue", (hand,), {}),
            ("getTriggerValue", (), {"hand": hand}),
        ],
    )
    try:
        if value is None:
            return 0.0
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) == 0:
                return 0.0
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_joystick_y(quest_reader: MetaQuestReader, hand: str) -> float:
    """Get joystick Y axis (forward/backward) from Meta Quest controller."""
    value = _call_first_successful(
        quest_reader,
        [
            ("get_joystick_value", (hand,), {}),
            ("getJoystickValue", (hand,), {}),
            ("get_joystick", (hand,), {}),
        ],
    )
    try:
        if value is None:
            return 0.0
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
            return float(value[1])
        return 0.0
    except (TypeError, ValueError):
        return 0.0


def _get_hand_transform_ros(
    quest_reader: MetaQuestReader, hand: str
) -> np.ndarray | None:
    transform = _call_first_successful(
        quest_reader,
        [
            ("get_hand_controller_transform_ros", (), {"hand": hand}),
            ("get_hand_controller_transform_ros", (hand,), {}),
            ("get_hand_transform_ros", (), {"hand": hand}),
            ("get_hand_transform_ros", (hand,), {}),
        ],
    )
    if transform is None:
        return None
    try:
        arr = np.asarray(transform, dtype=np.float64)
        return arr.copy() if arr.shape == (4, 4) else None
    except Exception:
        return None


def quest_reader_thread(
    data_manager: DataManager,
    quest_reader: MetaQuestReader,
    hands: tuple[str, ...] = ("left", "right"),
    stop_event: threading.Event | None = None,
    error_callback: Callable[[Exception], None] | None = None,
) -> None:
    """Quest reader thread - reads controller data and manages teleop state.

    This thread runs at high frequency to ensure responsive controller input.
    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        data_manager: DataManager object for thread-safe communication
        quest_reader: MetaQuestReader instance
    """
    active_hands = tuple(h for h in hands if h in ("left", "right"))
    if not active_hands:
        raise ValueError("quest_reader_thread needs at least one active hand")

    log.info("quest reader thread started (%s)", ", ".join(active_hands))

    dt: float = 1.0 / CONTROLLER_DATA_RATE
    prev_grip_active_by_hand: dict[str, bool] = {hand: False for hand in active_hands}
    deadman_release_count_by_hand: dict[str, int] = {hand: 0 for hand in active_hands}
    deadman_press_count_by_hand: dict[str, int] = {hand: 0 for hand in active_hands}
    # A hand may only arm teleop after its trigger has been observed genuinely
    # released — i.e. below the release threshold while tracking is VALID. The
    # leader zeroes the trigger whenever tracking drops, so trigger==0 on an
    # invalid frame is ambiguous and does NOT count as a release. This blocks a
    # stuck trigger (reads ~1.0 on every valid frame, 0 only when invalid) from
    # ever arming, while a healthy controller at rest qualifies immediately.
    # Last leader frame seq applied per hand, so a poll that outruns the
    # operator's frame rate does not re-filter a pose the filter has already
    # seen (see DataManager.set_controller_data's pose_is_new).
    last_seq_by_hand: dict[str, int | None] = {hand: None for hand in active_hands}
    armable_by_hand: dict[str, bool] = {hand: False for hand in active_hands}
    armable_low_count_by_hand: dict[str, int] = {hand: 0 for hand in active_hands}
    # A trigger squeezed while the hand is POLICY_CONTROLLED does not arm teleop
    # by itself: it files a take-over request (the policy runner holds both arms
    # and flips them to ENABLED within a tick) and stays pending until ENABLED is
    # seen with the trigger still held -- then teleop arms on that same squeeze.
    # Resuming the policy is an explicit dashboard command, never automatic.
    takeover_pending_by_hand: dict[str, bool] = {hand: False for hand in active_hands}
    release_sample_limit = max(1, int(TELEOP_DEADMAN_RELEASE_CONSECUTIVE_SAMPLES))
    release_observe_limit = max(1, int(TELEOP_DEADMAN_RELEASE_OBSERVE_SAMPLES))
    press_sample_limit = max(1, int(TELEOP_DEADMAN_PRESS_CONSECUTIVE_SAMPLES))
    printed_api_warning: bool = False
    debug_inputs: bool = os.getenv("QUEST_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    last_debug_print_time: float = 0.0
    last_seen_transform_keys: str = ""

    def _activate_teleop(hand: str, controller_transform_filt: np.ndarray) -> None:
        """Anchor teleop for ``hand`` at the current controller / end-effector poses."""
        controller_initial_transform = controller_transform_filt.copy()
        robot_initial_transform = data_manager.get_current_end_effector_pose(hand)
        data_manager.set_teleop_state(
            hand,
            True,
            controller_initial_transform,
            robot_initial_transform,
        )
        robot_pos = (
            robot_initial_transform[:3, 3]
            if robot_initial_transform is not None
            else None
        )
        log.info(
            "control activated (%s) controller_pos=%s robot_pos=%s",
            hand,
            controller_initial_transform[:3, 3],
            robot_pos,
        )

    fresh_input = getattr(quest_reader, "input_available", None)
    input_revision = getattr(quest_reader, "input_revision", lambda: None)
    last_input_revision = input_revision()
    connection_status = getattr(quest_reader, "get_connection_status", None)
    if connection_status is not None:
        data_manager.set_operator_connection_provider(connection_status)
    last_connection_metrics = 0.0

    try:
        while not data_manager.is_shutdown_requested() and not (
            stop_event is not None and stop_event.is_set()
        ):
            iteration_start = time.time()

            controller_transform_raw_by_hand: dict[str, np.ndarray | None] = {}
            grip_value_by_hand: dict[str, float] = {}
            trigger_value_by_hand: dict[str, float] = {}

            revision_before = input_revision()
            pose_is_new_by_hand = {}
            # Get controller data (be tolerant to MetaQuestReader API differences)
            for hand in active_hands:
                try:
                    grip_value_by_hand[hand] = _get_grip_value(quest_reader, hand)
                    trigger_value_by_hand[hand] = _get_trigger_value(quest_reader, hand)
                    controller_transform_raw_by_hand[hand] = _get_hand_transform_ros(
                        quest_reader, hand
                    )
                    # No dead-man gate here on purpose. Requiring a held grip to
                    # jog J6 was added against a frozen thumbstick (left J6 was
                    # driven to its +100 limit by an axis stuck at 0.9228 after a
                    # transport drop), but it made the stick silently dead in
                    # normal use and the requirement lived nowhere near the
                    # ik_solver gate it constrained. Staleness is already handled
                    # where it belongs: the transport's ``input_available`` fails
                    # on a pose older than 250 ms, a lost lease or a changed
                    # connection epoch, and ``get_joystick_value`` returns
                    # neutral then -- so a frozen axis cannot reach this line.
                    joystick_y = _get_joystick_y(quest_reader, hand)
                    data_manager.set_joystick_y(joystick_y, hand)
                except Exception as read_err:
                    if not printed_api_warning:
                        printed_api_warning = True
                        log.warning(
                            "reader polling failed; continuing with defaults: %s",
                            read_err,
                        )
                    grip_value_by_hand[hand] = 0.0
                    trigger_value_by_hand[hand] = 0.0
                    controller_transform_raw_by_hand[hand] = None

                # Update shared state with controller data for this hand.
                # pose_is_new is False when the transport has not delivered a new
                # frame since the last poll: grip/trigger still refresh (the
                # dead-man must never be stale) but the pose filter is left
                # alone rather than being fed a duplicate.
                seq = _get_frame_seq(quest_reader, hand)
                pose_is_new = seq is None or seq != last_seq_by_hand[hand]
                pose_is_new_by_hand[hand] = pose_is_new
                if seq is not None:
                    last_seq_by_hand[hand] = seq
                data_manager.set_controller_data(
                    hand,
                    controller_transform_raw_by_hand[hand],
                    grip_value_by_hand[hand],
                    trigger_value_by_hand[hand],
                    pose_is_new=pose_is_new,
                )

            revision_after = input_revision()
            input_changed = (
                revision_before != revision_after
                or revision_after != last_input_revision
            )
            last_input_revision = revision_after
            # Trigger button logic (dead man's switch) per hand
            for hand in active_hands:
                if fresh_input is not None and (input_changed or not fresh_input(hand)):
                    # Loss is immediate; neutral synthetic input never counts as
                    # a physical release and never cancels Stop/Teach or resumes policy.
                    if data_manager.get_teleop_active(hand):
                        data_manager.set_teleop_state(hand, False, None, None)
                    data_manager.set_joystick_y(0.0, hand)
                    prev_grip_active_by_hand[hand] = False
                    deadman_release_count_by_hand[hand] = 0
                    deadman_press_count_by_hand[hand] = 0
                    armable_by_hand[hand] = False
                    armable_low_count_by_hand[hand] = 0
                    takeover_pending_by_hand[hand] = False
                    continue
                controller_transform_filt, _, _ = data_manager.get_controller_data(hand)

                robot_activity_state = data_manager.get_robot_activity_state(hand)
                prev_grip_active = prev_grip_active_by_hand[hand]
                trigger_value = trigger_value_by_hand[hand]

                # Teleop can only be activated if robot is ENABLED (not HOMING or DISABLED).
                # Use hysteresis + consecutive-sample filtering so brief trigger
                # or input dropouts under recording load do not repeatedly reset
                # the teleop origin. POLICY_CONTROLLED runs the same debounce so a
                # squeeze under policy control is a genuine, tracked press (the
                # take-over request below), not a bounce.
                if robot_activity_state in (
                    RobotActivityState.ENABLED,
                    RobotActivityState.POLICY_CONTROLLED,
                ):
                    if prev_grip_active:
                        if trigger_value >= TELEOP_TRIGGER_RELEASE_THRESHOLD:
                            deadman_release_count_by_hand[hand] = 0
                        else:
                            deadman_release_count_by_hand[hand] += 1
                        grip_active = (
                            deadman_release_count_by_hand[hand] < release_sample_limit
                        )
                    else:
                        # A genuine release is only observable on a VALID frame
                        # (see armable_by_hand note above), and must persist for
                        # several consecutive samples so a flaky/intermittently
                        # stuck trigger that only dips low for a frame or two
                        # can't masquerade as a release. Use the RAW transform
                        # (same tick as the raw trigger) so the leader zeroing the
                        # trigger on a tracking dropout — which arrives with an
                        # empty transform — is never counted as a release.
                        if (
                            controller_transform_raw_by_hand[hand] is not None
                            and trigger_value < TELEOP_TRIGGER_RELEASE_THRESHOLD
                        ):
                            if fresh_input is None or pose_is_new_by_hand[hand]:
                                armable_low_count_by_hand[hand] += 1
                            if armable_low_count_by_hand[hand] >= release_observe_limit:
                                armable_by_hand[hand] = True
                        else:
                            armable_low_count_by_hand[hand] = 0
                        # Press debounce: require the trigger held above threshold
                        # AND valid controller tracking for several consecutive
                        # samples before activating. A single bouncy high-trigger
                        # frame on tracking recovery must not anchor teleop, and
                        # requiring continuous validity means we only capture the
                        # origin once the recovered pose has settled (no lurch).
                        if (
                            armable_by_hand[hand]
                            and trigger_value >= GRIP_THRESHOLD
                            and controller_transform_filt is not None
                        ):
                            if fresh_input is None or pose_is_new_by_hand[hand]:
                                deadman_press_count_by_hand[hand] += 1
                        else:
                            deadman_press_count_by_hand[hand] = 0
                        grip_active = (
                            deadman_press_count_by_hand[hand] >= press_sample_limit
                        )
                        if grip_active:
                            deadman_release_count_by_hand[hand] = 0
                            # Require a fresh observed release before re-arming.
                            armable_by_hand[hand] = False
                            armable_low_count_by_hand[hand] = 0
                else:
                    grip_active = False
                    deadman_release_count_by_hand[hand] = 0
                    deadman_press_count_by_hand[hand] = 0
                    armable_by_hand[hand] = False
                    armable_low_count_by_hand[hand] = 0

                # Rising edge - trigger just pressed AND robot is enabled (or the
                # policy is driving: then it is a take-over request, see above)
                if (
                    grip_active
                    and not prev_grip_active
                    and controller_transform_filt is not None
                ):
                    if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
                        takeover_pending_by_hand[hand] = True
                        data_manager.request_policy_takeover(hand)
                        log.info(
                            "policy take-over requested (%s): trigger squeezed "
                            "under policy control",
                            hand,
                        )
                    else:
                        _activate_teleop(hand, controller_transform_filt)

                # Take-over hand-off: the policy runner has flipped the arms to
                # ENABLED and the operator is still squeezing -- arm teleop now,
                # anchored at the pose the arm was held at.
                elif (
                    grip_active
                    and takeover_pending_by_hand[hand]
                    and robot_activity_state == RobotActivityState.ENABLED
                    and controller_transform_filt is not None
                ):
                    takeover_pending_by_hand[hand] = False
                    _activate_teleop(hand, controller_transform_filt)
                    log.info(
                        "policy take-over complete (%s): teleop armed on the "
                        "same squeeze",
                        hand,
                    )

                # Falling edge - grip just released OR robot disabled
                elif not grip_active and prev_grip_active:
                    if takeover_pending_by_hand[hand]:
                        takeover_pending_by_hand[hand] = False
                        log.info(
                            "policy take-over abandoned (%s): trigger released "
                            "before the hand-off",
                            hand,
                        )
                    data_manager.set_teleop_state(hand, False, None, None)
                    # Require a fresh, debounced hold before the next activation.
                    deadman_press_count_by_hand[hand] = 0
                    release_count = deadman_release_count_by_hand[hand]
                    # raw_valid distinguishes the two ways trigger reaches 0: an
                    # operator genuinely letting go (transform still tracking) vs
                    # the leader zeroing the trigger on a tracking dropout, which
                    # arrives with an empty transform. The re-arm path already
                    # guards on this; the release path does not, so surface it.
                    log.info(
                        "control deactivated (%s) trigger=%.3f grip=%.3f "
                        "low_samples=%s/%s raw_valid=%s robot_state=%s",
                        hand,
                        trigger_value,
                        grip_value_by_hand[hand],
                        release_count,
                        release_sample_limit,
                        controller_transform_raw_by_hand[hand] is not None,
                        robot_activity_state.value,
                    )

                prev_grip_active_by_hand[hand] = grip_active

            if debug_inputs and (iteration_start - last_debug_print_time) >= 1.0:
                last_debug_print_time = iteration_start
                # Attempt to read any non-controller transform as "headset" for debugging.
                headset_pos_str = "None"
                try:
                    transforms, _buttons = (
                        quest_reader.get_transformations_and_buttons()
                    )
                    keys = sorted(list(transforms.keys())) if transforms else []
                    last_seen_transform_keys = ",".join(keys) if keys else ""

                    headset_key = None
                    for candidate in ("h", "head", "H", "Head", "headset", "0"):
                        if transforms and candidate in transforms:
                            headset_key = candidate
                            break
                    if headset_key is None and transforms:
                        for k in keys:
                            if k not in {"r", "l"}:
                                headset_key = k
                                break

                    if headset_key is not None and transforms is not None:
                        t = np.asarray(transforms[headset_key], dtype=np.float64)
                        if t.shape == (4, 4):
                            p = t[:3, 3]
                            headset_pos_str = (
                                f"{headset_key}=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                            )
                        else:
                            headset_pos_str = f"{headset_key}=(bad_shape)"
                except Exception:
                    # Keep debug output robust; headset pose is optional.
                    pass

                def _pos_str(T: np.ndarray | None) -> str:
                    if T is None:
                        return "None"
                    p = T[:3, 3]
                    return f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"

                def _delta_str(hand: str) -> tuple[str, str]:
                    teleop_active = data_manager.get_teleop_active(hand)
                    controller_transform_filt, _, _ = data_manager.get_controller_data(
                        hand
                    )
                    if not teleop_active or controller_transform_filt is None:
                        return ("None", "None")
                    controller_initial, _robot_initial = (
                        data_manager.get_initial_robot_controller_transforms(hand)
                    )
                    if controller_initial is None:
                        return ("None", "None")
                    dp = controller_transform_filt[:3, 3] - controller_initial[:3, 3]
                    delta_pos_str = f"({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f})|norm={float(np.linalg.norm(dp)):.3f}"
                    dR = (
                        controller_transform_filt[:3, :3] @ controller_initial[:3, :3].T
                    )
                    tr = float(np.trace(dR))
                    cos_angle = (tr - 1.0) / 2.0
                    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))
                    delta_rot_str = (
                        f"{angle_deg:.1f}deg|scaled={angle_deg * ROTATION_SCALE:.1f}"
                    )
                    return (delta_pos_str, delta_rot_str)

                def _ee_str(T: np.ndarray | None) -> str:
                    if T is None or T.shape != (4, 4):
                        return "None"
                    p = T[:3, 3]
                    return f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"

                def _joints_str(j: np.ndarray | None) -> str:
                    if j is None:
                        return "None"
                    return "[" + ",".join(f"{x:.1f}" for x in j) + "]"

                hand_parts = []
                for hand in active_hands:
                    raw = controller_transform_raw_by_hand.get(hand)
                    filt, _, _ = data_manager.get_controller_data(hand)
                    dp, dr = _delta_str(hand)
                    state = data_manager.get_robot_activity_state(hand)
                    teleop_active = data_manager.get_teleop_active(hand)
                    current_joint_angles = data_manager.get_current_joint_angles(hand)
                    current_ee = data_manager.get_current_end_effector_pose(hand)
                    hand_parts.append(
                        f"{hand}: robot_state={state.value} "
                        f"grip={grip_value_by_hand[hand]:.3f} "
                        f"trigger={trigger_value_by_hand[hand]:.3f} "
                        f"teleop_active={teleop_active} "
                        f"transform_raw={'ok' if raw is not None else 'None'} "
                        f"transform_filt={'ok' if filt is not None else 'None'} "
                        f"ctrl_pos_raw={_pos_str(raw)} "
                        f"ctrl_pos_filt={_pos_str(filt)} "
                        f"delta_pos={dp} delta_rot={dr} "
                        f"ee_pos={_ee_str(current_ee)} "
                        f"joints_deg={_joints_str(current_joint_angles)}"
                    )
                log.debug(
                    "input | "
                    f"hands={','.join(active_hands)} "
                    f"activate_threshold={GRIP_THRESHOLD} "
                    f"release_threshold={TELEOP_TRIGGER_RELEASE_THRESHOLD} "
                    f"release_low_samples={release_sample_limit} | "
                    f"headset_pos={headset_pos_str} | "
                    f"translation_scale={TRANSLATION_SCALE} "
                    f"rotation_scale={ROTATION_SCALE} | " + " | ".join(hand_parts)
                )
                if last_seen_transform_keys:
                    log.debug("input | transform_keys=%s", last_seen_transform_keys)

            # Sleep to maintain loop rate (check shutdown more frequently)
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        log.error("quest reader thread error: %s", e, exc_info=True)
        if error_callback is not None:
            try:
                error_callback(e)
            except Exception:
                pass
        if stop_event is not None:
            stop_event.set()
        else:
            data_manager.request_shutdown()
    finally:
        # Ensure clean exit - deactivate teleop
        for hand in active_hands:
            data_manager.set_teleop_state(hand, False, None, None)
        log.info("quest reader thread stopped (%s)", ", ".join(active_hands))
