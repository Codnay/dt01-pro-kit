#!/usr/bin/env python3
"""VR teleoperation of the two arms.

    python -m arms.teleop --robot-process

The operator's Quest browser sends controller poses over LiveKit; this process
turns them into joint commands and streams those to the arms over CAN. It opens
no camera: live video is a separate process (``python -m arms.cameras``).

Stop it with Ctrl+C (SIGINT) and wait. That is the only shutdown path that
parks the arms cleanly; killing the process can leave an arm latched in an
emergency stop that needs a power cycle to clear.
"""

from __future__ import annotations

import argparse
import faulthandler
import signal

from arms.configs import (
    CONTROLLER_DATA_RATE,
    IK_SOLVER_RATE,
    JOINT_STATE_STREAMING_RATE,
    ROBOT_RATE,
)
from arms.livekit_rt.livekit_quest_reader import add_livekit_args
from arms.logging_setup import (
    configure_logging,
    get_logger,
    install_third_party_bridge,
)
from arms.perf.cv_threads import limit_opencv_threads
from arms.perf.gil_tuning import tune_gil_switch_interval
from arms.robot_identity import add_robot_id_arg, resolve_robot_id

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Build the teleop command line."""
    parser = argparse.ArgumentParser(description="VR teleoperation of the two arms")
    parser.add_argument(
        "--arm",
        choices=("both", "right", "left"),
        default="both",
        help="Arm(s) to run (default: both).",
    )
    parser.add_argument(
        "--can-right",
        default="can_right",
        help="CAN interface of the right arm (default: can_right).",
    )
    parser.add_argument(
        "--can-left",
        default="can_left",
        help="CAN interface of the left arm (default: can_left).",
    )
    parser.add_argument(
        "--robot-process",
        action="store_true",
        help="Run each arm's controller in its own process. Recommended on the "
        "robot computer: it keeps CAN decoding from starving the control loops, "
        "which is felt as jitter.",
    )
    parser.add_argument(
        "--visualizer",
        action="store_true",
        help="Also serve a 3D view of the arms (needs: pip install viser "
        "yourdfpy trimesh). Off by default.",
    )
    add_robot_id_arg(parser)
    add_livekit_args(parser)

    logs = parser.add_argument_group("logging")
    logs.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default=None,
        help="Console verbosity (default: info, or $PIPER_LOG_LEVEL).",
    )
    logs.add_argument(
        "--log-only",
        default=None,
        metavar="C1,C2",
        help="Show only these components, e.g. 'teleop,livekit-rtm,can_right'.",
    )
    logs.add_argument(
        "--log-mute",
        default=None,
        metavar="C1,C2",
        help="Hide these components (applied after --log-only).",
    )
    logs.add_argument(
        "--log-components",
        default=None,
        metavar="C=LEVEL,...",
        help="Per-component levels, e.g. 'livekit-rtm=debug,joint=warning'.",
    )
    logs.add_argument("--no-log-color", action="store_true", help="No ANSI colors.")
    logs.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Also append plain log lines to this file.",
    )
    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    configure_logging(
        level=args.log_level,
        only=args.log_only,
        mute=args.log_mute,
        component_levels=args.log_components,
        color=False if args.no_log_color else None,
        log_file=args.log_file,
        force=True,
    )
    install_third_party_bridge()


def _active_hands(arm: str) -> tuple[str, ...]:
    return ("right", "left") if arm == "both" else (arm,)


def _log_startup(args: argparse.Namespace, active_hands: tuple[str, ...]) -> None:
    cans = ", ".join(
        args.can_right if hand == "right" else args.can_left for hand in active_hands
    )
    log.info(
        "config: robot_id=%s  arms=%s (%s)  controllers=%s  visualizer=%s",
        resolve_robot_id(args),
        ",".join(active_hands),
        cans,
        "subprocess" if args.robot_process else "in-process",
        "on" if args.visualizer else "off",
    )
    log.info(
        "rates: quest=%gHz ik=%gHz control=%gHz joint=%gHz",
        CONTROLLER_DATA_RATE,
        IK_SOLVER_RATE,
        ROBOT_RATE,
        JOINT_STATE_STREAMING_RATE,
    )


def run(args: argparse.Namespace) -> int:
    """Run teleop until Ctrl+C, then park the arms."""
    # Imported here so --help works without the robot libraries installed.
    from arms.bimanual_teleop_runtime import (
        BimanualTeleopRuntime,
        BimanualTeleopRuntimeConfig,
    )
    from arms.operator_services import OperatorServices, OperatorServicesConfig

    active_hands = _active_hands(args.arm)
    _log_startup(args, active_hands)

    services = OperatorServices(OperatorServicesConfig(active_hands=active_hands))
    runtime: BimanualTeleopRuntime | None = None
    try:
        services.start()
        runtime = BimanualTeleopRuntime(
            data_manager=services.data_manager,
            config=BimanualTeleopRuntimeConfig(
                active_hands=active_hands,
                can_right=args.can_right,
                can_left=args.can_left,
                robot_process=args.robot_process,
                quest_source="livekit",
                livekit_url=args.livekit_url,
                livekit_room=args.livekit_room,
                livekit_identity=args.livekit_identity,
                livekit_connect_timeout=args.livekit_connect_timeout,
                livekit_stats_sec=args.livekit_stats_sec,
                livekit_input_timeout=args.livekit_input_timeout,
                enable_visualizer=args.visualizer,
            ),
            services=services,
        )
        try:
            runtime.start()
        except Exception as exc:
            log.error("teleop failed to start: %s", exc, exc_info=True)
            runtime.stop()
            runtime = None
            raise

        try:
            runtime.run_visualizer_loop()
        except KeyboardInterrupt:
            log.info("interrupt received — stopping teleop...")
        return 0
    except KeyboardInterrupt:
        log.info("interrupt received — shutting down...")
        return 0
    finally:
        if runtime is not None:
            runtime.stop()
        services.stop()


def main() -> int:
    """Parse the command line, tune the process, run teleop."""
    # A native crash should name the thread it happened on.
    faulthandler.enable()
    # Ctrl+C is how this process is told to park the arms. A process started in
    # the background by a script inherits "ignore SIGINT" and would never see
    # it, so put the normal handler back.
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    args = build_parser().parse_args()
    _configure_logging(args)

    # Both must happen before any thread starts. OpenCV's thread pool spin-waits
    # for no gain here, and CPython's default 5 ms GIL handoff alone can stall a
    # 10 ms control tick.
    tuning = []
    cv_threads = limit_opencv_threads()
    if cv_threads is not None:
        tuning.append(f"opencv_threads={cv_threads}")
    gil_interval = tune_gil_switch_interval()
    if gil_interval is not None:
        tuning.append(f"gil_switch={gil_interval * 1000.0:.2f}ms")
    if tuning:
        log.info("host tuning: %s", " ".join(tuning))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
