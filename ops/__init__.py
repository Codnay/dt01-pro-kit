"""Ops dashboard: start/stop the teleop stack and the camera recorder remotely.

Standalone by design -- stdlib only, no robot-stack imports -- so it keeps
working (and keeps its stop button) when the teleop venv or ``piper_sdk`` is
broken. See ops/CLAUDE.md.
"""
