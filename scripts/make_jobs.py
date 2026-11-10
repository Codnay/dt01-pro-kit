#!/usr/bin/env python3
"""Write ops/jobs.json for this machine from ops/jobs.example.json.

    make_jobs.py <kit_dir> [serial ...]

Sets ``project_dir`` to where the kit actually lives. If camera serials are
given and they differ from the ones in the template, the cameras job is
rewritten to use them (named camera-1, camera-2, ...). An existing
ops/jobs.json is left alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def camera_args(serials: list[str]) -> list[str]:
    """Return the --camera arguments for these serials."""
    args: list[str] = []
    for index, serial in enumerate(serials, start=1):
        args += ["--camera", f"camera-{index}:{serial}"]
    return args


def template_serials(argv: list[str]) -> set[str]:
    """Return the camera serials named in a cameras job command line."""
    specs = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--camera"]
    return {spec.split(":")[1] for spec in specs if ":" in spec}


def strip_camera_args(argv: list[str]) -> list[str]:
    """Return the command line without its --camera pairs."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--camera":
            skip = True
        else:
            out.append(arg)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    kit = Path(sys.argv[1]).resolve()
    serials = sorted(set(sys.argv[2:]))
    target = kit / "ops" / "jobs.json"
    if target.exists():
        print(f"keeping existing {target}")
        return 0

    jobs = json.loads((kit / "ops" / "jobs.example.json").read_text(encoding="utf-8"))
    jobs.pop("_comment", None)
    jobs["project_dir"] = str(kit)
    cameras = next(job for job in jobs["jobs"] if job["id"] == "cameras")
    if serials and not set(serials) <= template_serials(cameras["argv"]):
        cameras["argv"] = strip_camera_args(cameras["argv"]) + camera_args(serials)
        print(f"cameras job set to {len(serials)} detected camera(s)")
    target.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
