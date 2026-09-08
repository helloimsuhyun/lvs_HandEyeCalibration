#!/usr/bin/env python3
"""Flatten the official UR5e xacro for the MuJoCo visual preview.

Run this after sourcing ROS 2. The script intentionally uses the installed
`ur_description` package instead of vendoring Universal Robots meshes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "assets" / "ur5e" / "ur5e_robot.urdf.xacro"
    output = root / "build" / "ur5e_ljv7080_visual" / "ur5e_robot.urdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    xacro = shutil.which("xacro")
    if not xacro:
        raise SystemExit(
            "xacro not found. Run: source /opt/ros/humble/setup.bash "
            "and install the UR ROS 2 description package."
        )

    try:
        subprocess.run(
            ["ros2", "pkg", "prefix", "ur_description"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "ur_description is not visible. Source ROS and install ros-humble-ur."
        ) from error

    with output.open("wb") as stream:
        subprocess.run([xacro, str(source)], check=True, stdout=stream)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
