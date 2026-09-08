#!/usr/bin/env python3
"""Generate the RB5 + Keyence URDF used by MoveIt.

The source hand-eye JSON defines ``^TCP T_S`` and ``^S T_P``.  MoveIt plans
pose goals for link P while laser data and calibration remain expressed at S.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
SOURCE_URDF = ROOT / "mujoco_handeye_sim/assets/rb5_850e/rbpodo_description/robots/rb5_850e.urdf"
SENSOR_CONFIG = ROOT / "mujoco_handeye_sim/configs/rb5_ljv7080.yaml"
HANDEYE = ROOT / "real_laser_handeye/initial_T_tcp_sensor.json"
OUTPUT = Path(__file__).resolve().parents[1] / "config/rb5_laser.urdf"


def fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def origin(parent: ET.Element, transform: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    ET.SubElement(
        parent,
        "origin",
        xyz=fmt(transform[:3, 3]),
        rpy=fmt(rpy),
    )


def fixed_joint(root: ET.Element, name: str, parent: str, child: str, transform: np.ndarray) -> None:
    joint = ET.SubElement(root, "joint", name=name, type="fixed")
    origin(joint, transform)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)


def collision_geometry(link: ET.Element, name: str, transform: np.ndarray, kind: str, size) -> None:
    collision = ET.SubElement(link, "collision", name=name)
    origin(collision, transform)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, kind, **size)


def segment_transform(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
    delta = second - first
    length = float(np.linalg.norm(delta))
    direction = delta / length
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(reference @ direction)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, direction)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, direction])
    transform[:3, 3] = (first + second) * 0.5
    return transform, length


def main() -> None:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    # This is a planning-only model. Loading the vendor ros2_control plugin is
    # unnecessary and would couple move_group startup to the real controller.
    for control in root.findall("ros2_control"):
        root.remove(control)
    handeye = json.loads(HANDEYE.read_text(encoding="utf-8"))
    T_tcp_s = np.asarray(handeye["T_tcp_sensor"], dtype=float)
    T_s_p = np.asarray(handeye["T_sensor_physical"], dtype=float)
    for name, transform in (("T_tcp_sensor", T_tcp_s), ("T_sensor_physical", T_s_p)):
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"{name} must be a finite 4x4 matrix in {HANDEYE}")
    T_tcp_s = T_tcp_s.copy(); T_tcp_s[:3, 3] *= 1e-3
    T_s_p = T_s_p.copy(); T_s_p[:3, 3] *= 1e-3

    settings = yaml.safe_load(SENSOR_CONFIG.read_text(encoding="utf-8"))["sensor"]
    collision_settings = settings["collision"]
    padding = float(collision_settings.get("padding_m", 0.0))
    sensor_link = ET.SubElement(root, "link", name="sensor_measurement_frame")

    for box in collision_settings.get("boxes", []):
        transform = np.eye(4)
        transform[:3, 3] = np.asarray(box["center_mm"], dtype=float) * 1e-3
        half = np.asarray(box["half_size_mm"], dtype=float) * 1e-3 + padding
        collision_geometry(
            sensor_link,
            str(box["name"]),
            transform,
            "box",
            {"size": fmt(2.0 * half)},
        )

    for capsule in collision_settings.get("capsules", []):
        first = np.asarray(capsule["from_mm"], dtype=float) * 1e-3
        second = np.asarray(capsule["to_mm"], dtype=float) * 1e-3
        radius = float(capsule["radius_mm"]) * 1e-3 + padding
        transform, length = segment_transform(first, second)
        collision_geometry(
            sensor_link,
            str(capsule["name"]),
            transform,
            "cylinder",
            {"radius": f"{radius:.12g}", "length": f"{length:.12g}"},
        )

    ET.SubElement(root, "link", name="sensor_physical_origin")
    fixed_joint(root, "tcp_to_sensor_measurement", "tcp", "sensor_measurement_frame", T_tcp_s)
    fixed_joint(root, "sensor_measurement_to_physical", "sensor_measurement_frame", "sensor_physical_origin", T_s_p)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
