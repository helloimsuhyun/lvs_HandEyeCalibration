from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from scipy.spatial.transform import Rotation


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "rb5_850e" / "collision"


def _as_transform(value: Any, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-7
    ):
        raise ValueError(f"{name} rotation is not in SO(3)")
    return T.copy()


def _mujoco_pose(T_mm: np.ndarray) -> tuple[str, str]:
    T = _as_transform(T_mm, "pose")
    pos = " ".join(f"{value / 1000.0:.12g}" for value in T[:3, 3])
    xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    quat = " ".join(
        f"{value:.12g}" for value in (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    )
    return pos, quat


@dataclass(frozen=True)
class SceneObstacle:
    name: str
    center_mm: np.ndarray
    size_mm: np.ndarray
    rgba: str = "0.55 0.55 0.60 1"

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "SceneObstacle":
        center = np.asarray(data["center_mm"], dtype=float).reshape(3)
        size = np.asarray(data["size_mm"], dtype=float).reshape(3)
        if np.any(size <= 0.0) or not np.all(np.isfinite(center)):
            raise ValueError("obstacle size must be positive and center finite")
        return cls(
            name=str(data.get("name", f"obstacle_{index}")),
            center_mm=center,
            size_mm=size,
            rgba=str(data.get("rgba", "0.55 0.55 0.60 1")),
        )


@dataclass(frozen=True)
class RB5SceneConfig:
    T_tcp_sensor: np.ndarray
    T_base_plane: np.ndarray
    plane_size_mm: np.ndarray = field(
        default_factory=lambda: np.array([400.0, 400.0], dtype=float)
    )
    plane_thickness_mm: float = 10.0
    table_center_mm: np.ndarray = field(
        default_factory=lambda: np.array([550.0, 0.0, 165.0], dtype=float)
    )
    table_size_mm: np.ndarray = field(
        default_factory=lambda: np.array([700.0, 700.0, 330.0], dtype=float)
    )
    sensor_housing_size_mm: np.ndarray = field(
        default_factory=lambda: np.array([120.0, 50.0, 80.0], dtype=float)
    )
    sensor_housing_center_z_mm: float = -45.0
    collision_margin_mm: float = 2.0
    obstacles: tuple[SceneObstacle, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "T_tcp_sensor", _as_transform(self.T_tcp_sensor, "T_tcp_sensor")
        )
        object.__setattr__(
            self, "T_base_plane", _as_transform(self.T_base_plane, "T_base_plane")
        )
        for name in (
            "plane_size_mm",
            "table_center_mm",
            "table_size_mm",
            "sensor_housing_size_mm",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            expected = 2 if name == "plane_size_mm" else 3
            value = value.reshape(expected)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            if name != "table_center_mm" and np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value.copy())
        if self.plane_thickness_mm <= 0.0 or self.collision_margin_mm < 0.0:
            raise ValueError("plane thickness must be positive and margin non-negative")


def build_rb5_keyence_scene_xml(config: RB5SceneConfig) -> str:
    """Build an RB5-850E scene using Rainbow Robotics collision meshes.

    The mathematical calibration plane is the top face of ``target_board``.
    All MuJoCo lengths are metres; the public workflow remains in millimetres.
    """

    if not ASSET_DIR.is_dir():
        raise FileNotFoundError(f"RB5 mesh directory is missing: {ASSET_DIR}")
    mesh_dir = escape(str(ASSET_DIR))
    sensor_pos, sensor_quat = _mujoco_pose(config.T_tcp_sensor)
    plane_pos, plane_quat = _mujoco_pose(config.T_base_plane)
    plane_half = np.array(
        [
            0.5 * config.plane_size_mm[0],
            0.5 * config.plane_size_mm[1],
            0.5 * config.plane_thickness_mm,
        ]
    ) / 1000.0
    table_center = config.table_center_mm / 1000.0
    table_half = 0.5 * config.table_size_mm / 1000.0
    sensor_half = 0.5 * config.sensor_housing_size_mm / 1000.0
    margin = config.collision_margin_mm / 1000.0

    obstacle_xml = []
    for obstacle in config.obstacles:
        pos = " ".join(f"{value / 1000.0:.12g}" for value in obstacle.center_mm)
        size = " ".join(f"{0.5 * value / 1000.0:.12g}" for value in obstacle.size_mm)
        obstacle_xml.append(
            f'<geom name="{escape(obstacle.name)}" class="environment" '
            f'type="box" pos="{pos}" size="{size}" rgba="{escape(obstacle.rgba)}"/>'
        )
    obstacles = "\n    ".join(obstacle_xml)

    return f"""<mujoco model="rb5_850e_keyence_single_plane">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size memory="64M"/>
  <visual>
    <global azimuth="145" elevation="-24" offwidth="1024" offheight="768"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <default>
    <default class="robot">
      <geom contype="1" conaffinity="3" group="1" margin="{margin:.12g}"
            rgba="0.93 0.48 0.08 1" friction="0.8 0.02 0.002"/>
      <joint damping="2" armature="0.01" frictionloss="0.05"/>
    </default>
    <default class="environment">
      <geom contype="2" conaffinity="1" group="2" margin="{margin:.12g}"
            friction="0.9 0.02 0.002"/>
    </default>
  </default>
  <asset>
    <mesh name="rb5_link0" file="link0.stl"/>
    <mesh name="rb5_link1" file="link1.stl"/>
    <mesh name="rb5_link2" file="link2.stl"/>
    <mesh name="rb5_link3" file="link3.stl"/>
    <mesh name="rb5_link4" file="link4.stl"/>
    <mesh name="rb5_link5" file="link5.stl"/>
    <mesh name="rb5_link6" file="link6.stl"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.18 0.22 0.25" rgb2="0.25 0.29 0.32"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="3 3" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor_visual" type="plane" size="2 2 0.05" material="floor_mat"
          contype="0" conaffinity="0" group="3"/>
    <geom name="work_table" class="environment" type="box"
          pos="{' '.join(f'{v:.12g}' for v in table_center)}"
          size="{' '.join(f'{v:.12g}' for v in table_half)}" rgba="0.42 0.45 0.48 1"/>
    <body name="target" pos="{plane_pos}" quat="{plane_quat}">
      <geom name="target_board" class="environment" type="box"
            pos="0 0 {-plane_half[2]:.12g}"
            size="{' '.join(f'{v:.12g}' for v in plane_half)}" rgba="0.82 0.82 0.76 1"/>
      <site name="plane_frame" pos="0 0 0" size="0.012" rgba="0.2 0.8 0.2 1"/>
    </body>
    {obstacles}
    <body name="link0">
      <geom name="link0_geom" class="robot" type="mesh" mesh="rb5_link0" rgba="0.18 0.20 0.22 1"/>
      <body name="link1" pos="0 0 0.1692">
        <inertial pos="5.7e-05 -0.005795 -0.034516" quat="0.676677 0.181125 -0.183239 0.689729" mass="4.147" diaginertia="0.00940157 0.00878691 0.0073332"/>
        <joint name="base" class="robot" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" class="robot" type="mesh" mesh="rb5_link1"/>
        <body name="link2">
          <inertial pos="-3.7e-05 -0.121603 0.212933" quat="0.999997 0.000348233 7.80187e-05 -0.00223104" mass="9.633" diaginertia="0.392214 0.388915 0.0219492"/>
          <joint name="shoulder" class="robot" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_geom" class="robot" type="mesh" mesh="rb5_link2"/>
          <body name="link3" pos="0 0 0.425">
            <inertial pos="2.1e-05 -0.015845 0.202568" quat="0.999948 0.0100341 -5.40187e-05 -0.00170427" mass="3.915" diaginertia="0.11204 0.111832 0.0075271"/>
            <joint name="elbow" class="robot" axis="0 1 0" range="-3.14 3.14"/>
            <geom name="link3_geom" class="robot" type="mesh" mesh="rb5_link3"/>
            <body name="link4" pos="0 0 0.392">
              <inertial pos="5.7e-05 -0.106332 0.025881" quat="0.986693 0.162394 -0.00142514 0.00799378" mass="1.452" diaginertia="0.00162904 0.00160108 0.00118182"/>
              <joint name="wrist1" class="robot" axis="0 1 0" range="-3.14 3.14"/>
              <geom name="link4_geom" class="robot" type="mesh" mesh="rb5_link4"/>
              <body name="link5" pos="0 -0.1107 0.1107">
                <inertial pos="-2.8e-05 -0.025911 -0.004363" quat="0.812497 0.582798 -0.00662653 0.0123056" mass="1.454" diaginertia="0.00163147 0.00160354 0.00118381"/>
                <joint name="wrist2" class="robot" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="link5_geom" class="robot" type="mesh" mesh="rb5_link5"/>
                <body name="link6">
                  <inertial pos="-2e-06 -0.079964 -0.000494" quat="0.706709 -0.0427371 -0.0387873 0.705146" mass="0.243" diaginertia="0.000243707 0.000147897 0.000142261"/>
                  <joint name="wrist3" class="robot" axis="0 1 0" range="-3.14 3.14"/>
                  <geom name="link6_geom" class="robot" type="mesh" mesh="rb5_link6"/>
                  <body name="tcp" pos="0 -0.0967 0">
                    <site name="tcp_site" size="0.009" rgba="0.1 0.4 1 1"/>
                    <body name="keyence_housing" pos="{sensor_pos}" quat="{sensor_quat}">
                      <geom name="keyence_body" class="robot" type="box"
                            pos="0 0 {config.sensor_housing_center_z_mm / 1000.0:.12g}"
                            size="{' '.join(f'{v:.12g}' for v in sensor_half)}"
                            rgba="0.12 0.14 0.16 1"/>
                      <geom name="laser_sheet" type="box" pos="0 0 0.075" size="0.04 0.00035 0.075"
                            contype="0" conaffinity="0" group="4" rgba="1 0.03 0.03 0.22"/>
                      <site name="sensor_site" size="0.007" rgba="1 0.05 0.05 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="link0" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link6" body2="keyence_housing"/>
  </contact>
</mujoco>
"""
