from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import mujoco
import numpy as np


@dataclass(frozen=True)
class Contact:
    geom1: str
    geom2: str
    distance_m: float


@dataclass(frozen=True)
class CollisionReport:
    collision_free: bool
    minimum_contact_distance_m: float | None
    contacts: tuple[Contact, ...]


@dataclass(frozen=True)
class IKResult:
    success: bool
    qpos: np.ndarray
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    reason: str


class HandEyeSimulation:
    """Small planning-facing API around a generated MuJoCo model."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(self.model.njnt)
        )
        self._joint_qpos_addresses = {
            name: int(self.model.jnt_qposadr[index])
            for index, name in enumerate(self.joint_names)
            if name is not None
        }
        self._joint_dof_addresses = {
            name: int(self.model.jnt_dofadr[index])
            for index, name in enumerate(self.joint_names)
            if name is not None
        }

    def reset_home(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, qpos: Iterable[float] | dict[str, float]) -> None:
        if isinstance(qpos, dict):
            unknown = set(qpos) - set(self._joint_qpos_addresses)
            if unknown:
                raise KeyError(f"Unknown joints: {sorted(unknown)}")
            for name, value in qpos.items():
                self.data.qpos[self._joint_qpos_addresses[name]] = float(value)
        else:
            values = np.asarray(tuple(qpos), dtype=float)
            if values.shape != (self.model.nq,):
                raise ValueError(f"Expected {self.model.nq} qpos values, received {values.shape}")
            self.data.qpos[:] = values
        mujoco.mj_forward(self.model, self.data)

    def qpos_with_named_joints(
        self,
        joint_positions: dict[str, float],
        *,
        base_qpos: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Compose a full qpos vector from a portable named-joint mapping."""
        unknown = set(joint_positions) - set(self._joint_qpos_addresses)
        if unknown:
            raise KeyError(f"Unknown joints: {sorted(unknown)}")
        if base_qpos is None:
            result = self.data.qpos.copy()
        else:
            result = np.asarray(tuple(base_qpos), dtype=float).copy()
            if result.shape != (self.model.nq,):
                raise ValueError(f"base_qpos must have shape ({self.model.nq},)")
        for name, value in joint_positions.items():
            result[self._joint_qpos_addresses[name]] = float(value)
        return result

    def named_joint_positions(
        self, qpos: Iterable[float], joint_names: Iterable[str]
    ) -> np.ndarray:
        values = np.asarray(tuple(qpos), dtype=float)
        if values.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},)")
        names = tuple(joint_names)
        unknown = set(names) - set(self._joint_qpos_addresses)
        if unknown:
            raise KeyError(f"Unknown joints: {sorted(unknown)}")
        return np.asarray(
            [values[self._joint_qpos_addresses[name]] for name in names], dtype=float
        )

    def sensor_pose_world(self) -> np.ndarray:
        return self.site_pose_world("sensor_origin")

    def site_pose_world(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise KeyError(f"Generated model has no {site_name!r} site")
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[site_id]
        return transform

    def solve_site_ik(
        self,
        target_world: np.ndarray,
        seed_qpos: Iterable[float],
        *,
        site_name: str = "sensor_origin",
        joint_names: Iterable[str] | None = None,
        max_iterations: int = 300,
        position_tolerance_m: float = 5e-4,
        rotation_tolerance_rad: float = np.deg2rad(0.25),
        damping: float = 1e-3,
        rotation_weight_m: float = 0.20,
        max_step_rad: float = 0.20,
    ) -> IKResult:
        """Numerical damped-least-squares IK for a named MuJoCo site.

        The target translation is in metres, matching MuJoCo.  Supplying the
        robot's measured/safe joint state as ``seed_qpos`` keeps the selected
        IK branch consistent with the path that will actually be executed.
        """
        target = np.asarray(target_world, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_world must be a finite 4x4 transform")
        qpos = np.asarray(tuple(seed_qpos), dtype=float)
        if qpos.shape != (self.model.nq,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"seed_qpos must have shape ({self.model.nq},)")
        if max_iterations < 1 or damping <= 0 or rotation_weight_m <= 0:
            raise ValueError("IK iteration, damping, and rotation weight must be positive")

        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise KeyError(f"Generated model has no {site_name!r} site")

        selected = tuple(self.joint_names if joint_names is None else joint_names)
        unknown = set(selected) - set(self._joint_qpos_addresses)
        if unknown:
            raise KeyError(f"Unknown IK joints: {sorted(unknown)}")
        q_addresses = np.asarray(
            [self._joint_qpos_addresses[name] for name in selected], dtype=int
        )
        dof_addresses = np.asarray(
            [self._joint_dof_addresses[name] for name in selected], dtype=int
        )
        if len(set(q_addresses.tolist())) != len(q_addresses):
            raise ValueError("IK joints must map to distinct scalar qpos values")

        self.data.qpos[:] = qpos
        position_error = float("inf")
        rotation_error = float("inf")
        for iteration in range(1, max_iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            current_position = self.data.site_xpos[site_id].copy()
            current_rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
            position_vector = target[:3, 3] - current_position
            rotation_vector = 0.5 * sum(
                np.cross(current_rotation[:, axis], target[:3, :3][:, axis])
                for axis in range(3)
            )
            position_error = float(np.linalg.norm(position_vector))
            rotation_error = float(np.linalg.norm(rotation_vector))
            if (
                position_error <= position_tolerance_m
                and rotation_error <= rotation_tolerance_rad
            ):
                return IKResult(
                    True,
                    self.data.qpos.copy(),
                    iteration,
                    position_error,
                    rotation_error,
                    "converged",
                )

            jac_position = np.zeros((3, self.model.nv), dtype=float)
            jac_rotation = np.zeros((3, self.model.nv), dtype=float)
            mujoco.mj_jacSite(
                self.model, self.data, jac_position, jac_rotation, site_id
            )
            jacobian = np.vstack(
                [
                    jac_position[:, dof_addresses],
                    rotation_weight_m * jac_rotation[:, dof_addresses],
                ]
            )
            error = np.concatenate(
                [position_vector, rotation_weight_m * rotation_vector]
            )
            normal = jacobian @ jacobian.T + (damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(normal, error)
            delta = np.clip(delta, -max_step_rad, max_step_rad)
            self.data.qpos[q_addresses] += delta

            for joint_name, q_address in zip(selected, q_addresses):
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if bool(self.model.jnt_limited[joint_id]):
                    low, high = self.model.jnt_range[joint_id]
                    self.data.qpos[q_address] = np.clip(
                        self.data.qpos[q_address], low, high
                    )

        return IKResult(
            False,
            self.data.qpos.copy(),
            max_iterations,
            position_error,
            rotation_error,
            "maximum_iterations",
        )

    def collision_report(self) -> CollisionReport:
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            contacts.append(Contact(name1 or str(contact.geom1), name2 or str(contact.geom2), float(contact.dist)))
        minimum = min((contact.distance_m for contact in contacts), default=None)
        # Positive-distance contacts are already inside the configured safety
        # margin, so every generated contact is deliberately unsafe for planning.
        collision_free = not contacts
        return CollisionReport(collision_free, minimum, tuple(contacts))

    def trajectory_is_collision_free(
        self, qpos_trajectory: Iterable[Iterable[float]]
    ) -> tuple[bool, int | None, CollisionReport | None]:
        for index, qpos in enumerate(qpos_trajectory):
            self.set_joint_positions(qpos)
            report = self.collision_report()
            if not report.collision_free:
                return False, index, report
        return True, None, None

    def render_trajectory(
        self,
        qpos_trajectory: Iterable[Iterable[float]],
        *,
        width: int = 640,
        height: int = 480,
        max_frames: int = 18,
    ) -> list[np.ndarray]:
        """Render a compact off-screen preview of a collision-checked path."""
        if width <= 0 or height <= 0 or max_frames <= 0:
            raise ValueError("width, height, and max_frames must be positive")
        trajectory = np.asarray(tuple(qpos_trajectory), dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != self.model.nq:
            raise ValueError(f"trajectory must have shape (N, {self.model.nq})")
        if len(trajectory) == 0:
            raise ValueError("trajectory cannot be empty")
        indices = np.unique(
            np.linspace(0, len(trajectory) - 1, min(max_frames, len(trajectory))).astype(int)
        )
        self.set_joint_positions(trajectory[indices[0]])
        plane_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "estimated_plane_collision",
        )
        scene_positions = self.data.geom_xpos.copy()
        if plane_id >= 0:
            plane_position = self.data.geom_xpos[plane_id]
            low = np.minimum(np.min(scene_positions, axis=0), plane_position)
            high = np.maximum(np.max(scene_positions, axis=0), plane_position)
        else:
            low = np.min(scene_positions, axis=0)
            high = np.max(scene_positions, axis=0)
        extent = float(np.linalg.norm(high - low))
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = 0.5 * (low + high)
        camera.distance = max(1.0, min(2.2, extent * 1.35))
        # This side view keeps the sensor head and both P/S frame triads clear
        # of the RB5 forearm for the configured calibration-board placement.
        camera.azimuth = -45.0
        camera.elevation = -22.0
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        scene_option.geomgroup[:] = 1
        has_robot_visuals = any(
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")
            .startswith("robot_visual_")
            for index in range(self.model.ngeom)
        )
        if has_robot_visuals:
            # Collision meshes/proxies remain active in physics; only omit
            # them from the normal GUI preview.
            scene_option.geomgroup[1] = 0
            scene_option.geomgroup[3] = 0
        frames: list[np.ndarray] = []
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            for index in indices:
                self.set_joint_positions(trajectory[index])
                renderer.update_scene(
                    self.data,
                    camera=camera,
                    scene_option=scene_option,
                )
                frames.append(renderer.render().copy())
        return frames

    def view_trajectory_interactive(
        self,
        qpos_trajectory: Iterable[Iterable[float]],
        *,
        playback_s: float = 2.0,
        max_playback_frames: int = 120,
    ) -> None:
        """Open a native MuJoCo viewer: play once, then stay user-orbitable."""
        import mujoco.viewer

        if max_playback_frames <= 0:
            raise ValueError("max_playback_frames must be positive")

        trajectory = np.asarray(tuple(qpos_trajectory), dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != self.model.nq:
            raise ValueError(f"trajectory must have shape (N, {self.model.nq})")
        if len(trajectory) == 0:
            raise ValueError("trajectory cannot be empty")
        indices = np.unique(
            np.linspace(
                0,
                len(trajectory) - 1,
                min(max_playback_frames, len(trajectory)),
            ).astype(int)
        )
        frame_delay = max(0.01, float(playback_s) / max(1, len(indices)))
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.geomgroup[:] = 1
            if any(
                (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")
                .startswith("robot_visual_")
                for index in range(self.model.ngeom)
            ):
                viewer.opt.geomgroup[1] = 0
                viewer.opt.geomgroup[3] = 0
            for index in indices:
                if not viewer.is_running():
                    return
                self.set_joint_positions(trajectory[index])
                viewer.sync()
                time.sleep(frame_delay)
            self.set_joint_positions(trajectory[-1])
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
