from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .compat.hardware import LaserInterface, ProfileSample, RobotInterface

from .scene import RB5SceneConfig, SceneObstacle, build_rb5_keyence_scene_xml


JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "MuJoCo is not installed. Run: python3 -m pip install 'mujoco>=3.2,<4'"
        ) from exc
    return mujoco


def _transform_from_pose(center_mm: Any, rpy_deg: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(center_mm, dtype=float).reshape(3)
    T[:3, :3] = Rotation.from_euler(
        "xyz", np.asarray(rpy_deg, dtype=float).reshape(3), degrees=True
    ).as_matrix()
    return T


def _pose_key(T: np.ndarray) -> bytes:
    return np.round(np.asarray(T, dtype=float).reshape(4, 4), decimals=8).tobytes()


class MujocoRB5Robot(RobotInterface):
    """RB5-850E Cartesian adapter backed by MuJoCo IK and mesh collision checks.

    The adapter intentionally exposes the same millimetre-based TCP contract as
    the physical workflow. MuJoCo itself is kept in metres and radians.
    """

    def __init__(
        self,
        *,
        T_tcp_sensor: list[list[float]],
        plane_center_mm: list[float] = (550.0, 0.0, 350.0),
        plane_rpy_deg: list[float] = (0.0, 0.0, 0.0),
        plane_size_mm: list[float] = (400.0, 400.0),
        plane_thickness_mm: float = 10.0,
        table_center_mm: list[float] = (550.0, 0.0, 165.0),
        table_size_mm: list[float] = (700.0, 700.0, 330.0),
        sensor_housing_size_mm: list[float] = (120.0, 50.0, 80.0),
        sensor_housing_center_z_mm: float = -45.0,
        collision_margin_mm: float = 2.0,
        obstacles: list[dict[str, Any]] | None = None,
        initial_joint_positions_deg: list[float] = (0.0, -50.0, 100.0, -50.0, 90.0, 0.0),
        ik_position_tolerance_mm: float = 0.35,
        ik_rotation_tolerance_deg: float = 0.15,
        ik_max_iterations: int = 180,
        ik_damping: float = 1e-3,
        joint_collision_step_deg: float = 2.0,
        viewer: bool = False,
        realtime_scale: float = 0.0,
    ) -> None:
        self.mujoco = _import_mujoco()
        self.T_tcp_sensor_true = np.asarray(T_tcp_sensor, dtype=float).reshape(4, 4)
        self.T_base_plane = _transform_from_pose(plane_center_mm, plane_rpy_deg)
        obstacle_items = tuple(
            SceneObstacle.from_dict(item, index)
            for index, item in enumerate(obstacles or [])
        )
        self.scene_config = RB5SceneConfig(
            T_tcp_sensor=self.T_tcp_sensor_true,
            T_base_plane=self.T_base_plane,
            plane_size_mm=np.asarray(plane_size_mm, dtype=float),
            plane_thickness_mm=float(plane_thickness_mm),
            table_center_mm=np.asarray(table_center_mm, dtype=float),
            table_size_mm=np.asarray(table_size_mm, dtype=float),
            sensor_housing_size_mm=np.asarray(sensor_housing_size_mm, dtype=float),
            sensor_housing_center_z_mm=float(sensor_housing_center_z_mm),
            collision_margin_mm=float(collision_margin_mm),
            obstacles=obstacle_items,
        )
        xml = build_rb5_keyence_scene_xml(self.scene_config)
        self.model = self.mujoco.MjModel.from_xml_string(xml)
        self.data = self.mujoco.MjData(self.model)
        self._joint_ids = np.array(
            [
                self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in JOINT_NAMES
            ],
            dtype=int,
        )
        if np.any(self._joint_ids < 0):
            raise RuntimeError("RB5 model is missing one or more arm joints")
        self._qpos_indices = self.model.jnt_qposadr[self._joint_ids].astype(int)
        self._dof_indices = self.model.jnt_dofadr[self._joint_ids].astype(int)
        self._tcp_site_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, "tcp_site"
        )
        self._joint_lower = self.model.jnt_range[self._joint_ids, 0].copy()
        self._joint_upper = self.model.jnt_range[self._joint_ids, 1].copy()
        self._initial_q = np.radians(
            np.asarray(initial_joint_positions_deg, dtype=float).reshape(6)
        )
        if np.any(self._initial_q <= self._joint_lower) or np.any(
            self._initial_q >= self._joint_upper
        ):
            raise ValueError("initial RB5 joints must lie strictly inside their limits")
        if ik_position_tolerance_mm <= 0.0 or ik_rotation_tolerance_deg <= 0.0:
            raise ValueError("IK tolerances must be positive")
        if ik_max_iterations <= 0 or ik_damping <= 0.0 or joint_collision_step_deg <= 0.0:
            raise ValueError("IK iteration/damping and collision step must be positive")
        self.ik_position_tolerance_m = float(ik_position_tolerance_mm) / 1000.0
        self.ik_rotation_tolerance_rad = float(np.radians(ik_rotation_tolerance_deg))
        self.ik_max_iterations = int(ik_max_iterations)
        self.ik_damping = float(ik_damping)
        self.joint_collision_step_rad = float(np.radians(joint_collision_step_deg))
        self.viewer_enabled = bool(viewer)
        self.realtime_scale = float(realtime_scale)
        if self.realtime_scale < 0.0:
            raise ValueError("realtime_scale must be non-negative")
        self._viewer = None
        self._connected = False
        self._stop_requested = threading.Event()
        self._lock = threading.RLock()
        self._preflight_cache: dict[bytes, list[np.ndarray]] = {}
        self.last_path_error: str | None = None
        self.motion_log: list[dict[str, Any]] = []
        self._motion_frame_callback: Callable[["MujocoRB5Robot"], None] | None = None
        self.reset()

    @property
    def joint_positions_rad(self) -> np.ndarray:
        return self.data.qpos[self._qpos_indices].copy()

    def reset(self) -> None:
        with self._lock:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qpos[self._qpos_indices] = self._initial_q
            self.mujoco.mj_forward(self.model, self.data)
            self._stop_requested.clear()
            self._preflight_cache.clear()
            self.motion_log.clear()

    def connect(self) -> None:
        with self._lock:
            self._connected = True
            self._stop_requested.clear()
            contacts = self._collision_contacts(self.data)
            if contacts:
                self._connected = False
                raise RuntimeError(
                    "initial RB5 configuration is in collision: "
                    + "; ".join(contacts[:5])
                )
            if self.viewer_enabled and self._viewer is None:
                try:
                    from mujoco import viewer as mj_viewer

                    self._viewer = mj_viewer.launch_passive(self.model, self.data)
                except Exception as exc:
                    self._connected = False
                    raise RuntimeError(
                        "failed to open MuJoCo viewer; use viewer=false on a headless host"
                    ) from exc

    def close(self) -> None:
        with self._lock:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            self._connected = False

    def supports_independent_stop(self) -> bool:
        return True

    def stop(self) -> None:
        self._stop_requested.set()

    def _site_transform_mm(self, data: Any) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = np.asarray(data.site_xmat[self._tcp_site_id]).reshape(3, 3)
        T[:3, 3] = np.asarray(data.site_xpos[self._tcp_site_id]) * 1000.0
        return T

    def current_T_base_tcp(self) -> np.ndarray:
        with self._lock:
            self.mujoco.mj_forward(self.model, self.data)
            return self._site_transform_mm(self.data)

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        scratch = self.mujoco.MjData(self.model)
        scratch.qpos[self._qpos_indices] = np.asarray(
            joint_positions_rad, dtype=float
        ).reshape(6)
        self.mujoco.mj_forward(self.model, scratch)
        return self._site_transform_mm(scratch)

    @staticmethod
    def _pose_error(target_mm: np.ndarray, current_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position_m = (target_mm[:3, 3] - current_mm[:3, 3]) / 1000.0
        rotation = Rotation.from_matrix(
            target_mm[:3, :3] @ current_mm[:3, :3].T
        ).as_rotvec()
        return position_m, rotation

    def _solve_ik_once(self, target_mm: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        data = self.mujoco.MjData(self.model)
        q = np.asarray(seed, dtype=float).reshape(6).copy()
        lower = self._joint_lower + 1e-5
        upper = self._joint_upper - 1e-5
        q = np.clip(q, lower, upper)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        best = (float("inf"), q.copy(), float("inf"), float("inf"))
        damping = self.ik_damping

        for _ in range(self.ik_max_iterations):
            data.qpos[self._qpos_indices] = q
            self.mujoco.mj_forward(self.model, data)
            current = self._site_transform_mm(data)
            position_error, rotation_error = self._pose_error(target_mm, current)
            pos_norm = float(np.linalg.norm(position_error))
            rot_norm = float(np.linalg.norm(rotation_error))
            objective = pos_norm * pos_norm + (0.28 * rot_norm) ** 2
            if objective < best[0]:
                best = (objective, q.copy(), pos_norm, rot_norm)
            if (
                pos_norm <= self.ik_position_tolerance_m
                and rot_norm <= self.ik_rotation_tolerance_rad
            ):
                return q.copy(), pos_norm, rot_norm

            self.mujoco.mj_jacSite(
                self.model, data, jacp, jacr, self._tcp_site_id
            )
            J = np.vstack(
                [jacp[:, self._dof_indices], 0.28 * jacr[:, self._dof_indices]]
            )
            error = np.concatenate([position_error, 0.28 * rotation_error])
            lhs = J.T @ J + (damping * damping) * np.eye(6)
            try:
                delta = np.linalg.solve(lhs, J.T @ error)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, J.T @ error, rcond=None)[0]
            max_step = float(np.max(np.abs(delta)))
            if max_step > 0.18:
                delta *= 0.18 / max_step

            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = np.clip(q + scale * delta, lower, upper)
                data.qpos[self._qpos_indices] = candidate
                self.mujoco.mj_forward(self.model, data)
                candidate_pose = self._site_transform_mm(data)
                dp, dr = self._pose_error(target_mm, candidate_pose)
                candidate_objective = float(dp @ dp + (0.28 * np.linalg.norm(dr)) ** 2)
                if candidate_objective < objective:
                    q = candidate
                    damping = max(self.ik_damping, 0.7 * damping)
                    accepted = True
                    break
            if not accepted:
                damping = min(0.2, 4.0 * damping)
                if damping >= 0.2:
                    break

        _, best_q, best_pos, best_rot = best
        return best_q, best_pos, best_rot

    def _refine_ik_least_squares(
        self, target_mm: np.ndarray, seed: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Polish a DLS solution near singular configurations."""

        data = self.mujoco.MjData(self.model)

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[self._qpos_indices] = q
            self.mujoco.mj_forward(self.model, data)
            current = self._site_transform_mm(data)
            position_error, rotation_error = self._pose_error(target_mm, current)
            return np.concatenate([position_error, 0.28 * rotation_error])

        result = least_squares(
            residual,
            np.clip(seed, self._joint_lower + 1e-6, self._joint_upper - 1e-6),
            bounds=(self._joint_lower + 1e-6, self._joint_upper - 1e-6),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=250,
            x_scale="jac",
        )
        q = result.x
        data.qpos[self._qpos_indices] = q
        self.mujoco.mj_forward(self.model, data)
        position_error, rotation_error = self._pose_error(
            target_mm, self._site_transform_mm(data)
        )
        return q, float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))

    def solve_ik(self, T_base_tcp: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        target = np.asarray(T_base_tcp, dtype=float).reshape(4, 4)
        primary = self.joint_positions_rad if seed is None else np.asarray(seed, dtype=float).reshape(6)
        seeds = [primary]
        # Alternative shoulder/elbow seeds help when the current branch is near
        # a wrist singularity, while the primary seed preserves path continuity.
        for offset_deg in (
            (0, -35, 70, -35, 0, 0),
            (0, 35, -70, 35, 0, 0),
            (90, 0, 0, 0, 0, 0),
            (-90, 0, 0, 0, 0, 0),
        ):
            seeds.append(
                np.clip(
                    primary + np.radians(offset_deg),
                    self._joint_lower + 1e-5,
                    self._joint_upper - 1e-5,
                )
            )
        base_hint = float(np.arctan2(target[1, 3], target[0, 3]))
        for base_angle in (base_hint, base_hint + np.pi, base_hint - np.pi):
            for shoulder, elbow, wrist1, wrist2 in (
                (-90, 90, 0, 90),
                (-45, 90, -45, 90),
                (0, 90, -90, 90),
                (-100, 135, -35, -90),
                (35, -90, 55, 90),
            ):
                seeds.append(
                    np.clip(
                        np.array(
                            [
                                base_angle,
                                np.radians(shoulder),
                                np.radians(elbow),
                                np.radians(wrist1),
                                np.radians(wrist2),
                                primary[5],
                            ]
                        ),
                        self._joint_lower + 1e-5,
                        self._joint_upper - 1e-5,
                    )
                )
        candidates: list[tuple[np.ndarray, float, float]] = []
        for item in seeds:
            candidate = self._solve_ik_once(target, item)
            candidates.append(candidate)
            if (
                candidate[1] <= self.ik_position_tolerance_m
                and candidate[2] <= self.ik_rotation_tolerance_rad
            ):
                q, pos_error, rot_error = candidate
                break
        else:
            polished = [
                self._refine_ik_least_squares(target, item[0])
                for item in sorted(
                    candidates,
                    key=lambda candidate: candidate[1] ** 2 + (0.28 * candidate[2]) ** 2,
                )[:3]
            ]
            q, pos_error, rot_error = min(
                [*candidates, *polished],
                key=lambda item: item[1] ** 2 + (0.28 * item[2]) ** 2,
            )
        if pos_error > self.ik_position_tolerance_m or rot_error > self.ik_rotation_tolerance_rad:
            raise RuntimeError(
                "RB5 IK failed: position error "
                f"{1000.0 * pos_error:.3f} mm, rotation error "
                f"{np.degrees(rot_error):.3f} deg"
            )
        return q

    def _collision_contacts(self, data: Any) -> list[str]:
        contacts: list[str] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1 = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            )
            geom2 = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            )
            contacts.append(
                f"{geom1}<->{geom2} distance={1000.0 * float(contact.dist):.3f} mm"
            )
        return contacts

    def collision_contacts(self, joint_positions_rad: np.ndarray | None = None) -> list[str]:
        data = self.mujoco.MjData(self.model)
        q = self.joint_positions_rad if joint_positions_rad is None else np.asarray(
            joint_positions_rad, dtype=float
        ).reshape(6)
        data.qpos[self._qpos_indices] = q
        self.mujoco.mj_forward(self.model, data)
        return self._collision_contacts(data)

    def _assert_joint_path_collision_free(self, q_start: np.ndarray, q_end: np.ndarray) -> None:
        delta = np.asarray(q_end) - np.asarray(q_start)
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / self.joint_collision_step_rad)))
        scratch = self.mujoco.MjData(self.model)
        for index, fraction in enumerate(np.linspace(0.0, 1.0, count + 1)):
            q = (1.0 - fraction) * q_start + fraction * q_end
            scratch.qpos[self._qpos_indices] = q
            self.mujoco.mj_forward(self.model, scratch)
            contacts = self._collision_contacts(scratch)
            if contacts:
                raise RuntimeError(
                    f"collision at joint interpolation {index}/{count}: "
                    + "; ".join(contacts[:5])
                )

    def controller_path_is_safe(self, poses: list[np.ndarray]) -> bool:
        with self._lock:
            self.last_path_error = None
            self._preflight_cache.clear()
            if not poses:
                self.last_path_error = "empty Cartesian path"
                return False
            q_previous = self.joint_positions_rad
            try:
                for pose_index, pose in enumerate(poses):
                    target = np.asarray(pose, dtype=float).reshape(4, 4)
                    if pose_index == 0 and np.allclose(
                        target, self.current_T_base_tcp(), atol=1e-6, rtol=0.0
                    ):
                        q_target = q_previous.copy()
                    else:
                        q_target = self.solve_ik(target, seed=q_previous)
                    self._assert_joint_path_collision_free(q_previous, q_target)
                    self._preflight_cache.setdefault(_pose_key(target), []).append(
                        q_target.copy()
                    )
                    q_previous = q_target
            except Exception as exc:
                self.last_path_error = str(exc)
                self._preflight_cache.clear()
                return False
            return True

    def _cached_or_solved_target(self, target: np.ndarray) -> np.ndarray:
        key = _pose_key(target)
        values = self._preflight_cache.get(key)
        if values:
            result = values.pop(0)
            if not values:
                self._preflight_cache.pop(key, None)
            return result
        return self.solve_ik(target)

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def set_motion_frame_callback(
        self, callback: Callable[["MujocoRB5Robot"], None] | None
    ) -> None:
        """Attach an optional off-screen GUI renderer to completed waypoints."""

        self._motion_frame_callback = callback

    def move_tcp(
        self,
        T_base_tcp: np.ndarray,
        *,
        linear_speed_mm_s: float,
        angular_speed_deg_s: float,
        timeout_s: float,
    ) -> None:
        del angular_speed_deg_s
        target = np.asarray(T_base_tcp, dtype=float).reshape(4, 4)
        start_time = time.monotonic()
        with self._lock:
            if self._stop_requested.is_set():
                raise RuntimeError("MuJoCo RB5 motion was stopped")
            q_start = self.joint_positions_rad
            q_target = self._cached_or_solved_target(target)
            self._assert_joint_path_collision_free(q_start, q_target)
            delta = q_target - q_start
            count = max(
                1,
                int(np.ceil(np.max(np.abs(delta)) / self.joint_collision_step_rad)),
            )
            start_pose = self.current_T_base_tcp()
            distance_mm = float(np.linalg.norm(target[:3, 3] - start_pose[:3, 3]))
            nominal_duration = distance_mm / max(float(linear_speed_mm_s), 1e-9)
            for fraction in np.linspace(1.0 / count, 1.0, count):
                if self._stop_requested.is_set():
                    raise RuntimeError("MuJoCo RB5 motion was stopped")
                if time.monotonic() - start_time > float(timeout_s):
                    raise TimeoutError("MuJoCo RB5 Cartesian motion timed out")
                self.data.qpos[self._qpos_indices] = q_start + fraction * delta
                self.data.qvel[:] = 0.0
                self.mujoco.mj_forward(self.model, self.data)
                self._sync_viewer()
                if self.realtime_scale > 0.0 and nominal_duration > 0.0:
                    time.sleep(self.realtime_scale * nominal_duration / count)
            reached = self.current_T_base_tcp()
            dp, dr = self._pose_error(target, reached)
            self.motion_log.append(
                {
                    "T_base_tcp": reached.tolist(),
                    "joint_positions_deg": np.degrees(q_target).tolist(),
                    "position_error_mm": 1000.0 * float(np.linalg.norm(dp)),
                    "rotation_error_deg": float(np.degrees(np.linalg.norm(dr))),
                }
            )
            if self._motion_frame_callback is not None:
                self._motion_frame_callback(self)


class MujocoKeyenceLaser(LaserInterface):
    """Analytic Keyence LJ-X-style profile source attached to the MuJoCo RB5."""

    def __init__(
        self,
        *,
        robot: RobotInterface,
        x_min_mm: float = -40.0,
        x_max_mm: float = 40.0,
        z_min_mm: float = 20.0,
        z_max_mm: float = 170.0,
        point_count: int = 321,
        noise_std_x_mm: float = 0.015,
        noise_std_z_mm: float = 0.035,
        dropout_probability: float = 0.0,
        seed: int = 7,
    ) -> None:
        if not isinstance(robot, MujocoRB5Robot):
            raise TypeError("MujocoKeyenceLaser requires a MujocoRB5Robot")
        if point_count < 2 or x_max_mm <= x_min_mm or z_max_mm <= z_min_mm:
            raise ValueError("invalid Keyence profile range or point count")
        if noise_std_x_mm < 0.0 or noise_std_z_mm < 0.0:
            raise ValueError("Keyence noise standard deviations must be non-negative")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        self.robot = robot
        self.x_values = np.linspace(float(x_min_mm), float(x_max_mm), int(point_count))
        self.z_min_mm = float(z_min_mm)
        self.z_max_mm = float(z_max_mm)
        self.noise_std_x_mm = float(noise_std_x_mm)
        self.noise_std_z_mm = float(noise_std_z_mm)
        self.dropout_probability = float(dropout_probability)
        self.rng = np.random.default_rng(seed)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        del timeout_s
        if not self.connected:
            raise RuntimeError("MuJoCo Keyence sensor is not connected")
        T_base_sensor = self.robot.current_T_base_tcp() @ self.robot.T_tcp_sensor_true
        normal = self.robot.T_base_plane[:3, 2]
        plane_origin = self.robot.T_base_plane[:3, 3]
        plane_offset = float(normal @ plane_origin)
        R = T_base_sensor[:3, :3]
        origin = T_base_sensor[:3, 3]
        denominator = float(normal @ R[:, 2])
        if abs(denominator) < 1e-9:
            raise RuntimeError("laser scan plane is parallel to the calibration plane")
        z = (
            plane_offset
            - float(normal @ origin)
            - self.x_values * float(normal @ R[:, 0])
        ) / denominator
        points = np.column_stack([self.x_values, np.zeros_like(z), z])
        points_base = (R @ points.T).T + origin
        points_plane = (
            self.robot.T_base_plane[:3, :3].T
            @ (points_base - plane_origin).T
        ).T
        half_size = 0.5 * self.robot.scene_config.plane_size_mm
        valid = (
            (z >= self.z_min_mm)
            & (z <= self.z_max_mm)
            & (np.abs(points_plane[:, 0]) <= half_size[0])
            & (np.abs(points_plane[:, 1]) <= half_size[1])
        )
        if self.dropout_probability > 0.0:
            valid &= self.rng.random(len(valid)) >= self.dropout_probability
        points = points[valid]
        if len(points) == 0:
            raise RuntimeError("Keyence simulation returned no points on the target")
        if self.noise_std_x_mm > 0.0:
            points[:, 0] += self.rng.normal(0.0, self.noise_std_x_mm, len(points))
        if self.noise_std_z_mm > 0.0:
            points[:, 2] += self.rng.normal(0.0, self.noise_std_z_mm, len(points))
        return ProfileSample(points_s=points, timestamp_ns=time.time_ns())
