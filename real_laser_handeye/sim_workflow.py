from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from handeye_mujoco import HandEyeSimulation

from .estimate_initial_plane import next_capture_path
from .workflow import (
    VisualizationSample,
    WorkflowConfig,
    WorkflowController,
    _load_transform,
)


class SimWorkflowController(WorkflowController):
    """Hardware-free workflow using a synthetic board and MuJoCo joint state."""

    def __init__(self, config: WorkflowConfig):
        super().__init__(self._make_session_config(config))
        self.connected = False
        self.edge_index = 0
        self.simulation = HandEyeSimulation(self.config.path("mujoco_model"))
        self.simulation.reset_home()
        self.current_joint_deg = np.degrees(
            self.simulation.named_joint_positions(
                self.simulation.data.qpos, self.config.joint_names
            )
        )
        self.visual_sample_id = 0
        self.last_points_s: np.ndarray | None = None
        self.last_T_base_tcp: np.ndarray | None = None
        self._auto_safe_initialized = False

        # Board geometry exists independently of the stable robot pose.
        self._initialize_virtual_board()

        # Keep a valid provisional state until Connect is pressed.  In SIM,
        # Connect replaces this with a MoveIt-generated collision-free stable
        # pose when simulation.auto_safe_pose is enabled.
        self._set_safe_pose_from_current_mujoco()

    @staticmethod
    def _make_session_config(config: WorkflowConfig) -> WorkflowConfig:
        values = deepcopy(config.values)
        sim = values.get("simulation", {})
        if not bool(sim.get("new_session_per_launch", True)):
            return WorkflowConfig(config.source_path, values)
        base = config.path("session_root")
        stamp = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        session = base / stamp
        replacements = {
            "initial_dataset": session / "initial_plane/dataset",
            "estimate_dir": session / "initial_plane/estimate",
            "scan_dataset": session / "calibration_scans",
            "planning_model": session / "planning/model_with_infinite_plane.xml",
            "scan_plan": session / "planning/scan_plan.json",
            "calibrated_transform": session / "T_tcp_sensor_calibrated.csv",
        }
        for key, path in replacements.items():
            values["paths"][key] = str(path.resolve())
        return WorkflowConfig(config.source_path, values)

    def _initialize_virtual_board(self) -> None:
        settings = self.config.values.get("simulation", {})
        sensor_mm = self._current_sensor_pose_mm()
        forward_axis = str(
            self.config.values["planning"].get("sensor_forward_axis", "+z")
        ).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError("planning.sensor_forward_axis must be '+z' or '-z'")
        forward_sign = 1.0 if forward_axis == "+z" else -1.0

        configured_center = settings.get("board_center_base_mm")
        configured_normal = settings.get("board_normal_base")
        if configured_center is not None and configured_normal is not None:
            self.board_center_mm = np.asarray(configured_center, dtype=float).reshape(3)
            normal = np.asarray(configured_normal, dtype=float).reshape(3)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-12:
                raise ValueError("simulation.board_normal_base must be non-zero")
            normal /= normal_norm
            minimum_axis_tilt = float(settings.get("minimum_plane_axis_tilt_deg", 5.0))
            plane_axis_angles = np.degrees(np.arcsin(np.clip(np.abs(normal), 0.0, 1.0)))
            if np.min(plane_axis_angles) < minimum_axis_tilt:
                raise ValueError(
                    "simulation board plane must be tilted from every base axis by at "
                    f"least {minimum_axis_tilt:.1f} deg; got {plane_axis_angles.tolist()}"
                )
            up_hint = np.array([0.0, 0.0, 1.0])
            u_axis = np.cross(up_hint, normal)
            if np.linalg.norm(u_axis) < 1e-8:
                u_axis = np.cross(np.array([0.0, 1.0, 0.0]), normal)
            u_axis /= np.linalg.norm(u_axis)
            v_axis = np.cross(normal, u_axis)
            v_axis /= np.linalg.norm(v_axis)
        else:
            # Legacy fallback: infer a board in front of the MuJoCo home pose.
            z_axis = sensor_mm[:3, 2]
            view_axis = forward_sign * z_axis
            normal = -view_axis / np.linalg.norm(view_axis)
            u_axis = sensor_mm[:3, 0]
            u_axis = u_axis - normal * float(normal @ u_axis)
            u_axis /= np.linalg.norm(u_axis)
            v_axis = np.cross(normal, u_axis)
            v_axis /= np.linalg.norm(v_axis)
            distance = float(settings.get("board_distance_from_home_sensor_mm", 300.0))
            self.board_center_mm = sensor_mm[:3, 3] - distance * normal

        self.board_normal = normal
        self.board_u = u_axis
        self.board_v = v_axis
        self.board_half_width_mm = float(settings.get("board_half_width_mm", 120.0))
        self.board_half_height_mm = float(settings.get("board_half_height_mm", 90.0))
        self.initial_sensor_distance_mm = float(
            settings.get("initial_sensor_distance_mm", 220.0)
        )

    def _current_sensor_pose_mm(self) -> np.ndarray:
        sensor = self.simulation.sensor_pose_world().copy()
        sensor[:3, 3] *= 1e3
        return sensor

    def _set_robot_joint_rad(self, q_robot_rad: np.ndarray) -> None:
        q_robot_rad = np.asarray(q_robot_rad, dtype=float).reshape(
            len(self.config.joint_names)
        )
        full_qpos = self.simulation.qpos_with_named_joints(
            dict(zip(self.config.joint_names, q_robot_rad)),
            base_qpos=self.simulation.data.qpos.copy(),
        )
        self.simulation.set_joint_positions(full_qpos)
        self.current_joint_deg = np.degrees(q_robot_rad)

    def _set_safe_pose_from_current_mujoco(self) -> None:
        self.safe_joint_deg = self.current_joint_deg.copy()
        handeye = _load_transform(self.config.path("handeye"))
        T_base_sensor = self._current_sensor_pose_mm()
        self.safe_tcp = T_base_sensor @ np.linalg.inv(handeye)

        settings = self.config.values.get("simulation", {})

        # When automatic safe-pose generation is enabled, the current MuJoCo
        # home state is only a provisional pose.  Do not require it to form a
        # valid board-facing geometry; Connect will replace it with the
        # MoveIt-generated collision-free stable pose.
        if bool(settings.get("auto_safe_pose", True)):
            self.safe_forward_center_dot = float("nan")
            self.safe_plus_z_normal_dot = float("nan")
            return

        self._update_safe_pose_metrics(T_base_sensor)

    def _update_safe_pose_metrics(self, T_base_sensor_mm: np.ndarray) -> None:
        T_base_sensor_mm = np.asarray(T_base_sensor_mm, dtype=float)
        forward_axis = str(
            self.config.values["planning"].get("sensor_forward_axis", "+z")
        ).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError("planning.sensor_forward_axis must be '+z' or '-z'")
        forward_sign = 1.0 if forward_axis == "+z" else -1.0
        to_center = self.board_center_mm - T_base_sensor_mm[:3, 3]
        center_distance = float(np.linalg.norm(to_center))
        if center_distance < 1e-9:
            raise ValueError("simulation stable sensor origin coincides with board centre")
        sensor_forward = forward_sign * T_base_sensor_mm[:3, 2]
        self.safe_forward_center_dot = float(
            sensor_forward @ (to_center / center_distance)
        )
        self.safe_plus_z_normal_dot = float(
            T_base_sensor_mm[:3, 2] @ self.board_normal
        )

    def _moveit_settings_for_sim(self) -> dict[str, object]:
        planning = self.config.values["planning"]
        settings = dict(planning.get("moveit", {}))
        setup = Path(
            str(settings.get("workspace_setup", "../../moveit2_ws/install/setup.bash"))
        )
        if not setup.is_absolute():
            setup = (self.config.source_path.parent / setup).resolve()
        settings["workspace_setup"] = str(setup)

        safe_settings = self.config.values.get("simulation", {}).get("safe_pose", {})
        if isinstance(safe_settings, dict) and safe_settings.get("ik_timeout_s") is not None:
            settings["ik_timeout_s"] = float(safe_settings["ik_timeout_s"])
        return settings

    def _board_plane_transform_mm(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 0] = self.board_u
        transform[:3, 1] = self.board_v
        transform[:3, 2] = self.board_normal
        transform[:3, 3] = self.board_center_mm
        return transform

    def _center_facing_physical_pose(
        self, distance_mm: float, roll_deg: float
    ) -> np.ndarray:
        """Place physical origin P on +plane-normal side, looking at board centre.

        The requested roll is applied about the measurement frame viewing axis.
        The fixed S<->P rotation is then included so that the actual laser
        measurement frame S, rather than merely P, faces the board centre.
        """
        distance_mm = float(distance_mm)
        if not np.isfinite(distance_mm) or distance_mm <= 0:
            raise ValueError("automatic stable-pose distance must be positive")

        forward_axis = str(
            self.config.values["planning"].get("sensor_forward_axis", "+z")
        ).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError("planning.sensor_forward_axis must be '+z' or '-z'")

        # Desired measurement-frame orientation.  The sensor is on +n side.
        # +Z forward => +Z points toward board (-n).
        # -Z forward => +Z points away from board (+n).
        z_sensor = -self.board_normal if forward_axis == "+z" else self.board_normal
        x0 = self.board_u - z_sensor * float(z_sensor @ self.board_u)
        x0 /= np.linalg.norm(x0)
        y0 = np.cross(z_sensor, x0)
        y0 /= np.linalg.norm(y0)

        roll = np.deg2rad(float(roll_deg))
        x_sensor = np.cos(roll) * x0 + np.sin(roll) * y0
        x_sensor /= np.linalg.norm(x_sensor)
        y_sensor = np.cross(z_sensor, x_sensor)
        y_sensor /= np.linalg.norm(y_sensor)
        R_base_sensor = np.column_stack([x_sensor, y_sensor, z_sensor])

        # ^B R_P = ^B R_S * ^S R_P.  Position is intentionally defined for P
        # because P is the MoveIt tip and the scan-pattern path-design origin.
        T_measurement_physical = self.config.T_measurement_physical_mm  # ^S T_P
        target = np.eye(4)
        target[:3, :3] = R_base_sensor @ T_measurement_physical[:3, :3]
        target[:3, 3] = self.board_center_mm + distance_mm * self.board_normal
        return target

    def _automatic_safe_distance_candidates(self) -> list[float]:
        simulation_settings = self.config.values.get("simulation", {})
        safe_settings = simulation_settings.get("safe_pose", {})
        if not isinstance(safe_settings, dict):
            safe_settings = {}
        planning = self.config.values["planning"]

        nominal = safe_settings.get("distance_mm")
        if nominal is None:
            near = float(planning.get("distance_near_mm", 360.0))
            far = float(planning.get("distance_far_mm", 440.0))
            nominal = 0.5 * (near + far)
        nominal = float(nominal)

        offsets = safe_settings.get(
            "distance_offsets_mm", [0.0, 50.0, -50.0, 100.0, -100.0]
        )
        values: list[float] = []
        for offset in offsets:
            value = nominal + float(offset)
            if value <= 0 or not np.isfinite(value):
                continue
            if not any(abs(value - existing) < 1e-9 for existing in values):
                values.append(value)
        if not values:
            raise ValueError("simulation.safe_pose produced no positive distance candidates")
        return values

    def _initialize_auto_safe_pose(self) -> None:
        """Find a collision-free centre-looking SIM stable pose using MoveIt IK."""
        from .moveit_planner import MoveItServicePlanner

        planning = self.config.values["planning"]
        if str(planning.get("backend", "moveit")).strip().lower() != "moveit":
            raise RuntimeError(
                "simulation.auto_safe_pose currently requires planning.backend=moveit"
            )

        sim_settings = self.config.values.get("simulation", {})
        safe_settings = sim_settings.get("safe_pose", {})
        if not isinstance(safe_settings, dict):
            safe_settings = {}
        rolls = [
            float(value)
            for value in safe_settings.get(
                "roll_candidates_deg", [0.0, 90.0, -90.0, 180.0]
            )
        ]
        if not rolls:
            raise ValueError("simulation.safe_pose.roll_candidates_deg must not be empty")

        names = self.config.joint_names
        current_seed = np.radians(self.current_joint_deg)
        seed_candidates = [current_seed, np.zeros(len(names), dtype=float)]
        distances = self._automatic_safe_distance_candidates()
        settings = self._moveit_settings_for_sim()
        plane = self._board_plane_transform_mm()

        failures: list[str] = []
        selected: tuple[np.ndarray, np.ndarray, float, float] | None = None
        with MoveItServicePlanner(settings) as moveit:
            moveit.apply_scene(
                plane,
                clearance_mm=float(planning.get("plane_keepout_clearance_mm", 10.0)),
            )

            for distance_mm in distances:
                for roll_deg in rolls:
                    target_physical = self._center_facing_physical_pose(
                        distance_mm, roll_deg
                    )
                    for seed_index, seed in enumerate(seed_candidates):
                        ok, reason, q_safe, error_code = moveit.solve_pose_ik(
                            names,
                            seed,
                            target_physical,
                            avoid_collisions=True,
                        )
                        if ok:
                            selected = (q_safe, target_physical, distance_mm, roll_deg)
                            break
                        failures.append(
                            f"d={distance_mm:.1f} roll={roll_deg:.1f} "
                            f"seed={seed_index}: {reason} ({error_code})"
                        )
                    if selected is not None:
                        break
                if selected is not None:
                    break

        if selected is None:
            detail = "; ".join(failures[-8:])
            raise RuntimeError(
                "MoveIt could not generate a collision-free automatic SIM stable pose. "
                f"Tried {len(distances)} distances x {len(rolls)} rolls x "
                f"{len(seed_candidates)} seeds. Last failures: {detail}"
            )

        q_safe, T_base_physical, distance_mm, roll_deg = selected

        # Apply the MoveIt IK solution to MuJoCo.
        self._set_robot_joint_rad(q_safe)
        self.safe_joint_deg = self.current_joint_deg.copy()

        # IMPORTANT:
        # Read the actual measurement-frame pose from MuJoCo after applying q_safe.
        # Do not reconstruct S from the requested physical-frame target because
        # MoveIt/MuJoCo tool-frame definitions and the fixed S<->P transform must
        # agree exactly for that reconstruction to be valid.
        T_base_sensor = self._current_sensor_pose_mm()

        handeye = _load_transform(self.config.path("handeye"))
        self.safe_tcp = T_base_sensor @ np.linalg.inv(handeye)

        self._update_safe_pose_metrics(T_base_sensor)
        self._auto_safe_initialized = True

        print(
            "[SIM] automatic stable pose: "
            f"distance={distance_mm:.1f} mm, roll={roll_deg:.1f} deg, "
            f"joint_deg={np.round(self.safe_joint_deg, 3).tolist()}"
        )

    @property
    def session_directory(self) -> Path:
        return self.config.path("estimate_dir").parents[1]

    def connect(self) -> None:
        settings = self.config.values.get("simulation", {})
        if bool(settings.get("auto_safe_pose", True)) and not self._auto_safe_initialized:
            self._initialize_auto_safe_pose()
        self.connected = True
        self.motion_verified = True

    def close(self) -> None:
        self.connected = False
        self.motion_verified = False

    def _require_hardware(self) -> None:
        if not self.connected:
            raise RuntimeError("SIM is not connected; press Connect first")

    def record_safe_pose(self) -> None:
        self._require_hardware()
        self.safe_joint_deg = self.current_joint_deg.copy()
        handeye = _load_transform(self.config.path("handeye"))
        full_qpos = self.simulation.qpos_with_named_joints(
            dict(zip(self.config.joint_names, np.radians(self.current_joint_deg))),
            base_qpos=self.simulation.data.qpos.copy(),
        )
        self.simulation.set_joint_positions(full_qpos)
        sensor = self.simulation.sensor_pose_world()
        sensor[:3, 3] *= 1e3
        self.safe_tcp = sensor @ np.linalg.inv(handeye)

    def capture_initial_line(self) -> Path:
        self._require_hardware()
        # top, right, bottom, left; every line is expressed in its own sensor XZ plane.
        edges = (
            (self.board_v * self.board_half_height_mm, self.board_u, self.board_half_width_mm, "top"),
            (self.board_u * self.board_half_width_mm, self.board_v, self.board_half_height_mm, "right"),
            (-self.board_v * self.board_half_height_mm, self.board_u, self.board_half_width_mm, "bottom"),
            (-self.board_u * self.board_half_width_mm, self.board_v, self.board_half_height_mm, "left"),
        )
        edge_cycle = self.edge_index % len(edges)
        offset, line_axis, half_length, edge_name = edges[edge_cycle]
        midpoint = self.board_center_mm + offset
        forward_axis = str(
            self.config.values["planning"].get("sensor_forward_axis", "+z")
        ).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError("planning.sensor_forward_axis must be '+z' or '-z'")
        forward_sign = 1.0 if forward_axis == "+z" else -1.0
        view_axis = -self.board_normal
        z_axis = forward_sign * view_axis
        x_axis = line_axis / np.linalg.norm(line_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        sensor_origin = midpoint - self.initial_sensor_distance_mm * view_axis
        T_base_sensor = np.eye(4)
        T_base_sensor[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        T_base_sensor[:3, 3] = sensor_origin
        samples = int(self.config.values.get("simulation", {}).get("profile_points", 401))
        x = np.linspace(-half_length, half_length, samples)
        points_s = np.column_stack(
            [
                x,
                np.zeros_like(x),
                np.full_like(x, forward_sign * self.initial_sensor_distance_mm),
            ]
        )
        handeye = _load_transform(self.config.path("handeye"))
        T_base_tcp = T_base_sensor @ np.linalg.inv(handeye)
        output = self._save_capture(
            self.config.path("initial_dataset"),
            T_base_tcp,
            points_s,
            extra={"sim_edge": edge_name},
        )
        self._set_visual_sample(points_s, T_base_tcp)
        self.edge_index += 1
        return output

    def capture_additional_scan(self) -> Path:
        self._require_hardware()
        handeye = _load_transform(self.config.path("handeye"))
        full_qpos = self.simulation.qpos_with_named_joints(
            dict(zip(self.config.joint_names, np.radians(self.current_joint_deg))),
            base_qpos=self.simulation.data.qpos.copy(),
        )
        self.simulation.set_joint_positions(full_qpos)
        T_base_sensor = self.simulation.sensor_pose_world()
        T_base_sensor[:3, 3] *= 1e3
        T_base_tcp = T_base_sensor @ np.linalg.inv(handeye)
        points_s = self._profile_on_estimated_plane(T_base_sensor)
        output = self._save_capture(
            self.calibration_dataset_path(),
            T_base_tcp,
            points_s,
            extra={"sim_manual_extra": True},
        )
        self._set_visual_sample(points_s, T_base_tcp)
        return output

    def go_to_plan_start(self) -> str:
        """Move SIM state to the recorded plan START/safe joint state."""
        self._require_hardware()
        if self.plan is None:
            raise RuntimeError("generate a scan plan first")
        if int(self.plan.execution_step_index) != 0:
            raise RuntimeError(
                "GO TO START is only available before the first execution step"
            )

        target_rad = np.asarray(self.plan.safe_joint_rad, dtype=float)
        target_deg = np.degrees(target_rad)
        tolerance_deg = float(
            self.config.values["motion"].get("start_tolerance_deg", 1.0)
        )
        error_deg = self._joint_error_deg(self.current_joint_deg, target_deg)
        if error_deg <= tolerance_deg:
            self.current_safe_joint_rad = target_rad.copy()
            return f"Already at START (max joint error {error_deg:.3f} deg)"

        delay_s = float(
            self.config.values["motion"].get("simulated_delay_s", 0.15)
        )
        self._emit_motion_progress(
            phase="moving",
            label="SIM go to START / recorded safe",
            completed=0,
            total=1,
            fraction=0.0,
            elapsed_s=0.0,
            remaining_s=delay_s,
        )
        self._set_robot_joint_rad(target_rad)
        time.sleep(delay_s)
        self._emit_motion_progress(
            phase="moving",
            label="SIM go to START / recorded safe",
            completed=1,
            total=1,
            fraction=1.0,
            elapsed_s=delay_s,
            remaining_s=0.0,
        )

        final_error_deg = self._joint_error_deg(
            self.current_joint_deg, target_deg
        )
        if final_error_deg > tolerance_deg:
            raise RuntimeError(
                "SIM did not reach plan START: "
                f"max joint error={final_error_deg:.3f} deg > "
                f"{tolerance_deg:.3f} deg"
            )

        self.current_safe_joint_rad = target_rad.copy()
        self.active_scan = None
        self.previewed_scan_id = None
        self.previewed_return_scan_id = None
        return f"Reached START (max joint error {final_error_deg:.3f} deg)"

    def execute_next_execution_step(self) -> Path | None:
        """SIM counterpart of the unified SCAN/SAFE execution step."""
        self._require_hardware()
        if self.plan is None:
            raise RuntimeError("scan plan is not available")
        step = self._next_execution_step()

        planned_start_deg = np.degrees(np.asarray(step.trajectory_rad[0], dtype=float))
        error_deg = self._joint_error_deg(self.current_joint_deg, planned_start_deg)
        tolerance_deg = float(
            self.config.values["motion"].get("start_tolerance_deg", 1.0)
        )
        if error_deg > tolerance_deg:
            raise RuntimeError(
                "SIM state does not match the next planned step start: "
                f"max joint error={error_deg:.6f} deg > {tolerance_deg:.6f} deg; "
                f"current={np.round(self.current_joint_deg, 4).tolist()}, "
                f"planned_start={np.round(planned_start_deg, 4).tolist()}"
            )

        delay_s = float(self.config.values["motion"].get("simulated_delay_s", 0.15))
        self._emit_motion_progress(
            phase="moving",
            label=f"SIM execute {step.target_label}",
            completed=0,
            total=1,
            fraction=0.0,
            elapsed_s=0.0,
            remaining_s=delay_s,
        )
        self._set_robot_joint_rad(np.asarray(step.trajectory_rad[-1], dtype=float))
        time.sleep(delay_s)
        self._emit_motion_progress(
            phase="moving",
            label=f"SIM execute {step.target_label}",
            completed=1,
            total=1,
            fraction=1.0,
            elapsed_s=delay_s,
            remaining_s=0.0,
        )

        captured: Path | None = None
        if step.target_type == "SCAN":
            scan = next(
                (item for item in self.plan.scans if item.pose.scan_id == step.scan_id),
                None,
            )
            if scan is None:
                raise RuntimeError(f"planned scan {step.scan_id} is missing")
            self._emit_motion_progress(
                phase="capture",
                label=f"SIM capture scan {step.scan_id}",
                completed=1,
                total=1,
                fraction=1.0,
                elapsed_s=0.0,
                remaining_s=None,
            )
            points_s = self._profile_on_estimated_plane(scan.pose.T_base_sensor)
            captured = self._save_capture(
                self.config.path("scan_dataset"),
                scan.pose.T_base_tcp,
                points_s,
                extra={"sim_scan_id": scan.pose.scan_id},
            )
            self._set_visual_sample(points_s, scan.pose.T_base_tcp)
            scan.status = "SCANNED"
            scan.captured_path = str(captured)

        step.status = "DONE"
        self.plan.execution_step_index += 1
        self.active_scan = None
        self.previewed_scan_id = None
        self.previewed_return_scan_id = None
        self.plan.save(self.config.path("scan_plan"))
        return captured

    def next_scan(self) -> Path:
        self._require_hardware()
        if self.plan is None:
            raise RuntimeError("generate and validate a scan plan first")
        continuous = self.plan.execution_mode in {"continuous", "greedy", "optimized"}
        if self.active_scan is not None:
            if not continuous:
                raise RuntimeError("return to the stable pose before requesting the next scan")
            if self._joint_error_deg(
                self.current_joint_deg,
                np.degrees(self.active_scan.q_robot_rad),
            ) > 1e-6:
                raise RuntimeError("SIM joint state changed outside the planned path")
        else:
            self._assert_at_safe_pose()
        scan = self._next_safe_scan()
        if self.previewed_scan_id != scan.pose.scan_id:
            raise RuntimeError(
                f"preview scan {scan.pose.scan_id} in MuJoCo before pressing Next"
            )
        assert scan.q_robot_rad is not None
        assert scan.movej_robot_trajectory_rad is not None
        if self._joint_error_deg(
            self.current_joint_deg,
            np.degrees(scan.movej_robot_trajectory_rad[0]),
        ) > 1e-5:
            raise RuntimeError("SIM state does not match the planned path start")
        self.active_scan = scan
        delay_s = float(self.config.values["motion"].get("simulated_delay_s", 0.15))
        self._emit_motion_progress(
            phase="moving",
            label=f"SIM move to scan {scan.pose.scan_id}",
            completed=0,
            total=1,
            fraction=0.0,
            elapsed_s=0.0,
            remaining_s=delay_s,
        )
        self._set_robot_joint_rad(scan.q_robot_rad)
        time.sleep(delay_s)
        self._emit_motion_progress(
            phase="moving",
            label=f"SIM move to scan {scan.pose.scan_id}",
            completed=1,
            total=1,
            fraction=1.0,
            elapsed_s=delay_s,
            remaining_s=0.0,
        )
        self._emit_motion_progress(
            phase="capture",
            label=f"SIM capture scan {scan.pose.scan_id}",
            completed=1,
            total=1,
            fraction=1.0,
            elapsed_s=0.0,
            remaining_s=None,
        )
        points_s = self._profile_on_estimated_plane(scan.pose.T_base_sensor)
        output = self._save_capture(
            self.config.path("scan_dataset"),
            scan.pose.T_base_tcp,
            points_s,
            extra={"sim_scan_id": scan.pose.scan_id},
        )
        self._set_visual_sample(points_s, scan.pose.T_base_tcp)
        scan.status = "SCANNED"
        scan.captured_path = str(output)
        self.previewed_scan_id = None
        self.previewed_return_scan_id = None
        self.plan.save(self.config.path("scan_plan"))
        return output

    def return_to_safe(self) -> None:
        self._require_hardware()
        if self.active_scan is None:
            if self._joint_error_deg(self.current_joint_deg, self.safe_joint_deg) <= 1e-6:
                return
            raise RuntimeError("SIM state is not at a planned scan pose")
        assert self.active_scan.q_robot_rad is not None
        if self._joint_error_deg(
            self.current_joint_deg, np.degrees(self.active_scan.q_robot_rad)
        ) > 1e-6:
            raise RuntimeError("SIM joint state changed outside the planned path")
        delay_s = float(self.config.values["motion"].get("simulated_delay_s", 0.15))
        self._emit_motion_progress(
            phase="moving",
            label="SIM return to safe pose",
            completed=0,
            total=1,
            fraction=0.0,
            elapsed_s=0.0,
            remaining_s=delay_s,
        )
        time.sleep(delay_s)
        self._set_robot_joint_rad(np.radians(self.safe_joint_deg))
        self._emit_motion_progress(
            phase="moving",
            label="SIM return to safe pose",
            completed=1,
            total=1,
            fraction=1.0,
            elapsed_s=delay_s,
            remaining_s=0.0,
        )
        self.active_scan = None
        self.previewed_return_scan_id = None

    def _validated_return_to_safe_robot_trajectory_rad(self) -> np.ndarray:
        """SIM equivalent of the real controller's guarded return-path lookup."""
        self._require_hardware()
        if self.plan is None:
            raise RuntimeError("scan plan is not available")
        if self.active_scan is None:
            if self._joint_error_deg(self.current_joint_deg, self.safe_joint_deg) <= 1e-6:
                raise RuntimeError("robot is already at the safe pose")
            raise RuntimeError("SIM state is not at a planned scan pose")
        assert self.active_scan.q_robot_rad is not None
        if self._joint_error_deg(
            self.current_joint_deg, np.degrees(self.active_scan.q_robot_rad)
        ) > 1e-6:
            raise RuntimeError("SIM joint state changed outside the planned path")
        return_path = self.active_scan.return_to_safe_robot_trajectory_rad
        if return_path is None:
            if self.active_scan.movej_robot_trajectory_rad is None:
                raise RuntimeError("MoveIt-planned return-to-safe trajectory is unavailable")
            return_path = self.active_scan.movej_robot_trajectory_rad[::-1].copy()
        return np.asarray(return_path, dtype=float).copy()

    def _set_visual_sample(
        self, points_s: np.ndarray, T_base_tcp: np.ndarray
    ) -> None:
        self.visual_sample_id += 1
        self.last_points_s = np.asarray(points_s, dtype=float).copy()
        self.last_T_base_tcp = np.asarray(T_base_tcp, dtype=float).copy()

    def read_visualization_sample(self) -> VisualizationSample | None:
        self._require_hardware()

        # Mount check must follow the current MuJoCo joint state, not the TCP
        # cached at the most recent laser capture.
        handeye = _load_transform(self.config.path("handeye"))
        T_base_sensor = self._current_sensor_pose_mm()
        T_base_tcp = T_base_sensor @ np.linalg.inv(handeye)

        if self.last_points_s is None:
            sample_id = -1
            points_s = np.empty((0, 3))
        else:
            sample_id = self.visual_sample_id
            points_s = self.last_points_s.copy()

        return VisualizationSample(
            sample_id,
            points_s,
            np.asarray(T_base_tcp, dtype=float).copy(),
            self.current_joint_deg.copy(),
        )

    def _assert_at_safe_pose(self) -> None:
        if self._joint_error_deg(self.current_joint_deg, self.safe_joint_deg) > 1e-6:
            raise RuntimeError("SIM must return to the stable pose before Next")

    def _profile_on_estimated_plane(self, T_base_sensor: np.ndarray) -> np.ndarray:
        metadata = json.loads(
            (self.config.path("estimate_dir") / "plane_estimate.json").read_text(
                encoding="utf-8"
            )
        )
        center = np.asarray(metadata["centroid_w_mm"], dtype=float)
        normal = np.asarray(metadata["normal_w"], dtype=float)
        origin = T_base_sensor[:3, 3]
        x_axis = T_base_sensor[:3, 0]
        z_axis = T_base_sensor[:3, 2]
        denominator = float(normal @ z_axis)
        if abs(denominator) < 1e-6:
            raise RuntimeError("virtual laser XZ plane is parallel to the estimated plane")
        settings = self.config.values.get("simulation", {})
        half_width = float(settings.get("scan_profile_half_width_mm", 25.0))
        samples = int(settings.get("profile_points", 401))
        x = np.linspace(-half_width, half_width, samples)
        z = np.asarray(
            [float(normal @ (center - origin - value * x_axis)) / denominator for value in x]
        )
        return np.column_stack([x, np.zeros_like(x), z])

    def _save_capture(
        self,
        dataset_dir: Path,
        T_base_tcp: np.ndarray,
        points_s: np.ndarray,
        *,
        extra: dict[str, object],
    ) -> Path:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output = next_capture_path(dataset_dir)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        handeye = _load_transform(self.config.path("handeye"))
        T_base_sensor = T_base_tcp @ handeye
        points_w = points_s @ T_base_sensor[:3, :3].T + T_base_sensor[:3, 3]
        timestamp = np.int64(time.time_ns())
        payload = {
            "T_world_tcp": np.asarray(T_base_tcp, dtype=float),
            "T_base_tcp": np.asarray(T_base_tcp, dtype=float),
            "T_tcp_sensor": handeye,
            "T_world_sensor": T_base_sensor,
            "points_s": np.asarray(points_s, dtype=float),
            "points_w": points_w,
            "tcp_timestamp_ns": timestamp,
            "profile_timestamp_ns": timestamp,
            "captured_at": np.array(datetime.now(timezone.utc).isoformat()),
            "simulation": np.array(True),
        }
        payload.update({key: np.array(value) for key, value in extra.items()})
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output