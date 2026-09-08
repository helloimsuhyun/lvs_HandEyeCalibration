from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from handeye_mujoco import (
    HandEyeSimulation,
    automatic_scan_radius_mm,
    centered_convex_inradius,
    write_model_with_plane_box,
)

from .generate_optimal_tcp_poses import CalibrationPose, generate_optimal_tcp_poses


def _load_object(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError(f"object specification must be module:attribute, got {specification!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _load_transform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("T_tcp_sensor", "T_ef_s", "transform"):
                if key in value:
                    value = value[key]
                    break
        matrix = np.asarray(value, dtype=float)
    else:
        matrix = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid transform: {path}")
    return matrix


@dataclass(frozen=True)
class WorkflowConfig:
    source_path: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowConfig":
        source = Path(path).resolve()
        value = cls._load_values(source, ())
        if not isinstance(value, dict):
            raise ValueError("workflow configuration root must be a mapping")
        for section in ("paths", "equipment", "planning", "motion"):
            if not isinstance(value.get(section), dict):
                raise ValueError(f"workflow configuration requires a {section!r} mapping")
        return cls(source, value)

    @classmethod
    def _load_values(
        cls, source: Path, loading: tuple[Path, ...]
    ) -> dict[str, Any]:
        """Load a YAML config, optionally overlaying a nearby base config."""
        source = source.resolve()
        if source in loading:
            chain = " -> ".join(str(path) for path in (*loading, source))
            raise ValueError(f"cyclic workflow config inheritance: {chain}")
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"workflow configuration root must be a mapping: {source}")

        parent_name = value.pop("extends", None)
        if parent_name is None:
            return value
        parent_path = Path(str(parent_name))
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        parent = cls._load_values(parent_path, (*loading, source))
        return cls._deep_merge(parent, value)

    @classmethod
    def _deep_merge(
        cls, base: dict[str, Any], overlay: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overlay.items():
            prior = merged.get(key)
            if isinstance(prior, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(prior, value)
            else:
                merged[key] = value
        return merged

    def path(self, key: str) -> Path:
        raw = Path(str(self.values["paths"][key]))
        return raw.resolve() if raw.is_absolute() else (self.source_path.parent / raw).resolve()

    @property
    def joint_names(self) -> tuple[str, ...]:
        names = tuple(str(value) for value in self.values["equipment"]["joint_names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("equipment.joint_names must be a non-empty unique list")
        return names

    @property
    def T_measurement_physical_mm(self) -> np.ndarray:
        """Return ``^S T_P`` from the hand-eye JSON, with YAML fallback."""
        handeye_path = self.path("handeye")
        if handeye_path.suffix.lower() == ".json":
            payload = json.loads(handeye_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "T_sensor_physical" in payload:
                transform = np.asarray(payload["T_sensor_physical"], dtype=float)
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    raise ValueError(
                        f"invalid T_sensor_physical in hand-eye file: {handeye_path}"
                    )
                return transform

        # Backward-compatible equipment fallback for older hand-eye files.
        frames = self.values.get("sensor_frames", {})
        translation = np.asarray(
            frames.get("physical_to_measurement_translation_mm", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        rpy = np.asarray(
            frames.get("physical_to_measurement_rpy_deg", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        transform[:3, 3] = translation
        return np.linalg.inv(transform)

    @property
    def T_physical_measurement_mm(self) -> np.ndarray:
        """Return ``^P T_S``, the inverse of the supplied ``^S T_P``."""
        return np.linalg.inv(self.T_measurement_physical_mm)


@dataclass
class PlannedScan:
    pose: CalibrationPose
    status: str
    reason: str
    q_robot_rad: np.ndarray | None
    path_sample_count: int

    # Selected calibration-equivalent azimuth branch for this planned scan.
    # Values are "alpha" or "alpha+180" when a path is accepted.
    selected_branch: str | None = None

    # Weighted endpoint joint travel used to rank the two alpha branches:
    #   sum_j w_j * |q_approach[j] - q_current[j]|
    # Stored in degrees for GUI / scan-plan readability.
    approach_joint_cost_deg: float | None = None

    captured_path: str | None = None

    T_base_sensor_approach: np.ndarray | None = None
    T_base_tcp_approach: np.ndarray | None = None
    T_base_physical: np.ndarray | None = None
    T_base_physical_approach: np.ndarray | None = None
    q_approach_robot_rad: np.ndarray | None = None
    preview_qpos_trajectory: np.ndarray | None = None
    preview_time_from_start_s: np.ndarray | None = None

    # RRTConnect: previous APPROACH/START -> current APPROACH
    movej_robot_trajectory_rad: np.ndarray | None = None
    movej_time_from_start_s: np.ndarray | None = None
    movej_velocities_rad_s: np.ndarray | None = None
    movej_accelerations_rad_s2: np.ndarray | None = None

    # Historical ``lin_*`` field names are retained for scan-plan / GUI
    # compatibility. They are MoveIt /compute_cartesian_path trajectories.
    lin_to_scan_robot_trajectory_rad: np.ndarray | None = None
    lin_to_scan_time_from_start_s: np.ndarray | None = None
    lin_to_scan_velocities_rad_s: np.ndarray | None = None
    lin_to_scan_accelerations_rad_s2: np.ndarray | None = None

    lin_to_approach_robot_trajectory_rad: np.ndarray | None = None
    lin_to_approach_time_from_start_s: np.ndarray | None = None
    lin_to_approach_velocities_rad_s: np.ndarray | None = None
    lin_to_approach_accelerations_rad_s2: np.ndarray | None = None

    @property
    def executable(self) -> bool:
        return self.status in {"SAFE", "SCANNED"} and self.q_robot_rad is not None


@dataclass
class ExecutionStep:
    step_id: int
    target_type: str  # "SCAN" or "RETRACT"
    scan_id: int
    route_label: str
    trajectory_rad: np.ndarray
    time_from_start_s: np.ndarray
    velocities_rad_s: np.ndarray | None = None
    accelerations_rad_s2: np.ndarray | None = None
    capture_after: bool = False
    reason: str = ""
    status: str = "PENDING"

    @property
    def target_label(self) -> str:
        return f"{self.target_type} {self.scan_id}"


@dataclass(frozen=True)
class VisualizationSample:
    sample_id: int
    points_s: np.ndarray
    T_base_tcp: np.ndarray
    joint_positions_deg: np.ndarray | None = None


@dataclass
class ScanPlan:
    radius_mm: float
    recommended_radius_mm: float
    hull_inradius_mm: float
    safe_joint_rad: np.ndarray
    joint_names: tuple[str, ...]
    board_model_path: Path
    scans: list[PlannedScan]
    radius_warning: str | None = None
    planner_backend: str = "moveit"
    route_mode: str = "circular_greedy"
    execution_steps: list[ExecutionStep] | None = None
    execution_step_index: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def matrix_or_none(value):
            return None if value is None else np.asarray(value, dtype=float).tolist()

        def rad_trajectory_deg(value):
            return None if value is None else np.degrees(
                np.asarray(value, dtype=float)
            ).tolist()

        def array_or_none(value):
            return None if value is None else np.asarray(value, dtype=float).tolist()

        value = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "radius_mm": self.radius_mm,
            "recommended_radius_mm": self.recommended_radius_mm,
            "hull_inradius_mm": self.hull_inradius_mm,
            "radius_warning": self.radius_warning,
            "planner_backend": self.planner_backend,
            "route_mode": self.route_mode,
            "joint_names": list(self.joint_names),
            "safe_joint_deg": np.degrees(self.safe_joint_rad).tolist(),
            "board_model_path": str(self.board_model_path),
            "execution_step_index": int(self.execution_step_index),
            "execution_steps": [] if self.execution_steps is None else [
                {
                    "step_id": step.step_id,
                    "target_type": step.target_type,
                    "scan_id": step.scan_id,
                    "target": step.target_label,
                    "route": step.route_label,
                    "capture_after": step.capture_after,
                    "status": step.status,
                    "reason": step.reason,
                }
                for step in self.execution_steps
            ],
            "scans": [
                {
                    "scan_id": scan.pose.scan_id,
                    "status": scan.status,
                    "reason": scan.reason,
                    "support_id": scan.pose.support_id,
                    "target_uv_mm": [
                        scan.pose.target_u_mm,
                        scan.pose.target_v_mm,
                    ],
                    "tilt_deg": scan.pose.tilt_deg,
                    "distance_mm": scan.pose.distance_mm,
                    "selected_branch": scan.selected_branch,
                    "approach_joint_cost_deg": scan.approach_joint_cost_deg,
                    "T_base_sensor_mm": scan.pose.T_base_sensor.tolist(),
                    "T_base_tcp_mm": scan.pose.T_base_tcp.tolist(),
                    "T_base_physical_mm": scan.pose.T_base_physical.tolist(),
                    "T_base_sensor_approach_mm": matrix_or_none(
                        scan.T_base_sensor_approach
                    ),
                    "T_base_tcp_approach_mm": matrix_or_none(
                        scan.T_base_tcp_approach
                    ),
                    "T_base_physical_approach_mm": matrix_or_none(
                        scan.T_base_physical_approach
                    ),
                    "approach_joint_deg": None
                    if scan.q_approach_robot_rad is None
                    else np.degrees(scan.q_approach_robot_rad).tolist(),
                    "movej_joint_trajectory_deg": rad_trajectory_deg(
                        scan.movej_robot_trajectory_rad
                    ),
                    "movej_time_from_start_s": array_or_none(
                        scan.movej_time_from_start_s
                    ),
                    "cartesian_to_scan_joint_trajectory_deg": rad_trajectory_deg(
                        scan.lin_to_scan_robot_trajectory_rad
                    ),
                    "cartesian_to_scan_time_from_start_s": array_or_none(
                        scan.lin_to_scan_time_from_start_s
                    ),
                    "cartesian_retract_joint_trajectory_deg": rad_trajectory_deg(
                        scan.lin_to_approach_robot_trajectory_rad
                    ),
                    "cartesian_retract_time_from_start_s": array_or_none(
                        scan.lin_to_approach_time_from_start_s
                    ),
                    "joint_deg": None
                    if scan.q_robot_rad is None
                    else np.degrees(scan.q_robot_rad).tolist(),
                    "path_sample_count": scan.path_sample_count,
                    "preview_time_from_start_s": array_or_none(
                        scan.preview_time_from_start_s
                    ),
                    "captured_path": scan.captured_path,
                }
                for scan in self.scans
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


class ScanPlanner:
    """Optimized-only MoveIt scan planner."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def _load_plane_geometry(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        estimate_dir = self.config.path("estimate_dir")
        boundary = np.loadtxt(
            estimate_dir / "plane_boundary_uv.csv", delimiter=",", skiprows=1
        ).reshape(-1, 2)
        transform = np.loadtxt(
            estimate_dir / "T_world_plane.csv", delimiter=","
        )
        metadata = json.loads(
            (estimate_dir / "plane_estimate.json").read_text(encoding="utf-8")
        )
        return boundary, transform, metadata

    def recommended_radius_mm(self) -> float:
        boundary, _transform, _metadata = self._load_plane_geometry()
        planning = self.config.values["planning"]
        return automatic_scan_radius_mm(
            boundary,
            edge_margin_mm=float(planning.get("hull_edge_margin_mm", 10.0)),
            fill_ratio=float(planning.get("radius_fill_ratio", 0.90)),
        )

    @staticmethod
    def _mujoco_timed_qpos_trajectory(
        simulation: HandEyeSimulation,
        joint_names: tuple[str, ...],
        robot_trajectory_rad: np.ndarray,
        time_from_start_s: np.ndarray,
        base_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a MoveIt trajectory into MuJoCo qpos without discarding time.

        No joint-distance-based densification is performed here.  The original
        MoveIt timestamps are preserved exactly.  The isolated MuJoCo viewer
        interpolates these qpos samples against wall-clock time, so preview
        speed matches the trajectory that EXECUTE sends to the robot.
        """
        robot_path = np.asarray(robot_trajectory_rad, dtype=float)
        times = np.asarray(time_from_start_s, dtype=float).reshape(-1)
        base = np.asarray(base_qpos, dtype=float).reshape(-1)

        if robot_path.ndim != 2 or robot_path.shape[1] != len(joint_names):
            raise ValueError(
                "robot trajectory shape does not match configured joints"
            )
        if len(robot_path) == 0:
            return np.empty((0, len(base)), dtype=float), np.empty((0,), dtype=float)
        if times.shape != (len(robot_path),):
            raise ValueError(
                "MuJoCo preview timing does not match trajectory samples"
            )
        if not np.all(np.isfinite(robot_path)) or not np.all(np.isfinite(times)):
            raise ValueError(
                "MuJoCo preview trajectory/timing contains non-finite values"
            )
        if np.any(times < 0.0):
            raise ValueError("MuJoCo preview timing contains negative values")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "MuJoCo preview timing must be strictly increasing"
            )

        full = np.asarray(
            [
                simulation.qpos_with_named_joints(
                    dict(zip(joint_names, row)),
                    base_qpos=base,
                )
                for row in robot_path
            ],
            dtype=float,
        )
        return full, times.copy()

    @staticmethod
    def _timed_moveit_trajectory(
        result,
        expected_start_rad: np.ndarray,
        joint_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Validate and preserve MoveIt's time-parameterized joint trajectory."""
        positions = np.asarray(result.positions_rad, dtype=float)
        times = np.asarray(result.time_from_start_s, dtype=float).reshape(-1)
        expected_start = np.asarray(expected_start_rad, dtype=float).reshape(joint_count)

        if positions.ndim != 2 or positions.shape[1] != joint_count or len(positions) == 0:
            raise ValueError("MoveIt returned an invalid joint trajectory shape")
        if times.shape != (len(positions),):
            raise ValueError("MoveIt trajectory timing does not match its positions")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(times)):
            raise ValueError("MoveIt returned non-finite trajectory data")
        if np.any(times < 0.0):
            raise ValueError("MoveIt returned a negative time_from_start")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("MoveIt trajectory time_from_start is not strictly increasing")
        if times[-1] <= 0.0:
            raise ValueError("MoveIt trajectory duration is not positive")
        if np.max(np.abs(positions[0] - expected_start)) > 1e-5:
            raise ValueError(
                "MoveIt trajectory does not start at the requested planning state"
            )

        def optional(value) -> np.ndarray | None:
            array = np.asarray(value, dtype=float)
            if array.size == 0:
                return None
            if array.shape != positions.shape or not np.all(np.isfinite(array)):
                raise ValueError("MoveIt returned an invalid velocity/acceleration array")
            return array.copy()

        return (
            positions.copy(),
            times.copy(),
            optional(result.velocities_rad_s),
            optional(result.accelerations_rad_s2),
        )

    def _moveit_settings(self) -> dict[str, Any]:
        planning = self.config.values["planning"]
        settings = dict(planning.get("moveit", {}))
        setup = Path(
            str(
                settings.get(
                    "workspace_setup",
                    "../../moveit2_ws/install/setup.bash",
                )
            )
        )
        if not setup.is_absolute():
            setup = (self.config.source_path.parent / setup).resolve()
        settings["workspace_setup"] = str(setup)
        settings.setdefault("seed", int(planning.get("ik", {}).get("multistart_seed", 1701)))
        return settings

    @staticmethod
    def _greedy_joint_cost(
        current_rad: np.ndarray,
        candidate_rad: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Weighted joint travel used to choose the next greedy scan pose."""
        current = np.asarray(current_rad, dtype=float).reshape(-1)
        candidate = np.asarray(candidate_rad, dtype=float).reshape(-1)
        joint_weights = np.asarray(weights, dtype=float).reshape(-1)
        if current.shape != candidate.shape or current.shape != joint_weights.shape:
            raise ValueError("greedy joint cost vectors must have identical shapes")
        if not (
            np.all(np.isfinite(current))
            and np.all(np.isfinite(candidate))
            and np.all(np.isfinite(joint_weights))
        ):
            raise ValueError("greedy joint cost vectors must be finite")
        if np.any(joint_weights <= 0.0):
            raise ValueError("greedy joint weights must be positive")
        # Planning joints are bounded revolute joints. Do not wrap across +/-pi:
        # crossing a joint limit is real motion, not a zero-distance shortcut.
        return float(np.sum(joint_weights * np.abs(candidate - current)))

    @classmethod
    def _global_alpha_dp_order(
        cls,
        safe_rad: np.ndarray,
        solutions_by_index: dict[int, dict[str, np.ndarray]],
        weights: np.ndarray,
        *,
        max_joint_delta_deg: float | None,
    ) -> tuple[list[tuple[int, str]], list[float], float, float]:
        """Find the minimum-total open route and alpha branch jointly.

        Each scan index contributes exactly one state: either ``alpha`` or
        ``alpha+180``.  Edge cost is the existing weighted endpoint joint
        travel.  A finite ``max_joint_delta_deg`` rejects an edge when any one
        joint changes by more than that limit, preventing a route with one
        excessively large transition from winning on total cost alone.

        The returned route is open: it starts at ``safe_rad`` and does not add
        a final return-to-SAFE edge.
        """
        safe = np.asarray(safe_rad, dtype=float).reshape(-1)
        joint_weights = np.asarray(weights, dtype=float).reshape(-1)
        cls._greedy_joint_cost(safe, safe, joint_weights)

        scan_indices = tuple(sorted(int(index) for index in solutions_by_index))
        if not scan_indices:
            return [], [], 0.0, 0.0
        if len(scan_indices) > 20:
            raise ValueError("global_alpha_dp supports at most 20 scans")

        branches = ("alpha", "alpha+180")
        candidates: dict[tuple[int, str], np.ndarray] = {}
        for index in scan_indices:
            per_branch = solutions_by_index[index]
            for branch in branches:
                if branch not in per_branch:
                    continue
                solution = np.asarray(per_branch[branch], dtype=float).reshape(-1)
                if solution.shape != safe.shape or not np.all(np.isfinite(solution)):
                    raise ValueError(
                        "global_alpha_dp candidate joint vectors must match SAFE"
                    )
                candidates[(index, branch)] = solution.copy()
            if not any((index, branch) in candidates for branch in branches):
                raise ValueError(
                    f"global_alpha_dp scan index {index} has no IK candidate"
                )

        max_delta_rad: float | None
        if max_joint_delta_deg is None:
            max_delta_rad = None
        else:
            value = float(max_joint_delta_deg)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "planning.global_alpha_max_joint_delta_deg must be positive "
                    "or null"
                )
            max_delta_rad = float(np.deg2rad(value))

        bit_by_index = {
            index: 1 << position for position, index in enumerate(scan_indices)
        }

        def edge_cost(start: np.ndarray, goal: np.ndarray) -> float | None:
            delta = np.abs(goal - start)
            if max_delta_rad is not None and float(np.max(delta)) > max_delta_rad:
                return None
            return cls._greedy_joint_cost(start, goal, joint_weights)

        # state -> (total cost, maximum edge cost, previous state, step cost)
        State = tuple[int, int, str]
        records: dict[State, tuple[float, float, State | None, float]] = {}
        states_by_mask: dict[int, list[State]] = {}

        for (index, branch), goal in candidates.items():
            cost = edge_cost(safe, goal)
            if cost is None:
                continue
            state = (bit_by_index[index], index, branch)
            records[state] = (cost, cost, None, cost)
            states_by_mask.setdefault(state[0], []).append(state)

        full_mask = (1 << len(scan_indices)) - 1
        for mask in range(1, full_mask + 1):
            for state in states_by_mask.get(mask, ()):
                record = records[state]
                _mask, last_index, last_branch = state
                last_q = candidates[(last_index, last_branch)]
                total, largest, _previous, _step = record

                for next_index in scan_indices:
                    next_bit = bit_by_index[next_index]
                    if mask & next_bit:
                        continue
                    for next_branch in branches:
                        key = (next_index, next_branch)
                        if key not in candidates:
                            continue
                        step_cost = edge_cost(last_q, candidates[key])
                        if step_cost is None:
                            continue
                        next_state = (
                            mask | next_bit,
                            next_index,
                            next_branch,
                        )
                        candidate_record = (
                            total + step_cost,
                            max(largest, step_cost),
                            state,
                            step_cost,
                        )
                        existing = records.get(next_state)
                        if existing is None or (
                            candidate_record[0], candidate_record[1]
                        ) < (existing[0], existing[1]):
                            records[next_state] = candidate_record
                            if existing is None:
                                states_by_mask.setdefault(
                                    next_state[0], []
                                ).append(next_state)

        terminal_states = [
            (state, record)
            for state, record in records.items()
            if state[0] == full_mask
        ]
        if not terminal_states:
            limit = (
                "disabled"
                if max_joint_delta_deg is None
                else f"{float(max_joint_delta_deg):.3f} deg"
            )
            raise ValueError(
                "global_alpha_dp could not connect all reachable scans with "
                f"max per-joint transition {limit}"
            )

        terminal, terminal_record = min(
            terminal_states,
            key=lambda item: (
                item[1][0],
                item[1][1],
                item[0][1],
                branches.index(item[0][2]),
            ),
        )

        reversed_route: list[tuple[int, str]] = []
        reversed_costs: list[float] = []
        state: State | None = terminal
        while state is not None:
            total, largest, previous, step_cost = records[state]
            del total, largest
            reversed_route.append((state[1], state[2]))
            reversed_costs.append(float(step_cost))
            state = previous

        route = list(reversed(reversed_route))
        step_costs = list(reversed(reversed_costs))
        return (
            route,
            step_costs,
            float(terminal_record[0]),
            float(terminal_record[1]),
        )

    def _fallback_candidates(
        self,
        *,
        original_pose: CalibrationPose,
        T_world_plane: np.ndarray,
        T_measurement_physical: np.ndarray,
        handeye: np.ndarray,
        sensor_forward_axis: str,
    ) -> list[CalibrationPose]:
        """Generate local collision fallbacks for one scan.

        Search order:
          1) keep original tilt and increase distance to distance_far_mm,
          2) if all fail, reduce tilt by 5 deg (configurable),
          3) sweep distance again,
          4) repeat until tilt_min_deg.

        Scan id, support and target UV are preserved. ``distance_mm`` is treated
        as the physical sensor origin P -> board hit-point distance, while the
        viewing direction comes from measurement frame S +/-Z.
        """
        planning = self.config.values["planning"]

        distance_step_mm = float(
            planning.get("collision_distance_fallback_step_mm", 5.0)
        )
        tilt_step_deg = float(
            planning.get("collision_tilt_fallback_step_deg", 5.0)
        )
        tilt_min_deg = float(planning.get("tilt_min_deg", 5.0))
        distance_far_mm = float(planning.get("distance_far_mm", 120.0))

        if not np.isfinite(distance_step_mm) or distance_step_mm <= 0.0:
            raise ValueError(
                "planning.collision_distance_fallback_step_mm must be positive"
            )
        if not np.isfinite(tilt_step_deg) or tilt_step_deg <= 0.0:
            raise ValueError(
                "planning.collision_tilt_fallback_step_deg must be positive"
            )

        T_physical_measurement = np.linalg.inv(T_measurement_physical)
        physical_original = np.asarray(
            original_pose.T_base_physical, dtype=float
        ).copy()
        sensor_original = physical_original @ T_physical_measurement

        if sensor_forward_axis == "+z":
            original_view = sensor_original[:3, 2].copy()
        elif sensor_forward_axis == "-z":
            original_view = -sensor_original[:3, 2].copy()
        else:
            raise ValueError("sensor_forward_axis must be '+z' or '-z'")
        original_view /= np.linalg.norm(original_view)

        plane = np.asarray(T_world_plane, dtype=float).reshape(4, 4)
        plane_normal = plane[:3, 2].copy()
        plane_normal /= np.linalg.norm(plane_normal)
        if float(np.dot(original_view, plane_normal)) < 0.0:
            plane_normal = -plane_normal

        # Distance is defined from the physical origin P, not S.
        original_distance = float(original_pose.distance_mm)
        hit_point = (
            physical_original[:3, 3]
            + original_view * original_distance
        )

        dot = float(np.clip(np.dot(original_view, plane_normal), -1.0, 1.0))
        actual_tilt_rad = float(np.arccos(dot))
        axis = np.cross(original_view, plane_normal)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-12:
            axis /= axis_norm

        distances = [original_distance]
        if distance_far_mm > original_distance + 1e-9:
            value = original_distance + distance_step_mm
            while value < distance_far_mm - 1e-9:
                distances.append(float(value))
                value += distance_step_mm
            if abs(distances[-1] - distance_far_mm) > 1e-9:
                distances.append(float(distance_far_mm))

        original_tilt = float(original_pose.tilt_deg)
        tilts = [original_tilt]
        value = original_tilt - tilt_step_deg
        while value > tilt_min_deg + 1e-9:
            tilts.append(float(value))
            value -= tilt_step_deg
        if original_tilt > tilt_min_deg + 1e-9:
            if abs(tilts[-1] - tilt_min_deg) > 1e-9:
                tilts.append(float(tilt_min_deg))

        candidates: list[CalibrationPose] = []
        for desired_tilt_deg in tilts:
            desired_tilt_rad = np.deg2rad(float(desired_tilt_deg))
            delta_rad = max(0.0, actual_tilt_rad - desired_tilt_rad)

            if delta_rad <= 1e-12 or axis_norm <= 1e-12:
                R_delta = np.eye(3)
            else:
                R_delta = Rotation.from_rotvec(
                    delta_rad * axis
                ).as_matrix()

            R_physical = R_delta @ physical_original[:3, :3]

            for distance_mm in distances:
                physical = np.eye(4)
                physical[:3, :3] = R_physical

                sensor_rotation = (
                    physical[:3, :3]
                    @ T_physical_measurement[:3, :3]
                )
                if sensor_forward_axis == "+z":
                    view = sensor_rotation[:, 2].copy()
                else:
                    view = -sensor_rotation[:, 2].copy()
                view /= np.linalg.norm(view)

                physical[:3, 3] = hit_point - view * distance_mm
                sensor = physical @ T_physical_measurement
                tcp = sensor @ np.linalg.inv(handeye)

                candidates.append(
                    replace(
                        original_pose,
                        tilt_deg=float(desired_tilt_deg),
                        distance_mm=float(distance_mm),
                        T_base_sensor=sensor,
                        T_base_tcp=tcp,
                        T_base_physical=physical,
                    )
                )

        return candidates

    def _plan_scans_with_moveit(
        self,
        *,
        moveit,
        poses: list[CalibrationPose],
        alternate_poses: list[CalibrationPose],
        T_world_plane: np.ndarray,
        simulation: HandEyeSimulation,
        safe_robot: np.ndarray,
        safe_full: np.ndarray,
        names: tuple[str, ...],
        path_step_rad: float,
    ) -> list[PlannedScan]:
        """Plan the only supported industrial sequence.

        Per scan:
            current APPROACH/START
              -> OMPL RRTConnect -> APPROACH
              -> Cartesian path  -> SCAN
              -> Cartesian path  -> APPROACH

        Cartesian failure fallback:
            distance -> far, then tilt -= 5 deg and distance -> far again.
        Final execution ends at the final APPROACH; there is no SAFE return.
        """
        if len(alternate_poses) != len(poses):
            raise ValueError("alternate_poses must match poses one-to-one")

        planning = self.config.values["planning"]
        motion = self.config.values["motion"]
        weights = np.asarray(
            planning.get("greedy_joint_weights", [1.0] * len(names)),
            dtype=float,
        ).reshape(-1)
        if weights.shape != (len(names),):
            raise ValueError(
                "planning.greedy_joint_weights must contain one value per joint"
            )
        self._greedy_joint_cost(safe_robot, safe_robot, weights)

        approach_distance_mm = float(
            motion.get(
                "approach_distance_mm",
                motion.get("approach_offset_mm", 80.0),
            )
        )
        if not np.isfinite(approach_distance_mm) or approach_distance_mm <= 0.0:
            raise ValueError(
                "motion.approach_distance_mm must be positive and finite"
            )

        sensor_forward_axis = str(
            planning.get("sensor_forward_axis", "+z")
        ).strip().lower()
        if sensor_forward_axis not in {"+z", "-z"}:
            raise ValueError(
                "planning.sensor_forward_axis must be '+z' or '-z'"
            )

        T_measurement_physical = self.config.T_measurement_physical_mm
        T_physical_measurement = np.linalg.inv(T_measurement_physical)
        handeye = _load_transform(self.config.path("handeye"))

        def approach_transforms(scan_pose: CalibrationPose):
            physical = np.asarray(
                scan_pose.T_base_physical, dtype=float
            ).copy()
            sensor_scan = physical @ T_physical_measurement
            viewing_direction = sensor_scan[:3, 2].copy()
            if sensor_forward_axis == "-z":
                viewing_direction *= -1.0
            viewing_direction /= np.linalg.norm(viewing_direction)

            # Same orientation, translated opposite the viewing direction.
            physical[:3, 3] -= (
                viewing_direction * approach_distance_mm
            )
            sensor = physical @ T_physical_measurement
            tcp = sensor @ np.linalg.inv(handeye)
            return physical, sensor, tcp

        route_mode = str(
            planning.get("route_mode", "circular_greedy")
        ).strip().lower()
        if route_mode not in {"circular_greedy", "global_alpha_dp"}:
            raise ValueError(
                "planning.route_mode must be 'circular_greedy' or "
                "'global_alpha_dp'"
            )

        # Evaluate both calibration-equivalent alpha branches from START.  The
        # existing circular mode keeps the closer representative.  The global
        # mode retains both states so order and branch can be solved jointly.
        reachable_pose_indices: list[int] = []
        representative_by_index: dict[int, np.ndarray] = {}
        nominal_solutions_by_index: dict[int, dict[str, np.ndarray]] = {}
        representative_failures: list[tuple[int, str]] = []
        for pose_index, pose in enumerate(poses):
            branch_solutions: list[tuple[float, np.ndarray]] = []
            branch_failures: list[str] = []

            for branch_label, branch_pose in (
                ("alpha", pose),
                ("alpha+180", alternate_poses[pose_index]),
            ):
                approach_physical, _, _ = approach_transforms(branch_pose)
                ok, reason, q_goal, _ = moveit.solve_pose_ik(
                    names,
                    safe_robot,
                    approach_physical,
                    avoid_collisions=True,
                    diagnose_failure=False,
                )
                if ok:
                    nominal_solutions_by_index.setdefault(pose_index, {})[
                        branch_label
                    ] = q_goal.copy()
                    branch_solutions.append(
                        (
                            self._greedy_joint_cost(
                                safe_robot, q_goal, weights
                            ),
                            q_goal.copy(),
                        )
                    )
                else:
                    branch_failures.append(f"{branch_label}: {reason}")

            if branch_solutions:
                _, representative = min(
                    branch_solutions, key=lambda item: item[0]
                )
                reachable_pose_indices.append(pose_index)
                representative_by_index[pose_index] = representative
            else:
                representative_failures.append(
                    (pose_index, "; ".join(branch_failures))
                )

        if not reachable_pose_indices:
            return [
                PlannedScan(
                    pose=poses[pose_index],
                    status="NO_PATH",
                    reason="MoveIt approach IK failed: " + reason,
                    q_robot_rad=None,
                    path_sample_count=0,
                    T_base_physical=np.asarray(
                        poses[pose_index].T_base_physical, dtype=float
                    ).copy(),
                )
                for pose_index, reason in representative_failures
            ]

        preferred_branch_by_index: dict[int, str] = {}
        global_max_transition_deg: float | None = None
        if route_mode == "global_alpha_dp":
            max_transition = planning.get(
                "global_alpha_max_joint_delta_deg", 90.0
            )
            if max_transition is not None:
                global_max_transition_deg = float(max_transition)
            optimized, step_costs, total_cost, largest_cost = (
                self._global_alpha_dp_order(
                    safe_robot,
                    nominal_solutions_by_index,
                    weights,
                    max_joint_delta_deg=max_transition,
                )
            )
            order = [index for index, _branch in optimized]
            preferred_branch_by_index = dict(optimized)
            route_note = (
                "global alpha DP selected order/branches "
                f"(total weighted endpoint travel "
                f"{np.degrees(total_cost):.2f} deg, largest weighted edge "
                f"{np.degrees(largest_cost):.2f} deg, "
                f"{len(step_costs)} transitions)"
            )
        else:
            first_index = min(
                reachable_pose_indices,
                key=lambda index: self._greedy_joint_cost(
                    safe_robot,
                    representative_by_index[index],
                    weights,
                ),
            )

            reachable = set(reachable_pose_indices)
            ring = sorted(
                range(len(poses)),
                key=lambda index: int(poses[index].scan_id),
            )
            first_ring_index = ring.index(first_index)

            def circular(direction: int) -> list[int]:
                result: list[int] = []
                for step in range(1, len(ring)):
                    index = ring[
                        (first_ring_index + direction * step) % len(ring)
                    ]
                    if index in reachable:
                        result.append(index)
                return result

            ascending = circular(+1)
            descending = circular(-1)
            first_q = representative_by_index[first_index]

            if ascending and descending:
                asc_cost = self._greedy_joint_cost(
                    first_q,
                    representative_by_index[ascending[0]],
                    weights,
                )
                desc_cost = self._greedy_joint_cost(
                    first_q,
                    representative_by_index[descending[0]],
                    weights,
                )
                remaining = ascending if asc_cost <= desc_cost else descending
                direction_note = (
                    f"ascending selected ({np.degrees(asc_cost):.2f} <= "
                    f"{np.degrees(desc_cost):.2f} deg)"
                    if asc_cost <= desc_cost
                    else
                    f"descending selected ({np.degrees(desc_cost):.2f} < "
                    f"{np.degrees(asc_cost):.2f} deg)"
                )
            else:
                remaining = ascending or descending
                direction_note = "only reachable circular direction selected"

            order = [first_index, *remaining]
            route_note = f"circular route {direction_note}"
        scans: list[PlannedScan] = []
        current_robot = np.asarray(safe_robot, dtype=float).copy()

        for sequence_index, pose_index in enumerate(order, start=1):
            original_pose = poses[pose_index]
            alternate_pose = alternate_poses[pose_index]

            primary_candidates = self._fallback_candidates(
                original_pose=original_pose,
                T_world_plane=T_world_plane,
                T_measurement_physical=T_measurement_physical,
                handeye=handeye,
                sensor_forward_axis=sensor_forward_axis,
            )
            alternate_candidates = self._fallback_candidates(
                original_pose=alternate_pose,
                T_world_plane=T_world_plane,
                T_measurement_physical=T_measurement_physical,
                handeye=handeye,
                sensor_forward_axis=sensor_forward_axis,
            )
            if len(primary_candidates) != len(alternate_candidates):
                raise RuntimeError(
                    "alpha branch fallback candidate counts do not match"
                )

            attempt_notes: list[str] = []
            accepted: PlannedScan | None = None

            # For each identical (tilt, distance) geometry, solve both alpha
            # branches first. Try the branch with less weighted joint travel;
            # if its full path fails, immediately try the opposite branch.
            # Only after both branches fail do we advance distance/tilt fallback.
            for candidate_index, (primary, alternate) in enumerate(
                zip(primary_candidates, alternate_candidates)
            ):
                branch_candidates = []

                for branch_label, candidate in (
                    ("alpha", primary),
                    ("alpha+180", alternate),
                ):
                    approach_physical, approach_sensor, approach_tcp = (
                        approach_transforms(candidate)
                    )

                    ok, ik_reason, q_approach, _ = moveit.solve_pose_ik(
                        names,
                        current_robot,
                        approach_physical,
                        avoid_collisions=True,
                        diagnose_failure=False,
                    )
                    if not ok:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm approach "
                            f"IK: {ik_reason}"
                        )
                        continue

                    if global_max_transition_deg is not None:
                        actual_max_delta_deg = float(
                            np.max(
                                np.abs(
                                    np.degrees(q_approach - current_robot)
                                )
                            )
                        )
                        if actual_max_delta_deg > global_max_transition_deg:
                            attempt_notes.append(
                                f"{branch_label}, tilt "
                                f"{candidate.tilt_deg:.1f} deg, distance "
                                f"{candidate.distance_mm:.1f} mm rejected: "
                                f"max joint transition "
                                f"{actual_max_delta_deg:.2f} deg > "
                                f"{global_max_transition_deg:.2f} deg"
                            )
                            continue

                    branch_cost = self._greedy_joint_cost(
                        current_robot, q_approach, weights
                    )
                    branch_candidates.append(
                        (
                            branch_cost,
                            branch_label,
                            candidate,
                            approach_physical,
                            approach_sensor,
                            approach_tcp,
                            q_approach.copy(),
                        )
                    )

                preferred_branch = preferred_branch_by_index.get(pose_index)
                if preferred_branch is None:
                    branch_candidates.sort(key=lambda item: item[0])
                else:
                    # Preserve the globally selected alpha branch whenever it
                    # is executable.  The opposite branch remains a recovery
                    # path if the selected branch's MoveIt path fails.
                    branch_candidates.sort(
                        key=lambda item: (
                            item[1] != preferred_branch,
                            item[0],
                        )
                    )

                for (
                    branch_cost,
                    branch_label,
                    candidate,
                    approach_physical,
                    approach_sensor,
                    approach_tcp,
                    q_approach,
                ) in branch_candidates:
                    movej_result = moveit.plan_to_joint_positions(
                        names,
                        current_robot,
                        q_approach,
                    )
                    if not movej_result.success:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm "
                            f"RRTConnect: {movej_result.reason}"
                        )
                        continue

                    try:
                        movej = self._timed_moveit_trajectory(
                            movej_result,
                            current_robot,
                            len(names),
                        )
                    except ValueError as exc:
                        attempt_notes.append(
                            f"{branch_label}, invalid RRTConnect trajectory: {exc}"
                        )
                        continue
                    q_approach = movej[0][-1].copy()

                    cart_in_result = moveit.plan_cartesian_to_pose(
                        names,
                        q_approach,
                        np.asarray(candidate.T_base_physical, dtype=float),
                        start_pose_mm=approach_physical,
                        debug_label=(
                            f"scan {candidate.scan_id} APPROACH->SCAN "
                            f"({branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm)"
                        ),
                    )
                    if not cart_in_result.success:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm Cartesian "
                            f"approach failed: {cart_in_result.reason}"
                        )
                        continue

                    try:
                        cart_in = self._timed_moveit_trajectory(
                            cart_in_result,
                            q_approach,
                            len(names),
                        )
                    except ValueError as exc:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm invalid "
                            f"Cartesian approach: {exc}"
                        )
                        continue
                    q_scan = cart_in[0][-1].copy()

                    cart_out_result = moveit.plan_cartesian_to_pose(
                        names,
                        q_scan,
                        approach_physical,
                        start_pose_mm=np.asarray(
                            candidate.T_base_physical, dtype=float
                        ),
                        debug_label=(
                            f"scan {candidate.scan_id} SCAN->APPROACH "
                            f"({branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm)"
                        ),
                    )
                    if not cart_out_result.success:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm Cartesian "
                            f"retract failed: {cart_out_result.reason}"
                        )
                        continue

                    try:
                        cart_out = self._timed_moveit_trajectory(
                            cart_out_result,
                            q_scan,
                            len(names),
                        )
                    except ValueError as exc:
                        attempt_notes.append(
                            f"{branch_label}, tilt {candidate.tilt_deg:.1f} deg, "
                            f"distance {candidate.distance_mm:.1f} mm invalid "
                            f"Cartesian retract: {exc}"
                        )
                        continue

                    fallback_note = ""
                    if candidate_index:
                        fallback_note = (
                            f"; geometry fallback accepted: tilt "
                            f"{original_pose.tilt_deg:.1f}->"
                            f"{candidate.tilt_deg:.1f} deg, distance "
                            f"{original_pose.distance_mm:.1f}->"
                            f"{candidate.distance_mm:.1f} mm after "
                            f"{candidate_index} failed geometry candidate(s)"
                        )

                    branch_recovery_note = ""
                    preferred_branch = preferred_branch_by_index.get(pose_index)
                    if (
                        preferred_branch is not None
                        and branch_label != preferred_branch
                    ):
                        branch_recovery_note = (
                            f"; global branch {preferred_branch} was not "
                            f"executable, recovered with {branch_label}"
                        )

                    reason = (
                        f"scan planning {sequence_index}/{len(order)}; "
                        f"selected {branch_label} branch "
                        f"(approach joint cost "
                        f"{np.degrees(branch_cost):.2f} deg); "
                        f"{route_note}; "
                        "RRTConnect approach + MoveIt Cartesian scan + "
                        f"MoveIt Cartesian retract{fallback_note}"
                        f"{branch_recovery_note}"
                    )

                    combined = np.vstack(
                        [movej[0], cart_in[0][1:], cart_out[0][1:]]
                    )
                    movej_end_s = float(movej[1][-1])
                    cart_in_end_s = float(cart_in[1][-1])
                    combined_times = np.concatenate(
                        [
                            movej[1],
                            movej_end_s + cart_in[1][1:],
                            movej_end_s + cart_in_end_s + cart_out[1][1:],
                        ]
                    )
                    preview = None
                    preview_times = None
                    try:
                        candidate_preview, candidate_preview_times = (
                            self._mujoco_timed_qpos_trajectory(
                                simulation,
                                names,
                                combined,
                                combined_times,
                                safe_full,
                            )
                        )
                        if len(candidate_preview):
                            preview = candidate_preview
                            preview_times = candidate_preview_times
                    except Exception as exc:
                        reason += (
                            "; MuJoCo preview unavailable "
                            f"({type(exc).__name__}: {exc})"
                        )

                    accepted = PlannedScan(
                        pose=candidate,
                        status="SAFE",
                        reason=reason,
                        q_robot_rad=q_scan,
                        path_sample_count=(
                            0 if preview is None else len(preview)
                        ),
                        selected_branch=branch_label,
                        approach_joint_cost_deg=float(
                            np.degrees(branch_cost)
                        ),
                        T_base_sensor_approach=approach_sensor,
                        T_base_tcp_approach=approach_tcp,
                        T_base_physical=np.asarray(
                            candidate.T_base_physical, dtype=float
                        ).copy(),
                        T_base_physical_approach=approach_physical,
                        q_approach_robot_rad=q_approach,
                        preview_qpos_trajectory=preview,
                        preview_time_from_start_s=preview_times,
                        movej_robot_trajectory_rad=movej[0],
                        movej_time_from_start_s=movej[1],
                        movej_velocities_rad_s=movej[2],
                        movej_accelerations_rad_s2=movej[3],
                        lin_to_scan_robot_trajectory_rad=cart_in[0],
                        lin_to_scan_time_from_start_s=cart_in[1],
                        lin_to_scan_velocities_rad_s=cart_in[2],
                        lin_to_scan_accelerations_rad_s2=cart_in[3],
                        lin_to_approach_robot_trajectory_rad=cart_out[0],
                        lin_to_approach_time_from_start_s=cart_out[1],
                        lin_to_approach_velocities_rad_s=cart_out[2],
                        lin_to_approach_accelerations_rad_s2=cart_out[3],
                    )
                    current_robot = cart_out[0][-1].copy()
                    break

                if accepted is not None:
                    break

            if accepted is None:
                scans.append(
                    PlannedScan(
                        pose=original_pose,
                        status="NO_PATH",
                        reason=(
                            "scan planning failed: "
                            + "; ".join(attempt_notes)
                        ),
                        q_robot_rad=None,
                        path_sample_count=0,
                        T_base_physical=np.asarray(
                            original_pose.T_base_physical, dtype=float
                        ).copy(),
                    )
                )
            else:
                scans.append(accepted)

        for pose_index, reason in representative_failures:
            pose = poses[pose_index]
            scans.append(
                PlannedScan(
                    pose=pose,
                    status="NO_PATH",
                    reason="MoveIt approach IK failed: " + reason,
                    q_robot_rad=None,
                    path_sample_count=0,
                    T_base_physical=np.asarray(
                        pose.T_base_physical, dtype=float
                    ).copy(),
                )
            )

        return scans

    def _plan_with_moveit(
        self,
        *,
        poses: list[CalibrationPose],
        alternate_poses: list[CalibrationPose],
        T_world_plane: np.ndarray,
        plane_boundary_uv: np.ndarray,
        simulation: HandEyeSimulation,
        safe_robot: np.ndarray,
        safe_full: np.ndarray,
        names: tuple[str, ...],
        path_step_rad: float,
    ) -> list[PlannedScan]:
        """Apply the MoveIt scene and plan the scan sequence."""
        from .moveit_planner import MoveItServicePlanner

        planning = self.config.values["planning"]
        settings = self._moveit_settings()

        boundary_uv = np.asarray(
            plane_boundary_uv, dtype=float
        ).reshape(-1, 2)
        if len(boundary_uv) < 3 or not np.all(np.isfinite(boundary_uv)):
            raise ValueError(
                "plane_boundary_uv must contain at least 3 finite UV points"
            )

        observed_size_mm = (
            np.max(boundary_uv, axis=0)
            - np.min(boundary_uv, axis=0)
        )
        plane_scale = float(
            planning.get("plane_collision_scale", 2.0)
        )
        if not np.isfinite(plane_scale) or plane_scale <= 0.0:
            raise ValueError(
                "planning.plane_collision_scale must be positive"
            )
        plane_size_xy_mm = observed_size_mm * plane_scale

        with MoveItServicePlanner(settings) as moveit:
            moveit.apply_scene(
                T_world_plane,
                clearance_mm=float(
                    planning.get("plane_keepout_clearance_mm", 10.0)
                ),
                plane_size_xy_mm=plane_size_xy_mm,
            )
            return self._plan_scans_with_moveit(
                moveit=moveit,
                poses=poses,
                alternate_poses=alternate_poses,
                T_world_plane=T_world_plane,
                simulation=simulation,
                safe_robot=safe_robot,
                safe_full=safe_full,
                names=names,
                path_step_rad=path_step_rad,
            )

    @staticmethod
    def _concatenate_timed_trajectories(
        first_positions: np.ndarray,
        first_times: np.ndarray,
        first_velocities: np.ndarray | None,
        first_accelerations: np.ndarray | None,
        second_positions: np.ndarray,
        second_times: np.ndarray,
        second_velocities: np.ndarray | None,
        second_accelerations: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Join RRTConnect-to-APPROACH + Cartesian-to-SCAN into one command."""
        p1 = np.asarray(first_positions, dtype=float)
        t1 = np.asarray(first_times, dtype=float).reshape(-1)
        p2 = np.asarray(second_positions, dtype=float)
        t2 = np.asarray(second_times, dtype=float).reshape(-1)

        if len(p1) < 2 or len(p2) < 2:
            raise ValueError(
                "each trajectory segment must contain at least two waypoints"
            )
        if p1.shape[1:] != p2.shape[1:]:
            raise ValueError("trajectory joint dimensions do not match")
        if np.max(np.abs(p1[-1] - p2[0])) > 1e-5:
            raise ValueError(
                "RRTConnect and Cartesian segments do not share APPROACH"
            )

        positions = np.vstack([p1, p2[1:]])
        times = np.concatenate([t1, t1[-1] + t2[1:]])

        def combine_optional(first, second):
            if first is None or second is None:
                return None
            a = np.asarray(first, dtype=float)
            b = np.asarray(second, dtype=float)
            if a.shape != p1.shape or b.shape != p2.shape:
                return None
            return np.vstack([a, b[1:]])

        velocities = combine_optional(
            first_velocities, second_velocities
        )
        accelerations = combine_optional(
            first_accelerations, second_accelerations
        )

        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "concatenated trajectory timing is not strictly increasing"
            )
        return positions, times, velocities, accelerations

    @classmethod
    def _build_execution_steps(
        cls,
        scans: list[PlannedScan],
    ) -> list[ExecutionStep]:
        """Build the operator route: SCAN, RETRACT, ..."""
        executable = [
            scan
            for scan in scans
            if scan.status in {"SAFE", "SCANNED"}
        ]
        steps: list[ExecutionStep] = []
        previous_label = "START"

        for scan in executable:
            required = (
                scan.movej_robot_trajectory_rad,
                scan.movej_time_from_start_s,
                scan.lin_to_scan_robot_trajectory_rad,
                scan.lin_to_scan_time_from_start_s,
                scan.lin_to_approach_robot_trajectory_rad,
                scan.lin_to_approach_time_from_start_s,
            )
            if any(value is None for value in required):
                continue

            scan_id = int(scan.pose.scan_id)
            incoming = cls._concatenate_timed_trajectories(
                scan.movej_robot_trajectory_rad,
                scan.movej_time_from_start_s,
                scan.movej_velocities_rad_s,
                scan.movej_accelerations_rad_s2,
                scan.lin_to_scan_robot_trajectory_rad,
                scan.lin_to_scan_time_from_start_s,
                scan.lin_to_scan_velocities_rad_s,
                scan.lin_to_scan_accelerations_rad_s2,
            )

            steps.append(
                ExecutionStep(
                    step_id=len(steps) + 1,
                    target_type="SCAN",
                    scan_id=scan_id,
                    route_label=(
                        f"{previous_label} -> APPROACH {scan_id} -> "
                        f"SCAN {scan_id}"
                    ),
                    trajectory_rad=incoming[0],
                    time_from_start_s=incoming[1],
                    velocities_rad_s=incoming[2],
                    accelerations_rad_s2=incoming[3],
                    capture_after=True,
                    reason=(
                        "OMPL RRTConnect to APPROACH + collision-checked "
                        "Cartesian approach; " + scan.reason
                    ),
                )
            )
            steps.append(
                ExecutionStep(
                    step_id=len(steps) + 1,
                    target_type="RETRACT",
                    scan_id=scan_id,
                    route_label=(
                        f"SCAN {scan_id} -> APPROACH {scan_id}"
                    ),
                    trajectory_rad=np.asarray(
                        scan.lin_to_approach_robot_trajectory_rad,
                        dtype=float,
                    ).copy(),
                    time_from_start_s=np.asarray(
                        scan.lin_to_approach_time_from_start_s,
                        dtype=float,
                    ).copy(),
                    velocities_rad_s=None
                    if scan.lin_to_approach_velocities_rad_s is None
                    else np.asarray(
                        scan.lin_to_approach_velocities_rad_s,
                        dtype=float,
                    ).copy(),
                    accelerations_rad_s2=None
                    if scan.lin_to_approach_accelerations_rad_s2 is None
                    else np.asarray(
                        scan.lin_to_approach_accelerations_rad_s2,
                        dtype=float,
                    ).copy(),
                    capture_after=False,
                    reason="collision-checked MoveIt Cartesian retract",
                )
            )
            previous_label = f"APPROACH {scan_id}"

        return steps

    def plan(
        self,
        safe_joint_deg: np.ndarray,
        safe_T_base_tcp_mm: np.ndarray | None = None,
        radius_mm: float | None = None,
    ) -> ScanPlan:
        """Generate the single supported MoveIt scan plan."""
        # Stable TCP is retained for compatibility with SIM/older callers;
        # planning itself is fully determined by the recorded joint state.
        _ = safe_T_base_tcp_mm

        boundary, T_world_plane, metadata = self._load_plane_geometry()
        center = np.asarray(metadata["centroid_w_mm"], dtype=float)
        normal = np.asarray(metadata["normal_w"], dtype=float)

        planning = self.config.values["planning"]
        hull_inradius = centered_convex_inradius(boundary)
        recommended_radius = automatic_scan_radius_mm(
            boundary,
            edge_margin_mm=float(
                planning.get("hull_edge_margin_mm", 10.0)
            ),
            fill_ratio=float(
                planning.get("radius_fill_ratio", 0.90)
            ),
        )

        if radius_mm is None:
            radius_mm = planning.get("radius_mm")
        if radius_mm is None:
            raise ValueError(
                "scan radius is required; enter it in the GUI or set "
                "planning.radius_mm"
            )
        radius = float(radius_mm)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("scan radius must be positive and finite")

        observed_limit = hull_inradius - float(
            planning.get("hull_edge_margin_mm", 10.0)
        )
        radius_warning = None
        if radius > observed_limit:
            radius_warning = (
                f"selected radius {radius:.3f} mm exceeds the observed-hull "
                f"reference limit {observed_limit:.3f} mm"
            )

        handeye = _load_transform(self.config.path("handeye"))
        T_measurement_physical = self.config.T_measurement_physical_mm
        T_physical_measurement = np.linalg.inv(
            T_measurement_physical
        )
        forward_axis = str(
            planning.get("sensor_forward_axis", "+z")
        ).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError(
                "planning.sensor_forward_axis must be '+z' or '-z'"
            )

        tilt_min_deg = float(planning.get("tilt_min_deg", 5.0))
        tilt_max_deg = float(planning.get("tilt_max_deg", 40.0))
        distance_near_mm = float(
            planning.get("distance_near_mm", 60.0)
        )
        distance_far_mm = float(
            planning.get("distance_far_mm", 120.0)
        )

        poses = generate_optimal_tcp_poses(
            board_center_base_mm=center,
            board_normal_base=normal,
            T_tcp_sensor_init=handeye,
            radius_mm=radius,
            tilt_min_deg=tilt_min_deg,
            tilt_max_deg=tilt_max_deg,
            distance_near_mm=distance_near_mm,
            distance_far_mm=distance_far_mm,
            sensor_forward_axis=forward_axis,
            T_physical_measurement=T_physical_measurement,
        )
        alternate_poses = generate_optimal_tcp_poses(
            board_center_base_mm=center,
            board_normal_base=normal,
            T_tcp_sensor_init=handeye,
            radius_mm=radius,
            tilt_min_deg=tilt_min_deg,
            tilt_max_deg=tilt_max_deg,
            distance_near_mm=distance_near_mm,
            distance_far_mm=distance_far_mm,
            sensor_forward_axis=forward_axis,
            T_physical_measurement=T_physical_measurement,
            alpha_branch_offset_deg=180.0,
        )

        board_model = self.config.path("planning_model")
        boundary_uv = np.asarray(boundary, dtype=float).reshape(-1, 2)
        observed_size_mm = (
            np.max(boundary_uv, axis=0)
            - np.min(boundary_uv, axis=0)
        )
        plane_size_xy_mm = observed_size_mm * float(
            planning.get("plane_collision_scale", 2.0)
        )
        moveit_settings = planning.get("moveit", {})

        sensor_frames = self.config.values.get("sensor_frames", {})
        write_model_with_plane_box(
            self.config.path("mujoco_model"),
            board_model,
            T_world_plane_mm=T_world_plane,
            plane_size_xy_mm=plane_size_xy_mm,
            plane_thickness_mm=(
                1e3
                * float(
                    moveit_settings.get("plane_thickness_m", 0.01)
                )
            ),
            keepout_clearance_mm=float(
                planning.get("plane_keepout_clearance_mm", 10.0)
            ),
            contact_margin_mm=0.0,
            sensor_body_name=str(
                sensor_frames.get("body_name", "lj_v7080")
            ),
            physical_sensor_site_name=str(
                sensor_frames.get(
                    "physical_site_name",
                    "sensor_physical_origin",
                )
            ),
            T_measurement_physical_mm=T_measurement_physical,
            sensor_frame_axis_length_mm=float(
                self.config.values.get("visualization", {}).get(
                    "sensor_frame_axis_length_mm", 70.0
                )
            ),
        )

        simulation = HandEyeSimulation(board_model)
        simulation.reset_home()

        names = self.config.joint_names
        safe_robot = np.radians(
            np.asarray(
                safe_joint_deg, dtype=float
            ).reshape(len(names))
        )
        safe_full = simulation.qpos_with_named_joints(
            dict(zip(names, safe_robot)),
            base_qpos=simulation.data.qpos.copy(),
        )

        preview_step_deg = float(
            planning.get(
                "preview_step_deg",
                planning.get("path_step_deg", 1.0),
            )
        )
        if preview_step_deg <= 0.0:
            raise ValueError(
                "planning.preview_step_deg must be positive"
            )

        scans = self._plan_with_moveit(
            poses=poses,
            alternate_poses=alternate_poses,
            T_world_plane=T_world_plane,
            plane_boundary_uv=boundary,
            simulation=simulation,
            safe_robot=safe_robot,
            safe_full=safe_full,
            names=names,
            path_step_rad=np.deg2rad(preview_step_deg),
        )

        plan = ScanPlan(
            radius_mm=radius,
            recommended_radius_mm=recommended_radius,
            hull_inradius_mm=hull_inradius,
            safe_joint_rad=safe_robot,
            joint_names=names,
            board_model_path=board_model,
            scans=scans,
            radius_warning=radius_warning,
            planner_backend="moveit",
            route_mode=str(
                planning.get("route_mode", "circular_greedy")
            ).strip().lower(),
        )
        plan.execution_steps = self._build_execution_steps(scans)
        plan.execution_step_index = 0
        plan.save(self.config.path("scan_plan"))
        return plan


class WorkflowController:
    """Hardware workflow controller for the single MoveIt scan sequence."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.robot = None
        self.laser = None
        self.safe_joint_deg: np.ndarray | None = None
        self.safe_tcp: np.ndarray | None = None
        self.plan: ScanPlan | None = None
        # Kept for SimWorkflowController compatibility; GUI execution uses execution_steps.
        self.active_scan: PlannedScan | None = None
        self.previewed_scan_id: int | None = None
        self.calibration_dataset_override: Path | None = None
        self.driver_launcher = None
        self.motion_verified = False
        self.connection_report = ""
        self.robot_connection_report = ""
        self.laser_connection_report = ""
        # Optional GUI callback. Motion code never depends on this callback;
        # it only reports progress so a failed UI update cannot stop the robot.
        self.motion_progress_callback = None

    def calibration_dataset_path(self) -> Path:
        """Dataset selected for stage 4, independent of stages 2 and 3."""
        if self.calibration_dataset_override is not None:
            return self.calibration_dataset_override
        return self.config.path("scan_dataset")

    def set_calibration_dataset_path(self, dataset: str | Path) -> Path:
        path = Path(dataset).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"calibration dataset directory not found: {path}")
        if not any(path.glob("capture_*.npz")):
            raise RuntimeError(f"no capture_*.npz files found in {path}")
        self.calibration_dataset_override = path
        return path

    def accumulated_scan_path(self) -> Path:
        return self.config.path("calibrated_transform").with_suffix(
            ".accumulated_scans.npz"
        )

    def calibration_diagnostics_path(self) -> Path:
        return self.config.path("calibrated_transform").with_suffix(
            ".diagnostics.json"
        )

    def connect(self) -> None:
        """Connect both devices, preserving the original all-or-nothing API."""
        self.close()
        try:
            self.connect_robot()
            self.connect_laser()
        except BaseException:
            self.close()
            raise

    def connect_robot(self) -> None:
        hardware = self.config.values["equipment"]
        from .ros2_driver_launcher import ROS2DriverLauncher

        if self.robot is not None:
            return
        self.disconnect_robot()
        self.driver_launcher = ROS2DriverLauncher(self.config)
        self.motion_verified = False
        try:
            self.driver_launcher.start(str(hardware["robot_host"]))
            if self.driver_launcher.calibration_path is not None:
                moveit = self.config.values["planning"].setdefault("moveit", {})
                launch_arguments = moveit.setdefault("launch_arguments", {})
                launch_arguments["kinematics_params_file"] = str(
                    self.driver_launcher.calibration_path
                )
            robot_class = _load_object(str(hardware["robot_adapter"]))
            self.robot = robot_class(
                str(hardware["robot_host"]), hardware.get("robot_port")
            )
            adapter_joints = tuple(getattr(self.robot, "JOINT_NAMES", ()))
            if adapter_joints and adapter_joints != self.config.joint_names:
                raise ValueError(
                    "robot adapter joint order does not match equipment.joint_names: "
                    f"adapter={adapter_joints}, config={self.config.joint_names}"
                )
            adapter_base = str(getattr(self.robot, "BASE_FRAME", ""))
            configured_base = str(hardware.get("base_frame", ""))
            if adapter_base and configured_base and adapter_base != configured_base:
                raise ValueError(
                    "robot adapter base TF does not match equipment.base_frame: "
                    f"adapter={adapter_base!r}, config={configured_base!r}"
                )
            driver = hardware.get("driver", {})
            self.robot.connect(
                startup_timeout_s=float(driver.get("startup_timeout_s", 20.0))
            )
            self.driver_launcher.raise_if_exited()
            joints, transform = self._read_robot_state_snapshot()
            self.robot_connection_report = "robot joints/TF/action OK"
            self._update_connection_report()
        except BaseException as error:
            log_path = getattr(self.driver_launcher, "log_path", None)
            self.disconnect_robot()
            if log_path is not None:
                raise RuntimeError(f"{error}\nROS driver log: {log_path}") from error
            raise

    def connect_laser(self) -> None:
        hardware = self.config.values["equipment"]
        if self.laser is not None:
            return
        laser_class = _load_object(str(hardware["laser_adapter"]))
        laser = laser_class(
            ip=str(hardware["laser_ip"]),
            control_port=int(hardware.get("laser_control_port", 24691)),
            high_speed_port=int(hardware.get("laser_high_speed_port", 24692)),
            batch_profiles=int(hardware.get("batch_profiles", 5)),
            aggregate=str(hardware.get("aggregate", "median")),
        )
        try:
            laser.connect()
            profile = np.asarray(
                laser.read_profile(
                    timeout_s=float(hardware.get("connection_profile_timeout_s", 5.0))
                ),
                dtype=float,
            )
            if profile.ndim != 2 or profile.shape[1] != 3 or len(profile) == 0:
                raise RuntimeError(
                    f"Keyence returned an invalid startup profile shape: {profile.shape}"
                )
            self.laser = laser
            self.laser_connection_report = (
                f"laser {hardware['laser_ip']} profile OK ({len(profile)} points)"
            )
            self._update_connection_report()
        except BaseException:
            try:
                laser.close()
            except Exception:
                pass
            raise

    def _update_connection_report(self) -> None:
        reports = [
            report
            for report in (self.robot_connection_report, self.laser_connection_report)
            if report
        ]
        self.connection_report = " · ".join(reports)

    def _read_robot_state_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Read one measured robot state for plotting/FK verification.

        RB5 overrides read_state_snapshot() so joints and TCP come from the
        same controller packet. Other adapters use their own snapshot method
        or fall back to the two standard measured-state calls.
        """
        self._require_robot()
        snapshot = getattr(self.robot, "read_state_snapshot", None)
        if callable(snapshot):
            joints, transform = snapshot()
        else:
            joints = self.robot.read_joint_positions_deg()
            transform = self.robot.read_T_base_tcp()
        joints = np.asarray(joints, dtype=float)
        transform = np.asarray(transform, dtype=float)
        self._validate_robot_state(joints, transform)
        return joints, transform

    @staticmethod
    def _validate_robot_state(joints: np.ndarray, transform: np.ndarray) -> None:
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"invalid six-axis robot state: {joints}")
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise RuntimeError("invalid T_base_tcp returned by robot adapter")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise RuntimeError("T_base_tcp has an invalid homogeneous bottom row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise RuntimeError("T_base_tcp rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise RuntimeError("T_base_tcp rotation determinant is not +1")
        if np.linalg.norm(transform[:3, 3]) > 10000.0:
            raise RuntimeError("T_base_tcp translation is implausible for millimetres")

    def verify_motion_ready(self) -> str:
        """Exercise the real trajectory command path with a zero-displacement hold."""
        self._require_robot()
        self.motion_verified = False
        motion = self.config.values["motion"]
        before_q, before_T = self._read_robot_state_snapshot()
        execute = getattr(self.robot, "execute_joint_trajectory", None)
        if not callable(execute):
            raise RuntimeError(
                "robot adapter does not support complete FollowJointTrajectory execution"
            )
        hold_rad = np.radians(before_q)
        hold_positions = np.vstack([hold_rad, hold_rad])
        hold_times = np.asarray([0.0, 1.0], dtype=float)
        hold_zero = np.zeros_like(hold_positions)
        execute(
            hold_positions,
            hold_times,
            velocities_rad_s=hold_zero,
            accelerations_rad_s2=hold_zero,
            tolerance_deg=min(0.5, float(motion.get("joint_tolerance_deg", 0.5))),
            timeout_s=min(15.0, float(motion.get("timeout_s", 60.0))),
        )
        after_q, after_T = self._read_robot_state_snapshot()
        joint_drift = float(np.max(np.abs(after_q - before_q)))
        tcp_drift = float(np.linalg.norm(after_T[:3, 3] - before_T[:3, 3]))
        rotation_drift = float(
            np.degrees(
                Rotation.from_matrix(before_T[:3, :3].T @ after_T[:3, :3]).magnitude()
            )
        )
        joint_drift_limit = max(
            0.2, 2.0 * float(motion.get("joint_tolerance_deg", 0.1))
        )
        tcp_drift_limit = max(
            0.5, float(motion.get("linear_position_tolerance_mm", 0.5))
        )
        rotation_drift_limit = max(
            0.2, float(motion.get("linear_rotation_tolerance_deg", 0.2))
        )
        if (
            joint_drift > joint_drift_limit
            or tcp_drift > tcp_drift_limit
            or rotation_drift > rotation_drift_limit
        ):
            raise RuntimeError(
                "zero-displacement trajectory verification drifted unexpectedly: "
                f"joint={joint_drift:.3f} deg (limit {joint_drift_limit:.3f}), "
                f"tcp={tcp_drift:.3f} mm (limit {tcp_drift_limit:.3f}), "
                f"rotation={rotation_drift:.3f} deg "
                f"(limit {rotation_drift_limit:.3f})"
            )
        self.motion_verified = True
        return (
            f"trajectory verified ({self.robot.trajectory_action}); "
            f"joint drift {joint_drift:.3f}°, TCP drift {tcp_drift:.3f} mm"
        )

    def close(self) -> None:
        self.disconnect_laser()
        self.disconnect_robot()

    def disconnect_laser(self) -> None:
        if self.laser is not None:
            try:
                self.laser.close()
            except Exception:
                pass
        self.laser = None
        self.laser_connection_report = ""
        self._update_connection_report()

    def disconnect_robot(self) -> None:
        self.motion_verified = False
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception:
                pass
        if self.driver_launcher is not None:
            try:
                self.driver_launcher.close()
            except Exception:
                pass
        self.robot = None
        self.driver_launcher = None
        self.robot_connection_report = ""
        self._update_connection_report()

    def _require_motion_verified(self) -> None:
        if not self.motion_verified:
            raise RuntimeError(
                "real motion interlock is locked; enable real motion and complete "
                "the zero-displacement trajectory verification first"
            )

    def _require_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("robot is not connected")

    def _require_laser(self) -> None:
        if self.laser is None:
            raise RuntimeError("laser is not connected")

    def _require_hardware(self) -> None:
        self._require_robot()
        self._require_laser()

    def record_safe_pose(self) -> None:
        self._require_robot()
        self.safe_joint_deg, self.safe_tcp = self._read_robot_state_snapshot()

    def stable_sensor_pose_mm(self) -> np.ndarray | None:
        """Return the taught stable measurement frame S in base coordinates."""
        if self.safe_tcp is None:
            return None
        return np.asarray(self.safe_tcp, dtype=float) @ _load_transform(
            self.config.path("handeye")
        )

    def recommended_radius_mm(self) -> float:
        return ScanPlanner(self.config).recommended_radius_mm()

    def create_plan(
        self,
        radius_mm: float | None = None,
    ) -> ScanPlan:
        """Create the single supported MoveIt scan plan."""
        if self.safe_joint_deg is None:
            raise RuntimeError("record the manually taught START pose first")
        self.plan = ScanPlanner(self.config).plan(
            self.safe_joint_deg,
            self.safe_tcp,
            radius_mm=radius_mm,
        )
        self.active_scan = None
        self.previewed_scan_id = None
        return self.plan

    def go_to_plan_start(self) -> str:
        """Move the robot from its current state to the plan's recorded START state.

        This is intentionally separate from the execution route.  It is used only
        before step 1 so the first validated trajectory starts from exactly the
        joint state assumed during planning.
        """
        self._require_robot()
        if self.plan is None:
            raise RuntimeError("generate a scan plan first")
        if int(self.plan.execution_step_index) != 0:
            raise RuntimeError(
                "GO TO START is only available before the first execution step"
            )
        if self.plan.planner_backend != "moveit":
            raise RuntimeError("GO TO START currently requires the MoveIt backend")

        target_rad = np.asarray(self.plan.safe_joint_rad, dtype=float).reshape(
            len(self.plan.joint_names)
        )
        current_deg = np.asarray(self.robot.read_joint_positions_deg(), dtype=float)
        current_rad = np.radians(current_deg)

        motion = self.config.values["motion"]
        tolerance_deg = float(motion.get("start_tolerance_deg", 1.0))
        error_deg = self._joint_error_deg(current_deg, np.degrees(target_rad))
        if error_deg <= tolerance_deg:
            return f"Already at START (max joint error {error_deg:.3f} deg)"

        # Plan this recovery from the ACTUAL measured state.  Do not synthesize
        # or reverse an old path: the operator may have moved the robot since
        # the calibration plan was generated.
        from .moveit_planner import MoveItServicePlanner

        planner = ScanPlanner(self.config)
        boundary_uv, T_world_plane, _metadata = planner._load_plane_geometry()
        planning = self.config.values["planning"]
        boundary_uv = np.asarray(boundary_uv, dtype=float).reshape(-1, 2)
        observed_size_mm = np.max(boundary_uv, axis=0) - np.min(
            boundary_uv, axis=0
        )
        plane_scale = float(planning.get("plane_collision_scale", 2.0))
        plane_size_xy_mm = observed_size_mm * plane_scale

        with MoveItServicePlanner(planner._moveit_settings()) as moveit:
            moveit.apply_scene(
                T_world_plane,
                clearance_mm=float(
                    planning.get("plane_keepout_clearance_mm", 10.0)
                ),
                plane_size_xy_mm=plane_size_xy_mm,
            )
            result = moveit.plan_to_joint_positions(
                self.plan.joint_names,
                current_rad,
                target_rad,
            )
            if not result.success:
                raise RuntimeError(
                    "current -> START MoveIt planning failed: "
                    f"{result.reason}"
                )
            (
                positions,
                times,
                velocities,
                accelerations,
            ) = planner._timed_moveit_trajectory(
                result,
                current_rad,
                len(self.plan.joint_names),
            )

        self._execute_moveit_trajectory(
            positions,
            times,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            motion_label="Go to START",
        )

        reached_deg = np.asarray(self.robot.read_joint_positions_deg(), dtype=float)
        end_error_deg = self._joint_error_deg(
            reached_deg, np.degrees(target_rad)
        )
        if end_error_deg > tolerance_deg:
            raise RuntimeError(
                "robot did not reach plan START: "
                f"max joint error={end_error_deg:.3f} deg > "
                f"{tolerance_deg:.3f} deg"
            )

        self.active_scan = None
        self.previewed_scan_id = None
        return f"Reached START (max joint error {end_error_deg:.3f} deg)"

    def _next_execution_step(self) -> ExecutionStep:
        if self.plan is None or not self.plan.execution_steps:
            raise RuntimeError("generate an execution plan first")
        index = int(self.plan.execution_step_index)
        if index >= len(self.plan.execution_steps):
            raise RuntimeError("execution sequence is complete")
        return self.plan.execution_steps[index]

    def preview_next_execution_step(
        self,
    ) -> tuple[ExecutionStep, np.ndarray, np.ndarray]:
        """Preview the exact qpos *and timing* that EXECUTE will use."""
        step = self._next_execution_step()
        simulation = HandEyeSimulation(self.plan.board_model_path)
        simulation.reset_home()
        base_qpos = simulation.data.qpos.copy()
        preview, preview_times = ScanPlanner._mujoco_timed_qpos_trajectory(
            simulation,
            self.plan.joint_names,
            step.trajectory_rad,
            step.time_from_start_s,
            base_qpos,
        )
        if len(preview) == 0:
            raise RuntimeError("next execution-step preview trajectory is empty")
        return step, preview, preview_times

    @staticmethod
    def _concatenate_execution_step_trajectories(
        steps: list[ExecutionStep],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Join the complete planned route without changing its timing.

        Each execution step starts its local clock at zero and shares its first
        joint waypoint with the preceding step's final waypoint.  The duplicate
        boundary sample is removed while the original per-step durations are
        retained.
        """
        if not steps:
            raise RuntimeError("execution plan has no previewable steps")

        positions: list[np.ndarray] = []
        times: list[np.ndarray] = []
        elapsed_s = 0.0
        previous_end: np.ndarray | None = None
        joint_count: int | None = None

        for step in steps:
            q = np.asarray(step.trajectory_rad, dtype=float)
            t = np.asarray(step.time_from_start_s, dtype=float).reshape(-1)
            if q.ndim != 2 or len(q) < 2:
                raise ValueError(
                    f"execution step {step.step_id} must contain at least two "
                    "trajectory samples"
                )
            if joint_count is None:
                joint_count = int(q.shape[1])
            if q.shape[1] != joint_count or t.shape != (len(q),):
                raise ValueError(
                    f"execution step {step.step_id} trajectory shape mismatch"
                )
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)):
                raise ValueError(
                    f"execution step {step.step_id} contains non-finite values"
                )
            local_t = t - float(t[0])
            if np.any(local_t < 0.0) or (
                len(local_t) > 1 and np.any(np.diff(local_t) <= 0.0)
            ):
                raise ValueError(
                    f"execution step {step.step_id} timing is not increasing"
                )

            start = 0
            if previous_end is not None:
                if not np.allclose(previous_end, q[0], atol=1e-5, rtol=0.0):
                    raise ValueError(
                        f"execution step {step.step_id} does not start at the "
                        "previous step's final waypoint"
                    )
                start = 1
            if start < len(q):
                positions.append(q[start:].copy())
                times.append(elapsed_s + local_t[start:])
            elapsed_s += float(local_t[-1])
            previous_end = q[-1]

        combined_q = np.vstack(positions)
        combined_t = np.concatenate(times)
        if len(combined_t) > 1 and np.any(np.diff(combined_t) <= 0.0):
            raise ValueError("full-plan preview timing is not strictly increasing")
        return combined_q, combined_t

    def preview_full_execution_plan(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Build a MuJoCo-only preview of every validated execution step."""
        if self.plan is None or not self.plan.execution_steps:
            raise RuntimeError("generate an execution plan first")
        robot_path, times = self._concatenate_execution_step_trajectories(
            self.plan.execution_steps
        )
        simulation = HandEyeSimulation(self.plan.board_model_path)
        simulation.reset_home()
        base_qpos = simulation.data.qpos.copy()
        preview, preview_times = ScanPlanner._mujoco_timed_qpos_trajectory(
            simulation,
            self.plan.joint_names,
            robot_path,
            times,
            base_qpos,
        )
        if len(preview) == 0:
            raise RuntimeError("full execution-plan preview trajectory is empty")
        return preview, preview_times, len(self.plan.execution_steps)

    def execute_next_execution_step(self) -> Path | None:
        """Execute one operator-visible SCAN or SAFE step."""
        self._require_hardware()
        if self.plan is None:
            raise RuntimeError("scan plan is not available")
        step = self._next_execution_step()
        if self.plan.planner_backend != "moveit":
            raise RuntimeError("step-by-step execution currently requires MoveIt")

        motion = self.config.values["motion"]
        current_rad = np.radians(self.robot.read_joint_positions_deg())
        planned_start = np.asarray(step.trajectory_rad[0], dtype=float)
        start_error_deg = float(np.max(np.abs(np.degrees(current_rad - planned_start))))
        start_limit_deg = float(motion.get("start_tolerance_deg", 1.0))
        if start_error_deg > start_limit_deg:
            raise RuntimeError(
                "robot does not match the next planned step start: "
                f"max joint error={start_error_deg:.3f} deg > {start_limit_deg:.3f} deg"
            )

        self._execute_moveit_trajectory(
            step.trajectory_rad,
            step.time_from_start_s,
            velocities_rad_s=step.velocities_rad_s,
            accelerations_rad_s2=step.accelerations_rad_s2,
            motion_label=f"Execute {step.target_label}",
        )

        expected_deg = np.degrees(step.trajectory_rad[-1])
        current_deg = self.robot.read_joint_positions_deg()
        end_error = self._joint_error_deg(current_deg, expected_deg)
        if end_error > float(motion.get("joint_tolerance_deg", 0.1)):
            raise RuntimeError(
                f"robot did not reach {step.target_label}: max joint error={end_error:.3f} deg"
            )

        captured: Path | None = None
        if step.target_type == "SCAN":
            scan = next(
                (item for item in self.plan.scans if item.pose.scan_id == step.scan_id),
                None,
            )
            if scan is None:
                raise RuntimeError(f"planned scan {step.scan_id} is missing")
            settle_s = float(motion.get("settle_time_s", 1.0))
            self._emit_motion_progress(
                phase="settling", label=f"Settling at scan {step.scan_id}",
                completed=1, total=1, fraction=1.0, elapsed_s=0.0,
                remaining_s=settle_s,
            )
            time.sleep(settle_s)
            self._assert_at_planned_scan_pose(scan)
            from .estimate_initial_plane import capture_once
            self._emit_motion_progress(
                phase="capture", label=f"Capture scan {step.scan_id}",
                completed=1, total=1, fraction=1.0, elapsed_s=0.0,
                remaining_s=None,
            )
            captured = capture_once(
                self.robot, self.laser,
                self._capture_args(self.config.path("scan_dataset")),
            )
            scan.status = "SCANNED"
            scan.captured_path = str(captured)

        step.status = "DONE"
        self.plan.execution_step_index += 1
        self.active_scan = None
        self.previewed_scan_id = None
        self.plan.save(self.config.path("scan_plan"))
        return captured

    def _next_safe_scan(self) -> PlannedScan:
        if self.plan is None:
            raise RuntimeError("generate a scan plan first")
        scan = next((item for item in self.plan.scans if item.status == "SAFE"), None)
        if scan is None:
            raise RuntimeError("no remaining executable scan pose")
        return scan

    def preview_next_scan(self) -> tuple[int, list[np.ndarray]]:
        scan = self._next_safe_scan()
        return self.preview_scan(scan.pose.scan_id, authorize_next=True)

    def preview_scan(
        self, scan_id: int, *, authorize_next: bool = False
    ) -> tuple[int, list[np.ndarray]]:
        if self.plan is None:
            raise RuntimeError("generate a scan plan first")
        scan = next(
            (item for item in self.plan.scans if item.pose.scan_id == int(scan_id)),
            None,
        )
        if scan is None:
            raise KeyError(f"scan {scan_id} is not in the current plan")
        if scan.preview_qpos_trajectory is None:
            raise RuntimeError(
                f"scan {scan_id} has no MuJoCo preview trajectory: {scan.status} — {scan.reason}"
            )
        simulation = HandEyeSimulation(self.plan.board_model_path)
        preview = self.config.values["planning"].get("preview", {})
        frames = simulation.render_trajectory(
            scan.preview_qpos_trajectory,
            width=int(preview.get("width", 640)),
            height=int(preview.get("height", 480)),
            max_frames=int(preview.get("max_frames", 18)),
        )
        if authorize_next:
            next_scan = self._next_safe_scan()
            if next_scan.pose.scan_id == scan.pose.scan_id:
                self.previewed_scan_id = scan.pose.scan_id
        return scan.pose.scan_id, frames

    def view_scan_interactive(self, scan_id: int) -> None:
        if self.plan is None:
            raise RuntimeError("generate a scan plan first")
        scan = next(
            (item for item in self.plan.scans if item.pose.scan_id == int(scan_id)),
            None,
        )
        if scan is None or scan.preview_qpos_trajectory is None:
            raise RuntimeError(f"scan {scan_id} has no MuJoCo preview trajectory")
        simulation = HandEyeSimulation(self.plan.board_model_path)
        simulation.view_trajectory_interactive(scan.preview_qpos_trajectory)

    def read_visualization_sample(self) -> VisualizationSample | None:
        self._require_robot()
        joints, transform = self._read_robot_state_snapshot()
        if self.laser is None:
            return VisualizationSample(-1, np.empty((0, 3)), transform, joints)
        sample = self.laser.read_latest_profile_sample(max_age_s=1.0)
        if sample is None:
            return VisualizationSample(
                -1, np.empty((0, 3)), transform, joints
            )
        sample_id, _received_at, points_s = sample
        return VisualizationSample(
            int(sample_id),
            np.asarray(points_s, dtype=float),
            transform,
            joints,
        )

    def _emit_motion_progress(self, **payload) -> None:
        """Best-effort motion telemetry for the GUI; never part of robot safety."""
        callback = self.motion_progress_callback
        if not callable(callback):
            return
        try:
            callback(dict(payload))
        except BaseException:
            # A UI/reporting failure must never interrupt a robot command path.
            pass

    def _execute_moveit_trajectory(
        self,
        trajectory_rad: np.ndarray,
        time_from_start_s: np.ndarray,
        *,
        velocities_rad_s: np.ndarray | None = None,
        accelerations_rad_s2: np.ndarray | None = None,
        motion_label: str = "Robot motion",
    ) -> None:
        """Send one complete time-parameterized MoveIt trajectory goal."""
        self._require_motion_verified()
        path = np.asarray(trajectory_rad, dtype=float)
        times = np.asarray(time_from_start_s, dtype=float).reshape(-1)
        joint_count = len(self.config.joint_names)
        if path.ndim != 2 or path.shape[1] != joint_count or len(path) == 0:
            raise ValueError("MoveIt trajectory shape does not match configured joints")
        if times.shape != (len(path),):
            raise ValueError("MoveIt trajectory timing does not match its path")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("MoveIt trajectory timing must be strictly increasing")

        total_segments = max(0, len(path) - 1)
        total_duration = float(times[-1])
        motion = self.config.values["motion"]
        configured_timeout = float(motion.get("timeout_s", 60.0))
        # Keep the configured floor, but allow enough room for a valid MoveIt
        # trajectory and controller/action overhead (including speed scaling).
        timeout_s = max(configured_timeout, total_duration * 2.0 + 5.0)

        self._emit_motion_progress(
            phase="moving",
            label=motion_label,
            completed=0,
            total=total_segments,
            fraction=0.0,
            elapsed_s=0.0,
            remaining_s=total_duration,
        )

        def progress(info: dict[str, object]) -> None:
            self._emit_motion_progress(
                phase="moving",
                label=motion_label,
                completed=int(info.get("completed_segments", 0)),
                total=int(info.get("total_segments", total_segments)),
                fraction=float(info.get("fraction", 0.0)),
                elapsed_s=float(info.get("trajectory_time_s", 0.0)),
                remaining_s=info.get("remaining_s"),
            )

        execute = getattr(self.robot, "execute_joint_trajectory", None)
        if not callable(execute):
            raise RuntimeError(
                "robot adapter does not implement full FollowJointTrajectory execution"
            )
        execute(
            path,
            times,
            velocities_rad_s=velocities_rad_s,
            accelerations_rad_s2=accelerations_rad_s2,
            tolerance_deg=float(motion.get("joint_tolerance_deg", 0.1)),
            path_tolerance_deg=float(motion.get("path_tolerance_deg", 1.0)),
            timeout_s=timeout_s,
            progress_callback=progress,
        )

        self._emit_motion_progress(
            phase="moving",
            label=motion_label,
            completed=total_segments,
            total=total_segments,
            fraction=1.0,
            elapsed_s=total_duration,
            remaining_s=0.0,
        )

    def _capture_args(self, dataset_dir: Path):
        from argparse import Namespace

        capture = self.config.values.get("capture", {})
        return Namespace(
            dataset_dir=dataset_dir,
            T_tcp_sensor=_load_transform(self.config.path("handeye")),
            timeout_s=float(capture.get("timeout_s", 3.0)),
            min_points=int(capture.get("min_points", 50)),
            max_abs_sensor_y_mm=float(capture.get("max_abs_sensor_y_mm", 0.1)),
            min_sensor_x_mm=capture.get("min_sensor_x_mm"),
            max_sensor_x_mm=capture.get("max_sensor_x_mm"),
            min_sensor_z_mm=capture.get("min_sensor_z_mm"),
            max_sensor_z_mm=capture.get("max_sensor_z_mm"),
            max_stationarity_translation_mm=float(
                capture.get("max_stationarity_translation_mm", 0.2)
            ),
            max_stationarity_rotation_deg=float(
                capture.get("max_stationarity_rotation_deg", 0.2)
            ),
        )

    def capture_initial_line(self) -> Path:
        self._require_hardware()
        from .estimate_initial_plane import capture_once

        return capture_once(
            self.robot, self.laser, self._capture_args(self.config.path("initial_dataset"))
        )

    def capture_additional_scan(self) -> Path:
        """Capture a manually taught extra calibration line into scan_dataset."""
        self._require_hardware()
        from .estimate_initial_plane import capture_once

        return capture_once(
            self.robot, self.laser, self._capture_args(self.calibration_dataset_path())
        )

    def estimate_plane(self):
        from .estimate_initial_plane import estimate_from_dataset

        args = self._capture_args(self.config.path("initial_dataset"))
        args.estimate_dir = self.config.path("estimate_dir")
        args.normal_hint_world = None
        args.show_plot = False
        args.plot_axis_length_mm = 50.0
        return estimate_from_dataset(args)

    def _assert_at_planned_scan_pose(self, scan: PlannedScan) -> None:
        """Verify that the robot is on the planned SCAN joint branch and TCP pose."""
        if scan.q_robot_rad is None:
            raise RuntimeError(
                f"scan {scan.pose.scan_id} has no MoveIt-planned scan joint state"
            )

        motion = self.config.values["motion"]
        current_joint_deg, current_tcp = self._read_robot_state_snapshot()
        expected_joint_deg = np.degrees(
            np.asarray(scan.q_robot_rad, dtype=float)
        )
        joint_error = self._joint_error_deg(current_joint_deg, expected_joint_deg)
        joint_limit = float(motion.get("joint_tolerance_deg", 0.1))
        if joint_error > joint_limit:
            raise RuntimeError(
                "robot joint branch differs from the MoveIt-planned scan state: "
                f"max joint error={joint_error:.3f} deg > {joint_limit:.3f} deg"
            )

        expected_tcp = scan.pose.T_base_tcp
        position_error = float(
            np.linalg.norm(current_tcp[:3, 3] - expected_tcp[:3, 3])
        )
        relative = current_tcp[:3, :3].T @ expected_tcp[:3, :3]
        rotation_error = float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
                )
            )
        )
        position_limit = float(motion.get("linear_position_tolerance_mm", 1.0))
        rotation_limit = float(motion.get("linear_rotation_tolerance_deg", 1.0))
        if position_error > position_limit or rotation_error > rotation_limit:
            raise RuntimeError(
                "robot did not reach the requested planned scan TCP pose: "
                f"position={position_error:.3f} mm (limit {position_limit:.3f}), "
                f"rotation={rotation_error:.3f} deg (limit {rotation_limit:.3f})"
            )

    @staticmethod
    def _joint_error_deg(current: np.ndarray, target: np.ndarray) -> float:
        # The six RB5 planning joints are bounded revolute joints (URDF limits
        # approximately [-pi, pi]), not continuous joints. A branch on the
        # opposite side of a limit must not compare equal after an angle wrap.
        difference = np.asarray(current, dtype=float) - np.asarray(target, dtype=float)
        return float(np.max(np.abs(difference)))

    def calibrate(self) -> np.ndarray:
        if (
            self.plan is not None
            and self.plan.execution_steps
            and self.plan.execution_step_index < len(self.plan.execution_steps)
            and self.calibration_dataset_override is None
        ):
            remaining_steps = [
                step.target_label
                for step in self.plan.execution_steps[self.plan.execution_step_index :]
            ]
            raise RuntimeError(
                "execution route is not complete: " + " -> ".join(remaining_steps)
            )
        if self.plan is not None and self.calibration_dataset_override is None:
            remaining = [item.pose.scan_id for item in self.plan.scans if item.status == "SAFE"]
            if remaining:
                raise RuntimeError(f"collision-free scans are not complete: {remaining}")
        from argparse import Namespace
        from .main import calibrate

        calibration = self.config.values.get("calibration", {})
        args = Namespace(
            dataset_dir=self.calibration_dataset_path(),
            initial_transform=self.config.path("handeye"),
            output=self.config.path("calibrated_transform"),
            min_scans=int(calibration.get("min_scans", 4)),
            max_iter=int(calibration.get("max_iter", 3000)),
            tol=float(calibration.get("tol", 1e-9)),
            max_condition=float(calibration.get("max_condition", 1e6)),
            max_final_plane_rms_mm=float(
                calibration.get("max_final_plane_rms_mm", 2.0)
            ),
            disable_profile_ransac=bool(
                calibration.get("disable_profile_ransac", False)
            ),
            ransac_reject_policy=str(calibration.get("ransac_reject_policy", "skip")),
            ransac_threshold_mm=float(calibration.get("ransac_threshold_mm", 0.15)),
            ransac_max_iterations=int(
                calibration.get("ransac_max_iterations", 1000)
            ),
            ransac_min_inliers=int(calibration.get("ransac_min_inliers", 20)),
            ransac_min_inlier_ratio=float(
                calibration.get("ransac_min_inlier_ratio", 0.65)
            ),
            max_ransac_skip_ratio=float(
                calibration.get("max_ransac_skip_ratio", 0.3)
            ),
            ransac_seed=int(calibration.get("ransac_seed", 1701)),
            ransac_refine_iterations=int(
                calibration.get("ransac_refine_iterations", 3)
            ),
        )
        return calibrate(args)
