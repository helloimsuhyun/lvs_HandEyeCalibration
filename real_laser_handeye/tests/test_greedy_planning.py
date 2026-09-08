from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real_laser_handeye.moveit_planner import MoveItPath
from real_laser_handeye.workflow import ScanPlanner


class _FakeMoveIt:
    def solve_pose_ik(
        self,
        names,
        seed,
        target,
        *,
        avoid_collisions=True,
        diagnose_failure=False,
    ):
        del seed, avoid_collisions, diagnose_failure
        goal = float(target[0, 3])
        return True, "MoveIt IK succeeded", np.array([goal]), 1

    def plan_to_joint_positions(self, names, start, goal):
        positions = np.vstack(
            [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
        )
        return MoveItPath(
            success=True,
            reason="planned",
            joint_names=tuple(names),
            positions_rad=positions,
            time_from_start_s=np.array([0.0, 1.0]),
        )

    def plan_to_pose(self, names, start, target):
        goal = np.array([float(target[0, 3])])
        return self.plan_to_joint_positions(names, start, goal)

    def plan_pilz_lin_to_pose(self, names, start, target):
        goal = np.asarray(start, dtype=float).copy()
        goal[0] = float(target[0, 3])
        return self.plan_to_joint_positions(names, start, goal)


def _pose(scan_id: int, joint_goal: float):
    target = np.eye(4)
    target[0, 3] = joint_goal
    return SimpleNamespace(scan_id=scan_id, T_base_physical=target)


def _pose_with_geometry(
    scan_id: int,
    joint_goal: float,
    tilt_deg: float,
    distance_mm: float,
):
    pose = _pose(scan_id, joint_goal)
    pose.tilt_deg = float(tilt_deg)
    pose.distance_mm = float(distance_mm)
    return pose


def _pose_2d(scan_id: int, joint_goal: tuple[float, float]):
    target = np.eye(4)
    target[:2, 3] = joint_goal
    return SimpleNamespace(scan_id=scan_id, T_base_physical=target)


def test_greedy_joint_cost_rejects_invalid_weights():
    with np.testing.assert_raises(ValueError):
        ScanPlanner._greedy_joint_cost(
            np.zeros(2), np.ones(2), np.array([1.0, 0.0])
        )


def test_held_karp_open_order_is_exact_and_excludes_return_to_safe():
    safe = np.zeros(2)
    goals = [
        np.array([9.0, -10.0]),
        np.array([4.0, -2.0]),
        np.array([-3.0, 8.0]),
        np.array([-7.0, 0.0]),
    ]
    order, step_costs, total_cost = ScanPlanner._held_karp_open_order(
        safe, goals, np.ones(2)
    )

    assert order == (2, 3, 1, 0)
    assert np.isclose(sum(step_costs), total_cost)
    assert np.isclose(total_cost, 49.0)


def test_optimized_industrial_mode_retries_larger_distance_after_lin_failure(
    monkeypatch,
):
    original = _pose_with_geometry(0, 1.0, 10.0, 65.0)
    fallback = _pose_with_geometry(0, 2.0, 10.0, 70.0)
    for pose in (original, fallback):
        pose.T_base_sensor = pose.T_base_physical.copy()
        pose.T_base_tcp = pose.T_base_physical.copy()

    class LinFallbackMoveIt(_FakeMoveIt):
        def plan_pilz_lin_to_pose(self, names, start, target):
            if np.isclose(target[0, 3], 1.0):
                return MoveItPath(
                    False, "LIN collision", tuple(names), np.empty((0, 1))
                )
            return super().plan_pilz_lin_to_pose(names, start, target)

    planner = object.__new__(ScanPlanner)
    planner.config = SimpleNamespace(
        values={
            "planning": {"greedy_joint_weights": [1.0]},
            "motion": {"approach_offset_mm": 80.0},
        },
        T_measurement_physical_mm=np.eye(4),
        path=lambda _name: Path("unused.json"),
    )
    monkeypatch.setattr(
        "real_laser_handeye.workflow._load_transform", lambda _path: np.eye(4)
    )
    monkeypatch.setattr(
        ScanPlanner,
        "_mujoco_qpos_trajectory",
        staticmethod(lambda *args, **kwargs: np.zeros((2, 1))),
    )

    scans = planner._plan_optimized_with_moveit(
        moveit=LinFallbackMoveIt(),
        poses=[original],
        T_world_plane=np.eye(4),
        distance_fallback_poses_by_scan_id={0: (fallback,)},
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        names=("joint",),
        path_step_rad=0.1,
    )

    assert scans[0].status == "SAFE"
    assert scans[0].pose.distance_mm == 70.0
    assert "distance fallback accepted" in scans[0].reason
    steps = ScanPlanner._build_execution_steps(scans, "optimized")
    assert [step.target_type for step in steps] == [
        "APPROACH", "SCAN", "RETRACT", "SAFE"
    ]


def _adaptive_planner():
    planner = object.__new__(ScanPlanner)
    planner.config = SimpleNamespace(
        values={
            "planning": {
                "greedy_joint_weights": [1.0],
                "adaptive_trigger_max_joint_deg": 60.0,
            }
        }
    )
    return planner


def test_adaptive_mode_preserves_selected_route_in_execution_steps(monkeypatch):
    planner = _adaptive_planner()
    monkeypatch.setattr(
        ScanPlanner,
        "_mujoco_qpos_trajectory",
        staticmethod(lambda *args, **kwargs: np.zeros((2, 1))),
    )

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=_FakeMoveIt(),
        poses=[_pose(0, 1.0), _pose(1, 0.2), _pose(2, 0.5)],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
    )

    assert [scan.pose.scan_id for scan in scans] == [1, 2, 0]
    steps = ScanPlanner._build_execution_steps(scans, "adaptive_safe_return")
    assert [step.scan_id for step in steps] == [1, 2, 0]
    assert all(step.target_type == "SCAN" for step in steps)
    np.testing.assert_allclose(steps[0].trajectory_rad[0], [0.0])
    for previous, following in zip(steps, steps[1:]):
        np.testing.assert_allclose(
            previous.trajectory_rad[-1], following.trajectory_rad[0]
        )


def test_adaptive_mode_retains_direct_path_when_dynamic_bridge_is_larger(monkeypatch):
    planner = _adaptive_planner()
    planner.config.values["planning"]["adaptive_trigger_max_joint_deg"] = 20.0
    monkeypatch.setattr(
        ScanPlanner,
        "_mujoco_qpos_trajectory",
        staticmethod(lambda *args, **kwargs: np.zeros((2, 1))),
    )

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=_FakeMoveIt(),
        poses=[_pose(0, 1.0), _pose(1, 1.5)],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
    )

    assert [scan.pose.scan_id for scan in scans] == [0, 1]
    assert "direct RRT retained" in scans[1].reason
    assert "28.65 deg <= dynamic-safe 143.24 deg" in scans[1].reason
    assert scans[1].incoming_safe_split_index is None


def test_adaptive_mode_selects_dynamic_bridge_when_actual_travel_is_smaller(
    monkeypatch,
):
    class SeedSensitiveMoveIt(_FakeMoveIt):
        def solve_pose_ik(
            self,
            names,
            seed,
            target,
            *,
            avoid_collisions=True,
            diagnose_failure=False,
        ):
            del names, avoid_collisions, diagnose_failure
            target_x = float(target[0, 3])
            seed_value = float(np.asarray(seed)[0])
            if np.isclose(target_x, 1.0):
                goal = 0.1
            elif np.isclose(target_x, 2.0):
                goal = 3.0 if seed_value > 0.05 else 1.0
            else:
                goal = 0.0
            return True, "MoveIt IK succeeded", np.array([goal]), 1

    planner = _adaptive_planner()
    planner.config.values["planning"]["adaptive_trigger_max_joint_deg"] = 20.0
    monkeypatch.setattr(
        ScanPlanner,
        "_mujoco_qpos_trajectory",
        staticmethod(lambda *args, **kwargs: np.zeros((2, 1))),
    )

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=SeedSensitiveMoveIt(),
        poses=[_pose(0, 1.0), _pose(1, 2.0)],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
    )

    assert [scan.pose.scan_id for scan in scans] == [0, 1]
    assert "adaptive bridge selected" in scans[1].reason
    assert "dynamic 63.03 deg < direct 166.16 deg" in scans[1].reason
    assert scans[1].incoming_safe_split_index is not None


def test_adaptive_mode_retries_plane_sensor_ik_collision_at_larger_distance(
    monkeypatch,
):
    class PlaneCollisionMoveIt(_FakeMoveIt):
        def __init__(self):
            self.ik_targets = []

        def solve_pose_ik(
            self,
            names,
            seed,
            target,
            *,
            avoid_collisions=True,
            diagnose_failure=False,
        ):
            del names, seed, avoid_collisions, diagnose_failure
            target_x = float(target[0, 3])
            self.ik_targets.append(target_x)
            if np.isclose(target_x, 1.0):
                return (
                    False,
                    "MoveIt IK failed (error -31); collision-off IK succeeds -> "
                    "collision-aware IK rejected candidate; MoveIt collision: "
                    "calibration_plane <-> sensor_measurement_frame",
                    np.empty(0),
                    -31,
                )
            if np.isclose(target_x, 0.8):
                return False, "MoveIt IK failed (error -31)", np.empty(0), -31
            return True, "MoveIt IK succeeded", np.array([target_x]), 1

    planner = _adaptive_planner()
    monkeypatch.setattr(
        ScanPlanner,
        "_mujoco_qpos_trajectory",
        staticmethod(lambda *args, **kwargs: np.zeros((2, 1))),
    )
    primary = _pose_with_geometry(10, 1.0, 30.0, 75.0)
    fallback_80 = _pose_with_geometry(10, 0.8, 30.0, 80.0)
    fallback_85 = _pose_with_geometry(10, 0.9, 30.0, 85.0)
    moveit = PlaneCollisionMoveIt()

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=moveit,
        poses=[primary],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
        distance_fallback_poses_by_scan_id={10: (fallback_80, fallback_85)},
    )

    assert moveit.ik_targets == [1.0, 0.8, 0.9]
    assert scans[0].status == "SAFE"
    assert scans[0].pose is fallback_85
    assert scans[0].pose.tilt_deg == primary.tilt_deg
    assert scans[0].pose.distance_mm == 85.0
    assert "plane-collision distance fallback 75.00->85.00 mm" in scans[0].reason
    assert "80.00 mm=FAIL" in scans[0].reason
    assert "85.00 mm=OK" in scans[0].reason


def test_adaptive_mode_does_not_change_distance_for_unrelated_ik_failure():
    class GenericIKFailureMoveIt(_FakeMoveIt):
        def __init__(self):
            self.ik_calls = 0

        def solve_pose_ik(self, *args, **kwargs):
            self.ik_calls += 1
            return False, "MoveIt IK failed (error -31)", np.empty(0), -31

    planner = _adaptive_planner()
    primary = _pose_with_geometry(10, 1.0, 30.0, 75.0)
    fallback = _pose_with_geometry(10, 0.8, 30.0, 80.0)
    moveit = GenericIKFailureMoveIt()

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=moveit,
        poses=[primary],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
        distance_fallback_poses_by_scan_id={10: (fallback,)},
    )

    assert moveit.ik_calls == 1
    assert scans[0].status == "NO_PATH"
    assert scans[0].pose is primary


def test_adaptive_mode_reports_every_failed_distance_fallback_attempt():
    class AlwaysPlaneCollisionMoveIt(_FakeMoveIt):
        def solve_pose_ik(self, *args, **kwargs):
            return (
                False,
                "MoveIt IK failed (error -31); collision-off IK succeeds -> "
                "collision-aware IK rejected candidate; MoveIt collision: "
                "calibration_plane <-> sensor_measurement_frame",
                np.empty(0),
                -31,
            )

    planner = _adaptive_planner()
    primary = _pose_with_geometry(10, 1.0, 30.0, 75.0)
    fallback_80 = _pose_with_geometry(10, 0.8, 30.0, 80.0)
    fallback_85 = _pose_with_geometry(10, 0.9, 30.0, 85.0)

    scans = planner._plan_adaptive_safe_return_with_moveit(
        moveit=AlwaysPlaneCollisionMoveIt(),
        poses=[primary],
        simulation=object(),
        safe_robot=np.array([0.0]),
        safe_full=np.array([0.0]),
        safe_T_base_physical=np.eye(4),
        names=("joint",),
        path_step_rad=0.1,
        distance_fallback_poses_by_scan_id={10: (fallback_80, fallback_85)},
    )

    assert scans[0].status == "NO_PATH"
    assert "distance fallback attempts:" in scans[0].reason
    assert "80.00 mm=FAIL" in scans[0].reason
    assert "85.00 mm=FAIL" in scans[0].reason


def test_rrt_quality_gate_retries_excessive_detour():
    class DetourThenDirectMoveIt(_FakeMoveIt):
        def __init__(self):
            self.calls = 0

        def plan_to_joint_positions(self, names, start, goal):
            self.calls += 1
            start = np.asarray(start, dtype=float)
            goal = np.asarray(goal, dtype=float)
            if self.calls == 1:
                positions = np.vstack([start, np.array([3.0]), goal])
            else:
                positions = np.vstack([start, goal])
            return MoveItPath(
                success=True,
                reason="planned",
                joint_names=tuple(names),
                positions_rad=positions,
                time_from_start_s=np.arange(len(positions), dtype=float),
            )

    moveit = DetourThenDirectMoveIt()
    trajectory, cost, ratio, detail = (
        ScanPlanner._plan_joint_trajectory_with_retry(
            moveit=moveit,
            names=("joint",),
            start_rad=np.array([0.0]),
            goal_rad=np.array([1.0]),
            weights=np.array([1.0]),
            max_attempts=3,
            max_detour_ratio=1.5,
        )
    )

    assert moveit.calls == 2
    assert trajectory is not None
    np.testing.assert_allclose(trajectory[0][:, 0], [0.0, 1.0])
    assert np.isclose(cost, 1.0)
    assert np.isclose(ratio, 1.0)
    assert "attempt 2/3" in detail


def test_rrt_quality_gate_rejects_all_excessive_attempts():
    class AlwaysDetouringMoveIt(_FakeMoveIt):
        def __init__(self):
            self.calls = 0

        def plan_to_joint_positions(self, names, start, goal):
            self.calls += 1
            positions = np.vstack(
                [np.asarray(start, dtype=float), np.array([3.0]), goal]
            )
            return MoveItPath(
                success=True,
                reason="planned",
                joint_names=tuple(names),
                positions_rad=positions,
                time_from_start_s=np.arange(len(positions), dtype=float),
            )

    moveit = AlwaysDetouringMoveIt()
    trajectory, cost, ratio, detail = (
        ScanPlanner._plan_joint_trajectory_with_retry(
            moveit=moveit,
            names=("joint",),
            start_rad=np.array([0.0]),
            goal_rad=np.array([1.0]),
            weights=np.array([1.0]),
            max_attempts=3,
            max_detour_ratio=1.5,
        )
    )

    assert moveit.calls == 3
    assert trajectory is None
    assert np.isinf(cost)
    assert np.isinf(ratio)
    assert "best rejected detour 5.000" in detail
