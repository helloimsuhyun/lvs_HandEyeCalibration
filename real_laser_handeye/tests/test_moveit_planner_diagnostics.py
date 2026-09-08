from __future__ import annotations

import numpy as np

from real_laser_handeye.moveit_planner import MoveItServicePlanner


def test_error_minus_31_retries_once_with_longer_timeout():
    planner = object.__new__(MoveItServicePlanner)
    planner.settings = {"ik_timeout_s": 0.25, "ik_retry_timeout_s": 1.0}
    timeouts = []

    def fake_compute(*args, timeout_s=None, **kwargs):
        timeouts.append(timeout_s)
        if timeout_s is None:
            return False, "MoveIt IK failed (error -31)", np.empty(0), -31
        return True, "MoveIt IK succeeded", np.array([0.2]), 1

    planner._compute_ik = fake_compute
    ok, reason, solution, error_code = planner.solve_pose_ik(
        ("j1",), np.zeros(1), np.eye(4)
    )

    assert ok is True
    assert reason == "MoveIt IK succeeded"
    assert error_code == 1
    np.testing.assert_allclose(solution, [0.2])
    assert timeouts == [None, 1.0]


def test_solve_pose_ik_can_opt_in_to_collision_failure_diagnostics():
    planner = object.__new__(MoveItServicePlanner)
    planner._compute_ik = lambda *args, **kwargs: (
        False,
        "MoveIt IK failed (error -31)",
        np.empty((0,), dtype=float),
        -31,
    )
    planner._diagnose_collision_aware_ik_failure = lambda *args: (
        "MoveIt IK failed (error -31); MoveIt collision: "
        "calibration_plane <-> sensor_measurement_frame"
    )

    ok, reason, solution, error_code = planner.solve_pose_ik(
        ("j1",),
        np.zeros(1),
        np.eye(4),
        avoid_collisions=True,
        diagnose_failure=True,
    )

    assert ok is False
    assert error_code == -31
    assert solution.size == 0
    assert "calibration_plane <-> sensor_measurement_frame" in reason


def test_solve_pose_ik_skips_diagnostics_by_default():
    planner = object.__new__(MoveItServicePlanner)
    planner._compute_ik = lambda *args, **kwargs: (
        False,
        "MoveIt IK failed (error -31)",
        np.empty((0,), dtype=float),
        -31,
    )
    planner._diagnose_collision_aware_ik_failure = lambda *args: (_ for _ in ()).throw(
        AssertionError("diagnostics should be opt-in")
    )

    ok, reason, _solution, error_code = planner.solve_pose_ik(
        ("j1",),
        np.zeros(1),
        np.eye(4),
    )

    assert ok is False
    assert error_code == -31
    assert reason == "MoveIt IK failed (error -31)"
