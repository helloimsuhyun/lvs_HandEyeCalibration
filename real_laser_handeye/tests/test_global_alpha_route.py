from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from real_laser_handeye.workflow import ScanPlanner, WorkflowConfig


def test_global_alpha_dp_selects_order_and_branch_jointly():
    route, step_costs, total, largest = ScanPlanner._global_alpha_dp_order(
        np.array([0.0]),
        {
            0: {"alpha": np.array([1.0]), "alpha+180": np.array([10.0])},
            1: {"alpha": np.array([9.0]), "alpha+180": np.array([2.0])},
            2: {"alpha": np.array([3.0]), "alpha+180": np.array([8.0])},
        },
        np.array([1.0]),
        max_joint_delta_deg=None,
    )

    assert route == [
        (0, "alpha"),
        (1, "alpha+180"),
        (2, "alpha"),
    ]
    np.testing.assert_allclose(step_costs, [1.0, 1.0, 1.0])
    assert np.isclose(total, 3.0)
    assert np.isclose(largest, 1.0)


def test_global_alpha_dp_rejects_a_branch_over_the_per_joint_limit():
    route, _step_costs, _total, _largest = (
        ScanPlanner._global_alpha_dp_order(
            np.radians([0.0]),
            {
                0: {
                    "alpha": np.radians([100.0]),
                    "alpha+180": np.radians([10.0]),
                },
                1: {
                    "alpha": np.radians([110.0]),
                    "alpha+180": np.radians([20.0]),
                },
            },
            np.array([1.0]),
            max_joint_delta_deg=30.0,
        )
    )

    assert route == [(0, "alpha+180"), (1, "alpha+180")]


def test_global_alpha_dp_reports_when_limit_disconnects_all_routes():
    with pytest.raises(ValueError, match="could not connect all reachable scans"):
        ScanPlanner._global_alpha_dp_order(
            np.radians([0.0]),
            {
                0: {
                    "alpha": np.radians([1.0]),
                    "alpha+180": np.radians([2.0]),
                },
                1: {
                    "alpha": np.radians([100.0]),
                    "alpha+180": np.radians([101.0]),
                },
            },
            np.array([1.0]),
            max_joint_delta_deg=30.0,
        )


def test_robot_configs_preserve_existing_route_mode_by_default():
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    for filename in (
        "rb5_ljv7080_workflow.yaml",
        "ur5e_ljv7080_workflow.yaml",
        "sim_workflow.yaml",
        "ur5e_sim_workflow.yaml",
    ):
        planning = WorkflowConfig.load(config_dir / filename).values["planning"]
        assert planning["route_mode"] == "circular_greedy"
        assert planning["global_alpha_max_joint_delta_deg"] == 90.0
