from __future__ import annotations

import sys

import numpy as np
import pytest

from real_laser_handeye.workflow import ExecutionStep, WorkflowController
from real_laser_handeye.workflow_mujoco_viewer import (
    _scaled_playback_time,
    parse_args,
)


def _step(step_id: int, qpos: list[list[float]], times: list[float]):
    return ExecutionStep(
        step_id=step_id,
        target_type="SCAN",
        scan_id=step_id,
        route_label=f"step {step_id}",
        trajectory_rad=np.asarray(qpos, dtype=float),
        time_from_start_s=np.asarray(times, dtype=float),
    )


def test_full_plan_preview_joins_steps_and_preserves_each_duration():
    steps = [
        _step(1, [[0.0, 0.0], [1.0, 2.0]], [0.0, 2.0]),
        _step(2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.0, 1.0, 3.0]),
        _step(3, [[5.0, 6.0], [7.0, 8.0]], [2.0, 6.0]),
    ]

    qpos, times = WorkflowController._concatenate_execution_step_trajectories(
        steps
    )

    np.testing.assert_allclose(
        qpos,
        [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    )
    np.testing.assert_allclose(times, [0.0, 2.0, 3.0, 5.0, 9.0])


def test_full_plan_preview_rejects_disconnected_steps():
    steps = [
        _step(1, [[0.0], [1.0]], [0.0, 1.0]),
        _step(2, [[2.0], [3.0]], [0.0, 1.0]),
    ]

    with pytest.raises(ValueError, match="previous step"):
        WorkflowController._concatenate_execution_step_trajectories(steps)


def test_viewer_accepts_preview_only_playback_speed(monkeypatch, tmp_path):
    model = tmp_path / "model.xml"
    trajectory = tmp_path / "trajectory.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow_mujoco_viewer",
            "--model",
            str(model),
            "--trajectory",
            str(trajectory),
            "--playback-speed",
            "5",
        ],
    )

    args = parse_args()

    assert args.playback_speed == 5.0


def test_preview_speed_scales_wall_time_without_changing_trajectory_duration():
    assert _scaled_playback_time(1.25, 10.0, 2.0) == 2.5
    assert _scaled_playback_time(1.25, 10.0, 5.0) == 6.25
    assert _scaled_playback_time(3.0, 10.0, 5.0) == 10.0
