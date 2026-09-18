from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from real_laser_handeye.calibrate_only.with_ransac_calibrate_only import load_scans
from real_laser_handeye.main import atomic_accumulated_scans
from real_laser_handeye.workflow import WorkflowConfig, WorkflowController


def test_atomic_accumulated_scans_preserves_variable_profiles(tmp_path):
    scans = [
        SimpleNamespace(
            T_base_ef=np.eye(4),
            points_s=np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]),
            scan_id=4,
        ),
        SimpleNamespace(
            T_base_ef=np.eye(4),
            points_s=np.array([[5.0, 0.0, 6.0]]),
            scan_id=9,
        ),
    ]
    output = tmp_path / "accepted.accumulated_scans.npz"
    atomic_accumulated_scans(output, scans)

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["profile_offsets"], [0, 2, 3])
        np.testing.assert_array_equal(data["scan_ids"], [4, 9])
        assert data["points_s"].shape == (3, 3)
        assert data["T_base_tcp"].shape == (2, 4, 4)


def test_stage4_loader_accepts_workflow_world_tcp_capture(tmp_path):
    pose = np.eye(4)
    pose[:3, 3] = [100.0, 200.0, 300.0]
    points = np.column_stack(
        [np.linspace(-10.0, 10.0, 21), np.zeros(21), np.zeros(21)]
    )
    np.savez_compressed(
        tmp_path / "capture_0001.npz",
        T_world_tcp=pose,
        points_s=points,
    )

    scans, diagnostics = load_scans(
        tmp_path,
        use_profile_ransac=False,
        ransac_threshold_mm=0.15,
        ransac_max_iterations=10,
        ransac_min_inliers=2,
        ransac_min_inlier_ratio=0.5,
        ransac_seed=1,
        ransac_refine_iterations=1,
        ransac_reject_policy="error",
    )

    assert len(scans) == len(diagnostics) == 1
    np.testing.assert_allclose(scans[0].T_base_ef, pose)
    np.testing.assert_allclose(scans[0].points_s, points)


def test_stage4_manual_dataset_calibrates_without_scan_plan(tmp_path, monkeypatch):
    dataset = tmp_path / "manual_dataset"
    dataset.mkdir()
    np.savez_compressed(
        dataset / "capture_0001.npz",
        T_base_tcp=np.eye(4),
        points_s=np.zeros((2, 3)),
    )
    initial = tmp_path / "initial.csv"
    np.savetxt(initial, np.eye(4), delimiter=",")
    output = tmp_path / "final.csv"
    config = WorkflowConfig(
        tmp_path / "workflow.yaml",
        {
            "paths": {
                "scan_dataset": str(tmp_path / "unused"),
                "handeye": str(initial),
                "calibrated_transform": str(output),
            },
            "equipment": {"joint_names": ["j1"]},
            "planning": {},
            "motion": {},
            "calibration": {},
        },
    )
    controller = WorkflowController(config)
    controller.set_calibration_dataset_path(dataset)

    def fake_calibrate(args):
        assert args.dataset_dir == dataset.resolve()
        return np.eye(4)

    monkeypatch.setattr("real_laser_handeye.main.calibrate", fake_calibrate)
    result = controller.calibrate()
    np.testing.assert_allclose(result, np.eye(4))
    assert controller.plan is None
