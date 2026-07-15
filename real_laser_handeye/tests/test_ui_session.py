from __future__ import annotations

import copy
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_laser_handeye.hardware import (
    LaserInterface,
    MockPlanarLaser,
    MockRobot,
    ProfileSample,
)
from real_laser_handeye.planning import build_single_plane_plan
from real_laser_handeye.profile_broker import ProfileBroker
from real_laser_handeye.safety import AxisAlignedBox, SafetyConfig
from real_laser_handeye.ui_model import build_dashboard_snapshot
from real_laser_handeye.workflow import (
    CaptureConfig,
    MotionCancelled,
    ScanCaptureSession,
    finalize_bootstrap_plane,
    validate_bootstrap_boundary_quality,
)


def make_scene():
    T_plane = np.eye(4)
    T_plane[:3, 3] = [50.0, -20.0, 400.0]
    boundary = {
        "plane_frame": {"T_base_plane": T_plane.tolist()},
        "safe_bounds_uv_mm": {
            "u_min": -150.0,
            "u_max": 150.0,
            "v_min": -150.0,
            "v_max": 150.0,
        },
    }
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler(
        "xyz", [4.0, -7.0, 12.0], degrees=True
    ).as_matrix()
    T_true[:3, 3] = [35.0, -12.0, 70.0]
    T_init = T_true.copy()
    T_init[:3, 3] += [1.0, -1.0, 2.0]
    transit = np.eye(4)
    transit[:3, 3] = [0.0, 0.0, 750.0]
    safety = SafetyConfig(
        workspace=AxisAlignedBox(
            np.array([-800.0, -800.0, -100.0]),
            np.array([800.0, 800.0, 1000.0]),
        ),
        safe_transit_T_base_tcp=transit,
        max_linear_step_mm=100.0,
        max_angular_step_deg=30.0,
        settle_time_s=0.0,
        readback_position_tolerance_mm=0.01,
        readback_rotation_tolerance_deg=0.01,
        stationarity_position_tolerance_mm=0.01,
        stationarity_rotation_tolerance_deg=0.01,
        min_sensor_plane_clearance_mm=10.0,
    )
    capture = CaptureConfig(
        timeout_s=1.0,
        min_points=80,
        max_profile_age_ms=1000.0,
        max_initial_plane_rms_mm=10.0,
    )
    plan = build_single_plane_plan(
        plane_boundary=boundary,
        T_tcp_sensor_init=T_init,
        safety=safety,
    )
    robot = MockRobot(transit.tolist())
    laser = MockPlanarLaser(
        robot,
        T_true,
        T_plane[:3, 2],
        float(T_plane[:3, 2] @ T_plane[:3, 3]),
        point_count=101,
        noise_std_mm=0.05,
    )
    return T_plane, T_init, transit, safety, capture, plan, robot, laser


def test_scan_by_scan_session_snapshot_and_resume(tmp_path):
    _plane, _init, _transit, safety, capture, plan, robot, laser = make_scene()
    session = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=capture,
    )
    record = session.capture_next()
    assert record is not None and record["scan_id"] == 0
    assert session.completed_count == 1
    assert session.next_entry()["scan_id"] == 1

    live = laser.capture_profile(timeout_s=1.0)
    snapshot = build_dashboard_snapshot(
        plan=plan,
        dataset_dir=tmp_path,
        live_sequence=7,
        live_sample=live,
    )
    assert snapshot.completed_ids == (0,)
    assert snapshot.next_scan_id == 1
    assert snapshot.previous_scan_id == 0
    assert len(snapshot.previous_profile_s) == 101
    assert len(snapshot.accumulated_points_base) == 101
    assert not snapshot.plane_estimate.available
    assert not snapshot.live_profile_s.flags.writeable

    resumed = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=capture,
    )
    assert resumed.next_entry()["scan_id"] == 1
    resumed.capture_next()
    assert resumed.completed_ids == {0, 1}


def test_accumulated_plane_appears_after_second_circular_line(tmp_path):
    _plane, _init, _transit, safety, capture, plan, robot, laser = make_scene()
    session = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=capture,
    )
    for _ in range(10):  # 9 parameter poses on line 0, then line 1.
        session.capture_next()
    snapshot = build_dashboard_snapshot(plan=plan, dataset_dir=tmp_path)
    assert snapshot.plane_estimate.available
    assert snapshot.plane_estimate.distinct_line_count == 2
    assert snapshot.plane_estimate.rms_mm < 1.0


class SerialProbeLaser(LaserInterface):
    def __init__(self) -> None:
        self.active = 0
        self.overlap = False
        self.calls = 0
        self.lock = threading.Lock()

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        del timeout_s
        with self.lock:
            self.active += 1
            self.overlap |= self.active > 1
        time.sleep(0.003)
        points = np.column_stack(
            [np.linspace(-1, 1, 10), np.zeros(10), np.ones(10)]
        )
        with self.lock:
            self.calls += 1
            self.active -= 1
        return ProfileSample(points, time.time_ns())


def test_profile_broker_is_the_only_sensor_consumer():
    laser = SerialProbeLaser()
    broker = ProfileBroker(laser, acquisition_timeout_s=0.1)
    broker.start()
    results: list[ProfileSample] = []
    workers = [
        threading.Thread(
            target=lambda: results.append(broker.capture_profile(timeout_s=1.0))
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    broker.stop()
    assert len(results) == 4
    assert laser.calls >= 1
    assert not laser.overlap


class UninterruptibleLaser(LaserInterface):
    """Simulate a vendor call that ignores its requested timeout."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        del timeout_s
        self.entered.set()
        self.release.wait()
        points = np.column_stack(
            [np.linspace(-1.0, 1.0, 10), np.zeros(10), np.ones(10)]
        )
        return ProfileSample(points, time.time_ns())


def test_profile_broker_stop_keeps_live_thread_until_vendor_call_returns():
    laser = UninterruptibleLaser()
    broker = ProfileBroker(laser, acquisition_timeout_s=0.01)
    broker.start()
    assert laser.entered.wait(timeout=1.0)
    acquisition_thread = broker._thread

    try:
        with pytest.raises(TimeoutError, match="did not stop"):
            broker.stop(timeout_s=0.01)
        assert broker._thread is acquisition_thread
        assert acquisition_thread is not None and acquisition_thread.is_alive()
    finally:
        laser.release.set()
        broker.stop(timeout_s=1.0)

    assert broker._thread is None


class StopCountingRobot(MockRobot):
    def __init__(self, initial):
        super().__init__(initial)
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


def test_cancel_is_checked_between_waypoints_and_scan_is_not_committed(tmp_path):
    _plane, _init, transit, safety, capture, plan, _robot, laser_template = make_scene()
    robot = StopCountingRobot(transit.tolist())
    laser = MockPlanarLaser(
        robot,
        laser_template.T_tcp_sensor,
        laser_template.normal,
        laser_template.offset,
        point_count=101,
    )
    cancel = threading.Event()

    def on_event(event):
        if event.get("stage") == "MOVING":
            cancel.set()

    session = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=capture,
        cancel_event=cancel,
        on_event=on_event,
    )
    with pytest.raises(MotionCancelled):
        session.capture_next()
    assert session.completed_count == 0
    assert robot.stop_count == 1
    assert not (tmp_path / "manifest.json").exists()
    session.request_stop()
    assert robot.stop_count == 1


def test_resume_rejects_a_different_plan(tmp_path):
    _plane, _init, _transit, safety, capture, plan, robot, laser = make_scene()
    first = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=capture,
    )
    first.capture_next()
    other = dict(plan)
    other["pattern_radius_mm"] = float(plan["pattern_radius_mm"]) - 1.0
    from real_laser_handeye.planning import plan_identity

    other["plan_id"] = plan_identity(other)
    with pytest.raises(RuntimeError, match="different motion plan"):
        ScanCaptureSession(
            robot=robot,
            profile_source=laser,
            plan=other,
            output_dir=tmp_path,
            safety=safety,
            capture=capture,
        )


def _write_bootstrap_views(
    output_dir,
    *,
    lateral_positions_mm=(-60.0, -20.0, 20.0, 60.0),
    profile_z_perturbation=None,
):
    """Write four scan lines on base Z=0 with sensor origins at base Z=100."""
    x_sensor = np.linspace(-75.0, 75.0, 101)
    sensor_to_base_rotation = np.diag([1.0, -1.0, -1.0])
    for index, lateral_mm in enumerate(lateral_positions_mm, start=1):
        pose = np.eye(4)
        pose[:3, :3] = sensor_to_base_rotation
        pose[:3, 3] = [0.0, float(lateral_mm), 100.0]
        perturbation = (
            np.zeros_like(x_sensor)
            if profile_z_perturbation is None
            else np.asarray(profile_z_perturbation(index, x_sensor), dtype=float)
        )
        profile = np.column_stack(
            [x_sensor, np.zeros_like(x_sensor), 100.0 + perturbation]
        )
        np.savetxt(output_dir / f"bootstrap_pose_{index}.csv", pose, delimiter=",")
        np.savetxt(
            output_dir / f"bootstrap_profile_{index}.csv",
            profile,
            delimiter=",",
            header="x_s_mm,y_s_mm,z_s_mm",
            comments="",
        )


def test_finalize_bootstrap_plane_accepts_well_distributed_four_view_plane(tmp_path):
    _write_bootstrap_views(tmp_path)

    boundary = finalize_bootstrap_plane(
        output_dir=tmp_path,
        T_tcp_sensor_init=np.eye(4),
        margin_mm=5.0,
        max_plane_rms_mm=0.1,
        min_span_mm=50.0,
        min_sensor_distance_mm=10.0,
    )

    assert boundary["quality_gate"]["accepted"] is True
    assert boundary["plane"]["rms_error_mm"] < 1e-9
    assert boundary["quality_gate"]["observed_u_span_mm"] >= 150.0
    assert boundary["quality_gate"]["observed_v_span_mm"] >= 120.0
    assert min(boundary["quality_gate"]["sensor_plane_signed_distances_mm"]) == pytest.approx(100.0)
    assert (tmp_path / "plane_boundary.json").exists()
    validate_bootstrap_boundary_quality(
        boundary,
        CaptureConfig(
            max_bootstrap_plane_rms_mm=0.1,
            min_bootstrap_span_mm=50.0,
            min_bootstrap_sensor_plane_distance_mm=10.0,
        ),
        np.eye(4),
    )
    with pytest.raises(RuntimeError, match="current capture limits"):
        validate_bootstrap_boundary_quality(
            boundary,
            CaptureConfig(
                max_bootstrap_plane_rms_mm=0.1,
                min_bootstrap_span_mm=200.0,
                min_bootstrap_sensor_plane_distance_mm=10.0,
            ),
            np.eye(4),
        )
    invalid_bounds = copy.deepcopy(boundary)
    invalid_bounds["safe_bounds_uv_mm"]["u_min"] = (
        invalid_bounds["observed_bounds_uv_mm"]["u_min"] - 1.0
    )
    with pytest.raises(RuntimeError, match="safe UV bounds"):
        validate_bootstrap_boundary_quality(
            invalid_bounds,
            CaptureConfig(
                max_bootstrap_plane_rms_mm=0.1,
                min_bootstrap_span_mm=50.0,
                min_bootstrap_sensor_plane_distance_mm=10.0,
            ),
            np.eye(4),
        )
    different_handeye = np.eye(4)
    different_handeye[0, 3] = 1.0
    with pytest.raises(RuntimeError, match="differs from the seed"):
        validate_bootstrap_boundary_quality(
            boundary,
            CaptureConfig(
                max_bootstrap_plane_rms_mm=0.1,
                min_bootstrap_span_mm=50.0,
                min_bootstrap_sensor_plane_distance_mm=10.0,
            ),
            different_handeye,
        )


def test_finalize_bootstrap_plane_rejects_bad_rms_without_writing_result(tmp_path):
    _write_bootstrap_views(
        tmp_path,
        profile_z_perturbation=lambda _index, x: 5.0 * np.sin(x / 7.0),
    )

    with pytest.raises(RuntimeError, match="plane RMS"):
        finalize_bootstrap_plane(
            output_dir=tmp_path,
            T_tcp_sensor_init=np.eye(4),
            margin_mm=5.0,
            max_plane_rms_mm=0.1,
            min_span_mm=50.0,
            min_sensor_distance_mm=10.0,
        )

    assert not (tmp_path / "plane_boundary.json").exists()


@pytest.mark.parametrize("margin_mm", [-1.0, float("nan")])
def test_finalize_bootstrap_plane_rejects_invalid_margin(tmp_path, margin_mm):
    _write_bootstrap_views(tmp_path)
    with pytest.raises(ValueError, match="finite and non-negative"):
        finalize_bootstrap_plane(
            output_dir=tmp_path,
            T_tcp_sensor_init=np.eye(4),
            margin_mm=margin_mm,
        )
    assert not (tmp_path / "plane_boundary.json").exists()


def test_finalize_bootstrap_plane_rejects_narrow_span_without_writing_result(tmp_path):
    _write_bootstrap_views(
        tmp_path,
        lateral_positions_mm=(-10.0, -3.0, 3.0, 10.0),
    )

    with pytest.raises(RuntimeError, match="observed span"):
        finalize_bootstrap_plane(
            output_dir=tmp_path,
            T_tcp_sensor_init=np.eye(4),
            margin_mm=2.0,
            max_plane_rms_mm=0.1,
            min_span_mm=50.0,
            min_sensor_distance_mm=10.0,
        )

    assert not (tmp_path / "plane_boundary.json").exists()


def test_dashboard_snapshot_ignores_manifest_from_foreign_plan(tmp_path):
    _plane, _init, _transit, _safety, _capture, plan, _robot, _laser = make_scene()
    manifest = {
        "plan_id": "foreign-plan-id",
        "expected_scan_ids": [int(entry["scan_id"]) for entry in plan["entries"]],
        "scans": [
            {
                "scan_id": 0,
                "pose_file": "must_not_be_read_pose.csv",
                "profile_file": "must_not_be_read_profile.csv",
            }
        ],
    }
    from real_laser_handeye.planning import save_json

    save_json(tmp_path / "manifest.json", manifest)
    snapshot = build_dashboard_snapshot(plan=plan, dataset_dir=tmp_path)

    assert snapshot.completed_ids == ()
    assert snapshot.previous_scan_id is None
    assert snapshot.next_scan_id == int(plan["entries"][0]["scan_id"])
    assert snapshot.accumulated_points_base.shape == (0, 3)
    assert any("another motion plan" in error for error in snapshot.data_errors)
