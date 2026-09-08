from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from handeye_mujoco import (
    HandEyeSimulation,
    automatic_scan_radius_mm,
    centered_convex_inradius,
    interpolate_joint_path,
    write_model_with_convex_board,
    write_model_with_infinite_plane,
    write_model_with_plane_box,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/rb5_ljv7080/model.xml"


def test_centered_convex_inradius_and_automatic_radius():
    boundary = np.array([[-100, -80], [100, -80], [100, 80], [-100, 80]])
    assert centered_convex_inradius(boundary) == 80.0
    assert automatic_scan_radius_mm(
        boundary, edge_margin_mm=10.0, fill_ratio=0.9
    ) == 63.0


def test_joint_interpolation_respects_max_step():
    path = interpolate_joint_path(
        np.zeros(2), np.array([0.21, -0.1]), max_step_rad=0.05
    )
    np.testing.assert_allclose(path[0], [0, 0])
    np.testing.assert_allclose(path[-1], [0.21, -0.1])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 0.05 + 1e-12


def test_board_model_relocates_assets_and_ik_holds_current_pose(tmp_path):
    boundary = np.array([[-100, -80], [100, -80], [100, 80], [-100, 80]])
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, 1500]
    output = tmp_path / "model_with_board.xml"
    write_model_with_convex_board(
        MODEL,
        output,
        T_world_plane_mm=transform,
        boundary_uv_mm=boundary,
    )
    simulation = HandEyeSimulation(output)
    simulation.reset_home()
    target = simulation.sensor_pose_world()
    result = simulation.solve_site_ik(target, simulation.data.qpos.copy())
    assert result.success
    assert result.iterations == 1
    assert simulation.model.ngeom > HandEyeSimulation(MODEL).model.ngeom


def test_site_ik_reaches_nearby_known_joint_pose():
    simulation = HandEyeSimulation(MODEL)
    simulation.reset_home()
    seed = simulation.data.qpos.copy()
    known = seed + np.deg2rad([2.0, -2.0, 2.0, 1.0, -1.0, 2.0])
    simulation.set_joint_positions(known)
    target = simulation.sensor_pose_world()
    result = simulation.solve_site_ik(target, seed)
    assert result.success
    assert result.position_error_m <= 5e-4
    assert result.rotation_error_rad <= np.deg2rad(0.25)


def test_moveit_equivalent_plane_box_is_centered_and_compiles(tmp_path):
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, -1500]
    output = tmp_path / "model_with_infinite_plane.xml"
    write_model_with_plane_box(
        MODEL,
        output,
        T_world_plane_mm=transform,
        plane_size_xy_mm=[400.0, 300.0],
        plane_thickness_mm=10.0,
        keepout_clearance_mm=10.0,
        contact_margin_mm=0.0,
    )
    root = ET.parse(output).getroot()
    body = next(
        item for item in root.iter("body") if item.get("name") == "estimated_plane_keepout"
    )
    collision = next(
        item for item in body.findall("geom") if item.get("name") == "estimated_plane_collision"
    )
    visual = next(
        item for item in body.findall("geom") if item.get("name") == "estimated_plane_visual_board"
    )
    assert collision.get("type") == "box"
    np.testing.assert_allclose(
        np.fromstring(collision.get("size"), sep=" "), [0.2, 0.15, 0.015]
    )
    np.testing.assert_allclose(
        np.fromstring(body.get("pos"), sep=" "), [0.0, 0.0, -1.5]
    )
    assert visual.get("pos") == "0 0 0"
    assert visual.get("contype") == "0"
    simulation = HandEyeSimulation(output)
    simulation.reset_home()
    assert simulation.collision_report().collision_free


def test_physical_and_measurement_sensor_sites_keep_configured_offset(tmp_path):
    plane = np.eye(4)
    plane[:3, 3] = [0, 0, -1500]
    measurement_to_physical = np.eye(4)
    measurement_to_physical[:3, 3] = [0, 0, -80]
    output = tmp_path / "model_with_sensor_frames.xml"
    write_model_with_infinite_plane(
        MODEL,
        output,
        T_world_plane_mm=plane,
        sensor_body_name="lj_v7080",
        physical_sensor_site_name="sensor_physical_origin",
        T_measurement_physical_mm=measurement_to_physical,
    )
    simulation = HandEyeSimulation(output)
    simulation.reset_home()
    physical = simulation.site_pose_world("sensor_physical_origin")
    measurement = simulation.site_pose_world("sensor_origin")
    physical_to_measurement = np.linalg.inv(physical) @ measurement
    np.testing.assert_allclose(
        physical_to_measurement[:3, :3], np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(
        physical_to_measurement[:3, 3], [0.0, 0.0, 0.08], atol=1e-9
    )
    root = ET.parse(output).getroot()
    sites = {
        item.get("name"): item
        for item in root.iter("site")
        if item.get("name") is not None
    }
    assert sites["sensor_origin"].get("rgba") == "1.0 0.05 0.85 1.0"
    assert sites["sensor_origin"].get("size") == "0.009"
    assert "measurement_sensor_origin_preview" not in sites
    assert "physical_to_measurement_offset_preview" in sites
