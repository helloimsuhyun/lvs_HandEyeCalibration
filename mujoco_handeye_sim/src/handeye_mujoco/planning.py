from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .transforms import format_vector, matrix_to_quaternion, validate_transform


def centered_convex_inradius(boundary_uv_mm: np.ndarray) -> float:
    """Radius of the largest origin-centred circle inside a convex polygon."""
    boundary = np.asarray(boundary_uv_mm, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError("boundary_uv_mm must have shape (N, 2), N >= 3")
    if not np.all(np.isfinite(boundary)):
        raise ValueError("boundary_uv_mm contains NaN or Inf")

    distances: list[float] = []
    side_signs: list[float] = []
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        # Plane fitting can leave numerically duplicated hull vertices after
        # projection.  They do not change the polygon and are safe to skip.
        if length <= 1e-7:
            continue
        signed_cross = float(edge[0] * (-start[1]) - edge[1] * (-start[0]))
        side_signs.append(signed_cross)
        distances.append(abs(signed_cross) / length)

    if len(distances) < 3:
        raise ValueError("boundary_uv_mm has fewer than three non-zero edges")
    if min(side_signs) < -1e-9 and max(side_signs) > 1e-9:
        raise ValueError("polygon origin is outside the convex boundary")
    radius = min(distances)
    if radius <= 1e-9:
        raise ValueError("polygon origin is on or outside its boundary")
    return float(radius)


def automatic_scan_radius_mm(
    boundary_uv_mm: np.ndarray,
    *,
    edge_margin_mm: float = 10.0,
    fill_ratio: float = 0.90,
) -> float:
    if edge_margin_mm < 0 or not (0.0 < fill_ratio <= 1.0):
        raise ValueError("edge margin must be non-negative and fill_ratio in (0, 1]")
    available = centered_convex_inradius(boundary_uv_mm) - edge_margin_mm
    if available <= 0:
        raise ValueError(
            "convex hull is too small for the requested edge margin: "
            f"inradius={available + edge_margin_mm:.3f} mm, "
            f"margin={edge_margin_mm:.3f} mm"
        )
    return float(available * fill_ratio)


def interpolate_joint_path(
    start_qpos: np.ndarray,
    goal_qpos: np.ndarray,
    *,
    max_step_rad: float = np.deg2rad(1.0),
) -> np.ndarray:
    start = np.asarray(start_qpos, dtype=float)
    goal = np.asarray(goal_qpos, dtype=float)
    if start.shape != goal.shape or start.ndim != 1:
        raise ValueError("start_qpos and goal_qpos must be equal-length vectors")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("joint path endpoints must be finite")
    if max_step_rad <= 0:
        raise ValueError("max_step_rad must be positive")
    sample_count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / max_step_rad)) + 1)
    return np.linspace(start, goal, sample_count)


def write_model_with_convex_board(
    source_model: str | Path,
    output_model: str | Path,
    *,
    T_world_plane_mm: np.ndarray,
    boundary_uv_mm: np.ndarray,
    thickness_mm: float = 6.0,
    contact_margin_mm: float = 5.0,
) -> Path:
    """Add the observed convex hull as a thin, fixed MuJoCo collision prism."""
    source = Path(source_model).resolve()
    output = Path(output_model).resolve()
    transform = validate_transform(T_world_plane_mm, "T_world_plane_mm").copy()
    boundary = np.asarray(boundary_uv_mm, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError("boundary_uv_mm must have shape (N, 2), N >= 3")
    if thickness_mm <= 0 or contact_margin_mm < 0:
        raise ValueError("board thickness must be positive and margin non-negative")

    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("MuJoCo model must contain asset and worldbody elements")

    # The generated base model normally uses paths relative to its own build
    # directory.  The planning model is written into a run directory, so make
    # those references explicit before relocating the XML.
    for mesh in asset.findall("mesh"):
        raw_file = mesh.get("file")
        if raw_file and not Path(raw_file).is_absolute():
            mesh.set("file", str((source.parent / raw_file).resolve()))

    half_thickness_m = thickness_mm * 0.5e-3
    lower = np.column_stack(
        [boundary * 1e-3, np.full(len(boundary), -half_thickness_m)]
    )
    upper = np.column_stack(
        [boundary * 1e-3, np.full(len(boundary), half_thickness_m)]
    )
    vertices = np.vstack([lower, upper])
    ET.SubElement(
        asset,
        "mesh",
        name="estimated_board_hull_mesh",
        vertex=format_vector(vertices.reshape(-1)),
    )
    transform[:3, 3] *= 1e-3
    board = ET.SubElement(
        worldbody,
        "body",
        name="estimated_calibration_board",
        pos=format_vector(transform[:3, 3]),
        quat=format_vector(matrix_to_quaternion(transform[:3, :3])),
    )
    ET.SubElement(
        board,
        "geom",
        name="estimated_board_collision",
        type="mesh",
        mesh="estimated_board_hull_mesh",
        rgba="0.15 0.75 0.30 0.35",
        margin=str(contact_margin_mm * 1e-3),
        group="4",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def write_model_with_plane_box(
    source_model: str | Path,
    output_model: str | Path,
    *,
    T_world_plane_mm: np.ndarray,
    plane_size_xy_mm: np.ndarray | tuple[float, float] | list[float] = (600.0, 440.0),
    plane_thickness_mm: float = 10.0,
    keepout_clearance_mm: float = 10.0,
    contact_margin_mm: float = 0.0,
    sensor_body_name: str | None = None,
    physical_sensor_site_name: str = "sensor_physical_origin",
    T_measurement_physical_mm: np.ndarray | None = None,
    sensor_frame_axis_length_mm: float = 70.0,
) -> Path:
    """Add the same centered finite plane box used by MoveIt.

    Collision is a finite box centered on ``T_world_plane_mm`` with total thickness
    ``plane_thickness_mm + 2 * keepout_clearance_mm``.  This matches MoveIt's
    symmetric calibration-plane collision object.
    """
    source = Path(source_model).resolve()
    output = Path(output_model).resolve()
    transform = validate_transform(T_world_plane_mm, "T_world_plane_mm").copy()
    size_xy_mm = np.asarray(plane_size_xy_mm, dtype=float).reshape(-1)
    if size_xy_mm.size != 2 or not np.all(np.isfinite(size_xy_mm)):
        raise ValueError("plane_size_xy_mm must contain two finite values")
    if np.any(size_xy_mm <= 0.0):
        raise ValueError("plane_size_xy_mm values must be positive")
    if plane_thickness_mm <= 0:
        raise ValueError("plane_thickness_mm must be positive")
    if keepout_clearance_mm < 0 or contact_margin_mm < 0:
        raise ValueError("keep-out clearance and contact margin must be non-negative")

    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("MuJoCo model must contain asset and worldbody elements")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "960")
    global_visual.set("offheight", "720")
    for item in asset:
        raw_file = item.get("file")
        if raw_file and not Path(raw_file).is_absolute():
            item.set("file", str((source.parent / raw_file).resolve()))

    if T_measurement_physical_mm is not None:
        if not sensor_body_name:
            raise ValueError("sensor_body_name is required for a physical sensor site")
        local = validate_transform(
            T_measurement_physical_mm, "T_measurement_physical_mm"
        ).copy()
        local[:3, 3] *= 1e-3
        axis_length_m = float(sensor_frame_axis_length_mm) * 1e-3
        if not np.isfinite(axis_length_m) or axis_length_m <= 0:
            raise ValueError("sensor_frame_axis_length_mm must be positive and finite")
        sensor_body = next(
            (item for item in root.iter("body") if item.get("name") == sensor_body_name),
            None,
        )
        if sensor_body is None:
            raise ValueError(f"MuJoCo model has no sensor body {sensor_body_name!r}")
        measurement_site = next(
            (
                item
                for item in sensor_body.findall("site")
                if item.get("name") == "sensor_origin"
            ),
            None,
        )
        if measurement_site is None:
            raise ValueError("MuJoCo sensor body has no measurement site 'sensor_origin'")
        # Recolor the actual S site instead of drawing a second sphere on top
        # of the base model's red marker. One origin now means one visible ball.
        measurement_site.set("type", "sphere")
        measurement_site.set("size", "0.009")
        measurement_site.set("rgba", "1.0 0.05 0.85 1.0")
        if any(
            item.get("name") == physical_sensor_site_name
            for item in sensor_body.findall("site")
        ):
            raise ValueError(
                f"MuJoCo model already has site {physical_sensor_site_name!r}"
            )
        ET.SubElement(
            sensor_body,
            "site",
            name=physical_sensor_site_name,
            type="sphere",
            pos=format_vector(local[:3, 3]),
            quat=format_vector(matrix_to_quaternion(local[:3, :3])),
            size="0.011",
            rgba="1.0 0.9 0.05 1.0",
        )
        ET.SubElement(
            sensor_body,
            "site",
            name="physical_to_measurement_offset_preview",
            type="cylinder",
            fromto=format_vector(
                np.concatenate([local[:3, 3], np.zeros(3)])
            ),
            size="0.0025",
            rgba="1.0 0.75 0.15 0.85",
        )
        axis_colors = (
            ("x", "1.0 0.05 0.05 1.0"),
            ("y", "0.05 1.0 0.05 1.0"),
            ("z", "0.10 0.35 1.0 1.0"),
        )
        for axis_index, (axis_name, color) in enumerate(axis_colors):
            tip = local[:3, 3] + local[:3, :3] @ (
                np.eye(3)[axis_index] * axis_length_m
            )
            ET.SubElement(
                sensor_body,
                "site",
                name=f"{physical_sensor_site_name}_axis_{axis_name}",
                type="cylinder",
                fromto=format_vector(np.concatenate([local[:3, 3], tip])),
                size="0.0045",
                rgba=color,
            )

        # Larger measurement-frame axes. The existing sensor_origin sphere was
        # recolored above, so no duplicate S marker is created here.
        for axis_index, (axis_name, color) in enumerate(axis_colors):
            tip = np.eye(3)[axis_index] * axis_length_m
            ET.SubElement(
                sensor_body,
                "site",
                name=f"measurement_sensor_axis_{axis_name}_preview",
                type="cylinder",
                fromto=format_vector(np.concatenate([np.zeros(3), tip])),
                size="0.0028",
                rgba=color,
            )

    transform[:3, 3] *= 1e-3
    half_size_m = np.array(
        [
            0.5 * size_xy_mm[0] * 1e-3,
            0.5 * size_xy_mm[1] * 1e-3,
            0.5 * (plane_thickness_mm + 2.0 * keepout_clearance_mm) * 1e-3,
        ],
        dtype=float,
    )
    body = ET.SubElement(
        worldbody,
        "body",
        name="estimated_plane_keepout",
        pos=format_vector(transform[:3, 3]),
        quat=format_vector(matrix_to_quaternion(transform[:3, :3])),
    )
    ET.SubElement(
        body,
        "geom",
        name="estimated_plane_collision",
        type="box",
        size=format_vector(half_size_m),
        rgba="0.20 0.65 1.00 0.24",
        margin=str(contact_margin_mm * 1e-3),
        group="4",
    )
    # Draw the estimated physical board at the center of the translucent
    # collision volume. This visual geom is non-colliding.
    ET.SubElement(
        body,
        "geom",
        name="estimated_plane_visual_board",
        type="box",
        size=format_vector(
            [half_size_m[0], half_size_m[1], 0.5 * plane_thickness_mm * 1e-3]
        ),
        pos="0 0 0",
        rgba="1.00 0.48 0.04 0.82",
        contype="0",
        conaffinity="0",
        group="4",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


# Backward-compatible alias for existing callers and saved scripts. The model
# generated by this function is no longer an infinite, one-sided plane.
write_model_with_infinite_plane = write_model_with_plane_box
