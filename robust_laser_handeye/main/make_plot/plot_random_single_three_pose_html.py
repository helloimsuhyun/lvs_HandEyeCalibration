#!/usr/bin/env python3
"""Create clean interactive HTML views of paired random single/three poses."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from laser_handeye.calibration_dataset import load_calibration_dataset


PLANE_COLORS = ("#4C78A8", "#F58518", "#54A24B")
PLANE_FILL_COLORS = (
    "rgba(76,120,168,0.24)",
    "rgba(245,133,24,0.24)",
    "rgba(84,162,75,0.24)",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--plane-half-size-mm", type=float, default=72.0)
    parser.add_argument(
        "--three-plane-spacing-mm",
        type=float,
        default=320.0,
        help="Display-only center spacing for the separated three-plane view.",
    )
    parser.add_argument(
        "--max-orientation-axes",
        type=int,
        default=18,
        help="Maximum sparse sensor frames/profile lines shown in each HTML.",
    )
    parser.add_argument(
        "--target-spread-radius-mm",
        type=float,
        default=38.0,
        help=(
            "Presentation-only radius used to spread scan target centers "
            "across each plane."
        ),
    )
    parser.add_argument(
        "--full-azimuth-presentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rotate complete scan geometries about each plane normal so the "
            "presentation covers the full front-side azimuth."
        ),
    )
    parser.add_argument(
        "--plotly-js",
        choices=("inline", "cdn"),
        default="inline",
        help="inline produces self-contained HTML; cdn produces smaller files.",
    )
    return parser.parse_args(argv)


def _sensor_transforms(dataset) -> np.ndarray:
    if dataset.truth is None or dataset.truth.T_ef_s_true is None:
        raise ValueError("dataset has no ground-truth hand-eye transform")
    handeye = np.asarray(dataset.truth.T_ef_s_true, dtype=float)
    return np.stack(
        [
            np.asarray(scan.T_base_ef, dtype=float) @ handeye
            for scan in dataset.scans
        ]
    )


def _plane_patch(transform: np.ndarray, half_size: float) -> np.ndarray:
    local = np.asarray(
        [
            [-half_size, -half_size, 0.0, 1.0],
            [half_size, -half_size, 0.0, 1.0],
            [half_size, half_size, 0.0, 1.0],
            [-half_size, half_size, 0.0, 1.0],
        ],
        dtype=float,
    )
    return (np.asarray(transform, dtype=float) @ local.T).T[:, :3]


def _line_coordinates(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for start, end in segments:
        x.extend((float(start[0]), float(end[0]), None))
        y.extend((float(start[1]), float(end[1]), None))
        z.extend((float(start[2]), float(end[2]), None))
    return x, y, z


def _polyline_coordinates(
    lines: Sequence[np.ndarray],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for line in lines:
        x.extend([float(value) for value in line[:, 0]])
        y.extend([float(value) for value in line[:, 1]])
        z.extend([float(value) for value in line[:, 2]])
        x.append(None)
        y.append(None)
        z.append(None)
    return x, y, z


def _sparse_indices(
    plane_ids: np.ndarray,
    maximum: int,
) -> np.ndarray:
    unique = np.unique(plane_ids)
    per_plane = max(maximum // len(unique), 1)
    selected: list[int] = []
    for plane_id in unique:
        indices = np.flatnonzero(plane_ids == plane_id)
        count = min(per_plane, len(indices))
        selected.extend(
            indices[
                np.linspace(0, len(indices) - 1, count, dtype=int)
            ].tolist()
        )
    return np.asarray(sorted(set(selected)), dtype=int)


def _sunflower_offsets(count: int, radius: float) -> np.ndarray:
    """Deterministically distribute points over a disk without clustering."""
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    index = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radial = radius * np.sqrt((index + 0.5) / count)
    angle = golden_angle * index
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle)))


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=float)
    value /= np.linalg.norm(value)
    skew = np.array(
        [
            [0.0, -value[2], value[1]],
            [value[2], 0.0, -value[0]],
            [-value[1], value[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _target_frame(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    if abs(float(n[2])) < 0.9:
        v = np.array([0.0, 0.0, 1.0])
        u = np.cross(v, n)
        u /= np.linalg.norm(u)
    else:
        u = np.array([1.0, 0.0, 0.0])
        v = np.cross(n, u)
        v /= np.linalg.norm(v)
    return np.column_stack((u, v, n))


def _natural_separated_transforms(
    dataset,
    poses: np.ndarray,
    spacing: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    planes = sorted(dataset.truth.planes, key=lambda plane: int(plane.plane_id))
    if len(planes) != 3:
        raise ValueError("separated three-plane view requires exactly 3 planes")
    common_center = np.mean(
        [
            np.asarray(plane.T_base_plane, dtype=float)[:3, 3]
            for plane in planes
        ],
        axis=0,
    )
    # A natural open workspace: one table/floor panel and two separated walls.
    # The inward direction indicates where the associated sensor cloud should
    # appear. The truth-normal sign is adjusted because a plane normal itself
    # is sign-ambiguous, while the measured side is not.
    target_centers = (
        common_center
        + np.array([0.10 * spacing, -0.28 * spacing, -0.34 * spacing]),
        common_center
        + np.array([-0.68 * spacing, 0.02 * spacing, 0.02 * spacing]),
        common_center
        + np.array([0.18 * spacing, 0.68 * spacing, 0.02 * spacing]),
    )
    inward_directions = (
        np.array([0.0, 0.0, 1.0]),   # above the table/floor panel
        np.array([1.0, 0.0, 0.0]),   # right of the left wall
        np.array([0.0, -1.0, 0.0]),  # in front of the back wall
    )

    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for plane, target_center, inward in zip(
        planes,
        target_centers,
        inward_directions,
        strict=True,
    ):
        plane_id = int(plane.plane_id)
        old_frame = np.asarray(plane.T_base_plane, dtype=float)
        old_center = old_frame[:3, 3]
        scan_indices = [
            index
            for index, scan in enumerate(dataset.scans)
            if int(scan.plane_id) == plane_id
        ]
        mean_sensor_offset = np.mean(
            poses[scan_indices, :3, 3] - old_center,
            axis=0,
        )
        measured_side = float(mean_sensor_offset @ old_frame[:3, 2])
        normal_sign = 1.0 if measured_side >= 0.0 else -1.0
        new_frame = _target_frame(normal_sign * inward)
        rotation = new_frame @ old_frame[:3, :3].T
        translation = target_center - rotation @ old_center
        transforms[plane_id] = (rotation, translation)
    return transforms


def _apply_display_transform(
    points: np.ndarray,
    transform: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    rotation, translation = transform
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        return rotation @ values + translation
    return (rotation @ values.T).T + translation


def _identity_display_transforms(
    plane_ids: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        int(plane_id): (np.eye(3), np.zeros(3, dtype=float))
        for plane_id in np.unique(plane_ids)
    }


def _add_plane(
    figure: go.Figure,
    *,
    plane,
    display_transform: tuple[np.ndarray, np.ndarray],
    half_size: float,
) -> None:
    plane_id = int(plane.plane_id)
    transform = np.asarray(plane.T_base_plane, dtype=float)
    patch = _apply_display_transform(
        _plane_patch(transform, half_size),
        display_transform,
    )
    figure.add_trace(
        go.Mesh3d(
            x=patch[:, 0],
            y=patch[:, 1],
            z=patch[:, 2],
            i=(0, 0),
            j=(1, 2),
            k=(2, 3),
            color=PLANE_FILL_COLORS[plane_id],
            opacity=0.58,
            flatshading=True,
            hovertemplate=f"plane {plane_id}<extra></extra>",
            name=f"plane {plane_id}",
            legendgroup=f"plane_{plane_id}",
        )
    )
    closed = np.vstack((patch, patch[0]))
    figure.add_trace(
        go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines",
            line={"color": PLANE_COLORS[plane_id], "width": 5},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=f"plane_{plane_id}",
        )
    )
    rotation, _translation = display_transform
    center = _apply_display_transform(
        transform[:3, 3],
        display_transform,
    )
    normal_end = center + 45.0 * (rotation @ transform[:3, 2])
    x, y, z = _line_coordinates(((center, normal_end),))
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+text",
            line={"color": PLANE_COLORS[plane_id], "width": 7},
            text=(None, f"n{plane_id}", None),
            textposition="top center",
            hoverinfo="skip",
            name=f"normal n{plane_id}",
            legendgroup=f"plane_{plane_id}",
            showlegend=False,
        )
    )


def _add_scan_group(
    figure: go.Figure,
    *,
    dataset,
    poses: np.ndarray,
    plane_id: int,
    display_transforms: dict[int, tuple[np.ndarray, np.ndarray]],
    sparse_indices: set[int],
    target_spread_radius: float,
    full_azimuth_presentation: bool,
    mode_key: str,
    initially_visible: bool,
) -> tuple[list[int], list[bool | str]]:
    first_trace_index = len(figure.data)
    indices = np.asarray(
        [
            index
            for index, scan in enumerate(dataset.scans)
            if int(scan.plane_id) == plane_id
        ],
        dtype=int,
    )
    display_transform = display_transforms[plane_id]
    display_rotation, _display_translation = display_transform
    plane = next(
        item
        for item in dataset.truth.planes
        if int(item.plane_id) == plane_id
    )
    plane_transform = np.asarray(plane.T_base_plane, dtype=float)
    display_plane_center = _apply_display_transform(
        plane_transform[:3, 3],
        display_transform,
    )
    display_u = display_rotation @ plane_transform[:3, 0]
    display_v = display_rotation @ plane_transform[:3, 1]
    display_normal = display_rotation @ plane_transform[:3, 2]
    target_offsets = _sunflower_offsets(
        len(indices),
        target_spread_radius,
    )

    base_display_origins = _apply_display_transform(
        poses[indices, :3, 3],
        display_transform,
    )
    per_scan_shift: dict[int, np.ndarray] = {}
    per_scan_azimuth_rotation: dict[int, np.ndarray] = {}
    displayed_profiles: dict[int, np.ndarray] = {}
    for local_index, index in enumerate(indices):
        pose = poses[index]
        points_sensor = np.asarray(
            dataset.scans[index].valid_points_s,
            dtype=float,
        )
        points_base = (
            pose[:3, :3] @ points_sensor.T + pose[:3, 3, None]
        ).T
        display_points = _apply_display_transform(
            points_base,
            display_transform,
        )
        desired_center = (
            display_plane_center
            + target_offsets[local_index, 0] * display_u
            + target_offsets[local_index, 1] * display_v
        )
        shift = desired_center - np.mean(display_points, axis=0)
        if full_azimuth_presentation:
            azimuth_rotation = _axis_angle_rotation(
                display_normal,
                2.0 * np.pi * local_index / len(indices),
            )
        else:
            azimuth_rotation = np.eye(3)
        per_scan_shift[int(index)] = shift
        per_scan_azimuth_rotation[int(index)] = azimuth_rotation
        shifted_points = display_points + shift
        displayed_profiles[int(index)] = (
            azimuth_rotation
            @ (shifted_points - desired_center).T
        ).T + desired_center

    origins = np.stack(
        [
            (
                per_scan_azimuth_rotation[int(index)]
                @ (
                    origin
                    + per_scan_shift[int(index)]
                    - (
                        display_plane_center
                        + target_offsets[local_index, 0] * display_u
                        + target_offsets[local_index, 1] * display_v
                    )
                )
                + (
                    display_plane_center
                    + target_offsets[local_index, 0] * display_u
                    + target_offsets[local_index, 1] * display_v
                )
            )
            for local_index, (origin, index) in enumerate(
                zip(base_display_origins, indices, strict=True)
            )
        ],
        axis=0,
    )
    customdata = np.asarray(
        [
            (int(index), int(dataset.scans[index].scan_id), plane_id)
            for index in indices
        ],
        dtype=object,
    )
    figure.add_trace(
        go.Scatter3d(
            x=origins[:, 0],
            y=origins[:, 1],
            z=origins[:, 2],
            mode="markers",
            marker={
                "size": 4.2,
                "color": PLANE_COLORS[plane_id],
                "opacity": 0.82,
                "line": {"color": "white", "width": 0.6},
            },
            customdata=customdata,
            hovertemplate=(
                "scan index: %{customdata[0]}<br>"
                "scan id: %{customdata[1]}<br>"
                "plane: %{customdata[2]}<br>"
                "x: %{x:.1f} mm<br>"
                "y: %{y:.1f} mm<br>"
                "z: %{z:.1f} mm"
                "<extra></extra>"
            ),
            name=f"sensor origins · plane {plane_id}",
            legendgroup=f"{mode_key}_scans_{plane_id}",
            visible=initially_visible,
        )
    )

    profile_lines: list[np.ndarray] = []
    view_segments: list[tuple[np.ndarray, np.ndarray]] = []
    x_segments: list[tuple[np.ndarray, np.ndarray]] = []
    link_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for index in indices:
        if int(index) not in sparse_indices:
            continue
        pose = poses[index]
        unrotated_origin = (
            _apply_display_transform(
                pose[:3, 3],
                display_transform,
            )
            + per_scan_shift[int(index)]
        )
        points_display = displayed_profiles[int(index)]
        center = np.mean(points_display, axis=0)
        azimuth_rotation = per_scan_azimuth_rotation[int(index)]
        origin = (
            azimuth_rotation @ (unrotated_origin - center) + center
        )
        profile_lines.append(points_display)
        link_segments.append((origin, center))
        view_segments.append(
            (
                origin,
                origin
                + 28.0
                * (
                    azimuth_rotation
                    @ display_rotation
                    @ pose[:3, 2]
                ),
            )
        )
        x_segments.append(
            (
                origin,
                origin
                + 20.0
                * (
                    azimuth_rotation
                    @ display_rotation
                    @ pose[:3, 0]
                ),
            )
        )

    profile_x, profile_y, profile_z = _polyline_coordinates(profile_lines)
    figure.add_trace(
        go.Scatter3d(
            x=profile_x,
            y=profile_y,
            z=profile_z,
            mode="lines",
            line={"color": PLANE_COLORS[plane_id], "width": 4},
            opacity=0.85,
            hoverinfo="skip",
            name=f"sampled profiles · plane {plane_id}",
            legendgroup=f"{mode_key}_profiles_{plane_id}",
            visible=initially_visible,
        )
    )
    for segments, name, color, dash, width, visible in (
        (
            link_segments,
            f"sensor-to-profile · plane {plane_id}",
            "#777777",
            "dot",
            2,
            True,
        ),
        (
            view_segments,
            f"sensor +z · plane {plane_id}",
            "#202020",
            "solid",
            4,
            True,
        ),
        (
            x_segments,
            f"sensor +x · plane {plane_id}",
            "#D62728",
            "solid",
            3,
            "legendonly",
        ),
    ):
        x, y, z = _line_coordinates(segments)
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": color, "width": width, "dash": dash},
                opacity=0.68,
                hoverinfo="skip",
                name=name,
                legendgroup=f"{mode_key}_axes_{plane_id}",
                visible=visible if initially_visible else False,
                showlegend=False,
            )
        )
    indices_added = list(range(first_trace_index, len(figure.data)))
    active_visibility: list[bool | str] = [
        True,
        True,
        True,
        True,
        "legendonly",
    ]
    if len(indices_added) != len(active_visibility):
        raise RuntimeError("unexpected scan trace count")
    return indices_added, active_visibility


def _make_figure(
    dataset,
    *,
    separated: bool,
    spacing: float,
    plane_half_size: float,
    max_orientation_axes: int,
    target_spread_radius: float,
    full_azimuth_presentation: bool,
    title: str,
) -> go.Figure:
    poses = _sensor_transforms(dataset)
    plane_ids = np.asarray(
        [int(scan.plane_id) for scan in dataset.scans],
        dtype=int,
    )
    if separated:
        display_transforms = _natural_separated_transforms(
            dataset,
            poses,
            spacing,
        )
    else:
        display_transforms = _identity_display_transforms(plane_ids)
    sparse = set(
        map(int, _sparse_indices(plane_ids, max_orientation_axes))
    )

    figure = go.Figure()
    for plane in sorted(
        dataset.truth.planes, key=lambda item: int(item.plane_id)
    ):
        _add_plane(
            figure,
            plane=plane,
            display_transform=display_transforms[int(plane.plane_id)],
            half_size=plane_half_size,
        )
    plane_trace_count = len(figure.data)
    mode_indices: dict[str, list[int]] = {
        "experiment": [],
        "diverse": [],
    }
    mode_active_visibility: dict[str, list[bool | str]] = {
        "experiment": [],
        "diverse": [],
    }
    for mode_key, use_full_azimuth in (
        ("experiment", False),
        ("diverse", True),
    ):
        for plane_id in np.unique(plane_ids):
            indices_added, active_visibility = _add_scan_group(
                figure,
                dataset=dataset,
                poses=poses,
                plane_id=int(plane_id),
                display_transforms=display_transforms,
                sparse_indices=sparse,
                target_spread_radius=target_spread_radius,
                full_azimuth_presentation=use_full_azimuth,
                mode_key=mode_key,
                initially_visible=(
                    use_full_azimuth == full_azimuth_presentation
                ),
            )
            mode_indices[mode_key].extend(indices_added)
            mode_active_visibility[mode_key].extend(active_visibility)

    common_subtitle = (
        "Display-only natural workspace: every plane, associated sensor pose, "
        "and profile is rigidly transformed together; scan targets are spread "
        "across each surface."
        if separated
        else (
            "Presentation view: scan targets are spread across the plane "
            "while each sensor-to-profile geometry is preserved."
        )
    )
    subtitles = {
        "experiment": (
            common_subtitle
            + " Experiment tilt/azimuth/roll ranges are retained."
        ),
        "diverse": (
            common_subtitle
            + " Full front-side azimuth is used for visualization only."
        ),
    }
    titles = {
        mode_key: f"{title}<br><sup>{subtitle}</sup>"
        for mode_key, subtitle in subtitles.items()
    }

    visibility_by_mode: dict[str, list[bool | str]] = {}
    for selected_mode in ("experiment", "diverse"):
        visibility: list[bool | str] = [True] * plane_trace_count
        for mode_key in ("experiment", "diverse"):
            if mode_key == selected_mode:
                visibility.extend(mode_active_visibility[mode_key])
            else:
                visibility.extend([False] * len(mode_indices[mode_key]))
        visibility_by_mode[selected_mode] = visibility

    selected_mode = (
        "diverse" if full_azimuth_presentation else "experiment"
    )
    if len(visibility_by_mode[selected_mode]) != len(figure.data):
        raise RuntimeError("mode visibility vector does not match traces")
    for trace, visible in zip(
        figure.data,
        visibility_by_mode[selected_mode],
        strict=True,
    ):
        trace.visible = visible

    figure.update_layout(
        title={
            "text": titles[selected_mode],
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        paper_bgcolor="#F7F8FA",
        plot_bgcolor="#FFFFFF",
        margin={"l": 0, "r": 0, "t": 88, "b": 0},
        height=790,
        legend={
            "x": 0.01,
            "y": 0.99,
            "bgcolor": "rgba(255,255,255,0.84)",
            "bordercolor": "#D7DAE0",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        scene={
            "xaxis": {
                "title": "base x [mm]",
                "gridcolor": "#E4E7EC",
                "showbackground": True,
                "backgroundcolor": "#FBFCFD",
            },
            "yaxis": {
                "title": "base y [mm]",
                "gridcolor": "#E4E7EC",
                "showbackground": True,
                "backgroundcolor": "#FBFCFD",
            },
            "zaxis": {
                "title": "base z [mm]",
                "gridcolor": "#E4E7EC",
                "showbackground": True,
                "backgroundcolor": "#FBFCFD",
            },
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.55, "y": -1.65, "z": 1.15},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.08,
                "yanchor": "bottom",
                "showactive": True,
                "active": 1 if selected_mode == "diverse" else 0,
                "bgcolor": "white",
                "bordercolor": "#B8C0CC",
                "borderwidth": 1,
                "buttons": [
                    {
                        "label": "Experiment condition",
                        "method": "update",
                        "args": [
                            {
                                "visible": visibility_by_mode[
                                    "experiment"
                                ]
                            },
                            {"title.text": titles["experiment"]},
                        ],
                    },
                    {
                        "label": "Diverse display",
                        "method": "update",
                        "args": [
                            {"visible": visibility_by_mode["diverse"]},
                            {"title.text": titles["diverse"]},
                        ],
                    },
                ],
            }
        ],
    )
    return figure


def _write_html(
    figure: go.Figure,
    output: Path,
    *,
    heading: str,
    description: str,
    plotly_js: str,
) -> None:
    figure_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True if plotly_js == "inline" else "cdn",
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": output.stem,
                "scale": 2,
            },
        },
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(heading)}</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f2f4f7;
      color: #182230;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 22px; }}
    .card {{
      overflow: hidden;
      background: white;
      border: 1px solid #dfe3e8;
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(16, 24, 40, 0.08);
    }}
    header {{ padding: 22px 26px 10px; }}
    h1 {{ margin: 0 0 7px; font-size: 24px; letter-spacing: -0.02em; }}
    p {{ margin: 0; color: #667085; font-size: 14px; line-height: 1.55; }}
    .hint {{
      margin: 0 26px 18px;
      padding: 10px 13px;
      border-radius: 9px;
      background: #f8fafc;
      color: #475467;
      font-size: 13px;
    }}
    .plot {{ width: 100%; min-height: 790px; }}
    .plot .plotly-graph-div {{ width: 100% !important; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <header>
        <h1>{escape(heading)}</h1>
        <p>{escape(description)}</p>
      </header>
      <div class="hint">
        Switch Experiment/Diverse mode above the 3D view · drag to rotate ·
        wheel to zoom · click legend items to hide/show profiles · use the
        camera icon to export PNG.
      </div>
      <div class="plot">{figure_html}</div>
      <script>
        window.addEventListener("load", () => {{
          document.querySelectorAll(".plotly-graph-div").forEach((element) => {{
            if (window.Plotly) window.Plotly.Plots.resize(element);
          }});
        }});
      </script>
    </section>
  </main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def _write_index(
    output: Path,
    *,
    trial_index: int,
    single_name: str,
    three_name: str,
    condition_label: str,
    condition_summary: str,
) -> None:
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Random pose 3D comparison</title>
  <style>
    body {{ margin: 0; background: #f2f4f7; color: #182230;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 56px 24px; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #667085; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr);
      gap: 18px; margin-top: 28px; }}
    a {{ display: block; padding: 28px; background: white; color: #182230;
      text-decoration: none; border: 1px solid #dfe3e8; border-radius: 14px;
      box-shadow: 0 8px 24px rgba(16,24,40,.07); }}
    a:hover {{ border-color: #4c78a8; transform: translateY(-1px); }}
    strong {{ display: block; font-size: 19px; margin-bottom: 7px; }}
    span {{ color: #667085; font-size: 14px; line-height: 1.5; }}
    @media (max-width: 680px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Random pose 3D comparison</h1>
  <p>{escape(condition_label)} pose-diversity dataset ·
    {escape(condition_summary)} · trial {trial_index:06d}</p>
  <div class="grid">
    <a href="{escape(single_name)}"><strong>Single-plane random</strong>
      <span>True simulated base-frame geometry.</span></a>
    <a href="{escape(three_name)}"><strong>Three-plane random</strong>
      <span>Clean separated display with each measurement group moved
      together.</span></a>
  </div>
</main></body>
</html>
""",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trial_index < 0:
        raise SystemExit("--trial-index must be non-negative")
    if args.plane_half_size_mm <= 0.0:
        raise SystemExit("--plane-half-size-mm must be positive")
    if args.three_plane_spacing_mm <= 0.0:
        raise SystemExit("--three-plane-spacing-mm must be positive")
    if args.max_orientation_axes <= 0:
        raise SystemExit("--max-orientation-axes must be positive")
    if args.target_spread_radius_mm <= 0.0:
        raise SystemExit("--target-spread-radius-mm must be positive")
    if args.target_spread_radius_mm >= args.plane_half_size_mm:
        raise SystemExit(
            "--target-spread-radius-mm must be smaller than "
            "--plane-half-size-mm"
        )

    root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    condition_label = root.name.rsplit("_N", 1)[0].replace("_", " ").title()
    tilt_range = config["view_tilt_range_deg"]
    azimuth_range = config["view_azimuth_range_deg"]
    roll_range = config["sensor_roll_range_deg"]
    condition_summary = (
        f"tilt {tilt_range[0]:g}°–{tilt_range[1]:g}° · "
        f"azimuth {azimuth_range[0]:g}°–{azimuth_range[1]:g}° · "
        f"roll {roll_range[0]:g}°–{roll_range[1]:g}°"
    )
    trial = f"trial_{args.trial_index:06d}"
    single = load_calibration_dataset(
        root / "single_plane" / "trials" / trial
    )
    three = load_calibration_dataset(
        root / "three_plane" / "trials" / trial
    )

    single_name = f"single_random_{trial}_3d.html"
    three_name = f"three_random_separated_{trial}_3d.html"
    single_figure = _make_figure(
        single,
        separated=False,
        spacing=args.three_plane_spacing_mm,
        plane_half_size=args.plane_half_size_mm,
        max_orientation_axes=args.max_orientation_axes,
        target_spread_radius=args.target_spread_radius_mm,
        full_azimuth_presentation=args.full_azimuth_presentation,
        title=(
            f"{condition_label} · Single-plane random poses · "
            f"{len(single.scans)} scans · {trial}"
        ),
    )
    three_figure = _make_figure(
        three,
        separated=True,
        spacing=args.three_plane_spacing_mm,
        plane_half_size=args.plane_half_size_mm,
        max_orientation_axes=args.max_orientation_axes,
        target_spread_radius=args.target_spread_radius_mm,
        full_azimuth_presentation=args.full_azimuth_presentation,
        title=(
            f"{condition_label} · Three-plane random poses · "
            f"{len(three.scans)} scans · {trial}"
        ),
    )
    _write_html(
        single_figure,
        output_dir / single_name,
        heading="Single-plane random pose visualization",
        description=(
            f"{condition_summary}. Scan target centers are spread over the "
            "plane to avoid an artificial single-point focus. Use the mode "
            "buttons to switch between the experiment's orientation ranges "
            "and a visualization-only full-azimuth distribution."
        ),
        plotly_js=args.plotly_js,
    )
    _write_html(
        three_figure,
        output_dir / three_name,
        heading="Three-plane random pose visualization",
        description=(
            f"{condition_summary}. The three planes form a clean "
            "table/left-wall/back-wall layout for presentation. Each "
            "plane's sensors and profiles receive the same rigid display "
            "transform. Use the mode buttons to switch between the "
            "experiment's orientation ranges and visualization-only "
            "full-azimuth views."
        ),
        plotly_js=args.plotly_js,
    )
    _write_index(
        output_dir / "index.html",
        trial_index=args.trial_index,
        single_name=single_name,
        three_name=three_name,
        condition_label=condition_label,
        condition_summary=condition_summary,
    )
    print(f"Saved HTML index : {output_dir / 'index.html'}")
    print(f"Saved single HTML: {output_dir / single_name}")
    print(f"Saved three HTML : {output_dir / three_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
