from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .data import LaserScan
from .scene_generation import Plane, normalize_vector, plane_basis
from .se3 import transform_points


class CalibrationTrialLike(Protocol):
    """Fields used by the common benchmark report and plot functions."""

    converged: bool
    paper_success: bool
    trans_err_norm_mm: float
    rot_err_angle_deg: float
    init_trans_err_norm_mm: float
    init_rot_err_angle_deg: float
    plane_rms_history_mm: list[float]
    iter_T_frob_error: list[float]


def iter_frobenius_error(
    T_history: Sequence[np.ndarray],
    T_true: np.ndarray,
    T_initial: np.ndarray | None = None,
) -> list[float]:
    """Return ||T_true - T_est||_F per iteration.

    Translation entries are converted from millimetres to metres before the
    Frobenius norm is evaluated. This preserves the convention used by the
    existing Carlson-style benchmark plot.
    """

    estimates: list[np.ndarray] = []
    if T_initial is not None:
        estimates.append(T_initial)
    estimates.extend(T_history)

    T_true_m = _transform_with_translation_in_metres(T_true)
    return [
        float(
            np.linalg.norm(
                T_true_m - _transform_with_translation_in_metres(T_est),
                ord="fro",
            )
        )
        for T_est in estimates
    ]


def save_results_csv(results: Sequence[Any], path: Path) -> None:
    """Save dataclass benchmark results, JSON-encoding list-valued fields."""

    if not results:
        raise ValueError("no results to save")

    rows = [_result_to_csv_row(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_calibration_summary(results: Sequence[CalibrationTrialLike]) -> None:
    """Print the common translation, rotation, convergence and success summary."""

    if not results:
        print("No successful generated trials.")
        return

    translation_errors = np.asarray(
        [result.trans_err_norm_mm for result in results],
        dtype=float,
    )
    rotation_errors = np.asarray(
        [result.rot_err_angle_deg for result in results],
        dtype=float,
    )

    converged = sum(bool(result.converged) for result in results)
    succeeded = sum(bool(result.paper_success) for result in results)

    print(f"generated trials: {len(results)}")
    print(f"converged: {converged} / {len(results)}")
    print(f"paper-style success: {succeeded} / {len(results)}")
    print(
        "translation norm error [mm]: "
        f"median={np.median(translation_errors):.6g}, "
        f"mean={np.mean(translation_errors):.6g}, "
        f"max={np.max(translation_errors):.6g}"
    )
    print(
        "rotation angle error [deg]: "
        f"median={np.median(rotation_errors):.6g}, "
        f"mean={np.mean(rotation_errors):.6g}, "
        f"max={np.max(rotation_errors):.6g}"
    )


def save_calibration_plots(
    results: Sequence[CalibrationTrialLike],
    plot_dir: Path,
) -> list[Path]:
    """Save common convergence and before/after calibration plots."""

    if not results:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    transform_error_path = plot_dir / "T_true_minus_T_estimated_per_iter.png"
    if save_history_plot(
        histories=[result.iter_T_frob_error for result in results],
        out_path=transform_error_path,
        ylabel=r"$||T_{true}-\hat{T}||_F$",
        title="Per-iteration transform error to ground truth",
        semilogy=True,
        x_start=0,
    ):
        saved.append(transform_error_path)

    plane_rms_path = plot_dir / "rms_distance_from_points_to_plane_m.png"
    if save_history_plot(
        histories=[
            [float(value) * 1e-3 for value in result.plane_rms_history_mm]
            for result in results
        ],
        out_path=plane_rms_path,
        ylabel="RMS distance from points to plane [m]",
        title="Per-iteration RMS distance to fitted plane",
        semilogy=True,
        x_start=1,
    ):
        saved.append(plane_rms_path)

    distance_error_path = plot_dir / "distance_error_before_after_boxplot.png"
    if save_boxplot(
        data=[
            finite_array(
                [result.init_trans_err_norm_mm * 1e-3 for result in results]
            ),
            finite_array([result.trans_err_norm_mm * 1e-3 for result in results]),
        ],
        labels=["Before", "After"],
        out_path=distance_error_path,
        ylabel="Distance error [m]",
        title="Distance error before and after calibration",
        semilogy=True,
    ):
        saved.append(distance_error_path)

    rotation_error_path = plot_dir / "rotation_error_before_after_boxplot.png"
    if save_boxplot(
        data=[
            finite_array([result.init_rot_err_angle_deg for result in results]),
            finite_array([result.rot_err_angle_deg for result in results]),
        ],
        labels=["Before", "After"],
        out_path=rotation_error_path,
        ylabel="Rotation error [degrees]",
        title="Rotation error before and after calibration",
        semilogy=True,
    ):
        saved.append(rotation_error_path)

    return saved


def save_plane_scene_plot(
    T_ef_s_true: np.ndarray,
    scans_by_plane: dict[int, list[LaserScan]],
    planes: Sequence[Plane],
    out_path: Path,
    plane_size_mm: float = 1500.0,
) -> Path:
    """Plot planar targets and reconstructed profiles in the robot base frame.

    Args:
        T_ef_s_true: Ground-truth hand-eye transform used only for simulation
            visualization.
        scans_by_plane: Plane ID to laser-profile scans.
        planes: Plane equations as ``(unit_normal, distance_mm)``.
        out_path: Output PNG path.
        plane_size_mm: Minimum side length of each visualized plane patch.
    """

    if not scans_by_plane:
        raise ValueError("scans_by_plane must not be empty")
    if not planes:
        raise ValueError("planes must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    T_ef_s_true = np.asarray(T_ef_s_true, dtype=float).reshape(4, 4)
    profile_points_by_plane: dict[int, list[np.ndarray]] = {}
    residuals: list[np.ndarray] = []

    for plane_id, scans in scans_by_plane.items():
        if plane_id < 0 or plane_id >= len(planes):
            continue

        plane_n, plane_l = planes[plane_id]
        plane_n = normalize_vector(plane_n)
        profile_points_by_plane[plane_id] = []

        for scan in scans:
            T_base_s = scan.T_base_ef @ T_ef_s_true
            points_base = transform_points(T_base_s, scan.valid_points_s)
            profile_points_by_plane[plane_id].append(points_base)
            residuals.append(points_base @ plane_n - float(plane_l))

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    default_colors = ["tab:blue", "tab:orange", "tab:green"]

    for plane_id, (plane_n, plane_l) in enumerate(planes):
        plane_n = normalize_vector(plane_n)
        color = default_colors[plane_id % len(default_colors)]
        u, v = plane_basis(plane_n)
        point_sets = profile_points_by_plane.get(plane_id, [])

        if point_sets:
            points_all = np.vstack(point_sets)
            center = np.mean(points_all, axis=0)
            center -= (float(center @ plane_n) - float(plane_l)) * plane_n
            uv = np.column_stack(
                [(points_all - center) @ u, (points_all - center) @ v]
            )
            extent = max(
                float(np.max(np.abs(uv))) * 2.2,
                float(plane_size_mm),
            )
        else:
            center = plane_n * float(plane_l)
            extent = float(plane_size_mm)

        grid = np.linspace(-0.5 * extent, 0.5 * extent, 28)
        grid_u, grid_v = np.meshgrid(grid, grid)
        surface = (
            center.reshape(3, 1, 1)
            + u.reshape(3, 1, 1) * grid_u
            + v.reshape(3, 1, 1) * grid_v
        )

        ax.plot_surface(
            surface[0],
            surface[1],
            surface[2],
            color=color,
            alpha=0.18,
            linewidth=0,
            shade=False,
        )

        label_position = center + 0.42 * extent * (u + v)
        ax.text(
            float(label_position[0]),
            float(label_position[1]),
            float(label_position[2]),
            f"Plane {plane_id}",
            fontsize=10,
            color=color,
        )

    for plane_id, point_sets in profile_points_by_plane.items():
        color = default_colors[plane_id % len(default_colors)]
        for points_base in point_sets:
            ax.plot(
                points_base[:, 0],
                points_base[:, 1],
                points_base[:, 2],
                color=color,
                linewidth=1.35,
                alpha=0.72,
            )

    if residuals:
        residual_array = np.concatenate(
            [values.reshape(-1) for values in residuals]
        )
        residual_rms_mm = float(np.sqrt(np.mean(residual_array**2)))
    else:
        residual_rms_mm = float("nan")

    ax.set_title(
        "Debug scene: calibration planes and generated profiles\n"
        f"point-to-plane RMS = {residual_rms_mm:.4g} mm"
    )
    ax.set_xlabel("base X [mm]")
    ax.set_ylabel("base Y [mm]")
    ax.set_zlabel("base Z [mm]")
    ax.grid(True, alpha=0.3)
    _set_axes_equal_3d(ax)

    legend_handles = [
        Patch(
            facecolor=default_colors[index % len(default_colors)],
            edgecolor="none",
            alpha=0.18,
            label=f"Plane {index}",
        )
        for index in range(len(planes))
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=default_colors[index % len(default_colors)],
                linewidth=2.0,
                label=f"Profiles on Plane {index}",
            )
            for index in range(len(planes))
        ]
    )
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.view_init(elev=24, azim=-55)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_plane_scene_plot_3d(
    T_ef_s_true: np.ndarray,
    scans_by_plane: dict[int, list[LaserScan]],
    planes: Sequence[Plane],
    out_path: Path,
    plane_size_mm: float = 700.0,
) -> Path:
    """Save an interactive 3-D visualization of simulated GT planes.

    The output is an HTML file that supports mouse rotation, zooming,
    panning, and legend-based visibility control.

    The plane normals and offsets are the actual GT plane parameters used
    in the simulation:

        n.T @ p_base = l

    The laser profile points are reconstructed in the robot-base frame using:

        T_base_sensor = T_base_ef @ T_ef_s_true
    """

    if not scans_by_plane:
        raise ValueError("scans_by_plane must not be empty")

    if not planes:
        raise ValueError("planes must not be empty")

    import plotly.graph_objects as go

    T_ef_s_true = np.asarray(
        T_ef_s_true,
        dtype=float,
    ).reshape(4, 4)

    normalized_planes: list[tuple[np.ndarray, float]] = [
        (
            normalize_vector(plane_n),
            float(plane_l),
        )
        for plane_n, plane_l in planes
    ]

    colors = [
        "rgb(31,119,180)",
        "rgb(255,127,14)",
        "rgb(44,160,44)",
    ]

    # ------------------------------------------------------------
    # Reconstruct the simulated profiles in robot-base coordinates.
    # ------------------------------------------------------------
    profile_points_by_plane: dict[int, list[np.ndarray]] = {}
    residuals: list[np.ndarray] = []

    for plane_id, scans in scans_by_plane.items():
        if plane_id < 0 or plane_id >= len(normalized_planes):
            continue

        plane_n, plane_l = normalized_planes[plane_id]
        profile_points_by_plane[plane_id] = []

        for scan in scans:
            T_base_s = scan.T_base_ef @ T_ef_s_true

            points_base = transform_points(
                T_base_s,
                scan.valid_points_s,
            )

            profile_points_by_plane[plane_id].append(
                points_base
            )

            residuals.append(
                points_base @ plane_n - plane_l
            )

    # ------------------------------------------------------------
    # Calculate the common intersection for reference.
    # This does not alter any plane geometry.
    # ------------------------------------------------------------
    plane_intersection: np.ndarray | None = None

    if len(normalized_planes) == 3:
        normal_matrix = np.stack(
            [
                plane_n
                for plane_n, _ in normalized_planes
            ],
            axis=0,
        )

        offset_vector = np.array(
            [
                plane_l
                for _, plane_l in normalized_planes
            ],
            dtype=float,
        )

        if np.linalg.matrix_rank(normal_matrix) == 3:
            plane_intersection = np.linalg.solve(
                normal_matrix,
                offset_vector,
            )

    figure = go.Figure()

    patch_centers: dict[int, np.ndarray] = {}
    patch_extents: dict[int, float] = {}

    # ------------------------------------------------------------
    # Draw each actual GT plane.
    # ------------------------------------------------------------
    for plane_id, (plane_n, plane_l) in enumerate(
        normalized_planes
    ):
        color = colors[plane_id % len(colors)]
        u, v = plane_basis(plane_n)

        point_sets = profile_points_by_plane.get(
            plane_id,
            [],
        )

        if point_sets:
            points_all = np.vstack(point_sets)

            # Display patch near the part of the infinite plane that was scanned.
            center = np.mean(points_all, axis=0)

            # Project the center exactly onto the GT plane.
            center = center - (
                float(center @ plane_n) - plane_l
            ) * plane_n

            uv_coordinates = np.column_stack(
                [
                    (points_all - center) @ u,
                    (points_all - center) @ v,
                ]
            )

            extent = max(
                2.2 * float(np.max(np.abs(uv_coordinates))),
                float(plane_size_mm),
            )
        else:
            # Closest point on the plane to the base-frame origin.
            center = plane_l * plane_n
            extent = float(plane_size_mm)

        patch_centers[plane_id] = center
        patch_extents[plane_id] = extent

        grid = np.linspace(
            -0.5 * extent,
            0.5 * extent,
            20,
        )

        grid_u, grid_v = np.meshgrid(
            grid,
            grid,
        )

        surface = (
            center.reshape(3, 1, 1)
            + u.reshape(3, 1, 1) * grid_u
            + v.reshape(3, 1, 1) * grid_v
        )

        figure.add_trace(
            go.Surface(
                x=surface[0],
                y=surface[1],
                z=surface[2],
                surfacecolor=np.zeros_like(surface[0]),
                colorscale=[
                    [0.0, color],
                    [1.0, color],
                ],
                showscale=False,
                opacity=0.25,
                name=f"GT Plane {plane_id}",
                showlegend=True,
                hovertemplate=(
                    f"GT Plane {plane_id}<br>"
                    f"n = [{plane_n[0]:.4f}, "
                    f"{plane_n[1]:.4f}, "
                    f"{plane_n[2]:.4f}]<br>"
                    f"l = {plane_l:.3f} mm"
                    "<extra></extra>"
                ),
            )
        )

        # Plane normal at the displayed patch center.
        normal_end = (
            center
            + 0.22 * extent * plane_n
        )

        figure.add_trace(
            go.Scatter3d(
                x=[center[0], normal_end[0]],
                y=[center[1], normal_end[1]],
                z=[center[2], normal_end[2]],
                mode="lines+markers",
                line={
                    "color": color,
                    "width": 8,
                },
                marker={
                    "color": color,
                    "size": [3, 6],
                },
                name=f"Normal {plane_id}",
                hovertemplate=(
                    f"Normal {plane_id}<br>"
                    f"[{plane_n[0]:.4f}, "
                    f"{plane_n[1]:.4f}, "
                    f"{plane_n[2]:.4f}]"
                    "<extra></extra>"
                ),
            )
        )

    # ------------------------------------------------------------
    # Draw simulated laser profiles.
    # ------------------------------------------------------------
    for plane_id, point_sets in profile_points_by_plane.items():
        color = colors[plane_id % len(colors)]

        for scan_index, points_base in enumerate(point_sets):
            figure.add_trace(
                go.Scatter3d(
                    x=points_base[:, 0],
                    y=points_base[:, 1],
                    z=points_base[:, 2],
                    mode="lines",
                    line={
                        "color": color,
                        "width": 3,
                    },
                    opacity=0.75,
                    name=f"Profiles on Plane {plane_id}",
                    legendgroup=f"profile_{plane_id}",
                    showlegend=(scan_index == 0),
                    hovertemplate=(
                        f"Plane {plane_id} profile<br>"
                        "X: %{x:.3f} mm<br>"
                        "Y: %{y:.3f} mm<br>"
                        "Z: %{z:.3f} mm"
                        "<extra></extra>"
                    ),
                )
            )

    # ------------------------------------------------------------
    # Draw base-frame coordinate axes.
    # ------------------------------------------------------------
    all_profile_points = [
        points
        for point_sets in profile_points_by_plane.values()
        for points in point_sets
    ]

    if all_profile_points:
        all_points = np.vstack(all_profile_points)
        scene_span = float(
            np.max(
                np.ptp(all_points, axis=0)
            )
        )
    else:
        scene_span = float(plane_size_mm)

    axis_length = max(
        0.25 * scene_span,
        100.0,
    )

    base_origin = np.zeros(3)

    base_axes = [
        ("Base X", np.array([1.0, 0.0, 0.0]), "red"),
        ("Base Y", np.array([0.0, 1.0, 0.0]), "green"),
        ("Base Z", np.array([0.0, 0.0, 1.0]), "blue"),
    ]

    for axis_name, axis_direction, axis_color in base_axes:
        axis_end = (
            base_origin
            + axis_length * axis_direction
        )

        figure.add_trace(
            go.Scatter3d(
                x=[base_origin[0], axis_end[0]],
                y=[base_origin[1], axis_end[1]],
                z=[base_origin[2], axis_end[2]],
                mode="lines+text",
                line={
                    "color": axis_color,
                    "width": 8,
                },
                text=["", axis_name],
                textposition="top center",
                name=axis_name,
                hoverinfo="skip",
            )
        )

    # ------------------------------------------------------------
    # Draw mathematical common intersection only as a reference point.
    # ------------------------------------------------------------
    if plane_intersection is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[plane_intersection[0]],
                y=[plane_intersection[1]],
                z=[plane_intersection[2]],
                mode="markers",
                marker={
                    "color": "black",
                    "size": 6,
                    "symbol": "diamond",
                },
                name="Plane intersection",
                hovertemplate=(
                    "Mathematical plane intersection<br>"
                    "X: %{x:.3f} mm<br>"
                    "Y: %{y:.3f} mm<br>"
                    "Z: %{z:.3f} mm"
                    "<extra></extra>"
                ),
            )
        )

    # ------------------------------------------------------------
    # Numerical plane-angle information.
    # ------------------------------------------------------------
    angle_strings: list[str] = []

    for first_id in range(len(normalized_planes)):
        for second_id in range(
            first_id + 1,
            len(normalized_planes),
        ):
            first_normal = normalized_planes[first_id][0]
            second_normal = normalized_planes[second_id][0]

            cosine = float(
                np.clip(
                    abs(first_normal @ second_normal),
                    0.0,
                    1.0,
                )
            )

            angle_deg = float(
                np.degrees(
                    np.arccos(cosine)
                )
            )

            angle_strings.append(
                f"{first_id}-{second_id}: {angle_deg:.3f}°"
            )

    if residuals:
        residual_array = np.concatenate(
            [
                residual.reshape(-1)
                for residual in residuals
            ]
        )

        residual_rms_mm = float(
            np.sqrt(
                np.mean(residual_array**2)
            )
        )
    else:
        residual_rms_mm = float("nan")

    figure.update_layout(
        title={
            "text": (
                "Interactive 3-D simulation scene"
                "<br>"
                f"<sup>GT plane angles: "
                f"{', '.join(angle_strings)} | "
                f"point-to-plane RMS: "
                f"{residual_rms_mm:.4g} mm</sup>"
            ),
            "x": 0.5,
        },
        scene={
            "xaxis_title": "Base X [mm]",
            "yaxis_title": "Base Y [mm]",
            "zaxis_title": "Base Z [mm]",
            "aspectmode": "data",
            "camera": {
                "eye": {
                    "x": 1.5,
                    "y": -1.5,
                    "z": 1.2,
                }
            },
        },
        legend={
            "itemsizing": "constant",
        },
        width=1100,
        height=850,
        margin={
            "l": 20,
            "r": 20,
            "t": 90,
            "b": 20,
        },
    )

    out_path = Path(out_path)

    if out_path.suffix.lower() != ".html":
        out_path = out_path.with_suffix(".html")

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
    )

    return out_path

def save_history_plot(
    histories: Sequence[Sequence[float]],
    out_path: Path,
    ylabel: str,
    title: str,
    semilogy: bool = True,
    x_start: int = 1,
) -> bool:
    """Save multiple iteration histories and their per-iteration median."""

    values = _pad_histories(histories)
    if values.size == 0:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(x_start, x_start + values.shape[1])
    fig, ax = plt.subplots(figsize=(8, 5))

    for row in values:
        valid = np.isfinite(row)
        if not np.any(valid):
            continue
        if semilogy:
            ax.semilogy(x[valid], row[valid], linewidth=0.8, alpha=0.35)
        else:
            ax.plot(x[valid], row[valid], linewidth=0.8, alpha=0.35)

    median = np.nanmedian(values, axis=0)
    valid = np.isfinite(median)
    if np.any(valid):
        if semilogy:
            ax.semilogy(x[valid], median[valid], linewidth=2.5, label="median")
        else:
            ax.plot(x[valid], median[valid], linewidth=2.5, label="median")
        ax.legend()

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_boxplot(
    data: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: Path,
    ylabel: str,
    title: str,
    semilogy: bool = True,
) -> bool:
    """Save a common before/after or method-comparison boxplot."""

    if not data or any(len(values) == 0 for values in data):
        return False
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.boxplot(data, tick_labels=list(labels), showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=list(labels), showmeans=True)

    if semilogy:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def finite_array(values: Sequence[float]) -> np.ndarray:
    """Convert values to a one-dimensional array and remove NaN/Inf."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _result_to_csv_row(result: Any) -> dict[str, object]:
    if not is_dataclass(result):
        raise TypeError("each result must be a dataclass instance")

    row = asdict(result)
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = json.dumps(value)
    return row


def _transform_with_translation_in_metres(T: np.ndarray) -> np.ndarray:
    T_m = np.asarray(T, dtype=float).reshape(4, 4).copy()
    T_m[:3, 3] *= 1e-3
    return T_m


def _pad_histories(histories: Sequence[Sequence[float]]) -> np.ndarray:
    non_empty = [list(history) for history in histories if len(history) > 0]
    if not non_empty:
        return np.empty((0, 0), dtype=float)

    max_length = max(len(history) for history in non_empty)
    array = np.full((len(non_empty), max_length), np.nan, dtype=float)

    for index, history in enumerate(non_empty):
        array[index, : len(history)] = history

    return array



def _set_axes_equal_3d(ax: Any) -> None:
    limits = np.asarray(
        [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()],
        dtype=float,
    )
    centers = np.mean(limits, axis=1)
    radius = 0.5 * float(np.max(np.abs(limits[:, 1] - limits[:, 0])))

    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)