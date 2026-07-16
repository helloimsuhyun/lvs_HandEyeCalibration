from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

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


def iter_physical_transform_errors(
    T_history: Sequence[np.ndarray],
    T_true: np.ndarray,
    T_initial: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    """Return translation-norm [mm] and geodesic rotation [deg] histories."""

    estimates: list[np.ndarray] = []
    if T_initial is not None:
        estimates.append(T_initial)
    estimates.extend(T_history)

    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    translation_mm: list[float] = []
    rotation_deg: list[float] = []
    for T_est in estimates:
        T_est = np.asarray(T_est, dtype=float).reshape(4, 4)
        translation_mm.append(
            float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
        )
        relative_rotation = T_est[:3, :3].T @ T_true[:3, :3]
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        )
        rotation_deg.append(float(np.degrees(np.arccos(cosine))))

    return translation_mm, rotation_deg


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


def save_iteration_history_csv(results: Sequence[Any], path: Path) -> Path:
    """Save solver histories in analysis-friendly long format.

    Each row is one trial/solver-step pair.  Unlike the JSON arrays retained in
    the trial CSV for backwards compatibility, this layout can be grouped,
    filtered and plotted directly with pandas, R, MATLAB or a spreadsheet.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_fields = (
        ("iter_translation_error_norm_mm", "translation_error_mm"),
        ("iter_rotation_geodesic_error_deg", "rotation_error_deg"),
        ("plane_rms_history_mm", "plane_rms_mm"),
        ("iter_T_frob_error", "transform_frobenius_error"),
        ("iter_gauge_parallel_abs_error_mm", "gauge_parallel_abs_error_mm"),
        ("iter_gauge_perpendicular_error_mm", "gauge_perpendicular_error_mm"),
        ("iter_plane_offset_estimate_mm", "iter_plane_offset_estimate_mm"),
        ("iter_plane_offset_error_mm", "iter_plane_offset_error_mm"),
        ("iter_err_t_sensor_z_mm", "iter_err_t_sensor_z_mm"),
        (
            "iter_normal_sensor_z_dot_mean",
            "iter_normal_sensor_z_dot_mean",
        ),
    )
    identity_fields = (
        "system_idx",
        "mode",
        "init_mode",
        "plane_offset_mode",
        "pose_geometry",
        "n_planes",
        "n_scans",
        "n_points_per_scan",
        "converged",
        "paper_success",
    )
    fieldnames = [
        *identity_fields,
        "step",
        "is_initial",
        "is_final",
        *[column for _attribute, column in history_fields],
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        histories = {
            column: list(getattr(result, attribute, []))
            for attribute, column in history_fields
        }
        step_count = max((len(values) for values in histories.values()), default=0)
        for step in range(step_count):
            row = {
                field: _csv_config_value(getattr(result, field, ""))
                for field in identity_fields
            }
            row.update(
                {
                    "step": step,
                    "is_initial": step == 0,
                    "is_final": step == step_count - 1,
                }
            )
            for column, values in histories.items():
                row[column] = values[step] if step < len(values) else ""
            rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _csv_config_value(value: Any) -> Any:
    """Convert argparse/config values to stable single-cell CSV values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return value


def _finite_metric_pairs(
    results: Sequence[Any],
    attribute: str,
) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    for fallback_idx, result in enumerate(results):
        if not hasattr(result, attribute):
            continue
        try:
            value = float(getattr(result, attribute))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            pairs.append((value, int(getattr(result, "system_idx", fallback_idx))))
    return pairs


def _add_metric_summary(
    row: dict[str, Any],
    results: Sequence[Any],
    attribute: str,
    prefix: str,
    *,
    signed: bool = False,
) -> None:
    """Add robust scalar statistics and their extreme trial IDs to one row."""
    pairs = _finite_metric_pairs(results, attribute)
    values = np.asarray([value for value, _idx in pairs], dtype=float)
    row[f"{prefix}_count"] = int(values.size)
    if values.size == 0:
        for suffix in (
            "min", "p05", "p25", "median", "mean", "std", "rmse",
            "p75", "p95", "p99", "max",
        ):
            row[f"{prefix}_{suffix}"] = float("nan")
        row[f"{prefix}_min_system_idx"] = ""
        row[f"{prefix}_max_system_idx"] = ""
        if signed:
            row[f"{prefix}_max_abs"] = float("nan")
            row[f"{prefix}_max_abs_system_idx"] = ""
        return

    min_position = int(np.argmin(values))
    max_position = int(np.argmax(values))
    row.update(
        {
            f"{prefix}_min": float(values[min_position]),
            f"{prefix}_min_system_idx": int(pairs[min_position][1]),
            f"{prefix}_p05": float(np.percentile(values, 5.0)),
            f"{prefix}_p25": float(np.percentile(values, 25.0)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
            f"{prefix}_p75": float(np.percentile(values, 75.0)),
            f"{prefix}_p95": float(np.percentile(values, 95.0)),
            f"{prefix}_p99": float(np.percentile(values, 99.0)),
            f"{prefix}_max": float(values[max_position]),
            f"{prefix}_max_system_idx": int(pairs[max_position][1]),
        }
    )
    if signed:
        max_abs_position = int(np.argmax(np.abs(values)))
        row[f"{prefix}_max_abs"] = float(abs(values[max_abs_position]))
        row[f"{prefix}_max_abs_system_idx"] = int(
            pairs[max_abs_position][1]
        )


def save_experiment_summary_csv(
    results: Sequence[Any],
    path: Path,
    *,
    requested_systems: int,
    failed_systems: int,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Save one comprehensive aggregate row for a benchmark execution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(results)
    converged = sum(bool(getattr(result, "converged", False)) for result in results)
    paper_success = sum(
        bool(getattr(result, "paper_success", False)) for result in results
    )
    nonlinear_requested = bool((config or {}).get("nonlinear_refine", False)) or any(
        bool(getattr(result, "nonlinear_refined", False)) for result in results
    )
    nonlinear_success = sum(
        bool(getattr(result, "nonlinear_success", False)) for result in results
    )

    row: dict[str, Any] = {
        "requested_systems": int(requested_systems),
        "completed_systems": int(completed),
        "failed_systems": int(failed_systems),
        "completion_rate_requested": (
            completed / requested_systems if requested_systems else 0.0
        ),
        "converged_count": int(converged),
        "convergence_rate_completed": converged / completed if completed else 0.0,
        "convergence_rate_requested": (
            converged / requested_systems if requested_systems else 0.0
        ),
        "paper_success_count": int(paper_success),
        "paper_success_rate_completed": (
            paper_success / completed if completed else 0.0
        ),
        "paper_success_rate_requested": (
            paper_success / requested_systems if requested_systems else 0.0
        ),
        "nonlinear_requested": bool(nonlinear_requested),
        "nonlinear_success_count": int(nonlinear_success),
        "nonlinear_success_rate_completed": (
            nonlinear_success / completed
            if completed and nonlinear_requested
            else float("nan")
        ),
    }

    if results:
        first = results[0]
        for attribute in (
            "mode", "init_mode", "plane_offset_mode", "pose_geometry",
            "n_planes", "n_scans", "n_points_per_scan",
            "n_reference_scans",
        ):
            if hasattr(first, attribute):
                row[attribute] = _csv_config_value(getattr(first, attribute))

    metric_specs = (
        ("trans_err_norm_mm", "translation_error_mm", False),
        ("rot_err_angle_deg", "rotation_error_deg", False),
        ("err_tx_mm", "translation_x_error_mm", True),
        ("err_ty_mm", "translation_y_error_mm", True),
        ("err_tz_mm", "translation_z_error_mm", True),
        ("err_t_sensor_x_mm", "translation_sensor_x_error_mm", True),
        ("err_t_sensor_y_mm", "translation_sensor_y_error_mm", True),
        ("err_t_sensor_z_mm", "translation_sensor_z_error_mm", True),
        ("err_r_sensor_x_deg", "rotation_sensor_x_error_deg", True),
        ("err_r_sensor_y_deg", "rotation_sensor_y_error_deg", True),
        ("err_r_sensor_z_deg", "rotation_sensor_z_error_deg", True),
        ("estimated_plane_offset_mm", "estimated_plane_offset_mm", False),
        (
            "ground_truth_plane_offset_mm",
            "ground_truth_plane_offset_mm",
            False,
        ),
        ("plane_offset_error_mm", "plane_offset_error_mm", True),
        ("normal_sensor_z_dot_mean", "normal_sensor_z_dot_mean", True),
        ("err_rx_deg", "rotation_x_error_deg", True),
        ("err_ry_deg", "rotation_y_error_deg", True),
        ("err_rz_deg", "rotation_z_error_deg", True),
        ("init_trans_err_norm_mm", "initial_translation_error_mm", False),
        ("init_rot_err_angle_deg", "initial_rotation_error_deg", False),
        ("iterations", "iterations", False),
        ("rank_last", "final_rank", False),
        ("cond_last", "final_condition_number", False),
        ("final_linear_plane_rms_mm", "final_linear_plane_rms_mm", False),
        ("nonlinear_nfev", "nonlinear_nfev", False),
        ("nonlinear_initial_rms_mm", "nonlinear_initial_rms_mm", False),
        ("nonlinear_final_rms_mm", "nonlinear_final_rms_mm", False),
        (
            "nonlinear_delta_translation_mm",
            "nonlinear_translation_update_mm",
            False,
        ),
        (
            "nonlinear_delta_rotation_deg",
            "nonlinear_rotation_update_deg",
            False,
        ),
        ("gauge_parallel_error_mm", "gauge_parallel_error_mm", True),
        ("gauge_perpendicular_error_mm", "gauge_perpendicular_error_mm", False),
        ("gauge_axis_angle_deg", "gauge_axis_angle_deg", False),
    )
    for attribute, prefix, signed in metric_specs:
        _add_metric_summary(
            row,
            results,
            attribute,
            prefix,
            signed=signed,
        )

    # Histories carry the aligned initial and final solver states. Expose their
    # final values as ordinary summary metrics without altering trial schemas.
    history_results: list[Any] = []
    for result in results:
        plane_history = getattr(result, "plane_rms_history_mm", [])
        frobenius_history = getattr(result, "iter_T_frob_error", [])
        proxy = type("HistorySummary", (), {})()
        proxy.system_idx = int(getattr(result, "system_idx", len(history_results)))
        proxy.final_self_fitted_plane_rms_mm = (
            float(plane_history[-1]) if plane_history else float("nan")
        )
        proxy.final_transform_frobenius_error = (
            float(frobenius_history[-1]) if frobenius_history else float("nan")
        )
        history_results.append(proxy)
    _add_metric_summary(
        row,
        history_results,
        "final_self_fitted_plane_rms_mm",
        "final_self_fitted_plane_rms_mm",
    )
    _add_metric_summary(
        row,
        history_results,
        "final_transform_frobenius_error",
        "final_transform_frobenius_error",
    )

    for key, value in sorted((config or {}).items()):
        row[f"config_{key}"] = _csv_config_value(value)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def save_failures_csv(
    failures: Sequence[Mapping[str, Any]],
    path: Path,
) -> Path:
    """Save every failed trial, including an empty header-only failure log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    default_fields = ["system_idx", "error_type", "error_message"]
    fields = list(default_fields)
    for failure in failures:
        for key in failure:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for failure in failures:
            writer.writerow({key: _csv_config_value(value) for key, value in failure.items()})
    return path


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
    final_plane_rms = finite_array(
        [
            result.plane_rms_history_mm[-1]
            for result in results
            if result.plane_rms_history_mm
        ]
    )
    if final_plane_rms.size:
        print(
            "final fitted-plane RMS [mm]: "
            f"median={np.median(final_plane_rms):.6g}, "
            f"mean={np.mean(final_plane_rms):.6g}, "
            f"max={np.max(final_plane_rms):.6g}"
        )


def save_calibration_plots(
    results: Sequence[CalibrationTrialLike],
    plot_dir: Path,
    sensor_noise_std_mm: float | None = None,
) -> list[Path]:
    """Save only the four core benchmark figures.

    ``sensor_noise_std_mm`` is retained for CLI/API compatibility.  Noise and
    plane-residual diagnostics belong in the CSV outputs rather than in the
    default figure set.
    """

    if not results:
        return []
    if sensor_noise_std_mm is not None:
        sensor_noise_std_mm = float(sensor_noise_std_mm)
        if not np.isfinite(sensor_noise_std_mm) or sensor_noise_std_mm < 0.0:
            raise ValueError("sensor_noise_std_mm must be finite and non-negative")

    plot_dir.mkdir(parents=True, exist_ok=True)
    # A rerun into an existing directory must still leave only the current
    # default report.  Remove only filenames formerly created automatically;
    # user-requested debug plots and unrelated files are preserved.
    legacy_default_filenames = (
        "T_true_minus_T_estimated_per_iter.png",
        "T_true_minus_T_estimated_per_iter_after_step_1.png",
        "T_true_minus_T_estimated_physical_components_per_iter.png",
        "rms_distance_from_points_to_plane_m.png",
        "rms_distance_from_points_to_plane_m_after_step_1.png",
        "rms_distance_from_points_to_plane_per_iter.png",
        "rms_distance_from_points_to_plane_per_iter_after_step_1.png",
        "paper_style_gt_translation_rotation_boxplots.png",
        "gt_signed_component_error_boxplots.png",
        "distance_error_before_after_boxplot.png",
        "rotation_error_before_after_boxplot.png",
        "translation_error_vs_sensor_z_gauge.png",
        "translation_error_direction_sensor_frame.png",
        "translation_error_direction_diagnostics.png",
        "translation_error_direction_quick_view.png",
        "translation_error_direction_3d.html",
    )
    for filename in legacy_default_filenames:
        (plot_dir / filename).unlink(missing_ok=True)

    saved: list[Path] = []
    xlabel = "Solver step (0 = initial)"
    requested_plots = (
        (
            "iter_translation_error_norm_mm",
            "translation_error_by_iteration.png",
            "Translation error [mm]",
            "",
            "tab20",
            True,
            0.58,
        ),
        (
            "iter_rotation_geodesic_error_deg",
            "rotation_error_by_iteration.png",
            "Rotation error [deg]",
            "",
            "tab20",
            True,
            0.58,
        ),
        (
            "plane_rms_history_mm",
            "fitted_plane_rms_by_iteration.png",
            "Fitted-plane RMS [mm]",
            "",
            "tab20",
            True,
            0.58,
        ),
    )
    for (
        attribute,
        filename,
        ylabel,
        title,
        colormap,
        dense_log_ticks,
        line_alpha,
    ) in requested_plots:
        path = plot_dir / filename
        if save_history_plot(
            histories=[getattr(result, attribute, []) for result in results],
            out_path=path,
            ylabel=ylabel,
            title=title,
            semilogy=True,
            x_start=0,
            xlabel=xlabel,
            colormap=colormap,
            dense_log_ticks=dense_log_ticks,
            line_alpha=line_alpha,
        ):
            saved.append(path)

    final_error_path = plot_dir / "final_translation_rotation_error_boxplot.png"
    if save_paper_style_gt_error_distribution(results, final_error_path):
        saved.append(final_error_path)

    component_path = plot_dir / "final_error_sensor_axis_components.png"
    if save_gt_component_error_distribution(results, component_path):
        saved.append(component_path)

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
    sensor_origins_by_plane: dict[int, list[np.ndarray]] = {}
    sensor_view_directions_by_plane: dict[int, list[np.ndarray]] = {}
    residuals: list[np.ndarray] = []

    for plane_id, scans in scans_by_plane.items():
        if plane_id < 0 or plane_id >= len(normalized_planes):
            continue

        plane_n, plane_l = normalized_planes[plane_id]
        profile_points_by_plane[plane_id] = []
        sensor_origins_by_plane[plane_id] = []
        sensor_view_directions_by_plane[plane_id] = []

        for scan in scans:
            T_base_s = scan.T_base_ef @ T_ef_s_true
            sensor_origins_by_plane[plane_id].append(T_base_s[:3, 3].copy())
            sensor_view_directions_by_plane[plane_id].append(
                -T_base_s[:3, 2].copy()
            )

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
    sensor_axis_length = max(0.08 * scene_span, 25.0)

    # Sensor origins and their optical -Z viewing directions make the pose
    # distribution directly inspectable without drawing 3 full axes per scan.
    for plane_id, origins_list in sensor_origins_by_plane.items():
        if not origins_list:
            continue
        color = colors[plane_id % len(colors)]
        origins = np.asarray(origins_list, dtype=float)
        directions = np.asarray(
            sensor_view_directions_by_plane[plane_id],
            dtype=float,
        )
        figure.add_trace(
            go.Scatter3d(
                x=origins[:, 0],
                y=origins[:, 1],
                z=origins[:, 2],
                mode="markers",
                marker={
                    "color": color,
                    "size": 4,
                    "symbol": "circle-open",
                    "line": {"color": color, "width": 2},
                },
                name=f"Sensor origins on Plane {plane_id}",
                legendgroup=f"sensor_{plane_id}",
                hovertemplate=(
                    f"Sensor origin (Plane {plane_id})<br>"
                    "Base X: %{x:.3f} mm<br>"
                    "Base Y: %{y:.3f} mm<br>"
                    "Base Z: %{z:.3f} mm"
                    "<extra></extra>"
                ),
            )
        )

        view_x: list[float | None] = []
        view_y: list[float | None] = []
        view_z: list[float | None] = []
        for origin, direction in zip(origins, directions):
            endpoint = origin + sensor_axis_length * direction
            view_x.extend([float(origin[0]), float(endpoint[0]), None])
            view_y.extend([float(origin[1]), float(endpoint[1]), None])
            view_z.extend([float(origin[2]), float(endpoint[2]), None])
        figure.add_trace(
            go.Scatter3d(
                x=view_x,
                y=view_y,
                z=view_z,
                mode="lines",
                line={"color": color, "width": 3},
                opacity=0.55,
                name=f"Sensor -Z view axes on Plane {plane_id}",
                legendgroup=f"sensor_{plane_id}",
                hoverinfo="skip",
            )
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

    geometry_summary = (
        f"GT plane angles: {', '.join(angle_strings)}"
        if angle_strings
        else f"GT planes: {len(normalized_planes)}"
    )

    figure.update_layout(
        title={
            "text": (
                "Interactive 3-D simulation scene"
                "<br>"
                f"<sup>{geometry_summary} | "
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

def _histories_from_step(
    histories: Sequence[Sequence[float]],
    step: int,
) -> list[list[float]]:
    """Slice aligned state histories, skipping a zoom with only one state."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if not histories or max((len(history) for history in histories), default=0) <= step + 1:
        return []
    return [list(history)[step:] for history in histories if len(history) > step]


def _draw_history_axis(
    ax: Any,
    histories: Sequence[Sequence[float]],
    *,
    ylabel: str,
    title: str,
    semilogy: bool,
    x_start: int,
    xlabel: str,
    horizontal_reference_value: float | None = None,
    horizontal_reference_label: str | None = None,
    median_color: str = "#184e77",
    show_legend: bool = True,
    colormap: str = "turbo",
    dense_log_ticks: bool = False,
    line_alpha: float = 0.9,
) -> bool:
    """Draw one thin, uniquely colored line per system (paper Fig. 5 style)."""
    values = _pad_histories(histories)
    if values.size == 0:
        return False

    x = np.arange(x_start, x_start + values.shape[1])
    plotted = False
    import matplotlib.pyplot as plt

    color_map = plt.get_cmap(colormap)
    if hasattr(color_map, "colors"):
        base_colors = np.asarray(color_map.colors)
        colors = base_colors[np.arange(values.shape[0]) % len(base_colors)]
    else:
        colors = color_map(
            np.linspace(0.02, 0.98, max(values.shape[0], 2))
        )
    for row_index, row in enumerate(values):
        valid = np.isfinite(row)
        if semilogy:
            valid &= row > 0.0
        if not np.any(valid):
            continue
        ax.plot(
            x[valid],
            row[valid],
            color=colors[row_index],
            linewidth=0.75,
            alpha=line_alpha,
            zorder=1,
        )
        plotted = True

    if not plotted:
        return False

    if horizontal_reference_value is not None:
        reference_value = float(horizontal_reference_value)
        if not np.isfinite(reference_value):
            raise ValueError("horizontal reference value must be finite")
        if semilogy and reference_value <= 0.0:
            raise ValueError(
                "horizontal reference value must be positive on a log axis"
            )
        ax.axhline(
            reference_value,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=horizontal_reference_label,
            zorder=2,
        )

    if semilogy:
        ax.set_yscale("log")
        if dense_log_ticks:
            from matplotlib.ticker import (
                LogFormatterMathtext,
                LogLocator,
                NullFormatter,
            )

            positive_values = values[np.isfinite(values) & (values > 0.0)]
            lower_exponent = int(np.floor(np.log10(np.min(positive_values))))
            upper_exponent = int(np.ceil(np.log10(np.max(positive_values))))
            if upper_exponent <= lower_exponent:
                upper_exponent = lower_exponent + 1
            ax.set_ylim(10.0**lower_exponent, 10.0**upper_exponent)

            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0,))
            )
            ax.yaxis.set_major_formatter(
                LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=(2.0, 5.0))
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.65, alpha=0.65)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.margins(x=0.02)
    return True


def save_history_plot(
    histories: Sequence[Sequence[float]],
    out_path: Path,
    ylabel: str,
    title: str,
    semilogy: bool = True,
    x_start: int = 1,
    xlabel: str = "Iteration",
    horizontal_reference_value: float | None = None,
    horizontal_reference_label: str | None = None,
    colormap: str = "turbo",
    dense_log_ticks: bool = False,
    line_alpha: float = 0.9,
) -> bool:
    """Save trial histories with a marked and annotated per-step median."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted = _draw_history_axis(
        ax,
        histories,
        ylabel=ylabel,
        title=title,
        semilogy=semilogy,
        x_start=x_start,
        xlabel=xlabel,
        horizontal_reference_value=horizontal_reference_value,
        horizontal_reference_label=horizontal_reference_label,
        colormap=colormap,
        dense_log_ticks=dense_log_ticks,
        line_alpha=line_alpha,
    )
    if not plotted:
        plt.close(fig)
        return False

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return True


def save_transform_component_history_plot(
    results: Sequence[Any],
    out_path: Path,
    *,
    xlabel: str = "Solver step (0 = initial)",
) -> bool:
    """Save physical GT transform errors when component histories are present."""
    specifications = (
        (
            "iter_translation_error_norm_mm",
            r"$\|\Delta t\|$ [mm]",
            "Translation error norm",
            "#1f77b4",
        ),
        (
            "iter_rotation_geodesic_error_deg",
            "Geodesic rotation error [deg]",
            "Rotation geodesic error",
            "#9467bd",
        ),
        (
            "iter_gauge_parallel_abs_error_mm",
            r"$|\Delta t^T R_{true}e_z|$ [mm]",
            "Error parallel to true sensor-Z gauge",
            "#d95f02",
        ),
        (
            "iter_gauge_perpendicular_error_mm",
            r"$\|\Delta t_{\perp Z}\|$ [mm]",
            "Error perpendicular to true sensor-Z gauge",
            "#1b9e77",
        ),
    )
    histories_by_attribute = {
        attribute: [getattr(result, attribute, []) for result in results]
        for attribute, _ylabel, _title, _color in specifications
    }
    available_specifications = [
        specification
        for specification in specifications
        if any(
            len(history) > 0
            for history in histories_by_attribute[specification[0]]
        )
    ]
    if not available_specifications:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(available_specifications) <= 2:
        fig, axes = plt.subplots(
            1,
            len(available_specifications),
            figsize=(7.0 * len(available_specifications), 5.0),
            constrained_layout=True,
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            2, 2, figsize=(13.5, 9.0), constrained_layout=True, squeeze=False
        )
    plotted_count = 0
    for panel_index, (ax, specification) in enumerate(
        zip(axes.flat, available_specifications)
    ):
        attribute, ylabel, title, color = specification
        plotted = _draw_history_axis(
            ax,
            histories_by_attribute[attribute],
            ylabel=ylabel,
            title=title,
            semilogy=True,
            x_start=0,
            xlabel=xlabel,
            median_color=color,
            show_legend=panel_index == 0,
        )
        if plotted:
            plotted_count += 1
        else:
            ax.set_visible(False)

    for ax in list(axes.flat)[len(available_specifications) :]:
        ax.set_visible(False)

    if plotted_count == 0:
        plt.close(fig)
        return False

    fig.suptitle(
        "Physical ground-truth transform errors by solver step\n"
        "thin lines = trials; marked line = median",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return True


def save_paper_style_gt_error_distribution(
    results: Sequence[Any],
    out_path: Path,
) -> bool:
    """Save paper-style before/after GT error boxplots in physical units."""

    translation_before = finite_array(
        [getattr(result, "init_trans_err_norm_mm", np.nan) for result in results]
    )
    translation_after = finite_array(
        [getattr(result, "trans_err_norm_mm", np.nan) for result in results]
    )
    rotation_before = finite_array(
        [getattr(result, "init_rot_err_angle_deg", np.nan) for result in results]
    )
    rotation_after = finite_array(
        [getattr(result, "rot_err_angle_deg", np.nan) for result in results]
    )
    if translation_after.size == 0 or rotation_after.size == 0:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), constrained_layout=True)
    panels = (
        (
            translation_before,
            translation_after,
            "Translation error norm [mm]",
        ),
        (
            rotation_before,
            rotation_after,
            "Geodesic rotation error [deg]",
        ),
    )
    for ax, (before, after, ylabel) in zip(axes, panels):
        data = [before, after] if before.size else [after]
        labels = ["Before", "After"] if before.size else ["After"]
        box = ax.boxplot(
            data,
            widths=0.42,
            patch_artist=True,
            showmeans=False,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            flierprops={
                "marker": "+",
                "markeredgecolor": "black",
                "markersize": 4.0,
                "markeredgewidth": 0.65,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_linewidth(0.9)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return True


def save_paper_style_plane_rms_distribution(
    histories: Sequence[Sequence[float]],
    out_path: Path,
    *,
    sensor_noise_std_mm: float | None = None,
) -> bool:
    """Save initial/final self-fitted residuals using the paper's Fig. 4 grammar."""

    initial_values: list[float] = []
    final_values: list[float] = []
    for history in histories:
        values = finite_array(history)
        if values.size:
            initial_values.append(float(values[0]))
            final_values.append(float(values[-1]))

    initial = finite_array(initial_values)
    final = finite_array(final_values)
    if initial.size == 0 or final.size == 0:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), constrained_layout=True)
    panels = (
        (initial, "(a) Initial residual", "#9aa0a6"),
        (final, "(b) Final residual", "#4c78a8"),
    )
    rng = np.random.default_rng(20240612)
    for ax, (values, title, color) in zip(axes, panels):
        box = ax.boxplot(
            [values],
            positions=[1.0],
            widths=0.34,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
            medianprops={"color": "#8b0000", "linewidth": 2.0},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.62)
        jitter = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(
            np.full(values.size, 1.0) + jitter,
            values,
            s=15,
            color=color,
            alpha=min(0.58, max(0.12, 18.0 / max(values.size, 1))),
            linewidths=0,
            zorder=2,
        )
        if sensor_noise_std_mm is not None and sensor_noise_std_mm > 0.0:
            ax.axhline(
                sensor_noise_std_mm,
                color="#d62728",
                linestyle="--",
                linewidth=1.7,
                label=fr"commanded sensor $\sigma$ = {sensor_noise_std_mm:g} mm",
            )
            ax.legend(loc="best", fontsize=8)
        median = float(np.median(values))
        mean = float(np.mean(values))
        p25, p75 = np.percentile(values, [25.0, 75.0])
        ax.text(
            0.03,
            0.97,
            f"N = {values.size}\nmedian = {median:.4g} mm\n"
            f"mean = {mean:.4g} mm\nIQR = [{p25:.4g}, {p75:.4g}] mm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.88,
            },
        )
        ax.set_xlim(0.55, 1.45)
        ax.set_xticks([1.0], ["Self-fitted plane RMS"])
        ax.set_ylabel("Mean per-plane point-to-plane RMS [mm]")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.28)

    fig.suptitle(
        "Self-fitted plane residual distribution (single_plane.pdf Fig. 4 style)\n"
        "not a GT calibration error; all trials and outliers are retained",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return True


def save_gt_component_error_distribution(
    results: Sequence[Any],
    out_path: Path,
) -> bool:
    """Save final signed translation/rotation errors in true sensor axes."""

    attributes = (
        ("err_t_sensor_x_mm", "err_t_sensor_z_mm", "err_t_sensor_y_mm"),
        ("err_r_sensor_x_deg", "err_r_sensor_z_deg", "err_r_sensor_y_deg"),
    )
    data = [
        [
            finite_array([getattr(result, attribute, np.nan) for result in results])
            for attribute in group
        ]
        for group in attributes
    ]
    if any(any(values.size == 0 for values in group) for group in data):
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), constrained_layout=True)
    panels = (
        (
            data[0],
            r"sensor $X$",
            r"sensor $Z$",
            r"sensor $Y$",
            "Translation component error [mm]",
            "",
        ),
        (
            data[1],
            r"sensor $X$",
            r"sensor $Z$",
            r"sensor $Y$",
            "Rotation-vector error [deg]",
            "",
        ),
    )
    colors = ("#d9d9d9", "#bdbdbd", "#969696")
    for ax, (values, *labels_and_titles) in zip(axes, panels):
        labels = labels_and_titles[:3]
        ylabel, title = labels_and_titles[3:]
        box = ax.boxplot(
            values,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            flierprops={"markersize": 2.5, "markeredgewidth": 0.5},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xticks([1, 2, 3], labels)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
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
