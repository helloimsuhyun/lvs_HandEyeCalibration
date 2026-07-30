#!/usr/bin/env python3
"""Visualize plane-relative pose parameters and generated sensor poses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np

from laser_handeye.calibration_dataset import load_calibration_dataset


METHODS = (
    ("single_plane_uniform", "Single plane\nrelative LHS-uniform"),
    ("three_plane_random", "Three plane\nshared-global random"),
    (
        "three_plane_uniform",
        "Three plane relative LHS-uniform\nfull pose set on every plane",
    ),
)
THREE_LEVEL_METHODS = (
    ("single_uniform", "Single plane\nrelative LHS-uniform"),
    ("three_hard", "Three plane\nrestricted diversity"),
    ("three_moderate", "Three plane\nmoderate diversity"),
    ("three_easy", "Three plane\nwide diversity"),
)
SINGLE_UNIFORM_FISHER_METHODS = (
    ("single_plane_uniform", "Single plane\nUniform maximin"),
    ("single_plane_fisher", "Single plane\nactive Fisher"),
)
PLANE_COLORS = ("#4C78A8", "#F58518", "#54A24B")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--design",
        choices=(
            "uniform_fullset",
            "three_pose_levels",
            "single_uniform_vs_fisher",
        ),
        default="uniform_fullset",
    )
    parser.add_argument(
        "--tilt-range-deg",
        type=float,
        nargs=2,
        default=(10.0, 60.0),
    )
    parser.add_argument(
        "--azimuth-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument(
        "--roll-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument(
        "--max-orientation-axes",
        type=int,
        default=18,
        help="Maximum sensor frames drawn per method; all origins remain visible.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    delta: tuple[float, float],
    *,
    color: str,
    label: str | None = None,
    width: float = 0.012,
) -> None:
    axis.arrow(
        start[0],
        start[1],
        delta[0],
        delta[1],
        width=width,
        head_width=0.09,
        head_length=0.12,
        length_includes_head=True,
        color=color,
        label=label,
        zorder=4,
    )


def _range_label(values: Sequence[float]) -> str:
    return f"{values[0]:g}°–{values[1]:g}°"


def _save_parameter_definition(
    path: Path,
    dpi: int,
    *,
    tilt_range_deg: Sequence[float],
    azimuth_range_deg: Sequence[float],
    roll_range_deg: Sequence[float],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))

    # Tilt: side view in the plane spanned by n and the in-plane radial axis.
    axis = axes[0]
    axis.axhspan(-0.06, 0.0, color="#DCEAF7")
    axis.axhline(0.0, color="#4C78A8", linewidth=2.0)
    _arrow(axis, (0.0, 0.0), (0.0, 1.25), color="#54A24B", label=r"$n$")
    angle = np.deg2rad(45.0)
    direction = np.array([np.sin(angle), np.cos(angle)])
    _arrow(
        axis,
        (0.0, 0.0),
        tuple(1.25 * direction),
        color="#E45756",
        label=r"$d$: target $\rightarrow$ sensor",
    )
    axis.add_patch(
        Arc(
            (0.0, 0.0),
            0.9,
            0.9,
            theta1=45.0,
            theta2=90.0,
            color="#222222",
            linewidth=1.5,
        )
    )
    axis.text(0.16, 0.48, "tilt", fontsize=11)
    axis.scatter(*direction, color="#E45756", s=45, zorder=5)
    axis.text(direction[0] + 0.04, direction[1], "sensor")
    axis.text(-0.03, -0.13, "target", ha="center")
    axis.set_title(
        "Tilt: angle from plane normal\n"
        f"sampled {_range_label(tilt_range_deg)}"
    )
    axis.set_xlabel("in-plane radial direction")
    axis.set_ylabel("plane normal direction")
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    # Azimuth: top view of the target plane.
    axis = axes[1]
    axis.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="#A0A0A0"))
    _arrow(axis, (0.0, 0.0), (1.2, 0.0), color="#4C78A8", label=r"$u$")
    _arrow(axis, (0.0, 0.0), (0.0, 1.2), color="#54A24B", label=r"$v$")
    angle = np.deg2rad(55.0)
    radial = np.array([np.cos(angle), np.sin(angle)])
    _arrow(
        axis,
        (0.0, 0.0),
        tuple(radial),
        color="#E45756",
        label=r"projection of $d$",
    )
    axis.add_patch(
        Arc(
            (0.0, 0.0),
            0.85,
            0.85,
            theta1=0.0,
            theta2=55.0,
            color="#222222",
            linewidth=1.5,
        )
    )
    axis.text(0.43, 0.16, "azimuth", fontsize=11, rotation=22)
    axis.scatter(*radial, color="#E45756", s=45, zorder=5)
    axis.text(radial[0] + 0.04, radial[1], "sensor direction")
    axis.set_title(
        "Azimuth: direction within plane\n"
        f"sampled {_range_label(azimuth_range_deg)}"
    )
    axis.set_xlabel("plane $u$")
    axis.set_ylabel("plane $v$")
    axis.legend(frameon=False, fontsize=8, loc="lower left")

    # Roll: view along sensor +z; show zero-roll and rolled profile directions.
    axis = axes[2]
    axis.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="#A0A0A0"))
    _arrow(axis, (0.0, 0.0), (1.15, 0.0), color="#A0A0A0")
    _arrow(axis, (0.0, 0.0), (0.0, 1.15), color="#A0A0A0")
    axis.text(1.0, -0.13, r"$x_0$")
    axis.text(-0.14, 1.0, r"$y_0$")
    angle = np.deg2rad(45.0)
    x_axis = np.array([np.cos(angle), np.sin(angle)])
    y_axis = np.array([-np.sin(angle), np.cos(angle)])
    _arrow(
        axis,
        (0.0, 0.0),
        tuple(1.15 * x_axis),
        color="#E45756",
        label="sensor x / profile direction",
    )
    _arrow(
        axis,
        (0.0, 0.0),
        tuple(1.15 * y_axis),
        color="#54A24B",
        label="sensor y",
    )
    axis.add_patch(
        Arc(
            (0.0, 0.0),
            0.85,
            0.85,
            theta1=0.0,
            theta2=45.0,
            color="#222222",
            linewidth=1.5,
        )
    )
    axis.text(0.44, 0.14, "roll", fontsize=11, rotation=20)
    axis.text(
        0.0,
        0.0,
        r"$+z$",
        ha="center",
        va="center",
        bbox={"boxstyle": "circle", "facecolor": "white", "edgecolor": "#222222"},
    )
    axis.set_title(
        "Roll: rotation about sensor view axis\n"
        f"sampled {_range_label(roll_range_deg)}"
    )
    axis.set_xlabel("sensor image/profile plane")
    axis.set_ylabel("viewed along sensor $+z$")
    axis.legend(frameon=False, fontsize=8, loc="lower left")

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.35, 1.45)
        axis.set_ylim(-0.35 if axis is axes[0] else -1.35, 1.45)
        axis.grid(linestyle="--", alpha=0.2)
    figure.suptitle(
        "Plane-relative pose parameters (angles are not Euler angles)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _sensor_transforms(dataset) -> np.ndarray:
    if dataset.truth is None or dataset.truth.T_ef_s_true is None:
        raise ValueError("dataset has no ground-truth hand-eye transform")
    handeye = np.asarray(dataset.truth.T_ef_s_true, dtype=float)
    return np.stack(
        [np.asarray(scan.T_base_ef, dtype=float) @ handeye for scan in dataset.scans]
    )


def _plane_patch(transform: np.ndarray, half_size: float = 58.0) -> np.ndarray:
    grid = np.asarray(
        [
            [-half_size, -half_size, 0.0, 1.0],
            [half_size, -half_size, 0.0, 1.0],
            [half_size, half_size, 0.0, 1.0],
            [-half_size, half_size, 0.0, 1.0],
        ]
    )
    return (np.asarray(transform, dtype=float) @ grid.T).T[:, :3]


def _equal_3d_limits(axes: Sequence[plt.Axes], points: np.ndarray) -> None:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 1.0) * 1.08
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1.0, 1.0, 1.0))


def _save_pose_distribution(
    datasets: list,
    path: Path,
    *,
    trial_index: int,
    max_orientation_axes: int,
    dpi: int,
) -> None:
    transforms = [_sensor_transforms(dataset) for dataset in datasets]
    all_points = []
    figure = plt.figure(figsize=(5.4 * len(METHODS), 6.4))
    axes = [
        figure.add_subplot(1, len(METHODS), index + 1, projection="3d")
        for index in range(len(METHODS))
    ]

    for axis, dataset, poses, (_directory, title) in zip(
        axes, datasets, transforms, METHODS, strict=True
    ):
        origins = poses[:, :3, 3]
        all_points.append(origins)
        common_center = np.asarray(
            dataset.metadata["shared_target_center_base_mm"],
            dtype=float,
        )
        all_points.append(common_center[None, :])

        planes = {
            int(plane.plane_id): plane
            for plane in dataset.truth.planes
            if plane.T_base_plane is not None
        }
        for plane_id, plane in planes.items():
            patch = _plane_patch(plane.T_base_plane)
            all_points.append(patch)
            patch_closed = np.vstack([patch, patch[0]])
            axis.plot_trisurf(
                patch[:, 0],
                patch[:, 1],
                patch[:, 2],
                triangles=[[0, 1, 2], [0, 2, 3]],
                color=PLANE_COLORS[plane_id],
                alpha=0.16,
                shade=False,
            )
            axis.plot(
                patch_closed[:, 0],
                patch_closed[:, 1],
                patch_closed[:, 2],
                color=PLANE_COLORS[plane_id],
                linewidth=1.1,
                alpha=0.7,
            )
            frame = np.asarray(plane.T_base_plane, dtype=float)
            origin = frame[:3, 3]
            normal = frame[:3, 2]
            axis.quiver(
                *origin,
                *(32.0 * normal),
                color=PLANE_COLORS[plane_id],
                linewidth=1.8,
                arrow_length_ratio=0.18,
            )
            axis.text(
                *(origin + 37.0 * normal),
                f"$n_{plane_id}$",
                color=PLANE_COLORS[plane_id],
            )

        plane_ids = np.asarray(
            [int(scan.plane_id) for scan in dataset.scans],
            dtype=int,
        )
        for plane_id in np.unique(plane_ids):
            selected = plane_ids == plane_id
            axis.scatter(
                origins[selected, 0],
                origins[selected, 1],
                origins[selected, 2],
                s=13,
                alpha=0.62,
                color=PLANE_COLORS[plane_id],
                label=f"sensor origins, plane {plane_id}",
            )

        count = min(max_orientation_axes, len(poses))
        selected_indices = np.linspace(
            0,
            len(poses) - 1,
            count,
            dtype=int,
        )
        for pose_index in selected_indices:
            pose = poses[pose_index]
            origin = pose[:3, 3]
            x_axis = pose[:3, 0]
            z_axis = pose[:3, 2]
            points_sensor = dataset.scans[pose_index].valid_points_s
            points_base = (
                pose[:3, :3] @ points_sensor.T + origin[:, None]
            ).T
            all_points.append(points_base)
            plane_id = int(dataset.scans[pose_index].plane_id)
            axis.plot(
                points_base[:, 0],
                points_base[:, 1],
                points_base[:, 2],
                color=PLANE_COLORS[plane_id],
                linewidth=1.15,
                alpha=0.9,
            )
            profile_center = np.mean(points_base, axis=0)
            axis.plot(
                [origin[0], profile_center[0]],
                [origin[1], profile_center[1]],
                [origin[2], profile_center[2]],
                color="#555555",
                linestyle=":",
                linewidth=0.55,
                alpha=0.5,
            )
            axis.plot(
                [origin[0], origin[0] + 16.0 * x_axis[0]],
                [origin[1], origin[1] + 16.0 * x_axis[1]],
                [origin[2], origin[2] + 16.0 * x_axis[2]],
                color="#D62728",
                linewidth=0.8,
                alpha=0.8,
            )
            axis.plot(
                [origin[0], origin[0] + 24.0 * z_axis[0]],
                [origin[1], origin[1] + 24.0 * z_axis[1]],
                [origin[2], origin[2] + 24.0 * z_axis[2]],
                color="#202020",
                linewidth=0.75,
                alpha=0.65,
            )

        axis.scatter(
            *common_center,
            marker="*",
            s=80,
            color="#111111",
            label="common target center",
        )
        axis.set_title(title)
        axis.set_xlabel("base x [mm]")
        axis.set_ylabel("base y [mm]")
        axis.set_zlabel("base z [mm]")
        axis.view_init(elev=23.0, azim=-55.0)
        axis.legend(fontsize=7, frameon=False, loc="upper left")

    _equal_3d_limits(axes, np.vstack(all_points))
    figure.suptitle(
        f"Generated sensor poses — trial {trial_index:06d}\n"
        "dots: all sensor origins; red ticks: sensor x/profile direction; "
        "black ticks: sensor +z/view direction; colored lines: measured profiles",
        fontsize=13,
        y=0.98,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    global METHODS
    args = parse_args(argv)
    if args.design == "three_pose_levels":
        METHODS = THREE_LEVEL_METHODS
    elif args.design == "single_uniform_vs_fisher":
        METHODS = SINGLE_UNIFORM_FISHER_METHODS
    if args.trial_index < 0:
        raise SystemExit("--trial-index must be non-negative")
    if args.max_orientation_axes <= 0:
        raise SystemExit("--max-orientation-axes must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_name = f"trial_{args.trial_index:06d}"
    datasets = [
        load_calibration_dataset(dataset_root / directory / "trials" / trial_name)
        for directory, _title in METHODS
    ]
    _save_parameter_definition(
        output_dir / "plane_relative_pose_parameter_definition.png",
        args.dpi,
        tilt_range_deg=args.tilt_range_deg,
        azimuth_range_deg=args.azimuth_range_deg,
        roll_range_deg=args.roll_range_deg,
    )
    _save_pose_distribution(
        datasets,
        output_dir / f"generated_sensor_poses_{trial_name}.png",
        trial_index=args.trial_index,
        max_orientation_axes=args.max_orientation_axes,
        dpi=args.dpi,
    )
    print(f"Saved pose-geometry visualizations: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
