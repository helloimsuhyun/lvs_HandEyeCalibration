from __future__ import annotations

"""
Open3D-based viewer for point clouds saved by scan_for_validation.py or
Two-point stop-and-scan scripts.

Default rendering uses a muted uniform gray color rather than a height heatmap.

Examples
--------
# Default muted single-color point cloud
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/view_saved_point_cloud.py \
  --input /home/choisuhyun/lvs_HandEyeCalibration/runs/real/block.npz

# Z-height coloring
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/view_saved_point_cloud.py \
  --input /home/choisuhyun/lvs_HandEyeCalibration/runs/real/block.npz \
  --color-by z

# Capture-by-capture coloring
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/view_saved_point_cloud.py \
  --input   /home/choisuhyun/lvs_HandEyeCalibration/runs/real/block.npz\
  --color-by capture

Controls
--------
- Mouse drag: rotate
- Shift + mouse drag: pan
- Mouse wheel: zoom
- F: fit the current point cloud to the window
- Q or Esc: close
"""

import argparse
import colorsys
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d


def _capture_keys(keys: set[str], frame: str) -> list[str]:
    suffix = f"_points_{frame}"
    return sorted(
        key
        for key in keys
        if key.startswith("capture_") and key.endswith(suffix)
    )


def _clean_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")

    array = array[np.all(np.isfinite(array), axis=1)]
    if len(array) == 0:
        raise ValueError(f"{name} has no finite points")

    return np.ascontiguousarray(array, dtype=np.float64)


def load_groups(
    path: Path,
    *,
    frame: str,
    source: str,
) -> tuple[list[np.ndarray], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"point-cloud file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        capture_keys = _capture_keys(keys, frame)
        merged_key = f"points_{frame}_merged"

        if source == "merged":
            if capture_keys:
                selected_keys = capture_keys
            elif merged_key in keys:
                selected_keys = [merged_key]
            else:
                raise KeyError(
                    f"{path} has neither capture_*_points_{frame} nor {merged_key}"
                )
        elif source == "latest":
            if not capture_keys:
                raise KeyError(f"{path} has no capture_*_points_{frame}")
            selected_keys = [capture_keys[-1]]
        else:
            try:
                capture_index = int(source)
            except ValueError as exc:
                raise ValueError(
                    "--source must be merged, latest, or a capture index"
                ) from exc

            key = f"capture_{capture_index:04d}_points_{frame}"
            if key not in keys:
                raise KeyError(f"{path} has no key named {key}")
            selected_keys = [key]

        groups = [_clean_points(data[key], key) for key in selected_keys]

    names = [
        key.replace(f"_points_{frame}", "")
        if key.startswith("capture_")
        else key
        for key in selected_keys
    ]
    return groups, names


def crop_groups(
    groups: list[np.ndarray],
    names: list[str],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[str]]:
    bounds = (
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        (args.z_min, args.z_max),
    )

    cropped_groups: list[np.ndarray] = []
    cropped_names: list[str] = []

    for points, name in zip(groups, names):
        mask = np.ones(len(points), dtype=bool)
        for axis, (lower, upper) in enumerate(bounds):
            if lower is not None:
                mask &= points[:, axis] >= lower
            if upper is not None:
                mask &= points[:, axis] <= upper

        cropped = np.ascontiguousarray(points[mask], dtype=np.float64)
        if len(cropped):
            cropped_groups.append(cropped)
            cropped_names.append(name)

    if not cropped_groups:
        raise ValueError("crop removed all points")

    return cropped_groups, cropped_names


def downsample_groups(
    groups: list[np.ndarray],
    max_points: int,
) -> list[np.ndarray]:
    total = sum(len(points) for points in groups)
    if total <= max_points:
        return groups

    stride = max(1, math.ceil(total / max_points))
    return [np.ascontiguousarray(points[::stride]) for points in groups]


def _validate_rgb(values: Sequence[float], name: str) -> np.ndarray:
    color = np.asarray(values, dtype=np.float64)
    if color.shape != (3,) or not np.all(np.isfinite(color)):
        raise ValueError(f"{name} must contain three finite values")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return color


def muted_capture_colors(count: int) -> list[np.ndarray]:
    """Generate distinguishable but deliberately low-saturation capture colors."""
    colors: list[np.ndarray] = []
    for index in range(count):
        hue = index / max(1, count)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.42, 0.80)
        colors.append(np.array([red, green, blue], dtype=np.float64))
    return colors


def muted_height_colors(
    points: np.ndarray,
    *,
    color_min_z: float,
    color_max_z: float,
) -> np.ndarray:
    """Map Z to a subdued blue-to-gold palette."""
    z = np.asarray(points[:, 2], dtype=np.float64)
    span = max(float(color_max_z - color_min_z), 1e-12)
    normalized = np.clip((z - color_min_z) / span, 0.0, 1.0)

    positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
    anchors = np.array(
        [
            [0.18, 0.24, 0.38],
            [0.22, 0.42, 0.58],
            [0.34, 0.58, 0.56],
            [0.68, 0.64, 0.40],
            [0.80, 0.48, 0.30],
        ],
        dtype=np.float64,
    )

    colors = np.empty((len(points), 3), dtype=np.float64)
    for channel in range(3):
        colors[:, channel] = np.interp(
            normalized,
            positions,
            anchors[:, channel],
        )
    return colors


def build_open3d_cloud(
    shown_groups: list[np.ndarray],
    *,
    color_by: str,
    uniform_color: np.ndarray,
    heatmap_clip_percentile: float,
) -> tuple[o3d.geometry.PointCloud, tuple[float, float] | None]:
    points = np.concatenate(shown_groups, axis=0)

    if color_by == "uniform":
        colors = np.repeat(uniform_color.reshape(1, 3), len(points), axis=0)
        z_range = None

    elif color_by == "capture":
        palette = muted_capture_colors(len(shown_groups))
        colors = np.concatenate(
            [
                np.repeat(color.reshape(1, 3), len(group), axis=0)
                for group, color in zip(shown_groups, palette)
            ],
            axis=0,
        )
        z_range = None

    elif color_by == "z":
        lower, upper = np.percentile(
            points[:, 2],
            [heatmap_clip_percentile, 100.0 - heatmap_clip_percentile],
        )
        if upper - lower < 1e-12:
            lower = float(np.min(points[:, 2]))
            upper = float(np.max(points[:, 2]))

        colors = muted_height_colors(
            points,
            color_min_z=float(lower),
            color_max_z=float(upper),
        )
        z_range = (float(lower), float(upper))

    else:
        raise ValueError(f"unsupported color mode: {color_by}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud, z_range


def visualize(
    path: Path,
    groups: list[np.ndarray],
    names: list[str],
    args: argparse.Namespace,
) -> None:
    full_count = sum(len(points) for points in groups)
    shown_groups = downsample_groups(groups, args.max_display_points)
    shown_count = sum(len(points) for points in shown_groups)

    merged = np.concatenate(groups, axis=0)
    minimum = np.min(merged, axis=0)
    maximum = np.max(merged, axis=0)
    center = 0.5 * (minimum + maximum)

    cloud, z_range = build_open3d_cloud(
        shown_groups,
        color_by=args.color_by,
        uniform_color=np.asarray(args.uniform_color, dtype=np.float64),
        heatmap_clip_percentile=args.heatmap_clip_percentile,
    )

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    created = visualizer.create_window(
        window_name=f"Open3D point cloud: {path.name}",
        width=args.window_width,
        height=args.window_height,
        visible=True,
    )
    if not created:
        raise RuntimeError("Open3D failed to create a visualization window")

    visualizer.add_geometry(cloud, reset_bounding_box=True)

    if args.show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=args.axis_length_mm,
            origin=[0.0, 0.0, 0.0],
        )
        visualizer.add_geometry(axis, reset_bounding_box=False)

    render_option = visualizer.get_render_option()
    if render_option is None:
        visualizer.destroy_window()
        raise RuntimeError("Open3D did not provide render options")

    render_option.background_color = np.asarray(
        args.background_color,
        dtype=np.float64,
    )
    render_option.point_size = float(args.point_size)
    render_option.light_on = False

    def fit_view(vis: o3d.visualization.Visualizer) -> bool:
        vis.reset_view_point(True)
        return False

    visualizer.register_key_callback(ord("F"), fit_view)

    print(f"input: {path}")
    print(f"frame: {args.frame}")
    print(f"source: {args.source}")
    print(f"color mode: {args.color_by}")
    print(f"groups: {len(groups)}")
    print(f"points after crop: {full_count:,}")
    print(f"displayed points: {shown_count:,}")
    print(
        "bounds [mm]: "
        f"min=[{minimum[0]:.3f}, {minimum[1]:.3f}, {minimum[2]:.3f}], "
        f"max=[{maximum[0]:.3f}, {maximum[1]:.3f}, {maximum[2]:.3f}]"
    )
    print(
        f"center [mm]: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
    )
    if z_range is not None:
        print(f"Z color range [mm]: {z_range[0]:.3f} to {z_range[1]:.3f}")
    if args.color_by == "capture":
        print("capture groups: " + ", ".join(names))
    print("Open3D controls: F=fit view, mouse=rotate/pan/zoom, Q or Esc=close")

    visualizer.reset_view_point(True)
    visualizer.run()
    visualizer.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open and visualize a saved laser point-cloud NPZ with Open3D"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/real/two_point_stop_and_scan.npz"),
    )
    parser.add_argument(
        "--frame",
        choices=("base", "sensor"),
        default="base",
        help="Coordinate frame loaded from the NPZ. Default: base",
    )
    parser.add_argument(
        "--source",
        default="merged",
        help="merged, latest, or a capture index such as 0",
    )

    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color-by",
        choices=("uniform", "z", "capture"),
        default="uniform",
        help="Point coloring mode. Default: uniform",
    )
    color_group.add_argument(
        "--plain",
        action="store_true",
        help="Alias for --color-by uniform",
    )

    parser.add_argument("--uniform-color", nargs=3, type=float, default=(0.64, 0.67, 0.70))
    parser.add_argument("--background-color", nargs=3, type=float, default=(0.035, 0.035, 0.040))

    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)

    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-display-points", type=int, default=500_000)
    parser.add_argument("--show-axis", action="store_true")
    parser.add_argument("--axis-length-mm", type=float, default=50.0)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=850)
    parser.add_argument(
        "--heatmap-clip-percentile",
        type=float,
        default=2.0,
        help="Percentage clipped at each end of the Z color range. Default: 2",
    )

    args = parser.parse_args()

    if args.plain:
        args.color_by = "uniform"

    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    if args.max_display_points <= 0:
        parser.error("--max-display-points must be positive")
    if args.axis_length_mm <= 0:
        parser.error("--axis-length-mm must be positive")
    if args.window_width <= 0 or args.window_height <= 0:
        parser.error("window dimensions must be positive")
    if not 0.0 <= args.heatmap_clip_percentile < 50.0:
        parser.error("--heatmap-clip-percentile must be in [0, 50)")

    try:
        args.uniform_color = _validate_rgb(args.uniform_color, "--uniform-color")
        args.background_color = _validate_rgb(
            args.background_color,
            "--background-color",
        )
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main() -> None:
    args = parse_args()
    groups, names = load_groups(
        args.input,
        frame=args.frame,
        source=args.source,
    )
    groups, names = crop_groups(groups, names, args)
    visualize(args.input, groups, names, args)


if __name__ == "__main__":
    main()