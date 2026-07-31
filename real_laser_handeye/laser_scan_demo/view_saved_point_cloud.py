from __future__ import annotations

"""
Visualize a point cloud saved by scan_for_validation.py or
two_point_stop_and_scan.py.

Example:
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/view_saved_point_cloud.py \
  --input runs/real/two_point_stop_and_scan.npz \
  --z-min -10

The default color is a height heatmap based only on point Z coordinates.
No sphere fitting or sphere parameters are used.
"""

import argparse
import colorsys
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


def _capture_keys(keys: set[str], frame: str) -> list[str]:
    suffix = f"_points_{frame}"
    return sorted(
        key
        for key in keys
        if key.startswith("capture_") and key.endswith(suffix)
    )


def _clean_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    array = array[np.all(np.isfinite(array), axis=1)]
    if len(array) == 0:
        raise ValueError(f"{name} has no finite points")
    return np.ascontiguousarray(array, dtype=np.float32)


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
                    f"{path} has neither capture_*_points_{frame} nor "
                    f"{merged_key}"
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
        cropped = np.ascontiguousarray(points[mask])
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


def distinct_colors(count: int) -> list[tuple[float, float, float, float]]:
    colors = []
    for index in range(count):
        hue = index / max(1, count)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        colors.append((red, green, blue, 0.9))
    return colors


def height_heatmap_colors(
    points: np.ndarray,
    *,
    color_min_z: float,
    color_max_z: float,
) -> np.ndarray:
    """Map Z through a saturated, percentile-clipped height heatmap."""
    z = np.asarray(points[:, 2], dtype=float)
    span = max(float(color_max_z - color_min_z), 1e-9)
    normalized = np.clip((z - color_min_z) / span, 0.0, 1.0)

    positions = np.array(
        [0.0, 0.12, 0.28, 0.44, 0.60, 0.76, 0.90, 1.0],
        dtype=float,
    )
    anchors = np.array(
        [
            [0.15, 0.00, 0.35],  # dark purple
            [0.00, 0.05, 1.00],  # blue
            [0.00, 0.95, 1.00],  # cyan
            [0.00, 1.00, 0.15],  # green
            [0.95, 1.00, 0.00],  # yellow
            [1.00, 0.35, 0.00],  # orange
            [1.00, 0.00, 0.00],  # red
            [1.00, 1.00, 1.00],  # saturated high
        ],
        dtype=float,
    )
    colors = np.empty((len(points), 4), dtype=np.float32)
    for channel in range(3):
        colors[:, channel] = np.interp(
            normalized,
            positions,
            anchors[:, channel],
        )
    colors[:, 3] = 0.92
    return colors


def add_axes(view: gl.GLViewWidget, length: float) -> None:
    origin = np.zeros(3, dtype=np.float32)
    for endpoint, color in (
        ([length, 0.0, 0.0], (1.0, 0.1, 0.1, 1.0)),
        ([0.0, length, 0.0], (0.1, 1.0, 0.1, 1.0)),
        ([0.0, 0.0, length], (0.1, 0.5, 1.0, 1.0)),
    ):
        view.addItem(
            gl.GLLinePlotItem(
                pos=np.vstack([origin, np.asarray(endpoint, dtype=np.float32)]),
                color=color,
                width=3.0,
                antialias=False,
                mode="lines",
            )
        )


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
    extent = np.maximum(maximum - minimum, 1.0)
    color_min_z, color_max_z = np.percentile(
        merged[:, 2],
        [
            args.heatmap_clip_percentile,
            100.0 - args.heatmap_clip_percentile,
        ],
    )
    if color_max_z - color_min_z < 1e-9:
        color_min_z = float(minimum[2])
        color_max_z = float(maximum[2])

    app = pg.mkQApp("Saved point-cloud viewer")
    window = QtWidgets.QWidget()
    window.setWindowTitle(f"Saved point cloud: {path.name}")
    window.resize(1280, 850)
    layout = QtWidgets.QVBoxLayout(window)

    info = QtWidgets.QLabel(
        f"{path} | frame={args.frame} | groups={len(groups)} | "
        f"points={full_count:,} | displayed={shown_count:,}\n"
        f"min=[{minimum[0]:.3f}, {minimum[1]:.3f}, {minimum[2]:.3f}] mm | "
        f"max=[{maximum[0]:.3f}, {maximum[1]:.3f}, {maximum[2]:.3f}] mm"
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    view = gl.GLViewWidget()
    view.setBackgroundColor((12, 14, 18))
    layout.addWidget(view, stretch=1)

    grid = gl.GLGridItem()
    grid_size = max(10.0, math.ceil(float(max(extent[:2])) / 10.0) * 10.0)
    grid.setSize(x=grid_size, y=grid_size)
    grid.setSpacing(
        x=max(1.0, grid_size / 10.0),
        y=max(1.0, grid_size / 10.0),
    )
    grid.translate(float(center[0]), float(center[1]), float(minimum[2]))
    view.addItem(grid)

    if args.color_by == "capture":
        for points, color in zip(shown_groups, distinct_colors(len(shown_groups))):
            view.addItem(
                gl.GLScatterPlotItem(
                    pos=points,
                    color=color,
                    size=args.point_size,
                    pxMode=True,
                )
            )
    else:
        points = np.concatenate(shown_groups, axis=0)
        color = (
            height_heatmap_colors(
                points,
                color_min_z=float(color_min_z),
                color_max_z=float(color_max_z),
            )
            if args.color_by == "z"
            else (0.1, 0.85, 1.0, 0.9)
        )
        view.addItem(
            gl.GLScatterPlotItem(
                pos=points,
                color=color,
                size=args.point_size,
                pxMode=True,
            )
        )

    add_axes(view, args.axis_length_mm)
    view.opts["center"] = pg.Vector(
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )
    view.setCameraPosition(
        distance=max(30.0, 2.2 * float(np.linalg.norm(extent))),
        elevation=25,
        azimuth=45,
    )

    if args.color_by == "z":
        color_bar_row = QtWidgets.QHBoxLayout()
        color_bar_row.addWidget(
            QtWidgets.QLabel(f"Low Z: ≤{color_min_z:.3f} mm")
        )
        color_bar = QtWidgets.QFrame()
        color_bar.setMinimumHeight(22)
        color_bar.setStyleSheet(
            "background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #260059, stop:0.12 #000dff, "
            "stop:0.28 #00f2ff, stop:0.44 #00ff26, "
            "stop:0.60 #f2ff00, stop:0.76 #ff5900, "
            "stop:0.90 #ff0000, stop:1 #ffffff"
            "); border: 1px solid #888;"
        )
        color_bar_row.addWidget(color_bar, stretch=1)
        color_bar_row.addWidget(
            QtWidgets.QLabel(f"High Z: ≥{color_max_z:.3f} mm")
        )
        layout.addLayout(color_bar_row)
    else:
        legend = QtWidgets.QLabel(
            (
                "Capture colors: "
                + ", ".join(
                    f"{index}: {name}" for index, name in enumerate(names)
                )
            )
            if args.color_by == "capture"
            else "Color: uniform"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

    window.show()
    app.exec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open and visualize a saved laser point-cloud NPZ file"
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
    parser.add_argument(
        "--color-by",
        choices=("capture", "z", "uniform"),
        default="z",
        help="Point coloring mode. Default: z height heatmap",
    )
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--max-display-points", type=int, default=500_000)
    parser.add_argument("--axis-length-mm", type=float, default=50.0)
    parser.add_argument(
        "--heatmap-clip-percentile",
        type=float,
        default=2.0,
        help=(
            "Saturate this percentage at each end of the Z color range to "
            "increase contrast. Default: 2"
        ),
    )
    args = parser.parse_args()

    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    if args.max_display_points <= 0:
        parser.error("--max-display-points must be positive")
    if args.axis_length_mm <= 0:
        parser.error("--axis-length-mm must be positive")
    if not 0.0 <= args.heatmap_clip_percentile < 50.0:
        parser.error("--heatmap-clip-percentile must be in [0, 50)")
    return args


def main() -> None:
    args = parse_args()
    groups, names = load_groups(
        args.input,
        frame=args.frame,
        source=args.source,
    )
    groups, names = crop_groups(groups, names, args)

    print(f"input: {args.input}")
    print(f"frame: {args.frame}")
    print(f"source: {args.source}")
    print(f"groups: {len(groups)}")
    print(f"points after crop: {sum(len(points) for points in groups):,}")
    visualize(args.input, groups, names, args)


if __name__ == "__main__":
    main()
