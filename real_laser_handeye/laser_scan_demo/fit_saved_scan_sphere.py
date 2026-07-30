from __future__ import annotations

"""
PYTHONPATH=. python /home/choisuhyun/lvs_HandEyeCalibration/real_laser_handeye/laser_scan_demo/fit_saved_scan_sphere.py \
  --input runs/real/manual_world_scan.npz
"""

import argparse
from dataclasses import dataclass
import colorsys
import json
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


@dataclass
class SphereFitResult:
    center: np.ndarray
    radius_mm: float
    residuals_mm: np.ndarray
    inlier_mask: np.ndarray
    iterations: int
    converged: bool

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def rms_mm(self) -> float:
        r = self.residuals_mm[self.inlier_mask]
        return float(np.sqrt(np.mean(r * r)))

    @property
    def mae_mm(self) -> float:
        return float(np.mean(np.abs(self.residuals_mm[self.inlier_mask])))

    @property
    def median_abs_mm(self) -> float:
        return float(np.median(np.abs(self.residuals_mm[self.inlier_mask])))

    @property
    def p95_abs_mm(self) -> float:
        return float(np.percentile(np.abs(self.residuals_mm[self.inlier_mask]), 95))

    @property
    def max_abs_mm(self) -> float:
        return float(np.max(np.abs(self.residuals_mm[self.inlier_mask])))


@dataclass
class LoadedPoints:
    merged_points: np.ndarray
    groups: list[np.ndarray]
    group_names: list[str]


def _sorted_capture_keys(keys: set[str]) -> list[str]:
    return sorted(
        key for key in keys
        if key.startswith("capture_") and key.endswith("_points_base")
    )


def load_saved_points(path: Path, source: str) -> LoadedPoints:
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        capture_keys = _sorted_capture_keys(keys)

        if source == "merged":
            if capture_keys:
                groups = [np.asarray(data[key], dtype=float) for key in capture_keys]
                group_names = [key.replace("_points_base", "") for key in capture_keys]
                merged_points = np.concatenate(groups, axis=0)
            elif "points_base_merged" in keys:
                merged_points = np.asarray(data["points_base_merged"], dtype=float)
                groups = [merged_points]
                group_names = ["merged"]
            else:
                raise KeyError(
                    "NPZ has neither capture_*_points_base nor points_base_merged"
                )

        elif source == "latest":
            if not capture_keys:
                raise KeyError("NPZ has no capture_*_points_base arrays")
            groups = [np.asarray(data[capture_keys[-1]], dtype=float)]
            group_names = [capture_keys[-1].replace("_points_base", "")]
            merged_points = groups[0]

        else:
            key = f"capture_{int(source):04d}_points_base"
            if key not in keys:
                raise KeyError(f"NPZ has no key: {key}")
            groups = [np.asarray(data[key], dtype=float)]
            group_names = [key.replace("_points_base", "")]
            merged_points = groups[0]

    clean_groups: list[np.ndarray] = []
    clean_names: list[str] = []
    for name, group in zip(group_names, groups):
        if group.ndim != 2 or group.shape[1] != 3:
            raise ValueError(f"loaded points must have shape (N,3), got {group.shape}")
        group = group[np.all(np.isfinite(group), axis=1)]
        if len(group) == 0:
            continue
        clean_groups.append(np.ascontiguousarray(group))
        clean_names.append(name)

    if not clean_groups:
        raise ValueError("no finite points found")

    merged = np.concatenate(clean_groups, axis=0)
    return LoadedPoints(merged_points=np.ascontiguousarray(merged), groups=clean_groups, group_names=clean_names)


def apply_crop_to_groups(loaded: LoadedPoints, args: argparse.Namespace) -> LoadedPoints:
    cropped_groups: list[np.ndarray] = []
    kept_names: list[str] = []

    for name, points in zip(loaded.group_names, loaded.groups):
        mask = np.ones(len(points), dtype=bool)
        bounds = (
            (args.x_min, args.x_max),
            (args.y_min, args.y_max),
            (args.z_min, args.z_max),
        )
        for axis, (lower, upper) in enumerate(bounds):
            if lower is not None:
                mask &= points[:, axis] >= lower
            if upper is not None:
                mask &= points[:, axis] <= upper

        cropped = points[mask]
        if len(cropped) == 0:
            continue
        cropped_groups.append(np.ascontiguousarray(cropped))
        kept_names.append(name)

    if not cropped_groups:
        raise ValueError("crop removed all points")

    merged = np.concatenate(cropped_groups, axis=0)
    return LoadedPoints(merged_points=np.ascontiguousarray(merged), groups=cropped_groups, group_names=kept_names)


def sphere_from_four_points(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    p0 = points[0]
    matrix = 2.0 * (points[1:] - p0)
    rhs = np.sum(points[1:] ** 2, axis=1) - float(np.dot(p0, p0))
    if np.linalg.cond(matrix) > 1e8:
        return None
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
    if not np.all(np.isfinite(center)) or not np.isfinite(radius):
        return None
    return center, radius


def ransac_center(
    points: np.ndarray,
    radius_mm: float,
    iterations: int,
    inlier_threshold_mm: float,
    radius_tolerance_mm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 4:
        raise ValueError("at least 4 points are required")
    rng = np.random.default_rng(seed)
    best_center = None
    best_mask = None
    best_score = (-1, math.inf)

    for _ in range(iterations):
        sample = points[rng.choice(len(points), 4, replace=False)]
        estimate = sphere_from_four_points(sample)
        if estimate is None:
            continue
        center, estimated_radius = estimate
        if abs(estimated_radius - radius_mm) > radius_tolerance_mm:
            continue
        abs_r = np.abs(np.linalg.norm(points - center, axis=1) - radius_mm)
        mask = abs_r <= inlier_threshold_mm
        count = int(np.count_nonzero(mask))
        if count < 4:
            continue
        median = float(np.median(abs_r[mask]))
        score = (count, -median)
        if score > (best_score[0], -best_score[1]):
            best_center, best_mask = center, mask
            best_score = (count, median)

    if best_center is None or best_mask is None:
        center = np.mean(points, axis=0)
        abs_r = np.abs(np.linalg.norm(points - center, axis=1) - radius_mm)
        cutoff = max(inlier_threshold_mm, float(np.quantile(abs_r, 0.35)))
        return center, abs_r <= cutoff
    return best_center, best_mask


def fit_fixed_radius_sphere(
    points: np.ndarray,
    *,
    radius_mm: float,
    ransac_iterations: int,
    ransac_threshold_mm: float,
    radius_tolerance_mm: float,
    max_iterations: int,
    tolerance_mm: float,
    huber_delta_mm: float,
    final_inlier_threshold_mm: float,
    min_inliers: int,
    seed: int,
) -> SphereFitResult:
    if len(points) < max(4, min_inliers):
        raise ValueError(f"need at least {max(4, min_inliers)} points")

    center, active = ransac_center(
        points, radius_mm, ransac_iterations,
        ransac_threshold_mm, radius_tolerance_mm, seed,
    )
    if np.count_nonzero(active) < min_inliers:
        raise RuntimeError(
            f"RANSAC found only {np.count_nonzero(active)} inliers; "
            "crop around the sphere or relax thresholds"
        )

    converged = False
    completed = 0
    for iteration in range(1, max_iterations + 1):
        selected = points[active]
        vectors = selected - center
        distances = np.linalg.norm(vectors, axis=1)
        valid = distances > 1e-9
        vectors = vectors[valid]
        distances = distances[valid]
        if len(distances) < min_inliers:
            raise RuntimeError("too few valid inliers during ICP")

        residual = distances - radius_mm
        jacobian = -vectors / distances[:, None]
        abs_r = np.abs(residual)
        weights = np.ones_like(abs_r)
        large = abs_r > huber_delta_mm
        weights[large] = huber_delta_mm / abs_r[large]
        sw = np.sqrt(weights)
        delta, *_ = np.linalg.lstsq(jacobian * sw[:, None], -residual * sw, rcond=None)
        center += delta
        completed = iteration

        all_abs = np.abs(np.linalg.norm(points - center, axis=1) - radius_mm)
        robust_scale = 2.5 * float(np.median(all_abs[active])) if np.any(active) else 0.0
        active = all_abs <= max(final_inlier_threshold_mm, robust_scale)
        if np.count_nonzero(active) < min_inliers:
            raise RuntimeError("ICP lost too many inliers")
        if float(np.linalg.norm(delta)) <= tolerance_mm:
            converged = True
            break

    residuals = np.linalg.norm(points - center, axis=1) - radius_mm
    inliers = np.abs(residuals) <= final_inlier_threshold_mm
    if np.count_nonzero(inliers) < min_inliers:
        inliers = active

    return SphereFitResult(
        center=np.asarray(center),
        radius_mm=float(radius_mm),
        residuals_mm=residuals,
        inlier_mask=inliers,
        iterations=completed,
        converged=converged,
    )


def save_results(path: Path, input_path: Path, result: SphereFitResult) -> None:
    payload = {
        "input": str(input_path),
        "center_base_mm": result.center.tolist(),
        "fixed_radius_mm": result.radius_mm,
        "inlier_count": result.inlier_count,
        "total_point_count": int(len(result.residuals_mm)),
        "rms_radial_error_mm": result.rms_mm,
        "mae_radial_error_mm": result.mae_mm,
        "median_abs_radial_error_mm": result.median_abs_mm,
        "p95_abs_radial_error_mm": result.p95_abs_mm,
        "max_abs_radial_error_mm": result.max_abs_mm,
        "iterations": result.iterations,
        "converged": result.converged,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_distinct_colors(n: int) -> list[tuple[float, float, float, float]]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        s = 0.85
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((float(r), float(g), float(b), 0.85))
    return colors


def visualize(loaded: LoadedPoints, result: SphereFitResult, point_size: float, sphere_alpha: float) -> None:
    app = pg.mkQApp("Saved scan sphere fitting")
    window = QtWidgets.QWidget()
    window.setWindowTitle("Saved world scan: fixed-radius sphere fit")
    window.resize(1280, 850)
    root = QtWidgets.QVBoxLayout(window)

    c = result.center
    text = QtWidgets.QLabel(
        f"center=[{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}] mm | "
        f"radius={result.radius_mm:.3f} mm | "
        f"inliers={result.inlier_count}/{len(loaded.merged_points)} | "
        f"RMS={result.rms_mm:.4f} mm | MAE={result.mae_mm:.4f} mm | "
        f"median={result.median_abs_mm:.4f} mm | P95={result.p95_abs_mm:.4f} mm | "
        f"max={result.max_abs_mm:.4f} mm | iterations={result.iterations} | "
        f"converged={result.converged}"
    )
    text.setWordWrap(True)
    root.addWidget(text)

    legend_text = "Profiles: " + ", ".join(
        f"{i}:{name}" for i, name in enumerate(loaded.group_names)
    ) + " | yellow: sphere inliers | magenta: sphere center"
    legend = QtWidgets.QLabel(legend_text)
    legend.setWordWrap(True)
    root.addWidget(legend)

    view = gl.GLViewWidget()
    view.setBackgroundColor((12, 14, 18))
    view.setCameraPosition(distance=120.0, elevation=25, azimuth=45)
    root.addWidget(view, stretch=1)

    grid = gl.GLGridItem()
    grid.setSize(x=100, y=100)
    grid.setSpacing(x=10, y=10)
    grid.translate(float(c[0]), float(c[1]), float(c[2]))
    view.addItem(grid)

    colors = generate_distinct_colors(len(loaded.groups))

    # Draw each profile/capture in a different color.
    for idx, (group, color) in enumerate(zip(loaded.groups, colors)):
        draw = group
        if len(draw) > 50_000:
            draw = draw[::math.ceil(len(draw) / 50_000)]
        view.addItem(gl.GLScatterPlotItem(
            pos=draw.astype(np.float32),
            color=color,
            size=point_size,
            pxMode=True,
        ))

    # Draw fitted inliers on top so the fitted sphere support is easy to see.
    inliers = loaded.merged_points[result.inlier_mask]
    if len(inliers) > 40_000:
        inliers = inliers[::math.ceil(len(inliers) / 40_000)]
    view.addItem(gl.GLScatterPlotItem(
        pos=inliers.astype(np.float32),
        color=(1.0, 0.92, 0.10, 0.95),
        size=max(2.5, point_size + 1.5),
        pxMode=True,
    ))

    # More opaque reference sphere.
    mesh = gl.MeshData.sphere(rows=40, cols=80, radius=result.radius_mm)
    translated = gl.MeshData(
        vertexes=mesh.vertexes() + result.center.reshape(1, 3),
        faces=mesh.faces(),
    )
    view.addItem(gl.GLMeshItem(
        meshdata=translated,
        smooth=True,
        color=(0.70, 0.78, 0.92, sphere_alpha),
        shader="shaded",
        drawEdges=True,
        edgeColor=(1.0, 1.0, 1.0, 0.9),
    ))

    view.addItem(gl.GLScatterPlotItem(
        pos=result.center.astype(np.float32).reshape(1, 3),
        color=(1.0, 0.0, 1.0, 1.0),
        size=14.0,
        pxMode=True,
    ))

    view.opts["center"] = pg.Vector(float(c[0]), float(c[1]), float(c[2]))
    window.show()
    app.exec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a fixed-radius sphere to a saved manual_world_scan.npz and display each profile with a different color"
    )
    parser.add_argument("--input", type=Path, default=Path("runs/real/manual_world_scan.npz"))
    parser.add_argument(
        "--source", default="merged",
        help="merged, latest, or capture index such as 0, 1, 2",
    )
    parser.add_argument("--radius-mm", type=float, default=5.0)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--ransac-threshold-mm", type=float, default=0.4)
    parser.add_argument("--radius-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--huber-delta-mm", type=float, default=0.2)
    parser.add_argument("--inlier-threshold-mm", type=float, default=0.5)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--result-json", type=Path, default=Path("runs/real/sphere_fit_result.json"))
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--sphere-alpha", type=float, default=0.55)
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = apply_crop_to_groups(load_saved_points(args.input, args.source), args)
    points = loaded.merged_points

    result = fit_fixed_radius_sphere(
        points,
        radius_mm=args.radius_mm,
        ransac_iterations=args.ransac_iterations,
        ransac_threshold_mm=args.ransac_threshold_mm,
        radius_tolerance_mm=args.radius_tolerance_mm,
        max_iterations=args.max_iterations,
        tolerance_mm=args.tolerance_mm,
        huber_delta_mm=args.huber_delta_mm,
        final_inlier_threshold_mm=args.inlier_threshold_mm,
        min_inliers=args.min_inliers,
        seed=args.seed,
    )
    save_results(args.result_json, args.input, result)

    c = result.center
    print(f"input: {args.input}")
    print(f"points used: {len(points)}")
    print(f"profile groups: {len(loaded.groups)}")
    print(f"sphere center [mm]: [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    print(f"fixed radius [mm]: {result.radius_mm:.6f}")
    print(f"inliers: {result.inlier_count}/{len(points)}")
    print(f"RMS radial error [mm]: {result.rms_mm:.6f}")
    print(f"MAE radial error [mm]: {result.mae_mm:.6f}")
    print(f"median abs error [mm]: {result.median_abs_mm:.6f}")
    print(f"P95 abs error [mm]: {result.p95_abs_mm:.6f}")
    print(f"max abs error [mm]: {result.max_abs_mm:.6f}")
    print(f"iterations: {result.iterations}, converged={result.converged}")
    print(f"result JSON: {args.result_json}")

    if not args.no_gui:
        visualize(loaded, result, args.point_size, args.sphere_alpha)


if __name__ == "__main__":
    main()