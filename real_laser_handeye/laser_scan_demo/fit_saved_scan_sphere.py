from __future__ import annotations

"""
Example
-------
PYTHONPATH=. python3 real_laser_handeye/laser_scan_demo/fit_saved_scan_sphere.py \
  --input runs/real/two_point_stop_and_scan.npz \
  --auto-floor-z-min \
  --z-min-margin-mm 0 \
  --point-size 2
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    _OPEN3D_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Open3D may fail through optional ML dependencies.
    o3d = None
    _OPEN3D_IMPORT_ERROR = exc


@dataclass
class SphereFitResult:
    mode: str
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
    def total_count(self) -> int:
        return int(len(self.residuals_mm))

    def _inlier_residuals(self) -> np.ndarray:
        residuals = self.residuals_mm[self.inlier_mask]
        if len(residuals) == 0:
            raise RuntimeError(f"{self.mode} fit has no inlier residuals")
        return residuals

    @property
    def rms_mm(self) -> float:
        residuals = self._inlier_residuals()
        return float(np.sqrt(np.mean(residuals * residuals)))

    @property
    def mae_mm(self) -> float:
        return float(np.mean(np.abs(self._inlier_residuals())))

    @property
    def median_abs_mm(self) -> float:
        return float(np.median(np.abs(self._inlier_residuals())))

    @property
    def p95_abs_mm(self) -> float:
        return float(np.percentile(np.abs(self._inlier_residuals()), 95))

    @property
    def max_abs_mm(self) -> float:
        return float(np.max(np.abs(self._inlier_residuals())))


@dataclass
class LoadedPoints:
    merged_points: np.ndarray
    groups: list[np.ndarray]
    group_names: list[str]


@dataclass
class PreprocessStats:
    input_points: int
    after_crop_points: int
    z_min_removed_points: int
    profile_spike_removed_points: int
    sphere_outlier_removed_points: int
    final_points: int
    group_stats: list[dict[str, object]]


@dataclass(frozen=True)
class FloorZDetection:
    floor_z_mm: float
    z_min_mm: float
    clearance_mm: float
    profile_quantile: float
    candidate_profile_count: int
    inlier_profile_count: int
    robust_sigma_mm: float


def _sorted_capture_keys(keys: set[str]) -> list[str]:
    return sorted(
        key
        for key in keys
        if key.startswith("capture_") and key.endswith("_points_base")
    )


def _make_loaded(groups: list[np.ndarray], names: list[str]) -> LoadedPoints:
    if not groups:
        raise ValueError("no non-empty profile groups remain")
    merged = np.concatenate(groups, axis=0)
    return LoadedPoints(
        merged_points=np.ascontiguousarray(merged, dtype=float),
        groups=[np.ascontiguousarray(group, dtype=float) for group in groups],
        group_names=list(names),
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
                names = [key.replace("_points_base", "") for key in capture_keys]
            elif "points_base_merged" in keys:
                groups = [np.asarray(data["points_base_merged"], dtype=float)]
                names = ["merged"]
            else:
                raise KeyError(
                    "NPZ has neither capture_*_points_base nor points_base_merged"
                )
        elif source == "latest":
            if not capture_keys:
                raise KeyError("NPZ has no capture_*_points_base arrays")
            key = capture_keys[-1]
            groups = [np.asarray(data[key], dtype=float)]
            names = [key.replace("_points_base", "")]
        else:
            key = f"capture_{int(source):04d}_points_base"
            if key not in keys:
                raise KeyError(f"NPZ has no key: {key}")
            groups = [np.asarray(data[key], dtype=float)]
            names = [key.replace("_points_base", "")]

    clean_groups: list[np.ndarray] = []
    clean_names: list[str] = []
    for name, group in zip(names, groups):
        if group.ndim != 2 or group.shape[1] != 3:
            raise ValueError(f"loaded points must have shape (N,3), got {group.shape}")
        group = group[np.all(np.isfinite(group), axis=1)]
        if len(group) == 0:
            continue
        clean_groups.append(group)
        clean_names.append(name)

    return _make_loaded(clean_groups, clean_names)


def detect_floor_z_min(
    loaded: LoadedPoints,
    *,
    known_radius_mm: float,
    profile_quantile: float = 0.10,
    clearance_mm: float | None = None,
) -> FloorZDetection:
    """Detect the horizontal floor band and return an automatic Z crop.

    Each laser profile contributes one low-Z quantile so long profiles cannot
    dominate the estimate. A median/MAD gate rejects profiles containing deep
    spikes or object-only samples. The crop is placed above the detected floor;
    by default the clearance is 80% of the known sphere radius.
    """
    if known_radius_mm <= 0:
        raise ValueError("known radius must be positive")
    if not 0.0 < profile_quantile < 0.5:
        raise ValueError("floor profile quantile must be in (0, 0.5)")
    if clearance_mm is None:
        clearance_mm = 0.8 * known_radius_mm
    if clearance_mm < 0:
        raise ValueError("floor clearance must be non-negative")

    candidates = np.asarray(
        [
            np.quantile(group[:, 2], profile_quantile)
            for group in loaded.groups
            if len(group)
        ],
        dtype=float,
    )
    candidates = candidates[np.isfinite(candidates)]
    if len(candidates) < 3:
        raise RuntimeError("too few profiles for automatic floor detection")
    initial_median = float(np.median(candidates))
    initial_mad = float(np.median(np.abs(candidates - initial_median)))
    robust_sigma = 1.4826 * initial_mad
    gate_mm = max(0.25, 3.5 * max(robust_sigma, 1e-6))
    inliers = np.abs(candidates - initial_median) <= gate_mm
    if np.count_nonzero(inliers) < max(3, int(math.ceil(0.35 * len(candidates)))):
        inliers = np.ones(len(candidates), dtype=bool)
    floor_z = float(np.median(candidates[inliers]))
    inlier_mad = float(np.median(np.abs(candidates[inliers] - floor_z)))
    return FloorZDetection(
        floor_z_mm=floor_z,
        z_min_mm=float(floor_z + clearance_mm),
        clearance_mm=float(clearance_mm),
        profile_quantile=float(profile_quantile),
        candidate_profile_count=int(len(candidates)),
        inlier_profile_count=int(np.count_nonzero(inliers)),
        robust_sigma_mm=float(1.4826 * inlier_mad),
    )


def apply_crop_to_groups(
    loaded: LoadedPoints,
    args: argparse.Namespace,
) -> LoadedPoints:
    cropped_groups: list[np.ndarray] = []
    kept_names: list[str] = []
    bounds = (
        (args.x_min, args.x_max),
        (args.y_min, args.y_max),
        (args.z_min, args.z_max),
    )

    for name, points in zip(loaded.group_names, loaded.groups):
        mask = np.ones(len(points), dtype=bool)
        for axis, (lower, upper) in enumerate(bounds):
            if lower is not None:
                mask &= points[:, axis] >= lower
            if upper is not None:
                mask &= points[:, axis] <= upper
        cropped = points[mask]
        if len(cropped) > 0:
            cropped_groups.append(cropped)
            kept_names.append(name)

    return _make_loaded(cropped_groups, kept_names)


def remove_profile_z_min(
    loaded: LoadedPoints,
    margin_mm: float,
    min_points_per_group: int,
) -> tuple[LoadedPoints, list[dict[str, object]], int]:
    """Remove points close to each profile's minimum base-frame z value.

    A small positive margin removes a clipped or repeated z-min band. With
    margin=0, only points numerically equal to the profile minimum are removed.
    """
    if margin_mm < 0:
        raise ValueError("z-min margin must be non-negative")

    output_groups: list[np.ndarray] = []
    output_names: list[str] = []
    stats: list[dict[str, object]] = []
    total_removed = 0

    for name, points in zip(loaded.group_names, loaded.groups):
        z_min = float(np.min(points[:, 2]))
        threshold = z_min + margin_mm
        keep = points[:, 2] > threshold
        kept = points[keep]
        removed = int(len(points) - len(kept))
        total_removed += removed

        row: dict[str, object] = {
            "name": name,
            "before": int(len(points)),
            "z_min_mm": z_min,
            "z_min_threshold_mm": threshold,
            "z_min_removed": removed,
            "after_z_min": int(len(kept)),
            "kept": bool(len(kept) >= min_points_per_group),
        }
        stats.append(row)

        if len(kept) >= min_points_per_group:
            output_groups.append(kept)
            output_names.append(name)

    return _make_loaded(output_groups, output_names), stats, total_removed



def _short_run_peaks(
    mask: np.ndarray, scores: np.ndarray, max_run_length: int
) -> np.ndarray:
    """Select only the strongest sample from each short candidate run.

    One isolated spike often makes the deviations at its two neighbours large as
    well. Removing an entire candidate run would therefore erase valid adjacent
    sphere samples. Keeping only the maximum-score index avoids that dilation.
    """
    output = np.zeros_like(mask, dtype=bool)
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end - start <= max_run_length:
            peak = start + int(np.argmax(scores[start:end]))
            output[peak] = True
        start = end
    return output


def remove_profile_spikes(
    loaded: LoadedPoints,
    absolute_threshold_mm: float,
    sigma: float,
    max_run_length: int,
    min_points_per_group: int,
    group_stats: list[dict[str, object]],
) -> tuple[LoadedPoints, int]:
    """Remove isolated acquisition-order spikes without using a sphere model.

    For each interior point, the local deviation is the Euclidean distance from
    the midpoint of its two neighbours. A point is a spike only when the
    deviation exceeds both an absolute threshold and a robust MAD threshold.
    Long candidate runs are preserved because they are more likely to represent
    a real surface segment or an occlusion boundary than isolated sensor noise.
    """
    if absolute_threshold_mm < 0:
        raise ValueError("profile spike threshold must be non-negative")
    if sigma <= 0:
        raise ValueError("profile spike sigma must be positive")
    if max_run_length < 1:
        raise ValueError("max spike run length must be at least 1")

    stat_map = {str(row["name"]): row for row in group_stats}
    output_groups: list[np.ndarray] = []
    output_names: list[str] = []
    total_removed = 0

    for name, points in zip(loaded.group_names, loaded.groups):
        remove = np.zeros(len(points), dtype=bool)
        threshold = math.inf
        median = 0.0
        robust_sigma = 0.0

        if len(points) >= 3:
            local_prediction = 0.5 * (points[:-2] + points[2:])
            deviations = np.linalg.norm(points[1:-1] - local_prediction, axis=1)
            median = float(np.median(deviations))
            mad = float(np.median(np.abs(deviations - median)))
            robust_sigma = 1.4826 * mad
            threshold = max(
                absolute_threshold_mm,
                median + sigma * max(robust_sigma, 1e-9),
            )
            candidates = np.zeros(len(points), dtype=bool)
            scores = np.zeros(len(points), dtype=float)
            candidates[1:-1] = deviations > threshold
            scores[1:-1] = deviations
            remove = _short_run_peaks(candidates, scores, max_run_length)

        kept = points[~remove]
        removed = int(np.count_nonzero(remove))
        total_removed += removed

        row = stat_map.get(name)
        if row is not None:
            row["profile_spike_threshold_mm"] = (
                None if not np.isfinite(threshold) else float(threshold)
            )
            row["profile_spike_median_mm"] = median
            row["profile_spike_robust_sigma_mm"] = robust_sigma
            row["profile_spike_removed"] = removed
            row["after_profile_spike"] = int(len(kept))

        if len(kept) >= min_points_per_group:
            output_groups.append(kept)
            output_names.append(name)
        elif row is not None:
            row["kept"] = False

    return _make_loaded(output_groups, output_names), total_removed

def sphere_from_four_points(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape != (4, 3):
        raise ValueError("sphere_from_four_points expects shape (4,3)")
    p0 = points[0]
    matrix = 2.0 * (points[1:] - p0)
    rhs = np.sum(points[1:] ** 2, axis=1) - float(np.dot(p0, p0))
    if not np.all(np.isfinite(matrix)) or np.linalg.cond(matrix) > 1e8:
        return None
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
    if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
        return None
    return center, radius


def algebraic_sphere_initialization(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 4:
        raise ValueError("at least four points are required")
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    rhs = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    center = solution[:3]
    radius_sq = float(np.dot(center, center) + solution[3])
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        center = np.mean(points, axis=0)
        radius = float(np.median(np.linalg.norm(points - center, axis=1)))
    else:
        radius = math.sqrt(radius_sq)
    return np.asarray(center, dtype=float), float(radius)


def huber_weights(residuals: np.ndarray, delta_mm: float) -> np.ndarray:
    if delta_mm <= 0:
        return np.ones_like(residuals)
    absolute = np.abs(residuals)
    weights = np.ones_like(absolute)
    large = absolute > delta_mm
    weights[large] = delta_mm / np.maximum(absolute[large], 1e-12)
    return weights


def robust_active_mask(
    absolute_residuals: np.ndarray,
    current_mask: np.ndarray,
    hard_threshold_mm: float,
) -> np.ndarray:
    if np.any(current_mask):
        active_abs = absolute_residuals[current_mask]
    else:
        active_abs = absolute_residuals
    median = float(np.median(active_abs))
    mad = float(np.median(np.abs(active_abs - median)))
    robust_sigma = 1.4826 * mad
    adaptive_threshold = median + 3.5 * max(robust_sigma, 1e-6)
    threshold = max(hard_threshold_mm, adaptive_threshold)
    return absolute_residuals <= threshold


def ransac_free_sphere(
    points: np.ndarray,
    iterations: int,
    inlier_threshold_mm: float,
    expected_radius_mm: float,
    candidate_radius_tolerance_mm: float,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 4:
        raise ValueError("at least four points are required")
    rng = np.random.default_rng(seed)
    best_center: np.ndarray | None = None
    best_radius: float | None = None
    best_mask: np.ndarray | None = None
    best_count = -1
    best_median = math.inf

    for _ in range(iterations):
        sample = points[rng.choice(len(points), 4, replace=False)]
        estimate = sphere_from_four_points(sample)
        if estimate is None:
            continue
        center, radius = estimate
        if (
            candidate_radius_tolerance_mm > 0
            and abs(radius - expected_radius_mm) > candidate_radius_tolerance_mm
        ):
            continue
        absolute = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        mask = absolute <= inlier_threshold_mm
        count = int(np.count_nonzero(mask))
        if count < 4:
            continue
        median = float(np.median(absolute[mask]))
        if count > best_count or (count == best_count and median < best_median):
            best_center = center
            best_radius = radius
            best_mask = mask
            best_count = count
            best_median = median

    if best_center is None or best_radius is None or best_mask is None:
        best_center, best_radius = algebraic_sphere_initialization(points)
        absolute = np.abs(np.linalg.norm(points - best_center, axis=1) - best_radius)
        cutoff = max(inlier_threshold_mm, float(np.quantile(absolute, 0.5)))
        best_mask = absolute <= cutoff

    return best_center, best_radius, best_mask


def fit_free_radius_sphere(
    points: np.ndarray,
    *,
    expected_radius_mm: float,
    ransac_iterations: int,
    ransac_threshold_mm: float,
    candidate_radius_tolerance_mm: float,
    max_iterations: int,
    tolerance_mm: float,
    huber_delta_mm: float,
    final_inlier_threshold_mm: float,
    min_inliers: int,
    seed: int,
) -> SphereFitResult:
    if len(points) < max(4, min_inliers):
        raise ValueError(f"need at least {max(4, min_inliers)} points")

    center, radius, active = ransac_free_sphere(
        points=points,
        iterations=ransac_iterations,
        inlier_threshold_mm=ransac_threshold_mm,
        expected_radius_mm=expected_radius_mm,
        candidate_radius_tolerance_mm=candidate_radius_tolerance_mm,
        seed=seed,
    )
    if np.count_nonzero(active) < min_inliers:
        raise RuntimeError(
            f"free-radius RANSAC found only {np.count_nonzero(active)} inliers"
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
            raise RuntimeError("too few valid inliers during free-radius fitting")

        residuals = distances - radius
        jacobian = np.column_stack(
            (-vectors / distances[:, None], -np.ones(len(distances)))
        )
        weights = huber_weights(residuals, huber_delta_mm)
        sqrt_weights = np.sqrt(weights)
        delta, *_ = np.linalg.lstsq(
            jacobian * sqrt_weights[:, None],
            -residuals * sqrt_weights,
            rcond=None,
        )
        center += delta[:3]
        radius += float(delta[3])
        completed = iteration

        if not np.isfinite(radius) or radius <= 0:
            raise RuntimeError("free-radius fit produced an invalid radius")

        absolute = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        active = robust_active_mask(absolute, active, final_inlier_threshold_mm)
        if np.count_nonzero(active) < min_inliers:
            raise RuntimeError("free-radius fit lost too many inliers")
        if float(np.linalg.norm(delta)) <= tolerance_mm:
            converged = True
            break

    residuals = np.linalg.norm(points - center, axis=1) - radius
    final_mask = np.abs(residuals) <= final_inlier_threshold_mm
    if np.count_nonzero(final_mask) < min_inliers:
        final_mask = active

    return SphereFitResult(
        mode="free_radius",
        center=np.asarray(center, dtype=float),
        radius_mm=float(radius),
        residuals_mm=np.asarray(residuals, dtype=float),
        inlier_mask=np.asarray(final_mask, dtype=bool),
        iterations=completed,
        converged=converged,
    )


def fit_fixed_radius_sphere(
    points: np.ndarray,
    *,
    radius_mm: float,
    initial_center: np.ndarray,
    max_iterations: int,
    tolerance_mm: float,
    huber_delta_mm: float,
    final_inlier_threshold_mm: float,
    min_inliers: int,
) -> SphereFitResult:
    if len(points) < max(4, min_inliers):
        raise ValueError(f"need at least {max(4, min_inliers)} points")

    center = np.asarray(initial_center, dtype=float).copy()
    active = np.ones(len(points), dtype=bool)
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
            raise RuntimeError("too few valid inliers during fixed-radius fitting")

        residuals = distances - radius_mm
        jacobian = -vectors / distances[:, None]
        weights = huber_weights(residuals, huber_delta_mm)
        sqrt_weights = np.sqrt(weights)
        delta, *_ = np.linalg.lstsq(
            jacobian * sqrt_weights[:, None],
            -residuals * sqrt_weights,
            rcond=None,
        )
        center += delta
        completed = iteration

        absolute = np.abs(np.linalg.norm(points - center, axis=1) - radius_mm)
        active = robust_active_mask(absolute, active, final_inlier_threshold_mm)
        if np.count_nonzero(active) < min_inliers:
            raise RuntimeError("fixed-radius fit lost too many inliers")
        if float(np.linalg.norm(delta)) <= tolerance_mm:
            converged = True
            break

    residuals = np.linalg.norm(points - center, axis=1) - radius_mm
    final_mask = np.abs(residuals) <= final_inlier_threshold_mm
    if np.count_nonzero(final_mask) < min_inliers:
        final_mask = active

    return SphereFitResult(
        mode="fixed_radius",
        center=np.asarray(center, dtype=float),
        radius_mm=float(radius_mm),
        residuals_mm=np.asarray(residuals, dtype=float),
        inlier_mask=np.asarray(final_mask, dtype=bool),
        iterations=completed,
        converged=converged,
    )



def apply_merged_mask_to_groups(
    loaded: LoadedPoints,
    merged_mask: np.ndarray,
    min_points_per_group: int,
    min_profile_inlier_ratio: float,
    group_stats: list[dict[str, object]],
) -> LoadedPoints:
    if len(merged_mask) != len(loaded.merged_points):
        raise ValueError("merged mask length does not match merged points")
    if not 0.0 <= min_profile_inlier_ratio <= 1.0:
        raise ValueError("min profile inlier ratio must be in [0, 1]")

    output_groups: list[np.ndarray] = []
    output_names: list[str] = []
    offset = 0
    stat_map = {str(row["name"]): row for row in group_stats}

    for name, group in zip(loaded.group_names, loaded.groups):
        local_mask = merged_mask[offset : offset + len(group)]
        offset += len(group)
        kept = group[local_mask]
        ratio = float(len(kept) / max(len(group), 1))
        accepted = (
            len(kept) >= min_points_per_group
            and ratio >= min_profile_inlier_ratio
        )
        row = stat_map.get(name)
        if row is not None:
            row["sphere_outlier_removed"] = int(len(group) - len(kept))
            row["sphere_inlier_ratio"] = ratio
            row["final"] = int(len(kept))
            row["kept_after_outlier_filter"] = accepted
        if accepted:
            output_groups.append(kept)
            output_names.append(name)

    return _make_loaded(output_groups, output_names)


def _residual_stats(residuals: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    absolute = np.abs(residuals)
    return {
        "rms_radial_error_mm": float(np.sqrt(np.mean(residuals * residuals))),
        "mae_radial_error_mm": float(np.mean(absolute)),
        "median_abs_radial_error_mm": float(np.median(absolute)),
        "p95_abs_radial_error_mm": float(np.percentile(absolute, 95)),
        "max_abs_radial_error_mm": float(np.max(absolute)),
    }


def fit_payload(result: SphereFitResult) -> dict[str, object]:
    inlier_residuals = result.residuals_mm[result.inlier_mask]
    return {
        "mode": result.mode,
        "center_base_mm": result.center.tolist(),
        "radius_mm": result.radius_mm,
        "inlier_count": result.inlier_count,
        "total_point_count": result.total_count,
        "inlier_ratio": float(result.inlier_count / max(result.total_count, 1)),
        "inlier_metrics": _residual_stats(inlier_residuals),
        "all_precleaned_point_metrics": _residual_stats(result.residuals_mm),
        "iterations": result.iterations,
        "converged": result.converged,
    }


def save_results(
    path: Path,
    input_path: Path,
    known_radius_mm: float,
    free_result: SphereFitResult,
    fixed_result: SphereFitResult,
    preprocessing: PreprocessStats,
    settings: dict[str, object],
) -> None:
    payload = {
        "input": str(input_path),
        "known_radius_mm": known_radius_mm,
        "settings": settings,
        "preprocessing": {
            "input_points": preprocessing.input_points,
            "after_crop_points": preprocessing.after_crop_points,
            "z_min_removed_points": preprocessing.z_min_removed_points,
            "profile_spike_removed_points": preprocessing.profile_spike_removed_points,
            "sphere_outlier_removed_points": preprocessing.sphere_outlier_removed_points,
            "final_points": preprocessing.final_points,
            "groups": preprocessing.group_stats,
        },
        "free_radius_fit": fit_payload(free_result),
        "fixed_radius_fit": fit_payload(fixed_result),
        "comparison": {
            "radius_error_mm": free_result.radius_mm - known_radius_mm,
            "diameter_error_mm": 2.0 * (free_result.radius_mm - known_radius_mm),
            "center_difference_mm": float(
                np.linalg.norm(free_result.center - fixed_result.center)
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# Restrained, color-blind-friendly palette based on common paper figures.
PAPER_PALETTE = [
    (0.121, 0.466, 0.705),
    (0.850, 0.325, 0.098),
    (0.172, 0.627, 0.172),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.400, 0.400, 0.400),
    (0.737, 0.741, 0.133),
]


def _require_open3d() -> None:
    if o3d is None:
        raise RuntimeError(
            "Open3D visualization is not installed. Install it with "
            "`python -m pip install open3d==0.19.0` or run with --no-gui. "
            f"Import error: {_OPEN3D_IMPORT_ERROR}"
        )


def _point_cloud(points: np.ndarray, color: tuple[float, float, float]):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    cloud.paint_uniform_color(color)
    return cloud


def _polyline(points: np.ndarray, color: tuple[float, float, float]):
    points = np.asarray(points, dtype=float)
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    if len(points) >= 2:
        indices = np.column_stack(
            (np.arange(len(points) - 1), np.arange(1, len(points)))
        )
    else:
        indices = np.empty((0, 2), dtype=np.int32)
    lines.lines = o3d.utility.Vector2iVector(indices.astype(np.int32))
    lines.paint_uniform_color(color)
    return lines


def _sphere_wireframe(
    center: np.ndarray,
    radius_mm: float,
    color: tuple[float, float, float],
    resolution: int,
):
    """Create a sparse latitude/longitude sphere wireframe.

    Do not convert a triangulated sphere mesh directly to a LineSet. That draws
    every triangle edge, including diagonal edges, and makes the sphere appear
    almost black when viewed at publication scale.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    radius = float(radius_mm)

    # Keep the grid intentionally sparse for a clean publication-style view.
    meridian_count = max(8, min(12, int(resolution) // 3))
    parallel_count = max(4, min(7, int(resolution) // 5))
    samples_per_curve = max(64, int(resolution) * 2)

    all_points: list[np.ndarray] = []
    all_lines: list[tuple[int, int]] = []

    def append_curve(curve: np.ndarray, closed: bool) -> None:
        start = sum(len(points) for points in all_points)
        all_points.append(curve)
        count = len(curve)
        all_lines.extend((start + i, start + i + 1) for i in range(count - 1))
        if closed and count > 2:
            all_lines.append((start + count - 1, start))

    # Latitude circles, excluding the degenerate poles.
    latitudes = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        parallel_count + 2,
        endpoint=True,
    )[1:-1]
    theta = np.linspace(0.0, 2.0 * np.pi, samples_per_curve, endpoint=False)
    for latitude in latitudes:
        cos_lat = np.cos(latitude)
        curve = np.column_stack(
            (
                radius * cos_lat * np.cos(theta),
                radius * cos_lat * np.sin(theta),
                np.full_like(theta, radius * np.sin(latitude)),
            )
        ) + center
        append_curve(curve, closed=True)

    # Longitude arcs from south to north pole.
    phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, samples_per_curve)
    for longitude in np.linspace(0.0, 2.0 * np.pi, meridian_count, endpoint=False):
        cos_phi = np.cos(phi)
        curve = np.column_stack(
            (
                radius * cos_phi * np.cos(longitude),
                radius * cos_phi * np.sin(longitude),
                radius * np.sin(phi),
            )
        ) + center
        append_curve(curve, closed=False)

    wire = o3d.geometry.LineSet()
    wire.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    wire.lines = o3d.utility.Vector2iVector(np.asarray(all_lines, dtype=np.int32))
    wire.paint_uniform_color(color)
    return wire




def _sphere_mesh(
    center: np.ndarray,
    radius_mm: float,
    color: tuple[float, float, float],
    resolution: int = 40,
):
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(radius_mm), resolution=int(resolution)
    )
    mesh.translate(np.asarray(center, dtype=float))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def _center_marker(
    center: np.ndarray,
    marker_radius: float,
    color: tuple[float, float, float],
):
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(marker_radius), resolution=18
    )
    mesh.translate(np.asarray(center, dtype=float))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _point_material(color: tuple[float, float, float], point_size: float):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [*color, 1.0]
    material.point_size = float(point_size)
    return material


def _line_material(color: tuple[float, float, float], line_width: float):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "unlitLine"
    material.base_color = [*color, 1.0]
    material.line_width = float(line_width)
    return material


def _mesh_material(color: tuple[float, float, float]):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [*color, 1.0]
    material.base_roughness = 0.85
    material.base_metallic = 0.0
    return material


def _transparent_mesh_material(
    color: tuple[float, float, float],
    alpha: float,
):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [*color, float(alpha)]
    material.base_roughness = 0.92
    material.base_metallic = 0.0
    material.has_alpha = True
    return material


def _profile_specs(
    loaded: LoadedPoints,
    color_map: dict[str, tuple[float, float, float]],
    point_size: float,
    line_width: float,
    common_color: tuple[float, float, float] | None = None,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, (name, group) in enumerate(zip(loaded.group_names, loaded.groups)):
        color = common_color if common_color is not None else color_map[name]
        specs.append(
            {
                "name": f"{name}_line_{index}",
                "geometry": _polyline(group, color),
                "material": _line_material(color, line_width),
            }
        )
        specs.append(
            {
                "name": f"{name}_points_{index}",
                "geometry": _point_cloud(group, color),
                "material": _point_material(color, point_size),
            }
        )
    return specs


def _camera_from_points(
    points: np.ndarray,
    center_hint: np.ndarray,
    radius_hint: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    data_center = 0.5 * (minimum + maximum)
    lookat = 0.65 * np.asarray(center_hint, dtype=float) + 0.35 * data_center
    extent = max(float(np.max(maximum - minimum)), 2.0 * radius_hint, 1.0)
    direction = np.array([1.7, -1.9, 1.25], dtype=float)
    direction /= np.linalg.norm(direction)
    eye = lookat + direction * (2.5 * extent)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    return lookat, eye, up


def build_visualization_scenes(
    original: LoadedPoints,
    filtered: LoadedPoints,
    known_radius_mm: float,
    free_result: SphereFitResult,
    fixed_result: SphereFitResult,
    point_size: float,
    line_width: float,
) -> list[tuple[str, list[dict[str, object]]]]:
    _require_open3d()
    color_map = {
        name: PAPER_PALETTE[index % len(PAPER_PALETTE)]
        for index, name in enumerate(original.group_names)
    }

    original_specs = _profile_specs(
        original, color_map, point_size=point_size, line_width=line_width
    )
    filtered_specs = _profile_specs(
        filtered, color_map, point_size=point_size, line_width=line_width
    )

    known_color = (0.76, 0.77, 0.79)
    free_color = (0.68, 0.30, 0.27)
    fixed_center_color = (0.20, 0.20, 0.20)
    marker_radius = max(known_radius_mm * 0.035, 0.04)

    sphere_specs: list[dict[str, object]] = [
        {
            "name": "known_radius_sphere_surface",
            "geometry": _sphere_mesh(
                fixed_result.center,
                known_radius_mm,
                known_color,
                resolution=48,
            ),
            "material": _transparent_mesh_material(known_color, alpha=0.60),
        },
        {
            "name": "known_radius_sphere_wireframe",
            "geometry": _sphere_wireframe(
                fixed_result.center,
                known_radius_mm,
                known_color,
                resolution=30,
            ),
            "material": _line_material(known_color, max(1.0, line_width * 0.75)),
        },
        {
            "name": "free_radius_sphere",
            "geometry": _sphere_wireframe(
                free_result.center,
                free_result.radius_mm,
                free_color,
                resolution=24,
            ),
            "material": _line_material(free_color, max(1.0, line_width * 0.9)),
        },
        {
            "name": "fixed_center",
            "geometry": _center_marker(
                fixed_result.center, marker_radius, fixed_center_color
            ),
            "material": _mesh_material(fixed_center_color),
        },
        {
            "name": "free_center",
            "geometry": _center_marker(
                free_result.center, marker_radius * 0.85, free_color
            ),
            "material": _mesh_material(free_color),
        },
    ]
    # Keep each profile's original publication-palette color in the overlay
    # scene as well, so stages 2 and 3 are visually consistent.
    sphere_specs.extend(
        _profile_specs(
            filtered,
            color_map,
            point_size=max(point_size, 3.0),
            line_width=max(line_width, 1.5),
        )
    )

    return [
        ("1. Original profiles", original_specs),
        ("2. Filtered profiles", filtered_specs),
        ("3. Profiles on reference sphere", sphere_specs),
    ]


def show_open3d_scenes(
    scenes: list[tuple[str, list[dict[str, object]]]],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    field_of_view: float,
    stage: str,
) -> None:
    _require_open3d()
    lookat, eye, up = camera
    selected = {
        "original": [scenes[0]],
        "filtered": [scenes[1]],
        "sphere": [scenes[2]],
        "all": scenes,
    }[stage]

    for title, specs in selected:
        print(f"Opening Open3D view: {title}")
        o3d.visualization.draw(
            geometry=specs,
            title=title,
            width=int(width),
            height=int(height),
            lookat=np.asarray(lookat, dtype=float),
            eye=np.asarray(eye, dtype=float),
            up=np.asarray(up, dtype=float),
            field_of_view=float(field_of_view),
            bg_color=(1.0, 1.0, 1.0, 1.0),
            show_skybox=False,
            show_ui=True,
            raw_mode=False,
        )


def save_open3d_scenes(
    scenes: list[tuple[str, list[dict[str, object]]]],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
    width: int,
    height: int,
    field_of_view: float,
) -> None:
    _require_open3d()
    output_dir.mkdir(parents=True, exist_ok=True)
    lookat, eye, up = camera
    filenames = ["01_original.png", "02_filtered.png", "03_on_sphere.png"]

    for (title, specs), filename in zip(scenes, filenames):
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
            np.array([0.0, -1.0, -1.0], dtype=np.float32),
        )
        for spec in specs:
            renderer.scene.add_geometry(
                str(spec["name"]), spec["geometry"], spec["material"]
            )
        renderer.setup_camera(
            float(field_of_view),
            np.asarray(lookat, dtype=np.float32),
            np.asarray(eye, dtype=np.float32),
            np.asarray(up, dtype=np.float32),
        )
        image = renderer.render_to_image()
        destination = output_dir / filename
        if not o3d.io.write_image(str(destination), image, 9):
            raise RuntimeError(f"failed to save rendered image: {destination}")
        print(f"saved: {destination} ({title})")
        del renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit free-radius and fixed-radius spheres, robustly filter saved "
            "laser profiles, and visualize the three stages with Open3D."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/real/manual_world_scan.npz"),
    )
    parser.add_argument(
        "--source",
        default="merged",
        help="merged, latest, or capture index such as 0, 1, 2",
    )
    parser.add_argument("--radius-mm", type=float, default=5.0)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--ransac-threshold-mm", type=float, default=0.4)
    parser.add_argument(
        "--radius-tolerance-mm",
        type=float,
        default=2.0,
        help=(
            "Allowed radius deviation for free-sphere RANSAC initialization. "
            "Set <=0 to disable this initialization constraint."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--huber-delta-mm", type=float, default=0.2)
    parser.add_argument(
        "--inlier-threshold-mm",
        type=float,
        default=0.5,
        help=(
            "Absolute fixed-radius radial residual threshold used only to create "
            "the filtered visualization point set."
        ),
    )
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--min-points-per-group", type=int, default=4)
    parser.add_argument(
        "--min-profile-inlier-ratio",
        type=float,
        default=0.20,
        help="Reject a profile from the filtered view when too little of it lies on the sphere.",
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--z-min-margin-mm",
        type=float,
        default=1e-6,
        help=(
            "Remove points with z <= each profile's z_min + margin. This is a "
            "sensor-clipping heuristic; use --keep-profile-z-min to disable it."
        ),
    )
    parser.add_argument(
        "--keep-profile-z-min",
        action="store_true",
        help="Do not remove each profile's minimum-z sample/band.",
    )
    parser.add_argument(
        "--profile-spike-threshold-mm",
        type=float,
        default=0.5,
        help="Minimum neighbour-interpolation deviation required for spike removal.",
    )
    parser.add_argument(
        "--profile-spike-sigma",
        type=float,
        default=6.0,
        help="MAD multiplier for local profile spike detection.",
    )
    parser.add_argument(
        "--max-spike-run-length",
        type=int,
        default=3,
        help="Only remove isolated candidate runs no longer than this value.",
    )
    parser.add_argument(
        "--disable-profile-spike-filter",
        action="store_true",
        help="Disable acquisition-order local spike filtering.",
    )
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument(
        "--auto-floor-z-min",
        action="store_true",
        help=(
            "Detect the dominant horizontal floor Z band from per-profile "
            "low quantiles and use floor Z plus a clearance as --z-min"
        ),
    )
    parser.add_argument(
        "--floor-clearance-mm",
        type=float,
        help=(
            "Clearance above the detected floor used by --auto-floor-z-min; "
            "default is 0.8 * --radius-mm"
        ),
    )
    parser.add_argument(
        "--floor-profile-quantile",
        type=float,
        default=0.10,
        help="Low-Z quantile contributed by each profile for floor detection",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("runs/real/sphere_fit_both_result.json"),
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=1.5)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--field-of-view", type=float, default=38.0)
    parser.add_argument(
        "--stage",
        choices=("all", "original", "filtered", "sphere"),
        default="all",
        help="Open all three views sequentially or only one selected stage.",
    )
    parser.add_argument(
        "--save-render-dir",
        type=Path,
        help="Optionally save the three paper-style PNGs with OffscreenRenderer.",
    )
    parser.add_argument("--render-width", type=int, default=1800)
    parser.add_argument("--render-height", type=int, default=1350)
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args()


def print_fit(label: str, result: SphereFitResult) -> None:
    center = result.center
    all_stats = _residual_stats(result.residuals_mm)
    print(f"[{label}]")
    print(
        "  center [mm]: "
        f"[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]"
    )
    print(f"  radius [mm]: {result.radius_mm:.6f}")
    print(
        f"  robust inliers: {result.inlier_count}/{result.total_count} "
        f"({100.0 * result.inlier_count / max(result.total_count, 1):.1f}%)"
    )
    print(f"  inlier RMS radial error [mm]: {result.rms_mm:.6f}")
    print(f"  inlier MAE radial error [mm]: {result.mae_mm:.6f}")
    print(f"  inlier P95 abs error [mm]: {result.p95_abs_mm:.6f}")
    print(f"  all-precleaned RMS [mm]: {all_stats['rms_radial_error_mm']:.6f}")
    print(f"  all-precleaned P95 [mm]: {all_stats['p95_abs_radial_error_mm']:.6f}")
    print(f"  iterations: {result.iterations}, converged={result.converged}")


def main() -> None:
    args = parse_args()
    if args.radius_mm <= 0:
        raise ValueError("--radius-mm must be positive")
    if args.inlier_threshold_mm <= 0 or args.ransac_threshold_mm <= 0:
        raise ValueError("inlier thresholds must be positive")
    if args.auto_floor_z_min and args.z_min is not None:
        raise ValueError("use either --auto-floor-z-min or --z-min, not both")
    if args.floor_clearance_mm is not None and args.floor_clearance_mm < 0:
        raise ValueError("--floor-clearance-mm must be non-negative")
    if not 0.0 < args.floor_profile_quantile < 0.5:
        raise ValueError("--floor-profile-quantile must be in (0, 0.5)")

    loaded_original = load_saved_points(args.input, args.source)
    floor_detection: FloorZDetection | None = None
    if args.auto_floor_z_min:
        floor_detection = detect_floor_z_min(
            loaded_original,
            known_radius_mm=args.radius_mm,
            profile_quantile=args.floor_profile_quantile,
            clearance_mm=args.floor_clearance_mm,
        )
        args.z_min = floor_detection.z_min_mm
    loaded_cropped = apply_crop_to_groups(loaded_original, args)

    if args.keep_profile_z_min:
        loaded_z_clean = loaded_cropped
        group_stats = [
            {
                "name": name,
                "before": int(len(group)),
                "z_min_removed": 0,
                "after_z_min": int(len(group)),
                "kept": True,
            }
            for name, group in zip(loaded_cropped.group_names, loaded_cropped.groups)
        ]
        z_min_removed = 0
    else:
        loaded_z_clean, group_stats, z_min_removed = remove_profile_z_min(
            loaded_cropped,
            margin_mm=args.z_min_margin_mm,
            min_points_per_group=args.min_points_per_group,
        )

    if args.disable_profile_spike_filter:
        precleaned = loaded_z_clean
        profile_spike_removed = 0
        stat_map = {str(row["name"]): row for row in group_stats}
        for name, group in zip(precleaned.group_names, precleaned.groups):
            row = stat_map.get(name)
            if row is not None:
                row["profile_spike_removed"] = 0
                row["after_profile_spike"] = int(len(group))
    else:
        precleaned, profile_spike_removed = remove_profile_spikes(
            loaded_z_clean,
            absolute_threshold_mm=args.profile_spike_threshold_mm,
            sigma=args.profile_spike_sigma,
            max_run_length=args.max_spike_run_length,
            min_points_per_group=args.min_points_per_group,
            group_stats=group_stats,
        )

    # Both models are fit once on the same precleaned data. We do not delete
    # free-fit outliers and then refit the free model, which would make the
    # radius and residual evaluation artificially optimistic.
    free_result = fit_free_radius_sphere(
        precleaned.merged_points,
        expected_radius_mm=args.radius_mm,
        ransac_iterations=args.ransac_iterations,
        ransac_threshold_mm=args.ransac_threshold_mm,
        candidate_radius_tolerance_mm=args.radius_tolerance_mm,
        max_iterations=args.max_iterations,
        tolerance_mm=args.tolerance_mm,
        huber_delta_mm=args.huber_delta_mm,
        final_inlier_threshold_mm=args.inlier_threshold_mm,
        min_inliers=args.min_inliers,
        seed=args.seed,
    )
    fixed_result = fit_fixed_radius_sphere(
        precleaned.merged_points,
        radius_mm=args.radius_mm,
        initial_center=free_result.center,
        max_iterations=args.max_iterations,
        tolerance_mm=args.tolerance_mm,
        huber_delta_mm=args.huber_delta_mm,
        final_inlier_threshold_mm=args.inlier_threshold_mm,
        min_inliers=args.min_inliers,
    )

    # The filtered visualization uses the known-radius model because the target
    # radius is known. This mask is not reused to refit either reported model.
    filtered = apply_merged_mask_to_groups(
        precleaned,
        fixed_result.inlier_mask,
        min_points_per_group=args.min_points_per_group,
        min_profile_inlier_ratio=args.min_profile_inlier_ratio,
        group_stats=group_stats,
    )
    sphere_outlier_removed = int(
        len(precleaned.merged_points) - len(filtered.merged_points)
    )

    preprocessing = PreprocessStats(
        input_points=len(loaded_original.merged_points),
        after_crop_points=len(loaded_cropped.merged_points),
        z_min_removed_points=z_min_removed,
        profile_spike_removed_points=profile_spike_removed,
        sphere_outlier_removed_points=sphere_outlier_removed,
        final_points=len(filtered.merged_points),
        group_stats=group_stats,
    )
    settings = {
        "outlier_filtering_model": "fixed-radius residual mask for visualization only",
        "inlier_threshold_mm": args.inlier_threshold_mm,
        "profile_spike_threshold_mm": args.profile_spike_threshold_mm,
        "profile_spike_sigma": args.profile_spike_sigma,
        "max_spike_run_length": args.max_spike_run_length,
        "z_min_removal_enabled": not args.keep_profile_z_min,
        "z_min_margin_mm": args.z_min_margin_mm,
        "min_profile_inlier_ratio": args.min_profile_inlier_ratio,
        "z_min_mm": args.z_min,
        "auto_floor_z_min": args.auto_floor_z_min,
        "floor_detection": (
            None
            if floor_detection is None
            else {
                "floor_z_mm": floor_detection.floor_z_mm,
                "effective_z_min_mm": floor_detection.z_min_mm,
                "clearance_mm": floor_detection.clearance_mm,
                "profile_quantile": floor_detection.profile_quantile,
                "candidate_profile_count": floor_detection.candidate_profile_count,
                "inlier_profile_count": floor_detection.inlier_profile_count,
                "robust_sigma_mm": floor_detection.robust_sigma_mm,
            }
        ),
    }
    save_results(
        args.result_json,
        args.input,
        args.radius_mm,
        free_result,
        fixed_result,
        preprocessing,
        settings,
    )

    print(f"input: {args.input}")
    if floor_detection is not None:
        print(
            "auto floor Z [mm]: "
            f"{floor_detection.floor_z_mm:.6f} "
            f"(robust sigma={floor_detection.robust_sigma_mm:.6f}, "
            f"profiles={floor_detection.inlier_profile_count}/"
            f"{floor_detection.candidate_profile_count})"
        )
        print(
            "automatic z-min [mm]: "
            f"{floor_detection.z_min_mm:.6f} "
            f"(clearance={floor_detection.clearance_mm:.6f})"
        )
    print(f"original profile groups: {len(loaded_cropped.groups)}")
    print(f"filtered profile groups: {len(filtered.groups)}")
    print(f"input points: {preprocessing.input_points}")
    print(f"after crop: {preprocessing.after_crop_points}")
    print(f"z-min removed: {preprocessing.z_min_removed_points}")
    print(f"local profile spikes removed: {preprocessing.profile_spike_removed_points}")
    print(f"known-sphere outliers hidden: {preprocessing.sphere_outlier_removed_points}")
    print(f"final visualization points: {preprocessing.final_points}")
    print_fit("free radius", free_result)
    print(f"  radius error vs known [mm]: {free_result.radius_mm - args.radius_mm:+.6f}")
    print(
        "  diameter error vs known [mm]: "
        f"{2.0 * (free_result.radius_mm - args.radius_mm):+.6f}"
    )
    print_fit("fixed radius", fixed_result)
    print(
        "center difference free vs fixed [mm]: "
        f"{np.linalg.norm(free_result.center - fixed_result.center):.6f}"
    )
    print(f"result JSON: {args.result_json}")

    if args.no_gui and args.save_render_dir is None:
        return

    scenes = build_visualization_scenes(
        loaded_cropped,
        filtered,
        known_radius_mm=args.radius_mm,
        free_result=free_result,
        fixed_result=fixed_result,
        point_size=args.point_size,
        line_width=args.line_width,
    )
    camera = _camera_from_points(
        loaded_cropped.merged_points,
        center_hint=fixed_result.center,
        radius_hint=args.radius_mm,
    )

    if args.save_render_dir is not None:
        save_open3d_scenes(
            scenes,
            camera,
            output_dir=args.save_render_dir,
            width=args.render_width,
            height=args.render_height,
            field_of_view=args.field_of_view,
        )

    if not args.no_gui:
        show_open3d_scenes(
            scenes,
            camera,
            width=args.window_width,
            height=args.window_height,
            field_of_view=args.field_of_view,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
