"""
Example:
PYTHONPATH=. python real_laser_handeye/calibrate_only/with_ransac_calibrate_only.py \
  --dataset-dir runs/real/dataset \
  --initial-transform real_laser_handeye/no_initial.json \
  --output runs/real/T_tcp_sensor_calibrated_no__init_value.csv \
  --open-interactive-3d

ransac option
PYTHONPATH=. python real_laser_handeye/calibrate_only/with_ransac_calibrate_only.py \
  --dataset-dir runs/real/dataset \
  --initial-transform real_laser_handeye/no_initial.json \
  --output runs/real/no_initial/T_tcp_sensor_calibrated_no_initial.csv \
  --ransac-reject-policy skip \
  --ransac-threshold-mm 0.15 \
  --ransac-max-iterations 1000 \
  --ransac-min-inliers 20 \
  --ransac-min-inlier-ratio 0.65 \
  --max-ransac-skip-ratio 0.3 \
  --middle-iteration 3 \
  --open-interactive-3d

PYTHONPATH=. python real_laser_handeye/calibrate_only/with_ransac_calibrate_only.py \
  --dataset-dir runs/real/dataset \
  --initial-transform real_laser_handeye/initial_T_tcp_sensor.json \
  --output runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv \
  --ransac-reject-policy skip \
  --ransac-threshold-mm 0.15 \
  --ransac-max-iterations 1000 \
  --ransac-min-inliers 20 \
  --ransac-min-inlier-ratio 0.65 \
  --max-ransac-skip-ratio 0.3 \
  --middle-iteration 3 \
  --open-interactive-3d
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np

from robust_laser_handeye.laser_handeye.calibration import calibrate_single_plane
from robust_laser_handeye.laser_handeye.data import LaserScan


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a copy of a 4x4 rigid-body transform."""
    transform = np.asarray(value, dtype=float)

    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")

    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")

    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")

    return transform.copy()


def atomic_json(path: Path, value: dict) -> None:
    """Write JSON atomically so a failed run does not leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_matrix(path: Path, value: np.ndarray) -> None:
    """Write the calibrated 4x4 matrix atomically as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    try:
        with temporary.open("w", encoding="utf-8") as stream:
            np.savetxt(stream, value, delimiter=",", fmt="%.12g")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_transform(path: Path) -> np.ndarray:
    """Load T_tcp_sensor from JSON, CSV, or whitespace-separated text."""
    if not path.exists():
        raise FileNotFoundError(f"initial transform not found: {path}")

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if "T_tcp_sensor" not in value:
                raise KeyError(f"{path} does not contain 'T_tcp_sensor'")
            value = value["T_tcp_sensor"]
        return validate_transform(np.asarray(value, dtype=float), str(path))

    try:
        value = np.loadtxt(path, delimiter=",")
    except ValueError:
        value = np.loadtxt(path)

    return validate_transform(value, str(path))


def fit_line_tls_xz(points_xz: np.ndarray) -> tuple[float, float, float]:
    """
    Fit a 2-D line in the sensor profile plane using total least squares.

    The normalized implicit line is
        a * x + b * z + c = 0,
    where sqrt(a**2 + b**2) = 1. Therefore, the absolute algebraic
    residual is also the orthogonal point-to-line distance in millimetres.
    """
    points_xz = np.asarray(points_xz, dtype=float)
    if points_xz.ndim != 2 or points_xz.shape[1] != 2:
        raise ValueError("points_xz must have shape (N, 2)")
    if len(points_xz) < 2:
        raise ValueError("at least two points are required to fit a line")
    if not np.all(np.isfinite(points_xz)):
        raise ValueError("line-fitting points must be finite")

    centroid = np.mean(points_xz, axis=0)
    centered = points_xz - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] < 1e-12:
        raise ValueError("profile points are degenerate for line fitting")

    direction = np.asarray(vh[0], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("could not recover a valid profile-line normal")
    normal /= normal_norm

    # Resolve the arbitrary line-sign ambiguity for repeatable diagnostics.
    dominant_index = int(np.argmax(np.abs(normal)))
    if normal[dominant_index] < 0:
        normal = -normal

    c = -float(normal @ centroid)
    return float(normal[0]), float(normal[1]), c


def profile_line_distances_xz(
    points_s: np.ndarray,
    line_model: tuple[float, float, float],
) -> np.ndarray:
    """Return orthogonal distances from sensor-frame points to an x-z line."""
    points_s = np.asarray(points_s, dtype=float)
    if points_s.ndim != 2 or points_s.shape[1] != 3:
        raise ValueError("points_s must have shape (N, 3)")

    a, b, c = line_model
    return np.abs(a * points_s[:, 0] + b * points_s[:, 2] + c)


def raw_profile_line_statistics(points_s: np.ndarray) -> dict[str, float | int | list[float]]:
    """Fit one TLS line to every finite point and summarize pre-RANSAC residuals."""
    points_s = np.asarray(points_s, dtype=float)
    if points_s.ndim != 2 or points_s.shape[1] != 3:
        raise ValueError("points_s must have shape (N, 3)")
    if len(points_s) < 2:
        raise ValueError("at least two points are required for line statistics")

    model = fit_line_tls_xz(points_s[:, (0, 2)])
    signed = (
        model[0] * points_s[:, 0]
        + model[1] * points_s[:, 2]
        + model[2]
    )
    absolute = np.abs(signed)
    median_signed = float(np.median(signed))
    mad_sigma = float(
        1.4826 * np.median(np.abs(signed - median_signed))
    )

    return {
        "count": int(len(points_s)),
        "line_model_ax_bz_c": [float(value) for value in model],
        "mean_signed_mm": float(np.mean(signed)),
        "rms_mm": float(np.sqrt(np.mean(signed**2))),
        "mae_mm": float(np.mean(absolute)),
        "median_abs_mm": float(np.median(absolute)),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "p99_abs_mm": float(np.percentile(absolute, 99.0)),
        "max_abs_mm": float(np.max(absolute)),
        "mad_sigma_mm": mad_sigma,
    }


def ransac_profile_line_xz(
    points_s: np.ndarray,
    threshold_mm: float,
    max_iterations: int,
    min_inliers: int,
    min_inlier_ratio: float,
    seed: int,
    refine_iterations: int = 3,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    """
    Detect the dominant planar-target profile line in sensor x-z coordinates.

    A physical target plane and the laser sheet intersect in a line. Because
    the scanner already reports points in its own laser plane, that line is
    represented directly in the sensor-frame x-z profile. This test therefore
    does not depend on the unknown hand-eye transform.
    """
    points_s = np.asarray(points_s, dtype=float)
    if points_s.ndim != 2 or points_s.shape[1] != 3:
        raise ValueError("points_s must have shape (N, 3)")
    if len(points_s) < 2:
        raise ValueError("at least two points are required for line RANSAC")
    if threshold_mm <= 0:
        raise ValueError("RANSAC threshold must be positive")
    if max_iterations < 1:
        raise ValueError("RANSAC max_iterations must be at least 1")
    if min_inliers < 2:
        raise ValueError("RANSAC min_inliers must be at least 2")
    if not (0.0 < min_inlier_ratio <= 1.0):
        raise ValueError("RANSAC min_inlier_ratio must be in (0, 1]")
    if refine_iterations < 0:
        raise ValueError("RANSAC refine_iterations must be non-negative")

    points_xz = points_s[:, (0, 2)]
    rng = np.random.default_rng(seed)

    best_mask: np.ndarray | None = None
    best_model: tuple[float, float, float] | None = None
    best_count = -1
    best_median = float("inf")
    best_rms = float("inf")

    for _ in range(max_iterations):
        first_index, second_index = rng.choice(len(points_s), size=2, replace=False)
        first = points_xz[first_index]
        second = points_xz[second_index]
        direction = second - first
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            continue

        normal = np.array([-direction[1], direction[0]], dtype=float) / direction_norm
        c = -float(normal @ first)
        distances = np.abs(points_xz @ normal + c)
        mask = distances <= threshold_mm
        count = int(np.count_nonzero(mask))
        if count < 2:
            continue

        inlier_distances = distances[mask]
        median = float(np.median(inlier_distances))
        rms = float(np.sqrt(np.mean(inlier_distances**2)))

        # Consensus size is primary. Residual quality breaks ties.
        score_is_better = (
            count > best_count
            or (count == best_count and median < best_median)
            or (
                count == best_count
                and np.isclose(median, best_median)
                and rms < best_rms
            )
        )
        if score_is_better:
            best_count = count
            best_median = median
            best_rms = rms
            best_mask = mask
            best_model = (float(normal[0]), float(normal[1]), c)

    if best_mask is None or best_model is None:
        raise RuntimeError("line RANSAC could not generate a valid model")

    # Refit with total least squares, then update the consensus mask.
    mask = best_mask.copy()
    model = best_model
    for _ in range(refine_iterations):
        if int(np.count_nonzero(mask)) < 2:
            break
        model = fit_line_tls_xz(points_xz[mask])
        distances = profile_line_distances_xz(points_s, model)
        updated_mask = distances <= threshold_mm
        if np.array_equal(updated_mask, mask):
            mask = updated_mask
            break
        if int(np.count_nonzero(updated_mask)) < 2:
            break
        mask = updated_mask

    # One final TLS fit on the final consensus set.
    model = fit_line_tls_xz(points_xz[mask])
    distances = profile_line_distances_xz(points_s, model)
    mask = distances <= threshold_mm
    inlier_count = int(np.count_nonzero(mask))
    inlier_ratio = float(inlier_count / len(points_s))

    required_count = max(min_inliers, int(math.ceil(min_inlier_ratio * len(points_s))))
    accepted = bool(inlier_count >= required_count)

    inlier_distances = distances[mask]
    diagnostics: dict[str, float | int | bool | str | list[float]] = {
        "accepted": accepted,
        "required_inlier_count": int(required_count),
        "reject_reason": (
            ""
            if accepted
            else (
                "inlier consensus below requirement: "
                f"{inlier_count}/{len(points_s)} ({inlier_ratio:.1%}) < "
                f"required {required_count}/{len(points_s)}; "
                f"threshold={threshold_mm:g} mm"
            )
        ),
        "raw_count": int(len(points_s)),
        "inlier_count": inlier_count,
        "outlier_count": int(len(points_s) - inlier_count),
        "inlier_ratio": inlier_ratio,
        "threshold_mm": float(threshold_mm),
        "line_model_ax_bz_c": [float(value) for value in model],
        "inlier_rms_mm": float(np.sqrt(np.mean(inlier_distances**2))),
        "inlier_mae_mm": float(np.mean(inlier_distances)),
        "inlier_p95_mm": float(np.percentile(inlier_distances, 95.0)),
        "inlier_max_mm": float(np.max(inlier_distances)),
        "all_median_distance_mm": float(np.median(distances)),
        "all_max_distance_mm": float(np.max(distances)),
    }
    return mask, diagnostics


def save_profile_ransac_diagnostics(path: Path, rows: list[dict]) -> None:
    """Save one summary row per profile scan, including rejected scans."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_id",
        "capture_path",
        "status",
        "reject_reason",
        "raw_count",
        "finite_count",
        "raw_line_rms_mm",
        "raw_line_mae_mm",
        "raw_line_median_abs_mm",
        "raw_line_p95_abs_mm",
        "raw_line_mad_sigma_mm",
        "raw_line_max_abs_mm",
        "inlier_count",
        "required_inlier_count",
        "outlier_count",
        "inlier_ratio",
        "threshold_mm",
        "line_a_x",
        "line_b_z",
        "line_c",
        "inlier_rms_mm",
        "inlier_mae_mm",
        "inlier_p95_mm",
        "inlier_max_mm",
        "all_median_distance_mm",
        "all_max_distance_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_scans(
    dataset_dir: Path,
    *,
    use_profile_ransac: bool,
    ransac_threshold_mm: float,
    ransac_max_iterations: int,
    ransac_min_inliers: int,
    ransac_min_inlier_ratio: float,
    ransac_seed: int,
    ransac_refine_iterations: int,
    ransac_reject_policy: str,
) -> tuple[list[LaserScan], list[dict]]:
    """
    Load capture_*.npz files and optionally remove profile-line outliers.

    RANSAC is applied independently to every profile in sensor x-z coordinates.
    With reject_policy='skip', a rejected scan is recorded and omitted instead
    of terminating the entire calibration run.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    capture_paths = sorted(dataset_dir.glob("capture_*.npz"))
    if not capture_paths:
        raise RuntimeError(f"no capture_*.npz files found in {dataset_dir}")

    if ransac_reject_policy not in {"skip", "error"}:
        raise ValueError("ransac_reject_policy must be 'skip' or 'error'")

    scans: list[LaserScan] = []
    diagnostics_rows: list[dict] = []

    for scan_id, path in enumerate(capture_paths, start=1):
        with np.load(path, allow_pickle=False) as pair:
            if "T_base_tcp" not in pair or "points_s" not in pair:
                raise KeyError(
                    f"{path} must contain both 'T_base_tcp' and 'points_s'"
                )

            pose = validate_transform(pair["T_base_tcp"], str(path))
            raw_points = np.asarray(pair["points_s"], dtype=float)

        if raw_points.ndim != 2 or raw_points.shape[1] != 3:
            raise ValueError(f"{path}: points_s must have shape (N, 3)")

        finite_mask = np.all(np.isfinite(raw_points), axis=1)
        points = raw_points[finite_mask]
        if len(points) == 0:
            raise ValueError(f"{path}: no finite profile points")

        raw_line_stats = raw_profile_line_statistics(points)
        print(
            f"raw line scan {scan_id:03d}: N={raw_line_stats['count']}, "
            f"RMS={raw_line_stats['rms_mm']:.6g} mm, "
            f"MAE={raw_line_stats['mae_mm']:.6g} mm, "
            f"median|r|={raw_line_stats['median_abs_mm']:.6g} mm, "
            f"P95|r|={raw_line_stats['p95_abs_mm']:.6g} mm, "
            f"MAD-sigma={raw_line_stats['mad_sigma_mm']:.6g} mm, "
            f"max|r|={raw_line_stats['max_abs_mm']:.6g} mm"
        )

        base_row = {
            "scan_id": scan_id,
            "capture_path": str(path),
            "raw_count": int(len(raw_points)),
            "finite_count": int(len(points)),
            "raw_line_rms_mm": float(raw_line_stats["rms_mm"]),
            "raw_line_mae_mm": float(raw_line_stats["mae_mm"]),
            "raw_line_median_abs_mm": float(raw_line_stats["median_abs_mm"]),
            "raw_line_p95_abs_mm": float(raw_line_stats["p95_abs_mm"]),
            "raw_line_mad_sigma_mm": float(raw_line_stats["mad_sigma_mm"]),
            "raw_line_max_abs_mm": float(raw_line_stats["max_abs_mm"]),
        }

        if use_profile_ransac:
            try:
                inlier_mask, ransac_diag = ransac_profile_line_xz(
                    points,
                    threshold_mm=ransac_threshold_mm,
                    max_iterations=ransac_max_iterations,
                    min_inliers=ransac_min_inliers,
                    min_inlier_ratio=ransac_min_inlier_ratio,
                    seed=ransac_seed + scan_id,
                    refine_iterations=ransac_refine_iterations,
                )
            except Exception as exc:
                reason = f"RANSAC execution failed: {exc}"
                rejected_row = {
                    **base_row,
                    "status": "rejected",
                    "reject_reason": reason,
                    "inlier_count": 0,
                    "required_inlier_count": max(
                        ransac_min_inliers,
                        int(math.ceil(ransac_min_inlier_ratio * len(points))),
                    ),
                    "outlier_count": int(len(points)),
                    "inlier_ratio": 0.0,
                    "threshold_mm": float(ransac_threshold_mm),
                    "line_a_x": float("nan"),
                    "line_b_z": float("nan"),
                    "line_c": float("nan"),
                    "inlier_rms_mm": float("nan"),
                    "inlier_mae_mm": float("nan"),
                    "inlier_p95_mm": float("nan"),
                    "inlier_max_mm": float("nan"),
                    "all_median_distance_mm": float("nan"),
                    "all_max_distance_mm": float("nan"),
                }
                diagnostics_rows.append(rejected_row)
                print(f"profile RANSAC scan {scan_id:03d}: REJECTED - {reason}")
                if ransac_reject_policy == "error":
                    raise RuntimeError(f"{path}: {reason}") from exc
                continue

            line_a, line_b, line_c = ransac_diag["line_model_ax_bz_c"]
            accepted = bool(ransac_diag["accepted"])
            row = {
                **base_row,
                "status": "accepted" if accepted else "rejected",
                "reject_reason": str(ransac_diag["reject_reason"]),
                "inlier_count": int(ransac_diag["inlier_count"]),
                "required_inlier_count": int(
                    ransac_diag["required_inlier_count"]
                ),
                "outlier_count": int(ransac_diag["outlier_count"]),
                "inlier_ratio": float(ransac_diag["inlier_ratio"]),
                "threshold_mm": float(ransac_diag["threshold_mm"]),
                "line_a_x": float(line_a),
                "line_b_z": float(line_b),
                "line_c": float(line_c),
                "inlier_rms_mm": float(ransac_diag["inlier_rms_mm"]),
                "inlier_mae_mm": float(ransac_diag["inlier_mae_mm"]),
                "inlier_p95_mm": float(ransac_diag["inlier_p95_mm"]),
                "inlier_max_mm": float(ransac_diag["inlier_max_mm"]),
                "all_median_distance_mm": float(
                    ransac_diag["all_median_distance_mm"]
                ),
                "all_max_distance_mm": float(
                    ransac_diag["all_max_distance_mm"]
                ),
            }
            diagnostics_rows.append(row)

            if not accepted:
                print(
                    f"profile RANSAC scan {scan_id:03d}: REJECTED - "
                    f"{row['inlier_count']}/{row['finite_count']} inliers "
                    f"({100.0 * row['inlier_ratio']:.1f}%), "
                    f"required>={row['required_inlier_count']}; skipping"
                )
                if ransac_reject_policy == "error":
                    raise RuntimeError(f"{path}: {row['reject_reason']}")
                continue

            filtered_points = points[inlier_mask]
            print(
                f"profile RANSAC scan {scan_id:03d}: "
                f"{row['inlier_count']}/{row['finite_count']} inliers "
                f"({100.0 * row['inlier_ratio']:.1f}%), "
                f"RMS={row['inlier_rms_mm']:.6g} mm"
            )
        else:
            filtered_points = points
            row = {
                **base_row,
                "status": "accepted",
                "reject_reason": "",
                "inlier_count": int(len(points)),
                "required_inlier_count": 0,
                "outlier_count": 0,
                "inlier_ratio": 1.0,
                "threshold_mm": float("nan"),
                "line_a_x": float("nan"),
                "line_b_z": float("nan"),
                "line_c": float("nan"),
                "inlier_rms_mm": float("nan"),
                "inlier_mae_mm": float("nan"),
                "inlier_p95_mm": float("nan"),
                "inlier_max_mm": float("nan"),
                "all_median_distance_mm": float("nan"),
                "all_max_distance_mm": float("nan"),
            }
            diagnostics_rows.append(row)

        scans.append(
            LaserScan(
                pose,
                filtered_points,
                plane_id=0,
                scan_id=scan_id,
            )
        )

    return scans, diagnostics_rows


def save_plane_rms_plot(history: list[float], path: Path) -> bool:
    """Save the plane RMS convergence graph. Returns False if unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; RMS plot was not saved")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(len(history), dtype=int)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(iterations, history, marker="o", markersize=3)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Plane RMS [mm]")
    axis.set_title("Single-plane calibration convergence")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(math.acos(float(cosine)))
    if angle < 1e-12:
        return np.zeros(3, dtype=float)

    if abs(angle - math.pi) < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        if axis[0] < 1e-6:
            axis[0] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * max(axis[1], 1e-6))
        if axis[1] < 1e-6:
            axis[1] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * max(axis[2], 1e-6))
        if axis[2] < 1e-6:
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * max(axis[0], 1e-6))
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3, dtype=float)
        return angle * axis / norm

    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    return (0.5 * angle / math.sin(angle)) * vee


def so3_exp(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float).reshape(3)
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3, dtype=float) + skew(omega)

    axis = omega / angle
    K = skew(axis)
    return (
        np.eye(3, dtype=float)
        + math.sin(angle) * K
        + (1.0 - math.cos(angle)) * (K @ K)
    )


def midpoint_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the halfway transform between first and second."""
    first = validate_transform(first, "first transform")
    second = validate_transform(second, "second transform")

    result = np.eye(4, dtype=float)
    result[:3, 3] = 0.5 * (first[:3, 3] + second[:3, 3])

    relative = first[:3, :3].T @ second[:3, :3]
    omega = so3_log(relative)
    result[:3, :3] = first[:3, :3] @ so3_exp(0.5 * omega)
    return result


def get_scan_pose(scan: LaserScan) -> np.ndarray:
    """Return the robot pose stored in a LaserScan across supported field names."""
    for attribute in ("T_base_ef", "T_base_tcp", "pose"):
        value = getattr(scan, attribute, None)
        if value is not None:
            return validate_transform(
                value,
                f"scan {getattr(scan, 'scan_id', '?')} {attribute}",
            )

    metadata = getattr(scan, "meta", None)
    if isinstance(metadata, dict):
        for key in ("T_base_ef", "T_base_tcp", "pose"):
            if key in metadata:
                return validate_transform(
                    metadata[key],
                    f"scan {getattr(scan, 'scan_id', '?')} meta[{key}]",
                )

    raise AttributeError(
        "LaserScan has no supported robot-pose field. "
        "Expected one of: T_base_ef, T_base_tcp, pose."
    )


def get_scan_points(scan: LaserScan) -> np.ndarray:
    """Return sensor-frame profile points across supported field names."""
    for attribute in ("points_s", "points"):
        value = getattr(scan, attribute, None)
        if value is not None:
            points = np.asarray(value, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"scan {getattr(scan, 'scan_id', '?')} {attribute} "
                    "must have shape (N, 3)"
                )
            return points

    raise AttributeError(
        "LaserScan has no supported point field. "
        "Expected one of: points_s, points."
    )


def transform_scan_points_to_base(
    scans: list[LaserScan],
    T_tcp_sensor: np.ndarray,
    max_points_per_scan: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Convert each scan's sensor-frame points to the base frame.

    For each point p_s:
        p_base = T_base_tcp @ T_tcp_sensor @ p_s

    In this project LaserScan normally stores:
        T_base_ef : robot TCP/end-effector pose
        points_s  : profile points in the sensor frame
    """
    transformed_clouds: list[np.ndarray] = []
    sensor_origins: list[np.ndarray] = []
    T_tcp_sensor = validate_transform(T_tcp_sensor, "T_tcp_sensor")

    for scan in scans:
        T_base_tcp = get_scan_pose(scan)
        T_base_sensor = T_base_tcp @ T_tcp_sensor

        points = get_scan_points(scan)
        if max_points_per_scan > 0 and len(points) > max_points_per_scan:
            indices = np.linspace(
                0,
                len(points) - 1,
                num=max_points_per_scan,
                dtype=int,
            )
            points = points[indices]

        ones = np.ones((len(points), 1), dtype=float)
        points_h = np.hstack([points, ones])
        points_base = (T_base_sensor @ points_h.T).T[:, :3]
        transformed_clouds.append(points_base)
        sensor_origins.append(T_base_sensor[:3, 3].copy())

    return transformed_clouds, np.asarray(sensor_origins, dtype=float)



def fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit one plane to 3D points using PCA.

    Plane convention:
        normal^T point = offset
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) < 3:
        raise ValueError("at least three points are required to fit a plane")
    if not np.all(np.isfinite(points)):
        raise ValueError("plane-fitting points must be finite")

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    normal = np.asarray(vh[-1], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("could not recover a valid plane normal")
    normal /= normal_norm

    # Resolve the arbitrary PCA sign for repeatable output.
    dominant_index = int(np.argmax(np.abs(normal)))
    if normal[dominant_index] < 0:
        normal = -normal

    offset = float(normal @ centroid)
    return normal, offset, centroid


def point_to_plane_statistics(residuals: np.ndarray) -> dict[str, float | int]:
    """Return descriptive statistics for signed point-to-plane residuals."""
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        raise ValueError("no finite point-to-plane residuals")

    absolute = np.abs(residuals)
    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    centered = residuals - mean

    if std > 1e-15:
        normalized = centered / std
        skewness = float(np.mean(normalized**3))
        excess_kurtosis = float(np.mean(normalized**4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    stats: dict[str, float | int] = {
        "count": int(len(residuals)),
        "mean_mm": mean,
        "std_mm": std,
        "rms_mm": float(np.sqrt(np.mean(residuals**2))),
        "mae_mm": float(np.mean(absolute)),
        "median_mm": float(np.median(residuals)),
        "median_abs_mm": float(np.median(absolute)),
        "p95_abs_mm": float(np.percentile(absolute, 95.0)),
        "p99_abs_mm": float(np.percentile(absolute, 99.0)),
        "max_abs_mm": float(np.max(absolute)),
        "min_signed_mm": float(np.min(residuals)),
        "max_signed_mm": float(np.max(residuals)),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }

    for threshold_mm in (0.05, 0.1, 0.2, 0.5, 1.0):
        key = f"within_{str(threshold_mm).replace('.', 'p')}_mm_percent"
        stats[key] = float(100.0 * np.mean(absolute <= threshold_mm))

    return stats


def save_point_to_plane_histogram(
    residuals: np.ndarray,
    stats: dict[str, float | int],
    path: Path,
    bins: int,
    range_mm: float | None,
    show_plot: bool,
    central_percentile: float,
) -> bool:
    """
    Save a clean signed point-to-plane residual histogram.

    Plotting behavior:
      - statistics still use every residual,
      - the histogram automatically excludes only extreme tails,
      - a manual symmetric range overrides percentile clipping.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; residual histogram was not saved")
        return False

    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        raise ValueError("no residuals are available for the histogram")

    if bins < 1:
        raise ValueError("--residual-hist-bins must be at least 1")

    if not (50.0 < central_percentile <= 100.0):
        raise ValueError(
            "--residual-hist-percentile must be in the interval (50, 100]"
        )

    absolute = np.abs(residuals)

    if range_mm is not None:
        plot_limit = float(range_mm)
        if plot_limit <= 0:
            raise ValueError("--residual-hist-range-mm must be positive")
        range_source = "manual"
    else:
        plot_limit = float(np.percentile(absolute, central_percentile))
        range_source = f"abs residual P{central_percentile:g}"

    if not np.isfinite(plot_limit) or plot_limit <= 0:
        plot_limit = max(float(np.max(absolute)), 1e-6)

    plotted = residuals[absolute <= plot_limit]
    excluded_count = int(len(residuals) - len(plotted))
    excluded_percent = 100.0 * excluded_count / len(residuals)

    path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.3))

        axis.hist(
            plotted,
            bins=int(bins),
            range=(-plot_limit, plot_limit),
            alpha=0.85,
            edgecolor="none",
        )

        axis.axvline(
            0.0,
            linewidth=1.1,
            linestyle="--",
            label="zero",
        )

        rms = float(stats["rms_mm"])
        p95 = float(stats["p95_abs_mm"])
        count = int(stats["count"])

        axis.set_title("Final point-to-plane residual")
        axis.text(
            0.5,
            1.01,
            (
                f"N={count:,}  RMS={rms:.4g} mm  "
                f"P95_abs={p95:.4g} mm  "
                f"hidden={excluded_percent:.3g}%"
            ),
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

        axis.set_xlabel("Signed residual [mm]")
        axis.set_ylabel("Count")
        axis.set_xlim(-plot_limit, plot_limit)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(loc="upper right", frameon=False)

        axis.text(
            0.01,
            0.02,
            f"x range: +/-{plot_limit:.4g} mm ({range_source})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )

        figure.tight_layout()
        figure.savefig(path, dpi=200)

        if show_plot:
            plt.show()
        plt.close(figure)

    return True


def analyze_final_point_to_plane(
    scans: list[LaserScan],
    T_final: np.ndarray,
    histogram_path: Path,
    residual_csv_path: Path,
    per_scan_csv_path: Path,
    bins: int,
    histogram_range_mm: float | None,
    histogram_percentile: float,
    show_plot: bool,
) -> dict:
    """
    Reconstruct all final points, fit one final plane, and analyze residuals.

    This is a self-fit residual analysis: the plane is estimated from the same
    final reconstructed point set. It measures coplanarity and outliers, not
    absolute error relative to an independently surveyed plane.
    """
    point_clouds, _ = transform_scan_points_to_base(
        scans,
        T_final,
        max_points_per_scan=0,
    )
    if not point_clouds:
        raise RuntimeError("no reconstructed point clouds are available")

    all_points = np.vstack(point_clouds)
    normal, offset, centroid = fit_plane_pca(all_points)

    residual_sets = [
        np.asarray(cloud @ normal - offset, dtype=float)
        for cloud in point_clouds
    ]
    all_residuals = np.concatenate(residual_sets)
    overall_stats = point_to_plane_statistics(all_residuals)

    residual_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with residual_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scan_id",
                "point_index",
                "base_x_mm",
                "base_y_mm",
                "base_z_mm",
                "signed_residual_mm",
                "absolute_residual_mm",
            ]
        )
        for scan, cloud, residuals in zip(scans, point_clouds, residual_sets):
            scan_id = int(getattr(scan, "scan_id", -1))
            for point_index, (point, residual) in enumerate(
                zip(cloud, residuals)
            ):
                writer.writerow(
                    [
                        scan_id,
                        point_index,
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        float(residual),
                        float(abs(residual)),
                    ]
                )

    per_scan_rows: list[dict] = []
    per_scan_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with per_scan_csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "scan_id",
            "point_count",
            "mean_mm",
            "std_mm",
            "rms_mm",
            "mae_mm",
            "median_abs_mm",
            "p95_abs_mm",
            "p99_abs_mm",
            "max_abs_mm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for scan, residuals in zip(scans, residual_sets):
            scan_stats = point_to_plane_statistics(residuals)
            row = {
                "scan_id": int(getattr(scan, "scan_id", -1)),
                "point_count": int(scan_stats["count"]),
                "mean_mm": float(scan_stats["mean_mm"]),
                "std_mm": float(scan_stats["std_mm"]),
                "rms_mm": float(scan_stats["rms_mm"]),
                "mae_mm": float(scan_stats["mae_mm"]),
                "median_abs_mm": float(scan_stats["median_abs_mm"]),
                "p95_abs_mm": float(scan_stats["p95_abs_mm"]),
                "p99_abs_mm": float(scan_stats["p99_abs_mm"]),
                "max_abs_mm": float(scan_stats["max_abs_mm"]),
            }
            writer.writerow(row)
            per_scan_rows.append(row)

    histogram_saved = save_point_to_plane_histogram(
        residuals=all_residuals,
        stats=overall_stats,
        path=histogram_path,
        bins=bins,
        range_mm=histogram_range_mm,
        show_plot=show_plot,
        central_percentile=histogram_percentile,
    )

    worst_scan_by_rms = max(per_scan_rows, key=lambda row: row["rms_mm"])
    worst_scan_by_max = max(per_scan_rows, key=lambda row: row["max_abs_mm"])

    return {
        "definition": "signed_residual_mm = plane_normal^T point_base - plane_offset_mm",
        "interpretation": (
            "Self-fitted final-plane residual. Measures final coplanarity and "
            "outliers; it is not absolute error to an independently measured plane."
        ),
        "plane_normal_base": normal.tolist(),
        "plane_offset_mm": float(offset),
        "plane_centroid_base_mm": centroid.tolist(),
        "overall": overall_stats,
        "worst_scan_by_rms": worst_scan_by_rms,
        "worst_scan_by_max_abs": worst_scan_by_max,
        "histogram_path": str(histogram_path) if histogram_saved else None,
        "histogram_percentile": float(histogram_percentile),
        "histogram_manual_range_mm": (
            None if histogram_range_mm is None else float(histogram_range_mm)
        ),
        "residual_csv_path": str(residual_csv_path),
        "per_scan_csv_path": str(per_scan_csv_path),
    }


def collect_stage_bounds(
    initial_clouds: list[np.ndarray],
    mid_clouds: list[np.ndarray],
    final_clouds: list[np.ndarray],
    initial_origins: np.ndarray,
    mid_origins: np.ndarray,
    final_origins: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute common axis bounds using profile points only."""
    del initial_origins, mid_origins, final_origins

    all_points = [
        cloud
        for cloud in (*initial_clouds, *mid_clouds, *final_clouds)
        if np.size(cloud) > 0
    ]
    if not all_points:
        raise ValueError("no profile points are available for 3D visualization")

    stacked = np.vstack(all_points)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius < 1e-6:
        radius = 1.0
    return center, radius


def set_equal_3d_axes(axis, center: np.ndarray, radius: float) -> None:
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def draw_stage(
    axis,
    clouds: list[np.ndarray],
    origins: np.ndarray,
    title: str,
) -> None:
    """Draw reconstructed profile points only; sensor poses are intentionally hidden."""
    del origins

    for cloud in clouds:
        if len(cloud):
            axis.scatter(
                cloud[:, 0],
                cloud[:, 1],
                cloud[:, 2],
                s=2,
                alpha=0.45,
            )

    axis.set_title(title)
    axis.set_xlabel("Base X [mm]")
    axis.set_ylabel("Base Y [mm]")
    axis.set_zlabel("Base Z [mm]")
    axis.grid(True, alpha=0.3)


def save_base_point_visualization(
    scans: list[LaserScan],
    T_initial: np.ndarray,
    T_middle: np.ndarray,
    T_final: np.ndarray,
    middle_iteration: int | None,
    total_iterations: int,
    path: Path,
    max_points_per_scan: int = 600,
    show_plot: bool = False,
) -> bool:
    """
    Save a 1x3 3D visualization:
      1) initial T_tcp_sensor (iteration 0)
      2) actual recorded middle iteration, when T_history is available
      3) final T_tcp_sensor
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; base-frame 3D plot was not saved")
        return False

    T_mid = validate_transform(T_middle, "middle T_tcp_sensor")

    initial_clouds, initial_origins = transform_scan_points_to_base(
        scans, T_initial, max_points_per_scan
    )
    mid_clouds, mid_origins = transform_scan_points_to_base(
        scans, T_mid, max_points_per_scan
    )
    final_clouds, final_origins = transform_scan_points_to_base(
        scans, T_final, max_points_per_scan
    )

    center, radius = collect_stage_bounds(
        initial_clouds,
        mid_clouds,
        final_clouds,
        initial_origins,
        mid_origins,
        final_origins,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(18.0, 5.8))

    axis_initial = figure.add_subplot(1, 3, 1, projection="3d")
    draw_stage(axis_initial, initial_clouds, initial_origins, "Initial (iter 0) -> Base")
    set_equal_3d_axes(axis_initial, center, radius)

    axis_mid = figure.add_subplot(1, 3, 2, projection="3d")
    middle_title = (
        f"Actual middle (iter {middle_iteration}/{total_iterations}) -> Base"
        if middle_iteration is not None
        else "50% interpolation (no T_history) -> Base"
    )
    draw_stage(axis_mid, mid_clouds, mid_origins, middle_title)
    set_equal_3d_axes(axis_mid, center, radius)

    axis_final = figure.add_subplot(1, 3, 3, projection="3d")
    draw_stage(
        axis_final,
        final_clouds,
        final_origins,
        f"Final (iter {total_iterations}) -> Base",
    )
    set_equal_3d_axes(axis_final, center, radius)

    figure.suptitle("Sensor profile points transformed into the base frame", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    if show_plot:
        plt.show()
    plt.close(figure)
    return True



def save_interactive_base_point_visualization(
    scans: list[LaserScan],
    T_initial: np.ndarray,
    T_middle: np.ndarray,
    T_final: np.ndarray,
    middle_iteration: int | None,
    total_iterations: int,
    path: Path,
    max_points_per_scan: int = 600,
) -> bool:
    """
    Save an interactive Plotly HTML with three 3D scenes:
      1) initial T_tcp_sensor (iteration 0)
      2) actual recorded middle iteration, when T_history is available
      3) final T_tcp_sensor
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        print("warning: plotly is unavailable; interactive 3D HTML was not saved")
        return False

    T_mid = validate_transform(T_middle, "middle T_tcp_sensor")

    initial_clouds, initial_origins = transform_scan_points_to_base(
        scans, T_initial, max_points_per_scan
    )
    mid_clouds, mid_origins = transform_scan_points_to_base(
        scans, T_mid, max_points_per_scan
    )
    final_clouds, final_origins = transform_scan_points_to_base(
        scans, T_final, max_points_per_scan
    )

    center, radius = collect_stage_bounds(
        initial_clouds,
        mid_clouds,
        final_clouds,
        initial_origins,
        mid_origins,
        final_origins,
    )

    def add_stage(fig, col, clouds, origins):
        for cloud in clouds:
            if len(cloud):
                fig.add_trace(
                    go.Scatter3d(
                        x=cloud[:, 0],
                        y=cloud[:, 1],
                        z=cloud[:, 2],
                        mode="markers",
                        marker=dict(size=2, opacity=0.55),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
        del origins

    middle_title = (
        f"Actual middle iteration: {middle_iteration} / {total_iterations}"
        if middle_iteration is not None
        else "50% interpolation (T_history unavailable)"
    )
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Initial: iter 0",
            middle_title,
            f"Final: iter {total_iterations}",
        ),
        horizontal_spacing=0.02,
    )

    add_stage(fig, 1, initial_clouds, initial_origins)
    add_stage(fig, 2, mid_clouds, mid_origins)
    add_stage(fig, 3, final_clouds, final_origins)

    scene_common = dict(
        xaxis=dict(title="Base X [mm]", range=[center[0] - radius, center[0] + radius]),
        yaxis=dict(title="Base Y [mm]", range=[center[1] - radius, center[1] + radius]),
        zaxis=dict(title="Base Z [mm]", range=[center[2] - radius, center[2] + radius]),
        aspectmode="cube",
    )

    fig.update_layout(
        title="Sensor profile points transformed into the base frame (interactive 3D)",
        scene=scene_common,
        scene2=scene_common,
        scene3=scene_common,
        margin=dict(l=0, r=0, b=0, t=60),
        height=650,
        width=1800,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return True



def _fit_profiled_plane_residuals(
    scans: list[LaserScan],
    T_tcp_sensor: np.ndarray,
    reference_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return self-fitted plane residuals for one hand-eye transform.

    The plane is re-estimated for every transform.  This profiles out the
    unknown single-plane nuisance parameters and makes the numerical Jacobian
    consistent with the alternating single-plane calibration objective.
    """
    point_clouds, _ = transform_scan_points_to_base(
        scans,
        T_tcp_sensor,
        max_points_per_scan=0,
    )
    if not point_clouds:
        raise RuntimeError("no reconstructed points for uncertainty analysis")

    points = np.vstack(point_clouds)
    normal, offset, centroid = fit_plane_pca(points)
    if reference_normal is not None and float(normal @ reference_normal) < 0.0:
        normal = -normal
        offset = -offset

    residuals = np.asarray(points @ normal - offset, dtype=float)
    return residuals, normal, float(offset), centroid


def _perturb_handeye_scaled(
    T_tcp_sensor: np.ndarray,
    parameter_index: int,
    scaled_step_mm: float,
    characteristic_length_mm: float,
) -> np.ndarray:
    """Perturb one scaled hand-eye coordinate.

    The dimensionless/mixed-unit issue is handled with
        q = [L*phi_x, L*phi_y, L*phi_z, t_x, t_y, t_z],
    where every q component has millimetre-equivalent units.
    Rotation perturbations are right-multiplied in the sensor/local frame.
    """
    if abs(scaled_step_mm) <= 0.0:
        raise ValueError("scaled_step_mm must be non-zero")
    if characteristic_length_mm <= 0.0:
        raise ValueError("characteristic_length_mm must be positive")
    if parameter_index < 0 or parameter_index >= 6:
        raise IndexError("parameter_index must be in [0, 5]")

    perturbed = validate_transform(T_tcp_sensor, "T_tcp_sensor").copy()
    if parameter_index < 3:
        omega = np.zeros(3, dtype=float)
        omega[parameter_index] = scaled_step_mm / characteristic_length_mm
        perturbed[:3, :3] = perturbed[:3, :3] @ so3_exp(omega)
    else:
        perturbed[parameter_index - 3, 3] += scaled_step_mm
    return perturbed


def compute_handeye_observability_uncertainty(
    scans: list[LaserScan],
    T_final: np.ndarray,
    *,
    characteristic_length_mm: float,
    finite_difference_step_mm: float,
    svd_relative_tolerance: float = 1e-10,
) -> dict:
    """Compute profiled numerical Jacobian, SVD and local covariance.

    This is a local, first-order uncertainty analysis around the final result.
    It does not require ground truth.  The unknown plane is refitted for each
    finite-difference perturbation, so the resulting 6-column Jacobian measures
    the information remaining for the hand-eye parameters after eliminating
    the plane nuisance parameters.
    """
    if characteristic_length_mm <= 0.0:
        raise ValueError("characteristic_length_mm must be positive")
    if finite_difference_step_mm <= 0.0:
        raise ValueError("finite_difference_step_mm must be positive")
    if not (0.0 < svd_relative_tolerance < 1.0):
        raise ValueError("svd_relative_tolerance must be in (0, 1)")

    T_final = validate_transform(T_final, "T_final")
    residuals, normal, offset, centroid = _fit_profiled_plane_residuals(
        scans,
        T_final,
    )
    point_count = int(len(residuals))
    if point_count <= 6:
        raise RuntimeError("too few residuals for a 6-DoF uncertainty analysis")

    jacobian = np.empty((point_count, 6), dtype=float)
    for column in range(6):
        plus = _perturb_handeye_scaled(
            T_final,
            column,
            finite_difference_step_mm,
            characteristic_length_mm,
        )
        minus = _perturb_handeye_scaled(
            T_final,
            column,
            -finite_difference_step_mm,
            characteristic_length_mm,
        )
        residual_plus, _, _, _ = _fit_profiled_plane_residuals(
            scans,
            plus,
            reference_normal=normal,
        )
        residual_minus, _, _, _ = _fit_profiled_plane_residuals(
            scans,
            minus,
            reference_normal=normal,
        )
        jacobian[:, column] = (
            residual_plus - residual_minus
        ) / (2.0 * finite_difference_step_mm)

    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
    sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
    rank_threshold = svd_relative_tolerance * max(sigma_max, 1.0)
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    sigma_min = float(singular_values[-1]) if len(singular_values) else 0.0
    condition_number = (
        float(sigma_max / sigma_min)
        if sigma_min > rank_threshold
        else float("inf")
    )

    weak_mode_scaled = np.asarray(vh[-1], dtype=float)
    weak_mode_scaled /= max(float(np.linalg.norm(weak_mode_scaled)), 1e-15)

    # The self-fitted plane has three nuisance DoF (two for normal direction,
    # one for offset).  Use a conservative residual-variance denominator.
    dof = max(point_count - rank - 3, 1)
    residual_mean = float(np.mean(residuals))
    residual_variance_mm2 = float(
        np.sum((residuals - residual_mean) ** 2) / dof
    )
    residual_sigma_mm = float(math.sqrt(max(residual_variance_mm2, 0.0)))

    information_scaled = (
        jacobian.T @ jacobian / max(residual_variance_mm2, 1e-24)
    )
    covariance_scaled = residual_variance_mm2 * np.linalg.pinv(
        jacobian.T @ jacobian,
        rcond=svd_relative_tolerance,
    )

    # theta = [phi(rad), t(mm)] = D q, q=[L*phi(mm-equiv), t(mm)].
    D = np.diag(
        [
            1.0 / characteristic_length_mm,
            1.0 / characteristic_length_mm,
            1.0 / characteristic_length_mm,
            1.0,
            1.0,
            1.0,
        ]
    )
    covariance_physical = D @ covariance_scaled @ D.T
    std_physical = np.sqrt(np.maximum(np.diag(covariance_physical), 0.0))
    std_rotation_deg = np.degrees(std_physical[:3])
    std_translation_mm = std_physical[3:]

    denom = np.sqrt(
        np.maximum(np.diag(covariance_physical), 0.0)[:, None]
        * np.maximum(np.diag(covariance_physical), 0.0)[None, :]
    )
    correlation = np.divide(
        covariance_physical,
        denom,
        out=np.zeros_like(covariance_physical),
        where=denom > 0.0,
    )

    labels_scaled = ["L*rx", "L*ry", "L*rz", "tx", "ty", "tz"]
    labels_physical = ["rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm"]

    return {
        "definition": (
            "Profiled numerical Jacobian of self-fitted point-to-plane residuals "
            "with respect to q=[L*rotation(rad), translation(mm)]."
        ),
        "local_analysis_only": True,
        "parameterization": {
            "scaled_labels": labels_scaled,
            "physical_labels": labels_physical,
            "characteristic_length_mm": float(characteristic_length_mm),
            "finite_difference_step_mm": float(finite_difference_step_mm),
            "rotation_update": "right multiplication: R <- R exp([dphi]_x)",
            "translation_update": "additive in T_tcp_sensor translation coordinates",
        },
        "point_count": point_count,
        "effective_dof": int(dof),
        "plane_normal_base": normal.tolist(),
        "plane_offset_mm": float(offset),
        "plane_centroid_base_mm": centroid.tolist(),
        "residual_sigma_mm": residual_sigma_mm,
        "residual_variance_mm2": residual_variance_mm2,
        "rank": rank,
        "rank_threshold": float(rank_threshold),
        "singular_values": singular_values.tolist(),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition_number,
        "weak_mode_scaled": {
            label: float(value)
            for label, value in zip(labels_scaled, weak_mode_scaled)
        },
        "jacobian_scaled": jacobian.tolist(),
        "information_matrix_scaled": information_scaled.tolist(),
        "covariance_scaled_mm2": covariance_scaled.tolist(),
        "covariance_physical": covariance_physical.tolist(),
        "correlation_physical": correlation.tolist(),
        "std_rotation_deg": {
            label: float(value)
            for label, value in zip(("rx", "ry", "rz"), std_rotation_deg)
        },
        "std_translation_mm": {
            label: float(value)
            for label, value in zip(("tx", "ty", "tz"), std_translation_mm)
        },
    }


def save_observability_uncertainty_outputs(
    analysis: dict,
    output_base: Path,
) -> dict[str, str | None]:
    """Save numerical observability/uncertainty tables and plots."""
    output_base.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_base.with_suffix(".uncertainty.json")
    atomic_json(json_path, analysis)

    singular_csv = output_base.with_suffix(".jacobian_singular_values.csv")
    with singular_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "singular_value"])
        for index, value in enumerate(analysis["singular_values"], start=1):
            writer.writerow([index, float(value)])

    weak_csv = output_base.with_suffix(".weak_mode.csv")
    with weak_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter_scaled", "component"])
        for label, value in analysis["weak_mode_scaled"].items():
            writer.writerow([label, float(value)])

    covariance_csv = output_base.with_suffix(".parameter_covariance.csv")
    covariance = np.asarray(analysis["covariance_physical"], dtype=float)
    labels = list(analysis["parameterization"]["physical_labels"])
    with covariance_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", *labels])
        for label, row in zip(labels, covariance):
            writer.writerow([label, *[float(value) for value in row]])

    correlation_csv = output_base.with_suffix(".parameter_correlation.csv")
    correlation = np.asarray(analysis["correlation_physical"], dtype=float)
    with correlation_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", *labels])
        for label, row in zip(labels, correlation):
            writer.writerow([label, *[float(value) for value in row]])

    # Save an immediately readable per-parameter uncertainty summary.
    # These values describe local first-order uncertainty around the final
    # estimate; they are not absolute errors relative to ground truth.
    parameter_std_csv = output_base.with_suffix(".parameter_std.csv")
    std_rotation_deg = analysis["std_rotation_deg"]
    std_translation_mm = analysis["std_translation_mm"]
    std_rows = [
        ("rx", "deg", float(std_rotation_deg["rx"])),
        ("ry", "deg", float(std_rotation_deg["ry"])),
        ("rz", "deg", float(std_rotation_deg["rz"])),
        ("tx", "mm", float(std_translation_mm["tx"])),
        ("ty", "mm", float(std_translation_mm["ty"])),
        ("tz", "mm", float(std_translation_mm["tz"])),
    ]
    with parameter_std_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "parameter",
                "unit",
                "std_1sigma",
                "approx_95pct_half_width",
                "approx_95pct_lower_relative",
                "approx_95pct_upper_relative",
            ]
        )
        for parameter, unit, std_value in std_rows:
            half_width_95 = 1.96 * std_value
            writer.writerow(
                [
                    parameter,
                    unit,
                    std_value,
                    half_width_95,
                    -half_width_95,
                    half_width_95,
                ]
            )

    spectrum_plot: str | None = None
    weak_plot: str | None = None
    try:
        import matplotlib.pyplot as plt

        spectrum_path = output_base.with_suffix(".jacobian_spectrum.png")
        singular_values = np.asarray(analysis["singular_values"], dtype=float)
        figure, axis = plt.subplots(figsize=(6.6, 4.2))
        axis.semilogy(
            np.arange(1, len(singular_values) + 1),
            np.maximum(singular_values, 1e-18),
            marker="o",
        )
        axis.set_xlabel("Singular-value index")
        axis.set_ylabel("Singular value")
        axis.set_title("Profiled hand-eye Jacobian spectrum")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(spectrum_path, dpi=190)
        plt.close(figure)
        spectrum_plot = str(spectrum_path)

        weak_path = output_base.with_suffix(".weak_mode.png")
        weak = analysis["weak_mode_scaled"]
        figure, axis = plt.subplots(figsize=(6.8, 4.2))
        axis.bar(list(weak.keys()), [float(value) for value in weak.values()])
        axis.axhline(0.0, linewidth=1.0)
        axis.set_ylabel("Normalized component")
        axis.set_title("Weakest right singular vector (scaled coordinates)")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(weak_path, dpi=190)
        plt.close(figure)
        weak_plot = str(weak_path)
    except ImportError:
        print("warning: matplotlib unavailable; uncertainty plots were not saved")

    return {
        "json": str(json_path),
        "singular_values_csv": str(singular_csv),
        "weak_mode_csv": str(weak_csv),
        "covariance_csv": str(covariance_csv),
        "correlation_csv": str(correlation_csv),
        "parameter_std_csv": str(parameter_std_csv),
        "spectrum_plot": spectrum_plot,
        "weak_mode_plot": weak_plot,
    }

def run_calibration(args: argparse.Namespace) -> np.ndarray:
    if args.residual_hist_bins < 1:
        raise ValueError("--residual-hist-bins must be at least 1")
    if args.residual_hist_range_mm is not None and args.residual_hist_range_mm <= 0:
        raise ValueError("--residual-hist-range-mm must be positive")
    if not (50.0 < args.residual_hist_percentile <= 100.0):
        raise ValueError(
            "--residual-hist-percentile must be in the interval (50, 100]"
        )

    if args.uncertainty_characteristic_length_mm <= 0:
        raise ValueError("--uncertainty-characteristic-length-mm must be positive")
    if args.uncertainty_step_mm <= 0:
        raise ValueError("--uncertainty-step-mm must be positive")
    if not (0.0 < args.uncertainty_svd_rtol < 1.0):
        raise ValueError("--uncertainty-svd-rtol must be in (0, 1)")

    if args.ransac_threshold_mm <= 0:
        raise ValueError("--ransac-threshold-mm must be positive")
    if args.ransac_max_iterations < 1:
        raise ValueError("--ransac-max-iterations must be at least 1")
    if args.ransac_min_inliers < 2:
        raise ValueError("--ransac-min-inliers must be at least 2")
    if not (0.0 < args.ransac_min_inlier_ratio <= 1.0):
        raise ValueError("--ransac-min-inlier-ratio must be in (0, 1]")
    if args.ransac_refine_iterations < 0:
        raise ValueError("--ransac-refine-iterations must be non-negative")
    if not (0.0 <= args.max_ransac_skip_ratio <= 1.0):
        raise ValueError("--max-ransac-skip-ratio must be in [0, 1]")

    scans, profile_ransac_rows = load_scans(
        args.dataset_dir,
        use_profile_ransac=not args.disable_profile_ransac,
        ransac_threshold_mm=args.ransac_threshold_mm,
        ransac_max_iterations=args.ransac_max_iterations,
        ransac_min_inliers=args.ransac_min_inliers,
        ransac_min_inlier_ratio=args.ransac_min_inlier_ratio,
        ransac_seed=args.ransac_seed,
        ransac_refine_iterations=args.ransac_refine_iterations,
        ransac_reject_policy=args.ransac_reject_policy,
    )

    total_scan_count = len(profile_ransac_rows)
    rejected_rows = [
        row for row in profile_ransac_rows if row.get("status") == "rejected"
    ]
    rejected_ids = [int(row["scan_id"]) for row in rejected_rows]
    rejected_count = len(rejected_rows)
    skip_ratio = (
        float(rejected_count / total_scan_count) if total_scan_count > 0 else 0.0
    )

    print("\nProfile RANSAC dataset summary:")
    print(f"  total scans    : {total_scan_count}")
    print(f"  accepted scans : {len(scans)}")
    print(f"  rejected scans : {rejected_count}")
    print(f"  rejected ids   : {rejected_ids if rejected_ids else 'none'}")
    print(f"  skip ratio     : {100.0 * skip_ratio:.1f}%")

    # Save accepted/rejected diagnostics before any dataset-level safety abort.
    profile_ransac_csv_path = args.output.with_suffix(".profile_ransac.csv")
    save_profile_ransac_diagnostics(profile_ransac_csv_path, profile_ransac_rows)
    print(f"saved RANSAC diagnostics: {profile_ransac_csv_path}")

    if len(scans) < args.min_scans:
        raise RuntimeError(
            "too few accepted scans after profile RANSAC: "
            f"accepted={len(scans)}, required>={args.min_scans}, "
            f"rejected={rejected_count}/{total_scan_count}"
        )

    if (
        not args.disable_profile_ransac
        and skip_ratio > args.max_ransac_skip_ratio
    ):
        raise RuntimeError(
            "too many scans were rejected by profile RANSAC: "
            f"{rejected_count}/{total_scan_count} ({skip_ratio:.1%}) > "
            f"allowed {args.max_ransac_skip_ratio:.1%}. "
            "Inspect or reacquire the rejected scans."
        )

    initial = load_transform(args.initial_transform)

    print(f"dataset       : {args.dataset_dir}")
    print(f"scan count    : {len(scans)} accepted / {total_scan_count} total")
    print(f"rejected ids  : {rejected_ids if rejected_ids else 'none'}")
    print(f"point count   : {sum(scan.num_points for scan in scans)}")
    print(f"initial matrix: {args.initial_transform}")

    print(f"profile RANSAC: {'disabled' if args.disable_profile_ransac else 'enabled'}")

    print("\nInitial T_tcp_sensor =")
    print(np.array2string(initial, precision=12, suppress_small=True))

    result = calibrate_single_plane(
        scans,
        T_init=initial,
        max_iter=args.max_iter,
        tol=args.tol,
        plane_offset_mode=args.plane_offset_mode,
        max_translation_offset_condition=args.max_condition,
    )

    # Solver convergence diagnostics.
    delta_history = [
        float(value)
        for value in getattr(result, "delta_history", [])
    ]

    print("\nSolver convergence diagnostics:")
    print(f"  solver tol  : {args.tol:.12g}")
    print(f"  iterations  : {result.iterations}")
    print(f"  converged   : {result.converged}")

    if delta_history:
        print(f"  final delta : {delta_history[-1]:.12g}")
        print("  last 10 deltas:")
        for index, value in enumerate(
            delta_history[-10:],
            start=max(1, len(delta_history) - 9),
        ):
            print(f"    {index:4d}: {value:.12g}")
    else:
        print(
            "  delta history: unavailable "
            "(the calibration result does not expose delta_history)"
        )

    plane_rms_history = [float(value) for value in result.plane_rms_history]
    if not plane_rms_history:
        raise RuntimeError("calibration returned an empty plane RMS history")

    final_rms = plane_rms_history[-1]
    accepted = bool(
        result.converged
        and np.isfinite(final_rms)
        and final_rms <= args.max_final_plane_rms_mm
    )

    total_iterations = int(result.iterations)
    T_history = [
        validate_transform(np.asarray(value, dtype=float), f"T_history[{index}]")
        for index, value in enumerate(getattr(result, "T_history", []))
    ]

    requested_middle_iteration = args.middle_iteration

    if requested_middle_iteration is None:
        if T_history:
            middle_iteration = max(1, (len(T_history) + 1) // 2)
            T_mid = T_history[middle_iteration - 1]
            middle_source = "automatic_actual_iteration"
        else:
            middle_iteration = None
            T_mid = midpoint_transform(initial, result.T_ef_s)
            middle_source = "interpolated_fallback"
    else:
        middle_iteration = int(requested_middle_iteration)

        if middle_iteration < 0 or middle_iteration > total_iterations:
            raise ValueError(
                "--middle-iteration must be between 0 and "
                f"{total_iterations}; received {middle_iteration}"
            )

        if middle_iteration == 0:
            T_mid = initial.copy()
            middle_source = "selected_initial"
        elif middle_iteration == total_iterations:
            T_mid = validate_transform(result.T_ef_s, "final T_tcp_sensor")
            middle_source = "selected_final"
        elif middle_iteration <= len(T_history):
            # T_history[0] is the result after iteration 1.
            T_mid = T_history[middle_iteration - 1]
            middle_source = "selected_actual_iteration"
        else:
            raise RuntimeError(
                f"iteration {middle_iteration} was requested, but only "
                f"{len(T_history)} transforms are available in result.T_history"
            )

    print(
        "\n3D selected state: "
        + (
            f"iteration {middle_iteration} / {total_iterations} "
            f"({middle_source})"
            if middle_iteration is not None
            else "50% transform interpolation (T_history unavailable)"
        )
    )

    plot_path = args.output.with_suffix(".plane_rms.png")
    plot_saved = save_plane_rms_plot(plane_rms_history, plot_path)

    base_points_plot_path = args.output.with_suffix(".base_points_3d.png")
    base_points_plot_saved = save_base_point_visualization(
        scans=scans,
        T_initial=initial,
        T_middle=T_mid,
        T_final=result.T_ef_s,
        middle_iteration=middle_iteration,
        total_iterations=total_iterations,
        path=base_points_plot_path,
        max_points_per_scan=args.max_plot_points_per_scan,
        show_plot=args.show_plots,
    )

    interactive_3d_path = args.output.with_suffix(".base_points_3d.html")
    interactive_3d_saved = save_interactive_base_point_visualization(
        scans=scans,
        T_initial=initial,
        T_middle=T_mid,
        T_final=result.T_ef_s,
        middle_iteration=middle_iteration,
        total_iterations=total_iterations,
        path=interactive_3d_path,
        max_points_per_scan=args.max_plot_points_per_scan,
    )

    final_residual_hist_path = args.output.with_suffix(
        ".final_point_to_plane_hist.png"
    )
    final_residual_csv_path = args.output.with_suffix(
        ".final_point_to_plane_residuals.csv"
    )
    final_residual_per_scan_path = args.output.with_suffix(
        ".final_point_to_plane_per_scan.csv"
    )
    final_point_to_plane = analyze_final_point_to_plane(
        scans=scans,
        T_final=result.T_ef_s,
        histogram_path=final_residual_hist_path,
        residual_csv_path=final_residual_csv_path,
        per_scan_csv_path=final_residual_per_scan_path,
        bins=args.residual_hist_bins,
        histogram_range_mm=args.residual_hist_range_mm,
        histogram_percentile=args.residual_hist_percentile,
        show_plot=args.show_plots,
    )

    uncertainty_analysis = None
    uncertainty_outputs = None
    if not args.disable_uncertainty_analysis:
        uncertainty_analysis = compute_handeye_observability_uncertainty(
            scans=scans,
            T_final=result.T_ef_s,
            characteristic_length_mm=args.uncertainty_characteristic_length_mm,
            finite_difference_step_mm=args.uncertainty_step_mm,
            svd_relative_tolerance=args.uncertainty_svd_rtol,
        )
        uncertainty_outputs = save_observability_uncertainty_outputs(
            uncertainty_analysis,
            args.output,
        )

        print("\nFinal local observability / uncertainty analysis:")
        print(f"  rank        : {uncertainty_analysis['rank']} / 6")
        print(f"  sigma min   : {uncertainty_analysis['sigma_min']:.9g}")
        print(f"  condition   : {uncertainty_analysis['condition_number']:.9g}")
        print(f"  residual sd : {uncertainty_analysis['residual_sigma_mm']:.9g} mm")
        print("  weakest mode [L*rx, L*ry, L*rz, tx, ty, tz]:")
        print(
            "    "
            + ", ".join(
                f"{key}={value:+.5f}"
                for key, value in uncertainty_analysis[
                    'weak_mode_scaled'
                ].items()
            )
        )
        print(
            "  rotation std [deg]: "
            + ", ".join(
                f"{key}={value:.6g}"
                for key, value in uncertainty_analysis[
                    'std_rotation_deg'
                ].items()
            )
        )
        print(
            "  translation std [mm]: "
            + ", ".join(
                f"{key}={value:.6g}"
                for key, value in uncertainty_analysis[
                    'std_translation_mm'
                ].items()
            )
        )
        print(
            f"  saved parameter std CSV: "
            f"{uncertainty_outputs['parameter_std_csv']}"
        )
        print(
            f"  saved covariance CSV   : "
            f"{uncertainty_outputs['covariance_csv']}"
        )
        print(
            f"  saved correlation CSV  : "
            f"{uncertainty_outputs['correlation_csv']}"
        )

    diagnostics_path = args.output.with_suffix(".diagnostics.json")
    diagnostics = {
        "dataset_dir": str(args.dataset_dir),
        "initial_transform": str(args.initial_transform),
        "output_transform": str(args.output),
        "scan_count": len(scans),
        "total_capture_count": total_scan_count,
        "accepted_scan_count": len(scans),
        "rejected_scan_count": rejected_count,
        "rejected_scan_ids": rejected_ids,
        "profile_ransac_skip_ratio": skip_ratio,
        "profile_ransac_reject_policy": args.ransac_reject_policy,
        "max_ransac_skip_ratio": float(args.max_ransac_skip_ratio),
        "point_count": int(sum(scan.num_points for scan in scans)),
        "profile_ransac": {
            "enabled": not args.disable_profile_ransac,
            "threshold_mm": float(args.ransac_threshold_mm),
            "max_iterations": int(args.ransac_max_iterations),
            "min_inliers": int(args.ransac_min_inliers),
            "min_inlier_ratio": float(args.ransac_min_inlier_ratio),
            "seed": int(args.ransac_seed),
            "refine_iterations": int(args.ransac_refine_iterations),
            "diagnostics_csv": str(profile_ransac_csv_path),
            "per_scan": profile_ransac_rows,
        },
        "converged": bool(result.converged),
        "accepted": accepted,
        "iterations": int(result.iterations),
        "middle_iteration_requested": requested_middle_iteration,
        "middle_iteration": middle_iteration,
        "middle_transform_source": middle_source,
        "T_history_count": len(T_history),
        "plane_offset_mode": args.plane_offset_mode,
        "initial_plane_rms_mm": plane_rms_history[0],
        "final_plane_rms_mm": final_rms,
        "max_final_plane_rms_mm": float(args.max_final_plane_rms_mm),
        "plane_rms_history_mm": plane_rms_history,
        "plane_rms_plot": str(plot_path) if plot_saved else None,
        "base_points_3d_plot": str(base_points_plot_path) if base_points_plot_saved else None,
        "base_points_3d_html": str(interactive_3d_path) if interactive_3d_saved else None,
        "final_point_to_plane": final_point_to_plane,
        "observability_uncertainty": uncertainty_analysis,
        "observability_uncertainty_outputs": uncertainty_outputs,
        "rank_history": [int(value) for value in result.rank_history],
        "condition_history": [
            float(value) if np.isfinite(value) else None
            for value in result.cond_history
        ],
        "initial_T_tcp_sensor": initial.tolist(),
        "mid_T_tcp_sensor": T_mid.tolist(),
        "T_tcp_sensor": result.T_ef_s.tolist(),
    }
    atomic_json(diagnostics_path, diagnostics)
    print(f"saved diagnostics: {diagnostics_path}")

    if plot_saved:
        print(f"saved RMS plot   : {plot_path}")
    if base_points_plot_saved:
        print(f"saved 3D plot    : {base_points_plot_path}")
    if interactive_3d_saved:
        print(f"saved 3D html    : {interactive_3d_path}")

    residual_stats = final_point_to_plane["overall"]
    print("\nFinal self-fitted point-to-plane residual:")
    print(f"  mean      : {float(residual_stats['mean_mm']):.9g} mm")
    print(f"  std       : {float(residual_stats['std_mm']):.9g} mm")
    print(f"  RMS       : {float(residual_stats['rms_mm']):.9g} mm")
    print(f"  MAE       : {float(residual_stats['mae_mm']):.9g} mm")
    print(f"  median |r|: {float(residual_stats['median_abs_mm']):.9g} mm")
    print(f"  P95 |r|   : {float(residual_stats['p95_abs_mm']):.9g} mm")
    print(f"  P99 |r|   : {float(residual_stats['p99_abs_mm']):.9g} mm")
    print(f"  max |r|   : {float(residual_stats['max_abs_mm']):.9g} mm")
    print(
        "  worst RMS scan: "
        f"{final_point_to_plane['worst_scan_by_rms']['scan_id']} "
        f"({final_point_to_plane['worst_scan_by_rms']['rms_mm']:.9g} mm)"
    )
    print(f"saved residual hist : {final_residual_hist_path}")
    print(f"saved residual CSV  : {final_residual_csv_path}")
    print(f"saved per-scan CSV  : {final_residual_per_scan_path}")

    if interactive_3d_saved and args.open_interactive_3d:
        try:
            import webbrowser
            webbrowser.open(interactive_3d_path.resolve().as_uri())
        except Exception as exc:
            print(f"warning: could not open browser automatically: {exc}")

    if not accepted and not args.save_rejected:
        raise RuntimeError(
            "calibration rejected: "
            f"converged={result.converged}, "
            f"final RMS={final_rms:.6f} mm; "
            f"see {diagnostics_path}"
        )

    atomic_matrix(args.output, result.T_ef_s)
    print(f"saved transform  : {args.output}")
    print("\nFinal T_tcp_sensor =")
    print(np.array2string(result.T_ef_s, precision=12, suppress_small=True))

    print("\nFinal - Initial =")
    print(np.array2string(result.T_ef_s - initial, precision=12, suppress_small=False))
    translation_change = np.linalg.norm(result.T_ef_s[:3, 3] - initial[:3, 3])
    print(f"\nTranslation change norm [mm]: {translation_change:.12g}")

    if not accepted:
        print(
            "warning: result was saved because --save-rejected was set, "
            "but it did not pass the acceptance criterion"
        )

    return result.T_ef_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-plane 2D laser hand-eye calibration from previously "
            "saved capture_*.npz files. No robot or laser connection is used."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("runs/real/dataset"),
        help="Directory containing capture_*.npz files",
    )
    parser.add_argument(
        "--initial-transform",
        type=Path,
        default=Path("real_laser_handeye/initial_T_tcp_sensor.json"),
        help="Initial T_tcp_sensor in JSON, CSV, or text format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/real/T_tcp_sensor_calibrated.csv"),
        help="Output CSV path for the calibrated 4x4 matrix",
    )
    parser.add_argument("--min-scans", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--max-condition", type=float, default=1e6)
    parser.add_argument("--max-final-plane-rms-mm", type=float, default=2.0)
    parser.add_argument(
        "--disable-profile-ransac",
        action="store_true",
        help=(
            "Disable per-scan line RANSAC in sensor x-z coordinates. "
            "By default, only line inliers are passed to calibration."
        ),
    )
    parser.add_argument(
        "--ransac-threshold-mm",
        type=float,
        default=0.2,
        help=(
            "Maximum orthogonal x-z distance from the fitted profile line "
            "for an inlier. Start near 3-5 times the nominal profile noise."
        ),
    )
    parser.add_argument(
        "--ransac-max-iterations",
        type=int,
        default=1000,
        help="Maximum random line hypotheses evaluated per scan",
    )
    parser.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=20,
        help="Minimum absolute inlier count required for every scan",
    )
    parser.add_argument(
        "--ransac-min-inlier-ratio",
        type=float,
        default=0.5,
        help="Minimum inlier ratio required for every scan",
    )
    parser.add_argument(
        "--ransac-reject-policy",
        choices=("skip", "error"),
        default="skip",
        help=(
            "Behavior when one profile fails RANSAC. 'skip' records and omits "
            "that scan; 'error' terminates immediately. Default: skip."
        ),
    )
    parser.add_argument(
        "--max-ransac-skip-ratio",
        type=float,
        default=0.4,
        help=(
            "Abort after filtering when the rejected/total scan ratio exceeds "
            "this value. Default 0.4 means at most 40 percent may be skipped."
        ),
    )
    parser.add_argument(
        "--ransac-seed",
        type=int,
        default=1701,
        help="Base random seed for reproducible per-scan RANSAC",
    )
    parser.add_argument(
        "--ransac-refine-iterations",
        type=int,
        default=3,
        help="TLS refit/consensus-update passes after RANSAC",
    )
    parser.add_argument(
        "--plane-offset-mode",
        choices=("joint", "fitted"),
        default="joint",
        help="Plane-offset handling passed to calibrate_single_plane",
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save the matrix even when convergence/RMS acceptance fails",
    )
    parser.add_argument(
        "--middle-iteration",
        type=int,
        default=None,
        help=(
            "Iteration shown in the middle 3D panel. "
            "Use 0 for the initial transform. "
            "Omit this option to use the automatic halfway iteration."
        ),
    )
    parser.add_argument(
        "--max-plot-points-per-scan",
        type=int,
        default=600,
        help="Maximum number of points per scan drawn in the 3D base-frame plot",
    )
    parser.add_argument(
        "--residual-hist-bins",
        type=int,
        default=100,
        help="Number of bins in the final point-to-plane residual histogram",
    )
    parser.add_argument(
        "--residual-hist-percentile",
        type=float,
        default=99.0,
        help=(
            "Central absolute-residual percentile shown in the histogram "
            "when --residual-hist-range-mm is omitted. "
            "Default 99 excludes the most extreme 1 percent from the plot only."
        ),
    )
    parser.add_argument(
        "--residual-hist-range-mm",
        type=float,
        default=None,
        help=(
            "Optional symmetric histogram display range in millimetres. "
            "For example, 1.0 displays -1 to +1 mm. "
            "Residual CSV and statistics always include every point."
        ),
    )
    parser.add_argument(
        "--disable-uncertainty-analysis",
        action="store_true",
        help=(
            "Disable final profiled Jacobian SVD and local covariance analysis."
        ),
    )
    parser.add_argument(
        "--uncertainty-characteristic-length-mm",
        type=float,
        default=100.0,
        help=(
            "Characteristic length L used to scale rotation as L*rad so "
            "rotation and translation columns are comparable. Default: 100 mm."
        ),
    )
    parser.add_argument(
        "--uncertainty-step-mm",
        type=float,
        default=1e-3,
        help=(
            "Central finite-difference step in scaled millimetre-equivalent "
            "coordinates. Default: 1e-3 mm."
        ),
    )
    parser.add_argument(
        "--uncertainty-svd-rtol",
        type=float,
        default=1e-10,
        help="Relative SVD rank/pseudoinverse tolerance. Default: 1e-10.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show matplotlib windows in addition to saving the plots",
    )
    parser.add_argument(
        "--open-interactive-3d",
        action="store_true",
        help="Open the saved interactive 3D HTML in the default web browser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()