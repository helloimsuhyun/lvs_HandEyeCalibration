from __future__ import annotations

"""
Paired Monte Carlo validation for minimum-scan circular-pattern pose design.

Purpose
-------
Validate whether a small continuous Jacobian-optimized circular pose set can
match or outperform much larger / non-optimized pose sets in the *actual*
unknown-plane calibration solver.

Compared methods
----------------
For each requested N (e.g. 6,7,8):
  - opt_N:         continuous optimized design loaded from N_XX/best_design.csv
  - rand_cont_N:   random continuous design with the same N (one design/trial)
  - rand_disc_N:   random discrete design with the same N

Large baselines:
  - legacy81:      9 lines x d{60,90,120} x theta{30} x beta{60,90,120}
  - legacy117:     legacy81 + theta=60 scans on lines {1,2,5,6}

Pairing / common random numbers
-------------------------------
Within one trial, all methods share exactly the same:
  - ground-truth hand-eye transform
  - physical plane
  - initial hand-eye estimate
  - nominal noise seed

Different geometries cannot have literally identical noisy observations, but
resetting the same noise RNG seed for every method gives a common-random-number
pairing and removes avoidable Monte Carlo variation.

Outputs
-------
  trials.csv
  summary.csv
  paired_comparisons.csv
  translation_error_boxplot.png
  rotation_error_boxplot.png
  success_rate.png

Run from repository root, for example:

PYTHONPATH=./robust_laser_handeye python \
  robust_laser_handeye/examples/paired_mc_compare_pose_designs.py \
  --design-dir runs/circular_continuous_design \
  --n-values 6 7 8 \
  --trials 100 \
  --noise-std 0.5 \
  --output-dir runs/paired_mc_pose_validation
"""

import argparse
import csv
import inspect
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

# If placed in robust_laser_handeye/examples, make robust_laser_handeye importable
# even when the caller launches from the repository root.
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "examples":
    _PACKAGE_ROOT = _THIS_FILE.parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

from laser_handeye.calibration import calibrate_planes
from laser_handeye.initialization import InitialGuessMode, make_initial_guess
from laser_handeye.nonlinear_refinement import refine_handeye_nonlinear
from laser_handeye.patterns import circular_lines
from laser_handeye.se3 import inv_T, rot_error_deg
from laser_handeye.simulation import (
    is_reachable_simple,
    sample_random_handeye,
    sensor_pose_from_target_line,
    simulate_profile_on_plane,
)


def _call_with_supported_kwargs(func, *args, **kwargs):
    """Call a project function while filtering version-dependent kwargs.

    The repository has had multiple calibrate/refinement API revisions.  This
    keeps the benchmark compatible with the currently installed checkout while
    still using newer options when they actually exist.
    """
    signature = inspect.signature(func)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_var_kw:
        return func(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(*args, **filtered)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseSpec:
    line_id: int
    theta_deg: float
    beta_deg: float
    d_mm: float
    branch_sign: float


@dataclass
class TrialRow:
    trial: int
    method: str
    method_family: str
    N_nominal: int
    N_actual: int
    converged: bool
    calibration_success: bool
    iterations: int
    rank_last: int
    cond_last: float
    trans_err_mm: float
    rot_err_deg: float
    final_plane_rms_mm: float
    exception: str


# -----------------------------------------------------------------------------
# Geometry and design loading
# -----------------------------------------------------------------------------


def sample_random_plane_pose(
    rng: np.random.Generator,
    tilt_min_deg: float,
    tilt_max_deg: float,
    yaw_min_deg: float,
    yaw_max_deg: float,
    center_xy_range_mm: float,
    center_z_range_mm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Match the benchmark's random single-plane generation convention."""

    def signed_tilt() -> float:
        mag = float(rng.uniform(tilt_min_deg, tilt_max_deg))
        return mag if rng.random() >= 0.5 else -mag

    # Preserve the benchmark's intent of avoiding an exactly-zero yaw.
    yaw = float(rng.uniform(yaw_min_deg, yaw_max_deg))
    for _ in range(10000):
        if abs(yaw) >= tilt_min_deg:
            break
        yaw = float(rng.uniform(yaw_min_deg, yaw_max_deg))

    angles = np.array([signed_tilt(), signed_tilt(), yaw], dtype=float)
    plane_R = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    plane_t = np.array(
        [
            rng.uniform(-center_xy_range_mm, center_xy_range_mm),
            rng.uniform(-center_xy_range_mm, center_xy_range_mm),
            rng.uniform(*center_z_range_mm),
        ],
        dtype=float,
    )
    plane_n = np.asarray(plane_R[:, 2], dtype=float)
    plane_n /= np.linalg.norm(plane_n)
    plane_l = float(plane_n @ plane_t)
    if plane_l < 0.0:
        plane_n = -plane_n
        plane_l = -plane_l
        plane_R = plane_R.copy()
        plane_R[:, 0] *= -1.0
        plane_R[:, 2] *= -1.0
    return plane_R, plane_t, plane_n, plane_l


def branch_sign_for_line(line_id: int, mode: str) -> float:
    if mode == "positive":
        return 1.0
    if mode == "alternating":
        return 1.0 if int(line_id) % 2 == 0 else -1.0
    raise ValueError("branch mode must be 'alternating' or 'positive'")


def load_optimized_design(design_dir: Path, N: int) -> list[PoseSpec]:
    path = Path(design_dir) / f"N_{N:02d}" / "best_design.csv"
    if not path.exists():
        raise FileNotFoundError(f"optimized design not found: {path}")

    poses: list[PoseSpec] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append(
                PoseSpec(
                    line_id=int(row["line_id"]),
                    theta_deg=float(row["theta_deg"]),
                    beta_deg=float(row["beta_deg"]),
                    d_mm=float(row["d_mm"]),
                    branch_sign=float(row.get("branch_sign", 1.0)),
                )
            )
    if len(poses) != N:
        raise ValueError(f"{path}: expected {N} poses, found {len(poses)}")
    return poses


def sample_random_continuous_design(
    rng: np.random.Generator,
    N: int,
    *,
    theta_bounds: tuple[float, float],
    beta_bounds: tuple[float, float],
    d_bounds: tuple[float, float],
    pose_geometry: str,
    branch_mode: str,
) -> list[PoseSpec]:
    """Random same-budget design: N distinct lines, one pose per line."""
    if N > 9:
        raise ValueError("same-budget random design requires N <= 9")
    line_ids = sorted(int(x) for x in rng.choice(9, size=N, replace=False))
    out: list[PoseSpec] = []
    for line_id in line_ids:
        theta = float(rng.uniform(*theta_bounds))
        d_mm = float(rng.uniform(*d_bounds))
        if pose_geometry == "paper_incidence":
            # Exact feasible interval: beta in [90-theta, 90+theta],
            # intersected with the user-provided global bounds.
            lo = max(float(beta_bounds[0]), 90.0 - theta)
            hi = min(float(beta_bounds[1]), 90.0 + theta)
            if lo > hi:
                raise ValueError(
                    f"no feasible beta for theta={theta:g} within {beta_bounds}"
                )
            beta = float(rng.uniform(lo, hi))
        else:
            beta = float(rng.uniform(*beta_bounds))
        out.append(
            PoseSpec(
                line_id=line_id,
                theta_deg=theta,
                beta_deg=beta,
                d_mm=d_mm,
                branch_sign=branch_sign_for_line(line_id, branch_mode),
            )
        )
    return out


def sample_random_discrete_design(
    rng: np.random.Generator,
    N: int,
    *,
    theta_pool: Sequence[float],
    beta_pool: Sequence[float],
    d_pool: Sequence[float],
    pose_geometry: str,
    branch_mode: str,
) -> list[PoseSpec]:
    """Random same-budget design constrained to a discrete pose grid."""
    if N > 9:
        raise ValueError("same-budget discrete design requires N <= 9")
    line_ids = sorted(int(x) for x in rng.choice(9, size=N, replace=False))
    out: list[PoseSpec] = []
    for line_id in line_ids:
        feasible: list[tuple[float, float, float]] = []
        for theta in theta_pool:
            for beta in beta_pool:
                if pose_geometry == "paper_incidence":
                    if math.sin(math.radians(float(beta))) + 1e-12 < math.cos(
                        math.radians(float(theta))
                    ):
                        continue
                for d_mm in d_pool:
                    feasible.append((float(theta), float(beta), float(d_mm)))
        if not feasible:
            raise ValueError("discrete pool contains no feasible pose")
        theta, beta, d_mm = feasible[int(rng.integers(len(feasible)))]
        out.append(
            PoseSpec(
                line_id=line_id,
                theta_deg=theta,
                beta_deg=beta,
                d_mm=d_mm,
                branch_sign=branch_sign_for_line(line_id, branch_mode),
            )
        )
    return out


def legacy81_design(branch_mode: str) -> list[PoseSpec]:
    poses: list[PoseSpec] = []
    for line_id in range(9):
        for d_mm in (60.0, 90.0, 120.0):
            for beta in (60.0, 90.0, 120.0):
                poses.append(
                    PoseSpec(
                        line_id=line_id,
                        theta_deg=30.0,
                        beta_deg=beta,
                        d_mm=d_mm,
                        branch_sign=branch_sign_for_line(line_id, branch_mode),
                    )
                )
    assert len(poses) == 81
    return poses


def legacy117_design(branch_mode: str) -> list[PoseSpec]:
    poses = legacy81_design(branch_mode)
    for line_id in (1, 2, 5, 6):
        for d_mm in (60.0, 90.0, 120.0):
            for beta in (60.0, 90.0, 120.0):
                poses.append(
                    PoseSpec(
                        line_id=line_id,
                        theta_deg=60.0,
                        beta_deg=beta,
                        d_mm=d_mm,
                        branch_sign=branch_sign_for_line(line_id, branch_mode),
                    )
                )
    assert len(poses) == 117
    return poses


# -----------------------------------------------------------------------------
# Scan generation and calibration
# -----------------------------------------------------------------------------


def generate_scans_from_design(
    *,
    poses: Sequence[PoseSpec],
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    T_true: np.ndarray,
    x_values: np.ndarray,
    noise_std: float,
    noise_seed: int,
    radius_mm: float,
    pose_geometry: str,
    check_reachability: bool,
) -> list:
    lines = circular_lines(radius_mm, n_lines=9)
    rng_noise = np.random.default_rng(noise_seed)
    scans = []

    for scan_id, pose in enumerate(poses):
        line_p0, line_p1 = lines[int(pose.line_id)]
        T_base_s = sensor_pose_from_target_line(
            plane_R=plane_R,
            plane_t=plane_t,
            line_p0=line_p0,
            line_p1=line_p1,
            d_mm=float(pose.d_mm),
            theta_deg=float(pose.theta_deg),
            beta_deg=float(pose.beta_deg),
            branch_sign=float(pose.branch_sign),
            pose_geometry=pose_geometry,
        )
        # Ground-truth trajectory planning: command the flange pose that places
        # the true sensor exactly at the desired target-relative pose.
        T_base_ef = T_base_s @ inv_T(T_true)
        if check_reachability and not is_reachable_simple(T_base_ef):
            raise RuntimeError(
                f"unreachable pose: line={pose.line_id}, theta={pose.theta_deg}, "
                f"beta={pose.beta_deg}, d={pose.d_mm}"
            )

        scan = simulate_profile_on_plane(
            T_base_ef=T_base_ef,
            T_ef_s_true=T_true,
            plane_n=plane_n,
            plane_l=plane_l,
            x_values=x_values,
            noise_std=float(noise_std),
            rng=rng_noise,
            plane_id=0,
            scan_id=scan_id,
            meta={
                "line_id": int(pose.line_id),
                "theta_deg": float(pose.theta_deg),
                "beta_deg": float(pose.beta_deg),
                "d_mm": float(pose.d_mm),
                "theta_branch_sign": float(pose.branch_sign),
                "pose_geometry": pose_geometry,
            },
        )
        scans.append(scan)

    return scans


def reconstruct_points_base(scans: Sequence, T_ef_s: np.ndarray) -> np.ndarray:
    from laser_handeye.se3 import transform_points

    chunks: list[np.ndarray] = []
    for scan in scans:
        pts_s = np.asarray(scan.valid_points_s, dtype=float)
        if pts_s.size == 0:
            continue
        pts_ef = transform_points(T_ef_s, pts_s)
        pts_b = transform_points(np.asarray(scan.T_base_ef), pts_ef)
        chunks.append(pts_b)
    if not chunks:
        return np.empty((0, 3), dtype=float)
    return np.vstack(chunks)


def self_fit_plane_rms(scans: Sequence, T_ef_s: np.ndarray) -> float:
    from laser_handeye.geometry import fit_plane_pca

    pts = reconstruct_points_base(scans, T_ef_s)
    if len(pts) < 3:
        return float("inf")
    _n, _l, _c, rms = fit_plane_pca(pts)
    return float(rms)


def calibrate_one_method(
    *,
    trial: int,
    method: str,
    method_family: str,
    poses: Sequence[PoseSpec],
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    T_true: np.ndarray,
    T_init: np.ndarray,
    x_values: np.ndarray,
    noise_std: float,
    noise_seed: int,
    radius_mm: float,
    pose_geometry: str,
    check_reachability: bool,
    max_iter: int,
    tol: float,
    plane_offset_mode: str,
    solver_update_mode: str,
    nonlinear_refine: bool,
    nonlinear_loss: str,
    nonlinear_f_scale_mm: float,
    nonlinear_max_nfev: int,
) -> TrialRow:
    try:
        scans = generate_scans_from_design(
            poses=poses,
            plane_R=plane_R,
            plane_t=plane_t,
            plane_n=plane_n,
            plane_l=plane_l,
            T_true=T_true,
            x_values=x_values,
            noise_std=noise_std,
            noise_seed=noise_seed,
            radius_mm=radius_mm,
            pose_geometry=pose_geometry,
            check_reachability=check_reachability,
        )
        if len(scans) != len(poses):
            raise RuntimeError("scan count changed unexpectedly")

        result = _call_with_supported_kwargs(
            calibrate_planes,
            {0: scans},
            T_init=np.asarray(T_init, dtype=float),
            max_iter=int(max_iter),
            tol=float(tol),
            plane_offset_mode=plane_offset_mode,
            solver_update_mode=solver_update_mode,
        )
        T_est = np.asarray(result.T_ef_s, dtype=float)
        converged = bool(result.converged)
        iterations = int(result.iterations)
        rank_last = int(result.rank_history[-1]) if result.rank_history else -1
        cond_last = (
            float(result.cond_history[-1])
            if result.cond_history
            else float("nan")
        )

        if nonlinear_refine:
            nl = _call_with_supported_kwargs(
                refine_handeye_nonlinear,
                {0: scans},
                T_est,
                plane_mode="refit",
                loss=nonlinear_loss,
                f_scale_mm=float(nonlinear_f_scale_mm),
                max_nfev=int(nonlinear_max_nfev),
            )
            T_est = np.asarray(nl.T_ef_s, dtype=float)
            converged = converged and bool(nl.success)

        t_err = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
        r_err = float(rot_error_deg(T_est[:3, :3], T_true[:3, :3]))
        plane_rms = self_fit_plane_rms(scans, T_est)
        finite = np.isfinite(t_err) and np.isfinite(r_err) and np.isfinite(plane_rms)
        calibration_success = bool(converged and finite)

        return TrialRow(
            trial=trial,
            method=method,
            method_family=method_family,
            N_nominal=len(poses),
            N_actual=len(scans),
            converged=converged,
            calibration_success=calibration_success,
            iterations=iterations,
            rank_last=rank_last,
            cond_last=cond_last,
            trans_err_mm=t_err,
            rot_err_deg=r_err,
            final_plane_rms_mm=plane_rms,
            exception="",
        )
    except Exception as exc:
        return TrialRow(
            trial=trial,
            method=method,
            method_family=method_family,
            N_nominal=len(poses),
            N_actual=0,
            converged=False,
            calibration_success=False,
            iterations=0,
            rank_last=-1,
            cond_last=float("inf"),
            trans_err_mm=float("nan"),
            rot_err_deg=float("nan"),
            final_plane_rms_mm=float("nan"),
            exception=f"{type(exc).__name__}: {exc}",
        )


# -----------------------------------------------------------------------------
# Statistics / outputs
# -----------------------------------------------------------------------------


def save_trials(rows: Sequence[TrialRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(TrialRow.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def finite_values(rows: Sequence[TrialRow], attr: str) -> np.ndarray:
    vals = np.asarray([getattr(r, attr) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def summarize(rows: Sequence[TrialRow]) -> list[dict[str, object]]:
    methods = sorted({r.method for r in rows})
    out: list[dict[str, object]] = []
    for method in methods:
        group = [r for r in rows if r.method == method]
        t = finite_values(group, "trans_err_mm")
        rot = finite_values(group, "rot_err_deg")
        rms = finite_values(group, "final_plane_rms_mm")
        success = np.asarray([r.calibration_success for r in group], dtype=bool)
        converged = np.asarray([r.converged for r in group], dtype=bool)
        out.append(
            {
                "method": method,
                "method_family": group[0].method_family,
                "N": group[0].N_nominal,
                "trials": len(group),
                "success_rate": float(np.mean(success)) if len(success) else float("nan"),
                "convergence_rate": float(np.mean(converged)) if len(converged) else float("nan"),
                "median_trans_err_mm": float(np.median(t)) if len(t) else float("nan"),
                "p90_trans_err_mm": float(np.percentile(t, 90)) if len(t) else float("nan"),
                "mean_trans_err_mm": float(np.mean(t)) if len(t) else float("nan"),
                "median_rot_err_deg": float(np.median(rot)) if len(rot) else float("nan"),
                "p90_rot_err_deg": float(np.percentile(rot, 90)) if len(rot) else float("nan"),
                "mean_rot_err_deg": float(np.mean(rot)) if len(rot) else float("nan"),
                "median_plane_rms_mm": float(np.median(rms)) if len(rms) else float("nan"),
            }
        )
    return out


def save_dict_rows(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(rows: Sequence[TrialRow], n_values: Sequence[int]) -> list[dict[str, object]]:
    by_key = {(r.trial, r.method): r for r in rows}
    comparisons: list[dict[str, object]] = []

    for N in n_values:
        opt = f"opt_{N}"
        comparators = [f"rand_cont_{N}", f"rand_disc_{N}", "legacy81", "legacy117"]
        for other in comparators:
            pairs_t: list[tuple[float, float]] = []
            pairs_r: list[tuple[float, float]] = []
            both_success = 0
            opt_success = 0
            other_success = 0
            total = 0

            trial_ids = sorted({r.trial for r in rows})
            for trial in trial_ids:
                a = by_key.get((trial, opt))
                b = by_key.get((trial, other))
                if a is None or b is None:
                    continue
                total += 1
                opt_success += int(a.calibration_success)
                other_success += int(b.calibration_success)
                if a.calibration_success and b.calibration_success:
                    both_success += 1
                    if np.isfinite(a.trans_err_mm) and np.isfinite(b.trans_err_mm):
                        pairs_t.append((a.trans_err_mm, b.trans_err_mm))
                    if np.isfinite(a.rot_err_deg) and np.isfinite(b.rot_err_deg):
                        pairs_r.append((a.rot_err_deg, b.rot_err_deg))

            def paired_stats(pairs: list[tuple[float, float]]) -> dict[str, float]:
                if not pairs:
                    return {
                        "n_pairs": 0,
                        "win_rate": float("nan"),
                        "median_ratio_opt_over_other": float("nan"),
                        "median_difference_opt_minus_other": float("nan"),
                        "wilcoxon_p": float("nan"),
                    }
                arr = np.asarray(pairs, dtype=float)
                a = arr[:, 0]
                b = arr[:, 1]
                safe = b > np.finfo(float).eps
                ratios = a[safe] / b[safe]
                diff = a - b
                try:
                    p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
                except ValueError:
                    p = float("nan")
                return {
                    "n_pairs": len(arr),
                    "win_rate": float(np.mean(a < b)),
                    "median_ratio_opt_over_other": float(np.median(ratios)) if len(ratios) else float("nan"),
                    "median_difference_opt_minus_other": float(np.median(diff)),
                    "wilcoxon_p": p,
                }

            ts = paired_stats(pairs_t)
            rs = paired_stats(pairs_r)
            comparisons.append(
                {
                    "optimized": opt,
                    "comparator": other,
                    "total_trials": total,
                    "both_success": both_success,
                    "opt_success_rate": opt_success / total if total else float("nan"),
                    "other_success_rate": other_success / total if total else float("nan"),
                    "translation_pairs": ts["n_pairs"],
                    "translation_win_rate_opt": ts["win_rate"],
                    "translation_median_ratio_opt_over_other": ts[
                        "median_ratio_opt_over_other"
                    ],
                    "translation_median_diff_mm": ts[
                        "median_difference_opt_minus_other"
                    ],
                    "translation_wilcoxon_p": ts["wilcoxon_p"],
                    "rotation_pairs": rs["n_pairs"],
                    "rotation_win_rate_opt": rs["win_rate"],
                    "rotation_median_ratio_opt_over_other": rs[
                        "median_ratio_opt_over_other"
                    ],
                    "rotation_median_diff_deg": rs[
                        "median_difference_opt_minus_other"
                    ],
                    "rotation_wilcoxon_p": rs["wilcoxon_p"],
                }
            )
    return comparisons


def ordered_methods(n_values: Sequence[int]) -> list[str]:
    names: list[str] = []
    for N in n_values:
        names.extend([f"opt_{N}", f"rand_cont_{N}", f"rand_disc_{N}"])
    names.extend(["legacy81", "legacy117"])
    return names


def save_boxplot(
    rows: Sequence[TrialRow],
    n_values: Sequence[int],
    attr: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    methods = ordered_methods(n_values)
    data: list[np.ndarray] = []
    labels: list[str] = []
    for method in methods:
        vals = np.asarray(
            [getattr(r, attr) for r in rows if r.method == method and np.isfinite(getattr(r, attr))],
            dtype=float,
        )
        if len(vals):
            data.append(vals)
            labels.append(method)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(labels)), 6.2))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_success_plot(rows: Sequence[TrialRow], n_values: Sequence[int], out_path: Path) -> None:
    methods = ordered_methods(n_values)
    rates: list[float] = []
    labels: list[str] = []
    for method in methods:
        group = [r for r in rows if r.method == method]
        if not group:
            continue
        labels.append(method)
        rates.append(float(np.mean([r.calibration_success for r in group])))
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(labels)), 5.4))
    ax.bar(np.arange(len(labels)), rates)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Calibration success rate")
    ax.set_title("Paired Monte Carlo calibration success")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Monte Carlo validation of optimized minimum-scan circular pose sets."
    )
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--n-values", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paired_mc_pose_validation"))

    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--pose-geometry",
        choices=["paper_incidence", "observable_dihedral"],
        default="paper_incidence",
    )
    parser.add_argument(
        "--projection-branch-mode",
        choices=["alternating", "positive"],
        default="alternating",
    )
    parser.add_argument("--check-reachability", action="store_true")

    # Random continuous baseline bounds. Match these to the optimization run.
    parser.add_argument("--theta-min-deg", type=float, default=30.0)
    parser.add_argument("--theta-max-deg", type=float, default=60.0)
    parser.add_argument("--beta-min-deg", type=float, default=30.0)
    parser.add_argument("--beta-max-deg", type=float, default=150.0)
    parser.add_argument("--d-min-mm", type=float, default=60.0)
    parser.add_argument("--d-max-mm", type=float, default=120.0)

    # Random discrete same-budget baseline.
    parser.add_argument("--discrete-theta-deg", type=float, nargs="+", default=[30.0, 60.0])
    parser.add_argument("--discrete-beta-deg", type=float, nargs="+", default=[60.0, 90.0, 120.0])
    parser.add_argument("--discrete-d-mm", type=float, nargs="+", default=[60.0, 90.0, 120.0])

    # Random physical system.
    parser.add_argument("--plane-tilt-min-deg", type=float, default=1.0)
    parser.add_argument("--plane-tilt-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-yaw-min-deg", type=float, default=-5.0)
    parser.add_argument("--plane-yaw-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-center-xy-range-mm", type=float, default=100.0)
    parser.add_argument("--plane-center-z-min-mm", type=float, default=400.0)
    parser.add_argument("--plane-center-z-max-mm", type=float, default=550.0)

    # Initial error / solver. Defaults mirror the benchmark closely.
    parser.add_argument("--init-mode", choices=["relative", "carlson"], default="relative")
    parser.add_argument("--rel-offset", type=float, default=0.1)
    parser.add_argument("--init-translation-range-mm", type=float, default=200.0)
    parser.add_argument("--init-angle-range-deg", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument(
        "--plane-offset-mode",
        choices=["joint", "fitted", "difference"],
        default="joint",
    )
    parser.add_argument(
        "--solver-update-mode",
        choices=["simultaneous", "separated"],
        default="simultaneous",
    )
    parser.add_argument("--nonlinear-refine", action="store_true")
    parser.add_argument(
        "--nonlinear-loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="linear",
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument("--nonlinear-max-nfev", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be positive")
    n_values = sorted(set(int(N) for N in args.n_values))
    if any(N < 1 or N > 9 for N in n_values):
        parser.error("--n-values must lie in [1,9]")

    optimized = {N: load_optimized_design(args.design_dir, N) for N in n_values}
    baseline81 = legacy81_design(args.projection_branch_mode)
    baseline117 = legacy117_design(args.projection_branch_mode)

    x_values = np.linspace(
        -float(args.profile_half_width_mm),
        float(args.profile_half_width_mm),
        int(args.profile_points),
    )
    plane_center_z_range = (
        float(args.plane_center_z_min_mm),
        float(args.plane_center_z_max_mm),
    )

    rows: list[TrialRow] = []
    start = time.perf_counter()

    for trial in range(args.trials):
        # Split trial-level randomness into independent but reproducible streams.
        seeds = np.random.SeedSequence([args.seed, trial]).spawn(6)
        gt_rng = np.random.default_rng(seeds[0])
        plane_rng = np.random.default_rng(seeds[1])
        init_rng = np.random.default_rng(seeds[2])
        random_cont_rng = np.random.default_rng(seeds[3])
        random_disc_rng = np.random.default_rng(seeds[4])
        noise_seed = int(np.random.default_rng(seeds[5]).integers(0, 2**32 - 1))

        T_true, true_angles_deg, true_translation_mm = sample_random_handeye(gt_rng)
        plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
            plane_rng,
            tilt_min_deg=float(args.plane_tilt_min_deg),
            tilt_max_deg=float(args.plane_tilt_max_deg),
            yaw_min_deg=float(args.plane_yaw_min_deg),
            yaw_max_deg=float(args.plane_yaw_max_deg),
            center_xy_range_mm=float(args.plane_center_xy_range_mm),
            center_z_range_mm=plane_center_z_range,
        )
        T_init = make_initial_guess(
            reference_angles_deg=true_angles_deg,
            reference_translation_mm=true_translation_mm,
            rng=init_rng,
            mode=args.init_mode,
            rel_offset=float(args.rel_offset),
            translation_range_mm=float(args.init_translation_range_mm),
            angle_range_deg=float(args.init_angle_range_deg),
        )

        methods: list[tuple[str, str, list[PoseSpec]]] = []
        for N in n_values:
            methods.append((f"opt_{N}", "optimized", optimized[N]))
            methods.append(
                (
                    f"rand_cont_{N}",
                    "random_continuous",
                    sample_random_continuous_design(
                        random_cont_rng,
                        N,
                        theta_bounds=(args.theta_min_deg, args.theta_max_deg),
                        beta_bounds=(args.beta_min_deg, args.beta_max_deg),
                        d_bounds=(args.d_min_mm, args.d_max_mm),
                        pose_geometry=args.pose_geometry,
                        branch_mode=args.projection_branch_mode,
                    ),
                )
            )
            methods.append(
                (
                    f"rand_disc_{N}",
                    "random_discrete",
                    sample_random_discrete_design(
                        random_disc_rng,
                        N,
                        theta_pool=args.discrete_theta_deg,
                        beta_pool=args.discrete_beta_deg,
                        d_pool=args.discrete_d_mm,
                        pose_geometry=args.pose_geometry,
                        branch_mode=args.projection_branch_mode,
                    ),
                )
            )

        methods.extend(
            [
                ("legacy81", "many_scan_baseline", baseline81),
                ("legacy117", "many_scan_mixed_theta_baseline", baseline117),
            ]
        )

        for method, family, poses in methods:
            rows.append(
                calibrate_one_method(
                    trial=trial,
                    method=method,
                    method_family=family,
                    poses=poses,
                    plane_R=plane_R,
                    plane_t=plane_t,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    T_true=T_true,
                    T_init=T_init,
                    x_values=x_values,
                    noise_std=float(args.noise_std),
                    noise_seed=noise_seed,
                    radius_mm=float(args.radius_mm),
                    pose_geometry=args.pose_geometry,
                    check_reachability=bool(args.check_reachability),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    plane_offset_mode=args.plane_offset_mode,
                    solver_update_mode=args.solver_update_mode,
                    nonlinear_refine=bool(args.nonlinear_refine),
                    nonlinear_loss=args.nonlinear_loss,
                    nonlinear_f_scale_mm=float(args.nonlinear_f_scale_mm),
                    nonlinear_max_nfev=int(args.nonlinear_max_nfev),
                )
            )

        if (trial + 1) % max(1, args.log_every) == 0 or trial + 1 == args.trials:
            elapsed = time.perf_counter() - start
            done = [r for r in rows if r.trial <= trial]
            success = sum(r.calibration_success for r in done)
            print(
                f"[paired-MC] {trial+1}/{args.trials} trials | "
                f"method-runs={len(done)} | successes={success} | elapsed={elapsed:.1f}s"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_trials(rows, out_dir / "trials.csv")
    summary = summarize(rows)
    save_dict_rows(summary, out_dir / "summary.csv")
    paired = paired_comparisons(rows, n_values)
    save_dict_rows(paired, out_dir / "paired_comparisons.csv")

    save_boxplot(
        rows,
        n_values,
        "trans_err_mm",
        "Translation error norm [mm]",
        "Paired Monte Carlo: translation error",
        out_dir / "translation_error_boxplot.png",
    )
    save_boxplot(
        rows,
        n_values,
        "rot_err_deg",
        "Rotation geodesic error [deg]",
        "Paired Monte Carlo: rotation error",
        out_dir / "rotation_error_boxplot.png",
    )
    save_success_plot(rows, n_values, out_dir / "success_rate.png")

    print("\n=== SUMMARY ===")
    for s in summary:
        print(
            f"{s['method']:>12s} | N={int(s['N']):3d} | "
            f"success={100*float(s['success_rate']):6.1f}% | "
            f"median t={float(s['median_trans_err_mm']):.6g} mm | "
            f"median R={float(s['median_rot_err_deg']):.6g} deg | "
            f"p90 t={float(s['p90_trans_err_mm']):.6g} mm"
        )

    print("\n=== PAIRED OPTIMIZED COMPARISONS ===")
    for p in paired:
        print(
            f"{p['optimized']} vs {p['comparator']}: "
            f"t-win={100*float(p['translation_win_rate_opt']):.1f}% | "
            f"t-ratio={float(p['translation_median_ratio_opt_over_other']):.4g} | "
            f"R-win={100*float(p['rotation_win_rate_opt']):.1f}% | "
            f"R-ratio={float(p['rotation_median_ratio_opt_over_other']):.4g}"
        )

    print(f"\nsaved: {out_dir / 'trials.csv'}")
    print(f"saved: {out_dir / 'summary.csv'}")
    print(f"saved: {out_dir / 'paired_comparisons.csv'}")
    print(f"saved plots under: {out_dir}")


if __name__ == "__main__":
    main()