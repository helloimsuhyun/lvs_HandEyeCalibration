#!/usr/bin/env python3
"""
simulation_lhs_ds_subset_search.py

Purpose
-------
Discover repeatable geometric patterns in finite-N D_s-optimal pose subsets
selected from a broad, exact 6-D Latin-hypercube candidate pool.

Pipeline
--------
1. Sample one GT hand-eye transform and one single target plane.
2. Build an exact LHS in plane-relative pose parameters

       q = (u, v, d, tilt, azimuth, roll)

   while preserving every LHS stratum under feasibility rejection.
3. Simulate ideal/noise-free LaserScan profiles.
4. Evaluate every scan at the GT nominal point with the SAME 9-column
   joint Jacobian used by subset_ds_exchange_search.py:

       [hand-eye rotation(3), hand-eye translation(3),
        plane-normal tangent(2), plane offset(1)]

5. For each requested N, select a best-found subset by the existing
   D_s Schur-complement logdet objective and multi-start exchange search.
6. Export the selected relative-pose parameters and information-block
   diagnostics so patterns can be studied across N and across LHS/GT seeds.

This script intentionally does NOT recalibrate a full synthetic dataset to
obtain the Jacobian nominal point. Simulation already provides exact truth:

    T_nominal = T_ef_s_true
    n_nominal = plane.n
    l_nominal = plane.l

The plane remains an UNKNOWN nuisance parameter in D_s. Truth is used only as
the linearization point.

Repository dependencies
-----------------------
This wrapper reuses the user's existing validated simulation and subset-search
modules:

    main/generate_plane_uniform_comparison.py
    real_laser_handeye/calibrate_only/subset_ds_exchange_search.py

The script bootstraps both workspace package roots, so it can be run without a
custom PYTHONPATH. The example below assumes the current directory is
``robust_laser_handeye``; from the workspace root, prefix the script path with
``robust_laser_handeye/``.

Example
-------
python3 main/simulation_lhs_ds_subset_search.py \
  --output-dir runs/sim/lhs_ds_pattern \
  --trials 10 \
  --candidate-pool-size 300 \
  --n-values 5 6 7 8 10 12 \
  --profile-points 100 \
  --profile-half-width-mm 25 \
  --profile-depth-range-mm 60 150 \
  --candidate-center-depth-range-mm 90 120 \
  --candidate-target-u-range-mm -70 70 \
  --candidate-target-v-range-mm -70 70 \
  --candidate-view-tilt-range-deg 5 80 \
  --candidate-view-azimuth-range-deg -180 180 \
  --candidate-sensor-roll-range-deg -180 180 \
  --random-starts 100 \
  --two-exchange-top-k 3
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

# ``main`` and ``laser_handeye`` live under robust_laser_handeye, while the
# reused real-data D_s implementation lives at the workspace root.  Add both
# explicit roots before importing either stack so invocation does not depend on
# the caller's current directory or PYTHONPATH.
_ROBUST_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _ROBUST_ROOT.parent
for _import_root in (_WORKSPACE_ROOT, _ROBUST_ROOT):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

# Match the simulation stack used by the repository's exact sliced-LHS generator.
from main import generate_independent_random_plane_comparison as base
from main import generate_plane_uniform_comparison as uniform
from laser_handeye.pose_design import audit_latin_hypercube, latin_hypercube_strata
from laser_handeye.simulation import sample_random_handeye

# Reuse the exact D_s information and exchange-search implementation from
# the user's real-data subset code.
from real_laser_handeye.calibrate_only import subset_ds_exchange_search as ds


PARAMETER_NAMES = (
    "target_u_mm",
    "target_v_mm",
    "center_depth_mm",
    "view_tilt_deg",
    "view_azimuth_deg",
    "sensor_roll_deg",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must have exactly two values")
    lo, hi = map(float, values)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(f"invalid {name}: {values}")
    return lo, hi


def positive_condition(matrix: np.ndarray, rtol: float = 1e-12) -> float:
    A = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eig = np.linalg.eigvalsh(A)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    threshold = rtol * scale
    positive = eig[eig > threshold]
    if len(positive) != len(eig):
        return float("inf")
    return float(positive[-1] / positive[0])


def normalized_block_coupling(
    H_rr: np.ndarray,
    H_rt: np.ndarray,
    H_tt: np.ndarray,
) -> float:
    """Dimensionless Frobenius coupling diagnostic in the scaled coordinates."""
    a = float(np.linalg.norm(H_rr, ord="fro"))
    b = float(np.linalg.norm(H_tt, ord="fro"))
    denom = math.sqrt(max(a * b, 0.0))
    if denom <= 0.0:
        return float("inf")
    return float(np.linalg.norm(H_rt, ord="fro") / denom)


def circular_resultant_length_deg(values_deg: Iterable[float]) -> float:
    """0 ~= azimuths spread around circle, 1 ~= all concentrated together."""
    x = np.asarray(list(values_deg), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    z = np.exp(1j * np.deg2rad(x))
    return float(abs(np.mean(z)))


# ---------------------------------------------------------------------------
# Exact LHS candidate generation
# ---------------------------------------------------------------------------


def generate_exact_lhs_candidate_bank(
    *,
    rng: np.random.Generator,
    master_seed: int,
    trial_index: int,
    count: int,
    frame,
    common_center: np.ndarray,
    T_ef_s_true: np.ndarray,
    x_values: np.ndarray,
    fair_config,
    candidate_config,
    max_design_attempts: int,
    stratum_jitter_retries: int,
) -> tuple[list, list, np.ndarray, dict]:
    """Generate a feasible exact LHS without silently dropping LHS strata.

    A naive "generate LHS -> reject invalid rows -> keep the rest" is no longer
    an exact Latin hypercube. Here each point owns one stratum per dimension.

    For a fixed stratum combination, only the within-stratum jitter is retried.
    If that combination cannot become feasible, the ENTIRE LHS is regenerated.
    Thus every accepted final design still has exactly one sample per stratum
    in each active dimension.
    """
    if count < 1:
        raise ValueError("candidate count must be positive")

    total_pose_attempts = 0
    failed_designs = 0

    for design_attempt in range(1, max_design_attempts + 1):
        strata = latin_hypercube_strata(rng, count, len(PARAMETER_NAMES))

        poses = []
        scans = []
        normalized_rows = []
        failed = False

        for candidate_id in range(count):
            lower = strata[candidate_id].astype(float) / float(count)
            accepted = None

            for _ in range(stratum_jitter_retries):
                total_pose_attempts += 1
                row = lower + rng.random(len(PARAMETER_NAMES)) / float(count)

                pose = uniform._uniform_candidate_from_row(
                    row,
                    sample_id=candidate_id,
                    block_id=0,
                    index_in_block=candidate_id,
                    master_seed=master_seed,
                    trial_index=trial_index,
                    config=candidate_config,
                )

                try:
                    T_base_s = uniform._make_sensor_pose_relative_to_plane(
                        frame,
                        common_center,
                        pose,
                    )
                except (ValueError, FloatingPointError):
                    continue

                feasibility = base._profile_feasibility(
                    plane=frame,
                    common_center=common_center,
                    T_base_s=T_base_s,
                    x_values=x_values,
                    config=fair_config,
                )
                if feasibility is None:
                    continue

                scan = uniform._simulate_uniform_scan(
                    T_ef_s_true=T_ef_s_true,
                    plane=frame,
                    plane_id=0,
                    T_base_s=T_base_s,
                    pose=pose,
                    feasibility=feasibility,
                    strategy="single_plane_exact_lhs_pool",
                    x_values=x_values,
                )

                accepted = (pose, scan, row)
                break

            if accepted is None:
                failed = True
                failed_designs += 1
                break

            pose, scan, row = accepted
            poses.append(pose)
            scans.append(scan)
            normalized_rows.append(np.asarray(row, dtype=float))

        if failed:
            continue

        normalized = np.asarray(normalized_rows, dtype=float)
        audit_latin_hypercube(normalized)

        return poses, scans, normalized, {
            "exact_lhs": True,
            "candidate_count": int(count),
            "dimensions": int(len(PARAMETER_NAMES)),
            "design_attempt": int(design_attempt),
            "failed_designs_before_success": int(failed_designs),
            "total_pose_attempts": int(total_pose_attempts),
            "average_pose_attempts_per_accepted": float(
                total_pose_attempts / max(count, 1)
            ),
        }

    raise RuntimeError(
        "could not generate a feasible exact LHS candidate bank. "
        "Reduce pose ranges, increase --stratum-jitter-retries, or increase "
        "--max-design-attempts."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def make_simulation_configs(args: argparse.Namespace):
    profile_depth = finite_pair(
        args.profile_depth_range_mm,
        "profile depth range",
    )
    center_depth = finite_pair(
        args.candidate_center_depth_range_mm,
        "candidate center depth range",
    )
    if center_depth[0] <= 0.0:
        raise ValueError("candidate center depth must be positive")
    if profile_depth[0] <= 0.0:
        raise ValueError("profile depth range must be positive")

    tilt = finite_pair(
        args.candidate_view_tilt_range_deg,
        "candidate tilt range",
    )
    if tilt[0] < 0.0 or tilt[1] >= 89.0:
        raise ValueError("tilt range must lie within [0, 89) deg")

    u_range = finite_pair(args.candidate_target_u_range_mm, "candidate U range")
    v_range = finite_pair(args.candidate_target_v_range_mm, "candidate V range")
    az_range = finite_pair(
        args.candidate_view_azimuth_range_deg,
        "candidate azimuth range",
    )
    roll_range = finite_pair(
        args.candidate_sensor_roll_range_deg,
        "candidate roll range",
    )

    # Configuration expected by the shared validated profile-feasibility code.
    fair_config = base.FairGenerationConfig(
        total_scans=int(args.candidate_pool_size),
        profile_points=int(args.profile_points),
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.tangent_range_mm),
        profile_depth_range_mm=profile_depth,
        view_tilt_range_deg=tilt,
        view_azimuth_range_deg=az_range,
        sensor_roll_range_deg=roll_range,
        plane_angle_range_deg=finite_pair(
            args.plane_angle_range_deg,
            "plane angle range",
        ),
        plane_center_xy_range_mm=finite_pair(
            args.plane_center_xy_range_mm,
            "plane center XY range",
        ),
        plane_center_z_range_mm=finite_pair(
            args.plane_center_z_range_mm,
            "plane center Z range",
        ),
        max_local_pose_trials=max(
            int(args.max_design_attempts)
            * int(args.stratum_jitter_retries)
            * int(args.candidate_pool_size),
            200_000,
        ),
        min_abs_plane_normal_z=float(args.min_abs_plane_normal_z),
        verification_atol=float(args.verification_atol),
        min_effective_cross_plane_angle_deg=0.0,
    )

    candidate_config = uniform.UniformConfig(
        target_u_range_mm=u_range,
        target_v_range_mm=v_range,
        depth_range_mm=center_depth,
        tilt_range_deg=tilt,
        azimuth_range_deg=az_range,
        roll_range_deg=roll_range,
        max_batches=int(args.max_design_attempts),
        batch_multiplier=1,
    )

    return fair_config, candidate_config


# ---------------------------------------------------------------------------
# Diagnostics / output
# ---------------------------------------------------------------------------


def _pose_identifier(pose, fallback: int | None = None) -> int:
    """Return the repository pose/sample ID without assuming one dataclass version."""
    for name in ("candidate_id", "sample_id"):
        value = getattr(pose, name, None)
        if value is not None:
            return int(value)
    if fallback is None:
        raise AttributeError("pose has neither candidate_id nor sample_id")
    return int(fallback)


def pose_row(pose, normalized_row: np.ndarray, *, fallback_id: int | None = None) -> dict:
    pose_id = _pose_identifier(pose, fallback=fallback_id)
    row = {
        "candidate_id": pose_id,
        "target_u_mm": float(pose.target_u_mm),
        "target_v_mm": float(pose.target_v_mm),
        "center_depth_mm": float(pose.center_depth_mm),
        "view_tilt_deg": float(pose.view_tilt_deg),
        "view_azimuth_deg": float(pose.view_azimuth_deg),
        "sensor_roll_deg": float(pose.sensor_roll_deg),
        # Canonical aliases shared with the real-data view-pose analyses.  The
        # ideal simulator models no physical/coordinate-origin offset, hence
        # center_depth_mm and view_distance_mm are identical here.
        "view_u_mm": float(pose.target_u_mm),
        "view_v_mm": float(pose.target_v_mm),
        "view_distance_mm": float(pose.center_depth_mm),
        "view_roll_deg": float(pose.sensor_roll_deg),
    }
    if hasattr(pose, "block_id"):
        row["block_id"] = int(pose.block_id)
    if hasattr(pose, "index_in_block"):
        row["index_in_block"] = int(pose.index_in_block)
    for dim, name in enumerate(PARAMETER_NAMES):
        row[f"lhs_normalized_{name}"] = float(normalized_row[dim])
    return row


def subset_information_diagnostics(
    H_joint: np.ndarray,
    *,
    eigen_rtol: float,
) -> tuple[dict, np.ndarray | None]:
    score, H_eff, diag = ds.ds_score_from_information(
        H_joint,
        relative_eigen_tolerance=eigen_rtol,
    )

    out = {
        "ds_logdet": float(score),
        **diag,
    }

    if H_eff is None:
        out.update(
            {
                "rotation_block_condition": float("inf"),
                "translation_block_condition": float("inf"),
                "rotation_translation_coupling_fro": float("inf"),
                "rotation_translation_coupling_normalized": float("inf"),
            }
        )
        return out, None

    H_rr = H_eff[:3, :3]
    H_rt = H_eff[:3, 3:6]
    H_tt = H_eff[3:6, 3:6]

    out.update(
        {
            "rotation_block_condition": positive_condition(H_rr),
            "translation_block_condition": positive_condition(H_tt),
            "rotation_translation_coupling_fro": float(
                np.linalg.norm(H_rt, ord="fro")
            ),
            "rotation_translation_coupling_normalized": normalized_block_coupling(
                H_rr,
                H_rt,
                H_tt,
            ),
            "rotation_block_trace": float(np.trace(H_rr)),
            "translation_block_trace": float(np.trace(H_tt)),
            "rotation_block_eigenvalues": np.linalg.eigvalsh(
                0.5 * (H_rr + H_rr.T)
            ).tolist(),
            "translation_block_eigenvalues": np.linalg.eigvalsh(
                0.5 * (H_tt + H_tt.T)
            ).tolist(),
        }
    )
    return out, H_eff


def selected_geometry_summary(N: int, selected_poses: list) -> dict:
    u = np.asarray([p.target_u_mm for p in selected_poses], dtype=float)
    v = np.asarray([p.target_v_mm for p in selected_poses], dtype=float)
    depth = np.asarray([p.center_depth_mm for p in selected_poses], dtype=float)
    tilt = np.asarray([p.view_tilt_deg for p in selected_poses], dtype=float)
    azimuth = np.asarray([p.view_azimuth_deg for p in selected_poses], dtype=float)
    roll = np.asarray([p.sensor_roll_deg for p in selected_poses], dtype=float)

    radius = np.hypot(u, v)

    return {
        "N": int(N),
        "u_mean_mm": float(np.mean(u)),
        "v_mean_mm": float(np.mean(v)),
        "uv_centroid_radius_mm": float(np.hypot(np.mean(u), np.mean(v))),
        "u_range_mm": float(np.ptp(u)),
        "v_range_mm": float(np.ptp(v)),
        "uv_radius_mean_mm": float(np.mean(radius)),
        "uv_radius_std_mm": float(np.std(radius)),
        "depth_mean_mm": float(np.mean(depth)),
        "depth_std_mm": float(np.std(depth)),
        "depth_range_mm": float(np.ptp(depth)),
        "tilt_mean_deg": float(np.mean(tilt)),
        "tilt_std_deg": float(np.std(tilt)),
        "tilt_range_deg": float(np.ptp(tilt)),
        "azimuth_circular_span_deg": float(ds.circular_span_deg(azimuth)),
        "azimuth_resultant_length": circular_resultant_length_deg(azimuth),
        "roll_circular_span_deg": float(ds.circular_span_deg(roll)),
        "roll_resultant_length": circular_resultant_length_deg(roll),
    }


def save_selected_plot(
    path: Path,
    N: int,
    trial_index: int,
    selected_poses: list,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; skipping selected-pose plot")
        return

    u = np.asarray([p.target_u_mm for p in selected_poses])
    v = np.asarray([p.target_v_mm for p in selected_poses])
    depth = np.asarray([p.center_depth_mm for p in selected_poses])
    tilt = np.asarray([p.view_tilt_deg for p in selected_poses])
    azimuth = np.asarray([p.view_azimuth_deg for p in selected_poses])
    roll = np.asarray([p.sensor_roll_deg for p in selected_poses])
    ids = np.asarray([_pose_identifier(p) for p in selected_poses], dtype=int)

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(u, v)
    for sid, x, y in zip(ids, u, v):
        ax1.annotate(str(sid), (x, y), xytext=(3, 3), textcoords="offset points")
    ax1.set_xlabel("target U [mm]")
    ax1.set_ylabel("target V [mm]")
    ax1.set_title("Selected U/V")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2, projection="polar")
    ax2.scatter(np.deg2rad(azimuth), tilt)
    ax2.set_title("Azimuth / tilt")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(tilt, depth)
    ax3.set_xlabel("tilt [deg]")
    ax3.set_ylabel("center depth [mm]")
    ax3.set_title("Tilt vs depth")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(tilt, roll)
    ax4.set_xlabel("tilt [deg]")
    ax4.set_ylabel("roll [deg]")
    ax4.set_title("Tilt vs roll")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(azimuth, roll)
    ax5.set_xlabel("azimuth [deg]")
    ax5.set_ylabel("roll [deg]")
    ax5.set_title("Azimuth vs roll")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(np.hypot(u, v), tilt)
    ax6.set_xlabel("sqrt(U^2+V^2) [mm]")
    ax6.set_ylabel("tilt [deg]")
    ax6.set_title("In-plane radius vs tilt")
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f"trial={trial_index}, N={N}, best-found D_s subset")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


def run_trial(
    args: argparse.Namespace,
    fair_config,
    candidate_config,
    trial_index: int,
) -> list[dict]:
    trial_dir = args.output_dir / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    sequence = np.random.SeedSequence([args.seed, trial_index])
    handeye_seq, plane_seq, lhs_seq, search_seq = sequence.spawn(4)

    # Exact simulation truth.
    T_ef_s_true, _, _ = sample_random_handeye(
        np.random.default_rng(handeye_seq)
    )

    # Same single-plane geometry generator used by the existing experiments.
    plane_result = base._make_plane_frames(
        np.random.default_rng(plane_seq),
        fair_config.plane_angle_range_deg,
        fair_config.plane_center_xy_range_mm,
        fair_config.plane_center_z_range_mm,
    )

    # The repository version paired with lhs returns three values.
    if len(plane_result) == 3:
        frames, common_center, frame_angles_deg = plane_result
    elif len(plane_result) == 4:
        # Compatibility with the shared-global generator variant.
        frames, common_center, frame_angles_deg, _target_frame_R = plane_result
    else:
        raise RuntimeError(
            f"unexpected _make_plane_frames return length: {len(plane_result)}"
        )

    plane = frames[0]
    x_values = np.linspace(
        -fair_config.profile_half_width_mm,
        fair_config.profile_half_width_mm,
        fair_config.profile_points,
    )

    print(
        f"\n=== trial {trial_index} ===\n"
        f"GT plane n={np.asarray(plane.n)} l={float(plane.l):.6g}\n"
        f"candidate pool={args.candidate_pool_size}"
    )

    poses, scans, normalized, lhs_stats = generate_exact_lhs_candidate_bank(
        rng=np.random.default_rng(lhs_seq),
        master_seed=int(args.seed),
        trial_index=int(trial_index),
        count=int(args.candidate_pool_size),
        frame=plane,
        common_center=np.asarray(common_center, dtype=float),
        T_ef_s_true=T_ef_s_true,
        x_values=x_values,
        fair_config=fair_config,
        candidate_config=candidate_config,
        max_design_attempts=int(args.max_design_attempts),
        stratum_jitter_retries=int(args.stratum_jitter_retries),
    )

    pool_rows = [
        pose_row(pose, normalized_row, fallback_id=i)
        for i, (pose, normalized_row) in enumerate(zip(poses, normalized))
    ]
    write_rows_csv(trial_dir / "candidate_pool.csv", pool_rows)

    write_json(
        trial_dir / "truth_and_pool.json",
        {
            "trial_index": int(trial_index),
            "T_ef_s_true": T_ef_s_true,
            "plane_normal_base": np.asarray(plane.n),
            "plane_offset_mm": float(plane.l),
            "plane_center_base_mm": np.asarray(common_center),
            "plane_frame_euler_xyz_deg": np.asarray(frame_angles_deg),
            "lhs_statistics": lhs_stats,
            "parameter_names": PARAMETER_NAMES,
            "view_pose_convention": dict(uniform.VIEW_POSE_CONVENTION),
        },
    )

    # ---------------------------------------------------------------
    # Precompute H_i at exact GT. Plane nuisance columns are retained.
    # ---------------------------------------------------------------
    information = []
    info_rows = []

    print("precomputing GT-nominal 9-DoF scan information...")
    for index, scan in enumerate(scans, start=1):
        H_i, diag = ds.compute_joint_scan_information(
            scan,
            T_ef_s_true,
            np.asarray(plane.n, dtype=float),
            float(plane.l),
            handeye_characteristic_length_mm=float(
                args.handeye_characteristic_length_mm
            ),
            plane_characteristic_length_mm=float(
                args.plane_characteristic_length_mm
            ),
            finite_difference_step_mm=float(args.design_step_mm),
            design_points_per_scan=int(args.design_points_per_scan),
        )
        information.append(H_i)
        info_rows.append(diag)

        if index == 1 or index % args.progress_every == 0 or index == len(scans):
            print(
                f"  H_i {index:4d}/{len(scans)} | "
                f"scan_id={diag['scan_id']} | "
                f"rank(J_i)={diag['joint_jacobian_rank']}"
            )

    information_array = np.stack(information, axis=0)

    np.savez_compressed(
        trial_dir / "scan_joint_information.npz",
        scan_ids=np.asarray([_pose_identifier(p, fallback=i) for i, p in enumerate(poses)], dtype=int),
        H=information_array,
    )
    write_rows_csv(
        trial_dir / "scan_information_diagnostics.csv",
        info_rows,
    )

    # ---------------------------------------------------------------
    # N-wise best subset search
    # ---------------------------------------------------------------
    search_rng = np.random.default_rng(search_seq)
    summary_rows: list[dict] = []

    for N in args.n_values:
        if N > len(scans):
            continue

        n_dir = trial_dir / f"N_{N:03d}"
        n_dir.mkdir(parents=True, exist_ok=True)

        best, candidates = ds.search_one_n(
            int(N),
            scans,
            information_array,
            args,
            search_rng,
        )

        ds.write_candidate_csv(
            n_dir / "multistart_candidates.csv",
            candidates[: args.save_top_k] if args.save_top_k > 0 else candidates,
            scans,
            best_score=float(best["score"]),
        )

        best_indices = tuple(int(i) for i in best["subset"])
        selected_poses = [poses[i] for i in best_indices]
        selected_scan_ids = [_pose_identifier(p) for p in selected_poses]

        selected_rows = []
        for selection_rank, index in enumerate(best_indices):
            row = pose_row(
                poses[index],
                normalized[index],
                fallback_id=index,
            )
            row.update(
                {
                    "selection_rank": int(selection_rank),
                    "N": int(N),
                }
            )
            selected_rows.append(row)

        write_rows_csv(
            n_dir / "selected_pose_parameters.csv",
            selected_rows,
        )

        H_joint = np.sum(information_array[list(best_indices)], axis=0)
        info_diag, H_eff = subset_information_diagnostics(
            H_joint,
            eigen_rtol=float(args.design_eigen_rtol),
        )

        if H_eff is not None:
            np.savetxt(
                n_dir / "H_eff_6x6.csv",
                H_eff,
                delimiter=",",
                fmt="%.12g",
            )
            np.savetxt(
                n_dir / "H_eff_RR_3x3.csv",
                H_eff[:3, :3],
                delimiter=",",
                fmt="%.12g",
            )
            np.savetxt(
                n_dir / "H_eff_Rt_3x3.csv",
                H_eff[:3, 3:6],
                delimiter=",",
                fmt="%.12g",
            )
            np.savetxt(
                n_dir / "H_eff_tt_3x3.csv",
                H_eff[3:6, 3:6],
                delimiter=",",
                fmt="%.12g",
            )

        geometry = selected_geometry_summary(int(N), selected_poses)

        result = {
            "trial_index": int(trial_index),
            "N": int(N),
            "best_found_not_global_proof": True,
            "selected_candidate_indices_zero_based": list(best_indices),
            "selected_candidate_ids": selected_scan_ids,
            "ds_logdet": float(best["score"]),
            "normalized_logdet_per_scan": float(best["normalized_logdet_per_scan"]),
            "basin_count": int(best["basin_count"]),
            "one_exchange_moves": int(best["one_exchange_moves"]),
            "two_exchange_moves": int(best["two_exchange_moves"]),
            "search_score_evaluations": int(best["total_scorer_evaluations"]),
            "design_diagnostics": best["design_diagnostics"],
            "information_block_diagnostics": info_diag,
            "selected_geometry_summary": geometry,
        }
        write_json(n_dir / "best_subset.json", result)

        if not args.disable_plots:
            save_selected_plot(
                n_dir / "selected_pose_parameters.png",
                int(N),
                int(trial_index),
                selected_poses,
            )

        flat_summary = {
            "trial_index": int(trial_index),
            "N": int(N),
            "selected_candidate_ids": " ".join(map(str, selected_scan_ids)),
            "ds_logdet": float(best["score"]),
            "normalized_logdet_per_scan": float(best["normalized_logdet_per_scan"]),
            "eff_rank": int(best["design_diagnostics"]["handeye_eff_rank"]),
            "eff_sigma_min": float(best["design_diagnostics"]["sigma_min_eff"]),
            "eff_condition": float(best["design_diagnostics"]["condition_eff"]),
            "rotation_block_condition": info_diag["rotation_block_condition"],
            "translation_block_condition": info_diag[
                "translation_block_condition"
            ],
            "rotation_translation_coupling_normalized": info_diag[
                "rotation_translation_coupling_normalized"
            ],
            **geometry,
        }
        summary_rows.append(flat_summary)

        print(
            f"[trial={trial_index}, N={N}] "
            f"ids={selected_scan_ids} | "
            f"D_s={best['score']:.6g} | "
            f"cond_R={info_diag['rotation_block_condition']:.4g} | "
            f"cond_t={info_diag['translation_block_condition']:.4g} | "
            "coupling="
            f"{info_diag['rotation_translation_coupling_normalized']:.4g}"
        )

    write_rows_csv(trial_dir / "summary_by_N.csv", summary_rows)
    return summary_rows


# ---------------------------------------------------------------------------
# Cross-trial aggregation
# ---------------------------------------------------------------------------


def aggregate_across_trials(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    output = []
    for N in sorted(set(int(row["N"]) for row in rows)):
        group = [row for row in rows if int(row["N"]) == N]

        metric_names = (
            "ds_logdet",
            "eff_sigma_min",
            "eff_condition",
            "rotation_block_condition",
            "translation_block_condition",
            "rotation_translation_coupling_normalized",
            "uv_centroid_radius_mm",
            "uv_radius_mean_mm",
            "uv_radius_std_mm",
            "depth_mean_mm",
            "depth_std_mm",
            "tilt_mean_deg",
            "tilt_std_deg",
            "azimuth_circular_span_deg",
            "azimuth_resultant_length",
            "roll_circular_span_deg",
            "roll_resultant_length",
        )

        row_out = {
            "N": int(N),
            "trial_count": int(len(group)),
        }

        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in group],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                row_out[f"{metric}_mean"] = float("nan")
                row_out[f"{metric}_median"] = float("nan")
                row_out[f"{metric}_std"] = float("nan")
            else:
                row_out[f"{metric}_mean"] = float(np.mean(finite))
                row_out[f"{metric}_median"] = float(np.median(finite))
                row_out[f"{metric}_std"] = float(np.std(finite))

        output.append(row_out)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact-LHS simulation pool -> fixed-GT D_s finite-N subset search "
            "-> repeated geometric-pattern analysis."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/sim/lhs_ds_pattern"),
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)

    # Candidate pool / synthetic profile.
    parser.add_argument("--candidate-pool-size", type=int, default=300)
    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--tangent-range-mm", type=float, default=100.0)
    parser.add_argument(
        "--profile-depth-range-mm",
        type=float,
        nargs=2,
        default=(60.0, 150.0),
        help="Allowed sensor-frame Z range for every profile point.",
    )
    parser.add_argument(
        "--candidate-center-depth-range-mm",
        type=float,
        nargs=2,
        default=(90.0, 120.0),
        help="LHS range for the center-ray target depth d.",
    )
    parser.add_argument(
        "--candidate-target-u-range-mm",
        type=float,
        nargs=2,
        default=(-70.0, 70.0),
    )
    parser.add_argument(
        "--candidate-target-v-range-mm",
        type=float,
        nargs=2,
        default=(-70.0, 70.0),
    )
    parser.add_argument(
        "--candidate-view-tilt-range-deg",
        type=float,
        nargs=2,
        default=(5.0, 80.0),
    )
    parser.add_argument(
        "--candidate-view-azimuth-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument(
        "--candidate-sensor-roll-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
    )

    # Exact-LHS feasibility preservation.
    parser.add_argument("--max-design-attempts", type=int, default=30)
    parser.add_argument("--stratum-jitter-retries", type=int, default=200)

    # Random physical target / GT context. Repeated trials test whether the
    # selected RELATIVE geometry is invariant to global frame / GT changes.
    parser.add_argument(
        "--plane-angle-range-deg",
        type=float,
        nargs=2,
        default=(-15.0, 15.0),
    )
    parser.add_argument(
        "--plane-center-xy-range-mm",
        type=float,
        nargs=2,
        default=(-100.0, 100.0),
    )
    parser.add_argument(
        "--plane-center-z-range-mm",
        type=float,
        nargs=2,
        default=(400.0, 550.0),
    )
    parser.add_argument("--min-abs-plane-normal-z", type=float, default=1e-4)
    parser.add_argument("--verification-atol", type=float, default=1e-8)

    # N-wise search.
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=(5, 6, 7, 8, 10, 12),
    )
    parser.add_argument("--random-starts", type=int, default=100)
    parser.add_argument("--improvement-tol", type=float, default=1e-10)
    parser.add_argument(
        "--two-exchange-top-k",
        type=int,
        default=3,
        help=(
            "Refine only the top K unique 1-exchange optima with exhaustive "
            "2-exchange. Keep this small for candidate pools ~300-500."
        ),
    )
    parser.add_argument("--save-top-k", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=10)

    # Fixed-GT D_s Jacobian.
    parser.add_argument(
        "--handeye-characteristic-length-mm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--plane-characteristic-length-mm",
        type=float,
        default=100.0,
    )
    parser.add_argument("--design-step-mm", type=float, default=1e-3)
    parser.add_argument("--design-eigen-rtol", type=float, default=1e-10)
    parser.add_argument(
        "--design-points-per-scan",
        type=int,
        default=0,
        help="0 uses all synthetic profile points.",
    )

    parser.add_argument("--disable-plots", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.candidate_pool_size < 5:
        raise ValueError("--candidate-pool-size must be >= 5")
    if args.profile_points < 2:
        raise ValueError("--profile-points must be >= 2")
    if args.profile_half_width_mm <= 0.0:
        raise ValueError("--profile-half-width-mm must be positive")
    if args.tangent_range_mm <= 0.0:
        raise ValueError("--tangent-range-mm must be positive")
    if args.max_design_attempts < 1:
        raise ValueError("--max-design-attempts must be >= 1")
    if args.stratum_jitter_retries < 1:
        raise ValueError("--stratum-jitter-retries must be >= 1")
    if args.random_starts < 1:
        raise ValueError("--random-starts must be >= 1")
    if args.two_exchange_top_k < 0:
        raise ValueError("--two-exchange-top-k must be >= 0")
    if args.save_top_k < 0:
        raise ValueError("--save-top-k must be >= 0")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.design_step_mm <= 0.0:
        raise ValueError("--design-step-mm must be positive")
    if not (0.0 < args.design_eigen_rtol < 1.0):
        raise ValueError("--design-eigen-rtol must lie in (0,1)")
    if args.design_points_per_scan < 0:
        raise ValueError("--design-points-per-scan must be >= 0")
    if any(N < 1 for N in args.n_values):
        raise ValueError("every --n-values entry must be positive")
    if max(args.n_values) > args.candidate_pool_size:
        raise ValueError("N cannot exceed candidate-pool-size")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fair_config, candidate_config = make_simulation_configs(args)

    write_json(
        args.output_dir / "settings.json",
        {
            "args": vars(args),
            "fair_config": asdict(fair_config),
            "candidate_config": asdict(candidate_config),
            "important_convention": {
                "candidate_parameters": PARAMETER_NAMES,
                "view_pose": dict(uniform.VIEW_POSE_CONVENTION),
                "canonical_output_aliases": {
                    "view_u_mm": "target_u_mm",
                    "view_v_mm": "target_v_mm",
                    "view_distance_mm": "center_depth_mm",
                    "view_roll_deg": "sensor_roll_deg",
                },
                "jacobian_linearization": "exact simulation GT",
                "plane_truth_usage": (
                    "linearization point only; plane normal/offset remain "
                    "nuisance parameters in D_s"
                ),
                "candidate_design": (
                    "exact LHS; infeasible within-stratum jitter is retried, "
                    "and an impossible stratum combination regenerates the "
                    "entire LHS"
                ),
            },
        },
    )

    all_rows: list[dict] = []

    for trial_index in range(args.trials):
        rows = run_trial(
            args,
            fair_config,
            candidate_config,
            trial_index,
        )
        all_rows.extend(rows)

        # Keep interruption-safe global tables.
        write_rows_csv(
            args.output_dir / "all_trial_N_results.csv",
            all_rows,
        )
        write_rows_csv(
            args.output_dir / "aggregate_by_N.csv",
            aggregate_across_trials(all_rows),
        )

    print("\n=== DONE ===")
    print(f"results: {args.output_dir}")
    print(f"all trial/N: {args.output_dir / 'all_trial_N_results.csv'}")
    print(f"aggregate : {args.output_dir / 'aggregate_by_N.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
