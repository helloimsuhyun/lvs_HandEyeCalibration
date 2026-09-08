#!/usr/bin/env python3
"""Exhaustive 630-way D3 distance assignment study at fixed N=9 geometry."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fixed_translation_rotation_geometry import run as fixed  # noqa: E402


SCHEMA = "laser_handeye.distance_assignment_exhaustive_n9"
SCHEMA_VERSION = 1
FEATURES = (
    "abs_tilt_group_mean_difference",
    "abs_corr_d_bz",
    "R2_uv",
    "m_uv_norm",
    "m_bxy_norm",
    "m_b_norm",
    "R2_b",
    "R2_combined",
    "projection_norm",
    "retained_delta_JR_fro",
    "retained_ratio",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("geometry", "calibration", "analysis", "all"), required=True)
    parser.add_argument("--calibration-mode", choices=("none", "selected", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    # Analysis rows may deliberately use different predictor sets.  Build a
    # stable union schema so later-row coefficients are not dropped or rejected.
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_config(config: dict[str, Any]) -> dict[str, Any]:
    source = _load_json(_resolve(config["source_config"]))
    if int(source["N"]) != 9:
        raise ValueError("source experiment must use N=9")
    return source


def enumerate_assignments(config: dict[str, Any]) -> list[np.ndarray]:
    base = np.asarray(config["distance_multiset_mm"], dtype=float)
    if len(base) != 9:
        raise ValueError("distance multiset must contain 9 values")
    assignments: list[np.ndarray] = []
    positions = tuple(range(9))
    for mid in positions:
        remaining = tuple(index for index in positions if index != mid)
        for near in itertools.combinations(remaining, 4):
            distance = np.full(9, 150.0)
            distance[list(near)] = 60.0
            distance[mid] = 105.0
            assignments.append(distance)
    keys = {tuple(item.tolist()) for item in assignments}
    if len(assignments) != 630 or len(keys) != 630:
        raise RuntimeError("unique assignment enumeration did not produce 630 designs")
    for distance in assignments:
        values, counts = np.unique(distance, return_counts=True)
        if not np.array_equal(values, [60.0, 105.0, 150.0]) or not np.array_equal(counts, [4, 1, 4]):
            raise RuntimeError("distance histogram changed")
        if not np.isclose(np.mean(distance), 105.0, atol=1e-12):
            raise RuntimeError("mean distance changed")
    return assignments


def _design(
    assignment_id: int,
    distance: np.ndarray,
    context: fixed.ExperimentContext,
    source: dict[str, Any],
    radius: float,
) -> fixed.PoseDesign:
    angles = 2.0 * np.pi * np.arange(9) / 9.0
    return fixed._make_design(
        context,
        source,
        design_id=int(assignment_id),
        experiment="distance_assignment_exhaustive_N9",
        condition=f"assignment_{assignment_id:03d}",
        block_or_shift="",
        independent_value=float(assignment_id),
        gamma_deg=np.zeros(9),
        u_mm=radius * np.cos(angles),
        v_mm=radius * np.sin(angles),
        d_mm=np.asarray(distance, dtype=float),
    )


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) < 1e-15 or np.std(right) < 1e-15:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _r2_projection(target: np.ndarray, predictors: np.ndarray) -> tuple[float, float]:
    y = np.asarray(target, dtype=float)
    x = np.asarray(predictors, dtype=float)
    y = y - np.mean(y)
    x = x - np.mean(x, axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=0)
    keep = norms > 1e-12
    x = x[:, keep] / norms[keep]
    if x.shape[1] == 0 or np.linalg.norm(y) < 1e-15:
        return 0.0, 0.0
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    tolerance = max(1e-12, float(singular[0]) * 1e-12)
    q = u[:, singular > tolerance]
    projected = q @ (q.T @ y)
    ratio = float(np.linalg.norm(projected) / np.linalg.norm(y))
    return float(min(max(ratio * ratio, 0.0), 1.0)), ratio


def _normalized_moment(delta: np.ndarray, matrix: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    moment = matrix.T @ delta
    denominator = np.linalg.norm(delta) * np.linalg.norm(matrix, ord="fro")
    return float(np.linalg.norm(moment) / denominator) if denominator > 0 else 0.0


def _features(design: fixed.PoseDesign) -> dict[str, Any]:
    d = design.d_mm
    delta = d - np.mean(d)
    b = design.b_actual
    low = np.isclose(design.theta_deg, 10.0)
    high = np.isclose(design.theta_deg, 55.0)
    uv = np.column_stack([design.u_mm, design.v_mm])
    r2_uv, _ = _r2_projection(delta, uv)
    r2_b, _ = _r2_projection(delta, b)
    combined = np.column_stack([b, uv])
    r2_combined, projection_norm = _r2_projection(delta, combined)
    moment_uv = uv.T @ delta
    moment_bxy = b[:, :2].T @ delta
    moment_b = b.T @ delta
    tilt_difference = float(np.mean(d[high]) - np.mean(d[low]))
    return {
        "distance_vector": " ".join(f"{value:g}" for value in d),
        **{f"d_pose_{index}": float(value) for index, value in enumerate(d)},
        "mean_d_low_tilt": float(np.mean(d[low])),
        "mean_d_high_tilt": float(np.mean(d[high])),
        "std_d_low_tilt": float(np.std(d[low])),
        "std_d_high_tilt": float(np.std(d[high])),
        "n_near_low": int(np.sum(d[low] == 60.0)),
        "n_mid_low": int(np.sum(d[low] == 105.0)),
        "n_far_low": int(np.sum(d[low] == 150.0)),
        "n_near_high": int(np.sum(d[high] == 60.0)),
        "n_mid_high": int(np.sum(d[high] == 105.0)),
        "n_far_high": int(np.sum(d[high] == 150.0)),
        "tilt_group_mean_difference": tilt_difference,
        "abs_tilt_group_mean_difference": abs(tilt_difference),
        "corr_d_theta": _corr(d, design.theta_deg),
        "corr_d_bx": _corr(d, b[:, 0]),
        "corr_d_by": _corr(d, b[:, 1]),
        "corr_d_bz": _corr(d, b[:, 2]),
        "abs_corr_d_bz": abs(_corr(d, b[:, 2])),
        "m_bz": float(delta @ b[:, 2] / (np.linalg.norm(delta) * np.linalg.norm(b[:, 2]))),
        "corr_d_u": _corr(d, design.u_mm),
        "corr_d_v": _corr(d, design.v_mm),
        "m_uv_x_raw": float(moment_uv[0]),
        "m_uv_y_raw": float(moment_uv[1]),
        "m_uv_norm": _normalized_moment(delta, uv),
        "R2_uv": r2_uv,
        "m_bxy_x_raw": float(moment_bxy[0]),
        "m_bxy_y_raw": float(moment_bxy[1]),
        "m_bxy_norm": _normalized_moment(delta, b[:, :2]),
        "m_b_x_raw": float(moment_b[0]),
        "m_b_y_raw": float(moment_b[1]),
        "m_b_z_raw": float(moment_b[2]),
        "m_b_norm": _normalized_moment(delta, b),
        "R2_b": r2_b,
        "R2_combined": r2_combined,
        "projection_norm": projection_norm,
    }


def _delta_jacobian_metrics(
    matrices: dict[str, np.ndarray], baseline: dict[str, np.ndarray]
) -> dict[str, Any]:
    delta = matrices["J_X"][:, :3] - baseline["J_X"][:, :3]
    nuisance = np.column_stack([matrices["J_X"][:, 3:6], matrices["J_pi"]])
    retained = delta - nuisance @ np.linalg.lstsq(nuisance, delta, rcond=1e-12)[0]
    raw_norm = float(np.linalg.norm(delta, ord="fro"))
    retained_norm = float(np.linalg.norm(retained, ord="fro"))
    singular = np.linalg.svd(retained, compute_uv=False)
    return {
        "Delta_J_R_fro": raw_norm,
        "retained_delta_JR_fro": retained_norm,
        "retained_ratio": retained_norm / raw_norm if raw_norm > 0 else 0.0,
        "retained_delta_JR_sv1": float(singular[0]),
        "retained_delta_JR_sv2": float(singular[1]),
        "retained_delta_JR_sv3": float(singular[2]),
    }


def _percentile(values: np.ndarray, value: float, higher: bool = True) -> float:
    if higher:
        return float(100.0 * np.mean(values <= value))
    return float(100.0 * np.mean(values >= value))


def _anchor_map(assignments: Sequence[np.ndarray], config: dict[str, Any]) -> dict[int, int]:
    base = np.asarray(config["distance_multiset_mm"], dtype=float)
    lookup = {tuple(item.tolist()): index for index, item in enumerate(assignments)}
    return {shift: lookup[tuple(np.roll(base, shift).tolist())] for shift in range(9)}


def _mapping_rows(designs: Sequence[fixed.PoseDesign], anchors: dict[int, int]) -> list[dict[str, Any]]:
    reverse = {assignment: shift for shift, assignment in anchors.items()}
    rows: list[dict[str, Any]] = []
    for design in designs:
        for pose_id in range(9):
            rows.append({
                "assignment_id": design.design_id,
                "previous_shift_id": reverse.get(design.design_id, ""),
                "pose_id": pose_id,
                "theta_deg": float(design.theta_deg[pose_id]),
                "psi_b_deg": float(design.psi_b_deg[pose_id]),
                "gamma_deg": float(design.gamma_deg[pose_id]),
                "u_mm": float(design.u_mm[pose_id]),
                "v_mm": float(design.v_mm[pose_id]),
                "b_x": float(design.b_actual[pose_id, 0]),
                "b_y": float(design.b_actual[pose_id, 1]),
                "b_z": float(design.b_actual[pose_id, 2]),
                "d_mm": float(design.d_mm[pose_id]),
            })
    return rows


def _correlations(rows: Sequence[dict[str, Any]], outcomes: Sequence[str], features: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for outcome in outcomes:
        y = np.asarray([float(row[outcome]) for row in rows])
        for feature in features:
            x = np.asarray([float(row[feature]) for row in rows])
            result = spearmanr(x, y)
            output.append({
                "outcome": outcome,
                "feature": feature,
                "spearman_rho": float(result.correlation),
                "p_value_descriptive": float(result.pvalue),
                "assignment_count": len(rows),
                "population_note": "complete combinatorial population; p-value is descriptive",
            })
    return output


def _rank_table(rows: Sequence[dict[str, Any]], key: str, best_high: bool) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=best_high)[:20]
    return [dict(rank=index + 1, rank_metric=key, **row) for index, row in enumerate(ordered)]


def _rank_mapping_table(
    rows: Sequence[dict[str, Any]],
    mapping_rows: Sequence[dict[str, Any]],
    key: str,
    best_high: bool,
) -> list[dict[str, Any]]:
    """Expand each ranked assignment into nine directly inspectable pose rows."""
    ranked = _rank_table(rows, key, best_high)
    mapping_by_assignment: dict[int, list[dict[str, Any]]] = {}
    for pose in mapping_rows:
        mapping_by_assignment.setdefault(int(pose["assignment_id"]), []).append(pose)
    output: list[dict[str, Any]] = []
    for assignment in ranked:
        assignment_id = int(assignment["assignment_id"])
        for pose in sorted(mapping_by_assignment[assignment_id],key=lambda row:int(row["pose_id"])):
            output.append({
                "rank":assignment["rank"],
                "rank_metric":key,
                "rank_metric_value":float(assignment[key]),
                "assignment_id":assignment_id,
                "distance_vector":assignment["distance_vector"],
                **pose,
            })
    return output


def _scatter(rows: Sequence[dict[str, Any]], x_key: str, y_key: str, path: Path, anchors: dict[int, int]) -> None:
    x = np.asarray([float(row[x_key]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    result = spearmanr(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    ax.scatter(x, y, s=20, alpha=0.45, color="#4C78A8")
    lookup = {int(row["assignment_id"]): row for row in rows}
    for shift in (0, 2, 4):
        row = lookup[anchors[shift]]
        ax.scatter(float(row[x_key]), float(row[y_key]), s=85, marker="*", label=f"shift {shift}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Spearman rho={float(result.correlation):+.3f}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _layout_plot(rows: Sequence[dict[str, Any]], assignments: Sequence[np.ndarray], source: dict[str, Any], path: Path) -> None:
    best = sorted(rows, key=lambda row: float(row["H_R_given_t_logdet"]), reverse=True)[:4]
    worst = sorted(rows, key=lambda row: float(row["H_R_given_t_logdet"]))[:4]
    selected = best + worst
    theta, psi = fixed._baseline_angles(source)
    angles = 2.0 * np.pi * np.arange(9) / 9.0
    uv = 40.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    b = fixed._target_b(theta, psi)
    colors = {60.0: "#2C7BB6", 105.0: "#FEE08B", 150.0: "#D7191C"}
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), sharex=True, sharey=True)
    for index, (axis, row) in enumerate(zip(axes.flat, selected, strict=True)):
        distance = assignments[int(row["assignment_id"])]
        for pose in range(9):
            marker = "o" if np.isclose(theta[pose], 10.0) else "s"
            axis.scatter(uv[pose, 0], uv[pose, 1], s=125, marker=marker, color=colors[distance[pose]], edgecolor="black")
            axis.arrow(uv[pose, 0], uv[pose, 1], 8*b[pose,0], 8*b[pose,1], width=.35, head_width=2.2, color="#333333")
            axis.text(uv[pose, 0]+1.5, uv[pose, 1]+1.5, str(pose), fontsize=8)
        axis.axhline(0, color="#cccccc", lw=.6); axis.axvline(0, color="#cccccc", lw=.6)
        axis.set_aspect("equal")
        kind = "TOP" if index < 4 else "BOTTOM"
        axis.set_title(f"{kind} A{int(row['assignment_id']):03d}\nlogdet={float(row['H_R_given_t_logdet']):.4f}")
        axis.grid(alpha=.18)
    fig.supxlabel("u [mm]"); fig.supylabel("v [mm]")
    fig.suptitle("Distance assignment layouts: circle=10° tilt, square=55° tilt; blue/cream/red=60/105/150 mm")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_geometry(config: dict[str, Any], output: Path) -> None:
    path = output / "all_630_assignment_geometry.csv"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    source = _source_config(config)
    context = fixed._context(source)
    assignments = enumerate_assignments(config)
    anchors = _anchor_map(assignments, config)
    common = _design(-1, np.full(9, 105.0), context, source, float(config["uv_radius_mm"]))
    _, baseline_matrices = fixed._geometry(common, context, source)
    designs: list[fixed.PoseDesign] = []
    rows: list[dict[str, Any]] = []
    baseline_b: np.ndarray | None = None
    reverse_anchor = {assignment: shift for shift, assignment in anchors.items()}
    started = time.perf_counter()
    for assignment_id, distance in enumerate(assignments):
        design = _design(assignment_id, distance, context, source, float(config["uv_radius_mm"]))
        geometry, matrices = fixed._geometry(design, context, source)
        if baseline_b is None:
            baseline_b = matrices["b"].copy()
        b_drift = float(np.max(np.linalg.norm(matrices["b"] - baseline_b, axis=1)))
        if b_drift > float(source["numerics"]["identity_tolerance"]):
            raise RuntimeError(f"b changed at assignment {assignment_id}: {b_drift}")
        geometry.update(_features(design))
        geometry.update(_delta_jacobian_metrics(matrices, baseline_matrices))
        geometry["assignment_id"] = assignment_id
        geometry["previous_shift_id"] = reverse_anchor.get(assignment_id, "")
        geometry["max_b_drift_from_baseline"] = b_drift
        rows.append(geometry)
        designs.append(design)
        if (assignment_id + 1) % 50 == 0 or assignment_id == 629:
            elapsed = time.perf_counter() - started
            eta = elapsed * (630 - assignment_id - 1) / (assignment_id + 1)
            print(f"[geometry {assignment_id+1}/630] elapsed={fixed.phase1._format_duration(elapsed)} ETA={fixed.phase1._format_duration(eta)}", flush=True)
    fixed._assert_fixed_cb(rows, rows[0], source)
    _write_csv(path, rows)
    mapping_rows=_mapping_rows(designs, anchors)
    _write_csv(output / "assignment_pose_mapping.csv", mapping_rows)
    correlations = _correlations(rows, ("H_R_given_t_logdet", "H_R_given_t_trace_inv", "H_R_given_t_lambda_min"), FEATURES)
    _write_csv(output / "geometry_correlations.csv", correlations)

    source_geometry = _read_csv(_resolve(config["source_results"]) / "r2_distance_diversity_geometry.csv")
    source_d3 = {int(row["block_or_shift"]): row for row in source_geometry if row["condition"] == "D3"}
    anchor_rows: list[dict[str, Any]] = []
    for shift, assignment_id in anchors.items():
        row = rows[assignment_id]
        old = source_d3[shift]
        differences = {metric: abs(float(row[metric]) - float(old[metric])) for metric in ("H_R_given_t_lambda_min", "H_R_given_t_logdet", "H_R_given_t_trace_inv", "H_eff_logdet")}
        if max(differences.values()) > 1e-9:
            raise RuntimeError(f"anchor shift {shift} failed source reproduction: {differences}")
        anchor_rows.append({
            "previous_shift_id": shift,
            "assignment_id": assignment_id,
            "distance_vector": row["distance_vector"],
            **row,
            "logdet_percentile_high": _percentile(np.asarray([float(x["H_R_given_t_logdet"]) for x in rows]), float(row["H_R_given_t_logdet"])),
            "trace_inv_percentile_best": _percentile(np.asarray([float(x["H_R_given_t_trace_inv"]) for x in rows]), float(row["H_R_given_t_trace_inv"]), higher=False),
            "lambda_min_percentile_high": _percentile(np.asarray([float(x["H_R_given_t_lambda_min"]) for x in rows]), float(row["H_R_given_t_lambda_min"])),
            "tilt_coupling_percentile_low": _percentile(np.asarray([float(x["abs_tilt_group_mean_difference"]) for x in rows]), float(row["abs_tilt_group_mean_difference"]), higher=False),
            "R2_uv_percentile_low": _percentile(np.asarray([float(x["R2_uv"]) for x in rows]), float(row["R2_uv"]), higher=False),
            "R2_combined_percentile_low": _percentile(np.asarray([float(x["R2_combined"]) for x in rows]), float(row["R2_combined"]), higher=False),
            **{f"source_abs_difference_{key}": value for key, value in differences.items()},
        })
    _write_csv(output / "anchor_shift_report.csv", anchor_rows)
    tables = {
        "top20_logdet.csv": _rank_mapping_table(rows,mapping_rows,"H_R_given_t_logdet",True),
        "bottom20_logdet.csv": _rank_mapping_table(rows,mapping_rows,"H_R_given_t_logdet",False),
        "top20_trace_inv_best.csv": _rank_mapping_table(rows,mapping_rows,"H_R_given_t_trace_inv",False),
        "top20_low_R2_combined.csv": _rank_mapping_table(rows,mapping_rows,"R2_combined",False),
        "top20_high_R2_combined.csv": _rank_mapping_table(rows,mapping_rows,"R2_combined",True),
        "top20_low_tilt_coupling.csv": _rank_mapping_table(rows,mapping_rows,"abs_tilt_group_mean_difference",False),
        "top20_high_tilt_coupling.csv": _rank_mapping_table(rows,mapping_rows,"abs_tilt_group_mean_difference",True),
    }
    for name, table in tables.items(): _write_csv(output / name, table)
    for x_key, y_key, name in (
        ("abs_tilt_group_mean_difference", "H_R_given_t_logdet", "logdet_vs_tilt_coupling.png"),
        ("R2_uv", "H_R_given_t_logdet", "logdet_vs_R2_uv.png"),
        ("R2_combined", "H_R_given_t_logdet", "logdet_vs_R2_combined.png"),
        ("R2_combined", "H_R_given_t_trace_inv", "trace_inv_vs_R2_combined.png"),
        ("m_b_norm", "H_R_given_t_logdet", "logdet_vs_m_b_norm.png"),
    ):
        _scatter(rows, x_key, y_key, figures / name, anchors)
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); ax.hist([float(row["H_R_given_t_logdet"]) for row in rows], bins=35, color="#4C78A8", alpha=.85); ax.set_xlabel("H_R_given_t_logdet"); ax.set_ylabel("assignment count"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / "conditional_logdet_histogram.png", dpi=180); plt.close(fig)
    _layout_plot(rows, assignments, source, figures / "top_bottom_assignment_layouts.png")
    _write_json(output / "geometry_manifest.json", {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "stage": "geometry", "status": "complete",
        "assignment_count": len(rows), "enumeration": "mid_position ascending, then lexicographic 4-near combinations; remaining positions far",
        "anchor_assignment_ids": anchors, "fixed_b_verified": True,
        "files": {path.name: _sha256(path), "assignment_pose_mapping.csv": _sha256(output / "assignment_pose_mapping.csv")},
    })
    print(f"Geometry complete: {output}", flush=True)


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_SOURCE: dict[str, Any] | None = None
_WORKER_CONTEXT: fixed.ExperimentContext | None = None
_WORKER_SCENARIOS: dict[str, np.ndarray] | None = None


def _worker_init(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_SOURCE, _WORKER_CONTEXT, _WORKER_SCENARIOS
    _WORKER_CONFIG = config
    _WORKER_SOURCE = _source_config(config)
    _WORKER_CONTEXT = fixed._context(_WORKER_SOURCE)
    phase2a_dir = _resolve(_WORKER_SOURCE["phase2a_dir"])
    with np.load(phase2a_dir / "scenario_bank.npz", allow_pickle=False) as archive:
        _WORKER_SCENARIOS = {name: archive[name].copy() for name in archive.files}


def _calibrate_assignment(task: tuple[int, list[float]]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    assignment_id, distance_list = task
    assert _WORKER_CONFIG is not None and _WORKER_SOURCE is not None and _WORKER_CONTEXT is not None and _WORKER_SCENARIOS is not None
    config, source, context, scenarios = _WORKER_CONFIG, _WORKER_SOURCE, _WORKER_CONTEXT, _WORKER_SCENARIOS
    design = _design(assignment_id, np.asarray(distance_list), context, source, float(config["uv_radius_mm"]))
    phase2a_manifest = context.phase2a_manifest
    noise = phase2a_manifest["config"]["noise"]
    repeats = int(noise["repeats_per_environment"])
    classification = {
        "classification_translation_mm": float(phase2a_manifest["config"]["classification"]["translation_mm"]),
        "classification_rotation_deg": float(phase2a_manifest["config"]["classification"]["rotation_deg"]),
    }
    run_rows: list[dict[str, Any]] = []
    for environment_value in scenarios["environment_ids"]:
        environment = int(environment_value)
        resolved = fixed.phase2a._solver_resolved(context.phase1_manifest, classification, scenarios["T_init"][environment])
        for repeat in range(repeats):
            scans = fixed._paired_noisy_scans(design, environment=environment, repeat=repeat, noise_seed=int(source["seed"]), noise_std=float(noise["std_mm"]), noise_axis=str(noise["axis"]))
            row = fixed.phase1._run_one_calibration(resolved=resolved, scans=scans, scan_count=9, subset_id=assignment_id, repeat=repeat, candidate_ids=np.arange(9, dtype=np.int64))
            row.update(source="distance_assignment_exhaustive_N9", assignment_id=assignment_id, scenario_id=environment * repeats + repeat, environment_id=environment, noise_repeat=repeat)
            run_rows.append(row)
    rotation = np.asarray([float(row["rotation_error_deg"]) for row in run_rows])
    translation = np.asarray([float(row["translation_error_mm"]) for row in run_rows])
    residual = np.asarray([float(row["nonlinear_final_rms_mm"]) for row in run_rows])
    finite_rotation = rotation[np.isfinite(rotation)]; finite_translation = translation[np.isfinite(translation)]
    good = np.isfinite(rotation) & np.isfinite(residual) & (rotation <= .1)
    wrong = np.isfinite(rotation) & np.isfinite(residual) & (rotation > 90.)
    summary = {
        "assignment_id": assignment_id, "runs": len(run_rows),
        "rotation_error_median_deg": float(np.median(finite_rotation)), "rotation_error_p90_deg": float(np.quantile(finite_rotation,.9)),
        "translation_error_median_mm": float(np.median(finite_translation)), "translation_error_p90_mm": float(np.quantile(finite_translation,.9)),
        "full_success_rate": float(np.mean([bool(row["passes_success_threshold"]) for row in run_rows])),
        # Keep finite wrong-solution branches separate from solver failures.
        # In particular, +inf must not be double-counted as a >90 deg branch.
        "wrong_branch_rate_rotation_gt_90": float(np.mean(np.isfinite(rotation) & (rotation > 90.))),
        "nonfinite_result_rate": float(np.mean(~(np.isfinite(rotation) & np.isfinite(translation) & np.isfinite(residual)))),
        "nonfinite_result_count": int(np.sum(~(np.isfinite(rotation) & np.isfinite(translation) & np.isfinite(residual)))),
        "rotation_le_0p1_rate": float(np.mean(np.isfinite(rotation) & (rotation <= .1))),
        "good_branch_residual_median_mm": float(np.median(residual[good])) if np.any(good) else math.nan,
        "wrong_branch_residual_median_mm": float(np.median(residual[wrong])) if np.any(wrong) else math.nan,
        "catastrophic_failure_count": int(sum(bool(row["catastrophic_failure"]) for row in run_rows)),
    }
    return assignment_id, run_rows, summary


def _selected_assignment_ids(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> tuple[list[int], set[int]]:
    ordered = sorted(rows, key=lambda row: float(row["H_R_given_t_logdet"]))
    bins = np.array_split(np.arange(len(ordered)), int(config["selected_quantile_bins"]))
    rng = np.random.default_rng(int(config["seed"]))
    chosen: set[int] = set()
    for indices in bins:
        picked = rng.choice(indices, size=min(int(config["selected_per_bin"]), len(indices)), replace=False)
        chosen.update(int(ordered[int(index)]["assignment_id"]) for index in picked)
    anchors = {int(row["assignment_id"]) for row in _read_csv(_resolve(config["output_dir"]) / "anchor_shift_report.csv") if int(row["previous_shift_id"]) in (0,2,4)}
    anchor_only = anchors - chosen
    return sorted(chosen | anchors), anchor_only


def run_calibration(config: dict[str, Any], output: Path, mode: str, workers: int, resume: bool) -> None:
    if mode == "none":
        return
    if not (output / "geometry_manifest.json").exists():
        raise RuntimeError("geometry must complete before calibration")
    runs_path = output / f"calibration_runs_{mode}.csv"
    summary_path = output / f"calibration_summary_{mode}.csv"
    manifest_path = output / f"calibration_manifest_{mode}.json"
    if manifest_path.exists():
        raise FileExistsError(f"calibration already complete: {manifest_path}")
    if summary_path.exists() and not resume:
        raise FileExistsError("calibration checkpoint exists; use --resume")
    geometry = _read_csv(output / "all_630_assignment_geometry.csv")
    assignments = enumerate_assignments(config)
    if mode == "all":
        selected = list(range(630)); anchor_only: set[int] = set()
    else:
        selected, anchor_only = _selected_assignment_ids(geometry, config)
    completed = {int(row["assignment_id"]) for row in _read_csv(summary_path)} if summary_path.exists() else set()
    remaining = [assignment for assignment in selected if assignment not in completed]
    started = time.perf_counter()
    tasks = [(assignment, assignments[assignment].tolist()) for assignment in remaining]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(config,)) as executor:
        futures = {executor.submit(_calibrate_assignment, task): task[0] for task in tasks}
        done = 0
        for future in as_completed(futures):
            assignment_id, run_rows, summary = future.result()
            summary["selection_role"] = "descriptive_anchor" if assignment_id in anchor_only else ("confirmatory_stratified" if mode == "selected" else "exhaustive")
            _append_csv(runs_path, run_rows); _append_csv(summary_path, [summary])
            done += 1
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(tasks)-done) / max(done,1)
            print(f"[calibration {done}/{len(tasks)}] assignment={assignment_id:03d}; total complete={len(completed)+done}/{len(selected)}; elapsed={fixed.phase1._format_duration(elapsed)} ETA={fixed.phase1._format_duration(eta)}", flush=True)
    source = _source_config(config)
    scenario_path = _resolve(source["phase2a_dir"]) / "scenario_bank.npz"
    _write_json(manifest_path, {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "stage": "calibration", "status": "complete", "mode": mode,
        "assignment_count": len(selected), "runs_per_assignment": 90, "calibration_runs": len(selected)*90,
        "workers": workers, "scenario_bank_source": str(scenario_path), "scenario_bank_sha256": _sha256(scenario_path),
        "scenario_pairing": "same environment, noise repeat, and pose-slot noise seed for every assignment",
        "selected_assignment_ids": selected, "descriptive_anchor_only_ids": sorted(anchor_only),
    })


def _ols(rows: Sequence[dict[str, Any]], outcome: str, predictors: Sequence[str]) -> dict[str, Any]:
    y = np.asarray([float(row[outcome]) for row in rows]); x = np.column_stack([[float(row[key]) for row in rows] for key in predictors])
    x_mean=x.mean(0); x_std=x.std(0); xz=(x-x_mean)/np.where(x_std>1e-15,x_std,1.0); yz=y-y.mean(); design=np.column_stack([np.ones(len(y)),xz]); coefficients=np.linalg.lstsq(design,y,rcond=None)[0]; fitted=design@coefficients
    r2=1-float(np.sum((y-fitted)**2))/float(np.sum((y-y.mean())**2))
    return {"outcome":outcome,"predictors":" ".join(predictors),"intercept":float(coefficients[0]),**{f"beta_z_{key}":float(value) for key,value in zip(predictors,coefficients[1:],strict=True)},"R2_exploratory":r2}


def run_analysis(config: dict[str, Any], output: Path, mode: str) -> None:
    if mode == "none":
        raise ValueError("analysis requires selected or all calibration")
    report = output / f"report_{mode}.md"
    if report.exists(): raise FileExistsError(f"refusing to overwrite: {report}")
    geometry = _read_csv(output / "all_630_assignment_geometry.csv")
    calibration = _read_csv(output / f"calibration_summary_{mode}.csv")
    if not (output / f"calibration_manifest_{mode}.json").exists(): raise RuntimeError("calibration is not complete")
    geo = {int(row["assignment_id"]):row for row in geometry}
    merged: list[dict[str,Any]]=[]
    for cal in calibration:
        row=dict(geo[int(cal["assignment_id"])]); row.update(cal); merged.append(row)
    outcomes=("rotation_error_median_deg","rotation_error_p90_deg","wrong_branch_rate_rotation_gt_90","full_success_rate")
    metrics=("H_R_given_t_logdet","H_R_given_t_trace_inv","H_R_given_t_lambda_min",*FEATURES)
    correlations=_correlations(merged,outcomes,metrics)
    _write_csv(output/f"performance_correlations_{mode}.csv",correlations)
    epsilon=1e-6
    regression_rows=[]
    prepared=[]
    for row in merged:
        item=dict(row); rate=float(item["wrong_branch_rate_rotation_gt_90"]); n=float(item["runs"]); clipped=min(max(rate,.5/n),1-.5/n); item["wrong_branch_logit"]=math.log(clipped/(1-clipped)); item["log_rotation_median_plus_epsilon"]=math.log(float(item["rotation_error_median_deg"])+epsilon); prepared.append(item)
    regression_rows.append(_ols(prepared,"wrong_branch_logit",("H_R_given_t_logdet",)))
    regression_rows.append(_ols(prepared,"wrong_branch_logit",("H_R_given_t_logdet","abs_tilt_group_mean_difference","R2_uv")))
    regression_rows.append(_ols(prepared,"log_rotation_median_plus_epsilon",("H_R_given_t_logdet",)))
    for feature in ("abs_tilt_group_mean_difference","R2_uv","R2_b","R2_combined","retained_ratio"):
        regression_rows.append(_ols(prepared,"log_rotation_median_plus_epsilon",("H_R_given_t_logdet",feature)))
    regression_rows.append(_ols(prepared,"log_rotation_median_plus_epsilon",("H_R_given_t_logdet","abs_tilt_group_mean_difference","R2_uv")))
    _write_csv(output/f"exploratory_regressions_{mode}.csv",regression_rows)
    anchors=_read_csv(output/"anchor_shift_report.csv")
    anchor_ids={int(row["assignment_id"]):int(row["previous_shift_id"]) for row in anchors}
    figures=output/"figures"
    for x_key in ("H_R_given_t_logdet","H_R_given_t_trace_inv","H_R_given_t_lambda_min","R2_combined","abs_tilt_group_mean_difference"):
        _scatter(merged,x_key,"wrong_branch_rate_rotation_gt_90",figures/f"{mode}_wrong_branch_vs_{x_key}.png",{shift:aid for aid,shift in anchor_ids.items()})

    # Rank summaries and logdet-decile behavior make the exhaustive population
    # directly inspectable without rerunning calibration.
    _write_csv(output/f"top20_rotation_median_best_{mode}.csv",_rank_table(merged,"rotation_error_median_deg",False))
    _write_csv(output/f"bottom20_rotation_median_worst_{mode}.csv",_rank_table(merged,"rotation_error_median_deg",True))
    ordered=sorted(merged,key=lambda row:float(row["H_R_given_t_logdet"]))
    decile_rows=[]
    for decile, indices in enumerate(np.array_split(np.arange(len(ordered)),10),start=1):
        chunk=[ordered[int(index)] for index in indices]
        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in chunk])
        decile_rows.append({
            "logdet_decile_low_to_high":decile,
            "assignment_count":len(chunk),
            "logdet_min":float(np.min(values("H_R_given_t_logdet"))),
            "logdet_max":float(np.max(values("H_R_given_t_logdet"))),
            "rotation_median_of_assignment_medians_deg":float(np.median(values("rotation_error_median_deg"))),
            "mean_full_success_rate":float(np.mean(values("full_success_rate"))),
            "mean_wrong_branch_rate":float(np.mean(values("wrong_branch_rate_rotation_gt_90"))),
            "catastrophic_failure_count":int(np.sum(values("catastrophic_failure_count"))),
        })
    _write_csv(output/f"logdet_decile_performance_{mode}.csv",decile_rows)

    wrong_rows=[row for row in merged if float(row["wrong_branch_rate_rotation_gt_90"])>0]
    _write_csv(output/f"wrong_branch_assignments_{mode}.csv",sorted(wrong_rows,key=lambda row:float(row["wrong_branch_rate_rotation_gt_90"]),reverse=True))

    # Complement pairs exchange near<->far at every pose and keep the mid pose.
    # Their conditional information is numerically identical, so their outcome
    # difference is a direct diagnostic of what local information cannot explain.
    by_vector={tuple(float(value) for value in row["distance_vector"].split()):row for row in merged}
    complement_rows=[]
    for row in merged:
        vector=tuple(float(value) for value in row["distance_vector"].split())
        complement=tuple(210.0-value for value in vector)
        other=by_vector[complement]
        left=int(row["assignment_id"]); right=int(other["assignment_id"])
        if left>=right:
            continue
        complement_rows.append({
            "pair_id":len(complement_rows),"assignment_a":left,"assignment_b":right,
            "distance_vector_a":row["distance_vector"],"distance_vector_b":other["distance_vector"],
            "conditional_logdet":float(row["H_R_given_t_logdet"]),
            "conditional_logdet_abs_difference":abs(float(row["H_R_given_t_logdet"])-float(other["H_R_given_t_logdet"])),
            "rotation_median_a_deg":float(row["rotation_error_median_deg"]),"rotation_median_b_deg":float(other["rotation_error_median_deg"]),
            "rotation_median_abs_difference_deg":abs(float(row["rotation_error_median_deg"])-float(other["rotation_error_median_deg"])),
            "success_rate_a":float(row["full_success_rate"]),"success_rate_b":float(other["full_success_rate"]),
            "success_rate_abs_difference":abs(float(row["full_success_rate"])-float(other["full_success_rate"])),
            "wrong_branch_rate_a":float(row["wrong_branch_rate_rotation_gt_90"]),"wrong_branch_rate_b":float(other["wrong_branch_rate_rotation_gt_90"]),
            "wrong_branch_rate_abs_difference":abs(float(row["wrong_branch_rate_rotation_gt_90"])-float(other["wrong_branch_rate_rotation_gt_90"])),
        })
    _write_csv(output/f"complement_pair_summary_{mode}.csv",complement_rows)

    corr_lookup={(row["outcome"],row["feature"]):row for row in correlations}
    logdet=np.asarray([float(row["H_R_given_t_logdet"]) for row in geometry]); trace_inv=np.asarray([float(row["H_R_given_t_trace_inv"]) for row in geometry])
    lambda_min=np.asarray([float(row["H_R_given_t_lambda_min"]) for row in geometry])
    best=max(geometry,key=lambda row:float(row["H_R_given_t_logdet"])); worst=min(geometry,key=lambda row:float(row["H_R_given_t_logdet"]))
    merged_by_id={int(row["assignment_id"]):row for row in merged}
    best_perf=merged_by_id[int(best["assignment_id"])]; worst_perf=merged_by_id[int(worst["assignment_id"])]
    def population_stats(key: str) -> tuple[float,float,float]:
        data=np.asarray([float(row[key]) for row in merged])
        return float(np.min(data)),float(np.median(data)),float(np.max(data))
    rot_med=population_stats("rotation_error_median_deg"); rot_p90=population_stats("rotation_error_p90_deg")
    trans_med=population_stats("translation_error_median_mm"); trans_p90=population_stats("translation_error_p90_mm")
    success=population_stats("full_success_rate"); wrong=population_stats("wrong_branch_rate_rotation_gt_90")
    wrong_runs=int(round(sum(float(row["wrong_branch_rate_rotation_gt_90"])*float(row["runs"]) for row in merged)))
    catastrophic_runs=int(sum(int(float(row["catastrophic_failure_count"])) for row in merged))
    baseline_reg=next(row for row in regression_rows if row["outcome"]=="log_rotation_median_plus_epsilon" and row["predictors"]=="H_R_given_t_logdet")
    tilt_uv_reg=next(row for row in regression_rows if row["outcome"]=="log_rotation_median_plus_epsilon" and row["predictors"]=="H_R_given_t_logdet abs_tilt_group_mean_difference R2_uv")
    complement_logdet_max=max(float(row["conditional_logdet_abs_difference"]) for row in complement_rows)
    complement_rot_med=np.median([float(row["rotation_median_abs_difference_deg"]) for row in complement_rows])
    complement_success_med=np.median([float(row["success_rate_abs_difference"]) for row in complement_rows])
    geometry_corr=_read_csv(output/"geometry_correlations.csv")
    def rho(outcome: str, feature: str, table: Sequence[dict[str,Any]]=correlations) -> float:
        return float(next(row for row in table if row["outcome"]==outcome and row["feature"]==feature)["spearman_rho"])
    lines=[
        "# Exhaustive N=9 distance-assignment study", "", f"Status: geometry 630/630 and calibration mode `{mode}` complete ({len(merged)} assignments, {sum(int(row['runs']) for row in calibration)} paired runs).", "",
        "Every design uses the exact same ordered b_i, Cb, u/v circle, gamma=0, mean distance 105 mm, and distance histogram {60x4,105x1,150x4}. Only pose-to-distance assignment changes.", "",
        "## Executive conclusion", "",
        "Distance assignment matters strongly even when distance variance, ordered b_i, Cb, u/v and all pose counts are fixed. Conditional rotation information is a strong predictor of typical converged accuracy, but it does not explain the upper-tail/global-basin failures. The most defensible simple geometric proxy is to avoid a low-frequency u/v distance gradient; the exact design criterion remains conditional trace(inv) or logdet.", "",
        "## 1. How much conditional information changes", "",
        f"- conditional logdet min / median / max: {np.min(logdet):.6f} / {np.median(logdet):.6f} / {np.max(logdet):.6f}",
        f"- conditional trace(inv) min / median / max: {np.min(trace_inv):.6f} / {np.median(trace_inv):.6f} / {np.max(trace_inv):.6f}",
        f"- conditional lambda_min min / median / max: {np.min(lambda_min):.6f} / {np.median(lambda_min):.6f} / {np.max(lambda_min):.6f}",
        f"- logdet span is {np.max(logdet)-np.min(logdet):.6f}, equivalent to a determinant ratio of {math.exp(np.max(logdet)-np.min(logdet)):.2f}x.", "",
        "## 2. Best and worst structures", "",
        f"- best logdet: assignment {best['assignment_id']} = [{best['distance_vector']}], low/high mean distance {float(best['mean_d_low_tilt']):.2f}/{float(best['mean_d_high_tilt']):.2f} mm, R2_uv={float(best['R2_uv']):.3f}, retained ratio={float(best['retained_ratio']):.3f}.",
        f"- worst logdet: assignment {worst['assignment_id']} = [{worst['distance_vector']}], low/high mean distance {float(worst['mean_d_low_tilt']):.2f}/{float(worst['mean_d_high_tilt']):.2f} mm, R2_uv={float(worst['R2_uv']):.3f}, retained ratio={float(worst['retained_ratio']):.3f}.",
        f"- their median rotation errors are {float(best_perf['rotation_error_median_deg']):.5f} and {float(worst_perf['rotation_error_median_deg']):.5f} deg; success rates are {float(best_perf['full_success_rate']):.3f} and {float(worst_perf['full_success_rate']):.3f}.", "",
        "## 3. Is tilt-distance coupling sufficient?", "",
        f"No. |tilt mean difference| vs logdet has rho={rho('H_R_given_t_logdet','abs_tilt_group_mean_difference',geometry_corr):+.3f}, but the worst assignment has exactly zero tilt-group mean difference while the best has {float(best['abs_tilt_group_mean_difference']):.2f} mm. Shift 0 and shift 4 also show that the sign/reversal cannot be reduced to 'large coupling is always bad'.", "",
        "## 4. Does u/v alignment help explain assignment quality?", "",
        f"Yes descriptively: R2_uv vs logdet rho={rho('H_R_given_t_logdet','R2_uv',geometry_corr):+.3f}, the strongest of the simple scalar alignment scores. Lower u/v gradient alignment tends to preserve more conditional information. It is still a proxy, not a causal result.", "",
        "## 5. Is b-direction alignment more direct?", "",
        f"Simple R2_b is weaker (rho={rho('H_R_given_t_logdet','R2_b',geometry_corr):+.3f}) and m_b_norm is nearly uninformative (rho={rho('H_R_given_t_logdet','m_b_norm',geometry_corr):+.3f}). The matrix diagnostic is more direct: retained_delta_JR_fro rho={rho('H_R_given_t_logdet','retained_delta_JR_fro',geometry_corr):+.3f}, retained_ratio rho={rho('H_R_given_t_logdet','retained_ratio',geometry_corr):+.3f}.", "",
        "## 6. Which local information metric best predicts calibration?", "",
        f"For median rotation error, Spearman is logdet {rho('rotation_error_median_deg','H_R_given_t_logdet'):+.3f}, trace(inv) {rho('rotation_error_median_deg','H_R_given_t_trace_inv'):+.3f}, and lambda_min {rho('rotation_error_median_deg','H_R_given_t_lambda_min'):+.3f}. Trace(inv) is marginally strongest, but logdet is essentially tied and much stronger than lambda_min.",
        f"For full success rate the corresponding correlations are {rho('full_success_rate','H_R_given_t_logdet'):+.3f}, {rho('full_success_rate','H_R_given_t_trace_inv'):+.3f}, and {rho('full_success_rate','H_R_given_t_lambda_min'):+.3f}.", "",
        "## 7. A simple assignment rule", "",
        "Under fixed maximal distance variance, distribute near/far distances around the u/v circle so that distance is poorly explained by one linear u/v gradient (low R2_uv), rather than grouping distances into one side of the plane. Use conditional trace(inv)/logdet to rank ties. This is the simplest supported proxy, not a final universal rule.", "",
        "## 8. Simple balance versus exact conditional information", "",
        f"A standardized OLS for log(median rotation error) gives R2={float(baseline_reg['R2_exploratory']):.4f} with logdet alone and R2={float(tilt_uv_reg['R2_exploratory']):.4f} after adding tilt coupling and R2_uv. The increment is only {float(tilt_uv_reg['R2_exploratory'])-float(baseline_reg['R2_exploratory']):.4f}; the exact conditional metric subsumes nearly all of their typical-error signal.", "",
        "## 9. How far local information explains catastrophic basins", "",
        f"Not far. Across all assignments, median rotation error min/median/max is {rot_med[0]:.5f}/{rot_med[1]:.5f}/{rot_med[2]:.5f} deg, but p90 is {rot_p90[0]:.3f}/{rot_p90[1]:.3f}/{rot_p90[2]:.3f} deg. Success rate min/median/max is {success[0]:.3f}/{success[1]:.3f}/{success[2]:.3f}.",
        f"There are {wrong_runs} finite >90 deg runs in {len(wrong_rows)} assignments and {catastrophic_runs} non-finite/catastrophic runs. Wrong-branch rate vs logdet rho={rho('wrong_branch_rate_rotation_gt_90','H_R_given_t_logdet'):+.3f}; rotation p90 vs logdet rho={rho('rotation_error_p90_deg','H_R_given_t_logdet'):+.3f}.",
        f"The 315 near/far complement pairs have conditional logdet equal within {complement_logdet_max:.3e}, yet median absolute differences are {complement_rot_med:.5f} deg in median rotation and {complement_success_med:.3f} in success rate. This directly shows a nonlinear/noise/initialization effect outside the local information matrix.", "",
        "## 10. One N=11 / fresh-GT candidate to test", "",
        "Keep the b-set, u/v layout, distance histogram and scenario pairing controlled; choose assignments that minimize R2_uv, then rank them by conditional trace(inv) (lower) or logdet (higher). For every chosen assignment include its near/far complement. The complement control is essential because it holds conditional information fixed while exposing global-basin asymmetry.", "",
        "## Complete performance ranges", "",
        f"- translation median min/median/max: {trans_med[0]:.6f}/{trans_med[1]:.6f}/{trans_med[2]:.6f} mm",
        f"- translation p90 min/median/max: {trans_p90[0]:.6f}/{trans_p90[1]:.6f}/{trans_p90[2]:.6f} mm",
        f"- wrong-branch rate min/median/max: {wrong[0]:.3f}/{wrong[1]:.3f}/{wrong[2]:.3f}", "",
        "These are complete-population descriptive results for one fixed N=9 geometry and one paired Phase-2A scenario bank, not iid inferential samples. Fresh GT/plane and N=11 replication are required before promoting the proxy to a general pose-design rule.", "",
        "See the geometry/performance correlation tables, decile summary, complement-pair table, rank tables, anchors, raw runs, and figures for the complete evidence.", "",
    ]
    report.write_text("\n".join(lines),encoding="utf-8")

    # The suffixed files allow selected/all modes to coexist.  In exhaustive
    # all-mode also expose the exact canonical names requested by the experiment.
    if mode=="all":
        aliases={
            output/"calibration_runs_all.csv":output/"calibration_runs.csv",
            output/"calibration_summary_all.csv":output/"calibration_summary.csv",
            output/"performance_correlations_all.csv":output/"performance_correlations.csv",
            report:output/"report.md",
        }
        for source_path,target_path in aliases.items():
            if target_path.exists():
                if _sha256(source_path)!=_sha256(target_path):
                    raise FileExistsError(f"existing alias differs; refusing to overwrite: {target_path}")
                continue
            shutil.copyfile(source_path,target_path)
    _write_json(output/f"analysis_manifest_{mode}.json",{"schema":SCHEMA,"schema_version":SCHEMA_VERSION,"stage":"analysis","status":"complete","mode":mode,"assignment_count":len(merged),"report":report.name})
    print(f"Analysis complete: {report}",flush=True)


def main() -> int:
    args=_parser().parse_args(); config=_load_json(args.config); output=_resolve(args.output_dir or config["output_dir"]); workers=int(args.workers or config["calibration_workers"])
    if workers<1: raise ValueError("workers must be positive")
    if args.stage in ("geometry","all"): run_geometry(config,output)
    if args.stage in ("calibration","all"): run_calibration(config,output,args.calibration_mode,workers,args.resume)
    if args.stage in ("analysis","all") and args.calibration_mode!="none": run_analysis(config,output,args.calibration_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
