#!/usr/bin/env python3
"""Independent N=11 replication of trace-versus-condition effects.

The candidate bank, random subsets, initialization bank, and noise streams all
use new declared seeds. Geometry must pass strengthened matching checks before
the calibration stage is allowed. Calibration results are checkpointed per
subset and analyzed with the subset, not its 90 runs, as the independent unit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_random_subset_screening import run as phase1  # noqa: E402
from experiments.lhs_initialization_robustness import run as phase2a  # noqa: E402
from experiments.lhs_initialization_robustness import analyze_phase2a_geometry as statistics  # noqa: E402
from experiments.translation_trace_isotropy_controlled import run as controlled  # noqa: E402
from laser_handeye.data import LaserScan  # noqa: E402


GROUP_NAMES = (
    "A_high_trace_better_conditioned",
    "B_high_trace_worse_conditioned",
    "C_low_trace_better_conditioned",
    "D_low_trace_worse_conditioned",
)
COMPARISONS = {
    "A_vs_B": (GROUP_NAMES[0], GROUP_NAMES[1], "condition", "trace"),
    "C_vs_D": (GROUP_NAMES[2], GROUP_NAMES[3], "condition", "trace"),
    "A_vs_C": (GROUP_NAMES[0], GROUP_NAMES[2], "trace", "log_condition"),
    "B_vs_D": (GROUP_NAMES[1], GROUP_NAMES[3], "trace", "log_condition"),
}
PRIMARY_CONFOUNDERS = (
    "rotation_jacobian_logdet",
    "position_cov_trace_mm2",
    "position_span_mm",
)
SECONDARY_CONFOUNDERS = (
    "d_mean_mm",
    "d_std_mm",
    "uv_cov_trace_mm2",
)
OUTCOMES = (
    "translation_error_median_mm",
    "translation_error_p90_mm",
    "rotation_error_median_deg",
    "full_success_rate",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("geometry", "calibration", "analysis", "all"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise ValueError(f"JSON object required: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows: raise ValueError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _append_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows: return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));
        if not exists: writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(phase1._jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _population_standardization(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    fields = ("b_cov_trace", "b_cov_log_condition", *PRIMARY_CONFOUNDERS, *SECONDARY_CONFOUNDERS)
    means = {field: float(np.mean([row[field] for row in rows])) for field in fields}
    scales = {field: max(float(np.std([row[field] for row in rows])), 1e-12) for field in fields}
    return means, scales


def _strong_match(
    rows: Sequence[dict[str, Any]],
    comparison: str,
    config: dict[str, Any],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    """Match with hard constraints on controlled and priority confounder axes."""
    left_group, right_group, treatment, controlled_axis = COMPARISONS[comparison]
    left = [row for row in rows if row["group"] == left_group]
    right = [row for row in rows if row["group"] == right_group]
    controlled_field = "b_cov_trace" if controlled_axis == "trace" else "b_cov_log_condition"
    treatment_field = "b_cov_trace" if treatment == "trace" else "b_cov_log_condition"
    fields = (controlled_field, *PRIMARY_CONFOUNDERS, *SECONDARY_CONFOUNDERS)
    weights = np.asarray([12.0, 5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
    right_matrix = np.asarray([[(row[field] - means[field]) / scales[field] for field in fields] for row in right]) * weights
    tree = cKDTree(right_matrix)
    candidates: list[tuple[Any, ...]] = []
    for left_index, row in enumerate(left):
        vector = np.asarray([(row[field] - means[field]) / scales[field] for field in fields]) * weights
        distances, indices = tree.query(vector, k=min(300, len(right)))
        for distance, right_index_value in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            right_index = int(right_index_value); other = right[right_index]
            controlled_difference = abs(row[controlled_field] - other[controlled_field]) / scales[controlled_field]
            primary_differences = [abs(row[field] - other[field]) / scales[field] for field in PRIMARY_CONFOUNDERS]
            secondary_differences = [abs(row[field] - other[field]) / scales[field] for field in SECONDARY_CONFOUNDERS]
            treatment_gap = abs(row[treatment_field] - other[treatment_field]) / scales[treatment_field]
            secondary_rms = float(np.sqrt(np.mean(np.square(secondary_differences))))
            if controlled_difference > config["controlled_axis_max_std"]: continue
            if max(primary_differences) > config["primary_confounder_max_std"]: continue
            if secondary_rms > config["secondary_confounder_rms_max_std"]: continue
            if treatment_gap < config["minimum_treatment_gap_std"]: continue
            candidates.append((float(distance), left_index, right_index, controlled_difference, treatment_gap, secondary_rms, *primary_differences))
    candidates.sort()
    used_left: set[int] = set(); used_right: set[int] = set(); pairs: list[dict[str, Any]] = []
    for candidate in candidates:
        cost, left_index, right_index, controlled_difference, treatment_gap, secondary_rms, *primary_differences = candidate
        if left_index in used_left or right_index in used_right: continue
        a, b = left[left_index], right[right_index]
        pairs.append({
            "N": int(a["N"]), "comparison": comparison, "pair_id": len(pairs),
            "left_group": left_group, "right_group": right_group,
            "left_subset_id": int(a["subset_id"]), "right_subset_id": int(b["subset_id"]),
            "controlled_axis": controlled_axis, "treatment_axis": treatment,
            "matching_cost": cost, "controlled_difference_std": controlled_difference,
            "treatment_gap_std": treatment_gap, "secondary_confounder_rms_std": secondary_rms,
            **{f"{field}_difference_std": value for field, value in zip(PRIMARY_CONFOUNDERS, primary_differences, strict=True)},
            "left_trace": a["b_cov_trace"], "right_trace": b["b_cov_trace"],
            "left_log_condition": a["b_cov_log_condition"], "right_log_condition": b["b_cov_log_condition"],
        })
        used_left.add(left_index); used_right.add(right_index)
        if len(pairs) >= config["pairs_per_comparison"]: break
    return pairs


def _matching_quality(pairs: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    quality: dict[str, Any] = {}; passed = True
    for comparison in COMPARISONS:
        selected = [row for row in pairs if row["comparison"] == comparison]
        entry: dict[str, Any] = {"pair_count": len(selected)}
        for field in ("controlled_difference_std", "treatment_gap_std", "secondary_confounder_rms_std", *[f"{name}_difference_std" for name in PRIMARY_CONFOUNDERS]):
            values = np.asarray([row[field] for row in selected], dtype=float)
            entry[f"{field}_median"] = float(np.median(values)) if len(values) else math.inf
            entry[f"{field}_max"] = float(np.max(values)) if len(values) else math.inf
            entry[f"{field}_min"] = float(np.min(values)) if len(values) else -math.inf
        comparison_passed = (
            len(selected) >= config["minimum_pairs_per_comparison"]
            and entry["controlled_difference_std_median"] < config["controlled_axis_target_median_std"]
            and entry["controlled_difference_std_max"] < config["controlled_axis_max_std"]
            and entry["treatment_gap_std_min"] >= config["minimum_treatment_gap_std"]
        )
        for field in PRIMARY_CONFOUNDERS:
            comparison_passed &= entry[f"{field}_difference_std_median"] < config["primary_confounder_target_median_std"]
            comparison_passed &= entry[f"{field}_difference_std_max"] < config["primary_confounder_max_std"]
        entry["passed"] = bool(comparison_passed); passed &= comparison_passed; quality[comparison] = entry
    return quality, bool(passed)


def _save_candidate_bank(path: Path, bank: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        parameter_names=np.asarray(phase1.PARAMETER_NAMES),
        pose_convention=np.asarray(phase1.PLANE_RELATIVE_POSE_CONVENTION),
        **bank,
    )


def _assign_groups(rows: list[dict[str, Any]], fraction: float) -> dict[str, float]:
    """Assign the 2x2 design without implying near-perfect isotropy."""
    trace = np.asarray([row["b_cov_trace"] for row in rows], dtype=float)
    log_condition = np.asarray([row["b_cov_log_condition"] for row in rows], dtype=float)
    trace_low, trace_high = np.quantile(trace, [fraction, 1.0 - fraction])
    condition_better, condition_worse = np.quantile(log_condition, [fraction, 1.0 - fraction])
    for row in rows:
        high = row["b_cov_trace"] >= trace_high
        low = row["b_cov_trace"] <= trace_low
        better = row["b_cov_log_condition"] <= condition_better
        worse = row["b_cov_log_condition"] >= condition_worse
        if high and better:
            row["group"] = GROUP_NAMES[0]
        elif high and worse:
            row["group"] = GROUP_NAMES[1]
        elif low and better:
            row["group"] = GROUP_NAMES[2]
        elif low and worse:
            row["group"] = GROUP_NAMES[3]
    return {
        "trace_low": float(trace_low),
        "trace_high": float(trace_high),
        "log_condition_better": float(condition_better),
        "log_condition_worse": float(condition_worse),
    }


def run_geometry(config: dict[str, Any], output: Path) -> None:
    geometry_path = output / "fresh_N11_geometry_screening.csv"
    if geometry_path.exists(): raise FileExistsError(f"refusing to overwrite: {geometry_path}")
    output.mkdir(parents=True, exist_ok=True); plots = output / "plots"; plots.mkdir(exist_ok=True)
    replication = config["replication"]; n = int(replication["N"])
    phase1_dir = _resolve(config["phase1_dir"]); source_manifest = _load_json(phase1_dir / "manifest.json")
    source_config = copy.deepcopy(source_manifest["config"]); source_config["seed"] = int(replication["candidate_seed"]); source_config["output_dir"] = str(output / "unused_candidate_generation_output")
    resolved_phase1 = phase1._validate_config(source_config)
    print("Generating independent LHS candidate bank...", flush=True)
    bank = phase1._build_candidate_bank(resolved_phase1)
    _save_candidate_bank(output / "fresh_N11_candidate_bank.npz", bank)
    with np.load(phase1_dir / "candidate_bank.npz", allow_pickle=False) as archive:
        old_parameters = archive["pose_parameters"].copy()
    exact_candidate_overlap = len({tuple(row) for row in bank["pose_parameters"]} & {tuple(row) for row in old_parameters})
    if exact_candidate_overlap:
        raise RuntimeError("new candidate bank unexpectedly contains exact old candidates")

    rng = np.random.default_rng(int(replication["subset_seed"]))
    subsets = controlled._fresh_subsets(rng, len(bank["candidate_ids"]), n, int(replication["fresh_subsets"]), set())
    normal = np.asarray(source_manifest["resolved"]["plane"]["normal_base"], dtype=float); offset = float(source_manifest["resolved"]["plane"]["offset_mm"])
    rows: list[dict[str, Any]] = []; validation: dict[str, float] = {}
    print(f"Computing geometry for {len(subsets)} fresh N={n} subsets...", flush=True)
    for subset_id, candidate_ids in enumerate(subsets):
        row, checks = controlled._geometry_row(subset_id, candidate_ids, bank, normal, offset, float(replication["condition_eigenvalue_floor"]))
        rows.append(row)
        for name, value in checks.items(): validation[name] = max(validation.get(name, 0.0), float(value))
    thresholds = _assign_groups(rows, float(replication["extreme_fraction"])); means, scales = _population_standardization(rows)
    pairs: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        matched = _strong_match(rows, comparison, replication, means, scales); pairs.extend(matched)
        print(f"{comparison}: {len(matched)} pairs", flush=True)
    quality, matching_passed = _matching_quality(pairs, replication)
    identity_passed = all(value <= 1e-9 for value in validation.values()); matching_passed &= identity_passed
    selected_keys = {(int(row["N"]), int(row[side])) for row in pairs for side in ("left_subset_id", "right_subset_id")}
    matched_rows = [row for row in rows if (int(row["N"]), int(row["subset_id"])) in selected_keys]
    _write_csv(geometry_path, rows); _write_csv(output / "fresh_N11_matched_pairs.csv", pairs); _write_csv(output / "fresh_N11_matched_subsets.csv", matched_rows)
    np.savez_compressed(output / "fresh_N11_subsets.npz", candidate_ids=subsets)

    fig, ax = plt.subplots(figsize=(7,5.5)); ax.scatter([row["b_cov_trace"] for row in rows], [row["b_cov_condition"] for row in rows], s=4, alpha=.12); ax.set_yscale("log"); ax.set_xlabel("trace(Cov(b))"); ax.set_ylabel("condition(Cov(b))"); ax.set_title("Independent N=11 fresh geometry"); fig.tight_layout(); fig.savefig(plots / "fresh_N11_trace_condition_population.png", dpi=180); plt.close(fig)
    manifest = {"status": "complete", "matching_passed": matching_passed, "identity_passed": identity_passed, "candidate_bank_seed": replication["candidate_seed"], "subset_seed": replication["subset_seed"], "exact_old_candidate_overlap": exact_candidate_overlap, "subset_count": len(rows), "matched_unique_subset_count": len(matched_rows), "thresholds": thresholds, "population_means": means, "population_scales": scales, "validation_max_errors": validation, "matching_quality": quality, "calibration_started": False}
    _write_json(output / "fresh_N11_geometry_manifest.json", manifest)
    lines = ["# Fresh N=11 matching report", "", "This is an independent new candidate bank and subset population. Calibration has not run.", "", f"- Candidate seed: {replication['candidate_seed']}", f"- Subset seed: {replication['subset_seed']}", f"- Exact old candidate overlap: {exact_candidate_overlap}", f"- Fresh subsets: {len(rows)}", f"- Unique matched subsets: {len(matched_rows)}", f"- Matching passed: `{str(matching_passed).lower()}`", "", "## Matching quality", "", "| comparison | pairs | controlled median/max | treatment min | rotation-J median/max | position trace median/max | position span median/max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for comparison, entry in quality.items():
        lines.append(f"| {comparison} | {entry['pair_count']} | {entry['controlled_difference_std_median']:.3f}/{entry['controlled_difference_std_max']:.3f} | {entry['treatment_gap_std_min']:.3f} | {entry['rotation_jacobian_logdet_difference_std_median']:.3f}/{entry['rotation_jacobian_logdet_difference_std_max']:.3f} | {entry['position_cov_trace_mm2_difference_std_median']:.3f}/{entry['position_cov_trace_mm2_difference_std_max']:.3f} | {entry['position_span_mm_difference_std_median']:.3f}/{entry['position_span_mm_difference_std_max']:.3f} |")
    lines += ["", "Calibration is allowed only when `matching_passed=true`.", ""]
    (output / "fresh_N11_matching_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Geometry matching_passed={matching_passed}; output={output}")


def _load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: archive[name].copy()
            for name in archive.files
            if name not in ("parameter_names", "pose_convention")
        }


def _paired_scans(bank: dict[str, np.ndarray], candidate_ids: np.ndarray, environment: int, repeat: int, noise_seed: int, noise_std: float, noise_axis: str) -> list[LaserScan]:
    scans: list[LaserScan] = []
    for slot, candidate_value in enumerate(candidate_ids):
        candidate = int(candidate_value); points = bank["ideal_points_s"][candidate].copy(); rng = np.random.default_rng(np.random.SeedSequence([noise_seed, environment, repeat, slot]))
        if noise_axis == "z": points[:,2] += rng.normal(0.0, noise_std, len(points))
        else: points[:,[0,2]] += rng.normal(0.0, noise_std, (len(points),2))
        scans.append(LaserScan(T_base_ef=bank["T_base_ef"][candidate], points_s=points, plane_id=0, scan_id=candidate, meta={"candidate_id":candidate,"environment_id":environment,"noise_repeat":repeat,"noise_slot":slot}))
    return scans


def run_calibration(config: dict[str, Any], output: Path, resume: bool) -> None:
    geometry_manifest = _load_json(output / "fresh_N11_geometry_manifest.json")
    if geometry_manifest.get("matching_passed") is not True: raise RuntimeError("fresh N=11 matching failed; calibration blocked")
    summary_path = output / "fresh_N11_stage2_results.csv"; run_path = output / "fresh_N11_calibration_runs.csv"
    if summary_path.exists() and not resume: raise FileExistsError("calibration checkpoint exists; use --resume")
    completed = {(int(row["N"]), int(row["subset_id"])) for row in _read_csv(summary_path)} if summary_path.exists() else set()
    config_replication = config["replication"]
    source_phase1 = _load_json(_resolve(config["phase1_dir"]) / "manifest.json")
    controlled_stage1 = _load_json(_resolve(config["existing_stage2_dir"]) / "stage1_manifest.json")
    original_phase2_manifest = _load_json(_resolve(controlled_stage1["config"]["phase2a_dir"]) / "manifest.json")
    bank = _load_bank(output / "fresh_N11_candidate_bank.npz"); subsets = _read_csv(output / "fresh_N11_matched_subsets.csv")
    scenario_resolved = {"seed": int(config_replication["initialization_seed"]), "translation_range_mm": tuple(original_phase2_manifest["config"]["initial_error"]["translation_norm_mm"]), "rotation_range_deg": tuple(original_phase2_manifest["config"]["initial_error"]["rotation_deg"]), "environments": int(original_phase2_manifest["config"]["initial_error"]["environments"])}
    T_true = np.asarray(source_phase1["resolved"]["T_ef_s_true"], dtype=float); scenarios = phase2a._make_scenario_bank(scenario_resolved, T_true, smoke=False)
    scenario_path = output / "fresh_N11_scenario_bank.npz"
    if not scenario_path.exists(): np.savez_compressed(scenario_path, **scenarios)
    noise = original_phase2_manifest["config"]["noise"]; repeats = int(noise["repeats_per_environment"])
    remaining = [row for row in subsets if (int(row["N"]), int(row["subset_id"])) not in completed]
    total = len(remaining) * len(scenarios["environment_ids"]) * repeats; done = 0; started = time.perf_counter()
    for position, subset in enumerate(remaining, start=1):
        candidate_ids = np.asarray(json.loads(subset["candidate_ids"]), dtype=np.int64); subset_runs: list[dict[str, Any]] = []
        for environment_value in scenarios["environment_ids"]:
            environment = int(environment_value)
            solver_resolved = phase2a._solver_resolved(source_phase1, {"classification_translation_mm": float(original_phase2_manifest["config"]["classification"]["translation_mm"]), "classification_rotation_deg": float(original_phase2_manifest["config"]["classification"]["rotation_deg"])}, scenarios["T_init"][environment])
            for repeat in range(repeats):
                scans = _paired_scans(bank, candidate_ids, environment, repeat, int(config_replication["noise_seed"]), float(noise["std_mm"]), str(noise["axis"]))
                row = phase1._run_one_calibration(resolved=solver_resolved, scans=scans, scan_count=int(subset["N"]), subset_id=int(subset["subset_id"]), repeat=repeat, candidate_ids=candidate_ids)
                row.update(source="fresh_N11_replication", environment_id=environment, noise_repeat=repeat)
                subset_runs.append(row); done += 1
        numeric = {key: value for key, value in subset.items()}; numeric["N"] = int(subset["N"]); numeric["subset_id"] = int(subset["subset_id"])
        summary = controlled._stage2_summary(numeric, subset_runs, 1.0)
        _append_csv(run_path, subset_runs); _append_csv(summary_path, [summary])
        elapsed = time.perf_counter()-started; eta=elapsed*(total-done)/done
        print(f"[fresh N=11] subset {position}/{len(remaining)}, runs {done}/{total}, ETA {phase1._format_duration(eta)}", flush=True)
    _write_json(output / "fresh_N11_calibration_manifest.json", {"status":"complete","candidate_seed":config_replication["candidate_seed"],"initialization_seed":config_replication["initialization_seed"],"noise_seed":config_replication["noise_seed"],"subset_count":len(subsets),"runs_per_subset":len(scenarios["environment_ids"])*repeats,"subset_is_statistical_unit":True})


def _bootstrap_median(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float,float]:
    samples=np.empty(repeats)
    for i in range(repeats): samples[i]=np.median(values[rng.integers(0,len(values),len(values))])
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def _standardized_effect(pairs: Sequence[dict[str,str]], lookup: dict[tuple[int,int],dict[str,str]], treatment: str, trace_scale: float, condition_scale: float) -> tuple[float,np.ndarray]:
    dx=[]; dy=[]
    for pair in pairs:
        left=lookup[(int(pair["N"]),int(pair["left_subset_id"]))]; right=lookup[(int(pair["N"]),int(pair["right_subset_id"]))]
        if treatment=="trace": dx.append((float(left["b_cov_trace"])-float(right["b_cov_trace"]))/trace_scale)
        else: dx.append((float(left["b_cov_log_condition"])-float(right["b_cov_log_condition"]))/condition_scale)
        dy.append(np.log(float(left["translation_error_median_mm"]))-np.log(float(right["translation_error_median_mm"])))
    x=np.asarray(dx); y=np.asarray(dy); return float(x@y/(x@x)),np.column_stack([x,y])


def run_analysis(config: dict[str, Any], output: Path) -> None:
    result_path=output/"fresh_N11_stage2_results_report.md"
    if result_path.exists(): raise FileExistsError(f"refusing to overwrite: {result_path}")
    summaries=_read_csv(output/"fresh_N11_stage2_results.csv"); pairs=_read_csv(output/"fresh_N11_matched_pairs.csv"); lookup={(int(r["N"]),int(r["subset_id"])):r for r in summaries}
    repeats=int(config["bootstrap_repeats"]); rng=np.random.default_rng(int(config["seed"])+99); rows=[]
    p_values=[]
    for comparison in COMPARISONS:
        selected=[r for r in pairs if r["comparison"]==comparison]
        for outcome in OUTCOMES:
            left=np.asarray([float(lookup[(int(r["N"]),int(r["left_subset_id"]))][outcome]) for r in selected]); right=np.asarray([float(lookup[(int(r["N"]),int(r["right_subset_id"]))][outcome]) for r in selected]); difference=left-right
            statistic,p=wilcoxon(left,right) if np.any(difference) else (0.0,1.0); ci=_bootstrap_median(difference,repeats,rng)
            row={"comparison":comparison,"outcome":outcome,"pair_count":len(selected),"left_median":float(np.median(left)),"right_median":float(np.median(right)),"median_paired_difference":float(np.median(difference)),"bootstrap_ci_low":ci[0],"bootstrap_ci_high":ci[1],"left_win_rate":float(np.mean(left<right)),"paired_sign_effect":float((np.sum(difference>0)-np.sum(difference<0))/len(difference)),"wilcoxon_p":float(p),"fdr_q":math.nan}; rows.append(row); p_values.append(p)
    adjusted=statistics._bh_adjust(p_values)
    for row,q in zip(rows,adjusted,strict=True): row["fdr_q"]=float(q)
    _write_csv(output/"fresh_N11_stage2_paired_statistics.csv",rows)

    manifest=_load_json(output/"fresh_N11_geometry_manifest.json"); trace_scale=float(manifest["population_scales"]["b_cov_trace"]); condition_scale=float(manifest["population_scales"]["b_cov_log_condition"])
    trace_pairs=[r for r in pairs if r["comparison"] in ("A_vs_C","B_vs_D")]; condition_pairs=[r for r in pairs if r["comparison"] in ("A_vs_B","C_vs_D")]
    beta_t,trace_xy=_standardized_effect(trace_pairs,lookup,"trace",trace_scale,condition_scale); beta_k,condition_xy=_standardized_effect(condition_pairs,lookup,"condition",trace_scale,condition_scale)
    boot_t=[];boot_k=[]
    for _ in range(repeats):
        t=trace_xy[rng.integers(0,len(trace_xy),len(trace_xy))]; k=condition_xy[rng.integers(0,len(condition_xy),len(condition_xy))]
        boot_t.append(float(t[:,0]@t[:,1]/(t[:,0]@t[:,0]))); boot_k.append(float(k[:,0]@k[:,1]/(k[:,0]@k[:,0])))
    boot_t=np.asarray(boot_t);boot_k=np.asarray(boot_k); magnitude=np.abs(boot_t)-np.abs(boot_k)
    effect_rows=[{"effect":"trace_per_1sd","coefficient_log_error":beta_t,"ci_low":float(np.quantile(boot_t,.025)),"ci_high":float(np.quantile(boot_t,.975))},{"effect":"log_condition_per_1sd","coefficient_log_error":beta_k,"ci_low":float(np.quantile(boot_k,.025)),"ci_high":float(np.quantile(boot_k,.975))},{"effect":"abs_trace_minus_abs_condition","coefficient_log_error":abs(beta_t)-abs(beta_k),"ci_low":float(np.quantile(magnitude,.025)),"ci_high":float(np.quantile(magnitude,.975))}]
    _write_csv(output/"fresh_N11_standardized_effects.csv",effect_rows)

    def significant_direction(comparison:str)->bool:
        row=next(r for r in rows if r["comparison"]==comparison and r["outcome"]=="translation_error_median_mm")
        return float(row["median_paired_difference"])<0 and float(row["fdr_q"])<.05 and float(row["left_win_rate"])>.5
    trace_replicated=significant_direction("A_vs_C") and significant_direction("B_vs_D"); condition_replicated=significant_direction("A_vs_B") and significant_direction("C_vs_D")
    magnitude_trace_larger=(abs(beta_t)>abs(beta_k)); magnitude_ci_positive=float(np.quantile(magnitude,.025))>0
    if not trace_replicated: case=4; conclusion="N=11에서 trace effect가 엄격한 양쪽 비교 기준으로 반복되지 않아 N=9-specific 가능성을 유지하고 pose rule 확정을 보류한다."
    elif trace_replicated and condition_replicated and magnitude_trace_larger and magnitude_ci_positive: case=1; conclusion="total centered spread is the primary translation geometry factor; directional balance provides secondary refinement"
    elif trace_replicated and not condition_replicated: case=2; conclusion="trace만 반복되었으므로 high trace / low resultant를 주 원리로 두고 condition의 설계 우선순위를 낮춘다."
    else: case=3; conclusion="trace와 condition이 모두 반복되어 trace first, condition second 구조를 유지하고 logdet(Cov(b))를 combined screening score 후보로 둔다."

    plots=output/"plots"; plots.mkdir(exist_ok=True)
    def paired_plot(comparison:str,path:Path)->None:
        selected=[r for r in pairs if r["comparison"]==comparison]; left=[float(lookup[(int(r["N"]),int(r["left_subset_id"]))]["translation_error_median_mm"]) for r in selected]; right=[float(lookup[(int(r["N"]),int(r["right_subset_id"]))]["translation_error_median_mm"]) for r in selected]
        fig,ax=plt.subplots(figsize=(5.8,4.8));
        for a,b in zip(left,right): ax.plot([0,1],[a,b],color="#777",alpha=.35)
        ax.scatter(np.zeros(len(left)),left,label="left");ax.scatter(np.ones(len(right)),right,label="right");ax.set_xticks([0,1],[selected[0]["left_group"],selected[0]["right_group"]],rotation=10);ax.set_ylabel("translation median error (mm)");ax.set_title(comparison);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
    paired_plot("A_vs_B",plots/"fresh_N11_matched_A_B_translation_error.png");paired_plot("A_vs_C",plots/"fresh_N11_matched_A_C_translation_error.png")

    existing_rows=_read_csv(output/"existing_stage2_effect_decomposition.csv"); n9_t=next(float(r["coefficient"]) for r in existing_rows if r["model"]=="primary_log_error" and r["predictor"]=="z_trace");n9_k=next(float(r["coefficient"]) for r in existing_rows if r["model"]=="primary_log_error" and r["predictor"]=="z_log_condition")
    fig,ax=plt.subplots(figsize=(6,4.5));ax.bar([0,1,3,4],[n9_t,n9_k,beta_t,beta_k],color=["#4575b4","#fdae61","#4575b4","#fdae61"]);ax.set_xticks([0,1,3,4],["N9 trace","N9 condition","N11 trace","N11 condition"]);ax.axhline(0,color="black",linewidth=.8);ax.set_ylabel("log-error coefficient per geometry SD");ax.set_title("N=9 vs independent N=11 effects");fig.tight_layout();fig.savefig(plots/"N9_vs_N11_standardized_effects.png",dpi=180);plt.close(fig)
    report=["# Independent fresh N=11 Stage 2 results", "", f"- Automated conclusion case: **Case {case}**", f"- {conclusion}", "", "## Standardized effect comparison", "", f"- Trace beta per 1 SD: {beta_t:+.4f}, 95% CI [{np.quantile(boot_t,.025):+.4f}, {np.quantile(boot_t,.975):+.4f}]", f"- Log-condition beta per 1 SD: {beta_k:+.4f}, 95% CI [{np.quantile(boot_k,.025):+.4f}, {np.quantile(boot_k,.975):+.4f}]", f"- |trace|-|condition|: {abs(beta_t)-abs(beta_k):+.4f}, 95% CI [{np.quantile(magnitude,.025):+.4f}, {np.quantile(magnitude,.975):+.4f}]", "", "## Paired primary outcome", "", "| comparison | left median | right median | paired difference | 95% bootstrap CI | FDR q | left win rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for comparison in COMPARISONS:
        row=next(r for r in rows if r["comparison"]==comparison and r["outcome"]=="translation_error_median_mm");report.append(f"| {comparison} | {row['left_median']:.4f} | {row['right_median']:.4f} | {row['median_paired_difference']:+.4f} | [{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] | {row['fdr_q']:.3g} | {row['left_win_rate']:.3f} |")
    report += ["", "The independent replication uses a new candidate bank, subsets, initialization bank, and noise streams. Subset is the statistical unit. N=11 was selected for replication, not because it is theoretically optimal.", ""]
    result_path.write_text("\n".join(report),encoding="utf-8")
    _write_json(output/"fresh_N11_analysis_summary.json",{"case":case,"conclusion":conclusion,"trace_replicated":trace_replicated,"condition_replicated":condition_replicated,"trace_effect_per_sd":beta_t,"condition_effect_per_sd":beta_k,"magnitude_difference_ci":[float(np.quantile(magnitude,.025)),float(np.quantile(magnitude,.975))]})
    print(f"Fresh N=11 analysis complete: Case {case}")


def main() -> int:
    args=_parser();config=_load_json(args.config);output=_resolve(args.output_dir or config["output_dir"])
    if args.stage in ("geometry","all"): run_geometry(config,output)
    if args.stage in ("calibration","all"): run_calibration(config,output,args.resume)
    if args.stage in ("analysis","all"): run_analysis(config,output)
    return 0


if __name__=="__main__":raise SystemExit(main())
