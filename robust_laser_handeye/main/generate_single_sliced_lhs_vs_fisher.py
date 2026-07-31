#!/usr/bin/env python3
"""Sliced-LHS continuation vs active Fisher selection.

This module reuses the existing generate_single_uniform_vs_fisher.py generator,
but replaces its trial construction so that:

1. The first INITIAL_SCANS are an exact coarse LHS bootstrap.
2. The complete TOTAL_SCANS uniform branch is an exact refined LHS.
3. The Fisher branch starts from exactly the same noisy bootstrap scans.
4. Fisher selects the remaining scans from the same feasible pose domain.

For exact nested slicing, TOTAL_SCANS must be divisible by INITIAL_SCANS.
The output collection names remain ``single_plane_uniform`` and
``single_plane_fisher`` for compatibility with the existing validator/plotters.
The uniform collection metadata identifies the policy as
``sliced_lhs_continuation``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from main import generate_single_uniform_vs_fisher as legacy
from main import generate_independent_random_plane_comparison as base
from laser_handeye.calibration_dataset import CalibrationDataset
from laser_handeye.simulation import sample_random_handeye


def _nested_sliced_lhs(
    rng: np.random.Generator,
    *,
    initial_count: int,
    total_count: int,
    dimensions: int,
) -> np.ndarray:
    """Return an ordered nested LHS.

    The first ``initial_count`` rows form an LHS over coarse strata.
    All ``total_count`` rows form an LHS over refined strata.
    """
    if total_count % initial_count != 0:
        raise ValueError(
            "exact nested sliced-LHS requires total_scans % initial_scans == 0"
        )
    refinement = total_count // initial_count
    values = np.empty((total_count, dimensions), dtype=float)

    for dim in range(dimensions):
        coarse_order = rng.permutation(initial_count)
        sub_orders = np.vstack(
            [rng.permutation(refinement) for _ in range(initial_count)]
        )

        # Bootstrap: one refined sub-stratum from every coarse stratum.
        for row in range(initial_count):
            fine_index = coarse_order[row] * refinement + sub_orders[row, 0]
            values[row, dim] = (fine_index + rng.random()) / total_count

        # Continuation: fill the unused refined strata.
        output_row = initial_count
        for layer in range(1, refinement):
            row_order = rng.permutation(initial_count)
            for source_row in row_order:
                fine_index = (
                    coarse_order[source_row] * refinement
                    + sub_orders[source_row, layer]
                )
                values[output_row, dim] = (
                    fine_index + rng.random()
                ) / total_count
                output_row += 1

    return values


def _simulate_design_bank(
    *,
    rows: np.ndarray,
    extra_count: int,
    rng: np.random.Generator,
    master_seed: int,
    trial_index: int,
    frame,
    common_center: np.ndarray,
    T_ef_s_true: np.ndarray,
    x_values: np.ndarray,
    fair_config,
    candidate_config,
) -> tuple[list, list, np.ndarray, dict[str, Any]]:
    """Generate exact sliced design first, then extra Fisher candidates."""
    poses = []
    scans = []
    normalized = []
    attempted = 0
    rejected = 0

    def try_row(row: np.ndarray, candidate_id: int):
        nonlocal attempted, rejected
        attempted += 1
        pose = legacy._candidate_from_row(
            row,
            candidate_id=candidate_id,
            master_seed=master_seed,
            trial_index=trial_index,
            config=candidate_config,
        )
        try:
            T_base_s = legacy._make_sensor_pose_relative_to_plane(
                frame, common_center, pose
            )
        except (ValueError, FloatingPointError):
            rejected += 1
            return None
        feasibility = base._profile_feasibility(
            plane=frame,
            common_center=common_center,
            T_base_s=T_base_s,
            x_values=x_values,
            config=fair_config,
        )
        if feasibility is None:
            rejected += 1
            return None
        scan = legacy._simulate_candidate_scan(
            T_ef_s_true=T_ef_s_true,
            plane=frame,
            T_base_s=T_base_s,
            pose=pose,
            feasibility=feasibility,
            x_values=x_values,
        )
        return pose, scan

    # Preserve every LHS stratum. If a point is infeasible, retry only its
    # within-stratum jitter rather than moving it to another stratum.
    total_count = len(rows)
    for design_index, base_row in enumerate(rows):
        accepted = None
        fine_lower = np.floor(base_row * total_count) / total_count
        for _ in range(200):
            retry_row = fine_lower + rng.random(base_row.shape) / total_count
            accepted = try_row(retry_row, len(poses))
            if accepted is not None:
                poses.append(accepted[0])
                scans.append(accepted[1])
                normalized.append(retry_row)
                break
        if accepted is None:
            raise RuntimeError(
                f"could not find a feasible pose in sliced-LHS stratum "
                f"{design_index}; reduce pose ranges"
            )

    # Extra action bank used only by Fisher after the common bootstrap.
    while len(poses) < total_count + extra_count:
        batch = legacy._latin_hypercube(
            rng,
            max(64, total_count + extra_count - len(poses)),
            rows.shape[1],
        )
        for row in batch:
            if len(poses) >= total_count + extra_count:
                break
            accepted = try_row(row, len(poses))
            if accepted is not None:
                poses.append(accepted[0])
                scans.append(accepted[1])
                normalized.append(row)

    stats = {
        "attempted": attempted,
        "accepted": len(poses),
        "rejected": rejected,
        "nested_sliced_lhs_count": total_count,
        "extra_fisher_candidate_count": extra_count,
        "robot_ik_and_collision_checked": False,
    }
    return poses, scans, np.vstack(normalized), stats


def _sliced_uniform_output(
    *,
    selected_ids,
    acquire_scan,
    T_initial,
    parameter_scales,
    selection_config,
):
    acquired = [acquire_scan(int(i)) for i in selected_ids]
    initial_count = selection_config.initial_random_scans
    trace = [
        {
            "acquisition_step": step,
            "candidate_pool_index": int(candidate_id),
            "assigned_plane_id": 0,
            "selection_stage": (
                "common_sliced_lhs_bootstrap"
                if step < initial_count
                else "sliced_lhs_continuation"
            ),
            "selection_objective": "nested_sliced_lhs_space_filling",
            "objective_gain": None,
            "objective_value_predicted_after": None,
            "objective_value_after": None,
            "linearization_source": "not_used_by_sliced_lhs_policy",
        }
        for step, candidate_id in enumerate(selected_ids)
    ]

    estimate = legacy.estimate_joint_calibration(
        {0: acquired},
        T_initial,
        parameter_scales=parameter_scales,
        profile_noise_std_mm=selection_config.fisher_profile_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
        max_iterations=selection_config.estimator_max_iterations,
        tolerance=selection_config.estimator_tolerance,
    )
    summary = {
        "strategy": "sliced_lhs",
        "objective": "nested_sliced_lhs_space_filling",
        "initial_random_scan_count": initial_count,
        "selected_candidate_ids": list(map(int, selected_ids)),
        "selected_assigned_plane_counts": {"0": len(selected_ids), "1": 0, "2": 0},
        "final_information": base._observed_information_diagnostics(
            estimate.information, parameter_scales
        ),
        "final_estimate_sha256": base._estimate_hash(estimate),
        "final_estimate": base._estimate_snapshot(estimate),
        "final_estimator_whitened_cost": estimate.whitened_cost,
        "final_estimator_iterations": estimate.iterations,
        "final_estimator_converged": estimate.converged,
        "linearization_policy": "not_used_for_pose_selection",
        "information_prior": "none",
    }
    return list(map(int, selected_ids)), acquired, trace, summary


def _generate_trial(
    *,
    fair_config,
    candidate_config,
    selection_config,
    master_seed: int,
    trial_index: int,
) -> tuple[dict[str, CalibrationDataset], dict[str, Any]]:
    if fair_config.total_scans % selection_config.initial_random_scans != 0:
        raise ValueError(
            "TOTAL_SCANS must be divisible by INITIAL_SCANS for exact "
            "nested sliced-LHS"
        )

    sequence = np.random.SeedSequence([master_seed, trial_index])
    handeye_seq, plane_seq, design_seq, bank_seq = sequence.spawn(4)

    T_ef_s_true, _, _ = sample_random_handeye(np.random.default_rng(handeye_seq))
    frames, common_center, frame_angles_deg = base._make_plane_frames(
        np.random.default_rng(plane_seq),
        fair_config.plane_angle_range_deg,
        fair_config.plane_center_xy_range_mm,
        fair_config.plane_center_z_range_mm,
    )
    plane = frames[0]
    x_values = np.linspace(
        -fair_config.profile_half_width_mm,
        fair_config.profile_half_width_mm,
        fair_config.profile_points,
    )

    design_rows = _nested_sliced_lhs(
        np.random.default_rng(design_seq),
        initial_count=selection_config.initial_random_scans,
        total_count=fair_config.total_scans,
        dimensions=6,
    )
    extra_count = selection_config.candidate_pool_size - fair_config.total_scans
    poses, source_scans, normalized, bank_stats = _simulate_design_bank(
        rows=design_rows,
        extra_count=extra_count,
        rng=np.random.default_rng(bank_seq),
        master_seed=master_seed,
        trial_index=trial_index,
        frame=plane,
        common_center=common_center,
        T_ef_s_true=T_ef_s_true,
        x_values=x_values,
        fair_config=fair_config,
        candidate_config=candidate_config,
    )

    candidates = base._candidate_kinematics(
        source_scans, np.zeros(len(source_scans), dtype=int)
    )
    acquire_scan = base._make_candidate_acquirer(
        source_scans,
        measurement_seed=selection_config.measurement_seed,
        trial_index=trial_index,
        noise_std_mm=selection_config.measurement_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
    )
    scales = base._information_parameter_scales(1, selection_config)
    T_initial = base._selection_initial_transform(
        T_ef_s_true, trial_index=trial_index, config=selection_config
    )

    uniform_ids = list(range(fair_config.total_scans))
    initial_ids = uniform_ids[: selection_config.initial_random_scans]

    uniform_output = _sliced_uniform_output(
        selected_ids=uniform_ids,
        acquire_scan=acquire_scan,
        T_initial=T_initial,
        parameter_scales=scales,
        selection_config=selection_config,
    )
    fisher_output = base._run_online_pose_selection(
        candidates=candidates,
        acquire_scan=acquire_scan,
        initial_candidate_ids=initial_ids,
        total_scans=fair_config.total_scans,
        strategy="fisher",
        fixed_random_sequence=None,
        enforce_three_plane_balance=False,
        T_initial=T_initial,
        x_values=x_values,
        parameter_scales=scales,
        selection_config=selection_config,
        profile_depth_range_mm=fair_config.profile_depth_range_mm,
    )

    outputs = {
        "single_uniform": uniform_output,
        "single_fisher": fisher_output,
    }
    datasets = {}
    for key, (selected, acquired, trace, summary) in outputs.items():
        datasets[key] = base._build_selected_dataset(
            geometry_strategy="single_plane",
            pose_selection_strategy=(
                "sliced_lhs" if key == "single_uniform" else "fisher"
            ),
            acquired_scans=acquired,
            selected_candidate_ids=selected,
            selection_trace=trace,
            selection_summary=summary,
            T_ef_s_true=T_ef_s_true,
            frames=frames,
            common_center=common_center,
            frame_angles_deg=frame_angles_deg,
            trial_index=trial_index,
            config=fair_config,
            selection_config=selection_config,
        )

    b = len(initial_ids)
    lhs_bootstrap = np.stack(
        [scan.T_base_ef for scan in datasets["single_uniform"].scans[:b]]
    )
    fisher_bootstrap = np.stack(
        [scan.T_base_ef for scan in datasets["single_fisher"].scans[:b]]
    )
    initial_difference = float(np.max(np.abs(lhs_bootstrap - fisher_bootstrap)))
    if initial_difference > fair_config.verification_atol:
        raise RuntimeError("branches do not share identical bootstrap poses")

    uniform_info = uniform_output[3]["final_information"]
    fisher_info = fisher_output[3]["final_information"]
    comparison = {
        "trial_index": trial_index,
        "T_ef_s_true_sha256": base._array_sha256(T_ef_s_true),
        "candidate_bank_sha256": legacy._candidate_bank_hash(poses),
        "candidate_pool_size": len(poses),
        "candidate_bank_statistics": bank_stats,
        "candidate_config": asdict(candidate_config),
        "same_initial_bootstrap": True,
        "initial_candidate_ids": initial_ids,
        "initial_pose_max_abs_difference": initial_difference,
        "uniform_policy": "exact_nested_sliced_lhs_continuation",
        "fisher_policy": "estimated_state_active_fisher",
        "same_measurement_noise_for_same_candidate": True,
        "same_total_scan_count": True,
        "single_uniform_selected_candidate_ids": uniform_output[0],
        "single_fisher_selected_candidate_ids": fisher_output[0],
        "single_uniform_final_information": uniform_info,
        "single_fisher_final_information": fisher_info,
        "fisher_minus_uniform_marginal_logdet": float(
            fisher_info["handeye_marginal_logdet"]
            - uniform_info["handeye_marginal_logdet"]
        ),
        "fisher_over_uniform_min_eigenvalue_ratio": float(
            fisher_info["handeye_marginal_min_eigenvalue"]
            / uniform_info["handeye_marginal_min_eigenvalue"]
        ),
        "uniform_over_fisher_predicted_rotation_std_ratio": float(
            uniform_info["predicted_rotation_std_deg"]
            / fisher_info["predicted_rotation_std_deg"]
        ),
        "uniform_over_fisher_predicted_translation_std_ratio": float(
            uniform_info["predicted_translation_std_mm"]
            / fisher_info["predicted_translation_std_mm"]
        ),
    }
    return datasets, comparison


# Patch only the trial-generation policy; retain the established CLI and writer.
legacy._generate_trial = _generate_trial


def _output_dir_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--output-dir" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--output-dir="):
            return token.split("=", 1)[1]
    return None


def _patch_policy_metadata(output_dir: str) -> None:
    """Correct legacy maximin labels without changing the compatible schema."""
    from pathlib import Path
    import json

    root = Path(output_dir)
    comparison_path = root / "comparison_manifest.json"
    if comparison_path.is_file():
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        rules = payload.setdefault("comparison_rules", {})
        rules["uniform_policy"] = "exact_nested_sliced_lhs_continuation"
        rules["fisher_policy"] = "estimated_state_active_fisher"
        payload["uniform_policy"] = "exact_nested_sliced_lhs_continuation"
        comparison_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    uniform_path = root / "single_plane_uniform" / "collection.json"
    if uniform_path.is_file():
        payload = json.loads(uniform_path.read_text(encoding="utf-8"))
        payload["pose_selection_policy"] = (
            "exact_nested_sliced_lhs_continuation"
        )
        uniform_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    import sys

    output_dir = _output_dir_from_argv(sys.argv[1:])
    exit_code = legacy.main()
    if exit_code == 0 and output_dir is not None:
        _patch_policy_metadata(output_dir)
    raise SystemExit(exit_code)
