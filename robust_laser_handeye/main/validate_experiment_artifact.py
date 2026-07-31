#!/usr/bin/env python3
"""Validate cached datasets/results before an experiment runner reuses them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


COMPARISON_SCHEMA = "laser_handeye.shared_global_feasible_pose_comparison"
UNIFORM_COMPARISON_SCHEMA = "laser_handeye.uniform_relative_pose_comparison"
SINGLE_UNIFORM_FISHER_SCHEMA = (
    "laser_handeye.single_plane_uniform_vs_active_fisher"
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing JSON field: {dotted_key}")
        value = value[key]
    return value


def _same_number(actual: Any, expected: float) -> bool:
    try:
        number = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isclose(number, float(expected), rel_tol=1e-12, abs_tol=1e-12)


def _expect_equal(
    document: dict[str, Any],
    dotted_key: str,
    expected: Any,
) -> None:
    actual = _nested(document, dotted_key)
    if actual != expected:
        raise ValueError(
            f"{dotted_key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _expect_number(
    document: dict[str, Any],
    dotted_key: str,
    expected: float,
) -> None:
    actual = _nested(document, dotted_key)
    if not _same_number(actual, expected):
        raise ValueError(
            f"{dotted_key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _expect_number_pair(
    document: dict[str, Any],
    dotted_key: str,
    expected: Sequence[float],
) -> None:
    actual = _nested(document, dotted_key)
    if (
        not isinstance(actual, list)
        or len(actual) != 2
        or not all(
            _same_number(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    ):
        raise ValueError(
            f"{dotted_key} mismatch: expected {list(expected)!r}, "
            f"got {actual!r}"
        )


def _validate_dataset(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)

    _expect_equal(manifest, "schema", COMPARISON_SCHEMA)
    schema_version = _nested(manifest, "schema_version")
    if schema_version not in (1, 2, 3, 4):
        raise ValueError(
            "schema_version mismatch: expected 1, 2, 3, or 4, "
            f"got {schema_version!r}"
        )
    _expect_equal(manifest, "status", "complete")
    _expect_equal(manifest, "pose_sampling_frame", "shared_target_global")
    if schema_version == 1:
        _expect_equal(
            manifest,
            "single_and_three_robot_poses_identical",
            True,
        )
    else:
        _expect_equal(
            manifest,
            "random_single_and_three_robot_poses_identical",
            True,
        )
        _expect_equal(
            manifest,
            "all_branches_share_initial_random_poses",
            True,
        )
    _expect_equal(manifest, "requested_trials", args.trials)
    _expect_equal(manifest, "completed_trials", args.trials)
    _expect_equal(manifest, "master_seed", args.seed)
    _expect_equal(manifest, "config.total_scans", args.total_scans)
    _expect_equal(manifest, "config.profile_points", args.profile_points)
    _expect_number(
        manifest,
        "config.profile_half_width_mm",
        args.profile_half_width_mm,
    )
    _expect_number(
        manifest,
        "config.tangent_range_mm",
        args.tangent_range_mm,
    )
    _expect_number_pair(
        manifest,
        "config.profile_depth_range_mm",
        args.profile_depth_range_mm,
    )
    _expect_number_pair(
        manifest,
        "config.view_tilt_range_deg",
        args.view_tilt_range_deg,
    )
    _expect_number_pair(
        manifest,
        "config.view_azimuth_range_deg",
        args.view_azimuth_range_deg,
    )
    _expect_number_pair(
        manifest,
        "config.sensor_roll_range_deg",
        args.sensor_roll_range_deg,
    )
    rotation_scale_field = (
        "rotation_scale_deg"
        if schema_version >= 3
        else "prior_rotation_std_deg"
    )
    translation_scale_field = (
        "translation_scale_mm"
        if schema_version >= 3
        else "prior_translation_std_mm"
    )
    plane_normal_scale_field = (
        "plane_normal_scale_deg"
        if schema_version >= 3
        else "prior_plane_normal_std_deg"
    )
    plane_offset_scale_field = (
        "plane_offset_scale_mm"
        if schema_version >= 3
        else "prior_plane_offset_std_mm"
    )
    optional_selection_fields = (
        ("fisher_objective", args.fisher_objective, False),
        ("initial_random_scans", args.initial_random_scans, False),
        ("candidate_pool_size", args.candidate_pool_size, False),
        (
            "fisher_profile_noise_std_mm",
            args.fisher_profile_noise_std_mm,
            True,
        ),
        (
            rotation_scale_field,
            args.fisher_rotation_scale_deg,
            True,
        ),
        (
            translation_scale_field,
            args.fisher_translation_scale_mm,
            True,
        ),
        (
            plane_normal_scale_field,
            args.fisher_plane_normal_scale_deg,
            True,
        ),
        (
            plane_offset_scale_field,
            args.fisher_plane_offset_scale_mm,
            True,
        ),
        (
            "measurement_noise_std_mm",
            args.selection_measurement_noise_std_mm,
            True,
        ),
        (
            "measurement_noise_axis",
            args.selection_measurement_noise_axis,
            False,
        ),
        (
            "measurement_seed",
            args.selection_measurement_seed,
            False,
        ),
        (
            "initialization_seed",
            args.selection_initialization_seed,
            False,
        ),
        (
            "initial_translation_range_mm",
            args.selection_initial_translation_range_mm,
            True,
        ),
        (
            "initial_angle_range_deg",
            args.selection_initial_angle_range_deg,
            True,
        ),
        (
            "initial_rotation_perturbation",
            args.selection_initial_rotation_perturbation,
            False,
        ),
        (
            "initial_translation_perturbation",
            args.selection_initial_translation_perturbation,
            False,
        ),
        (
            "estimator_max_iterations",
            args.selection_estimator_max_iterations,
            False,
        ),
        (
            "estimator_tolerance",
            args.selection_estimator_tolerance,
            True,
        ),
    )
    for field, expected, is_number in optional_selection_fields:
        if expected is None:
            continue
        dotted_key = f"pose_selection_config.{field}"
        if is_number:
            _expect_number(manifest, dotted_key, expected)
        else:
            _expect_equal(manifest, dotted_key, expected)


def _validate_uniform_dataset(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)

    _expect_equal(manifest, "schema", UNIFORM_COMPARISON_SCHEMA)
    _expect_equal(manifest, "schema_version", 2)
    _expect_equal(manifest, "status", "complete")
    _expect_equal(manifest, "requested_trials", args.trials)
    _expect_equal(manifest, "completed_trials", args.trials)
    _expect_equal(manifest, "master_seed", args.seed)

    expected_collections = {
        "single_uniform": "single_plane_uniform",
        "three_random": "three_plane_random",
        "three_uniform": "three_plane_uniform",
    }
    expected_acquisition_modes = {
        "single_uniform": "single_plane_global_uniform",
        "three_random": "three_plane_global_random",
        "three_uniform": "three_plane_global_uniform",
    }
    _expect_equal(manifest, "collections", expected_collections)
    _expect_equal(
        manifest,
        "experiments.experiment_2",
        ["single_uniform", "three_random"],
    )
    _expect_equal(
        manifest,
        "experiments.additional",
        ["single_uniform", "three_uniform"],
    )
    _expect_equal(
        manifest,
        "scan_counts",
        {
            "single_uniform": args.total_scans,
            "three_random": args.total_scans,
            "three_uniform": 3 * args.total_scans,
        },
    )
    _expect_equal(
        manifest,
        "three_uniform_full_pose_set_applied_to_every_plane",
        True,
    )

    fair_values = (
        ("total_scans", args.total_scans, False),
        ("profile_points", args.profile_points, False),
        ("profile_half_width_mm", args.profile_half_width_mm, True),
        ("tangent_range_mm", args.tangent_range_mm, True),
        ("max_local_pose_trials", args.max_local_pose_trials, False),
        ("min_abs_plane_normal_z", args.min_abs_plane_normal_z, True),
        ("verification_atol", args.verification_atol, True),
    )
    for field, expected, is_number in fair_values:
        dotted_key = f"fair_config.{field}"
        if is_number:
            _expect_number(manifest, dotted_key, expected)
        else:
            _expect_equal(manifest, dotted_key, expected)
    for field, expected in (
        ("profile_depth_range_mm", args.profile_depth_range_mm),
        ("view_tilt_range_deg", args.random_view_tilt_range_deg),
        ("view_azimuth_range_deg", args.random_view_azimuth_range_deg),
        ("sensor_roll_range_deg", args.random_sensor_roll_range_deg),
        ("plane_angle_range_deg", args.plane_angle_range_deg),
        ("plane_center_xy_range_mm", args.plane_center_xy_range_mm),
        ("plane_center_z_range_mm", args.plane_center_z_range_mm),
    ):
        _expect_number_pair(manifest, f"fair_config.{field}", expected)

    uniform_values = (
        ("max_batches", args.uniform_max_batches, False),
        ("batch_multiplier", args.uniform_batch_multiplier, False),
    )
    for field, expected, is_number in uniform_values:
        dotted_key = f"uniform_config.{field}"
        if is_number:
            _expect_number(manifest, dotted_key, expected)
        else:
            _expect_equal(manifest, dotted_key, expected)
    for field, expected in (
        ("target_u_range_mm", args.uniform_target_u_range_mm),
        ("target_v_range_mm", args.uniform_target_v_range_mm),
        ("depth_range_mm", args.profile_depth_range_mm),
        ("tilt_range_deg", args.uniform_view_tilt_range_deg),
        ("azimuth_range_deg", args.uniform_view_azimuth_range_deg),
        ("roll_range_deg", args.uniform_sensor_roll_range_deg),
    ):
        _expect_number_pair(manifest, f"uniform_config.{field}", expected)

    for strategy, directory_name in expected_collections.items():
        collection_path = args.manifest.parent / directory_name / "collection.json"
        collection = _load_json(collection_path)
        _expect_equal(collection, "status", "complete")
        _expect_equal(
            collection,
            "generator_variant",
            UNIFORM_COMPARISON_SCHEMA,
        )
        _expect_equal(collection, "generator_schema_version", 2)
        _expect_equal(collection, "strategy", strategy)
        _expect_equal(
            collection,
            "acquisition_mode",
            expected_acquisition_modes[strategy],
        )
        _expect_equal(collection, "requested_trials", args.trials)
        _expect_equal(collection, "completed_trials", args.trials)
        _expect_equal(collection, "master_seed", args.seed)
        _expect_equal(
            collection,
            "scans_per_trial",
            3 * args.total_scans
            if strategy == "three_uniform"
            else args.total_scans,
        )
        _expect_equal(
            collection,
            "full_uniform_pose_set_applied_to_every_plane",
            strategy == "three_uniform",
        )
        entries = _nested(collection, "trials")
        if not isinstance(entries, list) or len(entries) != args.trials:
            raise ValueError(
                f"{strategy} collection trial-count mismatch: expected "
                f"{args.trials}, got "
                f"{len(entries) if isinstance(entries, list) else type(entries)}"
            )
        trial_indices = []
        expected_scans = (
            3 * args.total_scans
            if strategy == "three_uniform"
            else args.total_scans
        )
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{strategy} has a non-object trial entry")
            trial_indices.append(int(_nested(entry, "trial_index")))
            relative_path = _nested(entry, "relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(
                    f"{strategy} has an invalid trial relative_path"
                )
            dataset_manifest = _load_json(
                collection_path.parent / relative_path / "manifest.json"
            )
            _expect_equal(
                dataset_manifest,
                "acquisition_mode",
                expected_acquisition_modes[strategy],
            )
            _expect_equal(
                dataset_manifest,
                "counts.scans",
                expected_scans,
            )
        if sorted(trial_indices) != list(range(args.trials)):
            raise ValueError(
                f"{strategy} trial indices are not contiguous from zero"
            )


def _validate_single_uniform_fisher_dataset(
    args: argparse.Namespace,
) -> None:
    manifest = _load_json(args.manifest)

    _expect_equal(manifest, "schema", SINGLE_UNIFORM_FISHER_SCHEMA)
    _expect_equal(manifest, "schema_version", 1)
    _expect_equal(manifest, "status", "complete")
    _expect_equal(manifest, "requested_trials", args.trials)
    _expect_equal(manifest, "completed_trials", args.trials)
    _expect_equal(manifest, "master_seed", args.seed)
    expected_collections = {
        "single_uniform": "single_plane_uniform",
        "single_fisher": "single_plane_fisher",
    }
    _expect_equal(manifest, "collections", expected_collections)

    fair_values = (
        ("total_scans", args.total_scans, False),
        ("profile_points", args.profile_points, False),
        ("profile_half_width_mm", args.profile_half_width_mm, True),
        ("tangent_range_mm", args.tangent_range_mm, True),
        (
            "max_local_pose_trials",
            max(
                args.candidate_pool_size
                * args.candidate_batch_multiplier,
                200_000,
            ),
            False,
        ),
        ("min_abs_plane_normal_z", args.min_abs_plane_normal_z, True),
        ("verification_atol", args.verification_atol, True),
        ("min_effective_cross_plane_angle_deg", 0.0, True),
    )
    for field, expected, is_number in fair_values:
        dotted_key = f"fair_config.{field}"
        if is_number:
            _expect_number(manifest, dotted_key, expected)
        else:
            _expect_equal(manifest, dotted_key, expected)
    for field, expected in (
        ("profile_depth_range_mm", args.profile_depth_range_mm),
        ("view_tilt_range_deg", args.candidate_view_tilt_range_deg),
        ("view_azimuth_range_deg", args.candidate_view_azimuth_range_deg),
        ("sensor_roll_range_deg", args.candidate_sensor_roll_range_deg),
        ("plane_angle_range_deg", args.plane_angle_range_deg),
        ("plane_center_xy_range_mm", args.plane_center_xy_range_mm),
        ("plane_center_z_range_mm", args.plane_center_z_range_mm),
    ):
        _expect_number_pair(manifest, f"fair_config.{field}", expected)

    candidate_values = (
        ("max_batches", args.candidate_max_batches),
        ("batch_multiplier", args.candidate_batch_multiplier),
    )
    for field, expected in candidate_values:
        _expect_equal(manifest, f"candidate_config.{field}", expected)
    for field, expected in (
        ("target_u_range_mm", args.candidate_target_u_range_mm),
        ("target_v_range_mm", args.candidate_target_v_range_mm),
        ("depth_range_mm", args.profile_depth_range_mm),
        ("tilt_range_deg", args.candidate_view_tilt_range_deg),
        ("azimuth_range_deg", args.candidate_view_azimuth_range_deg),
        ("roll_range_deg", args.candidate_sensor_roll_range_deg),
    ):
        _expect_number_pair(manifest, f"candidate_config.{field}", expected)

    selection_values = (
        ("initial_random_scans", args.initial_scans, False),
        ("candidate_pool_size", args.candidate_pool_size, False),
        (
            "fisher_profile_noise_std_mm",
            args.fisher_profile_noise_std_mm,
            True,
        ),
        ("rotation_scale_deg", args.fisher_rotation_scale_deg, True),
        ("translation_scale_mm", args.fisher_translation_scale_mm, True),
        ("plane_normal_scale_deg", args.fisher_plane_normal_scale_deg, True),
        ("plane_offset_scale_mm", args.fisher_plane_offset_scale_mm, True),
        ("fisher_objective", args.fisher_objective, False),
        ("measurement_noise_std_mm", args.measurement_noise_std_mm, True),
        ("measurement_noise_axis", args.measurement_noise_axis, False),
        ("measurement_seed", args.measurement_seed, False),
        ("initialization_seed", args.initialization_seed, False),
        (
            "initial_translation_range_mm",
            args.initial_translation_range_mm,
            True,
        ),
        ("initial_angle_range_deg", args.initial_angle_range_deg, True),
        (
            "initial_rotation_perturbation",
            args.initial_rotation_perturbation,
            False,
        ),
        (
            "initial_translation_perturbation",
            args.initial_translation_perturbation,
            False,
        ),
        ("estimator_max_iterations", args.estimator_max_iterations, False),
        ("estimator_tolerance", args.estimator_tolerance, True),
    )
    for field, expected, is_number in selection_values:
        dotted_key = f"pose_selection_config.{field}"
        if is_number:
            _expect_number(manifest, dotted_key, expected)
        else:
            _expect_equal(manifest, dotted_key, expected)

    for key in (
        "shared_truth",
        "shared_physical_plane",
        "shared_candidate_bank",
        "shared_initial_bootstrap",
        "shared_candidate_keyed_noise",
        "same_total_scans",
    ):
        _expect_equal(manifest, f"comparison_rules.{key}", True)
    _expect_equal(
        manifest,
        "comparison_rules.uniform_policy",
        "plane_relative_parameter_maximin",
    )
    _expect_equal(
        manifest,
        "comparison_rules.fisher_policy",
        "estimated_state_active_greedy_marginal_handeye_"
        + args.fisher_objective,
    )

    comparison_trials = _nested(manifest, "trials")
    if (
        not isinstance(comparison_trials, list)
        or len(comparison_trials) != args.trials
    ):
        raise ValueError("comparison trial-count mismatch")
    for expected_index, trial in enumerate(comparison_trials):
        if not isinstance(trial, dict):
            raise ValueError("comparison has a non-object trial entry")
        _expect_equal(trial, "trial_index", expected_index)
        _expect_equal(trial, "candidate_pool_size", args.candidate_pool_size)
        _expect_equal(trial, "same_initial_bootstrap", True)
        _expect_equal(
            trial,
            "same_measurement_noise_for_same_candidate",
            True,
        )
        _expect_equal(trial, "same_total_scan_count", True)
        initial_ids = _nested(trial, "initial_candidate_ids")
        if not isinstance(initial_ids, list) or len(initial_ids) != args.initial_scans:
            raise ValueError(
                f"trial {expected_index} bootstrap-count mismatch"
            )

    expected_trial_modes = {
        "single_uniform": "single_plane_global_uniform",
        "single_fisher": "single_plane_global_fisher",
    }
    for strategy, directory_name in expected_collections.items():
        collection_path = args.manifest.parent / directory_name / "collection.json"
        collection = _load_json(collection_path)
        _expect_equal(collection, "status", "complete")
        _expect_equal(collection, "generator_variant", SINGLE_UNIFORM_FISHER_SCHEMA)
        _expect_equal(collection, "generator_schema_version", 1)
        _expect_equal(collection, "strategy", strategy)
        _expect_equal(collection, "acquisition_mode", strategy)
        _expect_equal(collection, "requested_trials", args.trials)
        _expect_equal(collection, "completed_trials", args.trials)
        _expect_equal(collection, "master_seed", args.seed)
        _expect_equal(collection, "profile_state", "measured")
        _expect_equal(
            collection,
            "noise_applied",
            args.measurement_noise_std_mm > 0.0,
        )
        entries = _nested(collection, "trials")
        if not isinstance(entries, list) or len(entries) != args.trials:
            raise ValueError(f"{strategy} collection trial-count mismatch")
        for expected_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{strategy} has a non-object trial entry")
            _expect_equal(entry, "trial_index", expected_index)
            relative_path = _nested(entry, "relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(
                    f"{strategy} has an invalid trial relative_path"
                )
            dataset_manifest = _load_json(
                collection_path.parent / relative_path / "manifest.json"
            )
            _expect_equal(
                dataset_manifest,
                "acquisition_mode",
                expected_trial_modes[strategy],
            )
            _expect_equal(dataset_manifest, "counts.scans", args.total_scans)
            _expect_equal(dataset_manifest, "profile_state", "measured")


def _resolve_config_path(text: Any) -> Path:
    if not isinstance(text, str) or not text:
        raise ValueError(f"invalid config.collection value: {text!r}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        # Runner paths are interpreted from the repository working directory.
        path = Path.cwd() / path
    return path.resolve()


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"trials CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"trials CSV has no header: {path}")
        if "trial_index" not in reader.fieldnames:
            raise ValueError(f"trials CSV has no trial_index column: {path}")
        return sum(1 for _ in reader)


def _validate_result(args: argparse.Namespace) -> None:
    summary = _load_json(args.summary)

    _expect_equal(summary, "requested_trials", args.trials)
    _expect_equal(summary, "config.mode", args.mode)
    _expect_equal(summary, "config.seed", args.seed)
    _expect_equal(summary, "config.noise_axis", args.noise_axis)
    _expect_number(summary, "config.noise_std_mm", args.noise_std_mm)
    _expect_equal(summary, "config.init_mode", "carlson")
    _expect_number(
        summary,
        "config.init_translation_range_mm",
        args.init_translation_range_mm,
    )
    _expect_number(
        summary,
        "config.init_angle_range_deg",
        args.init_angle_range_deg,
    )
    _expect_equal(
        summary,
        "config.init_rotation_perturbation",
        args.init_rotation_perturbation,
    )
    _expect_equal(
        summary,
        "config.init_translation_perturbation",
        args.init_translation_perturbation,
    )
    _expect_equal(summary, "config.max_iter", args.max_iter)
    _expect_number(summary, "config.tol", args.tol)
    if args.max_scans_per_trial is not None:
        _expect_equal(
            summary,
            "config.max_scans_per_trial",
            args.max_scans_per_trial,
        )
    if args.mode in {
        "iterative_refit_nonlinear",
        "iterative_joint_nonlinear",
    }:
        _expect_equal(
            summary,
            "config.nonlinear_loss",
            args.nonlinear_loss,
        )
        _expect_number(
            summary,
            "config.nonlinear_f_scale_mm",
            args.nonlinear_f_scale_mm,
        )
        _expect_equal(
            summary,
            "config.nonlinear_max_nfev",
            args.nonlinear_max_nfev,
        )
        _expect_number(
            summary,
            "config.nonlinear_ftol",
            args.nonlinear_ftol,
        )
        _expect_number(
            summary,
            "config.nonlinear_xtol",
            args.nonlinear_xtol,
        )
        _expect_number(
            summary,
            "config.nonlinear_gtol",
            args.nonlinear_gtol,
        )

    configured_collection = _resolve_config_path(
        _nested(summary, "config.collection"),
    )
    expected_collection = args.collection.expanduser().resolve()
    if configured_collection != expected_collection:
        raise ValueError(
            "config.collection mismatch: expected "
            f"{expected_collection}, got {configured_collection}"
        )

    trials_path = args.summary.parent / "trials.csv"
    row_count = _csv_row_count(trials_path)
    if row_count != args.trials:
        raise ValueError(
            f"trials.csv row-count mismatch: expected {args.trials}, "
            f"got {row_count}"
        )


def _pair(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(name, type=float, nargs=2, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--manifest", type=Path, required=True)
    dataset.add_argument("--trials", type=int, required=True)
    dataset.add_argument("--seed", type=int, required=True)
    dataset.add_argument("--total-scans", type=int, required=True)
    dataset.add_argument("--profile-points", type=int, required=True)
    dataset.add_argument(
        "--profile-half-width-mm",
        type=float,
        required=True,
    )
    dataset.add_argument("--tangent-range-mm", type=float, required=True)
    _pair(dataset, "--profile-depth-range-mm")
    _pair(dataset, "--view-tilt-range-deg")
    _pair(dataset, "--view-azimuth-range-deg")
    _pair(dataset, "--sensor-roll-range-deg")
    dataset.add_argument("--initial-random-scans", type=int)
    dataset.add_argument("--candidate-pool-size", type=int)
    dataset.add_argument(
        "--fisher-objective",
        choices=("d_optimal", "e_optimal"),
    )
    dataset.add_argument("--fisher-profile-noise-std-mm", type=float)
    dataset.add_argument(
        "--fisher-rotation-scale-deg",
        "--fisher-prior-rotation-std-deg",
        dest="fisher_rotation_scale_deg",
        type=float,
    )
    dataset.add_argument(
        "--fisher-translation-scale-mm",
        "--fisher-prior-translation-std-mm",
        dest="fisher_translation_scale_mm",
        type=float,
    )
    dataset.add_argument(
        "--fisher-plane-normal-scale-deg",
        "--fisher-prior-plane-normal-std-deg",
        dest="fisher_plane_normal_scale_deg",
        type=float,
    )
    dataset.add_argument(
        "--fisher-plane-offset-scale-mm",
        "--fisher-prior-plane-offset-std-mm",
        dest="fisher_plane_offset_scale_mm",
        type=float,
    )
    dataset.add_argument("--selection-measurement-noise-std-mm", type=float)
    dataset.add_argument("--selection-measurement-noise-axis")
    dataset.add_argument("--selection-measurement-seed", type=int)
    dataset.add_argument("--selection-initialization-seed", type=int)
    dataset.add_argument(
        "--selection-initial-translation-range-mm",
        type=float,
    )
    dataset.add_argument("--selection-initial-angle-range-deg", type=float)
    dataset.add_argument("--selection-initial-rotation-perturbation")
    dataset.add_argument("--selection-initial-translation-perturbation")
    dataset.add_argument(
        "--selection-estimator-max-iterations",
        type=int,
    )
    dataset.add_argument("--selection-estimator-tolerance", type=float)
    dataset.set_defaults(run=_validate_dataset)

    uniform_dataset = subparsers.add_parser("uniform-dataset")
    uniform_dataset.add_argument("--manifest", type=Path, required=True)
    uniform_dataset.add_argument("--trials", type=int, required=True)
    uniform_dataset.add_argument("--seed", type=int, required=True)
    uniform_dataset.add_argument("--total-scans", type=int, required=True)
    uniform_dataset.add_argument("--profile-points", type=int, required=True)
    uniform_dataset.add_argument(
        "--profile-half-width-mm",
        type=float,
        required=True,
    )
    uniform_dataset.add_argument(
        "--tangent-range-mm",
        type=float,
        required=True,
    )
    _pair(uniform_dataset, "--profile-depth-range-mm")
    _pair(uniform_dataset, "--random-view-tilt-range-deg")
    _pair(uniform_dataset, "--random-view-azimuth-range-deg")
    _pair(uniform_dataset, "--random-sensor-roll-range-deg")
    _pair(uniform_dataset, "--uniform-target-u-range-mm")
    _pair(uniform_dataset, "--uniform-target-v-range-mm")
    _pair(uniform_dataset, "--uniform-view-tilt-range-deg")
    _pair(uniform_dataset, "--uniform-view-azimuth-range-deg")
    _pair(uniform_dataset, "--uniform-sensor-roll-range-deg")
    _pair(uniform_dataset, "--plane-angle-range-deg")
    _pair(uniform_dataset, "--plane-center-xy-range-mm")
    _pair(uniform_dataset, "--plane-center-z-range-mm")
    uniform_dataset.add_argument(
        "--uniform-max-batches",
        type=int,
        required=True,
    )
    uniform_dataset.add_argument(
        "--uniform-batch-multiplier",
        type=int,
        required=True,
    )
    uniform_dataset.add_argument(
        "--max-local-pose-trials",
        type=int,
        required=True,
    )
    uniform_dataset.add_argument(
        "--min-abs-plane-normal-z",
        type=float,
        required=True,
    )
    uniform_dataset.add_argument(
        "--verification-atol",
        type=float,
        required=True,
    )
    uniform_dataset.set_defaults(run=_validate_uniform_dataset)

    single_uniform_fisher = subparsers.add_parser(
        "single-uniform-fisher-dataset"
    )
    single_uniform_fisher.add_argument("--manifest", type=Path, required=True)
    single_uniform_fisher.add_argument("--trials", type=int, required=True)
    single_uniform_fisher.add_argument("--seed", type=int, required=True)
    single_uniform_fisher.add_argument(
        "--total-scans", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--initial-scans", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--candidate-pool-size", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--profile-points", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--profile-half-width-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--tangent-range-mm", type=float, required=True
    )
    _pair(single_uniform_fisher, "--profile-depth-range-mm")
    _pair(single_uniform_fisher, "--candidate-target-u-range-mm")
    _pair(single_uniform_fisher, "--candidate-target-v-range-mm")
    _pair(single_uniform_fisher, "--candidate-view-tilt-range-deg")
    _pair(single_uniform_fisher, "--candidate-view-azimuth-range-deg")
    _pair(single_uniform_fisher, "--candidate-sensor-roll-range-deg")
    single_uniform_fisher.add_argument(
        "--candidate-max-batches", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--candidate-batch-multiplier", type=int, required=True
    )
    _pair(single_uniform_fisher, "--plane-angle-range-deg")
    _pair(single_uniform_fisher, "--plane-center-xy-range-mm")
    _pair(single_uniform_fisher, "--plane-center-z-range-mm")
    single_uniform_fisher.add_argument(
        "--min-abs-plane-normal-z", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--verification-atol", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--fisher-objective",
        choices=("d_optimal", "e_optimal"),
        required=True,
    )
    single_uniform_fisher.add_argument(
        "--fisher-profile-noise-std-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--fisher-rotation-scale-deg", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--fisher-translation-scale-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--fisher-plane-normal-scale-deg", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--fisher-plane-offset-scale-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--measurement-noise-std-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--measurement-noise-axis", choices=("z", "xz"), required=True
    )
    single_uniform_fisher.add_argument(
        "--measurement-seed", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--initialization-seed", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--initial-translation-range-mm", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--initial-angle-range-deg", type=float, required=True
    )
    single_uniform_fisher.add_argument(
        "--initial-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"),
        required=True,
    )
    single_uniform_fisher.add_argument(
        "--initial-translation-perturbation",
        choices=("direction_norm", "box_xyz"),
        required=True,
    )
    single_uniform_fisher.add_argument(
        "--estimator-max-iterations", type=int, required=True
    )
    single_uniform_fisher.add_argument(
        "--estimator-tolerance", type=float, required=True
    )
    single_uniform_fisher.set_defaults(
        run=_validate_single_uniform_fisher_dataset
    )

    result = subparsers.add_parser("result")
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument(
        "--mode",
        choices=(
            "iterative",
            "iterative_refit_nonlinear",
            "iterative_joint_nonlinear",
        ),
        default="iterative",
    )
    result.add_argument("--collection", type=Path, required=True)
    result.add_argument("--trials", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--noise-axis", required=True)
    result.add_argument("--noise-std-mm", type=float, required=True)
    result.add_argument(
        "--init-translation-range-mm",
        type=float,
        required=True,
    )
    result.add_argument(
        "--init-angle-range-deg",
        type=float,
        required=True,
    )
    result.add_argument(
        "--init-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"),
        required=True,
    )
    result.add_argument(
        "--init-translation-perturbation",
        choices=("direction_norm", "box_xyz"),
        required=True,
    )
    result.add_argument("--max-iter", type=int, required=True)
    result.add_argument("--tol", type=float, required=True)
    result.add_argument("--max-scans-per-trial", type=int)
    result.add_argument("--nonlinear-loss", default="linear")
    result.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    result.add_argument("--nonlinear-max-nfev", type=int, default=300)
    result.add_argument("--nonlinear-ftol", type=float, default=1e-10)
    result.add_argument("--nonlinear-xtol", type=float, default=1e-10)
    result.add_argument("--nonlinear-gtol", type=float, default=1e-10)
    result.set_defaults(run=_validate_result)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except ValueError as exc:
        print(
            f"ERROR: cached experiment artifact is incompatible: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
