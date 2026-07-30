from __future__ import annotations

import json

from main.validate_experiment_artifact import main as validate_artifact


def _dataset_arguments(manifest, trials: int = 3) -> list[str]:
    return [
        "dataset",
        "--manifest",
        str(manifest),
        "--trials",
        str(trials),
        "--seed",
        "17",
        "--total-scans",
        "108",
        "--profile-points",
        "100",
        "--profile-half-width-mm",
        "25",
        "--tangent-range-mm",
        "100",
        "--profile-depth-range-mm",
        "60",
        "150",
        "--view-tilt-range-deg",
        "48",
        "62",
        "--view-azimuth-range-deg",
        "35",
        "55",
        "--sensor-roll-range-deg",
        "-20",
        "20",
    ]


def test_dataset_validator_accepts_matching_shared_global_manifest(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "laser_handeye.shared_global_feasible_pose_comparison"
                ),
                "schema_version": 1,
                "status": "complete",
                "pose_sampling_frame": "shared_target_global",
                "single_and_three_robot_poses_identical": True,
                "requested_trials": 3,
                "completed_trials": 3,
                "master_seed": 17,
                "pose_selection_config": {
                    "fisher_objective": "d_optimal",
                    "initial_random_scans": 9,
                    "candidate_pool_size": 324,
                    "fisher_profile_noise_std_mm": 0.2,
                    "prior_rotation_std_deg": 10.0,
                    "prior_translation_std_mm": 200.0,
                    "prior_plane_normal_std_deg": 20.0,
                    "prior_plane_offset_std_mm": 100.0,
                },
                "config": {
                    "total_scans": 108,
                    "profile_points": 100,
                    "profile_half_width_mm": 25.0,
                    "tangent_range_mm": 100.0,
                    "profile_depth_range_mm": [60.0, 150.0],
                    "view_tilt_range_deg": [48.0, 62.0],
                    "view_azimuth_range_deg": [35.0, 55.0],
                    "sensor_roll_range_deg": [-20.0, 20.0],
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = _dataset_arguments(manifest_path) + [
        "--fisher-objective",
        "d_optimal",
        "--initial-random-scans",
        "9",
        "--candidate-pool-size",
        "324",
        "--fisher-profile-noise-std-mm",
        "0.2",
        "--fisher-prior-rotation-std-deg",
        "10",
        "--fisher-prior-translation-std-mm",
        "200",
        "--fisher-prior-plane-normal-std-deg",
        "20",
        "--fisher-prior-plane-offset-std-mm",
        "100",
    ]
    assert validate_artifact(arguments) == 0

    wrong_pool = list(arguments)
    wrong_pool[wrong_pool.index("--candidate-pool-size") + 1] = "648"
    assert validate_artifact(wrong_pool) == 1
    assert validate_artifact(
        _dataset_arguments(manifest_path, trials=100)
    ) == 1


def test_dataset_validator_accepts_fisher_comparison_schema_v2(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "laser_handeye.shared_global_feasible_pose_comparison"
                ),
                "schema_version": 2,
                "status": "complete",
                "pose_sampling_frame": "shared_target_global",
                "random_single_and_three_robot_poses_identical": True,
                "all_branches_share_initial_random_poses": True,
                "requested_trials": 3,
                "completed_trials": 3,
                "master_seed": 17,
                "config": {
                    "total_scans": 108,
                    "profile_points": 100,
                    "profile_half_width_mm": 25.0,
                    "tangent_range_mm": 100.0,
                    "profile_depth_range_mm": [60.0, 150.0],
                    "view_tilt_range_deg": [48.0, 62.0],
                    "view_azimuth_range_deg": [35.0, 55.0],
                    "sensor_roll_range_deg": [-20.0, 20.0],
                },
            }
        ),
        encoding="utf-8",
    )

    assert validate_artifact(_dataset_arguments(manifest_path)) == 0


def test_dataset_validator_checks_online_fisher_schema_v3_scales(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "laser_handeye.shared_global_feasible_pose_comparison"
                ),
                "schema_version": 3,
                "status": "complete",
                "pose_sampling_frame": "shared_target_global",
                "random_single_and_three_robot_poses_identical": True,
                "all_branches_share_initial_random_poses": True,
                "requested_trials": 3,
                "completed_trials": 3,
                "master_seed": 17,
                "pose_selection_config": {
                    "fisher_objective": "e_optimal",
                    "initial_random_scans": 9,
                    "candidate_pool_size": 324,
                    "fisher_profile_noise_std_mm": 0.25,
                    "rotation_scale_deg": 2.0,
                    "translation_scale_mm": 10.0,
                    "plane_normal_scale_deg": 20.0,
                    "plane_offset_scale_mm": 100.0,
                    "measurement_noise_std_mm": 0.25,
                    "measurement_noise_axis": "xz",
                    "measurement_seed": 1701,
                    "initialization_seed": 1701,
                    "initial_translation_range_mm": 10.0,
                    "initial_angle_range_deg": 2.0,
                    "initial_rotation_perturbation": "axis_angle",
                    "initial_translation_perturbation": "direction_norm",
                    "estimator_max_iterations": 30,
                    "estimator_tolerance": 1e-7,
                },
                "config": {
                    "total_scans": 108,
                    "profile_points": 100,
                    "profile_half_width_mm": 25.0,
                    "tangent_range_mm": 100.0,
                    "profile_depth_range_mm": [60.0, 150.0],
                    "view_tilt_range_deg": [48.0, 62.0],
                    "view_azimuth_range_deg": [35.0, 55.0],
                    "sensor_roll_range_deg": [-20.0, 20.0],
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = _dataset_arguments(manifest_path) + [
        "--fisher-objective",
        "e_optimal",
        "--initial-random-scans",
        "9",
        "--candidate-pool-size",
        "324",
        "--fisher-profile-noise-std-mm",
        "0.25",
        "--fisher-rotation-scale-deg",
        "2",
        "--fisher-translation-scale-mm",
        "10",
        "--fisher-plane-normal-scale-deg",
        "20",
        "--fisher-plane-offset-scale-mm",
        "100",
        "--selection-measurement-noise-std-mm",
        "0.25",
        "--selection-measurement-noise-axis",
        "xz",
        "--selection-measurement-seed",
        "1701",
        "--selection-initialization-seed",
        "1701",
        "--selection-initial-translation-range-mm",
        "10",
        "--selection-initial-angle-range-deg",
        "2",
        "--selection-initial-rotation-perturbation",
        "axis_angle",
        "--selection-initial-translation-perturbation",
        "direction_norm",
        "--selection-estimator-max-iterations",
        "30",
        "--selection-estimator-tolerance",
        "1e-7",
    ]
    assert validate_artifact(arguments) == 0

    wrong_scale = list(arguments)
    wrong_scale[
        wrong_scale.index("--fisher-translation-scale-mm") + 1
    ] = "20"
    assert validate_artifact(wrong_scale) == 1


def test_result_validator_checks_config_and_trials_row_count(tmp_path) -> None:
    collection = tmp_path / "dataset" / "single_plane"
    collection.mkdir(parents=True)
    result = tmp_path / "result"
    result.mkdir()

    summary_path = result / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "requested_trials": 2,
                "config": {
                    "collection": str(collection),
                    "mode": "iterative",
                    "seed": 1701,
                    "noise_axis": "xz",
                    "noise_std_mm": 0.25,
                    "init_mode": "carlson",
                    "init_translation_range_mm": 10.0,
                    "init_angle_range_deg": 2.0,
                    "init_rotation_perturbation": "axis_angle",
                    "init_translation_perturbation": "direction_norm",
                    "max_iter": 1500,
                    "tol": 1e-5,
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "trials.csv").write_text(
        "trial_index,translation_error_mm,rotation_error_deg\n"
        "0,0.1,0.01\n"
        "1,0.2,0.02\n",
        encoding="utf-8",
    )

    arguments = [
        "result",
        "--summary",
        str(summary_path),
        "--collection",
        str(collection),
        "--trials",
        "2",
        "--seed",
        "1701",
        "--noise-axis",
        "xz",
        "--noise-std-mm",
        "0.25",
        "--init-translation-range-mm",
        "10",
        "--init-angle-range-deg",
        "2",
        "--init-rotation-perturbation",
        "axis_angle",
        "--init-translation-perturbation",
        "direction_norm",
        "--max-iter",
        "1500",
        "--tol",
        "1e-5",
    ]
    assert validate_artifact(arguments) == 0

    wrong_noise_arguments = list(arguments)
    wrong_noise_arguments[
        wrong_noise_arguments.index("--noise-std-mm") + 1
    ] = "0.4"
    assert validate_artifact(wrong_noise_arguments) == 1
