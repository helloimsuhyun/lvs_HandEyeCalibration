from __future__ import annotations

from dataclasses import replace
import json

import numpy as np

import main.generate_independent_random_plane_comparison as comparison_generator
from laser_handeye.calibration_dataset import (
    load_calibration_dataset,
    logical_dataset_sha256,
)
from laser_handeye.se3 import transform_points
from main.generate_independent_random_plane_comparison import (
    CandidateKinematics,
    COMPARISON_SCHEMA,
    POSE_SAMPLING_FRAME,
    FairGenerationConfig,
    PoseSelectionConfig,
    _generate_fisher_random_trial,
    _generate_shared_global_trial,
    _validate_args,
    build_parser,
    main as generate_comparison,
)


def _small_config() -> FairGenerationConfig:
    return FairGenerationConfig(
        total_scans=18,
        profile_points=16,
        profile_half_width_mm=25.0,
        tangent_range_mm=100.0,
        profile_depth_range_mm=(60.0, 150.0),
        view_tilt_range_deg=(48.0, 62.0),
        view_azimuth_range_deg=(35.0, 55.0),
        sensor_roll_range_deg=(-20.0, 20.0),
        plane_angle_range_deg=(-15.0, 15.0),
        plane_center_xy_range_mm=(-100.0, 100.0),
        plane_center_z_range_mm=(400.0, 550.0),
        max_local_pose_trials=20_000,
        min_abs_plane_normal_z=1e-4,
        verification_atol=1e-8,
        min_effective_cross_plane_angle_deg=60.0,
    )


def _pose_stack(dataset) -> np.ndarray:
    return np.stack([scan.T_base_ef for scan in dataset.scans], axis=0)


def _small_selection_config() -> PoseSelectionConfig:
    return PoseSelectionConfig(
        initial_random_scans=9,
        candidate_pool_size=36,
        fisher_profile_noise_std_mm=0.25,
        rotation_scale_deg=2.0,
        translation_scale_mm=10.0,
        plane_normal_scale_deg=20.0,
        plane_offset_scale_mm=100.0,
        fisher_objective="d_optimal",
        initial_translation_range_mm=10.0,
        initial_angle_range_deg=2.0,
        estimator_max_iterations=30,
    )


def _assert_truth_plane_residuals(dataset) -> None:
    assert dataset.truth is not None
    assert dataset.truth.T_ef_s_true is not None
    planes = {plane.plane_id: plane for plane in dataset.truth.planes}
    for scan in dataset.scans:
        points_ef = transform_points(
            dataset.truth.T_ef_s_true,
            scan.valid_points_s,
        )
        points_base = transform_points(scan.T_base_ef, points_ef)
        plane = planes[scan.plane_id]
        residuals = points_base @ plane.normal_base - plane.offset_mm
        assert np.max(np.abs(residuals)) < 1e-8


def test_cli_disables_outcome_based_diversity_rejection_by_default(
    tmp_path,
) -> None:
    args = build_parser().parse_args(["--output-dir", str(tmp_path / "data")])
    config = _validate_args(args)
    assert config.min_effective_cross_plane_angle_deg == 0.0


def test_wide_negative_control_keeps_below_60_degree_trial() -> None:
    config = replace(
        _small_config(),
        total_scans=108,
        view_tilt_range_deg=(25.0, 82.0),
        view_azimuth_range_deg=(10.0, 80.0),
        sensor_roll_range_deg=(-180.0, 180.0),
        min_effective_cross_plane_angle_deg=0.0,
    )
    _, _, comparison = _generate_shared_global_trial(
        config=config,
        master_seed=17,
        trial_index=3,
    )
    assert (
        comparison["diversity_metrics"][
            "three_cross_plane_effective_normal_median_deg"
        ]
        < 60.0
    )


def test_shared_global_trial_preserves_three_plane_calibration_diversity() -> None:
    single, three, comparison = _generate_shared_global_trial(
        config=_small_config(),
        master_seed=17,
        trial_index=0,
    )

    assert comparison["pose_libraries_are_shared"] is True
    assert comparison["pose_libraries_are_independent"] is False
    assert comparison["single_and_three_robot_poses_identical"] is True
    assert comparison["paired_robot_pose_max_abs_difference"] == 0.0
    assert np.array_equal(_pose_stack(single), _pose_stack(three))
    assert np.array_equal(
        single.truth.T_ef_s_true,
        three.truth.T_ef_s_true,
    )

    three_plane_ids = np.asarray(
        [scan.plane_id for scan in three.scans],
        dtype=int,
    )
    unique, counts = np.unique(three_plane_ids, return_counts=True)
    assert np.array_equal(unique, np.arange(3))
    assert np.array_equal(counts, np.full(3, 6))

    metrics = comparison["diversity_metrics"]
    assert np.isclose(metrics["base_plane_normal_pairwise_min_deg"], 90.0)
    assert metrics["three_cross_plane_effective_normal_median_deg"] > 80.0
    assert (
        metrics["three_centered_effective_normal_condition"]
        < metrics["single_centered_effective_normal_condition"]
    )
    assert (
        metrics["centered_effective_normal_condition_improvement_ratio"]
        > 5.0
    )

    assert single.metadata["pose_sampling_frame"] == POSE_SAMPLING_FRAME
    assert three.metadata["pose_sampling_frame"] == POSE_SAMPLING_FRAME
    assert {
        scan.meta["pose_sampling_frame"] for scan in single.scans
    } == {POSE_SAMPLING_FRAME}
    assert all(scan.meta["paired_global_pose"] for scan in three.scans)
    _assert_truth_plane_residuals(single)
    _assert_truth_plane_residuals(three)


def test_shared_global_trial_is_deterministic() -> None:
    first_single, first_three, first_comparison = (
        _generate_shared_global_trial(
            config=_small_config(),
            master_seed=29,
            trial_index=3,
        )
    )
    second_single, second_three, second_comparison = (
        _generate_shared_global_trial(
            config=_small_config(),
            master_seed=29,
            trial_index=3,
        )
    )

    assert logical_dataset_sha256(first_single) == logical_dataset_sha256(
        second_single
    )
    assert logical_dataset_sha256(first_three) == logical_dataset_sha256(
        second_three
    )
    assert first_comparison == second_comparison


def test_fisher_and_random_share_bootstrap_and_compare_both_geometries() -> None:
    datasets, comparison = _generate_fisher_random_trial(
        config=_small_config(),
        selection_config=_small_selection_config(),
        master_seed=17,
        trial_index=0,
    )
    assert set(datasets) == {
        "single_random",
        "single_fisher",
        "three_random",
        "three_fisher",
    }
    initial_count = _small_selection_config().initial_random_scans
    initial_pose_stack = _pose_stack(datasets["single_random"])[
        :initial_count
    ]
    for dataset in datasets.values():
        assert np.array_equal(
            _pose_stack(dataset)[:initial_count],
            initial_pose_stack,
        )
    for geometry in ("single", "three"):
        random_dataset = datasets[f"{geometry}_random"]
        fisher_dataset = datasets[f"{geometry}_fisher"]
        for random_scan, fisher_scan in zip(
            random_dataset.scans[:initial_count],
            fisher_dataset.scans[:initial_count],
        ):
            assert np.array_equal(
                random_scan.points_s,
                fisher_scan.points_s,
            )

    # The historical single/three random baseline remains exactly pose-paired.
    assert np.array_equal(
        _pose_stack(datasets["single_random"]),
        _pose_stack(datasets["three_random"]),
    )
    selections = comparison["selections"]
    assert selections["single_random"]["selected_candidate_ids"] == (
        selections["three_random"]["selected_candidate_ids"]
    )
    for key in ("three_random", "three_fisher"):
        assert selections[key]["selected_assigned_plane_counts"] == {
            "0": 6,
            "1": 6,
            "2": 6,
        }

    # At the first post-bootstrap decision both strategies have the identical
    # posterior; Fisher must score at least as highly as that random choice.
    for geometry in ("single", "three"):
        fisher_trace = selections[f"{geometry}_fisher"]["selection_trace"]
        random_trace = selections[f"{geometry}_random"]["selection_trace"]
        assert fisher_trace[0]["bootstrap_estimate_sha256"] == (
            random_trace[0]["bootstrap_estimate_sha256"]
        )
        assert (
            fisher_trace[initial_count]["information_gain_nats"]
            >= random_trace[initial_count]["information_gain_nats"]
        )
        assert (
            selections[f"{geometry}_fisher"]["final_information"][
                "joint_parameter_dimension"
            ]
            == (9 if geometry == "single" else 15)
        )


def test_fisher_random_trial_is_deterministic() -> None:
    first_datasets, first_comparison = _generate_fisher_random_trial(
        config=_small_config(),
        selection_config=_small_selection_config(),
        master_seed=29,
        trial_index=3,
    )
    second_datasets, second_comparison = _generate_fisher_random_trial(
        config=_small_config(),
        selection_config=_small_selection_config(),
        master_seed=29,
        trial_index=3,
    )
    assert first_comparison == second_comparison
    for key in first_datasets:
        assert logical_dataset_sha256(first_datasets[key]) == (
            logical_dataset_sha256(second_datasets[key])
        )


def test_e_optimal_selection_targets_weakest_marginal_direction() -> None:
    selection_config = replace(
        _small_selection_config(),
        fisher_objective="e_optimal",
    )
    datasets, comparison = _generate_fisher_random_trial(
        config=_small_config(),
        selection_config=selection_config,
        master_seed=17,
        trial_index=0,
    )
    assert comparison["fisher_objective"] == (
        "estimated_state_handeye_marginal_e_optimal"
    )
    assert comparison["future_ground_truth_profiles_visible_to_policy"] is False
    assert comparison["policy_candidate_interface"] == (
        "candidate_kinematics_plus_acquire_selected_callback"
    )
    assert comparison["fisher_linearization"] == (
        "current_estimate_from_acquired_noisy_measurements"
    )
    initial_count = selection_config.initial_random_scans
    for geometry in ("single", "three"):
        fisher = comparison["selections"][f"{geometry}_fisher"]
        random = comparison["selections"][f"{geometry}_random"]
        fisher_step = fisher["selection_trace"][initial_count]
        random_step = random["selection_trace"][initial_count]
        assert fisher_step["selection_objective"] == "e_optimal"
        assert "marginal_min_eigenvalue_gain" in fisher_step
        assert "information_gain_nats" not in fisher_step
        assert (
            fisher_step["objective_gain"]
            >= random_step["objective_gain"]
        )
    assert {
        scan.meta["selection_objective"]
        for scan in datasets["single_fisher"].scans
    } == {"e_optimal"}
    assert datasets["single_fisher"].profile_state == "measured"
    assert datasets["single_fisher"].metadata["noise_applied"] is True
    for scan in datasets["single_fisher"].scans:
        assert scan.meta["scan_id"] == scan.scan_id
        assert scan.meta["candidate_pool_index"] == (
            scan.meta["source_candidate_scan_id"]
        )


def test_policy_visible_candidate_has_no_future_profile_points() -> None:
    candidate = CandidateKinematics(
        candidate_id=0,
        T_base_ef=np.eye(4),
        measurement_plane_id=0,
        quota_group_id=0,
    )
    assert not hasattr(candidate, "points_s")


def test_each_selected_scan_is_acquired_exactly_once(monkeypatch) -> None:
    original_factory = comparison_generator._make_candidate_acquirer
    acquired_ids: list[int] = []

    def counting_factory(*args, **kwargs):
        acquire = original_factory(*args, **kwargs)

        def counted_acquire(candidate_id: int):
            acquired_ids.append(int(candidate_id))
            return acquire(candidate_id)

        return counted_acquire

    monkeypatch.setattr(
        comparison_generator,
        "_make_candidate_acquirer",
        counting_factory,
    )
    datasets, _comparison = _generate_fisher_random_trial(
        config=_small_config(),
        selection_config=_small_selection_config(),
        master_seed=17,
        trial_index=0,
    )
    assert len(acquired_ids) == sum(
        len(dataset.scans) for dataset in datasets.values()
    )


def test_shared_global_cli_writes_auditable_pair(tmp_path) -> None:
    output = tmp_path / "comparison"
    result = generate_comparison(
        [
            "--trials",
            "1",
            "--seed",
            "17",
            "--output-dir",
            str(output),
            "--pose-selection-mode",
            "fisher_vs_random",
            "--total-scans",
            "18",
            "--initial-random-scans",
            "9",
            "--profile-points",
            "16",
            "--selection-estimator-max-iterations",
            "30",
        ]
    )
    assert result == 0

    manifest = json.loads(
        (output / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == COMPARISON_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 4
    assert manifest["pose_sampling_frame"] == POSE_SAMPLING_FRAME
    assert (
        manifest["fisher_vs_random_summary"]["single_plane"]["trial_count"]
        == 1
    )
    trial = manifest["trials"][0]
    assert trial["single_robot_pose_stack_sha256"] == (
        trial["three_robot_pose_stack_sha256"]
    )
    assert (
        trial["diversity_metrics"][
            "three_cross_plane_effective_normal_median_deg"
        ]
        > 80.0
    )

    single = load_calibration_dataset(
        output / trial["single_relative_path"]
    )
    three = load_calibration_dataset(
        output / trial["three_relative_path"]
    )
    assert np.array_equal(_pose_stack(single), _pose_stack(three))
    single_fisher = load_calibration_dataset(
        output / trial["single_fisher_relative_path"]
    )
    three_fisher = load_calibration_dataset(
        output / trial["three_fisher_relative_path"]
    )
    initial_count = manifest["pose_selection_config"][
        "initial_random_scans"
    ]
    for dataset in (single_fisher, three_fisher):
        assert np.array_equal(
            _pose_stack(dataset)[:initial_count],
            _pose_stack(single)[:initial_count],
        )
    assert {
        scan.meta["pose_selection_strategy"]
        for scan in single_fisher.scans
    } == {"fisher"}
    for relative_collection in (
        "single_plane",
        "single_plane_fisher",
        "three_plane",
        "three_plane_fisher",
    ):
        collection = json.loads(
            (output / relative_collection / "collection.json").read_text(
                encoding="utf-8"
            )
        )
        assert collection["profile_state"] == "measured"


def test_random_only_cli_writes_two_noise_free_paired_collections(
    tmp_path,
) -> None:
    output = tmp_path / "random_only"
    result = generate_comparison(
        [
            "--trials",
            "1",
            "--seed",
            "17",
            "--output-dir",
            str(output),
            "--pose-selection-mode",
            "random_only",
            "--total-scans",
            "18",
            "--profile-points",
            "16",
        ]
    )
    assert result == 0

    manifest = json.loads(
        (output / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["pose_selection_config"] is None
    assert manifest["completed_trials"] == 1
    assert not (output / "single_plane_fisher").exists()
    assert not (output / "three_plane_fisher").exists()

    single = load_calibration_dataset(
        output / manifest["trials"][0]["single_relative_path"]
    )
    three = load_calibration_dataset(
        output / manifest["trials"][0]["three_relative_path"]
    )
    assert single.profile_state == "ideal"
    assert three.profile_state == "ideal"
    assert single.metadata["noise_applied"] is False
    assert three.metadata["noise_applied"] is False
    assert np.array_equal(_pose_stack(single), _pose_stack(three))
    _assert_truth_plane_residuals(single)
    _assert_truth_plane_residuals(three)
