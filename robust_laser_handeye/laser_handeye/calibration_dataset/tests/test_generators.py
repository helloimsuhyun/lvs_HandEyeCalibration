from __future__ import annotations

from dataclasses import replace
import json

import numpy as np

from laser_handeye.calibration import calibrate_planes
from laser_handeye.calibration_dataset import (
    CalibrationDataset,
    load_calibration_dataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.calibration_dataset.generators import (
    GenerationSeeds,
    SinglePlaneCircularGenerationConfig,
    ThreePlaneGenerationConfig,
    TranslationCompositeGenerationConfig,
    generate_single_plane_circular_dataset,
    generate_three_plane_dataset,
    generate_translation_composite_dataset,
)
from laser_handeye.tan2025.simulation import (
    AnalyticTan2025Simulator,
    PaperSimulationConfig,
    paper_handeye_transform,
)
from laser_handeye.tan2025.estimator import Tan2025ClosedFormEstimator
from laser_handeye.se3 import transform_points


def _group_counts(dataset: CalibrationDataset) -> dict[str, int]:
    return {
        group.group_id: int(
            np.count_nonzero(dataset.scan_group_ids == group.group_id)
        )
        for group in dataset.groups
    }


def _assert_common_ideal_contract(
    dataset: CalibrationDataset,
    *,
    expected_scan_count: int,
    expected_profile_points: int,
) -> None:
    assert dataset.source == "simulation"
    assert dataset.profile_state == "ideal"
    assert dataset.channel_id_semantics == "stable_sensor_channel"
    assert len(dataset.scans) == expected_scan_count
    assert dataset.metadata["noise_applied"] is False
    assert len(np.unique([scan.scan_id for scan in dataset.scans])) == len(
        dataset.scans
    )
    for scan in dataset.scans:
        assert scan.num_points == expected_profile_points
        assert np.all(np.isfinite(scan.points_s))
        assert np.array_equal(
            scan.meta["channel_ids"],
            np.arange(expected_profile_points, dtype=np.int64),
        )


def _round_trip(
    dataset: CalibrationDataset,
    directory,
) -> CalibrationDataset:
    expected_hash = logical_dataset_sha256(dataset)
    manifest_path = save_calibration_dataset(dataset, directory)
    assert manifest_path.name == "manifest.json"
    assert (directory / "scans.npz").is_file()
    loaded = load_calibration_dataset(directory)
    assert logical_dataset_sha256(loaded) == expected_hash
    assert np.array_equal(loaded.scan_group_ids, dataset.scan_group_ids)
    assert np.array_equal(loaded.sequence_indices, dataset.sequence_indices)
    return loaded


def _translation_composite_config() -> TranslationCompositeGenerationConfig:
    return TranslationCompositeGenerationConfig(
        simulation=PaperSimulationConfig(
            num_translation_poses=4,
            num_composite_poses=6,
            num_profile_points=16,
        ),
        handeye_preset="tan2025",
    )


def test_translation_composite_small_dataset_contract_and_round_trip(
    tmp_path,
) -> None:
    dataset = generate_translation_composite_dataset(
        _translation_composite_config(),
        GenerationSeeds.derive(master_seed=17, trial_index=0),
    )

    assert dataset.acquisition_mode == "translation_composite"
    assert _group_counts(dataset) == {"translation": 4, "composite": 6}
    assert dataset.group("translation").motion_kind == "pure_translation"
    assert dataset.group("composite").motion_kind == "composite"
    _assert_common_ideal_contract(
        dataset,
        expected_scan_count=10,
        expected_profile_points=16,
    )

    assert dataset.truth is not None
    assert np.allclose(dataset.truth.T_ef_s_true, paper_handeye_transform())
    assert len(dataset.truth.planes) == 1
    assert dataset.truth.planes[0].plane_id == 0
    assert np.isclose(np.linalg.norm(dataset.truth.planes[0].normal_base), 1.0)
    assert "T_ef_s_true" not in json.dumps(dataset.metadata, sort_keys=True)
    assert all("true_T_base_ef" not in scan.meta for scan in dataset.scans)
    assert dataset.truth.T_base_ef_true is not None
    assert dataset.truth.T_base_ef_commanded is not None

    tan_dataset = dataset.to_tan2025_dataset()
    assert len(tan_dataset.translation_scans) == 4
    assert len(tan_dataset.composite_scans) == 6
    assert {
        scan.meta["motion_kind"] for scan in tan_dataset.translation_scans
    } == {"translation"}
    assert {
        scan.meta["motion_kind"] for scan in tan_dataset.composite_scans
    } == {"composite"}
    scans_by_plane = dataset.to_scans_by_plane()
    assert list(scans_by_plane) == [0]
    assert len(scans_by_plane[0]) == 10

    loaded = _round_trip(dataset, tmp_path / "translation_composite")
    loaded_tan = loaded.to_tan2025_dataset()
    assert len(loaded_tan.translation_scans) == 4
    estimate = Tan2025ClosedFormEstimator().estimate(loaded_tan)
    assert np.allclose(estimate.T_ef_s, loaded.truth.T_ef_s_true, atol=1e-8)


def test_translation_composite_random_plane_is_shared_and_seed_isolated() -> None:
    config = TranslationCompositeGenerationConfig(
        simulation=PaperSimulationConfig(
            num_translation_poses=5,
            num_composite_poses=7,
            num_profile_points=16,
        ),
        handeye_preset="random",
        plane_mode="random",
    )
    seeds = GenerationSeeds.derive(master_seed=19, trial_index=0)
    dataset = generate_translation_composite_dataset(config, seeds)
    repeated = generate_translation_composite_dataset(config, seeds)
    assert logical_dataset_sha256(dataset) == logical_dataset_sha256(repeated)

    plane = dataset.truth.planes[0]
    plane_metadata = dataset.truth.metadata["plane"]
    assert plane_metadata["plane_mode"] == "random"
    assert (
        plane_metadata["nearest_base_axis_angle_deg"]
        >= config.plane_min_axis_angle_deg
    )
    assert np.isclose(plane.offset_mm, plane_metadata["distance_mm"])
    assert np.isclose(
        plane.normal_base @ plane.T_base_plane[:3, 3],
        plane.offset_mm,
    )
    assert {scan.plane_id for scan in dataset.scans} == {0}
    for scan in dataset.scans:
        points_base = transform_points(
            scan.T_base_ef @ dataset.truth.T_ef_s_true,
            scan.valid_points_s,
        )
        residuals = points_base @ plane.normal_base - plane.offset_mm
        assert np.max(np.abs(residuals)) < 1e-8

    changed_environment = generate_translation_composite_dataset(
        config,
        replace(seeds, environment_seed=seeds.environment_seed + 1),
    )
    assert np.allclose(
        changed_environment.truth.T_ef_s_true,
        dataset.truth.T_ef_s_true,
    )
    assert not np.allclose(
        changed_environment.truth.planes[0].normal_base,
        plane.normal_base,
    )

    changed_motion = generate_translation_composite_dataset(
        config,
        replace(seeds, motion_seed=seeds.motion_seed + 1),
    )
    assert np.allclose(changed_motion.truth.planes[0].normal_base, plane.normal_base)
    assert np.isclose(changed_motion.truth.planes[0].offset_mm, plane.offset_mm)
    assert any(
        not np.allclose(first.T_base_ef, second.T_base_ef)
        for first, second in zip(dataset.scans, changed_motion.scans)
    )

    changed_handeye = generate_translation_composite_dataset(
        config,
        replace(seeds, handeye_seed=seeds.handeye_seed + 1),
    )
    assert np.allclose(
        changed_handeye.truth.planes[0].normal_base,
        plane.normal_base,
    )
    assert np.isclose(changed_handeye.truth.planes[0].offset_mm, plane.offset_mm)
    assert not np.allclose(
        changed_handeye.truth.T_ef_s_true,
        dataset.truth.T_ef_s_true,
    )
    assert any(
        not np.allclose(first.T_base_ef, second.T_base_ef)
        for first, second in zip(dataset.scans, changed_handeye.scans)
    )
    for first, second in zip(dataset.scans, changed_handeye.scans):
        assert np.allclose(
            first.T_base_ef @ dataset.truth.T_ef_s_true,
            second.T_base_ef @ changed_handeye.truth.T_ef_s_true,
        )
        assert np.allclose(first.points_s, second.points_s, equal_nan=True)

    estimate = Tan2025ClosedFormEstimator().estimate(dataset.to_tan2025_dataset())
    assert np.allclose(estimate.T_ef_s, dataset.truth.T_ef_s_true, atol=1e-8)


def test_translation_composite_fixed_plane_ignores_environment_seed() -> None:
    config = _translation_composite_config()
    seeds = GenerationSeeds.derive(master_seed=29, trial_index=0)
    first = generate_translation_composite_dataset(config, seeds)
    second = generate_translation_composite_dataset(
        config,
        replace(seeds, environment_seed=seeds.environment_seed + 1),
    )
    assert np.allclose(
        first.truth.planes[0].normal_base,
        second.truth.planes[0].normal_base,
    )
    assert np.isclose(
        first.truth.planes[0].offset_mm,
        second.truth.planes[0].offset_mm,
    )
    for first_scan, second_scan in zip(first.scans, second.scans):
        assert np.array_equal(first_scan.T_base_ef, second_scan.T_base_ef)
        assert np.array_equal(
            first_scan.points_s,
            second_scan.points_s,
            equal_nan=True,
        )

    baseline = AnalyticTan2025Simulator(config.simulation).generate(
        seed=None,
        rng=np.random.default_rng(seeds.motion_seed),
    )
    baseline_scans = [
        *baseline.dataset.translation_scans,
        *baseline.dataset.composite_scans,
    ]
    assert np.allclose(
        first.truth.planes[0].normal_base,
        baseline.truth.plane_normal_base,
    )
    assert np.isclose(
        first.truth.planes[0].offset_mm,
        baseline.truth.plane_offset_mm,
    )
    for generated, original in zip(first.scans, baseline_scans):
        assert np.array_equal(generated.T_base_ef, original.T_base_ef)
        assert np.array_equal(generated.points_s, original.points_s, equal_nan=True)


def test_single_plane_circular_small_grid_groups_metadata_and_round_trip(
    tmp_path,
) -> None:
    config = SinglePlaneCircularGenerationConfig(
        handeye_preset="tan2025",
        profile_points=16,
        heights_mm=(80.0,),
        theta_deg=(30.0,),
        beta_deg=(90.0,),
        reference_line_ids=(2,),
        reference_heights_mm=(80.0,),
        reference_theta_deg=60.0,
        reference_beta_deg=(90.0,),
    )
    dataset = generate_single_plane_circular_dataset(
        config,
        GenerationSeeds.derive(master_seed=23, trial_index=0),
    )

    assert dataset.acquisition_mode == "single_plane_circular"
    assert _group_counts(dataset) == {"primary_ring": 9, "reference_ring": 1}
    assert dataset.group("primary_ring").acquisition_role == "calibration"
    assert dataset.group("reference_ring").acquisition_role == "reference"
    _assert_common_ideal_contract(
        dataset,
        expected_scan_count=10,
        expected_profile_points=16,
    )

    primary = list(dataset.iter_group_scans("primary_ring"))
    reference = list(dataset.iter_group_scans("reference_ring"))
    assert sorted(int(scan.meta["line_id"]) for scan in primary) == list(range(9))
    assert {int(scan.meta["parameter_id"]) for scan in primary} == {0}
    assert int(reference[0].meta["line_id"]) == 2
    assert reference[0].meta["reference_pose"] is True
    assert reference[0].meta["additional_scan"] is True
    assert reference[0].meta["theta_deg"] == 60.0
    assert dataset.metadata["expected_and_generated_counts"] == {
        "primary_expected": 9,
        "primary_generated": 9,
        "reference_expected": 1,
        "reference_generated": 1,
    }

    assert dataset.truth is not None
    assert len(dataset.truth.planes) == 1
    assert dataset.truth.planes[0].T_base_plane is not None
    scans_by_plane = dataset.to_scans_by_plane()
    assert list(scans_by_plane) == [0]
    assert len(scans_by_plane[0]) == 10

    loaded = _round_trip(dataset, tmp_path / "single_plane_circular")
    assert len(loaded.to_scans_by_plane()[0]) == 10


def test_three_plane_small_dataset_plane_groups_truth_and_round_trip(
    tmp_path,
) -> None:
    dataset = generate_three_plane_dataset(
        ThreePlaneGenerationConfig(
            handeye_preset="tan2025",
            poses_per_plane=3,
            profile_points=16,
        ),
        GenerationSeeds.derive(master_seed=31, trial_index=0),
    )

    assert dataset.acquisition_mode == "three_plane_random"
    assert _group_counts(dataset) == {
        "plane_0": 3,
        "plane_1": 3,
        "plane_2": 3,
    }
    _assert_common_ideal_contract(
        dataset,
        expected_scan_count=9,
        expected_profile_points=16,
    )
    for plane_id in range(3):
        group = dataset.group(f"plane_{plane_id}")
        assert group.plane_id == plane_id
        assert group.motion_kind == "general_6dof"
        assert all(
            scan.plane_id == plane_id
            for scan in dataset.iter_group_scans(group.group_id)
        )

    scans_by_plane = dataset.to_scans_by_plane()
    assert list(scans_by_plane) == [0, 1, 2]
    assert [len(scans_by_plane[plane_id]) for plane_id in range(3)] == [3, 3, 3]
    assert dataset.truth is not None
    assert [plane.plane_id for plane in dataset.truth.planes] == [0, 1, 2]
    assert all(plane.T_base_plane is not None for plane in dataset.truth.planes)
    normals = np.stack([plane.normal_base for plane in dataset.truth.planes])
    assert np.allclose(normals @ normals.T, np.eye(3), atol=1e-12)

    loaded = _round_trip(dataset, tmp_path / "three_plane")
    assert [len(scans) for scans in loaded.to_scans_by_plane().values()] == [3, 3, 3]
    result = calibrate_planes(
        loaded.to_scans_by_plane(),
        loaded.truth.T_ef_s_true,
        max_iter=3,
        plane_offset_mode="joint",
    )
    assert result.converged
    assert np.allclose(result.T_ef_s, loaded.truth.T_ef_s_true, atol=1e-8)


def test_default_circular_motion_profile_is_observable_by_iterative_adapter() -> None:
    dataset = generate_single_plane_circular_dataset(
        SinglePlaneCircularGenerationConfig(
            handeye_preset="tan2025",
            profile_points=16,
        ),
        GenerationSeeds.derive(master_seed=23, trial_index=0),
    )
    assert _group_counts(dataset) == {
        "primary_ring": 81,
        "reference_ring": 36,
    }
    result = calibrate_planes(
        dataset.to_scans_by_plane(),
        dataset.truth.T_ef_s_true,
        max_iter=3,
        plane_offset_mode="joint",
    )
    assert result.converged
    assert np.allclose(result.T_ef_s, dataset.truth.T_ef_s_true, atol=1e-8)


def test_circular_signed_plane_frame_stays_consistent_for_negative_origin() -> None:
    dataset = generate_single_plane_circular_dataset(
        SinglePlaneCircularGenerationConfig(
            handeye_preset="tan2025",
            profile_points=8,
            heights_mm=(80.0,),
            theta_deg=(30.0,),
            beta_deg=(90.0,),
            reference_line_ids=(),
            plane_z_range_mm=(-550.0, -400.0),
        ),
        GenerationSeeds.derive(master_seed=5, trial_index=0),
    )
    plane = dataset.truth.planes[0]
    assert np.allclose(plane.T_base_plane[:3, 2], plane.normal_base)
    assert np.isclose(
        plane.normal_base @ plane.T_base_plane[:3, 3],
        plane.offset_mm,
    )


def test_generation_seed_reproducibility_changes_with_trial_index() -> None:
    config = _translation_composite_config()
    seeds = GenerationSeeds.derive(master_seed=47, trial_index=0)
    first = generate_translation_composite_dataset(config, seeds)
    repeated = generate_translation_composite_dataset(config, seeds)
    next_trial = generate_translation_composite_dataset(
        config,
        GenerationSeeds.derive(master_seed=47, trial_index=1),
    )

    assert logical_dataset_sha256(first) == logical_dataset_sha256(repeated)
    assert logical_dataset_sha256(first) != logical_dataset_sha256(next_trial)
