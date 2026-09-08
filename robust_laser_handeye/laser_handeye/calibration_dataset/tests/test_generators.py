from __future__ import annotations

import numpy as np

from laser_handeye.calibration import calibrate_planes
from laser_handeye.calibration_dataset import (
    load_calibration_dataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.calibration_dataset.generators import (
    GenerationSeeds,
    SinglePlaneCircularGenerationConfig,
    ThreePlaneGenerationConfig,
    generate_single_plane_circular_dataset,
    generate_three_plane_dataset,
)


def _round_trip(dataset, path):
    save_calibration_dataset(dataset, path)
    loaded = load_calibration_dataset(path)
    assert logical_dataset_sha256(loaded) == logical_dataset_sha256(dataset)
    return loaded


def test_single_plane_circular_small_grid_round_trip(tmp_path) -> None:
    dataset = generate_single_plane_circular_dataset(
        SinglePlaneCircularGenerationConfig(
            profile_points=16,
            heights_mm=(80.0,),
            theta_deg=(30.0,),
            beta_deg=(90.0,),
            reference_line_ids=(2,),
            reference_heights_mm=(80.0,),
            reference_beta_deg=(90.0,),
        ),
        GenerationSeeds.derive(master_seed=23, trial_index=0),
    )
    assert dataset.acquisition_mode == "single_plane_circular"
    assert len(list(dataset.iter_group_scans("primary_ring"))) == 9
    assert len(list(dataset.iter_group_scans("reference_ring"))) == 1
    assert len(_round_trip(dataset, tmp_path / "circular").scans) == 10


def test_three_plane_small_dataset_is_observable_and_round_trips(tmp_path) -> None:
    dataset = generate_three_plane_dataset(
        ThreePlaneGenerationConfig(poses_per_plane=3, profile_points=16),
        GenerationSeeds.derive(master_seed=31, trial_index=0),
    )
    assert dataset.acquisition_mode == "three_plane_random"
    normals = np.stack([plane.normal_base for plane in dataset.truth.planes])
    assert np.allclose(normals @ normals.T, np.eye(3), atol=1e-12)
    loaded = _round_trip(dataset, tmp_path / "three_plane")
    result = calibrate_planes(
        loaded.to_scans_by_plane(),
        loaded.truth.T_ef_s_true,
        max_iter=3,
        plane_offset_mode="joint",
    )
    assert result.converged
    assert np.allclose(result.T_ef_s, loaded.truth.T_ef_s_true, atol=1e-8)


def test_generation_seed_is_reproducible() -> None:
    config = SinglePlaneCircularGenerationConfig(
        profile_points=8,
        heights_mm=(80.0,),
        theta_deg=(30.0,),
        beta_deg=(90.0,),
        reference_line_ids=(),
    )
    seeds = GenerationSeeds.derive(master_seed=47, trial_index=0)
    first = generate_single_plane_circular_dataset(config, seeds)
    repeated = generate_single_plane_circular_dataset(config, seeds)
    next_trial = generate_single_plane_circular_dataset(
        config,
        GenerationSeeds.derive(master_seed=47, trial_index=1),
    )
    assert logical_dataset_sha256(first) == logical_dataset_sha256(repeated)
    assert logical_dataset_sha256(first) != logical_dataset_sha256(next_trial)
