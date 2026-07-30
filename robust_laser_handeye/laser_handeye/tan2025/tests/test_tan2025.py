from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from laser_handeye.data import LaserScan
from laser_handeye.se3 import euler_xyz_deg
from laser_handeye.tan2025 import (
    AnalyticTan2025Simulator,
    CalibrationPipeline,
    NonlinearPlaneRefiner,
    PaperSimulationConfig,
    SensorNoiseConfig,
    Tan2025ClosedFormEstimator,
    Tan2025Dataset,
    Tan2025EstimatorConfig,
)
from laser_handeye.tan2025.evaluation import euler_zyx_deg
from laser_handeye.tan2025.experiments import (
    run_gaussian_noise_sweep,
    run_paper_trial,
    trial_detail_dict,
)


def _small_config(**overrides) -> PaperSimulationConfig:
    config = PaperSimulationConfig(
        num_translation_poses=12,
        num_composite_poses=15,
        num_profile_points=96,
    )
    return replace(config, **overrides)


def test_paper_ground_truth_matches_equation_55() -> None:
    config = _small_config()
    expected_rotation = np.array(
        [
            [0.0, -0.9961946981, 0.0871557427],
            [1.0, 0.0, 0.0],
            [0.0, 0.0871557427, 0.9961946981],
        ]
    )
    assert np.allclose(config.T_ef_s_true[:3, :3], expected_rotation, atol=1e-9)
    assert np.allclose(
        config.T_ef_s_true[:3, 3],
        [-22.86848, 83.73314, 153.08619],
    )
    assert np.allclose(euler_zyx_deg(config.T_ef_s_true[:3, :3]), [90, 0, 5])


def test_noise_free_closed_form_recovers_ground_truth() -> None:
    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=11)
    result = Tan2025ClosedFormEstimator().estimate(simulated.dataset)

    normal_alignment = abs(
        float(result.plane_normal_base @ simulated.truth.plane_normal_base)
    )
    assert normal_alignment == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(
        result.T_ef_s[:3, :3], simulated.truth.T_ef_s[:3, :3], atol=1e-10
    )
    assert np.linalg.norm(
        result.T_ef_s[:3, 3] - simulated.truth.T_ef_s[:3, 3]
    ) < 1e-8
    assert result.normal_system.rank == 3
    assert result.rotation_system.rank == 6
    assert result.translation_system.rank == 3


def test_channel_ids_preserve_correspondence_with_dropouts() -> None:
    config = _small_config(
        sensor_noise=SensorNoiseConfig(mode="none", dropout_probability=0.03)
    )
    simulated = AnalyticTan2025Simulator(config).generate(seed=3)
    aligned = simulated.dataset.aligned_translation_points()
    assert 3 < len(aligned[0]) < config.num_profile_points

    result = Tan2025ClosedFormEstimator().estimate(simulated.dataset)
    assert np.allclose(
        result.T_ef_s[:3, :3], simulated.truth.T_ef_s[:3, :3], atol=1e-10
    )
    assert np.linalg.norm(
        result.T_ef_s[:3, 3] - simulated.truth.T_ef_s[:3, 3]
    ) < 1e-8


def test_pairwise_alignment_avoids_global_dropout_collapse() -> None:
    config = PaperSimulationConfig(
        num_translation_poses=48,
        num_composite_poses=24,
        num_profile_points=64,
        sensor_noise=SensorNoiseConfig(mode="none", dropout_probability=0.2),
    )
    simulated = AnalyticTan2025Simulator(config).generate(seed=23)
    # The probability that one channel survives all 48 profiles is tiny, but
    # every profile pair still has dozens of matching channels.
    with pytest.raises(ValueError, match="common finite channels"):
        simulated.dataset.aligned_translation_points()
    result = Tan2025ClosedFormEstimator().estimate(simulated.dataset)
    assert np.allclose(
        result.T_ef_s[:3, :3], simulated.truth.T_ef_s[:3, :3], atol=1e-10
    )
    assert np.linalg.norm(
        result.T_ef_s[:3, 3] - simulated.truth.T_ef_s[:3, 3]
    ) < 1e-8


def test_missing_channel_ids_rejects_variable_width_translation_profiles() -> None:
    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=5)
    scans = []
    for index, scan in enumerate(simulated.dataset.translation_scans):
        points = scan.points_s[:-index or None]
        scans.append(
            LaserScan(
                scan.T_base_ef,
                points,
                scan_id=index,
                meta={"motion_kind": "translation"},
            )
        )
    dataset = Tan2025Dataset(scans, simulated.dataset.composite_scans)
    with pytest.raises(ValueError, match="different lengths"):
        dataset.aligned_translation_points()


def test_degenerate_translation_motion_fails_with_rank_message() -> None:
    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=9)
    reference = simulated.dataset.translation_scans[0].T_base_ef
    scans = []
    direction = np.array([1.0, 0.0, 0.0])
    for index, scan in enumerate(simulated.dataset.translation_scans):
        pose = reference.copy()
        pose[:3, 3] += index * direction
        scans.append(
            LaserScan(
                pose,
                scan.points_s,
                scan_id=index,
                meta=scan.meta,
            )
        )
    dataset = Tan2025Dataset(scans, simulated.dataset.composite_scans)
    with pytest.raises(np.linalg.LinAlgError, match="rank 1 < 3"):
        Tan2025ClosedFormEstimator().estimate(dataset)


def test_translation_group_rotation_is_validated() -> None:
    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=13)
    scans = list(simulated.dataset.translation_scans)
    changed = scans[-1].T_base_ef.copy()
    changed[:3, :3] = changed[:3, :3] @ euler_xyz_deg(0.0, 0.0, 1.0)
    scans[-1] = LaserScan(
        changed,
        scans[-1].points_s,
        scan_id=scans[-1].scan_id,
        meta=scans[-1].meta,
    )
    dataset = Tan2025Dataset(scans, simulated.dataset.composite_scans)
    with pytest.raises(ValueError, match="changes tool orientation"):
        Tan2025ClosedFormEstimator().estimate(dataset)


def test_simulation_and_estimator_configs_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        PaperSimulationConfig(composite_roll_span_deg=np.nan)
    with pytest.raises(ValueError, match="positive"):
        Tan2025EstimatorConfig(line_constraint_weight=np.nan)


def test_simulation_config_rejects_non_rigid_ground_truth() -> None:
    transform = PaperSimulationConfig().T_ef_s_true.copy()
    transform[0, 1] *= 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        PaperSimulationConfig(T_ef_s_true=transform)


def test_tiny_translation_baseline_fails_excitation_gate() -> None:
    config = _small_config(
        num_translation_poses=4,
        translation_tangent_span_mm=0.1,
        translation_normal_span_mm=0.1,
    )
    simulated = AnalyticTan2025Simulator(config).generate(seed=41)
    with pytest.raises(np.linalg.LinAlgError, match="too small"):
        Tan2025ClosedFormEstimator().estimate(simulated.dataset)


def test_dataset_metadata_records_analytic_conventions() -> None:
    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=43)
    analytic_model = simulated.dataset.metadata["analytic_model"]
    assert analytic_model["version"] == 1
    assert analytic_model["target_plane"] == "infinite"
    assert analytic_model["ray_sampling"] == "uniform_angle"


def test_nonlinear_refiner_can_be_appended_to_closed_form_pipeline() -> None:
    config = _small_config(
        num_profile_points=48,
        sensor_noise=SensorNoiseConfig(mode="gaussian_xz", magnitude_mm=0.05),
    )
    simulated = AnalyticTan2025Simulator(config).generate(seed=17)
    pipeline = CalibrationPipeline(
        Tan2025ClosedFormEstimator(),
        [NonlinearPlaneRefiner(max_nfev=15)],
    )
    result = pipeline.run(simulated.dataset)
    assert [stage.name for stage in result.stages] == [
        "tan2025_closed_form",
        "nonlinear_refitted_plane",
    ]
    assert np.all(np.isfinite(result.T_ef_s))
    assert np.allclose(result.T_ef_s[3], [0.0, 0.0, 0.0, 1.0])


def test_pipeline_rejects_nonconverged_refiner() -> None:
    config = _small_config(
        num_profile_points=32,
        sensor_noise=SensorNoiseConfig(mode="gaussian_xz", magnitude_mm=0.1),
    )
    simulated = AnalyticTan2025Simulator(config).generate(seed=29)
    pipeline = CalibrationPipeline(
        Tan2025ClosedFormEstimator(),
        [NonlinearPlaneRefiner(max_nfev=1)],
    )
    with pytest.raises(RuntimeError, match="maximum number of function evaluations"):
        pipeline.run(simulated.dataset)


def test_refiner_cannot_mutate_previous_pipeline_stage() -> None:
    class MutatingRefiner:
        name = "mutating_test_refiner"

        @staticmethod
        def refine(dataset, T_init):
            del dataset
            T_init[0, 3] += 1.0
            return SimpleNamespace(T_ef_s=T_init, success=True)

    simulated = AnalyticTan2025Simulator(_small_config()).generate(seed=37)
    pipeline = CalibrationPipeline(
        Tan2025ClosedFormEstimator(),
        [MutatingRefiner()],
    )
    result = pipeline.run(simulated.dataset)
    assert result.stages[0].T_ef_s[0, 3] == pytest.approx(
        simulated.truth.T_ef_s[0, 3]
    )
    assert result.stages[1].T_ef_s[0, 3] == pytest.approx(
        simulated.truth.T_ef_s[0, 3] + 1.0
    )


def test_trial_json_separates_closed_form_and_final_pipeline_stage() -> None:
    config = _small_config(
        num_profile_points=32,
        sensor_noise=SensorNoiseConfig(mode="gaussian_xz", magnitude_mm=0.05),
    )
    pipeline = CalibrationPipeline(
        Tan2025ClosedFormEstimator(),
        [NonlinearPlaneRefiner(max_nfev=20)],
    )
    trial, estimate, pipeline_result = run_paper_trial(
        config,
        seed=31,
        pipeline=pipeline,
    )
    assert pipeline_result is not None
    detail = trial_detail_dict(trial, estimate, pipeline_result)
    assert detail["final"]["stage_name"] == "nonlinear_refitted_plane"
    assert np.allclose(detail["closed_form"]["T_ef_s"], estimate.T_ef_s)
    assert np.allclose(detail["final"]["T_ef_s"], pipeline_result.T_ef_s)
    assert detail["pipeline"][-1]["status"]["success"] is True


def test_seeded_noise_sweep_returns_one_summary_per_sigma() -> None:
    trials, summary = run_gaussian_noise_sweep(
        _small_config(num_profile_points=32),
        sigma_values_mm=[0.05, 0.1],
        trials_per_sigma=2,
        seed=19,
    )
    assert len(trials) == 4
    assert len(summary) == 2
    assert [row["noise_magnitude_mm"] for row in summary] == [0.05, 0.1]
    assert all(row["trials"] == 2 for row in summary)
