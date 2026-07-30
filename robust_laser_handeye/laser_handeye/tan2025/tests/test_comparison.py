from __future__ import annotations

import numpy as np
import pytest

from laser_handeye.tan2025 import (
    AnalyticTan2025Simulator,
    PaperSimulationConfig,
    SensorNoiseConfig,
    ThreeWayComparisonConfig,
    compare_three_methods_on_shared_simulation,
    run_three_way_monte_carlo,
    simulation_dataset_sha256,
    summarize_paired_differences,
    summarize_three_way_methods,
    three_way_trials_detail_dict,
)


def _simulation_config(*, noise_mm: float = 0.0) -> PaperSimulationConfig:
    return PaperSimulationConfig(
        num_translation_poses=12,
        num_composite_poses=12,
        num_profile_points=32,
        sensor_noise=SensorNoiseConfig(
            mode="none" if noise_mm == 0.0 else "gaussian_xz",
            magnitude_mm=noise_mm,
        ),
    )


def _method_lookup(trial):
    return {method.method: method for method in trial.methods}


def test_clean_three_way_comparison_reuses_tan_initial_and_shared_data() -> None:
    simulated = AnalyticTan2025Simulator(_simulation_config()).generate(seed=101)
    dataset_hash = simulation_dataset_sha256(simulated)
    trial = compare_three_methods_on_shared_simulation(
        simulated,
        ThreeWayComparisonConfig(trials=1, max_iter=50),
        trial_index=0,
        dataset_seed=101,
        initializer_seed=202,
    )
    methods = _method_lookup(trial)

    assert [method.method for method in trial.methods] == [
        "tan_closed_form",
        "tan_then_alternating",
        "alternating_only",
    ]
    assert {method.dataset_sha256 for method in trial.methods} == {dataset_hash}
    assert simulation_dataset_sha256(simulated) == dataset_hash
    assert np.array_equal(
        methods["tan_then_alternating"].T_init,
        methods["tan_closed_form"].T_ef_s,
    )
    assert np.array_equal(methods["alternating_only"].T_init, np.eye(4))
    assert all(method.completed and method.converged for method in trial.methods)
    assert all(
        method.metrics is not None
        and method.metrics.translation_error_mm < 1e-7
        for method in trial.methods
    )


def test_monte_carlo_seed_reproduces_data_initials_and_estimates() -> None:
    simulation_config = _simulation_config(noise_mm=0.1)
    comparison_config = ThreeWayComparisonConfig(
        trials=2,
        seed=17,
        iterative_only_init="relative_gt",
        max_iter=50,
    )
    first = run_three_way_monte_carlo(simulation_config, comparison_config)
    second = run_three_way_monte_carlo(simulation_config, comparison_config)

    for first_trial, second_trial in zip(first, second):
        assert first_trial.dataset_seed == second_trial.dataset_seed
        assert first_trial.initializer_seed == second_trial.initializer_seed
        assert first_trial.dataset_sha256 == second_trial.dataset_sha256
        for first_method, second_method in zip(
            first_trial.methods,
            second_trial.methods,
        ):
            assert first_method.status == second_method.status
            assert first_method.iterations == second_method.iterations
            if first_method.T_init is None:
                assert second_method.T_init is None
            else:
                assert np.allclose(first_method.T_init, second_method.T_init)
            assert np.allclose(first_method.T_ef_s, second_method.T_ef_s)


def test_tan_failure_skips_dependent_method_but_runs_alternating_only() -> None:
    class FailingEstimator:
        name = "failing_tan"

        @staticmethod
        def estimate(dataset):
            del dataset
            raise RuntimeError("synthetic Tan failure")

    simulated = AnalyticTan2025Simulator(_simulation_config()).generate(seed=303)
    trial = compare_three_methods_on_shared_simulation(
        simulated,
        ThreeWayComparisonConfig(trials=1, max_iter=50),
        trial_index=0,
        dataset_seed=303,
        initializer_seed=404,
        estimator=FailingEstimator(),
    )
    methods = _method_lookup(trial)

    assert methods["tan_closed_form"].status == "failed"
    assert methods["tan_then_alternating"].status == "skipped_dependency"
    assert methods["alternating_only"].completed
    assert methods["alternating_only"].converged


def test_nonconverged_iterative_results_remain_in_summary_denominator() -> None:
    simulated = AnalyticTan2025Simulator(_simulation_config(noise_mm=0.1)).generate(
        seed=505
    )
    trial = compare_three_methods_on_shared_simulation(
        simulated,
        ThreeWayComparisonConfig(trials=1, max_iter=1, tol=1e-15),
        trial_index=0,
        dataset_seed=505,
        initializer_seed=606,
    )
    methods = _method_lookup(trial)
    assert methods["alternating_only"].completed
    assert not methods["alternating_only"].converged
    assert methods["alternating_only"].metrics is not None

    summary = {row["method"]: row for row in summarize_three_way_methods([trial])}
    assert summary["alternating_only"]["trials_requested"] == 1
    assert summary["alternating_only"]["completed"] == 1
    assert summary["alternating_only"]["converged"] == 0
    paired = summarize_paired_differences([trial])
    assert any(row["paired_trials"] == 1 for row in paired)


def test_comparison_config_and_json_details_are_explicit() -> None:
    assert ThreeWayComparisonConfig().iterative_only_init == "identity"
    assert not ThreeWayComparisonConfig().iterative_init_uses_ground_truth
    assert ThreeWayComparisonConfig(
        iterative_only_init="relative_gt"
    ).iterative_init_uses_ground_truth
    with pytest.raises(ValueError, match="finite"):
        ThreeWayComparisonConfig(tol=np.nan)

    trials = run_three_way_monte_carlo(
        _simulation_config(),
        ThreeWayComparisonConfig(trials=1, max_iter=50),
    )
    details = three_way_trials_detail_dict(trials)
    methods = {item["method"]: item for item in details[0]["methods"]}
    assert methods["tan_then_alternating"]["T_init"] == methods[
        "tan_closed_form"
    ]["T_ef_s"]
    assert methods["alternating_only"]["init_source"] == "identity"
