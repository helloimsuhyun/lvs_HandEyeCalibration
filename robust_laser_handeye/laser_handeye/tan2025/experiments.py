from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from .estimator import Tan2025ClosedFormEstimator, Tan2025Estimate
from .evaluation import Tan2025Metrics, evaluate_tan2025_estimate
from .pipeline import CalibrationPipeline, PipelineResult
from .models import Tan2025Dataset
from .simulation import (
    AnalyticTan2025Simulator,
    PaperSimulationConfig,
    SensorNoiseConfig,
    SimulatedTan2025Dataset,
)


@dataclass
class PaperTrialResult:
    seed: int
    num_translation_poses: int
    num_composite_poses: int
    num_profile_points: int
    noise_mode: str
    noise_magnitude_mm: float
    stage_name: str
    elapsed_s: float
    metrics: Tan2025Metrics
    normal_rank: int
    normal_condition: float
    rotation_rank: int
    rotation_condition: float
    translation_rank: int
    translation_condition: float

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "seed": self.seed,
            "num_translation_poses": self.num_translation_poses,
            "num_composite_poses": self.num_composite_poses,
            "num_profile_points": self.num_profile_points,
            "noise_mode": self.noise_mode,
            "noise_magnitude_mm": self.noise_magnitude_mm,
            "stage_name": self.stage_name,
            "elapsed_s": self.elapsed_s,
            "normal_rank": self.normal_rank,
            "normal_condition": self.normal_condition,
            "rotation_rank": self.rotation_rank,
            "rotation_condition": self.rotation_condition,
            "translation_rank": self.translation_rank,
            "translation_condition": self.translation_condition,
        }
        row.update(self.metrics.to_dict())
        return row


def run_paper_trial(
    config: PaperSimulationConfig,
    *,
    seed: int,
    estimator: Tan2025ClosedFormEstimator | None = None,
    pipeline: CalibrationPipeline | None = None,
) -> tuple[PaperTrialResult, Tan2025Estimate, PipelineResult | None]:
    if estimator is not None and pipeline is not None:
        raise ValueError("pass estimator or pipeline, not both")
    simulated = AnalyticTan2025Simulator(config).generate(seed=seed)
    return _evaluate_simulated_trial(
        config,
        simulated,
        seed=seed,
        estimator=estimator,
        pipeline=pipeline,
    )


def _evaluate_simulated_trial(
    config: PaperSimulationConfig,
    simulated: SimulatedTan2025Dataset,
    *,
    seed: int,
    estimator: Tan2025ClosedFormEstimator | None = None,
    pipeline: CalibrationPipeline | None = None,
) -> tuple[PaperTrialResult, Tan2025Estimate, PipelineResult | None]:
    closed_form = estimator or Tan2025ClosedFormEstimator()

    if pipeline is None:
        initial = closed_form.estimate(simulated.dataset)
        final_transform = initial.T_ef_s
        stage_name = closed_form.name
        elapsed_s = initial.elapsed_s
        pipeline_result = None
    else:
        pipeline_started = time.perf_counter()
        pipeline_result = pipeline.run(simulated.dataset)
        first_result = pipeline_result.stages[0].result
        if not isinstance(first_result, Tan2025Estimate):
            raise TypeError("pipeline initializer must return Tan2025Estimate")
        initial = first_result
        final_transform = pipeline_result.T_ef_s
        stage_name = pipeline_result.stages[-1].name
        elapsed_s = time.perf_counter() - pipeline_started

    metrics = evaluate_tan2025_estimate(
        simulated.dataset,
        simulated.truth,
        final_transform,
        estimated_plane_normal_base=initial.plane_normal_base,
    )
    trial = PaperTrialResult(
        seed=int(seed),
        num_translation_poses=config.num_translation_poses,
        num_composite_poses=config.num_composite_poses,
        num_profile_points=config.num_profile_points,
        noise_mode=config.sensor_noise.mode,
        noise_magnitude_mm=config.sensor_noise.magnitude_mm,
        stage_name=stage_name,
        elapsed_s=elapsed_s,
        metrics=metrics,
        normal_rank=initial.normal_system.rank,
        normal_condition=initial.normal_system.condition,
        rotation_rank=initial.rotation_system.rank,
        rotation_condition=initial.rotation_system.condition,
        translation_rank=initial.translation_system.rank,
        translation_condition=initial.translation_system.condition,
    )
    return trial, initial, pipeline_result


def run_gaussian_noise_sweep(
    base_config: PaperSimulationConfig,
    *,
    sigma_values_mm: Sequence[float],
    trials_per_sigma: int = 40,
    seed: int = 7,
    estimator: Tan2025ClosedFormEstimator | None = None,
) -> tuple[list[PaperTrialResult], list[dict[str, object]]]:
    if trials_per_sigma <= 0:
        raise ValueError("trials_per_sigma must be positive")
    values = [float(value) for value in sigma_values_mm]
    if not values or any(value < 0.0 for value in values):
        raise ValueError("sigma_values_mm must contain non-negative values")

    seed_sequences = np.random.SeedSequence(seed).spawn(
        len(values) * trials_per_sigma
    )
    results: list[PaperTrialResult] = []
    cursor = 0
    for sigma in values:
        config = replace(
            base_config,
            sensor_noise=SensorNoiseConfig(
                mode="gaussian_xz",
                magnitude_mm=sigma,
                dropout_probability=base_config.sensor_noise.dropout_probability,
            ),
        )
        for _ in range(trials_per_sigma):
            child_seed = int(seed_sequences[cursor].generate_state(1)[0])
            cursor += 1
            trial, _, _ = run_paper_trial(
                config,
                seed=child_seed,
                estimator=estimator,
            )
            results.append(trial)

    summary = summarize_trials(results, group_fields=("noise_magnitude_mm",))
    return results, summary


def run_pose_count_sweep(
    base_config: PaperSimulationConfig,
    *,
    translation_counts: Sequence[int],
    composite_counts: Sequence[int],
    seed: int = 7,
    trials_per_cell: int = 1,
    estimator: Tan2025ClosedFormEstimator | None = None,
) -> tuple[list[PaperTrialResult], list[dict[str, object]]]:
    if trials_per_cell <= 0:
        raise ValueError("trials_per_cell must be positive")
    translation = [int(value) for value in translation_counts]
    composite = [int(value) for value in composite_counts]
    if not translation or min(translation) < 4:
        raise ValueError("translation counts must be at least four")
    if not composite or min(composite) < 6:
        raise ValueError("composite counts must be at least six")

    seed_sequences = np.random.SeedSequence(seed).spawn(trials_per_cell)
    results: list[PaperTrialResult] = []
    maximum_config = replace(
        base_config,
        num_translation_poses=max(translation),
        num_composite_poses=max(composite),
    )
    # One maximum dataset per repetition is sliced into nested prefixes.  This
    # is a controlled-variable count experiment: a cell changes the number of
    # poses without also changing all previously acquired poses/noise samples.
    for seed_sequence in seed_sequences:
        child_seed = int(seed_sequence.generate_state(1)[0])
        maximum = AnalyticTan2025Simulator(maximum_config).generate(seed=child_seed)
        for translation_count in translation:
            for composite_count in composite:
                config = replace(
                    base_config,
                    num_translation_poses=translation_count,
                    num_composite_poses=composite_count,
                )
                subset = SimulatedTan2025Dataset(
                    dataset=Tan2025Dataset(
                        maximum.dataset.translation_scans[:translation_count],
                        maximum.dataset.composite_scans[:composite_count],
                        metadata={
                            **maximum.dataset.metadata,
                            "nested_pose_count_subset": True,
                        },
                    ),
                    truth=maximum.truth,
                )
                trial, _, _ = _evaluate_simulated_trial(
                    config,
                    subset,
                    seed=child_seed,
                    estimator=estimator,
                )
                results.append(trial)
    summary = summarize_trials(
        results,
        group_fields=("num_translation_poses", "num_composite_poses"),
    )
    return results, summary


def summarize_trials(
    results: Sequence[PaperTrialResult],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    rows = [item.to_row() for item in results]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    metric_fields = (
        "plane_normal_error_deg",
        "rotation_geodesic_error_deg",
        "rotation_euler_l1_error_deg",
        "translation_error_mm",
        "reconstruction_mean_error_mm",
        "reconstruction_rms_error_mm",
        "self_fitted_mpde_mm",
        "elapsed_s",
    )
    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["trials"] = len(group)
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            )
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_min"] = float(np.min(values))
            summary[f"{field}_max"] = float(np.max(values))
        summaries.append(summary)
    return summaries


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def simulation_config_dict(config: PaperSimulationConfig) -> dict[str, object]:
    """Return every simulator field in a JSON-safe form."""
    return _jsonable(asdict(config))  # type: ignore[return-value]


def _pipeline_stage_dict(stage) -> dict[str, object]:
    result = stage.result
    status_fields = (
        "success",
        "converged",
        "status",
        "message",
        "iterations",
        "nfev",
        "njev",
        "cost",
        "optimality",
        "initial_rms_mm",
        "final_rms_mm",
        "delta_rotation_deg",
        "delta_translation_mm",
        "jacobian_rank",
        "jacobian_condition",
    )
    status = {
        name: _jsonable(getattr(result, name))
        for name in status_fields
        if hasattr(result, name)
    }
    plane_rms_history = getattr(result, "plane_rms_history", None)
    if plane_rms_history:
        status["initial_plane_rms_mm"] = float(plane_rms_history[0])
        status["final_plane_rms_mm"] = float(plane_rms_history[-1])
    return {
        "name": stage.name,
        "elapsed_s": float(stage.elapsed_s),
        "T_ef_s": stage.T_ef_s.tolist(),
        "status": status,
    }


def trial_detail_dict(
    trial: PaperTrialResult,
    estimate: Tan2025Estimate,
    pipeline_result: PipelineResult | None = None,
) -> dict[str, object]:
    diagnostics = {}
    for name in (
        "normal_system",
        "line_system",
        "center_system",
        "rotation_system",
        "translation_system",
    ):
        values = asdict(getattr(estimate, name))
        values["singular_values"] = list(values["singular_values"])
        for key, value in list(values.items()):
            if isinstance(value, float) and not np.isfinite(value):
                values[key] = None
        diagnostics[name] = values
    if pipeline_result is None:
        pipeline = [
            {
                "name": "tan2025_closed_form",
                "elapsed_s": float(estimate.elapsed_s),
                "T_ef_s": estimate.T_ef_s.tolist(),
                "status": {"closed_form": True},
            }
        ]
        final_transform = estimate.T_ef_s
    else:
        pipeline = [_pipeline_stage_dict(stage) for stage in pipeline_result.stages]
        final_transform = pipeline_result.T_ef_s
    return {
        "trial": trial.to_row(),
        "closed_form": {
            "T_ef_s": estimate.T_ef_s.tolist(),
            "plane_normal_base": estimate.plane_normal_base.tolist(),
            "diagnostics": diagnostics,
            "constraint_rms": estimate.constraint_rms,
        },
        "final": {
            "stage_name": trial.stage_name,
            "T_ef_s": final_transform.tolist(),
            "metrics": trial.metrics.to_dict(),
        },
        "pipeline": pipeline,
    }
