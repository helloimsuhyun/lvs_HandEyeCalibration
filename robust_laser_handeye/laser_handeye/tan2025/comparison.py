from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
import time
from typing import Callable, Literal, Sequence

import numpy as np

from ..calibration import (
    CalibrationResult,
    PlaneOffsetMode,
    mean_self_fitted_plane_rms,
)
from ..initialization import make_initial_guess
from ..se3 import rot_error_deg
from .estimator import Tan2025ClosedFormEstimator, Tan2025Estimate
from .evaluation import (
    Tan2025Metrics,
    evaluate_tan2025_estimate,
    euler_zyx_deg,
)
from .pipeline import AlternatingPlaneRefiner
from .simulation import (
    AnalyticTan2025Simulator,
    PaperSimulationConfig,
    SimulatedTan2025Dataset,
)


MethodName = Literal[
    "tan_closed_form",
    "tan_then_alternating",
    "alternating_only",
]
IterativeInitMode = Literal["relative_gt", "carlson_gt", "identity"]

METHOD_ORDER: tuple[MethodName, ...] = (
    "tan_closed_form",
    "tan_then_alternating",
    "alternating_only",
)


@dataclass(frozen=True)
class ThreeWayComparisonConfig:
    """Monte Carlo policy shared by the two alternating-solver branches."""

    trials: int = 100
    seed: int = 7
    iterative_only_init: IterativeInitMode = "identity"
    relative_offset: float = 0.1
    carlson_translation_range_mm: float = 200.0
    carlson_angle_range_deg: float = 30.0
    max_iter: int = 100
    tol: float = 1e-9
    plane_offset_mode: PlaneOffsetMode = "joint"
    max_translation_offset_condition: float = 1e6

    def __post_init__(self) -> None:
        for name in ("trials", "seed", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.iterative_only_init not in (
            "relative_gt",
            "carlson_gt",
            "identity",
        ):
            raise ValueError(
                "iterative_only_init must be relative_gt, carlson_gt, or identity"
            )
        for name in (
            "relative_offset",
            "carlson_translation_range_mm",
            "carlson_angle_range_deg",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.tol):
            raise ValueError("tol must be finite")
        if self.plane_offset_mode not in ("joint", "difference", "fitted"):
            raise ValueError(
                "plane_offset_mode must be joint, difference, or fitted"
            )
        if (
            not np.isfinite(self.max_translation_offset_condition)
            or self.max_translation_offset_condition <= 1.0
        ):
            raise ValueError(
                "max_translation_offset_condition must be finite and greater than one"
            )

    @property
    def iterative_init_uses_ground_truth(self) -> bool:
        return self.iterative_only_init in ("relative_gt", "carlson_gt")


@dataclass
class ThreeWayMethodResult:
    trial_index: int
    dataset_seed: int
    initializer_seed: int
    dataset_sha256: str
    method: MethodName
    status: str
    completed: bool
    converged: bool
    total_elapsed_s: float
    initializer_elapsed_s: float
    refinement_elapsed_s: float
    iterations: int
    init_source: str
    initial_translation_error_mm: float | None
    initial_rotation_error_deg: float | None
    final_plane_rms_mm: float | None
    rank_last: int | None
    condition_last: float | None
    metrics: Tan2025Metrics | None
    T_ef_s: np.ndarray | None
    T_init: np.ndarray | None
    error_type: str = ""
    error_message: str = ""

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "trial_index": self.trial_index,
            "dataset_seed": self.dataset_seed,
            "initializer_seed": self.initializer_seed,
            "dataset_sha256": self.dataset_sha256,
            "method": self.method,
            "status": self.status,
            "completed": self.completed,
            "converged": self.converged,
            "total_elapsed_s": self.total_elapsed_s,
            "initializer_elapsed_s": self.initializer_elapsed_s,
            "refinement_elapsed_s": self.refinement_elapsed_s,
            "iterations": self.iterations,
            "init_source": self.init_source,
            "initial_translation_error_mm": self.initial_translation_error_mm,
            "initial_rotation_error_deg": self.initial_rotation_error_deg,
            "final_plane_rms_mm": self.final_plane_rms_mm,
            "rank_last": self.rank_last,
            "condition_last": self.condition_last,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        metric_names = tuple(Tan2025Metrics.__dataclass_fields__)
        if self.metrics is None:
            row.update({name: None for name in metric_names})
        else:
            row.update(self.metrics.to_dict())
        return row

    def to_detail_dict(self) -> dict[str, object]:
        detail = self.to_row()
        detail["T_ef_s"] = None if self.T_ef_s is None else self.T_ef_s.tolist()
        detail["T_init"] = None if self.T_init is None else self.T_init.tolist()
        return detail


@dataclass
class ThreeWayTrialResult:
    trial_index: int
    dataset_seed: int
    initializer_seed: int
    dataset_sha256: str
    T_ef_s_true: np.ndarray
    plane_normal_base_true: np.ndarray
    plane_offset_mm_true: float
    methods: list[ThreeWayMethodResult]

    def __post_init__(self) -> None:
        self.T_ef_s_true = np.asarray(self.T_ef_s_true, dtype=float).copy()
        self.plane_normal_base_true = np.asarray(
            self.plane_normal_base_true,
            dtype=float,
        ).copy()

    def to_detail_dict(self) -> dict[str, object]:
        return {
            "trial_index": self.trial_index,
            "dataset_seed": self.dataset_seed,
            "initializer_seed": self.initializer_seed,
            "dataset_sha256": self.dataset_sha256,
            "truth": {
                "T_ef_s": self.T_ef_s_true.tolist(),
                "plane_normal_base": self.plane_normal_base_true.tolist(),
                "plane_offset_mm": self.plane_offset_mm_true,
            },
            "methods": [result.to_detail_dict() for result in self.methods],
        }


def simulation_dataset_sha256(simulated: SimulatedTan2025Dataset) -> str:
    """Hash the ordered environment, robot poses, profiles, and channel IDs."""
    digest = sha256()

    def update_array(value: np.ndarray) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())

    update_array(simulated.truth.T_ef_s)
    update_array(simulated.truth.plane_normal_base)
    update_array(np.asarray([simulated.truth.plane_offset_mm], dtype=np.float64))
    for group_name, scans in (
        ("translation", simulated.dataset.translation_scans),
        ("composite", simulated.dataset.composite_scans),
    ):
        digest.update(group_name.encode("ascii"))
        digest.update(str(len(scans)).encode("ascii"))
        for scan in scans:
            update_array(scan.T_base_ef)
            update_array(scan.points_s)
            channel_ids = scan.meta.get("channel_ids")
            if channel_ids is not None:
                update_array(np.asarray(channel_ids, dtype=np.int64))
    return digest.hexdigest()


def _trial_seeds(master_seed: int, trial_index: int) -> tuple[int, int]:
    children = np.random.SeedSequence([master_seed, trial_index]).spawn(2)
    dataset_seed = int(children[0].generate_state(1, dtype=np.uint32)[0])
    initializer_seed = int(children[1].generate_state(1, dtype=np.uint32)[0])
    return dataset_seed, initializer_seed


def _make_iterative_only_initial(
    simulated: SimulatedTan2025Dataset,
    config: ThreeWayComparisonConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.iterative_only_init == "identity":
        return np.eye(4, dtype=float)

    true = simulated.truth.T_ef_s
    # euler_zyx_deg returns [Z, Y, X], while make_initial_guess consumes [X, Y, Z].
    reference_angles_xyz_deg = euler_zyx_deg(true[:3, :3])[::-1]
    if config.iterative_only_init == "relative_gt":
        return make_initial_guess(
            reference_angles_deg=reference_angles_xyz_deg,
            reference_translation_mm=true[:3, 3],
            rng=rng,
            mode="relative",
            rel_offset=config.relative_offset,
        )
    return make_initial_guess(
        reference_angles_deg=reference_angles_xyz_deg,
        reference_translation_mm=true[:3, 3],
        rng=rng,
        mode="carlson",
        translation_range_mm=config.carlson_translation_range_mm,
        angle_range_deg=config.carlson_angle_range_deg,
    )


def _initial_errors(
    initial: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.linalg.norm(initial[:3, 3] - truth[:3, 3])),
        float(rot_error_deg(initial[:3, :3], truth[:3, :3])),
    )


def _iterative_method_result(
    *,
    method: MethodName,
    simulated: SimulatedTan2025Dataset,
    refiner: AlternatingPlaneRefiner,
    initial: np.ndarray,
    init_source: str,
    trial_index: int,
    dataset_seed: int,
    initializer_seed: int,
    dataset_hash: str,
    initializer_elapsed_s: float,
) -> ThreeWayMethodResult:
    initial_translation_error, initial_rotation_error = _initial_errors(
        initial,
        simulated.truth.T_ef_s,
    )
    started = time.perf_counter()
    try:
        result = refiner.refine(simulated.dataset, initial.copy())
        refinement_elapsed = time.perf_counter() - started
        if not isinstance(result, CalibrationResult):
            raise TypeError("alternating refiner must return CalibrationResult")
        if simulation_dataset_sha256(simulated) != dataset_hash:
            raise RuntimeError("calibration method mutated the shared dataset")
        normal = (
            result.plane_normals_history[-1][0]
            if result.plane_normals_history
            and result.plane_normals_history[-1]
            else None
        )
        metrics = evaluate_tan2025_estimate(
            simulated.dataset,
            simulated.truth,
            result.T_ef_s,
            estimated_plane_normal_base=normal,
        )
        return ThreeWayMethodResult(
            trial_index=trial_index,
            dataset_seed=dataset_seed,
            initializer_seed=initializer_seed,
            dataset_sha256=dataset_hash,
            method=method,
            status="converged" if result.converged else "nonconverged",
            completed=True,
            converged=bool(result.converged),
            total_elapsed_s=initializer_elapsed_s + refinement_elapsed,
            initializer_elapsed_s=initializer_elapsed_s,
            refinement_elapsed_s=refinement_elapsed,
            iterations=int(result.iterations),
            init_source=init_source,
            initial_translation_error_mm=initial_translation_error,
            initial_rotation_error_deg=initial_rotation_error,
            final_plane_rms_mm=float(result.plane_rms_history[-1]),
            rank_last=(
                int(result.rank_history[-1]) if result.rank_history else None
            ),
            condition_last=(
                float(result.cond_history[-1]) if result.cond_history else None
            ),
            metrics=metrics,
            T_ef_s=result.T_ef_s,
            T_init=initial.copy(),
        )
    except Exception as exception:
        refinement_elapsed = time.perf_counter() - started
        return ThreeWayMethodResult(
            trial_index=trial_index,
            dataset_seed=dataset_seed,
            initializer_seed=initializer_seed,
            dataset_sha256=dataset_hash,
            method=method,
            status="failed",
            completed=False,
            converged=False,
            total_elapsed_s=initializer_elapsed_s + refinement_elapsed,
            initializer_elapsed_s=initializer_elapsed_s,
            refinement_elapsed_s=refinement_elapsed,
            iterations=0,
            init_source=init_source,
            initial_translation_error_mm=initial_translation_error,
            initial_rotation_error_deg=initial_rotation_error,
            final_plane_rms_mm=None,
            rank_last=None,
            condition_last=None,
            metrics=None,
            T_ef_s=None,
            T_init=initial.copy(),
            error_type=type(exception).__name__,
            error_message=str(exception),
        )


def compare_three_methods_on_shared_simulation(
    simulated: SimulatedTan2025Dataset,
    comparison_config: ThreeWayComparisonConfig,
    *,
    trial_index: int,
    dataset_seed: int,
    initializer_seed: int,
    estimator: Tan2025ClosedFormEstimator | None = None,
) -> ThreeWayTrialResult:
    """Run all methods without regenerating or copying the simulated profiles."""
    dataset_hash = simulation_dataset_sha256(simulated)
    closed_form = estimator or Tan2025ClosedFormEstimator()
    refiner = AlternatingPlaneRefiner(
        max_iter=comparison_config.max_iter,
        tol=comparison_config.tol,
        plane_offset_mode=comparison_config.plane_offset_mode,
        max_translation_offset_condition=(
            comparison_config.max_translation_offset_condition
        ),
    )

    initializer_started = time.perf_counter()
    baseline_initial = _make_iterative_only_initial(
        simulated,
        comparison_config,
        np.random.default_rng(initializer_seed),
    )
    baseline_initializer_elapsed = time.perf_counter() - initializer_started

    methods: list[ThreeWayMethodResult] = []
    tan_estimate: Tan2025Estimate | None = None
    tan_started = time.perf_counter()
    try:
        tan_estimate = closed_form.estimate(simulated.dataset)
        tan_elapsed = time.perf_counter() - tan_started
        if simulation_dataset_sha256(simulated) != dataset_hash:
            raise RuntimeError("Tan estimator mutated the shared dataset")
        tan_metrics = evaluate_tan2025_estimate(
            simulated.dataset,
            simulated.truth,
            tan_estimate.T_ef_s,
            estimated_plane_normal_base=tan_estimate.plane_normal_base,
        )
        methods.append(
            ThreeWayMethodResult(
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
                dataset_sha256=dataset_hash,
                method="tan_closed_form",
                status="completed",
                completed=True,
                converged=True,
                total_elapsed_s=tan_elapsed,
                initializer_elapsed_s=tan_elapsed,
                refinement_elapsed_s=0.0,
                iterations=0,
                init_source="none_closed_form",
                initial_translation_error_mm=None,
                initial_rotation_error_deg=None,
                final_plane_rms_mm=mean_self_fitted_plane_rms(
                    {0: simulated.dataset.all_scans},
                    tan_estimate.T_ef_s,
                ),
                rank_last=tan_estimate.rotation_system.rank,
                condition_last=tan_estimate.rotation_system.condition,
                metrics=tan_metrics,
                T_ef_s=tan_estimate.T_ef_s,
                T_init=None,
            )
        )
    except Exception as exception:
        tan_elapsed = time.perf_counter() - tan_started
        methods.append(
            ThreeWayMethodResult(
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
                dataset_sha256=dataset_hash,
                method="tan_closed_form",
                status="failed",
                completed=False,
                converged=False,
                total_elapsed_s=tan_elapsed,
                initializer_elapsed_s=tan_elapsed,
                refinement_elapsed_s=0.0,
                iterations=0,
                init_source="none_closed_form",
                initial_translation_error_mm=None,
                initial_rotation_error_deg=None,
                final_plane_rms_mm=None,
                rank_last=None,
                condition_last=None,
                metrics=None,
                T_ef_s=None,
                T_init=None,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )
        )

    if tan_estimate is None:
        tan_then_result = ThreeWayMethodResult(
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
                dataset_sha256=dataset_hash,
                method="tan_then_alternating",
                status="skipped_dependency",
                completed=False,
                converged=False,
                total_elapsed_s=tan_elapsed,
                initializer_elapsed_s=tan_elapsed,
                refinement_elapsed_s=0.0,
                iterations=0,
                init_source="tan_closed_form",
                initial_translation_error_mm=None,
                initial_rotation_error_deg=None,
                final_plane_rms_mm=None,
                rank_last=None,
                condition_last=None,
                metrics=None,
                T_ef_s=None,
                T_init=None,
                error_type="DependencyError",
                error_message="Tan closed-form initialization failed",
            )
        alternating_only_result = _iterative_method_result(
            method="alternating_only",
            simulated=simulated,
            refiner=refiner,
            initial=baseline_initial,
            init_source=comparison_config.iterative_only_init,
            trial_index=trial_index,
            dataset_seed=dataset_seed,
            initializer_seed=initializer_seed,
            dataset_hash=dataset_hash,
            initializer_elapsed_s=baseline_initializer_elapsed,
        )
    else:
        def run_tan_then() -> ThreeWayMethodResult:
            return _iterative_method_result(
                method="tan_then_alternating",
                simulated=simulated,
                refiner=refiner,
                initial=tan_estimate.T_ef_s,
                init_source="tan_closed_form",
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
                dataset_hash=dataset_hash,
                initializer_elapsed_s=tan_elapsed,
            )

        def run_alternating_only() -> ThreeWayMethodResult:
            return _iterative_method_result(
                method="alternating_only",
                simulated=simulated,
                refiner=refiner,
                initial=baseline_initial,
                init_source=comparison_config.iterative_only_init,
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
                dataset_hash=dataset_hash,
                initializer_elapsed_s=baseline_initializer_elapsed,
            )

        # Alternate execution order to reduce systematic cache/order bias in
        # the paired runtime comparison. Stored method order remains fixed.
        if trial_index % 2 == 0:
            tan_then_result = run_tan_then()
            alternating_only_result = run_alternating_only()
        else:
            alternating_only_result = run_alternating_only()
            tan_then_result = run_tan_then()

    methods.extend(
        [
            tan_then_result,
            alternating_only_result,
        ]
    )
    if simulation_dataset_sha256(simulated) != dataset_hash:
        raise RuntimeError("a comparison method mutated the shared dataset")

    return ThreeWayTrialResult(
        trial_index=trial_index,
        dataset_seed=dataset_seed,
        initializer_seed=initializer_seed,
        dataset_sha256=dataset_hash,
        T_ef_s_true=simulated.truth.T_ef_s,
        plane_normal_base_true=simulated.truth.plane_normal_base,
        plane_offset_mm_true=simulated.truth.plane_offset_mm,
        methods=methods,
    )


def run_three_way_monte_carlo(
    simulation_config: PaperSimulationConfig,
    comparison_config: ThreeWayComparisonConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> list[ThreeWayTrialResult]:
    """Generate one Tan scene per trial and evaluate the three paired methods."""
    trials: list[ThreeWayTrialResult] = []
    simulator = AnalyticTan2025Simulator(simulation_config)
    for trial_index in range(comparison_config.trials):
        dataset_seed, initializer_seed = _trial_seeds(
            comparison_config.seed,
            trial_index,
        )
        simulated = simulator.generate(seed=dataset_seed)
        trials.append(
            compare_three_methods_on_shared_simulation(
                simulated,
                comparison_config,
                trial_index=trial_index,
                dataset_seed=dataset_seed,
                initializer_seed=initializer_seed,
            )
        )
        if progress is not None:
            progress(trial_index + 1, comparison_config.trials)
    return trials


def flatten_three_way_rows(
    trials: Sequence[ThreeWayTrialResult],
) -> list[dict[str, object]]:
    return [method.to_row() for trial in trials for method in trial.methods]


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def three_way_trials_detail_dict(
    trials: Sequence[ThreeWayTrialResult],
) -> list[dict[str, object]]:
    return _json_safe(  # type: ignore[return-value]
        [trial.to_detail_dict() for trial in trials]
    )


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1) if len(array) > 1 else 0.0),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def summarize_three_way_methods(
    trials: Sequence[ThreeWayTrialResult],
) -> list[dict[str, object]]:
    results = [method for trial in trials for method in trial.methods]
    summary: list[dict[str, object]] = []
    value_fields = (
        "rotation_geodesic_error_deg",
        "rotation_euler_l1_error_deg",
        "translation_error_mm",
        "reconstruction_mean_error_mm",
        "reconstruction_rms_error_mm",
        "self_fitted_mpde_mm",
        "total_elapsed_s",
        "refinement_elapsed_s",
        "iterations",
        "initial_translation_error_mm",
        "initial_rotation_error_deg",
        "final_plane_rms_mm",
    )
    for method_name in METHOD_ORDER:
        group = [item for item in results if item.method == method_name]
        completed = [item for item in group if item.completed]
        row: dict[str, object] = {
            "method": method_name,
            "trials_requested": len(group),
            "completed": len(completed),
            "converged": sum(item.converged for item in group),
            "failed": sum(item.status == "failed" for item in group),
            "skipped_dependency": sum(
                item.status == "skipped_dependency" for item in group
            ),
            "completion_rate": len(completed) / len(group) if group else 0.0,
            "convergence_rate": (
                sum(item.converged for item in group) / len(group) if group else 0.0
            ),
        }
        for field_name in value_fields:
            values: list[float] = []
            for item in completed:
                if item.metrics is not None and hasattr(item.metrics, field_name):
                    value = getattr(item.metrics, field_name)
                else:
                    value = getattr(item, field_name)
                if value is not None:
                    values.append(float(value))
            for statistic, value in _distribution(values).items():
                row[f"{field_name}_{statistic}"] = value
        summary.append(row)
    return summary


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float | None, float | None]:
    if not len(values):
        return None, None
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    bootstrap_means = np.mean(values[indices], axis=1)
    return (
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def summarize_paired_differences(
    trials: Sequence[ThreeWayTrialResult],
    *,
    bootstrap_seed: int = 0,
) -> list[dict[str, object]]:
    pairs: tuple[tuple[MethodName, MethodName], ...] = (
        ("tan_then_alternating", "tan_closed_form"),
        ("alternating_only", "tan_closed_form"),
        ("tan_then_alternating", "alternating_only"),
    )
    metrics = (
        "rotation_geodesic_error_deg",
        "translation_error_mm",
        "reconstruction_mean_error_mm",
        "total_elapsed_s",
        "iterations",
    )
    rows: list[dict[str, object]] = []
    for pair_index, (first_name, second_name) in enumerate(pairs):
        for metric_index, metric_name in enumerate(metrics):
            if metric_name == "iterations" and "tan_closed_form" in (
                first_name,
                second_name,
            ):
                continue
            differences: list[float] = []
            both_converged = 0
            for trial in trials:
                lookup = {item.method: item for item in trial.methods}
                first = lookup[first_name]
                second = lookup[second_name]
                if not first.completed or not second.completed:
                    continue
                if first.converged and second.converged:
                    both_converged += 1
                if metric_name in ("total_elapsed_s", "iterations"):
                    first_value = float(getattr(first, metric_name))
                    second_value = float(getattr(second, metric_name))
                else:
                    if first.metrics is None or second.metrics is None:
                        continue
                    first_value = float(getattr(first.metrics, metric_name))
                    second_value = float(getattr(second.metrics, metric_name))
                if np.isfinite(first_value) and np.isfinite(second_value):
                    differences.append(first_value - second_value)
            values = np.asarray(differences, dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                seed=bootstrap_seed + pair_index * 100 + metric_index,
            )
            tie_tolerance = {
                "rotation_geodesic_error_deg": 1e-8,
                "translation_error_mm": 1e-8,
                "reconstruction_mean_error_mm": 1e-8,
                "total_elapsed_s": 1e-6,
                "iterations": 0.5,
            }[metric_name]
            rows.append(
                {
                    "first_method": first_name,
                    "second_method": second_name,
                    "metric": metric_name,
                    "difference_definition": "first_minus_second",
                    "population": "completed pairs, including nonconverged",
                    "paired_trials": len(values),
                    "both_converged": both_converged,
                    "mean_difference": (
                        float(np.mean(values)) if len(values) else None
                    ),
                    "median_difference": (
                        float(np.median(values)) if len(values) else None
                    ),
                    "mean_difference_ci95_low": ci_low,
                    "mean_difference_ci95_high": ci_high,
                    "tie_tolerance": tie_tolerance,
                    "first_wins": int(
                        np.count_nonzero(values < -tie_tolerance)
                    ),
                    "ties": int(
                        np.count_nonzero(np.abs(values) <= tie_tolerance)
                    ),
                    "first_loses": int(
                        np.count_nonzero(values > tie_tolerance)
                    ),
                }
            )
    return rows
