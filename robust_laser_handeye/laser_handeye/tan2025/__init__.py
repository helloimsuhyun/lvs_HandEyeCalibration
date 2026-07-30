"""Tan et al. (IEEE TIM, 2025) single-plane hand-eye calibration.

The subpackage intentionally separates the paper's closed-form estimator from
the source of profile data.  An analytic simulator, MuJoCo, and a real robot
can all produce the same :class:`Tan2025Dataset` without changing the solver.
"""

from .estimator import (
    LinearSystemDiagnostics,
    Tan2025ClosedFormEstimator,
    Tan2025Estimate,
    Tan2025EstimatorConfig,
)
from .comparison import (
    METHOD_ORDER,
    ThreeWayComparisonConfig,
    ThreeWayMethodResult,
    ThreeWayTrialResult,
    compare_three_methods_on_shared_simulation,
    flatten_three_way_rows,
    run_three_way_monte_carlo,
    simulation_dataset_sha256,
    summarize_paired_differences,
    summarize_three_way_methods,
    three_way_trials_detail_dict,
)
from .evaluation import Tan2025Metrics, evaluate_tan2025_estimate
from .models import Tan2025Dataset, Tan2025GroundTruth
from .pipeline import (
    AlternatingPlaneRefiner,
    CalibrationPipeline,
    NonlinearPlaneRefiner,
    PipelineResult,
    PipelineStage,
)
from .simulation import (
    AnalyticTan2025Simulator,
    PaperSimulationConfig,
    SensorNoiseConfig,
    SimulatedTan2025Dataset,
    analytic_model_dict,
)

__all__ = [
    "AnalyticTan2025Simulator",
    "AlternatingPlaneRefiner",
    "CalibrationPipeline",
    "LinearSystemDiagnostics",
    "NonlinearPlaneRefiner",
    "PaperSimulationConfig",
    "PipelineResult",
    "PipelineStage",
    "SensorNoiseConfig",
    "SimulatedTan2025Dataset",
    "Tan2025ClosedFormEstimator",
    "Tan2025Dataset",
    "Tan2025Estimate",
    "Tan2025EstimatorConfig",
    "Tan2025GroundTruth",
    "Tan2025Metrics",
    "METHOD_ORDER",
    "ThreeWayComparisonConfig",
    "ThreeWayMethodResult",
    "ThreeWayTrialResult",
    "analytic_model_dict",
    "compare_three_methods_on_shared_simulation",
    "evaluate_tan2025_estimate",
    "flatten_three_way_rows",
    "run_three_way_monte_carlo",
    "simulation_dataset_sha256",
    "summarize_paired_differences",
    "summarize_three_way_methods",
    "three_way_trials_detail_dict",
]
