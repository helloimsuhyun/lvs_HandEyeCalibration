from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Protocol, Sequence

import numpy as np

from ..calibration import PlaneOffsetMode, calibrate_single_plane
from ..nonlinear_refinement import RobustLoss, refine_handeye_nonlinear
from .models import Tan2025Dataset


class TransformResult(Protocol):
    T_ef_s: np.ndarray


class HandEyeEstimator(Protocol):
    name: str

    def estimate(self, dataset: Tan2025Dataset) -> TransformResult:
        ...


class HandEyeRefiner(Protocol):
    name: str

    def refine(
        self,
        dataset: Tan2025Dataset,
        T_init: np.ndarray,
    ) -> TransformResult:
        ...


@dataclass
class PipelineStage:
    name: str
    T_ef_s: np.ndarray
    result: Any
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        transform = np.asarray(self.T_ef_s, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"pipeline stage {self.name!r} returned an invalid transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(
                f"pipeline stage {self.name!r} returned an invalid last row"
            )
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError(
                f"pipeline stage {self.name!r} returned a rotation outside SO(3)"
            )
        self.T_ef_s = transform.copy()
        self.elapsed_s = float(self.elapsed_s)


@dataclass
class PipelineResult:
    stages: list[PipelineStage] = field(default_factory=list)

    @property
    def T_ef_s(self) -> np.ndarray:
        if not self.stages:
            raise RuntimeError("calibration pipeline contains no stages")
        return self.stages[-1].T_ef_s.copy()


class CalibrationPipeline:
    """Closed-form initializer followed by zero or more interchangeable refiners."""

    def __init__(
        self,
        initializer: HandEyeEstimator,
        refiners: Sequence[HandEyeRefiner] = (),
        *,
        require_refiner_success: bool = True,
    ) -> None:
        self.initializer = initializer
        self.refiners = list(refiners)
        self.require_refiner_success = bool(require_refiner_success)

    def run(self, dataset: Tan2025Dataset) -> PipelineResult:
        started = time.perf_counter()
        initial = self.initializer.estimate(dataset)
        stages = [
            PipelineStage(
                name=self.initializer.name,
                T_ef_s=initial.T_ef_s,
                result=initial,
                elapsed_s=time.perf_counter() - started,
            )
        ]
        current = stages[-1].T_ef_s
        for refiner in self.refiners:
            started = time.perf_counter()
            refined = refiner.refine(dataset, current.copy())
            elapsed = time.perf_counter() - started
            self._assert_success(refiner.name, refined)
            stage = PipelineStage(
                name=refiner.name,
                T_ef_s=refined.T_ef_s,
                result=refined,
                elapsed_s=elapsed,
            )
            stages.append(stage)
            current = stage.T_ef_s
        return PipelineResult(stages=stages)

    def _assert_success(self, name: str, result: TransformResult) -> None:
        if not self.require_refiner_success:
            return
        success = getattr(result, "success", None)
        converged = getattr(result, "converged", None)
        if success is False:
            message = getattr(result, "message", "refiner reported failure")
            raise RuntimeError(f"refiner {name!r} failed: {message}")
        if converged is False:
            raise RuntimeError(f"refiner {name!r} did not converge")


@dataclass
class AlternatingPlaneRefiner:
    """Adapter for the repository's existing unknown-plane linear iteration."""

    max_iter: int = 30
    tol: float = 1e-9
    plane_offset_mode: PlaneOffsetMode = "joint"
    max_translation_offset_condition: float = 1e6
    name: str = "alternating_single_plane"

    def refine(
        self,
        dataset: Tan2025Dataset,
        T_init: np.ndarray,
    ) -> TransformResult:
        return calibrate_single_plane(
            dataset.all_scans,
            T_init=T_init,
            max_iter=self.max_iter,
            tol=self.tol,
            plane_offset_mode=self.plane_offset_mode,
            max_translation_offset_condition=self.max_translation_offset_condition,
        )


@dataclass
class NonlinearPlaneRefiner:
    """Adapter for the six-DOF point-to-refitted-plane SciPy refinement."""

    loss: RobustLoss = "linear"
    f_scale_mm: float = 1.0
    max_nfev: int = 200
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    name: str = "nonlinear_refitted_plane"

    def refine(
        self,
        dataset: Tan2025Dataset,
        T_init: np.ndarray,
    ) -> TransformResult:
        return refine_handeye_nonlinear(
            {0: dataset.all_scans},
            T_init=T_init,
            plane_mode="refit",
            loss=self.loss,
            f_scale_mm=self.f_scale_mm,
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
        )
