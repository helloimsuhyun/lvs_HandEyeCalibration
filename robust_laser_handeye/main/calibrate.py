#!/usr/bin/env python3
"""
PYTHONPATH=. python3 main/calibrate.py \
  --collection /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye/results/datasets/tan2025 \
  --output-dir results/calibration/iterative \
  --mode iterative \
  --noise-axis xz \
  --noise-std-mm 0.15 \
  --init-mode carlson \
  --init-translation-range-mm 100 \
  --init-angle-range-deg 15 \
  --init-translation-perturbation direction_norm \
  --init-rotation-perturbation axis_angle \
  --max-iter 3000 \
  --tol 1e-5 \
  --verbose

PYTHONPATH=. python3 main/calibrate.py \
  --collection /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye/results/datasets/tan2025 \
  --output-dir results/calibration/closed \
  --mode closed \
  --noise-axis xz \
  --noise-std-mm 0.35 \
  --verbose

PYTHONPATH=. python3 main/calibrate.py \
  --collection /home/choisuhyun/lvs_HandEyeCalibration/robust_laser_handeye/results/datasets/tan2025 \
  --output-dir results/calibration/closed_to_iterative \
  --mode closed_to_iterative \
  --noise-axis xz \
  --noise-std-mm 0.15 \
  --max-iter 3000 \
  --tol 1e-5 \
  --verbose

"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from laser_handeye.calibration import calibrate_planes
from laser_handeye.calibration_dataset import (
    CalibrationDataset,
    CalibrationTruth,
    load_calibration_dataset,
)
from laser_handeye.geometry import fit_plane_pca
from laser_handeye.initialization import make_initial_guess
from laser_handeye.nonlinear_refinement import (
    refine_handeye_planes_nonlinear,
)
from laser_handeye.se3 import (
    rot_error_deg,
    rotation_vector_error_deg,
    transform_points,
)
from laser_handeye.tan2025 import Tan2025ClosedFormEstimator
from laser_handeye.tan2025.models import Tan2025Dataset


COLLECTION_SCHEMA = "laser_handeye.calibration_dataset_collection"
ITERATIVE_PLANE_OFFSET_MODE = "joint"
ITERATIVE_SOLVER_UPDATE_MODE = "simultaneous"

MOTION_TAG_SCHEMA = "laser_handeye.motion_group"
MOTION_TAG_SCHEMA_VERSION = 1


def _split_translation_composite_scans(
    scans: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
    """Split scans only from explicit metadata written during acquisition."""
    translation: list[tuple[int, int, Any]] = []
    composite: list[tuple[int, int, Any]] = []
    missing: list[int] = []

    for fallback_index, scan in enumerate(scans):
        meta = dict(getattr(scan, "meta", None) or {})
        group = str(meta.get("motion_group", "")).strip().lower()

        if group not in {"translation", "composite"}:
            missing.append(fallback_index)
            continue

        if meta.get("motion_tag_schema") != MOTION_TAG_SCHEMA:
            raise ValueError(
                f"scan {fallback_index} has an invalid motion_tag_schema"
            )
        if int(meta.get("motion_tag_schema_version", -1)) != (
            MOTION_TAG_SCHEMA_VERSION
        ):
            raise ValueError(
                f"scan {fallback_index} has an unsupported motion tag version"
            )

        try:
            group_index = int(meta["group_scan_index"])
            acquisition_index = int(meta["acquisition_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"scan {fallback_index} has incomplete motion metadata: {meta}"
            ) from exc

        entry = (group_index, acquisition_index, scan)
        if group == "translation":
            translation.append(entry)
        else:
            composite.append(entry)

    if missing:
        raise ValueError(
            f"{len(missing)} scans have no explicit translation/composite tag. "
            "Regenerate the dataset with the tagged synthetic generator."
        )

    if not translation or not composite:
        raise ValueError(
            "both translation and composite scan groups are required"
        )

    translation.sort(key=lambda item: (item[0], item[1]))
    composite.sort(key=lambda item: (item[0], item[1]))

    translation_indices = [item[0] for item in translation]
    composite_indices = [item[0] for item in composite]
    if translation_indices != list(range(len(translation))):
        raise ValueError(
            f"translation group_scan_index is not contiguous: "
            f"{translation_indices}"
        )
    if composite_indices != list(range(len(composite))):
        raise ValueError(
            f"composite group_scan_index is not contiguous: "
            f"{composite_indices}"
        )

    return (
        [item[2] for item in translation],
        [item[2] for item in composite],
    )


MODE_ALIASES = {
    "iterative": "iterative",
    "iteraive": "iterative",
    "iterative_joint_nonlinear": "iterative_joint_nonlinear",
    "iterative_to_joint_nonlinear": "iterative_joint_nonlinear",
    "closed": "closed",
    "closed_to_iterative": "closed_to_iterative",
    "closed_to_tieraive": "closed_to_iterative",
    "closed_to_iteraive": "closed_to_iterative",
}


@dataclass
class TrialResult:
    trial_index: int
    relative_path: str
    acquisition_mode: str
    mode: str
    noise_axis: str
    noise_std_mm: float

    converged: bool
    iterations: int
    n_planes: int
    n_scans: int
    n_points: int

    closed_translation_error_mm: float
    closed_rotation_error_deg: float
    init_translation_error_mm: float
    init_rotation_error_deg: float

    translation_error_mm: float
    rotation_error_deg: float
    err_tx_mm: float
    err_ty_mm: float
    err_tz_mm: float
    err_rx_deg: float
    err_ry_deg: float
    err_rz_deg: float

    final_self_fit_plane_rms_mm: float
    final_true_plane_rms_mm: float
    rank_last: int
    condition_last: float
    runtime_s: float
    success: bool
    outlier: bool

    plane_offset_mode: str = ITERATIVE_PLANE_OFFSET_MODE
    solver_update_mode: str = ITERATIVE_SOLVER_UPDATE_MODE
    # Compatibility aliases used by the plotting utilities in the
    # single-plane benchmark script.  They duplicate the canonical fields
    # above so the same final-boxplot code can read this trials.csv directly.
    system_idx: int = -1
    trans_err_norm_mm: float = float("nan")
    rot_err_angle_deg: float = float("nan")
    cond_last: float = float("nan")
    init_trans_err_norm_mm: float = float("nan")
    init_rot_err_angle_deg: float = float("nan")

    alternating_converged: bool = False
    alternating_iterations: int = 0
    alternating_translation_error_mm: float = float("nan")
    alternating_rotation_error_deg: float = float("nan")
    alternating_self_fit_plane_rms_mm: float = float("nan")
    alternating_true_plane_rms_mm: float = float("nan")
    alternating_plane_normal_error_deg: float = float("nan")
    alternating_plane_offset_error_mm: float = float("nan")

    nonlinear_refined: bool = False
    nonlinear_success: bool = False
    nonlinear_status: int = 0
    nonlinear_nfev: int = 0
    nonlinear_runtime_s: float = float("nan")
    nonlinear_initial_rms_mm: float = float("nan")
    nonlinear_final_rms_mm: float = float("nan")
    nonlinear_delta_translation_mm: float = float("nan")
    nonlinear_delta_rotation_deg: float = float("nan")
    nonlinear_mean_plane_normal_delta_deg: float = float("nan")
    nonlinear_max_plane_normal_delta_deg: float = float("nan")
    nonlinear_mean_plane_offset_delta_mm: float = float("nan")
    nonlinear_max_plane_offset_delta_mm: float = float("nan")
    nonlinear_plane_normal_error_deg: float = float("nan")
    nonlinear_plane_offset_error_mm: float = float("nan")
    nonlinear_jacobian_rank: int = -1
    nonlinear_variable_count: int = 0
    nonlinear_jacobian_condition: float = float("nan")
    nonlinear_scaled_jacobian_condition: float = float("nan")
    nonlinear_plane_normals_json: str = ""
    nonlinear_plane_offsets_json: str = ""

    error_type: str = ""
    error_message: str = ""


@dataclass
class IterationResult:
    """One solver state for long-format iteration plotting.

    ``iteration == 0`` is the explicit initial estimate for iterative modes.
    Subsequent rows correspond to ``calibrate_planes().T_history``.  The
    closed-form-only mode contains a single final row.
    """

    trial_index: int
    system_idx: int
    relative_path: str
    acquisition_mode: str
    mode: str
    noise_axis: str
    noise_std_mm: float

    iteration: int
    is_initial: bool
    is_final: bool

    translation_error_mm: float
    translation_error_norm_mm: float
    rotation_error_deg: float
    rotation_error_geodesic_deg: float
    T_frobenius_error: float

    err_tx_mm: float
    err_ty_mm: float
    err_tz_mm: float
    err_rx_deg: float
    err_ry_deg: float
    err_rz_deg: float

    step_translation_mm: float
    step_rotation_deg: float
    self_fit_plane_rms_mm: float
    plane_rms_mm: float
    true_plane_rms_mm: float
    rank: int
    condition: float


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _calibration_mode(text: str) -> str:
    key = text.strip().lower()
    if key not in MODE_ALIASES:
        valid = ", ".join(
            (
                "iterative",
                "iterative_joint_nonlinear",
                "closed",
                "closed_to_iterative",
            )
        )
        raise argparse.ArgumentTypeError(f"invalid mode {text!r}; choose one of: {valid}")
    return MODE_ALIASES[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate every stored trial in an immutable dataset collection."
    )
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=_calibration_mode,
        default="iterative",
        metavar=(
            "{iterative,iterative_joint_nonlinear,closed,"
            "closed_to_iterative}"
        ),
        help=(
            "iterative: existing alternating solver; closed: Tan 2025 closed-form; "
            "closed_to_iterative: closed-form result used as iterative initialization; "
            "iterative_joint_nonlinear: alternating result initializes a joint "
            "nonlinear refinement of SE(3), plane normals, and plane offsets"
        ),
    )
    parser.add_argument("--only-trial", type=int, default=None)
    parser.add_argument("--max-trials", type=_positive_int, default=None)
    parser.add_argument(
        "--max-scans-per-trial",
        type=_positive_int,
        default=None,
        help=(
            "Use only the first N scans in the single acquisition group's "
            "explicit sequence order. Intended for sequential scan-budget "
            "prefix experiments."
        ),
    )

    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--noise-std-mm", type=float, default=0.0)
    parser.add_argument(
        "--noise-axis",
        choices=("z", "xz"),
        default="xz",
        help=(
            "z: independent Gaussian noise on profile Z only; "
            "xz: independent Gaussian noise on measured profile X and Z"
        ),
    )

    parser.add_argument(
        "--init-mode",
        choices=("relative", "carlson"),
        default="carlson",
        help="generic initialization used only by --mode iterative",
    )
    parser.add_argument("--rel-offset", type=float, default=0.1)
    parser.add_argument("--init-translation-range-mm", type=float, default=100.0)
    parser.add_argument("--init-angle-range-deg", type=float, default=15.0)
    parser.add_argument(
        "--init-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"),
        default="axis_angle",
        help=(
            "axis_angle: isotropic random axis with geodesic angle in "
            "[0, --init-angle-range-deg] (default); euler_xyz: legacy "
            "independent per-axis Euler box perturbation"
        ),
    )
    parser.add_argument(
        "--init-translation-perturbation",
        choices=("direction_norm", "box_xyz"),
        default="direction_norm",
        help=(
            "direction_norm: isotropic random direction with norm in "
            "[0, --init-translation-range-mm] (default); box_xyz: legacy "
            "independent per-axis translation box perturbation"
        ),
    )

    parser.add_argument("--max-iter", type=_positive_int, default=3500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument(
        "--nonlinear-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="linear",
        help="least-squares loss for iterative_joint_nonlinear",
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument("--nonlinear-max-nfev", type=_positive_int, default=300)
    parser.add_argument("--nonlinear-ftol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-xtol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-gtol", type=float, default=1e-10)

    parser.add_argument("--translation-success-mm", type=float, default=1.0)
    parser.add_argument("--rotation-success-deg", type=float, default=0.1)
    parser.add_argument("--outlier-translation-mm", type=float, default=5.0)
    parser.add_argument("--outlier-rotation-deg", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-traceback", action="store_true")
    return parser


def _read_collection(collection_dir: Path) -> dict[str, Any]:
    manifest_path = collection_dir / "collection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"collection manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != COLLECTION_SCHEMA:
        raise ValueError(
            f"unexpected collection schema: {payload.get('schema')!r}; "
            f"expected {COLLECTION_SCHEMA!r}"
        )
    if payload.get("status") != "complete":
        raise ValueError(
            f"collection is not complete: status={payload.get('status')!r}"
        )
    return payload


def _prefix_dataset(
    dataset: CalibrationDataset,
    maximum_scans: int | None,
) -> CalibrationDataset:
    if maximum_scans is None:
        return dataset
    if maximum_scans > len(dataset.scans):
        raise ValueError(
            f"requested {maximum_scans} scans, but trial contains only "
            f"{len(dataset.scans)}"
        )
    included_groups = [
        group for group in dataset.groups if group.include_in_calibration
    ]
    if len(included_groups) != 1:
        raise ValueError(
            "--max-scans-per-trial currently requires exactly one included "
            "acquisition group"
        )
    group_id = included_groups[0].group_id
    ordered_indices = [
        index
        for index, candidate in enumerate(map(str, dataset.scan_group_ids))
        if candidate == group_id
    ]
    ordered_indices.sort(
        key=lambda index: int(dataset.sequence_indices[index])
    )
    if len(ordered_indices) != len(dataset.scans):
        raise ValueError(
            "--max-scans-per-trial requires every scan to belong to the "
            "single included acquisition group"
        )
    selected = ordered_indices[:maximum_scans]

    truth = dataset.truth
    prefix_truth = None
    if truth is not None:
        prefix_truth = CalibrationTruth(
            T_ef_s_true=truth.T_ef_s_true,
            planes=truth.planes,
            T_base_ef_true=(
                None
                if truth.T_base_ef_true is None
                else np.asarray(truth.T_base_ef_true)[selected]
            ),
            T_base_ef_commanded=(
                None
                if truth.T_base_ef_commanded is None
                else np.asarray(truth.T_base_ef_commanded)[selected]
            ),
            metadata=truth.metadata,
        )
    metadata = copy.deepcopy(dataset.metadata)
    metadata.update(
        {
            "scan_budget_prefix_applied": True,
            "scan_budget_prefix_count": int(maximum_scans),
            "scan_budget_source_count": len(dataset.scans),
            "scan_budget_order": "explicit_sequence_index",
        }
    )
    return CalibrationDataset(
        scans=[dataset.scans[index] for index in selected],
        scan_group_ids=[str(dataset.scan_group_ids[index]) for index in selected],
        sequence_indices=list(range(len(selected))),
        groups=dataset.groups,
        acquisition_mode=dataset.acquisition_mode,
        source=dataset.source,
        profile_state=dataset.profile_state,
        truth=prefix_truth,
        metadata=metadata,
        channel_id_semantics=dataset.channel_id_semantics,
    )


def _get_member(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_transform(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4, got {array.shape}")
    return array


def _extract_truth_transform(dataset: Any) -> np.ndarray:
    direct = _get_member(
        dataset,
        (
            "T_ef_s_true",
            "T_es_true",
            "T_handeye_true",
            "handeye_true",
            "ground_truth_handeye",
        ),
    )
    if direct is not None:
        return _as_transform(direct, "ground-truth hand-eye")

    truth = _get_member(dataset, ("truth", "ground_truth", "gt"))
    if truth is not None:
        nested = _get_member(
            truth,
            (
                "T_ef_s",
                "T_ef_s_true",
                "T_es",
                "handeye",
                "handeye_transform",
            ),
        )
        if nested is not None:
            return _as_transform(nested, "ground-truth hand-eye")

    manifest = _get_member(dataset, ("manifest", "metadata", "meta"), {})
    for container in (
        manifest,
        _get_member(manifest, ("truth", "ground_truth"), {}),
    ):
        nested = _get_member(
            container,
            ("T_ef_s_true", "T_ef_s", "handeye_true", "handeye_transform"),
        )
        if nested is not None:
            return _as_transform(nested, "ground-truth hand-eye")

    raise AttributeError(
        "could not locate the saved GT hand-eye transform in the loaded dataset"
    )


def _looks_like_scan(value: Any) -> bool:
    return hasattr(value, "T_base_ef") and (
        hasattr(value, "valid_points_s") or hasattr(value, "points_s")
    )


def _scan_points(scan: Any) -> np.ndarray:
    points = getattr(scan, "valid_points_s", None)
    if points is None:
        points = getattr(scan, "points_s")
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _replace_scan_points(scan: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    if hasattr(scan, "points_s"):
        setattr(scan, "points_s", points)
        return
    if hasattr(scan, "valid_points_s"):
        try:
            setattr(scan, "valid_points_s", points)
            return
        except AttributeError:
            pass
    raise AttributeError("LaserScan point array is not mutable")


def _flatten_scans(value: Any) -> list[Any]:
    if value is None:
        return []
    if _looks_like_scan(value):
        return [value]
    if isinstance(value, Mapping):
        output: list[Any] = []
        for child in value.values():
            output.extend(_flatten_scans(child))
        return output
    if isinstance(value, (str, bytes, np.ndarray)):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    output: list[Any] = []
    for item in items:
        output.extend(_flatten_scans(item))
    return output


def _extract_all_scans(dataset: Any) -> list[Any]:
    candidates = [
        _get_member(
            dataset,
            (
                "scans_by_plane",
                "plane_scans",
                "scans",
                "laser_scans",
                "profiles",
                "translation_scans",
                "composite_scans",
            ),
        ),
        _get_member(dataset, ("acquisition", "data")),
    ]
    for candidate in candidates:
        scans = _flatten_scans(candidate)
        if scans:
            # Preserve object order while removing duplicate object references.
            seen: set[int] = set()
            unique: list[Any] = []
            for scan in scans:
                marker = id(scan)
                if marker not in seen:
                    seen.add(marker)
                    unique.append(scan)
            return unique
    raise AttributeError("could not locate LaserScan objects in loaded dataset")


def _extract_scans_by_plane(dataset: Any) -> dict[int, list[Any]]:
    scans = _extract_all_scans(dataset)
    grouped: dict[int, list[Any]] = {}
    for scan in scans:
        meta = getattr(scan, "meta", {}) or {}
        plane_id = int(meta.get("plane_id", getattr(scan, "plane_id", 0)))
        grouped.setdefault(plane_id, []).append(scan)
    return grouped


def _extract_translation_composite_groups(
    dataset: Any,
) -> tuple[list[Any], list[Any]]:
    """Split Tan motion groups strictly from explicit scan metadata tags."""
    return _split_translation_composite_scans(_extract_all_scans(dataset))


def _normalize(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(float).eps:
        raise ValueError("zero-length vector")
    return value / norm


def _extract_true_planes(dataset: Any) -> dict[int, tuple[np.ndarray, float]]:
    candidate = _get_member(
        dataset,
        ("planes", "true_planes", "ground_truth_planes", "plane_truth"),
    )
    if candidate is None:
        truth = _get_member(dataset, ("truth", "ground_truth", "gt"))
        candidate = _get_member(
            truth,
            ("planes", "true_planes", "ground_truth_planes", "plane_truth"),
        )

    if candidate is None:
        normal = _get_member(
            dataset,
            ("plane_n", "plane_normal", "plane_normal_base", "normal_base"),
        )
        offset = _get_member(
            dataset,
            ("plane_l", "plane_offset", "plane_offset_mm", "offset_mm"),
        )
        if normal is not None and offset is not None:
            n = _normalize(normal)
            l = float(offset)
            if l < 0.0:
                n, l = -n, -l
            return {0: (n, l)}
        return {}

    output: dict[int, tuple[np.ndarray, float]] = {}
    items = candidate.items() if isinstance(candidate, Mapping) else enumerate(candidate)

    for key, plane in items:
        if isinstance(plane, (tuple, list)) and len(plane) >= 2:
            normal, offset = plane[0], plane[1]
            plane_id = int(key)
        else:
            normal = _get_member(
                plane,
                ("n", "normal", "plane_n", "normal_base"),
            )
            offset = _get_member(
                plane,
                ("l", "offset", "plane_l", "distance_mm", "offset_mm"),
            )
            plane_id = int(_get_member(plane, ("plane_id", "id"), key))

        if normal is None or offset is None:
            continue

        n = _normalize(normal)
        l = float(offset)
        if l < 0.0:
            n, l = -n, -l
        output[plane_id] = (n, l)

    return output

def _add_noise(
    dataset: Any,
    rng: np.random.Generator,
    std_mm: float,
    axis_mode: str,
) -> Any:
    noisy_dataset = copy.deepcopy(dataset)
    if std_mm == 0.0:
        return noisy_dataset
    if std_mm < 0.0:
        raise ValueError("noise std must be non-negative")

    for scan in _extract_all_scans(noisy_dataset):
        points = _scan_points(scan).copy()
        if axis_mode == "z":
            points[:, 2] += rng.normal(0.0, std_mm, size=len(points))
        elif axis_mode == "xz":
            # Laser profile coordinates are [x, 0, z]. Preserve y=0.
            points[:, [0, 2]] += rng.normal(
                0.0,
                std_mm,
                size=(len(points), 2),
            )
        else:  # protected by argparse
            raise ValueError(f"unsupported noise-axis: {axis_mode}")
        _replace_scan_points(scan, points)

    return noisy_dataset


def _reconstruct_points(scans: Iterable[Any], T_ef_s: np.ndarray) -> np.ndarray:
    groups: list[np.ndarray] = []
    for scan in scans:
        points_s = _scan_points(scan)
        if len(points_s) == 0:
            continue
        points_ef = transform_points(T_ef_s, points_s)
        points_base = transform_points(np.asarray(scan.T_base_ef), points_ef)
        groups.append(points_base)
    return np.vstack(groups) if groups else np.empty((0, 3), dtype=float)


def _self_fit_rms(scans_by_plane: dict[int, list[Any]], T: np.ndarray) -> float:
    values: list[float] = []
    for scans in scans_by_plane.values():
        points = _reconstruct_points(scans, T)
        if len(points) < 3:
            return float("inf")
        _, _, _, rms = fit_plane_pca(points)
        values.append(float(rms))
    return float(np.mean(values)) if values else float("inf")


def _true_plane_rms(
    scans_by_plane: dict[int, list[Any]],
    T: np.ndarray,
    true_planes: dict[int, tuple[np.ndarray, float]],
) -> float:
    if not true_planes:
        return float("nan")
    values: list[float] = []
    for plane_id, scans in scans_by_plane.items():
        if plane_id not in true_planes:
            continue
        points = _reconstruct_points(scans, T)
        if len(points) == 0:
            continue
        n, l = true_planes[plane_id]
        residuals = points @ n - l
        values.append(float(np.sqrt(np.mean(residuals**2))))
    return float(np.mean(values)) if values else float("nan")


def _estimate_planes(
    scans_by_plane: dict[int, list[Any]],
    T: np.ndarray,
) -> dict[int, tuple[np.ndarray, float]]:
    """Fit one base-frame plane to each physical scan group."""
    estimates: dict[int, tuple[np.ndarray, float]] = {}
    for plane_id, scans in scans_by_plane.items():
        points = _reconstruct_points(scans, T)
        if len(points) < 3:
            raise ValueError(
                f"plane {plane_id} has fewer than three reconstructed points"
            )
        normal, offset, _centroid, _rms = fit_plane_pca(points)
        estimates[plane_id] = (
            np.asarray(normal, dtype=float).copy(),
            float(offset),
        )
    return estimates


def _mean_plane_parameter_errors(
    estimated_planes: Mapping[int, tuple[np.ndarray, float]],
    true_planes: Mapping[int, tuple[np.ndarray, float]],
) -> tuple[float, float]:
    """Return mean sign-aligned normal-angle and offset errors."""
    normal_errors: list[float] = []
    offset_errors: list[float] = []
    for plane_id, (normal_est, offset_est) in estimated_planes.items():
        if plane_id not in true_planes:
            continue
        normal_true, offset_true = true_planes[plane_id]
        normal_est = _normalize(normal_est)
        normal_true = _normalize(normal_true)
        sign = 1.0 if float(normal_est @ normal_true) >= 0.0 else -1.0
        aligned_normal = sign * normal_est
        aligned_offset = sign * float(offset_est)
        normal_errors.append(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(aligned_normal @ normal_true),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        )
        offset_errors.append(abs(aligned_offset - float(offset_true)))

    if not normal_errors:
        return float("nan"), float("nan")
    return float(np.mean(normal_errors)), float(np.mean(offset_errors))


def _generic_initial_guess(
    T_true: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    angles = (
        None
        if args.init_rotation_perturbation == "axis_angle"
        else Rotation.from_matrix(T_true[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        )
    )
    return make_initial_guess(
        reference_angles_deg=angles,
        reference_translation_mm=T_true[:3, 3],
        reference_rotation=T_true[:3, :3],
        rng=rng,
        mode=args.init_mode,
        rel_offset=args.rel_offset,
        translation_range_mm=args.init_translation_range_mm,
        angle_range_deg=args.init_angle_range_deg,
        rotation_perturbation=args.init_rotation_perturbation,
        translation_perturbation=args.init_translation_perturbation,
    )


def _extract_estimated_transform(result: Any) -> np.ndarray:
    direct_names = (
        "T_ef_s",
        "T_ef_s_est",
        "T_es",
        "T_handeye",
        "handeye",
        "transform",
        "T",
    )
    direct = _get_member(result, direct_names)
    if direct is not None:
        return _as_transform(direct, "closed-form estimate")

    if isinstance(result, np.ndarray):
        return _as_transform(result, "closed-form estimate")

    if isinstance(result, (tuple, list)):
        for item in result:
            try:
                return _extract_estimated_transform(item)
            except (AttributeError, ValueError, TypeError):
                continue

    nested = _get_member(result, ("estimate", "result", "solution"))
    if nested is not None and nested is not result:
        return _extract_estimated_transform(nested)

    raise AttributeError(
        "Tan2025ClosedFormEstimator returned an object without a recognizable "
        "4x4 hand-eye transform"
    )


def _invoke_closed_estimator(dataset: Any) -> np.ndarray:
    """Build the estimator's real Tan2025Dataset from explicit motion tags."""
    translation_scans, composite_scans = (
        _extract_translation_composite_groups(dataset)
    )

    tan_dataset = Tan2025Dataset(
        translation_scans=translation_scans,
        composite_scans=composite_scans,
        metadata={
            "source": "calibration_dataset_collection",
            "motion_group_source": "explicit_scan_tags",
        },
    )

    estimator = Tan2025ClosedFormEstimator()
    result = estimator.estimate(tan_dataset)
    return _extract_estimated_transform(result)


def _run_iterative(
    scans_by_plane: dict[int, list[Any]],
    T_init: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    """Run the alternating solver and retain its complete history object."""
    kwargs = {
        "T_init": T_init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "plane_offset_mode": ITERATIVE_PLANE_OFFSET_MODE,
        "solver_update_mode": ITERATIVE_SOLVER_UPDATE_MODE,
    }
    signature = inspect.signature(calibrate_planes)
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return calibrate_planes(scans_by_plane, **filtered)


def _transform_errors(
    T_est: np.ndarray,
    T_true: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])),
        float(rot_error_deg(T_est[:3, :3], T_true[:3, :3])),
    )


def _rotation_step_deg(T_previous: np.ndarray, T_current: np.ndarray) -> float:
    """Geodesic rotation change between successive solver states [deg]."""
    return float(
        rot_error_deg(
            np.asarray(T_current, dtype=float)[:3, :3],
            np.asarray(T_previous, dtype=float)[:3, :3],
        )
    )


def _make_iteration_rows(
    *,
    trial_index: int,
    relative_path: str,
    acquisition_mode: str,
    args: argparse.Namespace,
    T_true: np.ndarray,
    T_init: np.ndarray,
    T_est: np.ndarray,
    scans_by_plane: dict[int, list[Any]],
    true_planes: dict[int, tuple[np.ndarray, float]],
    solver_result: Any | None,
) -> list[IterationResult]:
    """Convert solver histories into one long-format table.

    The initial transform is always row 0 for iterative modes.  History arrays
    returned by the solver are aligned with post-update ``T_history`` states.
    Missing diagnostics are represented by NaN/-1 rather than dropping rows.
    """
    if solver_result is None:
        states = [np.asarray(T_est, dtype=float).reshape(4, 4)]
        rank_history: list[Any] = []
        cond_history: list[Any] = []
        plane_rms_history: list[Any] = []
        explicit_initial = False
    else:
        states = [np.asarray(T_init, dtype=float).reshape(4, 4)]
        states.extend(
            np.asarray(value, dtype=float).reshape(4, 4)
            for value in getattr(solver_result, "T_history", [])
        )
        rank_history = list(getattr(solver_result, "rank_history", []))
        cond_history = list(getattr(solver_result, "cond_history", []))
        plane_rms_history = list(
            getattr(solver_result, "plane_rms_history", [])
        )
        explicit_initial = True

        # Some solver implementations do not place the returned final transform
        # in T_history.  Append it only when it is genuinely a distinct state.
        final = np.asarray(T_est, dtype=float).reshape(4, 4)
        if not states or not np.allclose(
            states[-1], final, rtol=1e-11, atol=1e-11
        ):
            states.append(final)

    rows: list[IterationResult] = []
    for state_index, T_state in enumerate(states):
        history_index = state_index - 1 if explicit_initial else state_index
        is_initial = bool(explicit_initial and state_index == 0)
        is_final = state_index == len(states) - 1

        t_error, r_error = _transform_errors(T_state, T_true)
        dt = T_state[:3, 3] - T_true[:3, 3]
        dr = rotation_vector_error_deg(
            T_state[:3, :3],
            T_true[:3, :3],
        )

        if state_index == 0:
            step_translation = float("nan")
            step_rotation = float("nan")
        else:
            previous = states[state_index - 1]
            step_translation = float(
                np.linalg.norm(T_state[:3, 3] - previous[:3, 3])
            )
            step_rotation = _rotation_step_deg(previous, T_state)

        rank = -1
        condition = float("nan")
        stored_plane_rms = float("nan")
        if explicit_initial and history_index >= 0:
            if history_index < len(rank_history):
                rank = int(rank_history[history_index])
            if history_index < len(cond_history):
                condition = float(cond_history[history_index])
            if history_index < len(plane_rms_history):
                stored_plane_rms = float(plane_rms_history[history_index])

        self_fit_rms = _self_fit_rms(scans_by_plane, T_state)
        if not np.isfinite(stored_plane_rms):
            stored_plane_rms = self_fit_rms

        rows.append(
            IterationResult(
                trial_index=trial_index,
                system_idx=trial_index,
                relative_path=relative_path,
                acquisition_mode=acquisition_mode,
                mode=args.mode,
                noise_axis=args.noise_axis,
                noise_std_mm=float(args.noise_std_mm),
                iteration=state_index,
                is_initial=is_initial,
                is_final=is_final,
                translation_error_mm=t_error,
                translation_error_norm_mm=t_error,
                rotation_error_deg=r_error,
                rotation_error_geodesic_deg=r_error,
                T_frobenius_error=float(np.linalg.norm(T_state - T_true, ord="fro")),
                err_tx_mm=float(dt[0]),
                err_ty_mm=float(dt[1]),
                err_tz_mm=float(dt[2]),
                err_rx_deg=float(dr[0]),
                err_ry_deg=float(dr[1]),
                err_rz_deg=float(dr[2]),
                step_translation_mm=step_translation,
                step_rotation_deg=step_rotation,
                self_fit_plane_rms_mm=self_fit_rms,
                plane_rms_mm=stored_plane_rms,
                true_plane_rms_mm=_true_plane_rms(
                    scans_by_plane,
                    T_state,
                    true_planes,
                ),
                rank=rank,
                condition=condition,
            )
        )

    return rows


def _run_trial(
    trial_index: int,
    relative_path: str,
    acquisition_mode: str,
    dataset_clean: Any,
    args: argparse.Namespace,
) -> tuple[TrialResult, list[IterationResult]]:
    start = time.perf_counter()
    T_true = _extract_truth_transform(dataset_clean)

    trial_seed = np.random.SeedSequence([args.seed, trial_index])
    noise_seed, init_seed = trial_seed.spawn(2)
    dataset = _add_noise(
        dataset_clean,
        np.random.default_rng(noise_seed),
        args.noise_std_mm,
        args.noise_axis,
    )
    scans_by_plane = _extract_scans_by_plane(dataset)
    true_planes = _extract_true_planes(dataset)

    n_scans = sum(len(values) for values in scans_by_plane.values())
    n_points = sum(
        len(_scan_points(scan))
        for values in scans_by_plane.values()
        for scan in values
    )

    nan = float("nan")
    T_closed: np.ndarray | None = None
    closed_t_error = nan
    closed_r_error = nan
    rank_last = -1
    condition_last = nan
    solver_result: Any | None = None
    alternating_converged = False
    alternating_iterations = 0
    alternating_t_error = nan
    alternating_r_error = nan
    alternating_self_fit_rms = nan
    alternating_true_plane_rms = nan
    alternating_plane_normal_error = nan
    alternating_plane_offset_error = nan
    nonlinear_refined = False
    nonlinear_success = False
    nonlinear_status = 0
    nonlinear_nfev = 0
    nonlinear_runtime_s = nan
    nonlinear_initial_rms = nan
    nonlinear_final_rms = nan
    nonlinear_delta_translation = nan
    nonlinear_delta_rotation = nan
    nonlinear_mean_normal_delta = nan
    nonlinear_max_normal_delta = nan
    nonlinear_mean_offset_delta = nan
    nonlinear_max_offset_delta = nan
    nonlinear_plane_normal_error = nan
    nonlinear_plane_offset_error = nan
    nonlinear_jacobian_rank = -1
    nonlinear_variable_count = 0
    nonlinear_jacobian_condition = nan
    nonlinear_scaled_jacobian_condition = nan
    nonlinear_plane_normals_json = ""
    nonlinear_plane_offsets_json = ""

    if args.mode in {"closed", "closed_to_iterative"}:
        if acquisition_mode != "translation_composite":
            raise ValueError(
                f"--mode {args.mode} requires acquisition_mode='translation_composite', "
                f"got {acquisition_mode!r}"
            )
        T_closed = _invoke_closed_estimator(dataset)
        closed_t_error, closed_r_error = _transform_errors(T_closed, T_true)

    if args.mode == "closed":
        assert T_closed is not None
        T_init = T_closed.copy()
        T_est = T_closed
        init_t_error = closed_t_error
        init_r_error = closed_r_error
        converged = True
        iterations = 0

    elif args.mode == "closed_to_iterative":
        assert T_closed is not None
        T_init = T_closed.copy()
        init_t_error = closed_t_error
        init_r_error = closed_r_error
        solver_result = _run_iterative(scans_by_plane, T_init, args)
        T_est = np.asarray(solver_result.T_ef_s, dtype=float).reshape(4, 4)
        rank_history = list(getattr(solver_result, "rank_history", []))
        cond_history = list(getattr(solver_result, "cond_history", []))
        converged = bool(getattr(solver_result, "converged", False))
        iterations = int(
            getattr(
                solver_result,
                "iterations",
                len(getattr(solver_result, "T_history", [])),
            )
        )
        rank_last = int(rank_history[-1]) if rank_history else -1
        condition_last = (
            float(cond_history[-1]) if cond_history else float("nan")
        )
        alternating_converged = converged
        alternating_iterations = iterations
        alternating_t_error, alternating_r_error = _transform_errors(
            T_est,
            T_true,
        )
        alternating_self_fit_rms = _self_fit_rms(scans_by_plane, T_est)
        alternating_true_plane_rms = _true_plane_rms(
            scans_by_plane,
            T_est,
            true_planes,
        )

    elif args.mode in {"iterative", "iterative_joint_nonlinear"}:
        T_init = _generic_initial_guess(
            T_true,
            np.random.default_rng(init_seed),
            args,
        )
        init_t_error, init_r_error = _transform_errors(T_init, T_true)
        solver_result = _run_iterative(scans_by_plane, T_init, args)
        T_est = np.asarray(solver_result.T_ef_s, dtype=float).reshape(4, 4)
        rank_history = list(getattr(solver_result, "rank_history", []))
        cond_history = list(getattr(solver_result, "cond_history", []))
        converged = bool(getattr(solver_result, "converged", False))
        iterations = int(
            getattr(
                solver_result,
                "iterations",
                len(getattr(solver_result, "T_history", [])),
            )
        )
        rank_last = int(rank_history[-1]) if rank_history else -1
        condition_last = (
            float(cond_history[-1]) if cond_history else float("nan")
        )
        alternating_converged = converged
        alternating_iterations = iterations
        alternating_t_error, alternating_r_error = _transform_errors(
            T_est,
            T_true,
        )
        alternating_self_fit_rms = _self_fit_rms(scans_by_plane, T_est)
        alternating_true_plane_rms = _true_plane_rms(
            scans_by_plane,
            T_est,
            true_planes,
        )
        alternating_planes = _estimate_planes(scans_by_plane, T_est)
        (
            alternating_plane_normal_error,
            alternating_plane_offset_error,
        ) = _mean_plane_parameter_errors(
            alternating_planes,
            true_planes,
        )

        if args.mode == "iterative_joint_nonlinear":
            nonlinear_refined = True
            nonlinear_started = time.perf_counter()
            nonlinear_result = refine_handeye_planes_nonlinear(
                scans_by_plane,
                T_est,
                initial_planes=alternating_planes,
                loss=args.nonlinear_loss,
                f_scale_mm=args.nonlinear_f_scale_mm,
                max_nfev=args.nonlinear_max_nfev,
                ftol=args.nonlinear_ftol,
                xtol=args.nonlinear_xtol,
                gtol=args.nonlinear_gtol,
            )
            nonlinear_runtime_s = time.perf_counter() - nonlinear_started
            T_est = nonlinear_result.T_ef_s
            converged = bool(nonlinear_result.success)
            nonlinear_success = bool(nonlinear_result.success)
            nonlinear_status = int(nonlinear_result.status)
            nonlinear_nfev = int(nonlinear_result.nfev)
            nonlinear_initial_rms = float(
                nonlinear_result.initial_rms_mm
            )
            nonlinear_final_rms = float(nonlinear_result.final_rms_mm)
            nonlinear_delta_translation = float(
                nonlinear_result.delta_translation_mm
            )
            nonlinear_delta_rotation = float(
                nonlinear_result.delta_rotation_deg
            )
            nonlinear_mean_normal_delta = float(
                nonlinear_result.mean_plane_normal_delta_deg
            )
            nonlinear_max_normal_delta = float(
                nonlinear_result.max_plane_normal_delta_deg
            )
            nonlinear_mean_offset_delta = float(
                nonlinear_result.mean_plane_offset_delta_mm
            )
            nonlinear_max_offset_delta = float(
                nonlinear_result.max_plane_offset_delta_mm
            )
            nonlinear_jacobian_rank = int(
                nonlinear_result.jacobian_rank
            )
            nonlinear_variable_count = int(
                nonlinear_result.variable_count
            )
            nonlinear_jacobian_condition = float(
                nonlinear_result.jacobian_condition
            )
            nonlinear_scaled_jacobian_condition = float(
                nonlinear_result.scaled_jacobian_condition
            )
            refined_planes = {
                plane_id: (
                    nonlinear_result.plane_normals[plane_id],
                    nonlinear_result.plane_offsets_mm[plane_id],
                )
                for plane_id in nonlinear_result.plane_normals
            }
            (
                nonlinear_plane_normal_error,
                nonlinear_plane_offset_error,
            ) = _mean_plane_parameter_errors(
                refined_planes,
                true_planes,
            )
            nonlinear_plane_normals_json = json.dumps(
                {
                    str(plane_id): normal.tolist()
                    for plane_id, normal in (
                        nonlinear_result.plane_normals.items()
                    )
                },
                sort_keys=True,
            )
            nonlinear_plane_offsets_json = json.dumps(
                {
                    str(plane_id): offset
                    for plane_id, offset in (
                        nonlinear_result.plane_offsets_mm.items()
                    )
                },
                sort_keys=True,
            )

    else:  # protected by parser
        raise AssertionError(args.mode)

    dt = T_est[:3, 3] - T_true[:3, 3]
    dr = rotation_vector_error_deg(T_est[:3, :3], T_true[:3, :3])
    t_error, r_error = _transform_errors(T_est, T_true)
    success = bool(
        converged
        and t_error <= args.translation_success_mm
        and r_error <= args.rotation_success_deg
    )
    outlier = bool(
        (not converged)
        or t_error > args.outlier_translation_mm
        or r_error > args.outlier_rotation_deg
    )

    trial_result = TrialResult(
        trial_index=trial_index,
        relative_path=relative_path,
        acquisition_mode=acquisition_mode,
        mode=args.mode,
        noise_axis=args.noise_axis,
        noise_std_mm=float(args.noise_std_mm),
        converged=converged,
        iterations=iterations,
        n_planes=len(scans_by_plane),
        n_scans=n_scans,
        n_points=n_points,
        closed_translation_error_mm=closed_t_error,
        closed_rotation_error_deg=closed_r_error,
        init_translation_error_mm=init_t_error,
        init_rotation_error_deg=init_r_error,
        translation_error_mm=t_error,
        rotation_error_deg=r_error,
        err_tx_mm=float(dt[0]),
        err_ty_mm=float(dt[1]),
        err_tz_mm=float(dt[2]),
        err_rx_deg=float(dr[0]),
        err_ry_deg=float(dr[1]),
        err_rz_deg=float(dr[2]),
        final_self_fit_plane_rms_mm=_self_fit_rms(scans_by_plane, T_est),
        final_true_plane_rms_mm=_true_plane_rms(
            scans_by_plane,
            T_est,
            true_planes,
        ),
        rank_last=rank_last,
        condition_last=condition_last,
        runtime_s=time.perf_counter() - start,
        success=success,
        outlier=outlier,
        system_idx=trial_index,
        trans_err_norm_mm=t_error,
        rot_err_angle_deg=r_error,
        cond_last=condition_last,
        init_trans_err_norm_mm=init_t_error,
        init_rot_err_angle_deg=init_r_error,
        alternating_converged=alternating_converged,
        alternating_iterations=alternating_iterations,
        alternating_translation_error_mm=alternating_t_error,
        alternating_rotation_error_deg=alternating_r_error,
        alternating_self_fit_plane_rms_mm=alternating_self_fit_rms,
        alternating_true_plane_rms_mm=alternating_true_plane_rms,
        alternating_plane_normal_error_deg=(
            alternating_plane_normal_error
        ),
        alternating_plane_offset_error_mm=(
            alternating_plane_offset_error
        ),
        nonlinear_refined=nonlinear_refined,
        nonlinear_success=nonlinear_success,
        nonlinear_status=nonlinear_status,
        nonlinear_nfev=nonlinear_nfev,
        nonlinear_runtime_s=nonlinear_runtime_s,
        nonlinear_initial_rms_mm=nonlinear_initial_rms,
        nonlinear_final_rms_mm=nonlinear_final_rms,
        nonlinear_delta_translation_mm=nonlinear_delta_translation,
        nonlinear_delta_rotation_deg=nonlinear_delta_rotation,
        nonlinear_mean_plane_normal_delta_deg=(
            nonlinear_mean_normal_delta
        ),
        nonlinear_max_plane_normal_delta_deg=(
            nonlinear_max_normal_delta
        ),
        nonlinear_mean_plane_offset_delta_mm=(
            nonlinear_mean_offset_delta
        ),
        nonlinear_max_plane_offset_delta_mm=(
            nonlinear_max_offset_delta
        ),
        nonlinear_plane_normal_error_deg=(
            nonlinear_plane_normal_error
        ),
        nonlinear_plane_offset_error_mm=(
            nonlinear_plane_offset_error
        ),
        nonlinear_jacobian_rank=nonlinear_jacobian_rank,
        nonlinear_variable_count=nonlinear_variable_count,
        nonlinear_jacobian_condition=nonlinear_jacobian_condition,
        nonlinear_scaled_jacobian_condition=(
            nonlinear_scaled_jacobian_condition
        ),
        nonlinear_plane_normals_json=nonlinear_plane_normals_json,
        nonlinear_plane_offsets_json=nonlinear_plane_offsets_json,
    )

    iteration_rows = _make_iteration_rows(
        trial_index=trial_index,
        relative_path=relative_path,
        acquisition_mode=acquisition_mode,
        args=args,
        T_true=T_true,
        T_init=T_init,
        T_est=T_est,
        scans_by_plane=scans_by_plane,
        true_planes=true_planes,
        solver_result=solver_result,
    )
    return trial_result, iteration_rows


def _failure_result(
    trial_index: int,
    relative_path: str,
    acquisition_mode: str,
    exc: Exception,
    runtime_s: float,
    args: argparse.Namespace,
) -> TrialResult:
    nan = float("nan")
    return TrialResult(
        trial_index=trial_index,
        relative_path=relative_path,
        acquisition_mode=acquisition_mode,
        mode=args.mode,
        noise_axis=args.noise_axis,
        noise_std_mm=float(args.noise_std_mm),
        converged=False,
        iterations=0,
        n_planes=0,
        n_scans=0,
        n_points=0,
        closed_translation_error_mm=nan,
        closed_rotation_error_deg=nan,
        init_translation_error_mm=nan,
        init_rotation_error_deg=nan,
        translation_error_mm=nan,
        rotation_error_deg=nan,
        err_tx_mm=nan,
        err_ty_mm=nan,
        err_tz_mm=nan,
        err_rx_deg=nan,
        err_ry_deg=nan,
        err_rz_deg=nan,
        final_self_fit_plane_rms_mm=nan,
        final_true_plane_rms_mm=nan,
        rank_last=-1,
        condition_last=nan,
        runtime_s=runtime_s,
        success=False,
        outlier=True,
        system_idx=trial_index,
        trans_err_norm_mm=nan,
        rot_err_angle_deg=nan,
        cond_last=nan,
        init_trans_err_norm_mm=nan,
        init_rot_err_angle_deg=nan,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _write_dataclass_csv(
    path: Path,
    rows: Sequence[Any],
    row_type: type,
) -> None:
    """Write dataclass rows while preserving headers for an empty CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _summary(
    results: Sequence[TrialResult],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    completed = [row for row in results if not row.error_type]
    t = _finite(row.translation_error_mm for row in completed)
    r = _finite(row.rotation_error_deg for row in completed)
    summary = {
        "requested_trials": len(results),
        "completed_trials": len(completed),
        "failed_trials": sum(bool(row.error_type) for row in results),
        "converged_trials": sum(row.converged for row in completed),
        "successful_trials": sum(row.success for row in completed),
        "outlier_trials": sum(row.outlier for row in results),
        "success_rate": (
            float(np.mean([row.success for row in completed]))
            if completed
            else 0.0
        ),
        "translation_error_mm": {
            "median": float(np.median(t)) if len(t) else None,
            "mean": float(np.mean(t)) if len(t) else None,
            "p95": float(np.percentile(t, 95)) if len(t) else None,
            "max": float(np.max(t)) if len(t) else None,
        },
        "rotation_error_deg": {
            "median": float(np.median(r)) if len(r) else None,
            "mean": float(np.mean(r)) if len(r) else None,
            "p95": float(np.percentile(r, 95)) if len(r) else None,
            "max": float(np.max(r)) if len(r) else None,
        },
        "config": dict(config),
    }

    refined = [row for row in completed if row.nonlinear_refined]
    if refined:
        alternating_t = np.asarray(
            [row.alternating_translation_error_mm for row in refined],
            dtype=float,
        )
        alternating_r = np.asarray(
            [row.alternating_rotation_error_deg for row in refined],
            dtype=float,
        )
        refined_t = np.asarray(
            [row.translation_error_mm for row in refined],
            dtype=float,
        )
        refined_r = np.asarray(
            [row.rotation_error_deg for row in refined],
            dtype=float,
        )
        paired_finite = (
            np.isfinite(alternating_t)
            & np.isfinite(alternating_r)
            & np.isfinite(refined_t)
            & np.isfinite(refined_r)
        )
        alternating_t = alternating_t[paired_finite]
        alternating_r = alternating_r[paired_finite]
        refined_t = refined_t[paired_finite]
        refined_r = refined_r[paired_finite]
        delta_t = alternating_t - refined_t
        delta_r = alternating_r - refined_r

        summary["joint_nonlinear_comparison"] = {
            "paired_trials": int(len(delta_t)),
            "nonlinear_successful_trials": int(
                sum(row.nonlinear_success for row in refined)
            ),
            "alternating_translation_error_mm_median": (
                float(np.median(alternating_t))
                if len(alternating_t)
                else None
            ),
            "refined_translation_error_mm_median": (
                float(np.median(refined_t)) if len(refined_t) else None
            ),
            "translation_improvement_mm_median": (
                float(np.median(delta_t)) if len(delta_t) else None
            ),
            "translation_improved_fraction": (
                float(np.mean(delta_t > 0.0)) if len(delta_t) else None
            ),
            "alternating_rotation_error_deg_median": (
                float(np.median(alternating_r))
                if len(alternating_r)
                else None
            ),
            "refined_rotation_error_deg_median": (
                float(np.median(refined_r)) if len(refined_r) else None
            ),
            "rotation_improvement_deg_median": (
                float(np.median(delta_r)) if len(delta_r) else None
            ),
            "rotation_improved_fraction": (
                float(np.mean(delta_r > 0.0)) if len(delta_r) else None
            ),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.noise_std_mm < 0.0:
        raise SystemExit("--noise-std-mm must be non-negative")
    if args.nonlinear_f_scale_mm <= 0.0:
        raise SystemExit("--nonlinear-f-scale-mm must be positive")
    if args.only_trial is not None and args.only_trial < 0:
        raise SystemExit("--only-trial must be non-negative")

    collection_dir = args.collection.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    collection = _read_collection(collection_dir)
    acquisition_mode = str(collection.get("acquisition_mode", ""))
    if not acquisition_mode:
        raise SystemExit("collection.json does not contain acquisition_mode")

    if args.mode in {"closed", "closed_to_iterative"}:
        if acquisition_mode != "translation_composite":
            raise SystemExit(
                f"--mode {args.mode} requires a translation-composite dataset; "
                f"collection acquisition_mode is {acquisition_mode!r}"
            )

    entries = list(collection.get("trials", []))
    if args.only_trial is not None:
        entries = [
            entry
            for entry in entries
            if int(entry["trial_index"]) == args.only_trial
        ]
        if not entries:
            raise SystemExit(f"trial {args.only_trial} not found")
    if args.max_trials is not None:
        entries = entries[: args.max_trials]

    results: list[TrialResult] = []
    iteration_results: list[IterationResult] = []
    total_start = time.perf_counter()

    print(
        f"[dataset-calibration] mode={args.mode} | "
        f"acquisition={acquisition_mode} | "
        f"noise={args.noise_axis}, sigma={args.noise_std_mm:g} mm"
    )
    if args.mode in {
        "iterative",
        "iterative_joint_nonlinear",
        "closed_to_iterative",
    }:
        print(
            "[dataset-calibration] iterative solver fixed to "
            f"plane_offset_mode={ITERATIVE_PLANE_OFFSET_MODE}, "
            f"solver_update_mode={ITERATIVE_SOLVER_UPDATE_MODE}"
        )

    for ordinal, entry in enumerate(entries, start=1):
        trial_index = int(entry["trial_index"])
        relative_path = str(entry["relative_path"])
        trial_path = collection_dir / relative_path
        started = time.perf_counter()

        try:
            dataset = load_calibration_dataset(trial_path)
            dataset = _prefix_dataset(
                dataset,
                args.max_scans_per_trial,
            )
            row, trial_iteration_rows = _run_trial(
                trial_index,
                relative_path,
                acquisition_mode,
                dataset,
                args,
            )
        except Exception as exc:
            trial_iteration_rows = []
            row = _failure_result(
                trial_index,
                relative_path,
                acquisition_mode,
                exc,
                time.perf_counter() - started,
                args,
            )
            if args.debug_traceback:
                traceback.print_exc()

        results.append(row)
        iteration_results.extend(trial_iteration_rows)

        if args.verbose or ordinal == len(entries) or ordinal % 5 == 0:
            status = "FAIL" if row.error_type else ("OK" if row.success else "MISS")
            closed_text = (
                f" closed_t={row.closed_translation_error_mm:.6g} mm"
                if np.isfinite(row.closed_translation_error_mm)
                else ""
            )
            print(
                f"[{ordinal:04d}/{len(entries):04d}] trial={trial_index:06d} "
                f"{status} conv={row.converged} iter={row.iterations}"
                f"{closed_text} final_t={row.translation_error_mm:.6g} mm "
                f"final_r={row.rotation_error_deg:.6g} deg"
            )
            if row.nonlinear_refined:
                print(
                    "    alternating -> joint nonlinear: "
                    f"t {row.alternating_translation_error_mm:.6g} -> "
                    f"{row.translation_error_mm:.6g} mm, "
                    f"r {row.alternating_rotation_error_deg:.6g} -> "
                    f"{row.rotation_error_deg:.6g} deg, "
                    f"rank {row.nonlinear_jacobian_rank}/"
                    f"{row.nonlinear_variable_count}"
                )

    trials_csv = output_dir / "trials.csv"
    iterations_csv = output_dir / "iterations.csv"
    failures_csv = output_dir / "failures.csv"
    summary_json = output_dir / "summary.json"
    _write_dataclass_csv(trials_csv, results, TrialResult)
    _write_dataclass_csv(
        iterations_csv,
        iteration_results,
        IterationResult,
    )
    _write_dataclass_csv(
        failures_csv,
        [row for row in results if row.error_type],
        TrialResult,
    )

    config = {
        key: (value.as_posix() if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update(
        {
            "plane_offset_mode": ITERATIVE_PLANE_OFFSET_MODE,
            "solver_update_mode": ITERATIVE_SOLVER_UPDATE_MODE,
            "elapsed_seconds": time.perf_counter() - total_start,
        }
    )
    summary = _summary(results, config)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== DATASET CALIBRATION SUMMARY ===")
    print(
        f"mode={args.mode} | completed={summary['completed_trials']}/"
        f"{summary['requested_trials']} | converged={summary['converged_trials']} | "
        f"success={summary['successful_trials']} | "
        f"outliers={summary['outlier_trials']}"
    )
    print(
        "translation error [mm]: "
        f"median={summary['translation_error_mm']['median']}, "
        f"p95={summary['translation_error_mm']['p95']}, "
        f"max={summary['translation_error_mm']['max']}"
    )
    print(
        "rotation error [deg]: "
        f"median={summary['rotation_error_deg']['median']}, "
        f"p95={summary['rotation_error_deg']['p95']}, "
        f"max={summary['rotation_error_deg']['max']}"
    )
    if "joint_nonlinear_comparison" in summary:
        comparison = summary["joint_nonlinear_comparison"]
        print(
            "alternating -> joint nonlinear median: "
            f"translation "
            f"{comparison['alternating_translation_error_mm_median']} -> "
            f"{comparison['refined_translation_error_mm_median']} mm, "
            f"rotation "
            f"{comparison['alternating_rotation_error_deg_median']} -> "
            f"{comparison['refined_rotation_error_deg_median']} deg"
        )
    print(f"saved: {trials_csv}")
    print(f"saved: {iterations_csv}")
    print(f"saved: {failures_csv}")
    print(f"saved: {summary_json}")

    return 0 if summary["completed_trials"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
