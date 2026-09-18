#!/usr/bin/env python3
"""
exp_rl_adaptive_handeye.py

PYTHONPATH=. python \
robust_laser_handeye/well_made_experiment/exp_rl_adaptive_handeye_final.py \
--mode train \
--episodes 5000 \
--candidate-pool-size 96 \
--train-unavailable-prob-range 0.0 0.30 \
--model runs/rl_handeye/ddqn.pt

Core idea
---------
The RL policy does NOT learn robot IK.

For each decision:
    1. Generate a variable-size bank of board-relative sensor candidates from
       the CURRENT estimated plane.
    2. Score every candidate with Q(state, candidate).
    3. Rank all candidates by Q.
    4. Apply a deterministic robot feasibility callback top-down:
           IK / joint limit / collision
       If no callback is supplied, no fake/random IK mask is used.
    5. Execute the first feasible candidate.
    6. Simulate/measure the laser profile, update joint hand-eye + plane estimate.
    7. Repeat or choose STOP.

Important simulation correction
-------------------------------
A desired sensor pose is converted to a robot end-effector command using the
CURRENT hand-eye estimate:

    T_base_ef_cmd = T_base_s_desired @ inv(T_ef_s_est)

The physical sensor pose used to generate the measurement is then

    T_base_s_actual = T_base_ef_cmd @ T_ef_s_true

Therefore initial hand-eye error actually perturbs where the sensor ends up.
The same principle is used for initial plane-normal error:
candidate geometry is constructed around an assumed/estimated plane while
measurements are generated from the true plane.

The GT hand-eye and true plane are NEVER part of the RL observation. They are
used only by the simulator to generate measurements and training rewards.

Training randomization
----------------------
Per episode:
    - true hand-eye
    - initial hand-eye error
    - true plane normal
    - initial plane-normal error
    - initial plane offset error
    - measurement noise (optional interval)
    - candidate-domain scale (optional)

Execution availability
----------------------
Training can independently hide a fraction of otherwise feasible candidates.
This is NOT a physical IK model; it is ranking augmentation so the agent learns
that rank-1 may not be executable. A real deterministic IK/collision callback
can be plugged in later and is combined with this mask.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robust_laser_handeye.laser_handeye.active_fisher import (
    JointCalibrationEstimate,
    fisher_objective_value,
    predicted_candidate_information,
)
from robust_laser_handeye.laser_handeye.simulation import sample_random_handeye
from robust_laser_handeye.well_made_experiment import exp1_optimal as exp1
from robust_laser_handeye.well_made_experiment import (
    exp1_bootstrap_fisher_monte_carlo as base,
)


# =============================================================================
# Constants / utilities
# =============================================================================

EPS = 1e-10
ACTION_DIM = 10  # u,v,d + sin/cos(alpha, tilt, beta) + stop bit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("cannot normalize zero vector")
    return v / n


def upper_triangle(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    idx = np.triu_indices(M.shape[0])
    return M[idx]


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Dependency-free SO(3) logarithm, output in radians."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        )

    if abs(math.pi - theta) < 1e-5:
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis[0] = math.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = math.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = math.copysign(axis[2], R[1, 0] - R[0, 1])
        return normalize(axis) * theta

    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def random_tangent_axis(
    rng: np.random.Generator,
    normal: np.ndarray,
) -> np.ndarray:
    n = normalize(normal)
    for _ in range(100):
        a = rng.normal(size=3)
        a = a - float(a @ n) * n
        if np.linalg.norm(a) > 1e-8:
            return normalize(a)
    raise RuntimeError("failed to sample tangent axis")


def rotate_axis_angle(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    a = normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * v + s * np.cross(a, v) + (1.0 - c) * float(a @ v) * a


def perturb_normal(
    rng: np.random.Generator,
    normal: np.ndarray,
    max_error_deg: float,
) -> np.ndarray:
    if max_error_deg <= 0:
        return normalize(normal)
    axis = random_tangent_axis(rng, normal)
    angle = math.radians(float(rng.uniform(0.0, max_error_deg)))
    return normalize(rotate_axis_angle(normal, axis, angle))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def plane_offset_from_frame(frame) -> float:
    return float(frame.offset_mm)


def point_on_plane_near_reference(
    normal: np.ndarray,
    offset_mm: float,
    reference: np.ndarray,
) -> np.ndarray:
    """Project a known approximate board center onto n^T x = offset."""
    n = normalize(normal)
    r = np.asarray(reference, dtype=float).reshape(3)
    return r + (float(offset_mm) - float(n @ r)) * n


# =============================================================================
# Plane extraction from JointCalibrationEstimate
# =============================================================================

def _extract_plane_object(estimate: JointCalibrationEstimate):
    """
    Return one PlaneEstimate-like object from estimate.planes without trying
    to cast it to numpy.

    Common containers supported:
        dict[int, PlaneEstimate]
        list[PlaneEstimate]
        tuple[PlaneEstimate, ...]
        ndarray(dtype=object)
        PlaneEstimate directly
    """
    planes = estimate.planes

    if isinstance(planes, dict):
        if 0 in planes:
            return planes[0]
        if len(planes) == 1:
            return next(iter(planes.values()))
        raise KeyError(f"plane id 0 not found; keys={list(planes.keys())}")

    if isinstance(planes, (list, tuple)):
        if not planes:
            raise ValueError("estimate.planes is empty")
        return planes[0]

    if isinstance(planes, np.ndarray):
        if planes.size == 0:
            raise ValueError("estimate.planes ndarray is empty")
        if planes.dtype == object:
            return planes.reshape(-1)[0]

    return planes


def _public_fields(obj) -> dict[str, object]:
    """Best-effort field dictionary for dataclass / normal Python objects."""
    fields: dict[str, object] = {}

    if hasattr(obj, "__dict__"):
        fields.update(vars(obj))

    dataclass_fields = getattr(obj, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        for name in dataclass_fields:
            if name not in fields:
                try:
                    fields[name] = getattr(obj, name)
                except Exception:
                    pass

    # Some classes use slots.
    for name in getattr(obj, "__slots__", ()):
        if name not in fields:
            try:
                fields[name] = getattr(obj, name)
            except Exception:
                pass

    return {
        str(k): v
        for k, v in fields.items()
        if not str(k).startswith("_")
    }


def _as_scalar(value) -> float | None:
    """Convert only genuinely scalar numeric values."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        return x if np.isfinite(x) else None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.dtype == object or arr.size != 1:
        return None

    try:
        x = float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _as_vec3(value) -> np.ndarray | None:
    """Convert a numeric 3-vector only; never cast arbitrary objects."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.dtype == object or arr.size != 3:
        return None

    try:
        arr = np.asarray(arr, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None

    if not np.all(np.isfinite(arr)):
        return None

    norm = float(np.linalg.norm(arr))
    if norm <= EPS:
        return None
    return arr / norm


def extract_plane_normal_offset(
    estimate: JointCalibrationEstimate,
) -> tuple[np.ndarray, float]:
    """
    Extract estimated plane geometry robustly from PlaneEstimate.

    We deliberately avoid np.asarray(PlaneEstimate, dtype=float), which was the
    source of the previous crash.

    Priority:
      1) known normal/offset field names
      2) field-name heuristics
      3) unique numeric vec3 + unique numeric scalar fallback

    If the concrete PlaneEstimate schema is still unsupported, the error
    reports the object's type and public fields so adapting this function is
    trivial.
    """
    p = _extract_plane_object(estimate)
    fields = _public_fields(p)

    # ------------------------------------------------------------
    # 1. Explicit common field names.
    # ------------------------------------------------------------
    normal_names = (
        "normal",
        "n",
        "normal_base",
        "normal_b",
        "plane_normal",
        "normal_vector",
    )
    offset_names = (
        "offset_mm",
        "offset",
        "d_mm",
        "distance_mm",
        "plane_offset_mm",
        "plane_offset",
        "rho_mm",
        "rho",
    )

    normal = None
    normal_field = None
    for name in normal_names:
        if hasattr(p, name):
            candidate = _as_vec3(getattr(p, name))
            if candidate is not None:
                normal = candidate
                normal_field = name
                break
        if name in fields:
            candidate = _as_vec3(fields[name])
            if candidate is not None:
                normal = candidate
                normal_field = name
                break

    offset = None
    offset_field = None
    for name in offset_names:
        if hasattr(p, name):
            candidate = _as_scalar(getattr(p, name))
            if candidate is not None:
                offset = candidate
                offset_field = name
                break
        if name in fields:
            candidate = _as_scalar(fields[name])
            if candidate is not None:
                offset = candidate
                offset_field = name
                break

    # ------------------------------------------------------------
    # 2. Heuristics based on field names.
    # ------------------------------------------------------------
    if normal is None:
        for name, value in fields.items():
            lname = name.lower()
            if "normal" in lname or lname in {"n", "nhat", "n_hat"}:
                candidate = _as_vec3(value)
                if candidate is not None:
                    normal = candidate
                    normal_field = name
                    break

    if offset is None:
        for name, value in fields.items():
            lname = name.lower()
            if (
                "offset" in lname
                or "distance" in lname
                or lname in {"d", "rho"}
            ):
                candidate = _as_scalar(value)
                if candidate is not None:
                    offset = candidate
                    offset_field = name
                    break

    # ------------------------------------------------------------
    # 3. Conservative structural fallback.
    # ------------------------------------------------------------
    if normal is None:
        vec3_candidates = []
        for name, value in fields.items():
            candidate = _as_vec3(value)
            if candidate is not None:
                vec3_candidates.append((name, candidate))
        if len(vec3_candidates) == 1:
            normal_field, normal = vec3_candidates[0]

    if offset is None:
        scalar_candidates = []
        for name, value in fields.items():
            candidate = _as_scalar(value)
            if candidate is not None:
                scalar_candidates.append((name, candidate))

        # Prefer a scalar with length-unit-looking name if possible.
        mm_like = [
            item for item in scalar_candidates
            if "mm" in item[0].lower()
        ]
        if len(mm_like) == 1:
            offset_field, offset = mm_like[0]
        elif len(scalar_candidates) == 1:
            offset_field, offset = scalar_candidates[0]

    if normal is not None and offset is not None:
        return normalize(normal), float(offset)

    # Helpful diagnostic rather than another opaque float() exception.
    field_summary = {
        name: {
            "type": type(value).__name__,
            "shape": getattr(value, "shape", None),
            "repr": repr(value)[:160],
        }
        for name, value in fields.items()
    }

    raise TypeError(
        "Unsupported PlaneEstimate representation. "
        f"type={type(p).__module__}.{type(p).__name__}; "
        f"normal_field={normal_field}; offset_field={offset_field}; "
        f"fields={field_summary}"
    )


def estimated_plane_frame(estimate: JointCalibrationEstimate):
    """
    Build the plane frame used for candidate generation from the CURRENT
    estimated plane.

    Candidate poses must be generated around the estimate, not the GT plane.
    The true plane is used only when simulating the actually observed profile.
    """
    normal, offset_mm = extract_plane_normal_offset(estimate)

    # Keep the frame origin near the nominal board center while enforcing
    # n^T x = offset. This avoids arbitrary far-away origins on the same plane.
    board_center_est = point_on_plane_near_reference(
        normal,
        offset_mm,
        exp1.BOARD_CENTER,
    )

    return exp1.make_plane_frame(
        normal=normal,
        board_center=board_center_est,
    )


# =============================================================================
# RL candidate
# =============================================================================

@dataclass(frozen=True)
class RLAction:
    """
    Six physical pose parameters requested by the user.

    u_mm, v_mm:
        z-axis/plane intersection coordinates in the estimated plane frame.
    axis_azimuth_deg:
        azimuth of sensor +z projected onto the plane.
    view_tilt_deg:
        angle between sensor -z and plane normal.
    view_azimuth_deg:
        azimuth of the plane normal projected into sensor x-y.
    view_distance_mm:
        distance from sensor origin to the z-axis/plane intersection.

    pose:
        base.CandidatePose carrying the robot command and ACTUAL sensor pose
        for simulation.
    """
    u_mm: float
    v_mm: float
    axis_azimuth_deg: float
    view_tilt_deg: float
    view_azimuth_deg: float
    view_distance_mm: float
    pose: Optional[base.CandidatePose]
    is_stop: bool = False


@dataclass(frozen=True)
class ActionScales:
    uv_mm: float
    distance_mm: float


def encode_action(action: RLAction, scales: ActionScales) -> np.ndarray:
    if action.is_stop:
        return np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.float32)

    alpha = math.radians(action.axis_azimuth_deg)
    tilt = math.radians(action.view_tilt_deg)
    beta = math.radians(action.view_azimuth_deg)

    return np.array(
        [
            action.u_mm / scales.uv_mm,
            action.v_mm / scales.uv_mm,
            action.view_distance_mm / scales.distance_mm,
            math.sin(alpha),
            math.cos(alpha),
            math.sin(tilt),
            math.cos(tilt),
            math.sin(beta),
            math.cos(beta),
            0.0,
        ],
        dtype=np.float32,
    )


# =============================================================================
# Deterministic robot feasibility hook
# =============================================================================

RobotFeasibilityFn = Callable[[np.ndarray], bool]


def load_robot_feasibility_callback(
    module_path: Optional[Path],
) -> Optional[RobotFeasibilityFn]:
    """
    Optional external Python file:

        def is_pose_feasible(T_base_ef: np.ndarray) -> bool:
            # MoveIt / IK / collision / joint limit check
            return ...

    No path -> no fake/random IK filtering.
    """
    if module_path is None:
        return None

    module_path = Path(module_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("rl_robot_feasibility", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load feasibility module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "is_pose_feasible", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{module_path} must define callable "
            "is_pose_feasible(T_base_ef) -> bool"
        )
    return fn


# =============================================================================
# Candidate generation from CURRENT estimate
# =============================================================================

@dataclass
class CandidateDomain:
    uv_half_range_mm: float
    tilt_range_deg: tuple[float, float]
    distance_range_mm: tuple[float, float]
    axis_azimuth_range_deg: tuple[float, float]
    view_azimuth_range_deg: tuple[float, float]


def sample_action_candidate(
    *,
    candidate_id: int,
    estimated_frame,
    true_frame,
    T_ef_s_est: np.ndarray,
    T_ef_s_true: np.ndarray,
    rng: np.random.Generator,
    domain: CandidateDomain,
    depth_range_mm: tuple[float, float],
    max_attempts: int = 5000,
) -> RLAction:
    """
    Generate candidate geometry around the ESTIMATED plane.

    Causality:
      desired sensor pose -> current hand-eye estimate -> robot command
      robot command -> true hand-eye -> actual physical sensor pose
    """
    for _ in range(max_attempts):
        u = float(rng.uniform(-domain.uv_half_range_mm, domain.uv_half_range_mm))
        v = float(rng.uniform(-domain.uv_half_range_mm, domain.uv_half_range_mm))
        alpha = float(rng.uniform(*domain.axis_azimuth_range_deg))
        tilt = float(rng.uniform(*domain.tilt_range_deg))
        beta = float(rng.uniform(*domain.view_azimuth_range_deg))
        distance = float(rng.uniform(*domain.distance_range_mm))

        target_point = exp1.plane_uv_to_base_points(
            uv=np.asarray([[u, v]], dtype=float),
            target_center_base_mm=np.asarray(estimated_frame.center, dtype=float)
            if hasattr(estimated_frame, "center")
            else point_on_plane_near_reference(
                estimated_frame.n,
                estimated_frame.offset_mm,
                exp1.BOARD_CENTER,
            ),
            frame=estimated_frame,
        )[0]

        try:
            T_base_s_desired = exp1.sensor_pose_from_target_point(
                target_point_base_mm=target_point,
                frame=estimated_frame,
                distance_mm=distance,
                tilt_deg=tilt,
                azimuth_deg=alpha,
                normal_azimuth_sensor_deg=beta,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

        # Predicted sensor feasibility uses only CURRENT estimate.
        if not base.profile_is_feasible(
            estimated_frame,
            T_base_s_desired,
            depth_range_mm,
        ):
            continue

        # Command robot using current hand-eye estimate.
        T_base_ef_cmd = (
            np.asarray(T_base_s_desired, dtype=float).reshape(4, 4)
            @ np.linalg.inv(np.asarray(T_ef_s_est, dtype=float).reshape(4, 4))
        )

        # Actual sensor pose is determined by GT hand-eye in simulation.
        T_base_s_actual = (
            T_base_ef_cmd
            @ np.asarray(T_ef_s_true, dtype=float).reshape(4, 4)
        )

        pose = base.CandidatePose(
            candidate_id=int(candidate_id),
            target_u_mm=u,
            target_v_mm=v,
            # The legacy field is scan-line azimuth. It is not one of the six
            # RL action variables in this experiment, so keep it as NaN.
            psi_deg=float("nan"),
            tilt_deg=tilt,
            distance_mm=distance,
            beta_deg=beta,
            alpha_deg=alpha,
            T_base_s=T_base_s_actual,
            T_base_ef=T_base_ef_cmd,
        )

        return RLAction(
            u_mm=u,
            v_mm=v,
            axis_azimuth_deg=alpha,
            view_tilt_deg=tilt,
            view_azimuth_deg=beta,
            view_distance_mm=distance,
            pose=pose,
            is_stop=False,
        )

    raise RuntimeError("could not generate a predicted-visible RL candidate")


# =============================================================================
# Bootstrap generated from an intentionally imperfect initial estimate
# =============================================================================

def build_bootstrap_scans_from_assumed_state(
    *,
    assumed_frame,
    true_frame,
    T_ef_s_initial: np.ndarray,
    T_ef_s_true: np.ndarray,
    noise_std_mm: float,
    noise_axis: str,
    noise_seeds: np.ndarray,
    depth_range_mm: tuple[float, float],
) -> list:
    """
    Reuses the OPT12 geometric pattern, but constructs it around an ASSUMED plane
    and converts desired sensor poses to robot commands with the INITIAL hand-eye
    estimate. Thus both initial errors affect the physical observations.
    """
    uv = exp1.circular_uv_points(radius_mm=exp1.RADIUS_MM, N=exp1.N)
    target_points = exp1.plane_uv_to_base_points(
        uv=uv,
        target_center_base_mm=point_on_plane_near_reference(
            assumed_frame.n,
            assumed_frame.offset_mm,
            exp1.BOARD_CENTER,
        ),
        frame=assumed_frame,
    )

    support_ids = exp1.make_support_ids()
    beta = exp1.make_optimal_beta(support_ids)
    params = exp1.make_scan_pose_parameters(uv=uv, beta_deg=beta)

    desired_sensor_poses = exp1.make_sensor_poses(
        pose_params=params,
        target_points_base=target_points,
        frame=assumed_frame,
    )

    scans = []
    for idx, (param, T_desired, noise_seed) in enumerate(
        zip(params, desired_sensor_poses, noise_seeds)
    ):
        T_base_ef_cmd = (
            np.asarray(T_desired, dtype=float).reshape(4, 4)
            @ np.linalg.inv(np.asarray(T_ef_s_initial, dtype=float).reshape(4, 4))
        )
        T_actual = (
            T_base_ef_cmd
            @ np.asarray(T_ef_s_true, dtype=float).reshape(4, 4)
        )

        pose = base.CandidatePose(
            candidate_id=idx,
            target_u_mm=float(param.target_u_mm),
            target_v_mm=float(param.target_v_mm),
            psi_deg=float(param.scanline_azimuth_deg),
            tilt_deg=float(param.tilt_deg),
            distance_mm=float(param.distance_mm),
            beta_deg=float(param.normal_azimuth_sensor_deg),
            alpha_deg=float(param.azimuth_deg),
            T_base_s=T_actual,
            T_base_ef=T_base_ef_cmd,
        )

        # An initial wrong estimate can make a planned bootstrap observation miss
        # the sensor depth window. Skip impossible scans instead of fabricating data.
        if not base.profile_is_feasible(true_frame, T_actual, depth_range_mm):
            continue

        scans.append(
            base.noisy_scan_from_pose(
                pose,
                true_frame,
                noise_std_mm=noise_std_mm,
                noise_axis=noise_axis,
                noise_seed=int(noise_seed),
                scan_id=idx,
                source="RL_BOOTSTRAP",
            )
        )

    return scans



def supplement_bootstrap_scans(
    *,
    scans: list,
    target_count: int,
    assumed_frame,
    true_frame,
    T_ef_s_initial: np.ndarray,
    T_ef_s_true: np.ndarray,
    rng_candidate: np.random.Generator,
    rng_noise: np.random.Generator,
    domain: CandidateDomain,
    depth_range_mm: tuple[float, float],
    noise_std_mm: float,
    noise_axis: str,
    max_attempts: int,
) -> list:
    """
    Fill missing bootstrap scans by resampling predicted-visible candidates.

    This is intentionally NOT RL.
    It is a robust acquisition bootstrap:
        current initial estimate -> propose pose
        -> actual GT sensor pose
        -> if physically visible, acquire it
        -> otherwise discard and try another pose.

    This prevents an episode from failing merely because several of the
    fixed OPT12 bootstrap poses miss the physical sensor depth window under
    a large initial hand-eye / plane-normal error.
    """
    scans = list(scans)
    attempts = 0
    candidate_id = 1_000_000

    while len(scans) < target_count and attempts < max_attempts:
        attempts += 1

        try:
            action = sample_action_candidate(
                candidate_id=candidate_id,
                estimated_frame=assumed_frame,
                true_frame=true_frame,
                T_ef_s_est=T_ef_s_initial,
                T_ef_s_true=T_ef_s_true,
                rng=rng_candidate,
                domain=domain,
                depth_range_mm=depth_range_mm,
            )
        except RuntimeError:
            continue

        candidate_id += 1

        if action.pose is None:
            continue

        # Predicted-visible does not guarantee TRUE visibility when the initial
        # hand-eye / plane estimate is wrong. Check physical visibility here.
        if not base.profile_is_feasible(
            true_frame,
            action.pose.T_base_s,
            depth_range_mm,
        ):
            continue

        noise_seed = int(
            rng_noise.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )

        scan = base.noisy_scan_from_pose(
            action.pose,
            true_frame,
            noise_std_mm=noise_std_mm,
            noise_axis=noise_axis,
            noise_seed=noise_seed,
            scan_id=50_000 + len(scans),
            source="RL_BOOTSTRAP_RESAMPLED",
        )
        scans.append(scan)

    return scans

# =============================================================================
# State encoding
# =============================================================================

def make_state_vector(
    *,
    estimate: JointCalibrationEstimate,
    last_action_feature: np.ndarray,
    acquired_scan_count: int,
    max_scans: int,
    translation_scale_mm: float,
) -> np.ndarray:
    """
    Observation contains only estimated quantities.

    - hand-eye estimate: rotvec + translation
    - estimated plane normal + offset
    - shape + scale of current joint information matrix
    - previous action
    - normalized scan count
    """
    T = np.asarray(estimate.T_ef_s, dtype=float).reshape(4, 4)
    rotvec = rotation_matrix_to_rotvec(T[:3, :3]) / math.pi
    trans = T[:3, 3] / float(translation_scale_mm)

    n, offset = extract_plane_normal_offset(estimate)

    H = np.asarray(estimate.information, dtype=float)
    H = 0.5 * (H + H.T)
    trace = max(float(np.trace(H)), 0.0)
    H_shape = H / (trace + 1e-9)

    info_ut = upper_triangle(H_shape)
    info_scale = math.log1p(trace)

    step_fraction = float(acquired_scan_count) / max(float(max_scans), 1.0)

    return np.concatenate(
        [
            rotvec,
            trans,
            n,
            np.array([offset / 1000.0], dtype=float),
            info_ut,
            np.array([info_scale], dtype=float),
            np.asarray(last_action_feature, dtype=float),
            np.array([step_fraction], dtype=float),
        ]
    ).astype(np.float32)


# =============================================================================
# Environment
# =============================================================================

class AdaptiveHandEyeEnv:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        seed: int,
        robot_feasibility_fn: Optional[RobotFeasibilityFn] = None,
    ):
        self.args = args
        self.seed = int(seed)
        self.robot_feasibility_fn = robot_feasibility_fn
        self.stop_action = RLAction(0, 0, 0, 0, 0, 0, pose=None, is_stop=True)
        self.action_scales = ActionScales(
            uv_mm=max(float(args.uv_half_range_mm), 1.0),
            distance_mm=max(float(args.distance_range_mm[1]), 1.0),
        )
        self.episode_index = 0

    def _episode_rngs(self):
        root = np.random.SeedSequence([self.seed, self.episode_index])
        return [np.random.default_rng(x) for x in root.spawn(8)]

    def _sample_true_frame(self, rng: np.random.Generator):
        nominal_n = normalize(exp1.PLANE_NORMAL)
        n_true = perturb_normal(
            rng,
            nominal_n,
            self.args.true_plane_normal_random_deg,
        )
        center = np.asarray(exp1.BOARD_CENTER, dtype=float).copy()
        if self.args.true_plane_center_jitter_mm > 0:
            center += rng.normal(
                0.0,
                self.args.true_plane_center_jitter_mm,
                size=3,
            )
        return exp1.make_plane_frame(normal=n_true, board_center=center)

    def _sample_initial_assumed_frame(
        self,
        rng: np.random.Generator,
        true_frame,
    ):
        n0 = perturb_normal(
            rng,
            true_frame.n,
            self.args.init_plane_normal_error_deg,
        )
        offset0 = float(true_frame.offset_mm) + float(
            rng.uniform(
                -self.args.init_plane_offset_error_mm,
                self.args.init_plane_offset_error_mm,
            )
        )
        center0 = point_on_plane_near_reference(n0, offset0, exp1.BOARD_CENTER)
        return exp1.make_plane_frame(normal=n0, board_center=center0)

    def _sample_domain(self, rng: np.random.Generator) -> CandidateDomain:
        scale = 1.0
        if self.args.domain_randomization_fraction > 0:
            f = self.args.domain_randomization_fraction
            scale = float(rng.uniform(1.0 - f, 1.0 + f))

        tilt_lo, tilt_hi = self.args.tilt_range_deg
        dist_lo, dist_hi = self.args.distance_range_mm

        return CandidateDomain(
            uv_half_range_mm=max(5.0, self.args.uv_half_range_mm * scale),
            tilt_range_deg=(tilt_lo, max(tilt_lo + 1e-3, tilt_hi * scale)),
            distance_range_mm=(
                max(1.0, dist_lo * scale),
                max(dist_lo * scale + 1e-3, dist_hi * scale),
            ),
            axis_azimuth_range_deg=(0.0, 360.0),
            view_azimuth_range_deg=self.args.beta_range_deg,
        )

    def reset(self) -> np.ndarray:
        (
            rng_gt,
            rng_init,
            rng_plane,
            rng_plane_init,
            rng_noise,
            rng_domain,
            rng_candidate,
            rng_misc,
        ) = self._episode_rngs()
        self.episode_index += 1

        self.rng_candidate = rng_candidate
        self.rng_noise = rng_noise
        self.rng_misc = rng_misc
        # Dedicated RNG for training/test candidate availability augmentation.
        # This is NOT an IK model. It only teaches the ranker that its top
        # candidate may be unavailable at execution time.
        self.rng_availability = rng_misc

        self.T_true, _, _ = sample_random_handeye(
            rng=rng_gt,
            trans_range_mm=base.GT_TRANSLATION_COMPONENT_RANGE_MM,
            angle_range_deg=base.GT_EULER_COMPONENT_RANGE_DEG,
        )

        self.T_initial = exp1.make_initial_guess_GT(
            self.T_true,
            rng=rng_init,
            max_rotation_error_deg=self.args.init_rotation_deg,
            max_translation_error_mm=self.args.init_translation_mm,
        )

        self.true_frame = self._sample_true_frame(rng_plane)
        self.assumed_frame0 = self._sample_initial_assumed_frame(
            rng_plane_init,
            self.true_frame,
        )
        self.domain = self._sample_domain(rng_domain)

        if self.args.measurement_noise_std_random_mm > 0:
            lo = max(
                1e-6,
                self.args.measurement_noise_std_mm
                - self.args.measurement_noise_std_random_mm,
            )
            hi = (
                self.args.measurement_noise_std_mm
                + self.args.measurement_noise_std_random_mm
            )
            self.noise_std_mm = float(rng_noise.uniform(lo, hi))
        else:
            self.noise_std_mm = float(self.args.measurement_noise_std_mm)

        self.estimator_args = argparse.Namespace(**vars(self.args))
        self.estimator_args.measurement_noise_std_mm = self.noise_std_mm
        self.scales = base.parameter_scales(self.estimator_args)

        bootstrap_noise_seeds = rng_noise.integers(
            0,
            np.iinfo(np.uint32).max,
            size=exp1.N,
            dtype=np.uint32,
        )
        bootstrap_scans = build_bootstrap_scans_from_assumed_state(
            assumed_frame=self.assumed_frame0,
            true_frame=self.true_frame,
            T_ef_s_initial=self.T_initial,
            T_ef_s_true=self.T_true,
            noise_std_mm=self.noise_std_mm,
            noise_axis=self.args.noise_axis,
            noise_seeds=bootstrap_noise_seeds,
            depth_range_mm=self.args.profile_depth_range_mm,
        )

        initial_valid_bootstrap = len(bootstrap_scans)

        # Robust bootstrap:
        # if some fixed OPT12 scans miss because the current hand-eye / plane
        # estimate is wrong, resample alternative predicted-visible views.
        if len(bootstrap_scans) < self.args.bootstrap_target_scans:
            bootstrap_scans = supplement_bootstrap_scans(
                scans=bootstrap_scans,
                target_count=self.args.bootstrap_target_scans,
                assumed_frame=self.assumed_frame0,
                true_frame=self.true_frame,
                T_ef_s_initial=self.T_initial,
                T_ef_s_true=self.T_true,
                rng_candidate=self.rng_candidate,
                rng_noise=self.rng_noise,
                domain=self.domain,
                depth_range_mm=self.args.profile_depth_range_mm,
                noise_std_mm=self.noise_std_mm,
                noise_axis=self.args.noise_axis,
                max_attempts=self.args.bootstrap_resample_attempts,
            )

        if len(bootstrap_scans) < self.args.min_bootstrap_scans:
            raise RuntimeError(
                f"bootstrap acquisition failed: fixed OPT12 gave "
                f"{initial_valid_bootstrap} valid scans and resampling reached only "
                f"{len(bootstrap_scans)}; need >= {self.args.min_bootstrap_scans}. "
                f"Increase --bootstrap-resample-attempts or inspect the depth/domain ranges."
            )

        self.acquired = list(bootstrap_scans)
        self.estimate = base.estimate_state(
            self.acquired,
            self.T_initial,
            self.scales,
            self.estimator_args,
            previous=None,
        )

        self.last_action_feature = np.zeros(ACTION_DIM, dtype=np.float32)
        self.decision_count = 0
        self.failed_measurements = 0

        return self.state()

    def state(self) -> np.ndarray:
        return make_state_vector(
            estimate=self.estimate,
            last_action_feature=self.last_action_feature,
            acquired_scan_count=len(self.acquired),
            max_scans=self.args.max_total_scans,
            translation_scale_mm=self.args.state_translation_scale_mm,
        )

    def get_candidates(self) -> list[RLAction]:
        est_frame = estimated_plane_frame(self.estimate)
        actions: list[RLAction] = []

        for candidate_id in range(self.args.candidate_pool_size):
            try:
                action = sample_action_candidate(
                    candidate_id=candidate_id,
                    estimated_frame=est_frame,
                    true_frame=self.true_frame,
                    T_ef_s_est=self.estimate.T_ef_s,
                    T_ef_s_true=self.T_true,
                    rng=self.rng_candidate,
                    domain=self.domain,
                    depth_range_mm=self.args.profile_depth_range_mm,
                )
                actions.append(action)
            except RuntimeError:
                continue

        if not actions:
            raise RuntimeError("candidate generator returned no predicted-visible pose")

        if self.decision_count >= self.args.min_rl_decisions_before_stop:
            actions.append(self.stop_action)

        return actions

    def candidate_features(self, actions: Sequence[RLAction]) -> np.ndarray:
        return np.stack(
            [encode_action(a, self.action_scales) for a in actions]
        ).astype(np.float32)

    def robot_feasible(self, action: RLAction) -> bool:
        if action.is_stop:
            return True
        if action.pose is None:
            return False
        if self.robot_feasibility_fn is None:
            # IMPORTANT: no random fake mask.
            return True
        try:
            return bool(
                self.robot_feasibility_fn(
                    np.asarray(action.pose.T_base_ef, dtype=float).reshape(4, 4)
                )
            )
        except Exception:
            return False

    def deterministic_feasibility_mask(
        self,
        actions: Sequence[RLAction],
    ) -> np.ndarray:
        """
        Actual deterministic execution constraints.

        - If --robot-feasibility-module is absent: every non-invalid candidate
          is treated as robot-feasible.
        - Later, MoveIt IK / collision / joint-limit checks can be plugged in
          through the callback without changing the RL code.
        """
        return np.asarray([self.robot_feasible(a) for a in actions], dtype=bool)

    def availability_mask(
        self,
        actions: Sequence[RLAction],
        *,
        training: bool,
    ) -> np.ndarray:
        """
        Final execution mask used by the ranker.

        mask = deterministic robot feasibility
               AND
               synthetic candidate availability augmentation

        The synthetic part is deliberately independent of Q and hidden from
        the policy. It is NOT pretending to be a physical IK model. Its role is
        to ensure that training experiences:
            rank-1 unavailable -> rank-2/rank-3/... executed.

        At deployment set the synthetic probability to zero and optionally
        supply a real robot-feasibility callback.
        """
        mask = self.deterministic_feasibility_mask(actions)
        if len(actions) == 0:
            return mask

        if training:
            lo, hi = self.args.train_unavailable_prob_range
            p_unavailable = float(self.rng_availability.uniform(lo, hi))
        else:
            p_unavailable = float(self.args.test_unavailable_prob)

        p_unavailable = float(np.clip(p_unavailable, 0.0, 1.0))

        # STOP is never removed by synthetic availability augmentation.
        non_stop_ids = [
            i for i, action in enumerate(actions)
            if (not action.is_stop) and mask[i]
        ]

        if p_unavailable > 0.0 and non_stop_ids:
            drop = self.rng_availability.random(len(non_stop_ids)) < p_unavailable
            for idx, should_drop in zip(non_stop_ids, drop):
                if should_drop:
                    mask[idx] = False

            # Avoid creating an artificial state with zero executable motion.
            # Keep one deterministic-feasible motion available if all were
            # synthetically removed.
            surviving_motion = any(
                mask[i] and (not actions[i].is_stop)
                for i in range(len(actions))
            )
            if not surviving_motion:
                restore = int(self.rng_availability.choice(non_stop_ids))
                mask[restore] = True

        return mask

    # Backward-compatible alias: deterministic feasibility only.
    def feasibility_mask(self, actions: Sequence[RLAction]) -> np.ndarray:
        return self.deterministic_feasibility_mask(actions)

    def _terminal_reward(self) -> tuple[float, dict]:
        t_err, r_err = base.transform_errors(self.estimate.T_ef_s, self.T_true)
        reward = -(
            t_err / self.args.reward_translation_scale_mm
            + r_err / self.args.reward_rotation_scale_deg
        )
        info = {
            "translation_error_mm": float(t_err),
            "rotation_error_deg": float(r_err),
            "d_optimal": float(base.d_optimal_value(self.estimate)),
            "scan_count": len(self.acquired),
            "decision_count": self.decision_count,
            "measurement_failures": self.failed_measurements,
            "true_vs_est_plane_normal_deg": angle_deg(
                self.true_frame.n,
                extract_plane_normal_offset(self.estimate)[0],
            ),
        }
        return float(reward), info

    def step(self, action: RLAction):
        self.decision_count += 1

        if action.is_stop:
            terminal, info = self._terminal_reward()
            info["termination_reason"] = "stop"
            return self.state(), terminal, True, info

        if action.pose is None:
            raise ValueError("non-stop action has no pose")

        # The executor should call this only after feasibility filtering.
        if not self.robot_feasible(action):
            raise RuntimeError("attempted to execute an infeasible robot action")

        old_dopt = float(base.d_optimal_value(self.estimate))

        # Actual sensor visibility is checked against the TRUE plane AFTER selection.
        if not base.profile_is_feasible(
            self.true_frame,
            action.pose.T_base_s,
            self.args.profile_depth_range_mm,
        ):
            self.failed_measurements += 1
            self.last_action_feature = encode_action(action, self.action_scales)

            reward = -float(self.args.failed_measurement_penalty)
            done = self.decision_count >= self.args.max_rl_decisions

            info = {
                "measurement_failed": True,
                "reason": "actual profile outside depth range",
                "scan_count": len(self.acquired),
                "decision_count": self.decision_count,
            }
            if done:
                terminal, final_info = self._terminal_reward()
                reward += terminal
                info.update(final_info)
                info["termination_reason"] = "max_rl_decisions"

            return self.state(), float(reward), bool(done), info

        noise_seed = int(
            self.rng_noise.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        )
        scan = base.noisy_scan_from_pose(
            action.pose,
            self.true_frame,
            noise_std_mm=self.noise_std_mm,
            noise_axis=self.args.noise_axis,
            noise_seed=noise_seed,
            scan_id=100_000 + self.decision_count,
            source="RL_SELECTED",
        )

        self.acquired.append(scan)
        previous = self.estimate

        try:
            self.estimate = base.estimate_state(
                self.acquired,
                self.T_initial,
                self.scales,
                self.estimator_args,
                previous=previous,
            )
        except Exception:
            # A numerically bad update should be observable by the RL objective.
            self.acquired.pop()
            self.estimate = previous
            self.failed_measurements += 1
            self.last_action_feature = encode_action(action, self.action_scales)

            reward = -float(self.args.estimator_failure_penalty)
            done = self.decision_count >= self.args.max_rl_decisions

            info = {
                "measurement_failed": True,
                "reason": "estimator failure",
                "scan_count": len(self.acquired),
                "decision_count": self.decision_count,
            }
            if done:
                terminal, final_info = self._terminal_reward()
                reward += terminal
                info.update(final_info)
            return self.state(), float(reward), bool(done), info

        new_dopt = float(base.d_optimal_value(self.estimate))
        dopt_gain = new_dopt - old_dopt

        self.last_action_feature = encode_action(action, self.action_scales)

        reward = (
            self.args.info_gain_reward_weight * dopt_gain
            - self.args.pose_cost
        )

        done = (
            self.decision_count >= self.args.max_rl_decisions
            or len(self.acquired) >= self.args.max_total_scans
        )

        info = {
            "measurement_failed": False,
            "dopt_gain": float(dopt_gain),
            "scan_count": len(self.acquired),
            "decision_count": self.decision_count,
        }

        if done:
            terminal, final_info = self._terminal_reward()
            reward += terminal
            info.update(final_info)
            info["termination_reason"] = (
                "max_total_scans"
                if len(self.acquired) >= self.args.max_total_scans
                else "max_rl_decisions"
            )

        return self.state(), float(reward), bool(done), info


# =============================================================================
# Q network / replay
# =============================================================================

class ActionConditionalQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_DIM,
        hidden: Sequence[int] = (256, 256, 128),
    ):
        super().__init__()
        dims = [state_dim + action_dim, *hidden, 1]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(a, b), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


@dataclass
class Transition:
    state: np.ndarray
    action_feature: np.ndarray
    reward: float
    next_state: np.ndarray
    next_actions: np.ndarray
    next_feasible: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def add(self, x: Transition) -> None:
        self.data.append(x)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.data, n)


class DDQNRanker:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        device: Optional[str],
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.online = ActionConditionalQ(state_dim, action_dim).to(self.device)
        self.target = ActionConditionalQ(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.AdamW(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    @torch.no_grad()
    def scores(self, state: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        sb = s.unsqueeze(0).expand(a.shape[0], -1)
        return self.online(sb, a).cpu().numpy()

    def rank(self, state: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        return np.argsort(-self.scores(state, action_features))

    def select_executable(
        self,
        *,
        state: np.ndarray,
        action_features: np.ndarray,
        feasible: np.ndarray,
        epsilon: float,
    ) -> tuple[int, np.ndarray]:
        """
        Normal inference:
            score ALL -> rank ALL -> take first deterministic feasible.

        Epsilon exploration:
            occasionally execute a random feasible action for Q-learning.
        """
        q = self.scores(state, action_features)
        feasible_ids = np.flatnonzero(feasible)
        if feasible_ids.size == 0:
            raise RuntimeError("no robot-feasible action")

        if random.random() < epsilon:
            return int(np.random.choice(feasible_ids)), q

        for idx in np.argsort(-q):
            if feasible[int(idx)]:
                return int(idx), q

        raise RuntimeError("ranking had no feasible action")

    def update(self, replay: ReplayBuffer, batch_size: int) -> Optional[float]:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)

        S = torch.as_tensor(
            np.stack([x.state for x in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        A = torch.as_tensor(
            np.stack([x.action_feature for x in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        q_pred = self.online(S, A)

        targets = []
        with torch.no_grad():
            for tr in batch:
                if tr.done or tr.next_actions.shape[0] == 0:
                    targets.append(float(tr.reward))
                    continue

                mask_np = np.asarray(tr.next_feasible, dtype=bool)
                if not mask_np.any():
                    targets.append(float(tr.reward))
                    continue

                ns = torch.as_tensor(
                    tr.next_state,
                    dtype=torch.float32,
                    device=self.device,
                )
                na = torch.as_tensor(
                    tr.next_actions,
                    dtype=torch.float32,
                    device=self.device,
                )
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)

                ns_batch = ns.unsqueeze(0).expand(na.shape[0], -1)

                # Double DQN: online selects, target evaluates.
                q_online = self.online(ns_batch, na)
                q_online = q_online.masked_fill(~mask, -torch.inf)
                best = int(torch.argmax(q_online).item())

                q_next = self.target(
                    ns.unsqueeze(0),
                    na[best].unsqueeze(0),
                )[0]

                targets.append(
                    float(tr.reward) + self.gamma * float(q_next.item())
                )

        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        loss = self.loss_fn(q_pred, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_p, online_p in zip(
                self.target.parameters(),
                self.online.parameters(),
            ):
                target_p.mul_(1.0 - self.tau).add_(online_p, alpha=self.tau)

        return float(loss.item())

    def save(
        self,
        path: Path,
        state_dim: int,
        *,
        completed_episodes: int = 0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": int(state_dim),
                "action_dim": ACTION_DIM,
                "completed_episodes": int(completed_episodes),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt


# =============================================================================
# Fisher baseline using SAME candidate bank
# =============================================================================

def fisher_scores_for_actions(
    env: AdaptiveHandEyeEnv,
    actions: Sequence[RLAction],
) -> np.ndarray:
    values = np.full(len(actions), -np.inf, dtype=float)

    for i, action in enumerate(actions):
        if action.is_stop:
            # STOP is not an information-gain action.
            continue
        if action.pose is None:
            continue

        try:
            candidate_information = predicted_candidate_information(
                T_base_ef=action.pose.T_base_ef,
                plane_id=0,
                estimate=env.estimate,
                x_values=exp1.X_VALUES,
                parameter_scales=env.scales,
                profile_noise_std_mm=env.noise_std_mm,
                noise_axis=env.args.noise_axis,
                depth_range_mm=env.args.profile_depth_range_mm,
            )
            if candidate_information is None:
                continue

            values[i] = float(
                fisher_objective_value(
                    env.estimate.information + candidate_information,
                    base.FISHER_OBJECTIVE,
                )
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

    return values


# =============================================================================
# Train / evaluate
# =============================================================================

def epsilon_schedule(
    episode: int,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    x = min(max(episode / max(decay_episodes, 1), 0.0), 1.0)
    return float(start + x * (end - start))


def train(
    *,
    env: AdaptiveHandEyeEnv,
    agent: DDQNRanker,
    args: argparse.Namespace,
    start_episode: int = 0,
) -> ReplayBuffer:
    """
    Train for args.episodes ADDITIONAL episodes.

    When resuming:
      - online / target network weights continue
      - optimizer state continues
      - epsilon schedule continues from the global episode number
      - environment episode_index continues, so randomized episodes are NOT
        repeated from episode 1
      - replay buffer starts fresh (the old checkpoint did not store it)
    """
    replay = ReplayBuffer(args.replay_capacity)
    reward_window = deque(maxlen=100)
    step_window = deque(maxlen=100)
    loss_window = deque(maxlen=500)

    final_episode = int(start_episode) + int(args.episodes)

    for local_episode in range(1, args.episodes + 1):
        episode = int(start_episode) + local_episode
        # Some randomized episodes can be physically impossible at bootstrap.
        # Retry with the next episode seed rather than inventing data.
        for _ in range(args.reset_retry_count):
            try:
                state = env.reset()
                break
            except Exception as exc:
                last_reset_error = exc
        else:
            raise RuntimeError(
                f"environment reset failed repeatedly: {last_reset_error}"
            )

        done = False
        ep_reward = 0.0
        ep_decisions = 0

        # Important: sample a candidate bank ONCE for this state.
        # The same bank is used for action selection and is then carried into
        # the next transition, so replay targets do not refer to a different
        # random bank than the policy actually sees.
        actions = env.get_candidates()
        features = env.candidate_features(actions)
        feasible = env.availability_mask(actions, training=True)

        while not done:
            epsilon = epsilon_schedule(
                episode,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_episodes,
            )

            idx, _ = agent.select_executable(
                state=state,
                action_features=features,
                feasible=feasible,
                epsilon=epsilon,
            )
            chosen = actions[idx]

            next_state, reward, done, _info = env.step(chosen)

            if done:
                next_actions_obj = []
                next_features = np.zeros((0, ACTION_DIM), dtype=np.float32)
                next_feasible = np.zeros((0,), dtype=bool)
            else:
                next_actions_obj = env.get_candidates()
                next_features = env.candidate_features(next_actions_obj)
                next_feasible = env.availability_mask(next_actions_obj, training=True)

            replay.add(
                Transition(
                    state=state.copy(),
                    action_feature=features[idx].copy(),
                    reward=float(reward),
                    next_state=next_state.copy(),
                    next_actions=next_features.copy(),
                    next_feasible=next_feasible.copy(),
                    done=bool(done),
                )
            )

            state = next_state
            ep_reward += float(reward)
            ep_decisions += 1

            if len(replay) >= args.warmup_transitions:
                for _ in range(args.updates_per_step):
                    loss = agent.update(replay, args.batch_size)
                    if loss is not None:
                        loss_window.append(loss)

            # Carry exactly the previously sampled next candidate bank forward.
            if not done:
                actions = next_actions_obj
                features = next_features
                feasible = next_feasible

        reward_window.append(ep_reward)
        step_window.append(ep_decisions)

        if (
            local_episode == 1
            or episode % args.print_every == 0
            or episode == final_episode
        ):
            print(
                f"[train] ep={episode:6d}/{final_episode} "
                f"eps={epsilon_schedule(episode, args.epsilon_start, args.epsilon_end, args.epsilon_decay_episodes):.3f} "
                f"R100={np.mean(reward_window):+.4f} "
                f"decisions100={np.mean(step_window):.2f} "
                f"loss={np.mean(loss_window) if loss_window else float('nan'):.6f} "
                f"buffer={len(replay)}"
            )

    return replay

@torch.no_grad()
def evaluate_rl(
    *,
    env: AdaptiveHandEyeEnv,
    agent: DDQNRanker,
    episodes: int,
    output_dir: Path,
    show_rankings: int = 2,
) -> dict[str, float]:
    """
    Monte-Carlo test.

    Default recommendation:
        500 independent randomized episodes.

    Reports:
        translation median / P95
        rotation median / P95
        success rate under the existing 1 mm / 0.25 deg threshold
        scan-count median / P95
        decision-count median / P95
        measurement failure rate
        top-1 unavailable rate
        fallback-execution rate
        selected rank mean / P95
        STOP termination rate

    Test candidate availability is controlled by:
        --test-unavailable-prob

    Set it to 0 for pure geometry evaluation.
    Later, when a real IK callback is connected, it can also be 0 because
    actual deterministic feasibility already supplies the fallback events.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    completed = 0
    attempts = 0
    max_attempts = max(episodes * 30, 100)

    while completed < episodes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"too many evaluation reset failures: completed={completed}/{episodes}"
            )

        try:
            state = env.reset()
        except Exception:
            continue

        done = False
        final_info: dict[str, object] = {}

        selected_ranks: list[int] = []
        top1_unavailable_count = 0
        fallback_count = 0
        decision_events = 0

        while not done:
            actions = env.get_candidates()
            features = env.candidate_features(actions)

            # IMPORTANT:
            # Q ranks ALL candidates without seeing availability.
            q = agent.scores(state, features)
            order = np.argsort(-q)

            # Actual execution layer: deterministic feasibility + test-time
            # availability augmentation.
            feasible = env.availability_mask(actions, training=False)

            selected = None
            selected_rank = None
            for rank0, idx in enumerate(order):
                if feasible[int(idx)]:
                    selected = int(idx)
                    selected_rank = int(rank0 + 1)
                    break

            if selected is None:
                raise RuntimeError("RL ranking produced no executable candidate")

            decision_events += 1
            selected_ranks.append(selected_rank)

            top_idx = int(order[0])
            if not feasible[top_idx]:
                top1_unavailable_count += 1
            if selected_rank > 1:
                fallback_count += 1

            if completed < show_rankings:
                print(
                    f"\n[MC {completed:04d} | decision {decision_events:02d}] "
                    f"selected rank={selected_rank}"
                )
                for rank, idx in enumerate(order[:8], start=1):
                    a = actions[int(idx)]
                    print(
                        f"  {rank:2d}. Q={q[idx]:+.5f} "
                        f"available={bool(feasible[idx])} "
                        f"STOP={a.is_stop} "
                        f"u={a.u_mm:+.1f} v={a.v_mm:+.1f} "
                        f"alpha={a.axis_azimuth_deg:6.1f} "
                        f"tilt={a.view_tilt_deg:5.1f} "
                        f"beta={a.view_azimuth_deg:6.1f} "
                        f"d={a.view_distance_mm:6.1f}"
                    )

            state, _reward, done, final_info = env.step(actions[selected])

        t_err = float(final_info["translation_error_mm"])
        r_err = float(final_info["rotation_error_deg"])
        success = bool(
            t_err <= base.ACCURACY_TRANSLATION_THRESHOLD_MM
            and r_err <= base.ACCURACY_ROTATION_THRESHOLD_DEG
        )

        rows.append(
            {
                "trial": completed,
                "translation_error_mm": t_err,
                "rotation_error_deg": r_err,
                "accuracy_success": success,
                "scan_count": int(final_info["scan_count"]),
                "decision_count": int(final_info["decision_count"]),
                "measurement_failures": int(final_info["measurement_failures"]),
                "plane_normal_error_deg": float(
                    final_info["true_vs_est_plane_normal_deg"]
                ),
                "d_optimal": float(final_info["d_optimal"]),
                "termination_reason": str(
                    final_info.get("termination_reason", "unknown")
                ),
                "top1_unavailable_count": int(top1_unavailable_count),
                "fallback_count": int(fallback_count),
                "decision_events": int(decision_events),
                "top1_unavailable_rate": (
                    float(top1_unavailable_count / decision_events)
                    if decision_events else 0.0
                ),
                "fallback_rate": (
                    float(fallback_count / decision_events)
                    if decision_events else 0.0
                ),
                "selected_rank_mean": (
                    float(np.mean(selected_ranks))
                    if selected_ranks else float("nan")
                ),
                "selected_rank_p95": (
                    float(np.percentile(selected_ranks, 95))
                    if selected_ranks else float("nan")
                ),
            }
        )
        completed += 1

        if (
            completed == 1
            or completed % max(1, episodes // 10) == 0
            or completed == episodes
        ):
            print(f"[MC] completed {completed}/{episodes}")

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    t = np.asarray([r["translation_error_mm"] for r in rows], dtype=float)
    rr = np.asarray([r["rotation_error_deg"] for r in rows], dtype=float)
    scans = np.asarray([r["scan_count"] for r in rows], dtype=float)
    decisions = np.asarray([r["decision_count"] for r in rows], dtype=float)
    measurement_failures = np.asarray(
        [r["measurement_failures"] for r in rows], dtype=float
    )
    plane_err = np.asarray(
        [r["plane_normal_error_deg"] for r in rows], dtype=float
    )
    dopt = np.asarray([r["d_optimal"] for r in rows], dtype=float)
    success = np.asarray([r["accuracy_success"] for r in rows], dtype=bool)

    total_decision_events = int(sum(int(r["decision_events"]) for r in rows))
    total_top1_unavailable = int(
        sum(int(r["top1_unavailable_count"]) for r in rows)
    )
    total_fallback = int(sum(int(r["fallback_count"]) for r in rows))

    all_rank_means = np.asarray(
        [r["selected_rank_mean"] for r in rows], dtype=float
    )
    all_rank_p95 = np.asarray(
        [r["selected_rank_p95"] for r in rows], dtype=float
    )

    stop_rate = float(
        np.mean([r["termination_reason"] == "stop" for r in rows])
    )

    summary = {
        "monte_carlo_trials": int(episodes),
        "test_unavailable_prob": float(env.args.test_unavailable_prob),

        "translation_median_mm": float(np.median(t)),
        "translation_p95_mm": float(np.percentile(t, 95)),
        "translation_mean_mm": float(np.mean(t)),

        "rotation_median_deg": float(np.median(rr)),
        "rotation_p95_deg": float(np.percentile(rr, 95)),
        "rotation_mean_deg": float(np.mean(rr)),

        "accuracy_success_rate": float(np.mean(success)),

        "scan_count_median": float(np.median(scans)),
        "scan_count_p95": float(np.percentile(scans, 95)),
        "decision_count_median": float(np.median(decisions)),
        "decision_count_p95": float(np.percentile(decisions, 95)),

        "measurement_failures_mean": float(np.mean(measurement_failures)),
        "plane_normal_error_median_deg": float(np.median(plane_err)),
        "d_optimal_median": float(np.median(dopt)),

        "top1_unavailable_rate": (
            float(total_top1_unavailable / total_decision_events)
            if total_decision_events else 0.0
        ),
        "fallback_execution_rate": (
            float(total_fallback / total_decision_events)
            if total_decision_events else 0.0
        ),
        "selected_rank_mean": float(np.nanmean(all_rank_means)),
        "selected_rank_episode_p95_mean": float(np.nanmean(all_rank_p95)),
        "stop_termination_rate": stop_rate,

        "success_threshold_translation_mm": float(
            base.ACCURACY_TRANSLATION_THRESHOLD_MM
        ),
        "success_threshold_rotation_deg": float(
            base.ACCURACY_ROTATION_THRESHOLD_DEG
        ),
    }

    # ------------------------------------------------------------------
    # Save full trial table and summary.
    # ------------------------------------------------------------------
    trials_csv = output_dir / "mc_trials.csv"
    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = output_dir / "mc_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    print("\n" + "=" * 88)
    print(f"MONTE-CARLO TEST SUMMARY  (N={episodes})")
    print("=" * 88)
    print(
        f"Translation error : median={summary['translation_median_mm']:.6f} mm, "
        f"P95={summary['translation_p95_mm']:.6f} mm"
    )
    print(
        f"Rotation error    : median={summary['rotation_median_deg']:.6f} deg, "
        f"P95={summary['rotation_p95_deg']:.6f} deg"
    )
    print(
        f"Accuracy success  : {100.0 * summary['accuracy_success_rate']:.2f}% "
        f"(T<={summary['success_threshold_translation_mm']:g} mm, "
        f"R<={summary['success_threshold_rotation_deg']:g} deg)"
    )
    print(
        f"Scan count        : median={summary['scan_count_median']:.2f}, "
        f"P95={summary['scan_count_p95']:.2f}"
    )
    print(
        f"Fallback behavior : top1 unavailable="
        f"{100.0 * summary['top1_unavailable_rate']:.2f}%, "
        f"fallback executed={100.0 * summary['fallback_execution_rate']:.2f}%, "
        f"mean selected rank={summary['selected_rank_mean']:.3f}"
    )
    print(
        f"STOP termination  : {100.0 * summary['stop_termination_rate']:.2f}%"
    )
    print(f"Saved trials      : {trials_csv}")
    print(f"Saved summary     : {summary_json}")

    return summary


# =============================================================================
# CLI
# =============================================================================

def positive_int(x: str) -> int:
    v = int(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Action-conditional DDQN for adaptive laser hand-eye pose ranking."
    )

    # RL
    p.add_argument(
        "--mode",
        choices=("train", "test", "train-test"),
        default="train-test",
    )
    p.add_argument("--episodes", type=positive_int, default=5000)
    p.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=500,
        help="Monte-Carlo test trial count.",
    )
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--model", type=Path, default=Path("runs/rl_handeye/ddqn.pt"))
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue training from --model. --episodes is interpreted as the "
            "number of ADDITIONAL episodes."
        ),
    )
    p.add_argument(
        "--resume-episodes-done",
        type=int,
        default=None,
        help=(
            "For legacy checkpoints that do not contain completed_episodes, "
            "specify how many episodes were already trained, e.g. 5000."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs/rl_handeye/test"),
    )
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--target-tau", type=float, default=0.01)
    p.add_argument("--batch-size", type=positive_int, default=128)
    p.add_argument("--replay-capacity", type=positive_int, default=100_000)
    p.add_argument("--warmup-transitions", type=positive_int, default=2000)
    p.add_argument("--updates-per-step", type=positive_int, default=1)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=positive_int, default=3000)
    p.add_argument("--print-every", type=positive_int, default=100)

    # Variable candidate set / no fixed final K.
    p.add_argument("--candidate-pool-size", type=positive_int, default=96)
    p.add_argument("--min-rl-decisions-before-stop", type=int, default=2)
    p.add_argument("--max-rl-decisions", type=positive_int, default=20)
    p.add_argument("--max-total-scans", type=positive_int, default=32)

    # Six-dimensional candidate domain.
    p.add_argument("--uv-half-range-mm", type=float, default=100.0)
    p.add_argument("--tilt-range-deg", type=float, nargs=2, default=(5.0, 40.0))
    p.add_argument("--distance-range-mm", type=float, nargs=2, default=(60.0, 120.0))
    p.add_argument("--beta-range-deg", type=float, nargs=2, default=(0.0, 360.0))
    p.add_argument("--profile-depth-range-mm", type=float, nargs=2, default=(30.0, 150.0))

    # Randomization actually relevant to the calibration problem.
    p.add_argument("--init-translation-mm", type=float, default=100.0)
    p.add_argument("--init-rotation-deg", type=float, default=15.0)
    p.add_argument("--true-plane-normal-random-deg", type=float, default=25.0)
    p.add_argument("--true-plane-center-jitter-mm", type=float, default=0.0)
    p.add_argument("--init-plane-normal-error-deg", type=float, default=20.0)
    p.add_argument("--init-plane-offset-error-mm", type=float, default=20.0)
    p.add_argument("--measurement-noise-std-mm", type=float, default=0.25)
    p.add_argument("--measurement-noise-std-random-mm", type=float, default=0.10)
    p.add_argument("--noise-axis", choices=("z", "xz"), default="xz")
    p.add_argument("--domain-randomization-fraction", type=float, default=0.15)

    # Candidate availability augmentation.
    # This is NOT an IK model. It teaches the ranker that rank-1 may be
    # unavailable. Later it can be combined with/replaced by real IK checks.
    p.add_argument(
        "--train-unavailable-prob-range",
        type=float,
        nargs=2,
        default=(0.0, 0.30),
        metavar=("MIN", "MAX"),
        help="Per-decision candidate dropout probability range during training.",
    )
    p.add_argument(
        "--test-unavailable-prob",
        type=float,
        default=0.20,
        help=(
            "Candidate unavailable probability during Monte-Carlo test. "
            "Set 0 for pure geometry test or when relying only on real IK."
        ),
    )

    # Estimator/Fisher settings, same semantics as reference experiment.
    p.add_argument("--fisher-rotation-scale-deg", type=float, default=2.0)
    p.add_argument("--fisher-translation-scale-mm", type=float, default=10.0)
    p.add_argument("--fisher-plane-normal-scale-deg", type=float, default=20.0)
    p.add_argument("--fisher-plane-offset-scale-mm", type=float, default=100.0)
    p.add_argument("--estimator-max-iterations", type=positive_int, default=60)
    p.add_argument("--estimator-tolerance", type=float, default=1e-7)

    # Reward.
    p.add_argument("--info-gain-reward-weight", type=float, default=0.03)
    p.add_argument("--pose-cost", type=float, default=0.05)
    p.add_argument("--failed-measurement-penalty", type=float, default=1.0)
    p.add_argument("--estimator-failure-penalty", type=float, default=1.5)
    p.add_argument("--reward-translation-scale-mm", type=float, default=1.0)
    p.add_argument("--reward-rotation-scale-deg", type=float, default=0.25)

    # State scaling / reset robustness.
    p.add_argument("--state-translation-scale-mm", type=float, default=200.0)
    p.add_argument(
        "--bootstrap-target-scans",
        type=positive_int,
        default=12,
        help="Try to acquire this many physically valid scans before RL starts.",
    )
    p.add_argument(
        "--min-bootstrap-scans",
        type=positive_int,
        default=8,
        help="Absolute minimum needed to attempt the joint estimator.",
    )
    p.add_argument(
        "--bootstrap-resample-attempts",
        type=positive_int,
        default=300,
        help="Maximum alternative bootstrap pose proposals used to fill invalid OPT12 scans.",
    )
    p.add_argument("--reset-retry-count", type=positive_int, default=20)

    # Optional real deterministic IK/collision hook.
    p.add_argument(
        "--robot-feasibility-module",
        type=Path,
        default=None,
        help=(
            "Python file defining is_pose_feasible(T_base_ef)->bool. "
            "If omitted, no fake/random IK mask is used."
        ),
    )

    args = p.parse_args()

    # Convert list defaults to tuples, validate.
    args.tilt_range_deg = tuple(map(float, args.tilt_range_deg))
    args.distance_range_mm = tuple(map(float, args.distance_range_mm))
    args.beta_range_deg = tuple(map(float, args.beta_range_deg))
    args.profile_depth_range_mm = tuple(map(float, args.profile_depth_range_mm))
    args.train_unavailable_prob_range = tuple(
        map(float, args.train_unavailable_prob_range)
    )

    if args.min_rl_decisions_before_stop < 0:
        p.error("--min-rl-decisions-before-stop must be >= 0")
    if args.resume_episodes_done is not None and args.resume_episodes_done < 0:
        p.error("--resume-episodes-done must be >= 0")
    if args.resume and args.mode == "test":
        p.error("--resume is only valid with --mode train or train-test")
    if args.bootstrap_target_scans < args.min_bootstrap_scans:
        p.error("--bootstrap-target-scans must be >= --min-bootstrap-scans")
    if args.max_total_scans < args.bootstrap_target_scans:
        p.error("--max-total-scans must be >= --bootstrap-target-scans")
    if args.uv_half_range_mm <= 0:
        p.error("--uv-half-range-mm must be > 0")
    if args.distance_range_mm[0] <= 0 or args.distance_range_mm[0] > args.distance_range_mm[1]:
        p.error("invalid --distance-range-mm")
    if args.tilt_range_deg[0] <= 0 or args.tilt_range_deg[0] > args.tilt_range_deg[1]:
        p.error("invalid --tilt-range-deg")
    if args.reward_translation_scale_mm <= 0 or args.reward_rotation_scale_deg <= 0:
        p.error("reward scales must be positive")
    p0, p1 = args.train_unavailable_prob_range
    if not (0.0 <= p0 <= p1 <= 1.0):
        p.error("--train-unavailable-prob-range must satisfy 0 <= MIN <= MAX <= 1")
    if not (0.0 <= args.test_unavailable_prob <= 1.0):
        p.error("--test-unavailable-prob must be in [0, 1]")

    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    robot_feasibility_fn = load_robot_feasibility_callback(
        args.robot_feasibility_module
    )

    # Probe a valid environment once to determine observation dimension.
    probe_env = AdaptiveHandEyeEnv(
        args,
        seed=args.seed,
        robot_feasibility_fn=robot_feasibility_fn,
    )
    last_error = None
    for _ in range(args.reset_retry_count):
        try:
            state0 = probe_env.reset()
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"could not initialize environment: {last_error}")

    print("=" * 88)
    print("ADAPTIVE HAND-EYE RL")
    print("=" * 88)
    print(f"mode                 : {args.mode}")
    print(f"state dim            : {state0.shape[0]}")
    print(f"action dim           : {ACTION_DIM}")
    print(f"candidate pool       : {args.candidate_pool_size}")
    print(
        f"bootstrap            : target={args.bootstrap_target_scans}, "
        f"minimum={args.min_bootstrap_scans}, "
        f"resample attempts={args.bootstrap_resample_attempts}"
    )
    print(f"fixed final K        : NO")
    print(f"max RL decisions     : {args.max_rl_decisions} (safety cap)")
    print(
        "train unavailable   : "
        f"U{tuple(args.train_unavailable_prob_range)} per decision"
    )
    print(f"test unavailable    : {args.test_unavailable_prob:.3f}")
    print(
        f"robot feasibility    : "
        f"{'external deterministic callback' if robot_feasibility_fn else 'not connected yet'}"
    )
    print(
        f"device               : "
        f"{args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    agent = DDQNRanker(
        state_dim=state0.shape[0],
        action_dim=ACTION_DIM,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.target_tau,
        device=args.device,
    )

    completed_before = 0

    if args.mode in ("train", "train-test"):
        if args.resume:
            if not args.model.exists():
                raise FileNotFoundError(
                    f"resume checkpoint not found: {args.model}"
                )

            ckpt = agent.load(args.model)
            ckpt_completed = int(ckpt.get("completed_episodes", 0))

            if ckpt_completed > 0:
                completed_before = ckpt_completed
            elif args.resume_episodes_done is not None:
                completed_before = int(args.resume_episodes_done)
            else:
                raise RuntimeError(
                    "This is a legacy checkpoint without completed_episodes. "
                    "Resume with --resume-episodes-done <N>, e.g. 5000."
                )

            print(
                f"resume               : YES, checkpoint={args.model}, "
                f"completed={completed_before}, additional={args.episodes}, "
                f"target={completed_before + args.episodes}"
            )
            print(
                "resume replay         : fresh replay buffer "
                "(legacy checkpoint did not store replay)"
            )

        train_env = AdaptiveHandEyeEnv(
            args,
            seed=args.seed,
            robot_feasibility_fn=robot_feasibility_fn,
        )

        # Critical for resume:
        # do not replay the same randomized environments from episode index 0.
        train_env.episode_index = int(completed_before)

        train(
            env=train_env,
            agent=agent,
            args=args,
            start_episode=completed_before,
        )

        completed_after = completed_before + int(args.episodes)
        agent.save(
            args.model,
            state_dim=state0.shape[0],
            completed_episodes=completed_after,
        )
        print(
            f"\nsaved model: {args.model} "
            f"(completed_episodes={completed_after})"
        )

    if args.mode == "test":
        if not args.model.exists():
            raise FileNotFoundError(f"model not found: {args.model}")
        agent.load(args.model)

    if args.mode in ("test", "train-test"):
        test_env = AdaptiveHandEyeEnv(
            args,
            seed=args.seed + 10_000_000,
            robot_feasibility_fn=robot_feasibility_fn,
        )
        evaluate_rl(
            env=test_env,
            agent=agent,
            episodes=args.eval_episodes,
            output_dir=args.results_dir,
            show_rankings=2,
        )


if __name__ == "__main__":
    main()