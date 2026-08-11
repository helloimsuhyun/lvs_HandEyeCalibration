from __future__ import annotations

"""
Translation-isotropic constrained optimal design inside the existing 9-line
circular laser hand-eye calibration family.

Idea
----
For each scan budget N:

1) Choose an N-line subset from the 9-line circular target.
2) Search many continuous angular configurations (theta_i, beta_i) satisfying

       H_tt / tr(H_tt) = I / 3,

   where H_tt is the raw hand-eye translation information block.  With equal
   profile-point counts this is equivalent to

       H_tt \propto sum_i q_i q_i^T,
       q_i = R_base_ef_i^T n_base.

   Numerically, isotropy is represented by five independent equalities:

       Hxx-Hyy = 0,
       Hyy-Hzz = 0,
       Hxy = 0,
       Hxz = 0,
       Hyz = 0,

   after trace normalization.

3) The feasible roots form a continuous translation-isotropic manifold.  The
   script samples this manifold using multi-start SLSQP feasibility solves.
4) From those roots, optimize the FULL unknown-plane hand-eye design criterion
   while preserving the five isotropy equalities:

       maximize log det(H_eff)
       subject to translation isotropy,

   with

       H_eff = H_xx - H_xp inv(H_pp) H_px.

   During this second stage theta, beta, and distance d all remain optimization
   variables; theta and beta may move along the isotropy manifold.  Therefore
   this is NOT the special fixed-theta=54.7356 deg / uniform-azimuth solution.

Notes
-----
- "Exact" condition number 1 is numerical: --isotropy-tol controls the final
  equality tolerance.
- The continuous isotropic set is infinite.  "Pool" here means a multi-start
  numerical sample of that manifold, not an exhaustive enumeration.
- By default the mirror branch convention is the same fixed alternating branch
  used by the previous continuous circular-pattern script.  This keeps the new
  result directly comparable to the previous unconstrained search.
- Final minimum-N claims still require nonlinear paired Monte Carlo validation.
"""

import argparse
import csv
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Allow direct execution from robust_laser_handeye/examples.
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "examples" else _THIS_FILE.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from laser_handeye.patterns import circular_lines
from laser_handeye.se3 import inv_T, transform_points
from laser_handeye.simulation import (
    is_reachable_simple,
    sensor_pose_from_target_line,
    simulate_profile_on_plane,
)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseParam:
    line_id: int
    theta_deg: float
    beta_deg: float
    d_mm: float
    branch_sign: float


@dataclass
class InformationMetrics:
    strict_logdet: float
    search_logdet: float
    rank_eff: int
    lambda_min: float
    lambda_max: float
    condition: float
    eigenvalues: list[float]
    nuisance_rank: int
    joint_rank: int


@dataclass
class PoolRoot:
    line_ids: tuple[int, ...]
    angles: np.ndarray                 # [theta0,beta0,theta1,beta1,...]
    translation_condition: float
    isotropy_error: float
    equalities: np.ndarray
    source_start: int
    screen_vector: np.ndarray | None = None
    screen_metrics: InformationMetrics | None = None


@dataclass
class RefinedCandidate:
    line_ids: tuple[int, ...]
    vector: np.ndarray                 # [theta,beta,d] * N
    metrics: InformationMetrics
    translation_condition: float
    isotropy_error: float
    equalities: np.ndarray
    source_pool_index: int
    optimizer_success: bool
    optimizer_message: str


# -----------------------------------------------------------------------------
# Geometry / parameter helpers
# -----------------------------------------------------------------------------


def branch_sign_for_line(line_id: int, mode: str) -> float:
    if mode == "positive":
        return 1.0
    if mode == "alternating":
        return 1.0 if int(line_id) % 2 == 0 else -1.0
    raise ValueError("branch mode must be 'alternating' or 'positive'")


def unpack_full_design(
    line_ids: Sequence[int],
    vector: np.ndarray,
    branch_mode: str,
) -> list[PoseParam]:
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size != 3 * len(line_ids):
        raise ValueError("full design vector must have 3*N values")
    poses: list[PoseParam] = []
    for i, line_id in enumerate(line_ids):
        theta, beta, d_mm = x[3 * i : 3 * i + 3]
        poses.append(
            PoseParam(
                line_id=int(line_id),
                theta_deg=float(theta),
                beta_deg=float(beta),
                d_mm=float(d_mm),
                branch_sign=branch_sign_for_line(line_id, branch_mode),
            )
        )
    return poses


def angles_from_full_vector(vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size % 3 != 0:
        raise ValueError("full vector length must be divisible by 3")
    N = x.size // 3
    a = np.empty(2 * N, dtype=float)
    for i in range(N):
        a[2 * i : 2 * i + 2] = x[3 * i : 3 * i + 2]
    return a


def full_vector_from_angles_and_distances(
    angles: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    a = np.asarray(angles, dtype=float).reshape(-1)
    d = np.asarray(distances, dtype=float).reshape(-1)
    if a.size != 2 * d.size:
        raise ValueError("angles must contain 2*N entries and distances N entries")
    N = d.size
    x = np.empty(3 * N, dtype=float)
    for i in range(N):
        x[3 * i] = a[2 * i]
        x[3 * i + 1] = a[2 * i + 1]
        x[3 * i + 2] = d[i]
    return x


def paper_incidence_margin(theta_deg: float, beta_deg: float) -> float:
    """Feasible when margin >= 0: sin(beta) >= cos(theta)."""
    th = math.radians(float(theta_deg))
    be = math.radians(float(beta_deg))
    return float(math.sin(be) - math.cos(th))


def angular_incidence_margins(angles: np.ndarray) -> np.ndarray:
    a = np.asarray(angles, dtype=float).reshape(-1)
    N = a.size // 2
    return np.asarray(
        [paper_incidence_margin(a[2 * i], a[2 * i + 1]) for i in range(N)],
        dtype=float,
    )


def full_incidence_margins(vector: np.ndarray) -> np.ndarray:
    return angular_incidence_margins(angles_from_full_vector(vector))


# -----------------------------------------------------------------------------
# Translation block and exact-isotropy constraints
# -----------------------------------------------------------------------------


def normalized_translation_information_from_q(q_rows: np.ndarray) -> np.ndarray:
    q = np.asarray(q_rows, dtype=float).reshape(-1, 3)
    if q.shape[0] == 0:
        raise ValueError("need at least one translation normal")
    H = q.T @ q
    H = 0.5 * (H + H.T)
    tr = float(np.trace(H))
    if tr <= np.finfo(float).eps:
        raise ValueError("translation information has zero trace")
    return H / tr


def isotropy_equalities_from_Hn(Hn: np.ndarray) -> np.ndarray:
    """Five independent equalities whose zero set is Hn = I/3.

    Off-diagonal terms are multiplied by sqrt(2) so the Euclidean norm of this
    vector has a direct Frobenius-norm interpretation up to the two independent
    diagonal coordinates.
    """
    H = np.asarray(Hn, dtype=float).reshape(3, 3)
    return np.asarray(
        [
            H[0, 0] - H[1, 1],
            H[1, 1] - H[2, 2],
            math.sqrt(2.0) * H[0, 1],
            math.sqrt(2.0) * H[0, 2],
            math.sqrt(2.0) * H[1, 2],
        ],
        dtype=float,
    )


def translation_condition_from_Hn(Hn: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(0.5 * (Hn + Hn.T))
    if eig[0] <= np.finfo(float).eps:
        return float("inf")
    return float(eig[-1] / eig[0])


# -----------------------------------------------------------------------------
# Full residual/Jacobian/Schur information (same formulation as prior script)
# -----------------------------------------------------------------------------


def tangent_basis(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    axes = np.eye(3)
    seed = axes[int(np.argmin(np.abs(axes @ n)))]
    c1 = seed - n * float(seed @ n)
    c1 /= np.linalg.norm(c1)
    c2 = np.cross(n, c1)
    c2 /= np.linalg.norm(c2)
    return np.column_stack([c1, c2])


def full_problem_residuals(
    scans: Sequence,
    T_handeye: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
) -> np.ndarray:
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    blocks: list[np.ndarray] = []
    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float)
        if points_s.size == 0:
            continue
        points_ef = transform_points(T_handeye, points_s)
        points_b = transform_points(np.asarray(scan.T_base_ef), points_ef)
        blocks.append(points_b @ n - float(plane_l))
    if not blocks:
        return np.empty(0, dtype=float)
    return np.concatenate(blocks)


def scaled_joint_jacobian(
    scans: Sequence,
    T_reference: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    *,
    handeye_characteristic_length_mm: float,
    plane_characteristic_length_mm: float,
    finite_difference_step_mm: float,
) -> np.ndarray:
    Lh = float(handeye_characteristic_length_mm)
    Lp = float(plane_characteristic_length_mm)
    h = float(finite_difference_step_mm)
    if Lh <= 0.0 or Lp <= 0.0 or h <= 0.0:
        raise ValueError("characteristic lengths and finite-difference step must be positive")

    T0 = np.asarray(T_reference, dtype=float).reshape(4, 4)
    n0 = np.asarray(plane_n, dtype=float).reshape(3)
    n0 /= np.linalg.norm(n0)
    l0 = float(plane_l)
    C = tangent_basis(n0)

    r0 = full_problem_residuals(scans, T0, n0, l0)
    if r0.size == 0:
        raise ValueError("no residuals available for Jacobian")

    J = np.empty((r0.size, 9), dtype=float)

    # Hand-eye rotation, right perturbation, scaled by Lh.
    dphi = h / Lh
    for axis in range(3):
        w = np.zeros(3, dtype=float)
        w[axis] = dphi
        Tp = T0.copy()
        Tm = T0.copy()
        Tp[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(w).as_matrix()
        Tm[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(-w).as_matrix()
        rp = full_problem_residuals(scans, Tp, n0, l0)
        rm = full_problem_residuals(scans, Tm, n0, l0)
        J[:, axis] = (rp - rm) / (2.0 * h)

    # Hand-eye translation.
    for axis in range(3):
        Tp = T0.copy()
        Tm = T0.copy()
        Tp[axis, 3] += h
        Tm[axis, 3] -= h
        rp = full_problem_residuals(scans, Tp, n0, l0)
        rm = full_problem_residuals(scans, Tm, n0, l0)
        J[:, 3 + axis] = (rp - rm) / (2.0 * h)

    # Plane-normal tangent coordinates, scaled by Lp.
    dn = h / Lp
    for axis in range(2):
        direction = C[:, axis]
        np_ = n0 + dn * direction
        nm_ = n0 - dn * direction
        np_ /= np.linalg.norm(np_)
        nm_ /= np.linalg.norm(nm_)
        rp = full_problem_residuals(scans, T0, np_, l0)
        rm = full_problem_residuals(scans, T0, nm_, l0)
        J[:, 6 + axis] = (rp - rm) / (2.0 * h)

    J[:, 8] = -1.0
    return J


def information_metrics_from_jacobian(
    J: np.ndarray,
    *,
    eigen_rtol: float,
    search_regularization: float,
) -> tuple[InformationMetrics, np.ndarray]:
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[1] != 9:
        raise ValueError("J must have shape (M,9)")

    H = J.T @ J
    H = 0.5 * (H + H.T)
    H_xx = H[:6, :6]
    H_xp = H[:6, 6:]
    H_pp = H[6:, 6:]

    eig_pp = np.linalg.eigvalsh(0.5 * (H_pp + H_pp.T))
    pp_scale = max(float(np.max(np.abs(eig_pp))), 1.0)
    pp_thr = float(eigen_rtol) * pp_scale
    nuisance_rank = int(np.count_nonzero(eig_pp > pp_thr))
    joint_rank = int(np.linalg.matrix_rank(J))

    if nuisance_rank < 3:
        m = InformationMetrics(
            strict_logdet=float("-inf"),
            search_logdet=float("-inf"),
            rank_eff=0,
            lambda_min=0.0,
            lambda_max=0.0,
            condition=float("inf"),
            eigenvalues=[0.0] * 6,
            nuisance_rank=nuisance_rank,
            joint_rank=joint_rank,
        )
        return m, np.zeros((6, 6), dtype=float)

    try:
        correction = H_xp @ np.linalg.solve(H_pp, H_xp.T)
    except np.linalg.LinAlgError:
        correction = H_xp @ np.linalg.pinv(H_pp) @ H_xp.T

    H_eff = H_xx - correction
    H_eff = 0.5 * (H_eff + H_eff.T)
    eig = np.linalg.eigvalsh(H_eff)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    thr = float(eigen_rtol) * scale
    rank_eff = int(np.count_nonzero(eig > thr))
    lmin = float(max(eig[0], 0.0))
    lmax = float(max(eig[-1], 0.0))
    cond = float(lmax / lmin) if rank_eff == 6 and lmin > 0.0 else float("inf")

    if rank_eff == 6 and np.all(eig > thr):
        strict = float(np.sum(np.log(eig)))
    else:
        strict = float("-inf")

    reg = float(search_regularization)
    if reg <= 0.0:
        raise ValueError("search regularization must be positive")
    clipped = np.maximum(eig, reg)
    search = float(np.sum(np.log(clipped)))
    if np.any(eig < -10.0 * reg):
        search -= 1e3 * float(np.sum(np.abs(eig[eig < 0.0])))

    m = InformationMetrics(
        strict_logdet=strict,
        search_logdet=search,
        rank_eff=rank_eff,
        lambda_min=lmin,
        lambda_max=lmax,
        condition=cond,
        eigenvalues=[float(v) for v in eig],
        nuisance_rank=nuisance_rank,
        joint_rank=joint_rank,
    )
    return m, H_eff


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------


class Evaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lines = circular_lines(args.radius_mm, n_lines=9)

        self.plane_R = np.eye(3, dtype=float)
        self.plane_t = np.array([0.0, 0.0, args.canonical_plane_z_mm], dtype=float)
        self.plane_n = np.array([0.0, 0.0, 1.0], dtype=float)
        self.plane_l = float(self.plane_n @ self.plane_t)
        self.T_true = np.eye(4, dtype=float)
        self.x_values = np.linspace(
            -float(args.profile_half_width_mm),
            float(args.profile_half_width_mm),
            int(args.profile_points),
        )
        self.full_evaluations = 0
        self.translation_evaluations = 0

    @property
    def d_reference(self) -> float:
        return 0.5 * (float(self.args.d_min_mm) + float(self.args.d_max_mm))

    def translation_q_rows(
        self,
        line_ids: Sequence[int],
        angles: np.ndarray,
    ) -> np.ndarray:
        """q_i = R_base_ef_i^T n_base for one scan on each selected line.

        With T_true=I in canonical design coordinates, EF and sensor frames are
        identical.  d is set to a fixed reference because orientation, and thus
        q_i, is independent of standoff in this pose generator.
        """
        self.translation_evaluations += 1
        a = np.asarray(angles, dtype=float).reshape(-1)
        if a.size != 2 * len(line_ids):
            raise ValueError("angles must have 2*N values")

        q_rows = []
        for i, line_id in enumerate(line_ids):
            theta = float(a[2 * i])
            beta = float(a[2 * i + 1])
            if self.args.pose_geometry == "paper_incidence":
                if paper_incidence_margin(theta, beta) < -1e-10:
                    raise ValueError("infeasible paper-incidence angle pair")

            p0, p1 = self.lines[int(line_id)]
            T_base_s = sensor_pose_from_target_line(
                plane_R=self.plane_R,
                plane_t=self.plane_t,
                line_p0=p0,
                line_p1=p1,
                d_mm=self.d_reference,
                theta_deg=theta,
                beta_deg=beta,
                branch_sign=branch_sign_for_line(line_id, self.args.branch_mode),
                pose_geometry=self.args.pose_geometry,
            )
            T_base_ef = T_base_s @ inv_T(self.T_true)
            if self.args.check_reachability and not is_reachable_simple(T_base_ef):
                raise ValueError("simple reachability failed")

            R = np.asarray(T_base_ef, dtype=float)[:3, :3]
            q = R.T @ self.plane_n
            nq = float(np.linalg.norm(q))
            if nq <= 0.0:
                raise ValueError("zero translation normal")
            q_rows.append(q / nq)

        return np.asarray(q_rows, dtype=float)

    def translation_report(
        self,
        line_ids: Sequence[int],
        angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        q = self.translation_q_rows(line_ids, angles)
        Hn = normalized_translation_information_from_q(q)
        eq = isotropy_equalities_from_Hn(Hn)
        err = float(np.linalg.norm(Hn - np.eye(3) / 3.0, ord="fro"))
        cond = translation_condition_from_Hn(Hn)
        return Hn, eq, err, cond, q

    def make_scans(self, line_ids: Sequence[int], vector: np.ndarray) -> list:
        poses = unpack_full_design(line_ids, vector, self.args.branch_mode)
        scans = []
        for scan_id, pose in enumerate(poses):
            if self.args.pose_geometry == "paper_incidence":
                if paper_incidence_margin(pose.theta_deg, pose.beta_deg) < -1e-10:
                    raise ValueError("infeasible paper-incidence angle pair")

            p0, p1 = self.lines[pose.line_id]
            T_base_s = sensor_pose_from_target_line(
                plane_R=self.plane_R,
                plane_t=self.plane_t,
                line_p0=p0,
                line_p1=p1,
                d_mm=pose.d_mm,
                theta_deg=pose.theta_deg,
                beta_deg=pose.beta_deg,
                branch_sign=pose.branch_sign,
                pose_geometry=self.args.pose_geometry,
            )
            T_base_ef = T_base_s @ inv_T(self.T_true)
            if self.args.check_reachability and not is_reachable_simple(T_base_ef):
                raise ValueError("simple reachability failed")

            scan = simulate_profile_on_plane(
                T_base_ef=T_base_ef,
                T_ef_s_true=self.T_true,
                plane_n=self.plane_n,
                plane_l=self.plane_l,
                x_values=self.x_values,
                noise_std=0.0,
                rng=np.random.default_rng(0),
                plane_id=0,
                scan_id=scan_id,
                meta={
                    "line_id": pose.line_id,
                    "theta_deg": pose.theta_deg,
                    "beta_deg": pose.beta_deg,
                    "d_mm": pose.d_mm,
                    "theta_branch_sign": pose.branch_sign,
                    "pose_geometry": self.args.pose_geometry,
                },
            )
            scans.append(scan)

        if len(scans) != len(line_ids):
            raise RuntimeError("failed to generate exactly N scans")
        return scans

    def full_metrics(
        self,
        line_ids: Sequence[int],
        vector: np.ndarray,
    ) -> tuple[InformationMetrics, np.ndarray]:
        self.full_evaluations += 1
        try:
            scans = self.make_scans(line_ids, vector)
            J = scaled_joint_jacobian(
                scans,
                self.T_true,
                self.plane_n,
                self.plane_l,
                handeye_characteristic_length_mm=self.args.handeye_characteristic_length_mm,
                plane_characteristic_length_mm=self.args.plane_characteristic_length_mm,
                finite_difference_step_mm=self.args.design_step_mm,
            )
            return information_metrics_from_jacobian(
                J,
                eigen_rtol=self.args.design_eigen_rtol,
                search_regularization=self.args.search_regularization,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            bad = InformationMetrics(
                strict_logdet=float("-inf"),
                search_logdet=-1e12,
                rank_eff=0,
                lambda_min=0.0,
                lambda_max=0.0,
                condition=float("inf"),
                eigenvalues=[0.0] * 6,
                nuisance_rank=0,
                joint_rank=0,
            )
            return bad, np.zeros((6, 6), dtype=float)


# -----------------------------------------------------------------------------
# Sampling / feasibility-root search
# -----------------------------------------------------------------------------


def angle_bounds(N: int, args: argparse.Namespace) -> list[tuple[float, float]]:
    return [
        bound
        for _ in range(N)
        for bound in (
            (float(args.theta_min_deg), float(args.theta_max_deg)),
            (float(args.beta_min_deg), float(args.beta_max_deg)),
        )
    ]


def full_bounds(N: int, args: argparse.Namespace) -> list[tuple[float, float]]:
    return [
        bound
        for _ in range(N)
        for bound in (
            (float(args.theta_min_deg), float(args.theta_max_deg)),
            (float(args.beta_min_deg), float(args.beta_max_deg)),
            (float(args.d_min_mm), float(args.d_max_mm)),
        )
    ]


def sample_feasible_angles(
    rng: np.random.Generator,
    N: int,
    args: argparse.Namespace,
) -> np.ndarray:
    values: list[float] = []
    for _ in range(N):
        theta = float(rng.uniform(args.theta_min_deg, args.theta_max_deg))
        if args.pose_geometry == "paper_incidence":
            # Exact feasible interval from sin(beta) >= cos(theta), for
            # theta in [0,90): beta in [90-theta, 90+theta].
            lo = max(float(args.beta_min_deg), 90.0 - theta)
            hi = min(float(args.beta_max_deg), 90.0 + theta)
            if lo > hi:
                # Resample theta until a nonempty beta interval exists.
                for _attempt in range(10000):
                    theta = float(rng.uniform(args.theta_min_deg, args.theta_max_deg))
                    lo = max(float(args.beta_min_deg), 90.0 - theta)
                    hi = min(float(args.beta_max_deg), 90.0 + theta)
                    if lo <= hi:
                        break
                else:
                    raise RuntimeError("no feasible theta/beta interval inside requested bounds")
            beta = float(rng.uniform(lo, hi))
        else:
            beta = float(rng.uniform(args.beta_min_deg, args.beta_max_deg))
        values.extend([theta, beta])
    return np.asarray(values, dtype=float)


def pool_root_objective(
    angles: np.ndarray,
    line_ids: Sequence[int],
    evaluator: Evaluator,
) -> float:
    try:
        _Hn, eq, _err, _cond, _q = evaluator.translation_report(line_ids, angles)
        return 0.5 * float(eq @ eq)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 1e6


def find_translation_isotropic_root(
    *,
    line_ids: tuple[int, ...],
    start_angles: np.ndarray,
    evaluator: Evaluator,
    args: argparse.Namespace,
    source_start: int,
) -> PoolRoot | None:
    constraints = []
    if args.pose_geometry == "paper_incidence":
        constraints.append({"type": "ineq", "fun": angular_incidence_margins})

    result = minimize(
        pool_root_objective,
        np.asarray(start_angles, dtype=float),
        args=(line_ids, evaluator),
        method="SLSQP",
        bounds=angle_bounds(len(line_ids), args),
        constraints=constraints,
        options={
            "maxiter": int(args.pool_maxiter),
            "ftol": float(args.pool_ftol),
            "disp": False,
        },
    )

    a = np.asarray(result.x, dtype=float)
    try:
        _Hn, eq, err, cond, _q = evaluator.translation_report(line_ids, a)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None

    if not np.all(np.isfinite(eq)):
        return None
    if err > float(args.isotropy_tol):
        return None
    if args.pose_geometry == "paper_incidence" and np.min(angular_incidence_margins(a)) < -1e-8:
        return None

    return PoolRoot(
        line_ids=line_ids,
        angles=a,
        translation_condition=cond,
        isotropy_error=err,
        equalities=eq,
        source_start=int(source_start),
    )


def root_is_duplicate(
    root: PoolRoot,
    existing: Sequence[PoolRoot],
    tol_deg: float,
) -> bool:
    for other in existing:
        if other.line_ids != root.line_ids:
            continue
        if np.max(np.abs(other.angles - root.angles)) <= float(tol_deg):
            return True
    return False


def build_pool_for_subset(
    *,
    line_ids: tuple[int, ...],
    evaluator: Evaluator,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> list[PoolRoot]:
    roots: list[PoolRoot] = []
    for start_idx in range(int(args.pool_starts_per_subset)):
        a0 = sample_feasible_angles(rng, len(line_ids), args)
        root = find_translation_isotropic_root(
            line_ids=line_ids,
            start_angles=a0,
            evaluator=evaluator,
            args=args,
            source_start=start_idx,
        )
        if root is None:
            continue
        if not root_is_duplicate(root, roots, args.pool_dedup_tol_deg):
            roots.append(root)
    return roots


# -----------------------------------------------------------------------------
# Full-score screening and constrained refinement on isotropy manifold
# -----------------------------------------------------------------------------


def screen_pool_root(
    root: PoolRoot,
    evaluator: Evaluator,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    best_vector = None
    best_metrics = None
    n_samples = max(1, int(args.distance_screen_samples))

    for s in range(n_samples):
        if s == 0:
            distances = np.full(len(root.line_ids), evaluator.d_reference, dtype=float)
        else:
            distances = rng.uniform(
                float(args.d_min_mm),
                float(args.d_max_mm),
                size=len(root.line_ids),
            )
        x = full_vector_from_angles_and_distances(root.angles, distances)
        metrics, _ = evaluator.full_metrics(root.line_ids, x)
        if best_metrics is None or metrics.search_logdet > best_metrics.search_logdet:
            best_metrics = metrics
            best_vector = x

    root.screen_vector = best_vector
    root.screen_metrics = best_metrics


def full_translation_equalities(
    vector: np.ndarray,
    line_ids: Sequence[int],
    evaluator: Evaluator,
) -> np.ndarray:
    angles = angles_from_full_vector(vector)
    try:
        _Hn, eq, _err, _cond, _q = evaluator.translation_report(line_ids, angles)
        return eq
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        # SLSQP needs a finite vector. Large nonzero residual keeps it away.
        return np.full(5, 1e3, dtype=float)


def full_objective(
    vector: np.ndarray,
    line_ids: Sequence[int],
    evaluator: Evaluator,
) -> float:
    metrics, _ = evaluator.full_metrics(line_ids, vector)
    if not np.isfinite(metrics.search_logdet):
        return 1e12
    return -float(metrics.search_logdet)


def refine_pool_root(
    *,
    root: PoolRoot,
    pool_index: int,
    evaluator: Evaluator,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> RefinedCandidate | None:
    if root.screen_vector is None:
        raise ValueError("root must be screened before refinement")

    starts: list[np.ndarray] = [np.asarray(root.screen_vector, dtype=float).copy()]
    for _ in range(max(0, int(args.local_distance_starts) - 1)):
        distances = rng.uniform(
            float(args.d_min_mm),
            float(args.d_max_mm),
            size=len(root.line_ids),
        )
        starts.append(full_vector_from_angles_and_distances(root.angles, distances))

    best: RefinedCandidate | None = None
    for x0 in starts:
        constraints: list[dict] = [
            {
                "type": "eq",
                "fun": lambda x, ids=root.line_ids: full_translation_equalities(x, ids, evaluator),
            }
        ]
        if args.pose_geometry == "paper_incidence":
            constraints.append({"type": "ineq", "fun": full_incidence_margins})

        result = minimize(
            full_objective,
            x0,
            args=(root.line_ids, evaluator),
            method="SLSQP",
            bounds=full_bounds(len(root.line_ids), args),
            constraints=constraints,
            options={
                "maxiter": int(args.local_maxiter),
                "ftol": float(args.local_ftol),
                "disp": False,
            },
        )

        x = np.asarray(result.x, dtype=float)
        angles = angles_from_full_vector(x)
        try:
            _Hn, eq, iso_err, tcond, _q = evaluator.translation_report(root.line_ids, angles)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

        # Equality-constrained optimizers can terminate with a slightly violated
        # constraint.  Reject anything outside the requested numerical exactness.
        if iso_err > float(args.refined_isotropy_tol):
            continue
        if args.pose_geometry == "paper_incidence" and np.min(full_incidence_margins(x)) < -1e-7:
            continue

        metrics, _ = evaluator.full_metrics(root.line_ids, x)
        candidate = RefinedCandidate(
            line_ids=root.line_ids,
            vector=x,
            metrics=metrics,
            translation_condition=tcond,
            isotropy_error=iso_err,
            equalities=eq,
            source_pool_index=int(pool_index),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

        if best is None or candidate.metrics.search_logdet > best.metrics.search_logdet:
            best = candidate

    return best


# -----------------------------------------------------------------------------
# Search per N
# -----------------------------------------------------------------------------


def search_one_n(
    N: int,
    evaluator: Evaluator,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[RefinedCandidate | None, list[PoolRoot], dict]:
    start_time = time.perf_counter()
    all_roots: list[PoolRoot] = []
    subsets_with_roots = 0
    subset_count = math.comb(9, N)

    print(f"\n[N={N}] line subsets={subset_count}")

    for subset_idx, subset in enumerate(itertools.combinations(range(9), N), start=1):
        line_ids = tuple(int(v) for v in subset)
        roots = build_pool_for_subset(
            line_ids=line_ids,
            evaluator=evaluator,
            rng=rng,
            args=args,
        )
        if roots:
            subsets_with_roots += 1
            all_roots.extend(roots)

        if subset_idx % max(1, int(args.progress_every)) == 0 or subset_idx == subset_count:
            print(
                f"  pool {subset_idx:>4}/{subset_count} | "
                f"roots={len(all_roots):>5} | "
                f"subsets-with-roots={subsets_with_roots:>4}"
            )

    if not all_roots:
        return None, [], {
            "subset_count": subset_count,
            "subsets_with_roots": 0,
            "pool_roots": 0,
            "elapsed_sec": time.perf_counter() - start_time,
        }

    # Evaluate full D_s score at one/few distance assignments for every sampled
    # translation-isotropic root.  This is only screening; final roots are moved
    # along the equality manifold during constrained refinement.
    print(f"  screening {len(all_roots)} isotropic roots with full Jacobian ...")
    for idx, root in enumerate(all_roots, start=1):
        screen_pool_root(root, evaluator, rng, args)
        if idx % max(1, int(args.screen_progress_every)) == 0 or idx == len(all_roots):
            print(f"    full-score screen {idx:>5}/{len(all_roots)}")

    ranked_indices = sorted(
        range(len(all_roots)),
        key=lambda idx: (
            -np.inf
            if all_roots[idx].screen_metrics is None
            else all_roots[idx].screen_metrics.search_logdet
        ),
        reverse=True,
    )
    refine_indices = ranked_indices[: min(int(args.refine_top_pool), len(ranked_indices))]

    print(f"  constrained refinement of top {len(refine_indices)} pool roots ...")
    best: RefinedCandidate | None = None
    refined_count = 0
    for j, pool_index in enumerate(refine_indices):
        root = all_roots[pool_index]
        cand = refine_pool_root(
            root=root,
            pool_index=pool_index,
            evaluator=evaluator,
            rng=rng,
            args=args,
        )
        if cand is not None:
            refined_count += 1
            if best is None or cand.metrics.search_logdet > best.metrics.search_logdet:
                best = cand

        if (j + 1) % max(1, int(args.refine_progress_every)) == 0 or j + 1 == len(refine_indices):
            best_txt = "none" if best is None else f"{best.metrics.strict_logdet:.6g}"
            print(
                f"    refine {j+1:>4}/{len(refine_indices)} | "
                f"accepted={refined_count:>4} | best logdet={best_txt}"
            )

    diagnostics = {
        "subset_count": subset_count,
        "subsets_with_roots": subsets_with_roots,
        "pool_roots": len(all_roots),
        "refined_accepted": refined_count,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    return best, all_roots, diagnostics


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def save_pool_csv(pool: Sequence[PoolRoot], path: Path) -> None:
    rows = []
    for idx, root in enumerate(pool):
        row = {
            "pool_index": idx,
            "line_ids": " ".join(map(str, root.line_ids)),
            "translation_condition": root.translation_condition,
            "isotropy_error_fro": root.isotropy_error,
            "eq_0_Hxx_minus_Hyy": root.equalities[0],
            "eq_1_Hyy_minus_Hzz": root.equalities[1],
            "eq_2_sqrt2_Hxy": root.equalities[2],
            "eq_3_sqrt2_Hxz": root.equalities[3],
            "eq_4_sqrt2_Hyz": root.equalities[4],
            "screen_search_logdet": (
                np.nan if root.screen_metrics is None else root.screen_metrics.search_logdet
            ),
            "screen_strict_logdet": (
                np.nan if root.screen_metrics is None else root.screen_metrics.strict_logdet
            ),
            "screen_rank_eff": (
                -1 if root.screen_metrics is None else root.screen_metrics.rank_eff
            ),
        }
        for i in range(len(root.line_ids)):
            row[f"theta_{i}_deg"] = root.angles[2 * i]
            row[f"beta_{i}_deg"] = root.angles[2 * i + 1]
        rows.append(row)

    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_best_design(
    best: RefinedCandidate,
    evaluator: Evaluator,
    n_dir: Path,
    args: argparse.Namespace,
) -> None:
    n_dir.mkdir(parents=True, exist_ok=True)
    poses = unpack_full_design(best.line_ids, best.vector, args.branch_mode)
    angles = angles_from_full_vector(best.vector)
    Hn, eq, iso_err, tcond, q_rows = evaluator.translation_report(best.line_ids, angles)
    metrics, H_eff = evaluator.full_metrics(best.line_ids, best.vector)

    rows = []
    for i, pose in enumerate(poses):
        rows.append(
            {
                "scan_index": i,
                "line_id": pose.line_id,
                "theta_deg": pose.theta_deg,
                "beta_deg": pose.beta_deg,
                "d_mm": pose.d_mm,
                "branch_sign": pose.branch_sign,
                "q_tx": q_rows[i, 0],
                "q_ty": q_rows[i, 1],
                "q_tz": q_rows[i, 2],
            }
        )
    with (n_dir / "best_design.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savetxt(n_dir / "best_translation_H_normalized.csv", Hn, delimiter=",")
    np.savetxt(n_dir / "best_H_eff_6x6.csv", H_eff, delimiter=",")

    report = {
        "line_ids": list(best.line_ids),
        "strict_logdet": metrics.strict_logdet,
        "search_logdet": metrics.search_logdet,
        "rank_eff": metrics.rank_eff,
        "lambda_min": metrics.lambda_min,
        "lambda_max": metrics.lambda_max,
        "condition_eff": metrics.condition,
        "effective_eigenvalues": metrics.eigenvalues,
        "nuisance_rank": metrics.nuisance_rank,
        "joint_rank": metrics.joint_rank,
        "translation_condition": tcond,
        "translation_isotropy_error_fro": iso_err,
        "translation_equalities": [float(v) for v in eq],
        "optimizer_success": best.optimizer_success,
        "optimizer_message": best.optimizer_message,
        "source_pool_index": best.source_pool_index,
    }
    with (n_dir / "best_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def save_pool_condition_plot(pool: Sequence[PoolRoot], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not pool:
        return
    cond = np.asarray([r.translation_condition for r in pool], dtype=float)
    err = np.asarray([r.isotropy_error for r in pool], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(err, cond, s=14, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("translation isotropy error ||H/tr(H)-I/3||_F")
    ax.set_ylabel("translation condition number")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "summary_by_N.csv"
    if not rows:
        return path
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    if not (1 <= args.n_min <= args.n_max <= 9):
        raise ValueError("require 1 <= n_min <= n_max <= 9")
    if args.profile_points < 2:
        raise ValueError("profile_points must be >= 2")
    if args.theta_min_deg >= args.theta_max_deg:
        raise ValueError("invalid theta bounds")
    if args.beta_min_deg >= args.beta_max_deg:
        raise ValueError("invalid beta bounds")
    if args.d_min_mm >= args.d_max_mm:
        raise ValueError("invalid distance bounds")
    if args.pose_geometry == "paper_incidence":
        if args.theta_min_deg < 0.0 or args.theta_max_deg >= 90.0:
            raise ValueError("paper_incidence requires theta in [0,90)")
        if args.beta_min_deg <= 0.0 or args.beta_max_deg >= 180.0:
            raise ValueError("paper_incidence requires beta in (0,180)")
    if args.pool_starts_per_subset < 1:
        raise ValueError("pool_starts_per_subset must be >= 1")
    if args.refine_top_pool < 1:
        raise ValueError("refine_top_pool must be >= 1")
    if args.isotropy_tol <= 0.0 or args.refined_isotropy_tol <= 0.0:
        raise ValueError("isotropy tolerances must be positive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample the translation-condition-1 manifold inside the 9-line circular "
            "pose family, then maximize nuisance-marginalized hand-eye D_s information "
            "subject to exact translation isotropy."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/circular_translation_isotropic_design"),
    )
    p.add_argument("--seed", type=int, default=20260811)

    p.add_argument("--n-min", type=int, default=6)
    p.add_argument("--n-max", type=int, default=8)

    p.add_argument("--radius-mm", type=float, default=100.0)
    p.add_argument("--profile-points", type=int, default=50)
    p.add_argument("--profile-half-width-mm", type=float, default=25.0)
    p.add_argument("--canonical-plane-z-mm", type=float, default=500.0)

    # Keep defaults identical to the previous continuous search for fair comparison.
    p.add_argument("--theta-min-deg", type=float, default=30.0)
    p.add_argument("--theta-max-deg", type=float, default=60.0)
    p.add_argument("--beta-min-deg", type=float, default=60.0)
    p.add_argument("--beta-max-deg", type=float, default=120.0)
    p.add_argument("--d-min-mm", type=float, default=60.0)
    p.add_argument("--d-max-mm", type=float, default=120.0)
    p.add_argument(
        "--pose-geometry",
        choices=("paper_incidence", "observable_dihedral"),
        default="paper_incidence",
    )
    p.add_argument(
        "--branch-mode",
        choices=("alternating", "positive"),
        default="alternating",
        help="Fixed mirror-branch convention; alternating matches the previous continuous design.",
    )
    p.add_argument("--check-reachability", action="store_true")

    # Same full-Jacobian scaling as previous continuous design.
    p.add_argument("--handeye-characteristic-length-mm", type=float, default=100.0)
    p.add_argument("--plane-characteristic-length-mm", type=float, default=100.0)
    p.add_argument("--design-step-mm", type=float, default=1e-3)
    p.add_argument("--design-eigen-rtol", type=float, default=1e-10)
    p.add_argument("--search-regularization", type=float, default=1e-12)

    # Translation-isotropic manifold sampling.
    p.add_argument(
        "--pool-starts-per-subset",
        type=int,
        default=20,
        help="Random multi-start feasibility solves per line subset.",
    )
    p.add_argument("--pool-maxiter", type=int, default=400)
    p.add_argument("--pool-ftol", type=float, default=1e-13)
    p.add_argument(
        "--isotropy-tol",
        type=float,
        default=1e-7,
        help="Accept a pool root when ||H_tt/tr(H_tt)-I/3||_F <= this value.",
    )
    p.add_argument(
        "--pool-dedup-tol-deg",
        type=float,
        default=1e-3,
        help="Angular max-norm tolerance for removing numerically identical roots.",
    )

    # Full-Jacobian screening/refinement on the isotropy manifold.
    p.add_argument("--distance-screen-samples", type=int, default=2)
    p.add_argument(
        "--refine-top-pool",
        type=int,
        default=20,
        help="Number of best screened isotropic roots refined with equality constraints.",
    )
    p.add_argument("--local-distance-starts", type=int, default=2)
    p.add_argument("--local-maxiter", type=int, default=120)
    p.add_argument("--local-ftol", type=float, default=1e-9)
    p.add_argument(
        "--refined-isotropy-tol",
        type=float,
        default=2e-6,
        help="Final accepted isotropy error after constrained D_s refinement.",
    )

    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--screen-progress-every", type=int, default=50)
    p.add_argument("--refine-progress-every", type=int, default=5)

    args = p.parse_args()
    validate_args(args)
    return args


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "settings.json").open("w", encoding="utf-8") as f:
        serializable = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        }
        json.dump(serializable, f, indent=2)
        f.write("\n")

    rng = np.random.default_rng(args.seed)
    evaluator = Evaluator(args)
    summary_rows: list[dict] = []

    print("Translation-isotropic circular-pattern optimal design")
    print(f"  N range        : {args.n_min}..{args.n_max}")
    print(f"  geometry       : {args.pose_geometry}")
    print(f"  branch mode    : {args.branch_mode}")
    print(
        "  bounds         : "
        f"theta=[{args.theta_min_deg:g},{args.theta_max_deg:g}], "
        f"beta=[{args.beta_min_deg:g},{args.beta_max_deg:g}], "
        f"d=[{args.d_min_mm:g},{args.d_max_mm:g}]"
    )
    print(
        "  pool exactness : "
        f"||Htt/tr(Htt)-I/3||_F <= {args.isotropy_tol:g}"
    )

    for N in range(args.n_min, args.n_max + 1):
        full_before = evaluator.full_evaluations
        trans_before = evaluator.translation_evaluations

        best, pool, diag = search_one_n(N, evaluator, rng, args)
        n_dir = args.output_dir / f"N_{N:02d}"
        n_dir.mkdir(parents=True, exist_ok=True)
        save_pool_csv(pool, n_dir / "translation_isotropic_pool.csv")
        save_pool_condition_plot(pool, n_dir / "translation_pool_condition.png")

        row = {
            "N": N,
            "subset_count": diag["subset_count"],
            "subsets_with_isotropic_roots": diag["subsets_with_roots"],
            "pool_roots_found": diag["pool_roots"],
            "refined_accepted": diag.get("refined_accepted", 0),
            "best_strict_logdet": np.nan,
            "best_rank_eff": -1,
            "best_lambda_min": np.nan,
            "best_condition_eff": np.nan,
            "best_translation_condition": np.nan,
            "best_translation_isotropy_error": np.nan,
            "best_line_ids": "",
            "full_jacobian_evaluations": evaluator.full_evaluations - full_before,
            "translation_evaluations": evaluator.translation_evaluations - trans_before,
            "elapsed_sec": diag["elapsed_sec"],
        }

        if best is not None:
            save_best_design(best, evaluator, n_dir, args)
            row.update(
                {
                    "best_strict_logdet": best.metrics.strict_logdet,
                    "best_rank_eff": best.metrics.rank_eff,
                    "best_lambda_min": best.metrics.lambda_min,
                    "best_condition_eff": best.metrics.condition,
                    "best_translation_condition": best.translation_condition,
                    "best_translation_isotropy_error": best.isotropy_error,
                    "best_line_ids": " ".join(map(str, best.line_ids)),
                }
            )
            print(
                f"[N={N}] BEST | logdet={best.metrics.strict_logdet:.8g} | "
                f"rank={best.metrics.rank_eff}/6 | "
                f"lambda_min={best.metrics.lambda_min:.6g} | "
                f"cond_eff={best.metrics.condition:.6g} | "
                f"cond_t={best.translation_condition:.9g} | "
                f"iso_err={best.isotropy_error:.3e} | "
                f"lines={best.line_ids}"
            )
        else:
            print(f"[N={N}] no accepted full-D_s design on sampled isotropy manifold")

        summary_rows.append(row)
        write_summary(summary_rows, args.output_dir)

    summary_path = write_summary(summary_rows, args.output_dir)
    print(f"\nDone. Summary: {summary_path}")
    print(
        "Interpretation: pool roots are numerical samples of the continuous "
        "translation-condition-1 manifold; best_design.csv is the best found "
        "full-Jacobian D_s design while remaining on that manifold."
    )


if __name__ == "__main__":
    main()