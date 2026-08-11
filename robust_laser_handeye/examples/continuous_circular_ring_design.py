#!/usr/bin/env python3
"""
Continuous circular-ring D_s-optimal design for single-plane 2D laser hand-eye calibration.

Purpose
-------
Keep the repository's 9-line circular target pattern fixed.
For each tilt level ("ring"):

    - view tilt theta is common to the ring,
    - center distance d is common to the ring,
    - sensor-frame plane-normal azimuth psi is uniformly distributed over [0, 2pi).

Optimize only:
    theta_k, d_k, and ring weights w_k.

Compare K = 1, 2, ..., max_k and select the smallest K that reaches the requested
D-efficiency relative to the best tested K.

This script uses the repository's parameter scaling / nuisance-plane convention:
    hand-eye right rotation  : 3
    hand-eye right translation: 3
    plane normal tangent     : 2
    plane offset             : 1

The objective is D_s-optimality:
    H_eff = H_HH - H_HP H_PP^{-1} H_PH
    score = 0.5 * log det(H_eff)

The fast FIM implementation is validated once against
laser_handeye.active_fisher.joint_residual_jacobian before optimization.

Expected location
-----------------
Save as:
    robust_laser_handeye/main/continuous_circular_ring_design.py

Run from robust_laser_handeye/:
    PYTHONPATH=. python3 main/continuous_circular_ring_design.py \
        --output-dir result_continuous_ring \
        --max-k 3 \
        --tilt-range-deg 5 80 \
        --distance-range-mm 70 130 \
        --profile-depth-range-mm 40 160 \
        --radius-mm 150 \
        --restarts 3

Notes
-----
1) theta is the user's view tilt:
       theta = angle(-Z_sensor, plane normal).

2) psi is NOT the board-frame view azimuth gamma.
   psi is the azimuth of the plane normal expressed in the sensor XY plane:
       n_s = [sin(theta) cos(psi),
              sin(theta) sin(psi),
             -cos(theta)].

3) For a fixed target line, theta + psi uniquely determine the sensor orientation
   (up to the directed-line convention used here) while forcing the target line
   to lie in the sensor X-Z laser plane.

4) The optimization is an approximate continuous design. The returned weights
   are ideal proportions, not integer scan counts. Convert them to exact N-scan
   designs in the next finite-design stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from laser_handeye.active_fisher import (
    PlaneEstimate,
    joint_residual_jacobian,
    plane_tangent_basis,
)
from laser_handeye.patterns import circular_lines
from laser_handeye.se3 import make_T
from laser_handeye.simulation import simulate_profile_on_plane


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingSupport:
    theta_deg: float
    distance_mm: float
    weight: float


@dataclass(frozen=True)
class DesignResult:
    k: int
    valid: bool
    half_logdet: float
    logdet: float
    min_eig: float
    max_eig: float
    condition_number: float
    supports: tuple[RingSupport, ...]
    optimizer_fun: float
    optimizer_nfev: int
    optimizer_nit: int
    seed: int
    message: str


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lo, hi = map(float, values)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"invalid {name}: {values}")
    return lo, hi


def _softmax_with_zero_reference(raw: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.ones(1, dtype=float)
    raw = np.asarray(raw, dtype=float).reshape(k - 1)
    logits = np.concatenate([raw, np.zeros(1, dtype=float)])
    logits -= float(np.max(logits))
    values = np.exp(logits)
    return values / float(np.sum(values))


def _parameter_scales(args: argparse.Namespace) -> np.ndarray:
    return np.array(
        [
            *([np.deg2rad(float(args.rotation_scale_deg))] * 3),
            *([float(args.translation_scale_mm)] * 3),
            *([np.deg2rad(float(args.plane_normal_scale_deg))] * 2),
            float(args.plane_offset_scale_mm),
        ],
        dtype=float,
    )


def _normal_sensor(theta_rad: float, psi_rad: float) -> np.ndarray:
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    return np.array(
        [s * math.cos(psi_rad), s * math.sin(psi_rad), -c],
        dtype=float,
    )


def _line_sensor_from_normal(n_s: np.ndarray) -> np.ndarray:
    """
    Directed unit target-line vector in the sensor X-Z laser plane.

    Constraints:
        L_s[1] = 0
        L_s dot n_s = 0
        ||L_s|| = 1

    We choose the branch with positive sensor-X component.
    """
    n_s = np.asarray(n_s, dtype=float).reshape(3)
    raw = np.array([-n_s[2], 0.0, n_s[0]], dtype=float)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate line direction; theta is too close to 90 deg")
    line_s = raw / norm
    if line_s[0] < 0.0:
        line_s = -line_s
    return line_s


def sensor_pose_from_target_line_view(
    *,
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    distance_mm: float,
    theta_deg: float,
    psi_deg: float,
) -> np.ndarray:
    """
    Construct T_base_sensor from the user's (theta, psi, d) geometry.

    theta:
        angle(-Z_sensor, plane normal).

    psi:
        sensor-frame azimuth of the plane normal:
            atan2(n_s_y, n_s_x).

    distance_mm:
        target-line midpoint is [0, 0, d] in the sensor frame.

    The target line is forced into the sensor X-Z laser plane.
    """
    plane_R = np.asarray(plane_R, dtype=float).reshape(3, 3)
    plane_t = np.asarray(plane_t, dtype=float).reshape(3)
    p0 = np.asarray(line_p0, dtype=float).reshape(2)
    p1 = np.asarray(line_p1, dtype=float).reshape(2)

    if not np.isfinite(distance_mm) or distance_mm <= 0.0:
        raise ValueError("distance_mm must be positive and finite")

    theta = math.radians(float(theta_deg))
    psi = math.radians(float(psi_deg))
    if not (0.0 <= theta < 0.5 * math.pi):
        raise ValueError("theta must lie in [0, 90) deg")

    q0 = plane_t + plane_R @ np.array([p0[0], p0[1], 0.0], dtype=float)
    q1 = plane_t + plane_R @ np.array([p1[0], p1[1], 0.0], dtype=float)
    q_mid = 0.5 * (q0 + q1)

    line_base = q1 - q0
    line_base /= np.linalg.norm(line_base)
    normal_base = plane_R[:, 2].copy()
    normal_base /= np.linalg.norm(normal_base)

    n_s = _normal_sensor(theta, psi)
    line_s = _line_sensor_from_normal(n_s)

    cross_s = np.cross(line_s, n_s)
    cross_s /= np.linalg.norm(cross_s)
    cross_base = np.cross(line_base, normal_base)
    cross_base /= np.linalg.norm(cross_base)

    basis_sensor = np.column_stack([line_s, n_s, cross_s])
    basis_base = np.column_stack([line_base, normal_base, cross_base])
    R_base_s = basis_base @ basis_sensor.T

    if not np.isclose(np.linalg.det(R_base_s), 1.0, atol=1e-9):
        raise RuntimeError("constructed sensor rotation is not right-handed")

    # Midpoint is [0, 0, d] in the sensor frame.
    t_base_s = q_mid - float(distance_mm) * R_base_s[:, 2]

    # Strong geometry checks.
    realized_n_s = R_base_s.T @ normal_base
    if not np.allclose(realized_n_s, n_s, atol=1e-9):
        raise RuntimeError("normal-sensor construction check failed")

    realized_line_s = R_base_s.T @ line_base
    if abs(float(realized_line_s[1])) > 1e-9:
        raise RuntimeError("target line is not in sensor X-Z plane")
    if abs(float(realized_line_s @ realized_n_s)) > 1e-9:
        raise RuntimeError("target line is not tangent to calibration plane")

    return make_T(R_base_s, t_base_s)


# ---------------------------------------------------------------------------
# Fast per-scan / per-ring FIM
# ---------------------------------------------------------------------------

class ContinuousRingProblem:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tilt_range = _finite_pair(args.tilt_range_deg, "tilt range")
        self.distance_range = _finite_pair(
            args.distance_range_mm, "distance range"
        )
        self.depth_range = (
            None
            if args.profile_depth_range_mm is None
            else _finite_pair(args.profile_depth_range_mm, "profile depth range")
        )
        self.x_values = np.linspace(
            -float(args.profile_half_width_mm),
            float(args.profile_half_width_mm),
            int(args.profile_points),
            dtype=float,
        )
        self.scales = _parameter_scales(args)
        self.plane_R = np.eye(3, dtype=float)
        self.plane_t = np.array(
            [0.0, 0.0, float(args.plane_z_mm)], dtype=float
        )
        self.plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
        self.plane_offset = float(args.plane_z_mm)
        self.tangent_u, self.tangent_v = plane_tangent_basis(
            self.plane_normal
        )
        self.lines = circular_lines(
            float(args.radius_mm),
            n_lines=int(args.n_lines),
        )
        self.psi_values = np.linspace(
            0.0,
            2.0 * np.pi,
            int(args.azimuth_samples),
            endpoint=False,
            dtype=float,
        )
        self._line_geometry = self._precompute_line_geometry()

    def _precompute_line_geometry(self) -> list[tuple[np.ndarray, np.ndarray]]:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for p0, p1 in self.lines:
            p0 = np.asarray(p0, dtype=float).reshape(2)
            p1 = np.asarray(p1, dtype=float).reshape(2)
            q0 = self.plane_t + self.plane_R @ np.array(
                [p0[0], p0[1], 0.0], dtype=float
            )
            q1 = self.plane_t + self.plane_R @ np.array(
                [p1[0], p1[1], 0.0], dtype=float
            )
            q_mid = 0.5 * (q0 + q1)
            line_base = q1 - q0
            line_base /= np.linalg.norm(line_base)
            result.append((q_mid, line_base))
        return result

    def ring_is_feasible(self, theta_deg: float, distance_mm: float) -> bool:
        if (
            not np.isfinite(theta_deg)
            or not np.isfinite(distance_mm)
            or theta_deg < self.tilt_range[0]
            or theta_deg > self.tilt_range[1]
            or distance_mm < self.distance_range[0]
            or distance_mm > self.distance_range[1]
        ):
            return False

        theta = math.radians(float(theta_deg))
        c = math.cos(theta)
        if c <= 1e-8:
            return False

        # A true uniform psi ring includes |cos(psi)| = 1, so use the
        # worst-case profile-depth swing rather than only sampled psi values.
        max_depth_swing = (
            float(np.max(np.abs(self.x_values))) * math.tan(theta)
        )
        if self.depth_range is not None:
            z_min, z_max = self.depth_range
            if distance_mm - max_depth_swing < z_min - 1e-10:
                return False
            if distance_mm + max_depth_swing > z_max + 1e-10:
                return False

        if self.args.enforce_target_segment:
            # At |cos psi|=1, ds/dx along the target line is sec(theta).
            max_line_half_extent = (
                float(np.max(np.abs(self.x_values))) / c
            )
            # circular_lines uses [center -> circumference], so the midpoint
            # has radius/2 available in either direction.
            if max_line_half_extent > 0.5 * float(self.args.radius_mm) + 1e-10:
                return False

        return True

    def _residual_sigma(self, n_s: np.ndarray) -> float:
        sigma = float(self.args.profile_noise_std_mm)
        if self.args.noise_axis == "xz":
            projection = float(np.linalg.norm(n_s[[0, 2]]))
        elif self.args.noise_axis == "z":
            projection = abs(float(n_s[2]))
        else:
            raise ValueError("noise_axis must be 'xz' or 'z'")
        if projection <= 1e-12:
            raise ValueError("zero residual noise projection")
        return sigma * projection

    def scan_information_fast(
        self,
        *,
        theta_deg: float,
        distance_mm: float,
        psi_rad: float,
        line_id: int,
    ) -> np.ndarray:
        theta = math.radians(float(theta_deg))
        s = math.sin(theta)
        c = math.cos(theta)
        cp = math.cos(float(psi_rad))

        n_s = _normal_sensor(theta, float(psi_rad))
        residual_sigma = self._residual_sigma(n_s)

        # Because [0,0,d] is on the calibration plane in sensor coordinates:
        #   n_x x + n_z z + c0 = 0
        # -> z(x) = d + tan(theta) cos(psi) x
        z_values = (
            float(distance_mm)
            + math.tan(theta) * cp * self.x_values
        )
        points_s = np.column_stack(
            [
                self.x_values,
                np.zeros_like(self.x_values),
                z_values,
            ]
        )

        q_mid, line_base = self._line_geometry[int(line_id)]

        # Sensor target-line direction and conversion from sensor x coordinate
        # to signed distance along the physical target line.
        line_s = _line_sensor_from_normal(n_s)
        if line_s[0] <= 1e-12:
            raise ValueError("degenerate target-line X component")
        alpha = self.x_values / float(line_s[0])
        points_base = q_mid[None, :] + alpha[:, None] * line_base[None, :]

        dimension = 9
        J_phys = np.zeros((len(self.x_values), dimension), dtype=float)

        # Right-local hand-eye rotation: p_s x n_s.
        J_phys[:, 0:3] = np.cross(points_s, n_s[None, :])
        # Right-local hand-eye translation.
        J_phys[:, 3:6] = n_s[None, :]
        # Plane-normal tangent coordinates, matching active_fisher.py.
        tangent_matrix = np.column_stack([self.tangent_u, self.tangent_v])
        J_phys[:, 6:8] = points_base @ tangent_matrix
        # Plane offset.
        J_phys[:, 8] = -1.0

        # At the design point residual=0, the derivative of the whitening gain
        # does not contribute. This matches predicted-candidate Fisher in repo.
        J = (J_phys / residual_sigma) * self.scales[None, :]
        H = J.T @ J
        return 0.5 * (H + H.T)

    def ring_information(
        self,
        theta_deg: float,
        distance_mm: float,
    ) -> np.ndarray | None:
        if not self.ring_is_feasible(theta_deg, distance_mm):
            return None

        H = np.zeros((9, 9), dtype=float)
        count = 0
        for line_id in range(len(self.lines)):
            for psi in self.psi_values:
                H += self.scan_information_fast(
                    theta_deg=theta_deg,
                    distance_mm=distance_mm,
                    psi_rad=float(psi),
                    line_id=line_id,
                )
                count += 1
        if count == 0:
            return None
        H /= float(count)  # per-scan average information for this ring
        return 0.5 * (H + H.T)

    @staticmethod
    def marginal_handeye_information(
        joint_information: np.ndarray,
        *,
        eig_tol: float = 1e-11,
    ) -> np.ndarray | None:
        H = 0.5 * (
            np.asarray(joint_information, dtype=float)
            + np.asarray(joint_information, dtype=float).T
        )
        if H.shape != (9, 9):
            raise ValueError("single-plane joint information must be 9x9")

        H_pp = H[6:, 6:]
        H_hp = H[:6, 6:]
        evals_pp = np.linalg.eigvalsh(H_pp)
        pp_scale = max(float(evals_pp[-1]), 1.0)
        if float(evals_pp[0]) <= eig_tol * pp_scale:
            return None

        solved = np.linalg.solve(H_pp, H_hp.T)
        H_eff = H[:6, :6] - H_hp @ solved
        return 0.5 * (H_eff + H_eff.T)

    def score_information(
        self,
        joint_information: np.ndarray,
    ) -> tuple[float, np.ndarray] | None:
        H_eff = self.marginal_handeye_information(joint_information)
        if H_eff is None:
            return None

        eigvals = np.linalg.eigvalsh(H_eff)
        scale = max(float(eigvals[-1]), 1.0)
        if float(eigvals[0]) <= float(self.args.eig_tol) * scale:
            return None

        if self.args.objective == "d_optimal":
            sign, logdet = np.linalg.slogdet(H_eff)
            if sign <= 0.0:
                return None
            score = 0.5 * float(logdet)
        elif self.args.objective == "e_optimal":
            score = float(eigvals[0])
        else:
            raise ValueError(f"unsupported objective: {self.args.objective}")

        return score, H_eff

    def decode_vector(
        self,
        vector: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector = np.asarray(vector, dtype=float).reshape(3 * k - 1)
        theta = vector[:k]
        distance = vector[k : 2 * k]
        weights = _softmax_with_zero_reference(vector[2 * k :], k)
        return theta, distance, weights

    def joint_information_from_vector(
        self,
        vector: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        theta, distance, weights = self.decode_vector(vector, k)

        H = np.zeros((9, 9), dtype=float)
        for th, d, w in zip(theta, distance, weights):
            H_ring = self.ring_information(float(th), float(d))
            if H_ring is None:
                return None
            H += float(w) * H_ring

        return H, theta, distance, weights

    def objective_function(self, vector: np.ndarray, k: int) -> float:
        decoded = self.joint_information_from_vector(vector, k)
        if decoded is None:
            return 1e12
        H, _theta, _distance, _weights = decoded
        scored = self.score_information(H)
        if scored is None:
            return 1e12
        score, _ = scored
        return -float(score)

    def bounds_for_k(self, k: int) -> list[tuple[float, float]]:
        return (
            [self.tilt_range] * k
            + [self.distance_range] * k
            + [(-8.0, 8.0)] * (k - 1)
        )

    def validate_fast_fim_against_repo(self) -> None:
        """
        One automatic regression check against the repository implementation.

        This checks:
            custom (theta, psi, d) pose construction
            fast analytic zero-residual Jacobian
            parameter scale convention
            noise whitening
            plane tangent convention
        """
        theta_test = 0.5 * (self.tilt_range[0] + self.tilt_range[1])
        d_test = 0.5 * (self.distance_range[0] + self.distance_range[1])

        # Move to a feasible point if the midpoint violates depth/segment limits.
        candidates_theta = np.linspace(
            self.tilt_range[0],
            self.tilt_range[1],
            17,
        )
        candidates_d = np.linspace(
            self.distance_range[0],
            self.distance_range[1],
            17,
        )
        found = False
        for th in candidates_theta:
            for d in candidates_d:
                if self.ring_is_feasible(float(th), float(d)):
                    theta_test = float(th)
                    d_test = float(d)
                    found = True
                    break
            if found:
                break
        if not found:
            raise SystemExit(
                "No feasible (theta, d) exists for the requested ranges. "
                "Check depth ROI / target segment / profile width."
            )

        psi = 0.731
        line_id = min(1, len(self.lines) - 1)
        p0, p1 = self.lines[line_id]

        T_base_s = sensor_pose_from_target_line_view(
            plane_R=self.plane_R,
            plane_t=self.plane_t,
            line_p0=p0,
            line_p1=p1,
            distance_mm=d_test,
            theta_deg=theta_test,
            psi_deg=math.degrees(psi),
        )

        # Identity hand-eye is enough because the right-local Fisher geometry
        # is evaluated at the commanded sensor pose.
        T_ef_s = np.eye(4, dtype=float)
        T_base_ef = T_base_s.copy()

        scan = simulate_profile_on_plane(
            T_base_ef=T_base_ef,
            T_ef_s_true=T_ef_s,
            plane_n=self.plane_normal,
            plane_l=self.plane_offset,
            x_values=self.x_values,
            noise_std=0.0,
            rng=np.random.default_rng(0),
            plane_id=0,
            scan_id=0,
        )

        plane = PlaneEstimate(
            normal_base=self.plane_normal,
            offset_mm=self.plane_offset,
        )
        residual, J_repo = joint_residual_jacobian(
            {0: [scan]},
            T_ef_s,
            {0: plane},
            profile_noise_std_mm=float(self.args.profile_noise_std_mm),
            noise_axis=str(self.args.noise_axis),
            parameter_scales=self.scales,
        )
        if float(np.max(np.abs(residual))) > 1e-9:
            raise RuntimeError(
                "repo Jacobian validation expected zero residual at GT"
            )

        H_repo = J_repo.T @ J_repo
        H_fast = self.scan_information_fast(
            theta_deg=theta_test,
            distance_mm=d_test,
            psi_rad=psi,
            line_id=line_id,
        )

        denom = max(float(np.linalg.norm(H_repo, ord="fro")), 1.0)
        rel_error = float(
            np.linalg.norm(H_repo - H_fast, ord="fro") / denom
        )

        # Also check profile points themselves.
        z_fast = (
            d_test
            + math.tan(math.radians(theta_test))
            * math.cos(psi)
            * self.x_values
        )
        max_profile_error = float(
            np.max(np.abs(scan.valid_points_s[:, 2] - z_fast))
        )

        print(
            "[validation] "
            f"theta={theta_test:.6f} deg, d={d_test:.6f} mm, "
            f"FIM relative error={rel_error:.3e}, "
            f"profile max error={max_profile_error:.3e} mm"
        )

        if rel_error > float(self.args.validation_rtol):
            raise RuntimeError(
                "Fast analytic FIM does not match repository Jacobian: "
                f"relative error={rel_error:.3e} > "
                f"{self.args.validation_rtol:.3e}"
            )
        if max_profile_error > 1e-8:
            raise RuntimeError(
                "Fast profile formula does not match repository simulator"
            )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize_one_k(
    problem: ContinuousRingProblem,
    k: int,
    *,
    seed: int,
) -> DesignResult:
    best_opt = None

    for restart in range(int(problem.args.restarts)):
        run_seed = int(seed + 1009 * restart + 7919 * k)
        result = differential_evolution(
            lambda x: problem.objective_function(x, k),
            bounds=problem.bounds_for_k(k),
            strategy="best1bin",
            maxiter=int(problem.args.maxiter),
            popsize=int(problem.args.popsize),
            tol=float(problem.args.de_tol),
            atol=float(problem.args.de_atol),
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=run_seed,
            polish=bool(problem.args.polish),
            updating="immediate",
            workers=1,
            disp=bool(problem.args.verbose),
        )
        if best_opt is None or float(result.fun) < float(best_opt.fun):
            best_opt = result

        print(
            f"[K={k}] restart {restart + 1}/{problem.args.restarts} "
            f"seed={run_seed} "
            f"best_half_logdet={-float(result.fun):.9g} "
            f"nfev={result.nfev}"
        )

    assert best_opt is not None

    decoded = problem.joint_information_from_vector(best_opt.x, k)
    if decoded is None:
        return DesignResult(
            k=k,
            valid=False,
            half_logdet=float("-inf"),
            logdet=float("-inf"),
            min_eig=0.0,
            max_eig=0.0,
            condition_number=float("inf"),
            supports=tuple(),
            optimizer_fun=float(best_opt.fun),
            optimizer_nfev=int(best_opt.nfev),
            optimizer_nit=int(best_opt.nit),
            seed=int(seed),
            message="best optimizer point is infeasible",
        )

    H, theta, distance, weights = decoded
    scored = problem.score_information(H)
    if scored is None:
        return DesignResult(
            k=k,
            valid=False,
            half_logdet=float("-inf"),
            logdet=float("-inf"),
            min_eig=0.0,
            max_eig=0.0,
            condition_number=float("inf"),
            supports=tuple(),
            optimizer_fun=float(best_opt.fun),
            optimizer_nfev=int(best_opt.nfev),
            optimizer_nit=int(best_opt.nit),
            seed=int(seed),
            message="marginal hand-eye information is rank deficient",
        )

    score, H_eff = scored
    eigvals = np.linalg.eigvalsh(H_eff)
    supports = [
        RingSupport(
            theta_deg=float(th),
            distance_mm=float(d),
            weight=float(w),
        )
        for th, d, w in zip(theta, distance, weights)
    ]
    supports.sort(key=lambda item: item.theta_deg)

    return DesignResult(
        k=k,
        valid=True,
        half_logdet=float(score),
        logdet=2.0 * float(score),
        min_eig=float(eigvals[0]),
        max_eig=float(eigvals[-1]),
        condition_number=float(eigvals[-1] / eigvals[0]),
        supports=tuple(supports),
        optimizer_fun=float(best_opt.fun),
        optimizer_nfev=int(best_opt.nfev),
        optimizer_nit=int(best_opt.nit),
        seed=int(seed),
        message=str(best_opt.message),
    )


def d_efficiency(result: DesignResult, best: DesignResult) -> float:
    """
    D-efficiency for 6 hand-eye parameters.

    result.logdet = log det(H_eff), so:
        eff = exp((logdet - best_logdet) / 6)
    """
    if not result.valid or not best.valid:
        return 0.0
    return float(np.exp((result.logdet - best.logdet) / 6.0))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_summary(
    results: Sequence[DesignResult],
    best: DesignResult,
    chosen: DesignResult,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "continuous_ring_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "K",
            "valid",
            "half_logdet",
            "logdet",
            "d_efficiency_vs_best",
            "min_eig",
            "max_eig",
            "condition_number",
            "theta_deg",
            "distance_mm",
            "weight",
            "optimizer_nfev",
            "optimizer_nit",
            "message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            efficiency = d_efficiency(result, best)
            if result.supports:
                for support in result.supports:
                    writer.writerow(
                        {
                            "K": result.k,
                            "valid": result.valid,
                            "half_logdet": result.half_logdet,
                            "logdet": result.logdet,
                            "d_efficiency_vs_best": efficiency,
                            "min_eig": result.min_eig,
                            "max_eig": result.max_eig,
                            "condition_number": result.condition_number,
                            "theta_deg": support.theta_deg,
                            "distance_mm": support.distance_mm,
                            "weight": support.weight,
                            "optimizer_nfev": result.optimizer_nfev,
                            "optimizer_nit": result.optimizer_nit,
                            "message": result.message,
                        }
                    )
            else:
                writer.writerow(
                    {
                        "K": result.k,
                        "valid": result.valid,
                        "half_logdet": result.half_logdet,
                        "logdet": result.logdet,
                        "d_efficiency_vs_best": efficiency,
                        "min_eig": result.min_eig,
                        "max_eig": result.max_eig,
                        "condition_number": result.condition_number,
                        "theta_deg": "",
                        "distance_mm": "",
                        "weight": "",
                        "optimizer_nfev": result.optimizer_nfev,
                        "optimizer_nit": result.optimizer_nit,
                        "message": result.message,
                    }
                )

    payload = {
        "best_tested_K": int(best.k),
        "chosen_minimum_K": int(chosen.k),
        "chosen_efficiency_threshold": float(
            chosen.k and 0.0  # overwritten below by caller metadata if needed
        ),
        "results": [
            {
                "K": int(result.k),
                "valid": bool(result.valid),
                "half_logdet": float(result.half_logdet),
                "logdet": float(result.logdet),
                "d_efficiency_vs_best": d_efficiency(result, best),
                "min_eig": float(result.min_eig),
                "max_eig": float(result.max_eig),
                "condition_number": float(result.condition_number),
                "supports": [
                    {
                        "theta_deg": float(s.theta_deg),
                        "distance_mm": float(s.distance_mm),
                        "weight": float(s.weight),
                    }
                    for s in result.supports
                ],
                "optimizer_nfev": int(result.optimizer_nfev),
                "optimizer_nit": int(result.optimizer_nit),
                "message": result.message,
            }
            for result in results
        ],
    }
    with (out_dir / "continuous_ring_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_run_config(args: argparse.Namespace, out_dir: Path) -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def print_result(result: DesignResult, best: DesignResult | None = None) -> None:
    print()
    print("=" * 78)
    print(f"K = {result.k}")
    print("=" * 78)
    if not result.valid:
        print("INVALID / rank deficient")
        print(result.message)
        return

    print(f"0.5 logdet(H_eff) : {result.half_logdet:.9g}")
    print(f"logdet(H_eff)     : {result.logdet:.9g}")
    print(f"lambda_min        : {result.min_eig:.9g}")
    print(f"lambda_max        : {result.max_eig:.9g}")
    print(f"condition number  : {result.condition_number:.9g}")
    if best is not None:
        print(f"D-eff vs best     : {100.0*d_efficiency(result, best):.4f}%")
    print("supports:")
    for idx, support in enumerate(result.supports):
        print(
            f"  [{idx}] theta={support.theta_deg:10.6f} deg, "
            f"d={support.distance_mm:10.6f} mm, "
            f"w={support.weight:10.6f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Continuous D_s-optimal circular-ring design: optimize tilt, "
            "distance, and ring weights while enforcing uniform sensor-normal "
            "azimuth within each tilt ring."
        )
    )

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=_nonnegative_int, default=17)
    p.add_argument("--max-k", type=_positive_int, default=3)
    p.add_argument(
        "--efficiency-threshold",
        type=float,
        default=0.99,
        help=(
            "Choose the smallest K whose 6D D-efficiency is at least this "
            "fraction of the best tested K."
        ),
    )

    p.add_argument(
        "--tilt-range-deg",
        type=float,
        nargs=2,
        default=(5.0, 55.0),
    )
    p.add_argument(
        "--distance-range-mm",
        type=float,
        nargs=2,
        default=(70.0, 130.0),
    )

    p.add_argument("--n-lines", type=_positive_int, default=9)
    p.add_argument("--radius-mm", type=float, default=150.0)
    p.add_argument("--plane-z-mm", type=float, default=500.0)

    p.add_argument("--profile-points", type=_positive_int, default=51)
    p.add_argument("--profile-half-width-mm", type=float, default=25.0)
    p.add_argument(
        "--profile-depth-range-mm",
        type=float,
        nargs=2,
        default=None,
        help=(
            "Optional physical sensor-Z ROI. Example: --profile-depth-range-mm "
            "40 160. If omitted, depth ROI is not used as a design constraint."
        ),
    )
    p.add_argument(
        "--enforce-target-segment",
        action="store_true",
        help=(
            "Require the complete simulated profile to remain inside each "
            "finite center-to-circumference target-line segment."
        ),
    )

    p.add_argument(
        "--azimuth-samples",
        type=_positive_int,
        default=12,
        help=(
            "Quadrature samples for the uniform sensor-normal azimuth psi "
            "inside each ring."
        ),
    )

    p.add_argument(
        "--objective",
        choices=("d_optimal", "e_optimal"),
        default="d_optimal",
    )
    p.add_argument("--profile-noise-std-mm", type=float, default=0.20)
    p.add_argument("--noise-axis", choices=("xz", "z"), default="xz")

    # Match the repository's Fisher nondimensionalization defaults.
    p.add_argument("--rotation-scale-deg", type=float, default=2.0)
    p.add_argument("--translation-scale-mm", type=float, default=10.0)
    p.add_argument("--plane-normal-scale-deg", type=float, default=20.0)
    p.add_argument("--plane-offset-scale-mm", type=float, default=100.0)

    p.add_argument("--restarts", type=_positive_int, default=3)
    p.add_argument("--maxiter", type=_positive_int, default=80)
    p.add_argument("--popsize", type=_positive_int, default=10)
    p.add_argument("--de-tol", type=float, default=1e-6)
    p.add_argument("--de-atol", type=float, default=1e-8)
    p.add_argument("--polish", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--eig-tol", type=float, default=1e-10)
    p.add_argument("--validation-rtol", type=float, default=1e-9)
    p.add_argument(
        "--skip-repo-validation",
        action="store_true",
        help="Skip the one-pose fast-FIM vs repository-Jacobian regression check.",
    )

    return p


def validate_args(args: argparse.Namespace) -> None:
    tilt = _finite_pair(args.tilt_range_deg, "tilt range")
    dist = _finite_pair(args.distance_range_mm, "distance range")

    if tilt[0] < 0.0 or tilt[1] >= 89.0:
        raise SystemExit("--tilt-range-deg must lie inside [0, 89)")
    if dist[0] <= 0.0:
        raise SystemExit("--distance-range-mm must be positive")
    if args.radius_mm <= 0.0:
        raise SystemExit("--radius-mm must be positive")
    if args.profile_half_width_mm <= 0.0:
        raise SystemExit("--profile-half-width-mm must be positive")
    if args.profile_noise_std_mm <= 0.0:
        raise SystemExit("--profile-noise-std-mm must be positive")
    if args.azimuth_samples < 3:
        raise SystemExit("--azimuth-samples must be >= 3")
    if not (0.0 < args.efficiency_threshold <= 1.0):
        raise SystemExit("--efficiency-threshold must be in (0, 1]")
    if args.max_k > 5:
        raise SystemExit(
            "--max-k > 5 is intentionally blocked. Start with K<=3; "
            "increase only if K=3 is still materially improving."
        )

    for name in (
        "rotation_scale_deg",
        "translation_scale_mm",
        "plane_normal_scale_deg",
        "plane_offset_scale_mm",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, out_dir)

    problem = ContinuousRingProblem(args)

    if not args.skip_repo_validation:
        problem.validate_fast_fim_against_repo()

    results: list[DesignResult] = []
    for k in range(1, int(args.max_k) + 1):
        print()
        print("#" * 78)
        print(f"Optimizing K={k} tilt ring(s)")
        print("#" * 78)
        result = optimize_one_k(
            problem,
            k,
            seed=int(args.seed),
        )
        results.append(result)
        print_result(result)

    valid_results = [r for r in results if r.valid]
    if not valid_results:
        raise SystemExit(
            "No tested K produced positive-definite marginal hand-eye information."
        )

    best = max(
        valid_results,
        key=lambda result: (
            result.half_logdet
            if args.objective == "d_optimal"
            else result.min_eig
        ),
    )

    if args.objective == "d_optimal":
        chosen_candidates = [
            r
            for r in valid_results
            if d_efficiency(r, best) >= float(args.efficiency_threshold)
        ]
    else:
        # For E-optimality use a direct weakest-eigenvalue ratio.
        chosen_candidates = [
            r
            for r in valid_results
            if r.min_eig / best.min_eig >= float(args.efficiency_threshold)
        ]

    chosen = min(chosen_candidates, key=lambda r: r.k)

    print()
    print("\nFINAL COMPARISON")
    print("=" * 78)
    for result in results:
        print_result(result, best=best)

    print()
    print("=" * 78)
    print(f"Best tested K      : {best.k}")
    print(
        f"Chosen minimum K   : {chosen.k} "
        f"(threshold={100.0*args.efficiency_threshold:.2f}%)"
    )
    if args.objective == "d_optimal":
        print(
            f"Chosen D-efficiency: "
            f"{100.0*d_efficiency(chosen, best):.4f}%"
        )
    else:
        print(
            f"Chosen E-efficiency: "
            f"{100.0*chosen.min_eig/best.min_eig:.4f}%"
        )
    print("=" * 78)

    save_summary(results, best, chosen, out_dir)

    # Patch the threshold into JSON metadata.
    results_json = out_dir / "continuous_ring_results.json"
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    payload["chosen_efficiency_threshold"] = float(args.efficiency_threshold)
    payload["objective"] = str(args.objective)
    payload["chosen_supports"] = [
        {
            "theta_deg": float(s.theta_deg),
            "distance_mm": float(s.distance_mm),
            "weight": float(s.weight),
        }
        for s in chosen.supports
    ]
    results_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {out_dir / 'continuous_ring_summary.csv'}")
    print(f"Saved: {out_dir / 'continuous_ring_results.json'}")
    print(f"Saved: {out_dir / 'run_config.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())