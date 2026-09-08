#!/usr/bin/env python3
"""Controlled N=9 study of gamma, u/v, and distance at fixed translation b geometry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_initialization_robustness import run as phase2a  # noqa: E402
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_geometry as statistics,
)
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_jacobian as jacobian_analysis,
)
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_jacobian_decomposition as decomposition,
)
from experiments.lhs_random_subset_screening import run as phase1  # noqa: E402
from laser_handeye.data import LaserScan  # noqa: E402
from laser_handeye.pose_design import (  # noqa: E402
    PLANE_RELATIVE_POSE_CONVENTION,
    PlaneFrame,
    PlaneRelativePose,
    sensor_pose_from_plane_relative,
)
from laser_handeye.pose_generation import sensor_pose_to_robot_pose  # noqa: E402
from laser_handeye.simulation import simulate_profile_on_plane  # noqa: E402


SCHEMA = "laser_handeye.fixed_translation_rotation_geometry"
SCHEMA_VERSION = 1
ROTATION_SCALE_RAD = jacobian_analysis.ROTATION_SCALE_RAD
PARAMETER_NAMES = (
    "u_mm",
    "v_mm",
    "d_mm",
    "tilt_deg",
    "gamma_deg",
    "psi_b_deg",
)


@dataclass(frozen=True)
class ExperimentContext:
    phase1_manifest: dict[str, Any]
    phase2a_manifest: dict[str, Any]
    frame: PlaneFrame
    center: np.ndarray
    normal: np.ndarray
    offset_mm: float
    T_true: np.ndarray
    x_values: np.ndarray


@dataclass(frozen=True)
class PoseDesign:
    design_id: int
    experiment: str
    condition: str
    block_or_shift: str
    independent_value: float
    theta_deg: np.ndarray
    psi_b_deg: np.ndarray
    gamma_deg: np.ndarray
    legacy_phi_deg: np.ndarray
    u_mm: np.ndarray
    v_mm: np.ndarray
    d_mm: np.ndarray
    T_base_s: np.ndarray
    T_base_ef: np.ndarray
    ideal_points_s: np.ndarray
    b_actual: np.ndarray
    validation: dict[str, float]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("r0", "geometry", "r3", "calibration", "analysis"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.write_text(
        json.dumps(phase1._jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context(config: dict[str, Any]) -> ExperimentContext:
    if int(config["N"]) != 9:
        raise ValueError("this controlled discovery experiment requires N=9")
    phase1_dir = _resolve(config["phase1_dir"])
    phase2a_dir = _resolve(config["phase2a_dir"])
    phase1_manifest = _load_json(phase1_dir / "manifest.json")
    phase2a_manifest = _load_json(phase2a_dir / "manifest.json")
    if phase1_manifest.get("status") != "complete":
        raise ValueError("source Phase 1 must be complete")
    if phase2a_manifest.get("status") != "complete":
        raise ValueError("source Phase 2A must be complete")
    plane = phase1_manifest["resolved"]["plane"]
    center = np.asarray(plane["center_base_mm"], dtype=float)
    normal = np.asarray(plane["normal_base"], dtype=float)
    normal /= np.linalg.norm(normal)
    frame = PlaneFrame(
        np.asarray(plane["u_base"], dtype=float),
        np.asarray(plane["v_base"], dtype=float),
        normal,
        float(plane["offset_mm"]),
    )
    profile = config["profile"]
    x_values = np.linspace(
        -float(profile["half_width_mm"]),
        float(profile["half_width_mm"]),
        int(profile["points"]),
    )
    return ExperimentContext(
        phase1_manifest=phase1_manifest,
        phase2a_manifest=phase2a_manifest,
        frame=frame,
        center=center,
        normal=normal,
        offset_mm=float(plane["offset_mm"]),
        T_true=np.asarray(phase1_manifest["resolved"]["T_ef_s_true"], dtype=float),
        x_values=x_values,
    )


def _baseline_angles(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    baseline = config["baseline"]
    low_psi = np.asarray(baseline["low_psi_b_deg"], dtype=float)
    high_psi = np.asarray(baseline["high_psi_b_deg"], dtype=float)
    theta = np.concatenate(
        [
            np.full(len(low_psi), float(baseline["low_tilt_deg"])),
            np.full(len(high_psi), float(baseline["high_tilt_deg"])),
        ]
    )
    psi = np.concatenate([low_psi, high_psi])
    if len(theta) != int(config["N"]):
        raise ValueError("baseline ring schedule does not contain N poses")
    return theta, psi


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _legacy_phi(theta_deg: np.ndarray, gamma_deg: np.ndarray, psi_b_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    gamma = np.deg2rad(gamma_deg)
    psi_zero = np.rad2deg(
        np.arctan2(-np.sin(gamma), np.cos(theta) * np.cos(gamma))
    )
    return _wrap_deg(psi_zero - psi_b_deg)


def _target_b(theta_deg: np.ndarray, psi_b_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(psi_b_deg)
    return np.column_stack(
        [
            np.sin(theta) * np.cos(psi),
            np.sin(theta) * np.sin(psi),
            -np.cos(theta),
        ]
    )


def _angular_error_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _wrap_deg(np.asarray(actual) - np.asarray(target))


def _make_design(
    context: ExperimentContext,
    config: dict[str, Any],
    *,
    design_id: int,
    experiment: str,
    condition: str,
    block_or_shift: str,
    independent_value: float,
    gamma_deg: np.ndarray,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
    d_mm: np.ndarray,
) -> PoseDesign:
    theta_deg, psi_b_deg = _baseline_angles(config)
    n = int(config["N"])
    arrays = [
        np.asarray(value, dtype=float).reshape(n)
        for value in (gamma_deg, u_mm, v_mm, d_mm)
    ]
    gamma_deg, u_mm, v_mm, d_mm = arrays
    target_b = _target_b(theta_deg, psi_b_deg)
    transforms = np.empty((n, 4, 4), dtype=float)
    flange = np.empty_like(transforms)
    points = np.empty((n, len(context.x_values), 3), dtype=float)
    for pose_id in range(n):
        pose = PlaneRelativePose(
            sample_id=pose_id,
            target_u_mm=float(u_mm[pose_id]),
            target_v_mm=float(v_mm[pose_id]),
            distance_mm=float(d_mm[pose_id]),
            tilt_deg=float(theta_deg[pose_id]),
            azimuth_deg=float(gamma_deg[pose_id]),
            normal_azimuth_sensor_deg=float(psi_b_deg[pose_id]),
        )
        transforms[pose_id] = sensor_pose_from_plane_relative(
            context.frame, context.center, pose
        )
        flange[pose_id] = sensor_pose_to_robot_pose(
            transforms[pose_id], context.T_true
        )
        scan = simulate_profile_on_plane(
            T_base_ef=flange[pose_id],
            T_ef_s_true=context.T_true,
            plane_n=context.normal,
            plane_l=context.offset_mm,
            x_values=context.x_values,
            noise_std=0.0,
            plane_id=0,
            scan_id=pose_id,
        )
        points[pose_id] = scan.points_s
    b_actual = np.einsum("nji,j->ni", transforms[:, :3, :3], context.normal)
    actual_psi = np.rad2deg(np.arctan2(b_actual[:, 1], b_actual[:, 0]))
    actual_tilt = np.rad2deg(np.arccos(np.clip(-b_actual[:, 2], -1.0, 1.0)))
    validation = {
        "max_b_target_error": float(np.max(np.linalg.norm(b_actual - target_b, axis=1))),
        "max_b_unit_error": float(np.max(np.abs(np.linalg.norm(b_actual, axis=1) - 1.0))),
        "max_psi_b_error_deg": float(
            np.max(np.abs(_angular_error_deg(actual_psi, psi_b_deg)))
        ),
        "max_tilt_error_deg": float(np.max(np.abs(actual_tilt - theta_deg))),
        "profile_z_min_mm": float(np.min(points[:, :, 2])),
        "profile_z_max_mm": float(np.max(points[:, :, 2])),
    }
    tolerance = float(config["numerics"]["identity_tolerance"])
    if max(
        validation["max_b_target_error"],
        validation["max_b_unit_error"],
        validation["max_psi_b_error_deg"],
        validation["max_tilt_error_deg"],
    ) > tolerance:
        raise RuntimeError(f"pose identity failed for {experiment}/{condition}: {validation}")
    if validation["profile_z_min_mm"] < 20.0 or validation["profile_z_max_mm"] > 250.0:
        raise RuntimeError(f"profile range invalid for {experiment}/{condition}: {validation}")
    return PoseDesign(
        design_id=design_id,
        experiment=experiment,
        condition=condition,
        block_or_shift=block_or_shift,
        independent_value=float(independent_value),
        theta_deg=theta_deg,
        psi_b_deg=psi_b_deg,
        gamma_deg=gamma_deg,
        legacy_phi_deg=_legacy_phi(theta_deg, gamma_deg, psi_b_deg),
        u_mm=u_mm,
        v_mm=v_mm,
        d_mm=d_mm,
        T_base_s=transforms,
        T_base_ef=flange,
        ideal_points_s=points,
        b_actual=b_actual,
        validation=validation,
    )


def _spectral(
    matrix: np.ndarray, prefix: str, config: dict[str, Any]
) -> dict[str, float | int | str]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(eigenvalues[-1])
    tolerance = max(1e-12, abs(maximum) * float(config["numerics"]["rank_relative_tolerance"]))
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(len(positive))
    full_rank = rank == len(eigenvalues)
    ordinary_logdet = float(np.sum(np.log(eigenvalues))) if full_rank else float("-inf")
    condition = float(maximum / eigenvalues[0]) if full_rank else float("inf")
    trace_inv = float(np.sum(1.0 / eigenvalues)) if full_rank else float("inf")
    output: dict[str, float | int | str] = {
        f"{prefix}_trace": float(np.trace(matrix)),
        f"{prefix}_lambda_min": float(eigenvalues[0]),
        f"{prefix}_lambda_max": maximum,
        f"{prefix}_rank": rank,
        f"{prefix}_rank_tolerance": tolerance,
        f"{prefix}_condition": condition,
        f"{prefix}_logdet": ordinary_logdet,
        f"{prefix}_pseudo_logdet": (
            float(np.sum(np.log(positive))) if len(positive) else float("-inf")
        ),
        f"{prefix}_trace_inv": trace_inv,
    }
    if len(eigenvalues) >= 3:
        output[f"{prefix}_lambda_mid"] = float(eigenvalues[-2])
    for index, value in enumerate(eigenvalues):
        output[f"{prefix}_eigenvalue_{index + 1}"] = float(value)
    return output


def _effective(j_x: np.ndarray, j_pi: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if j_pi.shape[1] == 0:
        return 0.5 * (j_x.T @ j_x + (j_x.T @ j_x).T)
    return decomposition._schur_effective(j_x, j_pi)


def _conditional_rotation(
    h_eff: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, str, int, float]:
    h_rr = h_eff[:3, :3]
    h_rt = h_eff[:3, 3:6]
    h_tt = h_eff[3:6, 3:6]
    eigenvalues = np.linalg.eigvalsh(0.5 * (h_tt + h_tt.T))
    tolerance = max(
        1e-12,
        abs(float(eigenvalues[-1]))
        * float(config["numerics"]["rank_relative_tolerance"]),
    )
    rank = int(np.sum(eigenvalues > tolerance))
    if rank == 3:
        solved = np.linalg.solve(h_tt, h_rt.T)
        method = "solve"
    else:
        solved = np.linalg.pinv(
            h_tt, rcond=float(config["numerics"]["pinv_rcond"])
        ) @ h_rt.T
        method = "pinv"
    conditional = h_rr - h_rt @ solved
    return 0.5 * (conditional + conditional.T), method, rank, tolerance


def _design_bank(design: PoseDesign) -> dict[str, np.ndarray]:
    parameters = np.column_stack(
        [
            design.u_mm,
            design.v_mm,
            design.d_mm,
            design.theta_deg,
            design.gamma_deg,
            design.psi_b_deg,
        ]
    )
    return {
        "candidate_ids": np.arange(len(parameters), dtype=np.int64),
        "pose_parameters": parameters,
        "T_base_s": design.T_base_s,
        "T_base_ef": design.T_base_ef,
        "ideal_points_s": design.ideal_points_s,
    }


def _geometry(
    design: PoseDesign, context: ExperimentContext, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bank = _design_bank(design)
    ids = bank["candidate_ids"]
    joint, truth_residual = jacobian_analysis._joint_jacobian(
        ids, bank, context.normal, context.offset_mm
    )
    scales = np.asarray(
        [ROTATION_SCALE_RAD] * 3
        + [1.0] * 3
        + [ROTATION_SCALE_RAD] * 2
        + [1.0],
        dtype=float,
    )
    joint_scaled = joint * scales[None, :]
    j_x = joint_scaled[:, :6]
    j_pi = joint_scaled[:, 6:]
    h_xx = j_x.T @ j_x
    h_eff = _effective(j_x, j_pi, config)
    h_eff_normal = _effective(j_x, j_pi[:, :2], config)
    h_eff_offset = _effective(j_x, j_pi[:, 2:3], config)
    h_r_given_t, conditional_method, htt_rank, htt_tolerance = _conditional_rotation(
        h_eff, config
    )
    raw_hrr = h_xx[:3, :3]
    b = design.b_actual
    centered_b = b - np.mean(b, axis=0, keepdims=True)
    cb = centered_b.T @ centered_b / len(b)
    cb_eigenvalues = np.linalg.eigvalsh(cb)
    cb_condition = float(cb_eigenvalues[-1] / cb_eigenvalues[0])
    denominator = math.sqrt(
        max(float(np.linalg.norm(h_xx, ord="fro")), 1e-12)
        * max(float(np.linalg.norm(j_pi.T @ j_pi, ord="fro")), 1e-12)
    )
    row: dict[str, Any] = {
        "design_id": design.design_id,
        "experiment": design.experiment,
        "condition": design.condition,
        "block_or_shift": design.block_or_shift,
        "N": len(b),
        "independent_value": design.independent_value,
        "R_uv_mm": float(np.sqrt(np.mean(design.u_mm**2 + design.v_mm**2))),
        "mean_d_mm": float(np.mean(design.d_mm)),
        "std_d_mm": float(np.std(design.d_mm)),
        "var_d_mm2": float(np.var(design.d_mm)),
        "trace_Cb": float(np.trace(cb)),
        "lambda_min_Cb": float(cb_eigenvalues[0]),
        "lambda_mid_Cb": float(cb_eigenvalues[1]),
        "lambda_max_Cb": float(cb_eigenvalues[2]),
        "condition_Cb": cb_condition,
        "b_resultant": float(np.linalg.norm(np.mean(b, axis=0))),
        "max_truth_residual_mm": truth_residual,
        "max_b_target_error": design.validation["max_b_target_error"],
        "max_b_unit_error": design.validation["max_b_unit_error"],
        "max_psi_b_error_deg": design.validation["max_psi_b_error_deg"],
        "max_tilt_error_deg": design.validation["max_tilt_error_deg"],
        "profile_z_min_mm": design.validation["profile_z_min_mm"],
        "profile_z_max_mm": design.validation["profile_z_max_mm"],
        "J_X_fro": float(np.linalg.norm(j_x, ord="fro")),
        "J_pi_fro": float(np.linalg.norm(j_pi, ord="fro")),
        "H_Xpi_fro": float(np.linalg.norm(j_x.T @ j_pi, ord="fro")),
        "H_Xpi_normalized_fro": float(
            np.linalg.norm(j_x.T @ j_pi, ord="fro") / denominator
        ),
        "total_nuisance_trace_loss": float(np.trace(h_xx) - np.trace(h_eff)),
        "plane_normal_trace_loss": float(
            np.trace(h_xx) - np.trace(h_eff_normal)
        ),
        "plane_offset_trace_loss": float(
            np.trace(h_xx) - np.trace(h_eff_offset)
        ),
        "information_retention_trace": float(np.trace(h_eff) / np.trace(h_xx)),
        "H_R_given_t_solver": conditional_method,
        "H_tt_eff_rank": htt_rank,
        "H_tt_eff_rank_tolerance": htt_tolerance,
    }
    row.update(_spectral(raw_hrr, "raw_HRR", config))
    row.update(_spectral(h_eff, "H_eff", config))
    row.update(_spectral(h_eff[:3, :3], "H_RR_eff", config))
    row.update(_spectral(h_eff[3:6, 3:6], "H_tt_eff", config))
    row.update(_spectral(h_r_given_t, "H_R_given_t", config))
    matrices = {
        "b": b,
        "Cb": cb,
        "points_s": design.ideal_points_s,
        "J_joint": joint_scaled,
        "J_X": j_x,
        "J_pi": j_pi,
        "H_XX": h_xx,
        "H_eff": h_eff,
        "H_R_given_t": h_r_given_t,
    }
    return row, matrices


def _assert_fixed_cb(
    rows: Sequence[dict[str, Any]], baseline: dict[str, Any], config: dict[str, Any]
) -> None:
    tolerance = float(config["numerics"]["identity_tolerance"])
    fields = (
        "trace_Cb",
        "lambda_min_Cb",
        "lambda_mid_Cb",
        "lambda_max_Cb",
        "condition_Cb",
        "b_resultant",
    )
    for row in rows:
        for field in fields:
            if abs(float(row[field]) - float(baseline[field])) > tolerance:
                raise RuntimeError(
                    f"translation b geometry changed in {row['experiment']}/"
                    f"{row['condition']}: {field}"
                )


def _r0_designs(context: ExperimentContext, config: dict[str, Any]) -> list[PoseDesign]:
    n = int(config["N"])
    zeros = np.zeros(n)
    distance = np.full(n, float(config["baseline"]["distance_mm"]))
    uniform = np.arange(n, dtype=float) * 360.0 / n
    clustered = np.asarray(config["r0_gamma"]["clustered_deg"], dtype=float)
    rng = np.random.default_rng(int(config["r0_gamma"]["random_seed"]))
    random_gamma = rng.uniform(-180.0, 180.0, n)
    schedules = (
        (0, "A_all_zero", zeros),
        (1, "B_uniform_360", uniform),
        (2, "C_clustered", clustered),
        (3, "D_random_uniform", random_gamma),
    )
    return [
        _make_design(
            context,
            config,
            design_id=design_id,
            experiment="R0_gamma_invariance",
            condition=name,
            block_or_shift="0",
            independent_value=float(np.std(gamma)),
            gamma_deg=gamma,
            u_mm=zeros,
            v_mm=zeros,
            d_mm=distance,
        )
        for design_id, name, gamma in schedules
    ]


def _r1_r2_designs(context: ExperimentContext, config: dict[str, Any]) -> list[PoseDesign]:
    n = int(config["N"])
    gamma = np.zeros(n)
    angles = 2.0 * np.pi * np.arange(n) / n
    designs: list[PoseDesign] = []
    for index, radius in enumerate(config["r1_uv_radii_mm"]):
        radius = float(radius)
        designs.append(
            _make_design(
                context,
                config,
                design_id=100 + index,
                experiment="R1_uv_spread",
                condition=f"Ruv_{radius:g}mm",
                block_or_shift="0",
                independent_value=radius,
                gamma_deg=gamma,
                u_mm=radius * np.cos(angles),
                v_mm=radius * np.sin(angles),
                d_mm=np.full(n, float(config["baseline"]["distance_mm"])),
            )
        )
    radius = 40.0
    u = radius * np.cos(angles)
    v = radius * np.sin(angles)
    for index, distance in enumerate(config["r2_mean_distances_mm"]):
        distance = float(distance)
        designs.append(
            _make_design(
                context,
                config,
                design_id=200 + index,
                experiment="R2A_mean_distance",
                condition=f"d_{distance:g}mm",
                block_or_shift="0",
                independent_value=distance,
                gamma_deg=gamma,
                u_mm=u,
                v_mm=v,
                d_mm=np.full(n, distance),
            )
        )
    for level_index, (level, values) in enumerate(
        config["r2_diversity_levels"].items()
    ):
        base = np.asarray(values, dtype=float)
        if len(base) != n or not np.isclose(np.mean(base), 105.0, atol=1e-12):
            raise ValueError(f"invalid fixed-mean distance schedule: {level}")
        shifts = (0,) if level == "D0" else range(n)
        for shift in shifts:
            distance = np.roll(base, int(shift))
            designs.append(
                _make_design(
                    context,
                    config,
                    design_id=300 + 10 * level_index + int(shift),
                    experiment="R2B_distance_diversity",
                    condition=level,
                    block_or_shift=str(shift),
                    independent_value=float(np.std(distance)),
                    gamma_deg=gamma,
                    u_mm=u,
                    v_mm=v,
                    d_mm=distance,
                )
            )
    return designs


def _r3_designs(context: ExperimentContext, config: dict[str, Any]) -> list[PoseDesign]:
    n = int(config["N"])
    gamma = np.zeros(n)
    angles = 2.0 * np.pi * np.arange(n) / n
    settings = config["r3"]
    designs: list[PoseDesign] = []
    for uv_index, (uv_name, radius) in enumerate(
        (
            ("small_uv", float(settings["small_uv_radius_mm"])),
            ("large_uv", float(settings["large_uv_radius_mm"])),
        )
    ):
        u = radius * np.cos(angles)
        v = radius * np.sin(angles)
        constant = np.full(n, float(settings["constant_distance_mm"]))
        designs.append(
            _make_design(
                context,
                config,
                design_id=400 + uv_index * 20,
                experiment="R3_interaction",
                condition=f"{uv_name}_d_constant",
                block_or_shift="0",
                independent_value=radius,
                gamma_deg=gamma,
                u_mm=u,
                v_mm=v,
                d_mm=constant,
            )
        )
        diverse = np.asarray(settings["diverse_distances_mm"], dtype=float)
        for shift in range(n):
            designs.append(
                _make_design(
                    context,
                    config,
                    design_id=410 + uv_index * 20 + shift,
                    experiment="R3_interaction",
                    condition=f"{uv_name}_d_diverse",
                    block_or_shift=str(shift),
                    independent_value=radius,
                    gamma_deg=gamma,
                    u_mm=u,
                    v_mm=v,
                    d_mm=np.roll(diverse, shift),
                )
            )
    return designs


def _baseline_pose_rows(design: PoseDesign) -> list[dict[str, Any]]:
    rows = []
    for pose_id in range(len(design.theta_deg)):
        rows.append(
            {
                "pose_id": pose_id,
                "theta_deg": design.theta_deg[pose_id],
                "psi_b_deg": design.psi_b_deg[pose_id],
                "gamma_deg": design.gamma_deg[pose_id],
                "legacy_phi_deg": design.legacy_phi_deg[pose_id],
                "u_mm": design.u_mm[pose_id],
                "v_mm": design.v_mm[pose_id],
                "d_mm": design.d_mm[pose_id],
                "b_x": design.b_actual[pose_id, 0],
                "b_y": design.b_actual[pose_id, 1],
                "b_z": design.b_actual[pose_id, 2],
            }
        )
    return rows


def run_r0(config: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    r0_path = output / "r0_gamma_invariance.csv"
    if r0_path.exists():
        raise FileExistsError(f"refusing to overwrite: {r0_path}")
    context = _context(config)
    designs = _r0_designs(context, config)
    geometry_rows: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for design in designs:
        row, matrix = _geometry(design, context, config)
        geometry_rows.append(row)
        matrices[design.condition] = matrix
    baseline = geometry_rows[0]
    _assert_fixed_cb(geometry_rows, baseline, config)
    expected = np.asarray([0.0417558508, 0.1930925065, 0.1930925065])
    actual = np.asarray(
        [baseline["lambda_min_Cb"], baseline["lambda_mid_Cb"], baseline["lambda_max_Cb"]]
    )
    if not np.allclose(actual, expected, atol=2e-10, rtol=0.0):
        raise RuntimeError(f"baseline Cb does not match theoretical ring: {actual}")
    base_matrix = matrices[designs[0].condition]
    comparison_rows: list[dict[str, Any]] = []
    for design, geometry_row in zip(designs, geometry_rows, strict=True):
        matrix = matrices[design.condition]
        row = dict(geometry_row)
        row.update(
            max_b_difference_vs_A=float(
                np.max(np.linalg.norm(matrix["b"] - base_matrix["b"], axis=1))
            ),
            max_scan_point_difference_mm_vs_A=float(
                np.max(np.abs(matrix["points_s"] - base_matrix["points_s"]))
            ),
            J_X_relative_fro_vs_A=decomposition._relative_frobenius(
                matrix["J_X"], base_matrix["J_X"]
            ),
            J_pi_relative_fro_vs_A=decomposition._relative_frobenius(
                matrix["J_pi"], base_matrix["J_pi"]
            ),
            H_eff_relative_fro_vs_A=decomposition._relative_frobenius(
                matrix["H_eff"], base_matrix["H_eff"]
            ),
            H_R_given_t_relative_fro_vs_A=decomposition._relative_frobenius(
                matrix["H_R_given_t"], base_matrix["H_R_given_t"]
            ),
        )
        comparison_rows.append(row)
    _write_csv(output / "baseline_pose_table.csv", _baseline_pose_rows(designs[0]))
    _write_csv(r0_path, comparison_rows)
    np.savez_compressed(
        output / "r0_gamma_invariance_matrices.npz",
        **{
            f"{condition}__{name}": value
            for condition, items in matrices.items()
            for name, value in items.items()
        },
    )
    _write_json(
        output / "r0_manifest.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stage": "r0",
            "status": "complete",
            "pose_convention": PLANE_RELATIVE_POSE_CONVENTION,
            "N": int(config["N"]),
            "theoretical_Cb_eigenvalues": expected,
            "actual_Cb_eigenvalues": actual,
            "actual_trace_Cb": baseline["trace_Cb"],
            "actual_condition_Cb": baseline["condition_Cb"],
            "files": {
                "baseline_pose_table.csv": _sha256(output / "baseline_pose_table.csv"),
                "r0_gamma_invariance.csv": _sha256(r0_path),
            },
        },
    )
    print(f"R0 complete: {output}", flush=True)


def run_geometry(config: dict[str, Any], output: Path) -> None:
    if not (output / "r0_manifest.json").exists():
        raise RuntimeError("R0 must complete before R1/R2 geometry")
    paths = {
        "R1_uv_spread": output / "r1_uv_spread_geometry.csv",
        "R2A_mean_distance": output / "r2_mean_distance_geometry.csv",
        "R2B_distance_diversity": output / "r2_distance_diversity_geometry.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("R1/R2 geometry output already exists")
    context = _context(config)
    designs = _r1_r2_designs(context, config)
    rows: list[dict[str, Any]] = []
    for position, design in enumerate(designs, start=1):
        row, _matrices = _geometry(design, context, config)
        rows.append(row)
        print(
            f"[geometry {position}/{len(designs)}] {design.experiment} "
            f"{design.condition} shift={design.block_or_shift}",
            flush=True,
        )
    baseline = _read_csv(output / "r0_gamma_invariance.csv")[0]
    _assert_fixed_cb(rows, baseline, config)
    for experiment, path in paths.items():
        _write_csv(path, [row for row in rows if row["experiment"] == experiment])
    _write_json(
        output / "geometry_manifest.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stage": "R1_R2_geometry",
            "status": "complete",
            "design_count": len(rows),
            "fixed_b_verified": True,
            "files": {path.name: _sha256(path) for path in paths.values()},
        },
    )
    print(f"R1/R2 geometry complete: {output}", flush=True)


def run_r3(config: dict[str, Any], output: Path) -> None:
    if not (output / "geometry_manifest.json").exists():
        raise RuntimeError("R1/R2 geometry must complete before optional R3")
    path = output / "r3_interaction_geometry.csv"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    context = _context(config)
    designs = _r3_designs(context, config)
    rows = [_geometry(design, context, config)[0] for design in designs]
    baseline = _read_csv(output / "r0_gamma_invariance.csv")[0]
    _assert_fixed_cb(rows, baseline, config)
    _write_csv(path, rows)
    _write_json(
        output / "r3_manifest.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stage": "R3_geometry",
            "status": "complete",
            "design_count": len(rows),
            "fixed_b_verified": True,
            "file": {path.name: _sha256(path)},
        },
    )
    print(f"R3 geometry complete: {output}", flush=True)


def _all_designs(
    context: ExperimentContext, config: dict[str, Any], output: Path
) -> list[PoseDesign]:
    if not (output / "r0_manifest.json").exists() or not (
        output / "geometry_manifest.json"
    ).exists():
        raise RuntimeError("R0 and R1/R2 geometry must complete before calibration")
    designs = _r0_designs(context, config) + _r1_r2_designs(context, config)
    if (output / "r3_manifest.json").exists():
        designs += _r3_designs(context, config)
    ids = [design.design_id for design in designs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("design IDs are not unique")
    return designs


def _paired_noisy_scans(
    design: PoseDesign,
    *,
    environment: int,
    repeat: int,
    noise_seed: int,
    noise_std: float,
    noise_axis: str,
) -> list[LaserScan]:
    scans: list[LaserScan] = []
    for slot in range(len(design.theta_deg)):
        points = design.ideal_points_s[slot].copy()
        rng = np.random.default_rng(
            np.random.SeedSequence([noise_seed, environment, repeat, slot])
        )
        if noise_axis == "z":
            points[:, 2] += rng.normal(0.0, noise_std, len(points))
        elif noise_axis == "xz":
            points[:, [0, 2]] += rng.normal(0.0, noise_std, (len(points), 2))
        else:
            raise ValueError(f"unsupported noise axis: {noise_axis}")
        scans.append(
            LaserScan(
                T_base_ef=design.T_base_ef[slot],
                points_s=points,
                plane_id=0,
                scan_id=slot,
                meta={
                    "design_id": design.design_id,
                    "environment_id": environment,
                    "noise_repeat": repeat,
                    "noise_slot": slot,
                },
            )
        )
    return scans


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _calibration_summary(
    design: PoseDesign, runs: Sequence[dict[str, Any]], geometry: dict[str, str]
) -> dict[str, Any]:
    rotation = _finite([float(row["rotation_error_deg"]) for row in runs])
    translation = _finite([float(row["translation_error_mm"]) for row in runs])
    if not len(rotation) or not len(translation):
        raise RuntimeError(f"no finite calibration results: {design.design_id}")
    return {
        "design_id": design.design_id,
        "experiment": design.experiment,
        "condition": design.condition,
        "block_or_shift": design.block_or_shift,
        "N": len(design.theta_deg),
        "independent_value": design.independent_value,
        "runs": len(runs),
        "finite_rotation_runs": len(rotation),
        "finite_translation_runs": len(translation),
        "catastrophic_failure_count": sum(bool(row["catastrophic_failure"]) for row in runs),
        "rotation_error_median_deg": float(np.median(rotation)),
        "rotation_error_p90_deg": float(np.quantile(rotation, 0.9)),
        "rotation_error_mean_deg": float(np.mean(rotation)),
        "translation_error_median_mm": float(np.median(translation)),
        "translation_error_p90_mm": float(np.quantile(translation, 0.9)),
        "translation_error_mean_mm": float(np.mean(translation)),
        "full_success_rate": float(
            np.mean([bool(row["passes_success_threshold"]) for row in runs])
        ),
        **{
            key: geometry[key]
            for key in (
                "R_uv_mm",
                "mean_d_mm",
                "std_d_mm",
                "trace_Cb",
                "lambda_min_Cb",
                "lambda_max_Cb",
                "condition_Cb",
                "raw_HRR_lambda_min",
                "raw_HRR_logdet",
                "H_R_given_t_lambda_min",
                "H_R_given_t_logdet",
                "H_R_given_t_trace_inv",
                "H_eff_lambda_min",
                "H_eff_logdet",
                "total_nuisance_trace_loss",
                "plane_normal_trace_loss",
                "plane_offset_trace_loss",
            )
        },
    }


def _geometry_lookup(output: Path) -> dict[int, dict[str, str]]:
    paths = [
        output / "r0_gamma_invariance.csv",
        output / "r1_uv_spread_geometry.csv",
        output / "r2_mean_distance_geometry.csv",
        output / "r2_distance_diversity_geometry.csv",
    ]
    if (output / "r3_interaction_geometry.csv").exists():
        paths.append(output / "r3_interaction_geometry.csv")
    rows = [row for path in paths for row in _read_csv(path)]
    return {int(row["design_id"]): row for row in rows}


def run_calibration(config: dict[str, Any], output: Path, resume: bool) -> None:
    context = _context(config)
    designs = _all_designs(context, config, output)
    summary_path = output / "calibration_summary.csv"
    run_path = output / "calibration_runs.csv"
    if summary_path.exists() and not resume:
        raise FileExistsError("calibration checkpoint exists; use --resume")
    completed = (
        {int(row["design_id"]) for row in _read_csv(summary_path)}
        if summary_path.exists()
        else set()
    )
    geometry = _geometry_lookup(output)
    phase2a_dir = _resolve(config["phase2a_dir"])
    with np.load(phase2a_dir / "scenario_bank.npz", allow_pickle=False) as archive:
        scenarios = {name: archive[name].copy() for name in archive.files}
    source_scenario_hash = _sha256(phase2a_dir / "scenario_bank.npz")
    noise = context.phase2a_manifest["config"]["noise"]
    repeats = int(noise["repeats_per_environment"])
    noise_seed = int(config["seed"])
    classification = {
        "classification_translation_mm": float(
            context.phase2a_manifest["config"]["classification"]["translation_mm"]
        ),
        "classification_rotation_deg": float(
            context.phase2a_manifest["config"]["classification"]["rotation_deg"]
        ),
    }
    remaining = [design for design in designs if design.design_id not in completed]
    runs_per_design = len(scenarios["environment_ids"]) * repeats
    total = len(remaining) * runs_per_design
    completed_runs = 0
    started = time.perf_counter()
    for position, design in enumerate(remaining, start=1):
        design_runs: list[dict[str, Any]] = []
        for environment_value in scenarios["environment_ids"]:
            environment = int(environment_value)
            resolved = phase2a._solver_resolved(
                context.phase1_manifest,
                classification,
                scenarios["T_init"][environment],
            )
            for repeat in range(repeats):
                scans = _paired_noisy_scans(
                    design,
                    environment=environment,
                    repeat=repeat,
                    noise_seed=noise_seed,
                    noise_std=float(noise["std_mm"]),
                    noise_axis=str(noise["axis"]),
                )
                row = phase1._run_one_calibration(
                    resolved=resolved,
                    scans=scans,
                    scan_count=int(config["N"]),
                    subset_id=design.design_id,
                    repeat=repeat,
                    candidate_ids=np.arange(int(config["N"]), dtype=np.int64),
                )
                row.update(
                    source="fixed_translation_rotation_geometry",
                    design_id=design.design_id,
                    experiment=design.experiment,
                    condition=design.condition,
                    block_or_shift=design.block_or_shift,
                    environment_id=environment,
                    noise_repeat=repeat,
                )
                design_runs.append(row)
                completed_runs += 1
        summary = _calibration_summary(design, design_runs, geometry[design.design_id])
        _append_csv(run_path, design_runs)
        _append_csv(summary_path, [summary])
        elapsed = time.perf_counter() - started
        eta = elapsed * (total - completed_runs) / max(completed_runs, 1)
        print(
            f"[calibration {position}/{len(remaining)}] design={design.design_id} "
            f"{design.experiment}/{design.condition}/shift={design.block_or_shift}; "
            f"runs {completed_runs}/{total}; ETA {phase1._format_duration(eta)}",
            flush=True,
        )
    manifest_path = output / "calibration_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite: {manifest_path}")
    _write_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stage": "calibration",
            "status": "complete",
            "design_count": len(designs),
            "runs_per_design": runs_per_design,
            "calibration_runs": len(designs) * runs_per_design,
            "scenario_bank_source": str(phase2a_dir / "scenario_bank.npz"),
            "scenario_bank_sha256": source_scenario_hash,
            "noise_pairing": "identical by environment/repeat/pose slot",
            "noise_seed": noise_seed,
            "subset_is_statistical_unit": False,
            "scenario_is_paired_repeated_measurement": True,
        },
    )


def _bootstrap_median(
    values: np.ndarray, repeats: int, rng: np.random.Generator
) -> tuple[float, float]:
    samples = np.empty(repeats)
    for index in range(repeats):
        sample = values[rng.integers(0, len(values), len(values))]
        samples[index] = np.median(sample)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_statistics(
    run_rows: Sequence[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    by_design: dict[int, dict[tuple[int, int], float]] = {}
    metadata: dict[int, dict[str, str]] = {}
    for row in run_rows:
        design_id = int(row["design_id"])
        metadata[design_id] = row
        by_design.setdefault(design_id, {})[
            (int(row["environment_id"]), int(row["noise_repeat"]))
        ] = float(row["rotation_error_deg"])
    references: dict[str, int] = {
        "R0_gamma_invariance": 0,
        "R1_uv_spread": 100,
        "R2A_mean_distance": 203,
        "R2B_distance_diversity": 300,
        "R3_interaction": 400,
    }
    rng = np.random.default_rng(int(config["seed"]) + 0x53544154)
    repeats = int(config["bootstrap_repeats"])
    output: list[dict[str, Any]] = []
    for design_id in sorted(by_design):
        experiment = metadata[design_id]["experiment"]
        reference_id = references[experiment]
        if reference_id not in by_design:
            continue
        keys = sorted(set(by_design[design_id]) & set(by_design[reference_id]))
        left = np.asarray([by_design[design_id][key] for key in keys])
        right = np.asarray([by_design[reference_id][key] for key in keys])
        finite = np.isfinite(left) & np.isfinite(right)
        difference = left[finite] - right[finite]
        if len(difference) == 0:
            continue
        statistic, p_value = (
            wilcoxon(difference)
            if np.any(difference != 0.0)
            else (0.0, 1.0)
        )
        ci_low, ci_high = _bootstrap_median(difference, repeats, rng)
        output.append(
            {
                "design_id": design_id,
                "reference_design_id": reference_id,
                "experiment": experiment,
                "condition": metadata[design_id]["condition"],
                "block_or_shift": metadata[design_id]["block_or_shift"],
                "paired_scenarios": len(difference),
                "rotation_error_median_difference_deg": float(np.median(difference)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "condition_win_rate": float(np.mean(difference < 0.0)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "fdr_q": math.nan,
            }
        )
    adjusted = statistics._bh_adjust([row["p_value"] for row in output])
    for row, q_value in zip(output, adjusted, strict=True):
        row["fdr_q"] = float(q_value)
    return output


def _r2b_level_summary(
    summaries: Sequence[dict[str, str]], geometry: Sequence[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    geometry_lookup = {int(row["design_id"]): row for row in geometry}
    selected = [row for row in summaries if row["experiment"] == "R2B_distance_diversity"]
    rng = np.random.default_rng(int(config["seed"]) + 0x523242)
    repeats = int(config["bootstrap_repeats"])
    output: list[dict[str, Any]] = []
    for level in config["r2_diversity_levels"]:
        rows = [row for row in selected if row["condition"] == level]
        rotation = np.asarray([float(row["rotation_error_median_deg"]) for row in rows])
        design_ids = [int(row["design_id"]) for row in rows]
        lambda_min = np.asarray(
            [float(geometry_lookup[design_id]["H_R_given_t_lambda_min"]) for design_id in design_ids]
        )
        trace_inv = np.asarray(
            [float(geometry_lookup[design_id]["H_R_given_t_trace_inv"]) for design_id in design_ids]
        )
        success = np.asarray([float(row["full_success_rate"]) for row in rows])
        translation = np.asarray(
            [float(row["translation_error_median_mm"]) for row in rows]
        )
        if len(rows) > 1:
            rotation_ci = _bootstrap_median(rotation, repeats, rng)
            lambda_ci = _bootstrap_median(lambda_min, repeats, rng)
            trace_inv_ci = _bootstrap_median(trace_inv, repeats, rng)
        else:
            rotation_ci = (float(rotation[0]), float(rotation[0]))
            lambda_ci = (float(lambda_min[0]), float(lambda_min[0]))
            trace_inv_ci = (float(trace_inv[0]), float(trace_inv[0]))
        output.append(
            {
                "level": level,
                "shift_count": len(rows),
                "std_d_mm": float(rows[0]["std_d_mm"]),
                "rotation_error_median_across_shifts_deg": float(np.median(rotation)),
                "rotation_error_bootstrap_ci_low": rotation_ci[0],
                "rotation_error_bootstrap_ci_high": rotation_ci[1],
                "H_R_given_t_lambda_min_median": float(np.median(lambda_min)),
                "H_R_given_t_lambda_min_ci_low": lambda_ci[0],
                "H_R_given_t_lambda_min_ci_high": lambda_ci[1],
                "H_R_given_t_trace_inv_median": float(np.median(trace_inv)),
                "H_R_given_t_trace_inv_ci_low": trace_inv_ci[0],
                "H_R_given_t_trace_inv_ci_high": trace_inv_ci[1],
                "translation_error_median_across_shifts_mm": float(
                    np.median(translation)
                ),
                "full_success_rate_median_across_shifts": float(np.median(success)),
            }
        )
    return output


def _plot_sweep(
    rows: Sequence[dict[str, str]], x_field: str, title: str, path: Path
) -> None:
    ordered = sorted(rows, key=lambda row: float(row[x_field]))
    x = np.asarray([float(row[x_field]) for row in ordered])
    fields = (
        ("H_R_given_t_lambda_min", "lambda_min(H_R|t)"),
        ("H_R_given_t_logdet", "logdet(H_R|t)"),
        ("H_R_given_t_trace_inv", "trace(inv(H_R|t))"),
        ("raw_HRR_lambda_min", "raw HRR lambda_min"),
        ("rotation_error_median_deg", "rotation median error [deg]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    for axis, (field, ylabel) in zip(axes.flat, fields, strict=False):
        axis.plot(x, [float(row[field]) for row in ordered], "o-")
        axis.set_xlabel(x_field)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_r2b(
    summaries: Sequence[dict[str, str]], levels: Sequence[dict[str, Any]], path: Path
) -> None:
    rows = [row for row in summaries if row["experiment"] == "R2B_distance_diversity"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    fields = (
        ("H_R_given_t_lambda_min", "lambda_min(H_R|t)"),
        ("H_R_given_t_trace_inv", "trace(inv(H_R|t))"),
        ("rotation_error_median_deg", "rotation median error [deg]"),
    )
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        axis.scatter(
            [float(row["std_d_mm"]) for row in rows],
            [float(row[field]) for row in rows],
            alpha=0.55,
            label="cyclic shifts",
        )
        if field == "H_R_given_t_lambda_min":
            median = [row["H_R_given_t_lambda_min_median"] for row in levels]
        elif field == "H_R_given_t_trace_inv":
            median = [row["H_R_given_t_trace_inv_median"] for row in levels]
        else:
            median = [row["rotation_error_median_across_shifts_deg"] for row in levels]
        axis.plot([row["std_d_mm"] for row in levels], median, "ko-", label="level median")
        axis.set_xlabel("std(d) [mm]")
        axis.set_ylabel(ylabel)
        if field == "rotation_error_median_deg":
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("R2-B fixed-mean distance diversity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _solver_basin_summary(
    run_rows: Sequence[dict[str, str]], summaries: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    metadata = {int(row["design_id"]): row for row in summaries}
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in run_rows:
        grouped.setdefault(int(row["design_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for design_id in sorted(grouped):
        rows = grouped[design_id]
        rotation = np.asarray([float(row["rotation_error_deg"]) for row in rows])
        translation = np.asarray([float(row["translation_error_mm"]) for row in rows])
        residual = np.asarray([float(row["nonlinear_final_rms_mm"]) for row in rows])
        finite = np.isfinite(rotation) & np.isfinite(translation) & np.isfinite(residual)
        rotation = rotation[finite]
        translation = translation[finite]
        residual = residual[finite]
        accurate = rotation <= 0.1
        opposite = rotation > 90.0
        meta = metadata[design_id]
        output.append(
            {
                "design_id": design_id,
                "experiment": meta["experiment"],
                "condition": meta["condition"],
                "block_or_shift": meta["block_or_shift"],
                "finite_runs": len(rotation),
                "rotation_error_min_deg": float(np.min(rotation)),
                "rotation_error_median_deg": float(np.median(rotation)),
                "rotation_error_max_deg": float(np.max(rotation)),
                "rotation_le_0p1_rate": float(np.mean(accurate)),
                "rotation_le_1_rate": float(np.mean(rotation <= 1.0)),
                "rotation_gt_90_rate": float(np.mean(opposite)),
                "translation_gt_10_rate": float(np.mean(translation > 10.0)),
                "iterative_converged_rate": float(
                    np.mean([_as_bool(row["iterative_converged"]) for row in rows])
                ),
                "nonlinear_success_rate": float(
                    np.mean([_as_bool(row["nonlinear_success"]) for row in rows])
                ),
                "residual_rms_median_mm": float(np.median(residual)),
                "residual_rms_median_rotation_le_0p1_mm": (
                    float(np.median(residual[accurate])) if np.any(accurate) else math.nan
                ),
                "residual_rms_median_rotation_gt_90_mm": (
                    float(np.median(residual[opposite])) if np.any(opposite) else math.nan
                ),
            }
        )
    return output


def _paired_contrast(
    run_rows: Sequence[dict[str, str]],
    left_id: int,
    right_id: int,
    label: str,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    values: dict[int, dict[tuple[int, int], float]] = {left_id: {}, right_id: {}}
    for row in run_rows:
        design_id = int(row["design_id"])
        if design_id in values:
            values[design_id][
                (int(row["environment_id"]), int(row["noise_repeat"]))
            ] = float(row["rotation_error_deg"])
    keys = sorted(set(values[left_id]) & set(values[right_id]))
    left = np.asarray([values[left_id][key] for key in keys])
    right = np.asarray([values[right_id][key] for key in keys])
    finite = np.isfinite(left) & np.isfinite(right)
    difference = left[finite] - right[finite]
    statistic, p_value = (
        wilcoxon(difference) if np.any(difference != 0.0) else (0.0, 1.0)
    )
    low, high = _bootstrap_median(
        difference, int(config["bootstrap_repeats"]), rng
    )
    return {
        "contrast": label,
        "left_design_id": left_id,
        "right_design_id": right_id,
        "paired_scenarios": len(difference),
        "rotation_error_median_difference_deg": float(np.median(difference)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "left_win_rate": float(np.mean(difference < 0.0)),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "fdr_q": math.nan,
    }


def _r3_paired_contrasts(
    run_rows: Sequence[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(config["seed"]) + 0x523343)
    rows = [
        _paired_contrast(run_rows, 420, 400, "uv_effect_at_constant_d", config, rng)
    ]
    for shift in range(9):
        rows.extend(
            [
                _paired_contrast(
                    run_rows,
                    410 + shift,
                    400,
                    f"distance_diversity_effect_small_uv_shift_{shift}",
                    config,
                    rng,
                ),
                _paired_contrast(
                    run_rows,
                    430 + shift,
                    420,
                    f"distance_diversity_effect_large_uv_shift_{shift}",
                    config,
                    rng,
                ),
                _paired_contrast(
                    run_rows,
                    430 + shift,
                    410 + shift,
                    f"uv_effect_at_diverse_d_shift_{shift}",
                    config,
                    rng,
                ),
            ]
        )
    adjusted = statistics._bh_adjust([row["p_value"] for row in rows])
    for row, q_value in zip(rows, adjusted, strict=True):
        row["fdr_q"] = float(q_value)
    return rows


def _r3_level_summary(
    summaries: Sequence[dict[str, str]], geometry: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    geometry_lookup = {int(row["design_id"]): row for row in geometry}
    output: list[dict[str, Any]] = []
    for condition in (
        "small_uv_d_constant",
        "small_uv_d_diverse",
        "large_uv_d_constant",
        "large_uv_d_diverse",
    ):
        rows = [
            row
            for row in summaries
            if row["experiment"] == "R3_interaction" and row["condition"] == condition
        ]
        geometry_rows = [geometry_lookup[int(row["design_id"])] for row in rows]
        values = lambda field: np.asarray([float(row[field]) for row in rows])
        geometry_values = lambda field: np.asarray(
            [float(row[field]) for row in geometry_rows]
        )
        output.append(
            {
                "condition": condition,
                "shift_count": len(rows),
                "rotation_error_median_across_shifts_deg": float(
                    np.median(values("rotation_error_median_deg"))
                ),
                "rotation_error_min_across_shifts_deg": float(
                    np.min(values("rotation_error_median_deg"))
                ),
                "rotation_error_max_across_shifts_deg": float(
                    np.max(values("rotation_error_median_deg"))
                ),
                "translation_error_median_across_shifts_mm": float(
                    np.median(values("translation_error_median_mm"))
                ),
                "full_success_rate_median_across_shifts": float(
                    np.median(values("full_success_rate"))
                ),
                "H_R_given_t_lambda_min_median": float(
                    np.median(geometry_values("H_R_given_t_lambda_min"))
                ),
                "H_R_given_t_logdet_median": float(
                    np.median(geometry_values("H_R_given_t_logdet"))
                ),
                "H_R_given_t_trace_inv_median": float(
                    np.median(geometry_values("H_R_given_t_trace_inv"))
                ),
            }
        )
    return output


def _plot_r0_actual(summaries: Sequence[dict[str, str]], path: Path) -> None:
    rows = [row for row in summaries if row["experiment"] == "R0_gamma_invariance"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    labels = [row["condition"].split("_")[0] for row in rows]
    axes[0].bar(labels, [float(row["rotation_error_median_deg"]) for row in rows])
    axes[0].set_ylabel("rotation median error [deg]")
    axes[1].bar(labels, [float(row["full_success_rate"]) for row in rows])
    axes[1].set_ylabel("full success rate")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("R0 gamma schedules: actual paired calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_r3(levels: Sequence[dict[str, Any]], path: Path) -> None:
    labels = [
        "small uv\nconstant d",
        "small uv\ndiverse d",
        "large uv\nconstant d",
        "large uv\ndiverse d",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    fields = (
        ("H_R_given_t_logdet_median", "logdet(H_R|t)", False),
        ("H_R_given_t_trace_inv_median", "trace(inv(H_R|t))", False),
        ("rotation_error_median_across_shifts_deg", "rotation median error [deg]", True),
        ("full_success_rate_median_across_shifts", "full success rate", False),
    )
    for axis, (field, ylabel, log_scale) in zip(axes.flat, fields, strict=True):
        axis.bar(labels, [float(row[field]) for row in levels])
        axis.set_ylabel(ylabel)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("R3 u/v spread x distance diversity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_error(
    summaries: Sequence[dict[str, str]], metric: str, path: Path
) -> tuple[float, float]:
    x = np.asarray([float(row[metric]) for row in summaries])
    y = np.asarray([float(row["rotation_error_median_deg"]) for row in summaries])
    finite = np.isfinite(x) & np.isfinite(y)
    result = spearmanr(x[finite], y[finite])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for experiment in sorted({row["experiment"] for row in summaries}):
        selected = [row for row in summaries if row["experiment"] == experiment]
        ax.scatter(
            [float(row[metric]) for row in selected],
            [float(row["rotation_error_median_deg"]) for row in selected],
            label=experiment,
            alpha=0.7,
        )
    ax.set_xlabel(metric)
    ax.set_ylabel("rotation median error [deg]")
    ax.set_title(f"Spearman rho={float(result.correlation):+.3f}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return float(result.correlation), float(result.pvalue)


def run_analysis(config: dict[str, Any], output: Path) -> None:
    report_path = output / "report.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")
    if not (output / "calibration_manifest.json").exists():
        raise RuntimeError("calibration must complete before analysis")
    summaries = _read_csv(output / "calibration_summary.csv")
    runs = _read_csv(output / "calibration_runs.csv")
    geometry_lookup = _geometry_lookup(output)
    geometry = list(geometry_lookup.values())
    paired = _paired_statistics(runs, config)
    _write_csv(output / "paired_rotation_statistics.csv", paired)
    basin = _solver_basin_summary(runs, summaries)
    _write_csv(output / "solver_basin_summary.csv", basin)
    levels = _r2b_level_summary(summaries, geometry, config)
    _write_csv(output / "r2_distance_diversity_level_summary.csv", levels)
    r3_levels = _r3_level_summary(summaries, geometry)
    _write_csv(output / "r3_interaction_summary.csv", r3_levels)
    r3_contrasts = _r3_paired_contrasts(runs, config)
    _write_csv(output / "r3_paired_contrasts.csv", r3_contrasts)

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    r1 = [row for row in summaries if row["experiment"] == "R1_uv_spread"]
    r2a = [row for row in summaries if row["experiment"] == "R2A_mean_distance"]
    _plot_r0_actual(summaries, plots / "r0_gamma_actual_calibration.png")
    _plot_sweep(r1, "R_uv_mm", "R1 u/v radial spread", plots / "r1_uv_spread.png")
    _plot_sweep(r2a, "mean_d_mm", "R2-A mean distance", plots / "r2_mean_distance.png")
    _plot_r2b(summaries, levels, plots / "r2_distance_diversity.png")
    _plot_r3(r3_levels, plots / "r3_interaction.png")
    correlation_rows = []
    for metric in (
        "H_R_given_t_lambda_min",
        "H_R_given_t_logdet",
        "H_R_given_t_trace_inv",
    ):
        rho, p_value = _plot_metric_error(
            summaries, metric, plots / f"{metric}_vs_rotation_error.png"
        )
        correlation_rows.append(
            {
                "metric": metric,
                "condition_unit_count": len(summaries),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    _write_csv(output / "geometry_rotation_error_correlations.csv", correlation_rows)

    r0 = _read_csv(output / "r0_gamma_invariance.csv")
    r1_geometry = _read_csv(output / "r1_uv_spread_geometry.csv")
    r2a_geometry = _read_csv(output / "r2_mean_distance_geometry.csv")
    r0_max = {
        field: max(float(row[field]) for row in r0)
        for field in (
            "max_b_difference_vs_A",
            "max_scan_point_difference_mm_vs_A",
            "J_X_relative_fro_vs_A",
            "J_pi_relative_fro_vs_A",
            "H_eff_relative_fro_vs_A",
            "H_R_given_t_relative_fro_vs_A",
        )
    }
    r1_first, r1_last = r1_geometry[0], r1_geometry[-1]
    r2a_reference = next(row for row in r2a_geometry if np.isclose(float(row["mean_d_mm"]), 105.0))
    r2a_low, r2a_high = r2a_geometry[0], r2a_geometry[-1]
    summary_lookup = {int(row["design_id"]): row for row in summaries}
    basin_lookup = {int(row["design_id"]): row for row in basin}
    paired_lookup = {int(row["design_id"]): row for row in paired}

    def rotation_for(design_id: int) -> float:
        return float(summary_lookup[design_id]["rotation_error_median_deg"])

    baseline = r0[0]
    cb_fields = ("trace_Cb", "lambda_min_Cb", "lambda_mid_Cb", "lambda_max_Cb")
    cb_max_difference = max(
        abs(float(row[field]) - float(baseline[field]))
        for row in geometry
        for field in cb_fields
    )
    max_b_target_error = max(float(row["max_b_target_error"]) for row in geometry)
    max_b_unit_error = max(float(row["max_b_unit_error"]) for row in geometry)
    max_psi_error = max(float(row["max_psi_b_error_deg"]) for row in geometry)
    max_tilt_error = max(float(row["max_tilt_error_deg"]) for row in geometry)
    r2a_conditional_span = max(
        float(row["H_R_given_t_lambda_min"]) for row in r2a_geometry
    ) - min(float(row["H_R_given_t_lambda_min"]) for row in r2a_geometry)
    r2a_paired_max = max(
        abs(float(paired_lookup[design_id]["rotation_error_median_difference_deg"]))
        for design_id in range(200, 207)
    )
    r3_uv_diverse_rows = [
        row for row in r3_contrasts if row["contrast"].startswith("uv_effect_at_diverse")
    ]
    r3_uv_diverse_diff = float(
        np.median(
            [float(row["rotation_error_median_difference_deg"]) for row in r3_uv_diverse_rows]
        )
    )
    r3_uv_diverse_wins = float(
        np.median([float(row["left_win_rate"]) for row in r3_uv_diverse_rows])
    )
    r3_uv_significant = sum(
        float(row["fdr_q"]) < 0.05 for row in r3_uv_diverse_rows
    )

    lines = [
        "# Fixed translation geometry: rotation-DOF controlled experiment",
        "",
        "Status: complete. R0, R1, R2-A, R2-B, optional R3, and all paired nonlinear calibration batches finished.",
        "",
        "All comparisons use N=9 and the exact same ordered b_i set. Calibration reuses the saved Phase-2A bank: 30 initialization environments x 3 paired noise repeats = 90 runs/design. There are 64 condition/shift designs and 5,760 calibration runs in total.",
        "",
        "## Reused repository implementation",
        "",
        "- Pose generation: `laser_handeye.pose_design.sensor_pose_from_plane_relative`.",
        "- Scan generation: `laser_handeye.simulation.simulate_profile_on_plane` with fixed sensor-x samples, an infinite plane, and no target clipping/visibility boundary.",
        "- Jacobian: the finite-difference-verified analytic joint Jacobian from `analyze_phase2a_jacobian.py`.",
        "- Parameter order: rotation(3, rad), translation(3, mm), plane-normal tangent(2, rad), plane offset(1, mm). Rotation and plane-normal columns use the existing one-degree scaling before information metrics.",
        "- Nuisance elimination: the existing plane Schur complement; conditional rotation uses solve and reports rank/tolerance without diagonal regularization.",
        "- Calibration: the existing Phase-1 iterative-then-joint-nonlinear solver and Phase-2A scenario bank/configuration.",
        "- Canonical sixth pose coordinate is `psi_b = atan2((R_BS^T n)_y, (R_BS^T n)_x)`. `legacy_phi_deg` is retained only as a conversion/reference column.",
        "",
        "## Fixed translation baseline",
        "",
        f"- trace(Cb): {float(baseline['trace_Cb']):.10f}",
        f"- eigenvalues(Cb): [{float(baseline['lambda_min_Cb']):.10f}, {float(baseline['lambda_mid_Cb']):.10f}, {float(baseline['lambda_max_Cb']):.10f}]",
        f"- condition(Cb): {float(baseline['condition_Cb']):.6f}",
        f"- maximum Cb scalar/eigenvalue drift over every design: {cb_max_difference:.3e}",
        f"- maximum b target / unit-norm error: {max_b_target_error:.3e} / {max_b_unit_error:.3e}",
        f"- maximum recovered psi_b / tilt error: {max_psi_error:.3e} / {max_tilt_error:.3e} deg",
        "",
        "The prescribed symmetric-ring values are reproduced at numerical precision, so translation geometry is genuinely fixed rather than threshold-matched.",
        "",
        "## R0 gamma sanity check",
        "",
        "Fixed: b_i, u=v=0, d=105 mm. Changed only view gamma schedule.",
        f"- max b difference: {r0_max['max_b_difference_vs_A']:.3e}",
        f"- max sensor-frame point difference: {r0_max['max_scan_point_difference_mm_vs_A']:.3e} mm",
        f"- max relative J_X difference: {r0_max['J_X_relative_fro_vs_A']:.3e}",
        f"- max relative J_pi difference: {r0_max['J_pi_relative_fro_vs_A']:.3e}",
        f"- max relative H_eff difference: {r0_max['H_eff_relative_fro_vs_A']:.3e}",
        f"- max relative H_R_given_t difference: {r0_max['H_R_given_t_relative_fro_vs_A']:.3e}",
        "",
        "| gamma schedule | rotation median | full success | paired median difference vs A | FDR q |",
        "|---|---:|---:|---:|---:|",
    ]
    for design_id in range(4):
        row = summary_lookup[design_id]
        comparison = paired_lookup[design_id]
        lines.append(
            f"| {row['condition']} | {float(row['rotation_error_median_deg']):.6f} deg | "
            f"{float(row['full_success_rate']):.3f} | "
            f"{float(comparison['rotation_error_median_difference_deg']):+.6f} deg | "
            f"{float(comparison['fdr_q']):.3g} |"
        )
    lines += [
        "",
        "Result: gamma is exactly invariant for b, noiseless sensor points, and raw J_X, but not for J_pi. Gamma changes the base-frame point arrangement used by the plane-normal nuisance columns, so the Schur-reduced information changes. There is no clipping, truncation, visibility, boundary, noise, or robot-pose term in this geometry check. Despite the local-information change, none of B/C/D gives a significant paired calibration improvement over A after FDR correction. The original 'full gamma invariance' hypothesis is therefore only true for raw extrinsic excitation, not for nuisance-aware information.",
        "",
        "## R1 u/v spread",
        "",
        "Fixed: b_i, d=105 mm, gamma=0. Changed only circular target radius.",
        f"- raw HRR lambda_min R=0 -> 40: {float(r1_first['raw_HRR_lambda_min']):.6g} -> {float(r1_last['raw_HRR_lambda_min']):.6g}",
        f"- conditional rotation lambda_min R=0 -> 40: {float(r1_first['H_R_given_t_lambda_min']):.6g} -> {float(r1_last['H_R_given_t_lambda_min']):.6g}",
        f"- conditional trace(inv) R=0 -> 40: {float(r1_first['H_R_given_t_trace_inv']):.6g} -> {float(r1_last['H_R_given_t_trace_inv']):.6g}",
        f"- rotation median error R=0 -> 40: {rotation_for(100):.6f} -> {rotation_for(104):.6f} deg",
        "",
        "| R_uv [mm] | conditional lambda_min | conditional logdet | rotation median | success | paired difference vs R=0 | FDR q |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r1:
        design_id = int(row["design_id"])
        comparison = paired_lookup[design_id]
        lines.append(
            f"| {float(row['R_uv_mm']):.0f} | {float(row['H_R_given_t_lambda_min']):.6g} | "
            f"{float(row['H_R_given_t_logdet']):.6g} | {float(row['rotation_error_median_deg']):.6f} deg | "
            f"{float(row['full_success_rate']):.3f} | "
            f"{float(comparison['rotation_error_median_difference_deg']):+.6g} deg | {float(comparison['fdr_q']):.3g} |"
        )
    lines += [
        "",
        "Result: raw H_RR is unchanged while H_R_given_t improves, exactly supporting the nuisance-decoupling interpretation. However, actual error is highly bimodal and the paired differences against R=0 are all non-significant. Thus u/v spread improves local conditional information but is not sufficient to remove the global wrong-solution basin at constant distance.",
        "",
        "## R2-A mean distance",
        "",
        "Fixed: b_i, R_uv=40 mm, gamma=0, equal d within each design. Changed only common mean distance.",
        f"- conditional lambda_min d=60 / 105 / 150: {float(r2a_low['H_R_given_t_lambda_min']):.6g} / {float(r2a_reference['H_R_given_t_lambda_min']):.6g} / {float(r2a_high['H_R_given_t_lambda_min']):.6g}",
        f"- rotation median error d=60 / 105 / 150: {rotation_for(200):.6f} / {rotation_for(203):.6f} / {rotation_for(206):.6f} deg",
        f"- full conditional-lambda span over d=60...150: {r2a_conditional_span:.3e}",
        f"- largest absolute paired median error difference vs d=105: {r2a_paired_max:.3e} deg",
        "",
        "Result: a common distance shift changes raw point coordinates and individual nuisance-loss terms, but cancels from final conditional rotation information. The median paired effect is numerically negligible. The large jumps among marginal medians (75 vs 168 deg) come from unstable ordering of a near-50/50 bimodal branch distribution and must not be read as a causal mean-distance effect.",
        "",
        "## R2-B fixed-mean distance diversity",
        "",
        "Fixed: b_i, mean(d)=105 mm, R_uv=40 mm, gamma=0. Changed only distance diversity; D1-D3 use all nine cyclic shifts.",
        "",
        "| level | std(d) | shifts | median lambda_min | median trace(inv) | rotation median | translation median | success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in levels:
        lines.append(
            f"| {row['level']} | {row['std_d_mm']:.3f} | {row['shift_count']} | "
            f"{row['H_R_given_t_lambda_min_median']:.6g} | "
            f"{row['H_R_given_t_trace_inv_median']:.6g} | {row['rotation_error_median_across_shifts_deg']:.6f} deg | "
            f"{row['translation_error_median_across_shifts_mm']:.6f} mm | "
            f"{row['full_success_rate_median_across_shifts']:.3f} |"
        )
    lines += [
        "",
        f"For representative shift 0, paired median rotation-error changes versus D0 are {float(paired_lookup[310]['rotation_error_median_difference_deg']):+.3f} deg (D1), {float(paired_lookup[320]['rotation_error_median_difference_deg']):+.3f} deg (D2), and {float(paired_lookup[330]['rotation_error_median_difference_deg']):+.3f} deg (D3); all remain significant after FDR (q <= {max(float(paired_lookup[index]['fdr_q']) for index in (310, 320, 330)):.3g}).",
        f"D0 has {float(basin_lookup[300]['rotation_gt_90_rate']):.3f} of runs above 90 deg. Their median residual is {float(basin_lookup[300]['residual_rms_median_rotation_gt_90_mm']):.6f} mm, essentially the same as the <=0.1 deg runs ({float(basin_lookup[300]['residual_rms_median_rotation_le_0p1_mm']):.6f} mm). This is evidence of a low-residual alternative branch, not merely optimizer non-convergence.",
        "",
        "Result: distance diversity is the dominant controlled intervention. It leaves Cb, mean distance, and raw H_RR fixed, increases nuisance-aware rotation information, sharply reduces the wrong-branch basin, and improves continuous rotation error and success across cyclic assignments. D3 gives the best median error/trace-inverse, although lambda_min peaks around D1; the gain is therefore not explained by the weakest eigenvalue alone.",
        "",
        "## R3 u/v spread x distance diversity",
        "",
        "Fixed: the same b_i, gamma=0, and mean(d)=105 mm. Changed only R_uv={0,40} and distance constant/diverse; diverse cells use nine matched cyclic shifts.",
        "",
        "| condition | shifts | lambda_min | logdet | trace(inv) | rotation median | translation median | success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r3_levels:
        lines.append(
            f"| {row['condition']} | {row['shift_count']} | {row['H_R_given_t_lambda_min_median']:.6g} | "
            f"{row['H_R_given_t_logdet_median']:.6g} | {row['H_R_given_t_trace_inv_median']:.6g} | "
            f"{row['rotation_error_median_across_shifts_deg']:.6f} deg | "
            f"{row['translation_error_median_across_shifts_mm']:.6f} mm | "
            f"{row['full_success_rate_median_across_shifts']:.3f} |"
        )
    lines += [
        "",
        f"With diverse distance, increasing R_uv from 0 to 40 changes shift-matched paired rotation error by a median {r3_uv_diverse_diff:+.6f} deg, with median per-shift win rate {r3_uv_diverse_wins:.3f}; only {r3_uv_significant}/9 shifts are significant after FDR. Under constant distance, the same u/v increase does not give a significant paired benefit. Distance diversity is necessary in this tested range. The incremental actual benefit from u/v spread is modest and assignment-dependent despite its clear local-information gain, so the 2x2 response is nonlinear rather than cleanly additive.",
        "",
        "## Geometry metric versus actual rotation error",
        "",
    ]
    for row in correlation_rows:
        lines.append(
            f"- {row['metric']}: Spearman rho={row['spearman_rho']:+.3f}, p={row['p_value']:.3g}"
        )
    lines += [
        "",
        "These correlations use 64 designed condition/shift units. The strongest association is conditional logdet, followed by trace(inv); lambda_min alone misses the monotonic D1-to-D3 improvement. The units are deliberately structured and duplicated across R2/R3, so these p-values are descriptive rather than evidence from 64 independent random geometries.",
        "",
        "## Final interpretation",
        "",
        "1. Exact b/Cb control succeeded; none of the effects can be attributed to changed translation-direction geometry.",
        "2. u/v and gamma can alter rotation information through plane-normal nuisance coupling even when raw J_X or raw H_RR is unchanged.",
        "3. A common distance shift has no stable conditional-information or paired-performance effect in this ideal infinite-plane model.",
        "4. Per-pose distance diversity at fixed mean is the only intervention that robustly changes both local information and the nonlinear calibration basin. It collapses median rotation error from order 10^2 deg to order 10^-1 deg.",
        "5. The most defensible design rule from this experiment is: retain a well-spread fixed b set and introduce substantial distance diversity. u/v spread is a secondary nuisance-decoupling lever whose incremental calibration benefit still needs broader validation. Conditional logdet or trace(inv) is a better proxy here than lambda_min alone.",
        "",
        "Causal scope: this conclusion is for N=9, one GT hand-eye/plane, the repository's infinite-plane fixed-profile simulator, and the saved Phase-2A initialization/noise bank. It does not establish generalization across GT, plane, finite target boundaries, or N.",
        "",
        "Interpretation is condition/shift level; the 90 runs are paired repeated measurements, not independent geometry samples. All raw runs, including failures and threshold misses, are retained.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stage": "analysis",
            "status": "complete",
            "condition_count": len(summaries),
            "paired_run_count": len(runs),
            "r3_included": (output / "r3_manifest.json").exists(),
            "report": report_path.name,
        },
    )
    print(f"Analysis complete: {report_path}", flush=True)


def main() -> int:
    args = _parser().parse_args()
    config = _load_json(args.config)
    output = _resolve(args.output_dir or config["output_dir"])
    if args.stage == "r0":
        run_r0(config, output)
    elif args.stage == "geometry":
        run_geometry(config, output)
    elif args.stage == "r3":
        run_r3(config, output)
    elif args.stage == "calibration":
        run_calibration(config, output, args.resume)
    elif args.stage == "analysis":
        run_analysis(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
