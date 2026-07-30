from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from laser_handeye.active_fisher import (
    PlaneEstimate,
    apply_joint_local_update,
    fisher_objective_value,
    joint_residual_jacobian,
    marginal_handeye_information,
    predict_profile_points,
    predicted_candidate_information,
)
from laser_handeye.data import LaserScan
from laser_handeye.se3 import make_T


def _zero_residual_scan() -> tuple[LaserScan, np.ndarray, PlaneEstimate]:
    T_ef_s = make_T(
        Rotation.from_euler("xyz", [13.0, -7.0, 21.0], degrees=True)
        .as_matrix(),
        np.array([31.0, -17.0, 54.0]),
    )
    T_base_ef = make_T(
        Rotation.from_euler("xyz", [-18.0, 11.0, 9.0], degrees=True)
        .as_matrix(),
        np.array([120.0, -40.0, 380.0]),
    )
    plane = PlaneEstimate(
        np.array([0.23, -0.31, 0.922]),
        530.0,
    )
    x_values = np.linspace(-25.0, 25.0, 16)
    points = predict_profile_points(
        T_base_ef,
        T_ef_s,
        plane,
        x_values,
    )
    assert points is not None
    return (
        LaserScan(
            T_base_ef=T_base_ef,
            points_s=points,
            plane_id=0,
            scan_id=0,
        ),
        T_ef_s,
        plane,
    )


def test_joint_analytic_jacobian_matches_right_local_finite_difference() -> None:
    scan, T_ef_s, plane = _zero_residual_scan()
    groups = {0: [scan]}
    planes = {0: plane}
    scales = np.asarray(
        [
            *([np.deg2rad(10.0)] * 3),
            *([200.0] * 3),
            np.deg2rad(20.0),
            np.deg2rad(20.0),
            100.0,
        ]
    )
    residual, analytic = joint_residual_jacobian(
        groups,
        T_ef_s,
        planes,
        profile_noise_std_mm=0.25,
        noise_axis="xz",
        parameter_scales=scales,
    )
    assert np.max(np.abs(residual)) < 1e-10

    epsilon = 1e-6
    numerical = np.zeros_like(analytic)
    for column in range(analytic.shape[1]):
        normalized_update = np.zeros(analytic.shape[1])
        normalized_update[column] = epsilon
        plus_T, plus_planes = apply_joint_local_update(
            T_ef_s,
            planes,
            scales * normalized_update,
        )
        minus_T, minus_planes = apply_joint_local_update(
            T_ef_s,
            planes,
            -scales * normalized_update,
        )
        plus, _ = joint_residual_jacobian(
            groups,
            plus_T,
            plus_planes,
            profile_noise_std_mm=0.25,
            noise_axis="xz",
            parameter_scales=scales,
        )
        minus, _ = joint_residual_jacobian(
            groups,
            minus_T,
            minus_planes,
            profile_noise_std_mm=0.25,
            noise_axis="xz",
            parameter_scales=scales,
        )
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)

    assert np.allclose(analytic, numerical, rtol=2e-5, atol=2e-5)


def test_whitened_jacobian_matches_nonzero_residual_finite_difference() -> None:
    scan, T_ef_s, plane = _zero_residual_scan()
    perturbed_points = scan.points_s.copy()
    perturbed_points[:, 0] += np.linspace(-0.08, 0.06, len(scan.points_s))
    perturbed_points[:, 2] += np.linspace(0.12, -0.04, len(scan.points_s))
    perturbed_scan = LaserScan(
        T_base_ef=scan.T_base_ef,
        points_s=perturbed_points,
        plane_id=scan.plane_id,
        scan_id=scan.scan_id,
    )
    groups = {0: [perturbed_scan]}
    planes = {0: plane}
    scales = np.asarray(
        [
            *([np.deg2rad(2.0)] * 3),
            *([10.0] * 3),
            np.deg2rad(20.0),
            np.deg2rad(20.0),
            100.0,
        ]
    )

    for noise_axis in ("xz", "z"):
        _residual, analytic = joint_residual_jacobian(
            groups,
            T_ef_s,
            planes,
            profile_noise_std_mm=0.25,
            noise_axis=noise_axis,
            parameter_scales=scales,
        )
        epsilon = 1e-6
        numerical = np.zeros_like(analytic)
        for column in range(analytic.shape[1]):
            normalized_update = np.zeros(analytic.shape[1])
            normalized_update[column] = epsilon
            plus_T, plus_planes = apply_joint_local_update(
                T_ef_s,
                planes,
                scales * normalized_update,
            )
            minus_T, minus_planes = apply_joint_local_update(
                T_ef_s,
                planes,
                -scales * normalized_update,
            )
            plus, _ = joint_residual_jacobian(
                groups,
                plus_T,
                plus_planes,
                profile_noise_std_mm=0.25,
                noise_axis=noise_axis,
                parameter_scales=scales,
            )
            minus, _ = joint_residual_jacobian(
                groups,
                minus_T,
                minus_planes,
                profile_noise_std_mm=0.25,
                noise_axis=noise_axis,
                parameter_scales=scales,
            )
            numerical[:, column] = (plus - minus) / (2.0 * epsilon)

        assert np.allclose(analytic, numerical, rtol=3e-5, atol=3e-5)


def test_schur_covariance_matches_joint_inverse_handeye_block() -> None:
    rng = np.random.default_rng(9)
    design = rng.normal(size=(80, 15))
    information = design.T @ design
    marginal = marginal_handeye_information(information)
    joint_covariance = np.linalg.inv(information)
    assert np.allclose(
        np.linalg.inv(marginal),
        joint_covariance[:6, :6],
        rtol=1e-10,
        atol=1e-10,
    )


def test_predicted_candidate_information_does_not_need_future_points() -> None:
    scan, T_ef_s, plane = _zero_residual_scan()
    scales = np.asarray(
        [
            *([np.deg2rad(10.0)] * 3),
            *([200.0] * 3),
            np.deg2rad(20.0),
            np.deg2rad(20.0),
            100.0,
        ]
    )
    residual, jacobian = joint_residual_jacobian(
        {0: [scan]},
        T_ef_s,
        {0: plane},
        profile_noise_std_mm=0.25,
        noise_axis="xz",
        parameter_scales=scales,
    )

    from laser_handeye.active_fisher import JointCalibrationEstimate

    estimate = JointCalibrationEstimate(
        T_ef_s=T_ef_s,
        planes={0: plane},
        information=jacobian.T @ jacobian,
        whitened_cost=float(residual @ residual),
        iterations=0,
        converged=True,
        data_rank=int(np.linalg.matrix_rank(jacobian)),
    )
    candidate = predicted_candidate_information(
        T_base_ef=scan.T_base_ef,
        plane_id=0,
        estimate=estimate,
        x_values=scan.points_s[:, 0],
        parameter_scales=scales,
        profile_noise_std_mm=0.25,
        noise_axis="xz",
        depth_range_mm=None,
    )
    assert candidate is not None
    assert np.allclose(candidate, jacobian.T @ jacobian)


def test_fisher_objectives_use_marginal_handeye_information() -> None:
    rng = np.random.default_rng(13)
    design = rng.normal(size=(100, 9))
    information = design.T @ design
    marginal = marginal_handeye_information(information)
    assert np.isclose(
        fisher_objective_value(information, "e_optimal"),
        np.linalg.eigvalsh(marginal)[0],
    )
    assert np.isclose(
        fisher_objective_value(information, "d_optimal"),
        0.5 * np.linalg.slogdet(marginal)[1],
    )
