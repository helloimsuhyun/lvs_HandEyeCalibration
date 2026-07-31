from __future__ import annotations

import numpy as np
import pytest

from laser_handeye.nonlinear_refinement import (
    point_to_plane_residuals,
    refine_handeye_nonlinear,
    refine_handeye_planes_nonlinear,
)
from laser_handeye.se3 import euler_xyz_deg, make_T, rot_error_deg
from laser_handeye.simulation import (
    random_robot_poses,
    simulate_profile_on_plane,
)


def _synthetic_problem(
    n_planes: int,
) -> tuple[dict[int, list], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20 + n_planes)
    T_true = make_T(
        euler_xyz_deg(12.0, -7.0, 18.0),
        np.array([45.0, -20.0, 80.0]),
    )
    all_planes = [
        (np.array([0.1, 0.2, 1.0]), 450.0),
        (np.array([1.0, -0.2, 0.1]), 120.0),
        (np.array([-0.2, -1.0, 0.2]), 80.0),
    ]

    groups = {}
    for plane_id, (normal, offset) in enumerate(all_planes[:n_planes]):
        normal = normal / np.linalg.norm(normal)
        scans = []
        for pose in random_robot_poses(30, rng):
            try:
                scans.append(
                    simulate_profile_on_plane(
                        pose,
                        T_true,
                        normal,
                        offset,
                        np.linspace(-20.0, 20.0, 21),
                        noise_std=0.02,
                        rng=rng,
                        plane_id=plane_id,
                    )
                )
            except ValueError:
                continue
        groups[plane_id] = scans

    T_initial = T_true.copy()
    T_initial[:3, :3] = (
        T_true[:3, :3] @ euler_xyz_deg(1.0, -0.5, 0.7)
    )
    T_initial[:3, 3] += np.array([2.0, -1.0, 1.5])
    return groups, T_true, T_initial


@pytest.mark.parametrize("n_planes", [1, 3])
def test_joint_refinement_estimates_unit_normals_and_handeye(
    n_planes: int,
) -> None:
    groups, T_true, T_initial = _synthetic_problem(n_planes)

    result = refine_handeye_planes_nonlinear(
        groups,
        T_initial,
        loss="linear",
        max_nfev=100,
    )

    assert result.success
    assert result.variable_count == 6 + 3 * n_planes
    assert result.jacobian_rank == result.variable_count
    assert result.final_rms_mm < 0.03
    assert result.final_rms_mm < 0.05 * result.initial_rms_mm
    assert set(result.plane_normals) == set(groups)
    assert set(result.plane_offsets_mm) == set(groups)
    for normal in result.plane_normals.values():
        assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12)

    translation_error = np.linalg.norm(
        result.T_ef_s[:3, 3] - T_true[:3, 3]
    )
    rotation_error = rot_error_deg(
        result.T_ef_s[:3, :3],
        T_true[:3, :3],
    )
    assert translation_error < 0.02
    assert rotation_error < 0.03


def test_fixed_normals_profiles_offset_for_each_handeye_candidate() -> None:
    groups, _T_true, T_initial = _synthetic_problem(1)
    normal = np.array([0.1, 0.2, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    normalized_groups = list(groups.items())

    residuals_near = point_to_plane_residuals(
        normalized_groups,
        T_initial,
        plane_mode="fixed_normals",
        fixed_planes={0: (normal, 10.0)},
    )
    residuals_far = point_to_plane_residuals(
        normalized_groups,
        T_initial,
        plane_mode="fixed_normals",
        fixed_planes={0: (normal, 10_000.0)},
    )
    fixed_offset_residuals = point_to_plane_residuals(
        normalized_groups,
        T_initial,
        plane_mode="fixed",
        fixed_planes={0: (normal, 10.0)},
    )

    assert np.allclose(residuals_near, residuals_far)
    assert np.isclose(np.mean(residuals_near), 0.0, atol=1e-12)
    assert np.sum(residuals_near**2) < np.sum(fixed_offset_residuals**2)

    result_near = refine_handeye_nonlinear(
        groups,
        T_initial,
        plane_mode="fixed_normals",
        planes={0: (normal, 10.0)},
        loss="linear",
        max_nfev=100,
    )
    result_far = refine_handeye_nonlinear(
        groups,
        T_initial,
        plane_mode="fixed_normals",
        planes={0: (normal, 10_000.0)},
        loss="linear",
        max_nfev=100,
    )

    assert result_near.success
    assert result_far.success
    assert np.allclose(result_near.T_ef_s, result_far.T_ef_s, atol=1e-9)
    assert np.isclose(
        result_near.final_rms_mm,
        result_far.final_rms_mm,
        atol=1e-12,
    )
