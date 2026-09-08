from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from robust_laser_handeye.laser_handeye.data import PlaneFrame
from robust_laser_handeye.laser_handeye.scene_generation import plane_basis


@dataclass(frozen=True)
class JacobianMetrics:
    shape: tuple[int, int]
    rank: int

    sigma_max: float
    sigma_min: float
    cond_j: float

    lambda_max: float
    lambda_min: float
    cond_h: float

    trace_h: float
    logdet_h: float
    half_logdet_h: float
    trace_h_inv: float

    eigenvalues: np.ndarray


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def sensor_profile_points_on_plane(
    T_base_s: np.ndarray,
    frame: PlaneFrame,
    x_values: np.ndarray,
) -> np.ndarray:

    T_base_s = np.asarray(
        T_base_s,
        dtype=float,
    ).reshape(4, 4)

    x_values = np.asarray(
        x_values,
        dtype=float,
    ).reshape(-1)

    R = T_base_s[:3, :3]
    t = T_base_s[:3, 3]

    n = np.asarray(
        frame.n,
        dtype=float,
    ).reshape(3)

    l = float(frame.offset_mm)

    r_x = R[:, 0]
    r_z = R[:, 2]

    denominator = float(n @ r_z)

    if abs(denominator) < 1e-12:
        raise RuntimeError(
            "profile plane is nearly parallel to target plane"
        )

    z_values = (
        l
        - n @ t
        - x_values * (n @ r_x)
    ) / denominator

    return np.column_stack(
        [
            x_values,
            np.zeros_like(x_values),
            z_values,
        ]
    )


def fit_plane_to_points(
    points: np.ndarray,
    reference_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:

    points = np.asarray(
        points,
        dtype=float,
    ).reshape(-1, 3)

    center = points.mean(axis=0)

    centered = points - center

    H = centered.T @ centered

    _, eigenvectors = np.linalg.eigh(H)

    n = eigenvectors[:, 0]
    n = n / np.linalg.norm(n)

    if reference_normal is not None:
        reference_normal = np.asarray(
            reference_normal,
            dtype=float,
        ).reshape(3)

        if float(n @ reference_normal) < 0.0:
            n = -n

    l = float(n @ center)

    u, v = plane_basis(n)

    return n, l, u, v


def compute_jacobian_metrics(
    J: np.ndarray,
) -> JacobianMetrics:

    J = np.asarray(
        J,
        dtype=float,
    )

    if J.ndim != 2:
        raise ValueError("J must be a 2-D matrix")

    if J.shape[1] == 0:
        raise ValueError("J must have at least one column")

    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    H = J.T @ J

    eigenvalues = np.linalg.eigvalsh(H)

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])

    tol = (
        max(J.shape)
        * np.finfo(float).eps
        * sigma_max
    )

    rank = int(
        np.sum(singular_values > tol)
    )

    lambda_max = float(eigenvalues[-1])
    lambda_min = float(eigenvalues[0])

    full_column_rank = (
        rank == J.shape[1]
    )

    if full_column_rank and sigma_min > 0.0:
        cond_j = float(
            sigma_max / sigma_min
        )
    else:
        cond_j = np.inf

    if full_column_rank and lambda_min > 0.0:
        cond_h = float(
            lambda_max / lambda_min
        )

        logdet_h = float(
            np.sum(
                np.log(eigenvalues)
            )
        )

        half_logdet_h = (
            0.5 * logdet_h
        )

        trace_h_inv = float(
            np.sum(
                1.0 / eigenvalues
            )
        )

    else:
        cond_h = np.inf
        logdet_h = -np.inf
        half_logdet_h = -np.inf
        trace_h_inv = np.inf

    trace_h = float(
        np.trace(H)
    )

    return JacobianMetrics(
        shape=J.shape,
        rank=rank,

        sigma_max=sigma_max,
        sigma_min=sigma_min,
        cond_j=cond_j,

        lambda_max=lambda_max,
        lambda_min=lambda_min,
        cond_h=cond_h,

        trace_h=trace_h,
        logdet_h=logdet_h,
        half_logdet_h=half_logdet_h,
        trace_h_inv=trace_h_inv,

        eigenvalues=eigenvalues,
    )


def print_jacobian_metrics(
    name: str,
    J: np.ndarray,
) -> JacobianMetrics:

    metrics = compute_jacobian_metrics(J)

    print(f"\n[{name}]")

    print(
        "shape          :",
        metrics.shape,
    )

    print(
        "rank           :",
        metrics.rank,
        "/",
        metrics.shape[1],
    )

    print("\nSingular values")

    print(
        "sigma_max      :",
        metrics.sigma_max,
    )

    print(
        "sigma_min      :",
        metrics.sigma_min,
    )

    print(
        "cond(J)        :",
        metrics.cond_j,
    )

    print("\nInformation matrix H = J^T J")

    print(
        "lambda_max     :",
        metrics.lambda_max,
    )

    print(
        "lambda_min     :",
        metrics.lambda_min,
    )

    print(
        "cond(H)        :",
        metrics.cond_h,
    )

    print(
        "trace(H)       :",
        metrics.trace_h,
    )

    print(
        "logdet(H)      :",
        metrics.logdet_h,
    )

    print(
        "0.5 logdet(H)  :",
        metrics.half_logdet_h,
    )

    print(
        "trace(H^-1)    :",
        metrics.trace_h_inv,
    )

    print(
        "eigenvalues(H) :",
        metrics.eigenvalues,
    )

    return metrics


def build_calibration_jacobians(
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    T_ef_s_true: np.ndarray,
    T_eval: np.ndarray,
    x_values: np.ndarray,
    rotation_characteristic_length_mm: float = 100.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    if rotation_characteristic_length_mm <= 0.0:
        raise ValueError(
            "rotation_characteristic_length_mm must be positive"
        )

    T_ef_s_true = np.asarray(
        T_ef_s_true,
        dtype=float,
    ).reshape(4, 4)

    T_eval = np.asarray(
        T_eval,
        dtype=float,
    ).reshape(4, 4)

    R_es = T_eval[:3, :3]
    t_es = T_eval[:3, 3]

    T_s_ef_true = np.linalg.inv(
        T_ef_s_true
    )

    robot_poses = []
    sensor_points_per_scan = []

    for T_base_s in sensor_poses:

        T_base_s = np.asarray(
            T_base_s,
            dtype=float,
        ).reshape(4, 4)

        T_base_ef = (
            T_base_s
            @ T_s_ef_true
        )

        robot_poses.append(
            T_base_ef
        )

        sensor_points_per_scan.append(
            sensor_profile_points_on_plane(
                T_base_s=T_base_s,
                frame=frame,
                x_values=x_values,
            )
        )

    all_points_base = []

    for T_base_ef, points_s in zip(
        robot_poses,
        sensor_points_per_scan,
    ):

        R_be = T_base_ef[:3, :3]
        t_be = T_base_ef[:3, 3]

        points_ef = (
            R_es @ points_s.T
        ).T + t_es

        points_base = (
            R_be @ points_ef.T
        ).T + t_be

        all_points_base.append(
            points_base
        )

    all_points_base = np.vstack(
        all_points_base
    )

    n, _, plane_u, plane_v = (
        fit_plane_to_points(
            all_points_base,
            reference_normal=frame.n,
        )
    )

    J_x_rows = []
    J_plane_rows = []
    J_translation_rows = []

    point_index = 0

    for T_base_ef, points_s in zip(
        robot_poses,
        sensor_points_per_scan,
    ):

        R_be = T_base_ef[:3, :3]

        j_translation = (
            n @ R_be
        )

        J_translation_rows.append(
            j_translation
        )

        for p_s in points_s:

            j_rotation = (
                -n
                @ R_be
                @ R_es
                @ skew(p_s)
            )

            j_rotation_scaled = (
                j_rotation
                / rotation_characteristic_length_mm
            )

            J_x_rows.append(
                np.concatenate(
                    [
                        j_rotation_scaled,
                        j_translation,
                    ]
                )
            )

            p_base = all_points_base[
                point_index
            ]

            J_plane_rows.append(
                [
                    float(
                        plane_u
                        @ p_base
                    ),
                    float(
                        plane_v
                        @ p_base
                    ),
                    -1.0,
                ]
            )

            point_index += 1

    J_x = np.asarray(
        J_x_rows,
        dtype=float,
    )

    J_plane = np.asarray(
        J_plane_rows,
        dtype=float,
    )

    J_translation = np.asarray(
        J_translation_rows,
        dtype=float,
    )

    J_full = np.hstack(
        [
            J_x,
            J_plane,
        ]
    )

    P_plane = (
        J_plane
        @ np.linalg.pinv(
            J_plane
        )
    )

    J_eff = (
        np.eye(
            J_x.shape[0]
        )
        - P_plane
    ) @ J_x

    return (
        J_translation,
        J_x,
        J_plane,
        J_full,
        J_eff,
    )


def analyze_calibration_jacobian(
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    T_ef_s_true: np.ndarray,
    T_eval: np.ndarray,
    x_values: np.ndarray,
    rotation_characteristic_length_mm: float = 100.0,
    label: str = "",
) -> dict[str, JacobianMetrics]:

    (
        J_translation,
        J_x,
        J_plane,
        J_full,
        J_eff,
    ) = build_calibration_jacobians(
        sensor_poses=sensor_poses,
        frame=frame,
        T_ef_s_true=T_ef_s_true,
        T_eval=T_eval,
        x_values=x_values,
        rotation_characteristic_length_mm=(
            rotation_characteristic_length_mm
        ),
    )

    print("\n" + "=" * 72)
    print(
        f"Jacobian analysis: {label}"
    )
    print("=" * 72)

    translation_metrics = (
        print_jacobian_metrics(
            "Translation J_t",
            J_translation,
        )
    )

    fixed_plane_metrics = (
        print_jacobian_metrics(
            "Hand-eye J_X (plane fixed)",
            J_x,
        )
    )

    plane_metrics = (
        print_jacobian_metrics(
            "Plane J_pi",
            J_plane,
        )
    )

    full_metrics = (
        print_jacobian_metrics(
            "Full J = [J_X J_pi]",
            J_full,
        )
    )

    effective_metrics = (
        print_jacobian_metrics(
            "Hand-eye J_eff (plane nuisance removed)",
            J_eff,
        )
    )

    return {
        "translation": translation_metrics,
        "fixed_plane": fixed_plane_metrics,
        "plane": plane_metrics,
        "full": full_metrics,
        "effective": effective_metrics,
    }


# =============================================================================
# Global ambiguity analysis
# =============================================================================

@dataclass(frozen=True)
class ProfileLineGeometry:
    """
    Exact affine representation of each noiseless laser profile line:

        p_i^S(x_c) = c_i + x_c s_i

    where x_c = x - mean(x).

    centers_s     : shape (N, 3), c_i
    directions_s  : shape (N, 3), s_i = dp/dx
    point_count   : number of profile points per scan
    s_xx          : sum_j x_c,j^2
    """

    centers_s: np.ndarray
    directions_s: np.ndarray
    point_count: int
    s_xx: float


@dataclass(frozen=True)
class AmbiguityDecompositionResult:
    """
    Exact decomposition of the best-plane SSE at one fixed relative transform.

    For transformed profile lines

        p_i(x) = a_i + x b_i

    with centered x samples, the total scatter is

        H_total = H_center + H_direction

        H_center
          = M sum_i (a_i-a_bar)(a_i-a_bar)^T

        H_direction
          = S_xx sum_i b_i b_i^T.

    Let n_star be the eigenvector of H_total associated with lambda_min.
    This n_star is the normal of the single best-fit plane for the COMPLETE
    profile set.

    The exact attribution at that SAME plane normal is

        center_sse    = n_star^T H_center n_star
        direction_sse = n_star^T H_direction n_star
        total_sse     = center_sse + direction_sse.

    center_rms_contribution_mm and direction_rms_contribution_mm do not add
    linearly. Their squares add to total_rms_mm^2.
    """

    best_plane_normal: np.ndarray

    total_sse: float
    center_sse: float
    direction_sse: float

    total_rms_mm: float
    center_rms_contribution_mm: float
    direction_rms_contribution_mm: float

    center_fraction: float
    direction_fraction: float

    lambda_min_center: float
    lambda_min_direction: float

    eigenvalues_total: np.ndarray
    eigenvalues_center: np.ndarray
    eigenvalues_direction: np.ndarray


@dataclass(frozen=True)
class TranslationOptimizedRotationResult:
    rotation_angle_deg: float
    rotation_axis: np.ndarray
    translation_mm: np.ndarray
    rms_mm: float
    optimizer_success: bool
    decomposition: AmbiguityDecompositionResult


@dataclass(frozen=True)
class GlobalAmbiguityMetrics:
    """
    GT-independent global ambiguity diagnostics.

    Larger min_far_rms_mm is better.

    y180_decomposition and min_far_decomposition separate the residual at the
    SAME best-fit plane normal into center and direction contributions.
    """

    identity_rms_mm: float

    y180_rms_mm: float
    y180_translation_mm: np.ndarray
    y180_optimizer_success: bool
    y180_decomposition: AmbiguityDecompositionResult

    min_far_rms_mm: float
    min_far_angle_deg: float
    min_far_axis: np.ndarray
    min_far_translation_mm: np.ndarray
    min_far_decomposition: AmbiguityDecompositionResult

    min_angle_deg: float
    angle_step_deg: float
    num_axes: int
    evaluated_rotation_count: int



def rotation_matrix_from_axis_angle(
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:

    axis = np.asarray(
        axis,
        dtype=float,
    ).reshape(3)

    norm = float(
        np.linalg.norm(axis)
    )

    if norm <= 0.0:
        raise ValueError(
            "rotation axis must be non-zero"
        )

    axis = axis / norm

    theta = np.radians(
        float(angle_deg)
    )

    K = skew(axis)

    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )



def build_profile_line_geometry(
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    x_values: np.ndarray,
) -> ProfileLineGeometry:
    """
    Build the exact line representation of every noiseless laser profile.

    This uses only the designed sensor poses and target plane geometry.
    It does NOT require the true hand-eye transform or a current estimate.
    """

    x_values = np.asarray(
        x_values,
        dtype=float,
    ).reshape(-1)

    if len(x_values) < 2:
        raise ValueError(
            "x_values must contain at least two samples"
        )

    x_centered = (
        x_values
        - float(np.mean(x_values))
    )

    s_xx = float(
        x_centered
        @ x_centered
    )

    if s_xx <= 0.0:
        raise ValueError(
            "x_values must have non-zero spread"
        )

    centers_s = []
    directions_s = []

    for T_base_s in sensor_poses:

        points_s = sensor_profile_points_on_plane(
            T_base_s=T_base_s,
            frame=frame,
            x_values=x_values,
        )

        center_s = np.mean(
            points_s,
            axis=0,
        )

        direction_s = (
            x_centered[:, None]
            * points_s
        ).sum(axis=0) / s_xx

        # Exact line check. This should be at numerical precision because the
        # profile-plane intersection is affine in x.
        reconstructed = (
            center_s[None, :]
            + x_centered[:, None]
            * direction_s[None, :]
        )

        max_line_error = float(
            np.max(
                np.linalg.norm(
                    reconstructed - points_s,
                    axis=1,
                )
            )
        )

        if max_line_error > 1e-8:
            raise RuntimeError(
                "profile points are not sufficiently affine in x; "
                f"max error={max_line_error:.3e} mm"
            )

        centers_s.append(
            center_s
        )

        directions_s.append(
            direction_s
        )

    return ProfileLineGeometry(
        centers_s=np.asarray(
            centers_s,
            dtype=float,
        ),
        directions_s=np.asarray(
            directions_s,
            dtype=float,
        ),
        point_count=len(x_values),
        s_xx=s_xx,
    )



def relative_transform_profile_scatter_components(
    sensor_poses: list[np.ndarray],
    line_geometry: ProfileLineGeometry,
    delta_R: np.ndarray,
    delta_t_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact center / direction decomposition of transformed profile scatter.

    H_center
      = M sum_i (a_i-a_bar)(a_i-a_bar)^T

    H_direction
      = S_xx sum_i b_i b_i^T

    H_total
      = H_center + H_direction
    """

    delta_R = np.asarray(
        delta_R,
        dtype=float,
    ).reshape(3, 3)

    delta_t_mm = np.asarray(
        delta_t_mm,
        dtype=float,
    ).reshape(3)

    if len(sensor_poses) != len(
        line_geometry.centers_s
    ):
        raise ValueError(
            "sensor_poses and line geometry size mismatch"
        )

    centers_base = []
    directions_base = []

    for T_base_s, center_s, direction_s in zip(
        sensor_poses,
        line_geometry.centers_s,
        line_geometry.directions_s,
    ):
        T_base_s = np.asarray(
            T_base_s,
            dtype=float,
        ).reshape(4, 4)

        R_i = T_base_s[:3, :3]
        t_i = T_base_s[:3, 3]

        a_i = (
            t_i
            + R_i
            @ (
                delta_R @ center_s
                + delta_t_mm
            )
        )

        b_i = (
            R_i
            @ delta_R
            @ direction_s
        )

        centers_base.append(a_i)
        directions_base.append(b_i)

    centers_base = np.asarray(
        centers_base,
        dtype=float,
    )

    directions_base = np.asarray(
        directions_base,
        dtype=float,
    )

    mean_center = np.mean(
        centers_base,
        axis=0,
    )

    centered_centers = (
        centers_base
        - mean_center
    )

    H_center = (
        line_geometry.point_count
        * centered_centers.T
        @ centered_centers
    )

    H_direction = (
        line_geometry.s_xx
        * directions_base.T
        @ directions_base
    )

    H_center = 0.5 * (
        H_center + H_center.T
    )

    H_direction = 0.5 * (
        H_direction + H_direction.T
    )

    H_total = (
        H_center
        + H_direction
    )

    H_total = 0.5 * (
        H_total + H_total.T
    )

    return (
        H_center,
        H_direction,
        H_total,
    )



def relative_transform_profile_scatter(
    sensor_poses: list[np.ndarray],
    line_geometry: ProfileLineGeometry,
    delta_R: np.ndarray,
    delta_t_mm: np.ndarray,
) -> np.ndarray:

    _, _, H_total = (
        relative_transform_profile_scatter_components(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=delta_R,
            delta_t_mm=delta_t_mm,
        )
    )

    return H_total



def relative_transform_best_plane_decomposition(
    sensor_poses: list[np.ndarray],
    line_geometry: ProfileLineGeometry,
    delta_R: np.ndarray,
    delta_t_mm: np.ndarray,
) -> AmbiguityDecompositionResult:
    """
    Attribute the best-plane SSE at the SAME optimal plane normal n_star.
    """

    (
        H_center,
        H_direction,
        H_total,
    ) = relative_transform_profile_scatter_components(
        sensor_poses=sensor_poses,
        line_geometry=line_geometry,
        delta_R=delta_R,
        delta_t_mm=delta_t_mm,
    )

    (
        eigenvalues_total,
        eigenvectors_total,
    ) = np.linalg.eigh(
        H_total
    )

    eigenvalues_center = np.linalg.eigvalsh(
        H_center
    )

    eigenvalues_direction = np.linalg.eigvalsh(
        H_direction
    )

    n_star = np.asarray(
        eigenvectors_total[:, 0],
        dtype=float,
    )

    n_star = (
        n_star
        / np.linalg.norm(n_star)
    )

    center_sse = max(
        0.0,
        float(
            n_star
            @ H_center
            @ n_star
        ),
    )

    direction_sse = max(
        0.0,
        float(
            n_star
            @ H_direction
            @ n_star
        ),
    )

    total_sse = (
        center_sse
        + direction_sse
    )

    num_points = (
        len(sensor_poses)
        * line_geometry.point_count
    )

    total_rms = float(
        np.sqrt(
            total_sse / num_points
        )
    )

    center_rms = float(
        np.sqrt(
            center_sse / num_points
        )
    )

    direction_rms = float(
        np.sqrt(
            direction_sse / num_points
        )
    )

    if total_sse > 1e-20:
        center_fraction = float(
            center_sse / total_sse
        )
        direction_fraction = float(
            direction_sse / total_sse
        )
    else:
        center_fraction = np.nan
        direction_fraction = np.nan

    return AmbiguityDecompositionResult(
        best_plane_normal=n_star,

        total_sse=float(total_sse),
        center_sse=float(center_sse),
        direction_sse=float(direction_sse),

        total_rms_mm=total_rms,
        center_rms_contribution_mm=center_rms,
        direction_rms_contribution_mm=direction_rms,

        center_fraction=center_fraction,
        direction_fraction=direction_fraction,

        lambda_min_center=max(
            0.0,
            float(eigenvalues_center[0]),
        ),
        lambda_min_direction=max(
            0.0,
            float(eigenvalues_direction[0]),
        ),

        eigenvalues_total=np.asarray(
            eigenvalues_total,
            dtype=float,
        ),
        eigenvalues_center=np.asarray(
            eigenvalues_center,
            dtype=float,
        ),
        eigenvalues_direction=np.asarray(
            eigenvalues_direction,
            dtype=float,
        ),
    )



def relative_transform_best_plane_rms(
    sensor_poses: list[np.ndarray],
    line_geometry: ProfileLineGeometry,
    delta_R: np.ndarray,
    delta_t_mm: np.ndarray,
) -> float:

    decomposition = (
        relative_transform_best_plane_decomposition(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=delta_R,
            delta_t_mm=delta_t_mm,
        )
    )

    return float(
        decomposition.total_rms_mm
    )


def _default_translation_starts(
    line_geometry: ProfileLineGeometry,
    translation_bound_mm: float,
) -> list[np.ndarray]:
    """
    Deterministic starts for Delta t optimization.

    Besides zero, include +/- 2*d along sensor z for each distinct profile
    center depth d. These starts deliberately cover the common 180-deg
    profile-line ambiguity branch.
    """

    bound = float(
        translation_bound_mm
    )

    starts = [
        np.zeros(3, dtype=float)
    ]

    depths = np.unique(
        np.round(
            line_geometry.centers_s[:, 2],
            decimals=9,
        )
    )

    for depth in depths:

        for sign in (
            1.0,
            -1.0,
        ):

            z = float(
                np.clip(
                    sign * 2.0 * depth,
                    -bound,
                    bound,
                )
            )

            starts.append(
                np.array(
                    [0.0, 0.0, z],
                    dtype=float,
                )
            )

    # Remove duplicates while preserving order.
    unique_starts = []

    for start in starts:

        if not any(
            np.allclose(
                start,
                previous,
                atol=1e-12,
                rtol=0.0,
            )
            for previous in unique_starts
        ):
            unique_starts.append(
                start
            )

    return unique_starts



def optimize_translation_for_relative_rotation(
    sensor_poses: list[np.ndarray],
    line_geometry: ProfileLineGeometry,
    delta_R: np.ndarray,
    translation_bound_mm: float = 400.0,
    maxiter: int = 250,
) -> tuple[np.ndarray, float, bool]:
    """
    For a fixed Delta R, numerically solve

        min_{Delta t} RMS_best_plane(Delta R, Delta t).

    n and l are NOT optimized numerically; they have already been eliminated
    exactly by the 3x3 scatter-matrix eigenvalue.
    """

    if translation_bound_mm <= 0.0:
        raise ValueError(
            "translation_bound_mm must be positive"
        )

    delta_R = np.asarray(
        delta_R,
        dtype=float,
    ).reshape(3, 3)

    starts = _default_translation_starts(
        line_geometry=line_geometry,
        translation_bound_mm=(
            translation_bound_mm
        ),
    )

    bounds = [
        (
            -float(translation_bound_mm),
            float(translation_bound_mm),
        )
    ] * 3

    best_t = None
    best_value = np.inf
    best_success = False

    def objective(
        delta_t_flat: np.ndarray,
    ) -> float:

        rms = relative_transform_best_plane_rms(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=delta_R,
            delta_t_mm=delta_t_flat,
        )

        # Squared RMS gives the optimizer a smoother scale near zero.
        return float(
            rms * rms
        )

    for start in starts:

        result = minimize(
            objective,
            x0=start,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "xtol": 1e-7,
                "ftol": 1e-14,
            },
        )

        value = float(
            result.fun
        )

        if (
            np.isfinite(value)
            and value < best_value
        ):

            best_value = value

            best_t = np.asarray(
                result.x,
                dtype=float,
            ).reshape(3)

            best_success = bool(
                result.success
            )

    if best_t is None:
        raise RuntimeError(
            "Delta t optimization failed for all starts"
        )

    return (
        best_t,
        float(
            np.sqrt(
                max(0.0, best_value)
            )
        ),
        best_success,
    )



def evaluate_y180_ambiguity(
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    x_values: np.ndarray,
    translation_bound_mm: float = 400.0,
) -> TranslationOptimizedRotationResult:
    """
    Targeted test for the sensor-y 180-degree competing branch.

    Smaller rms_mm => stronger ambiguity.
    """

    line_geometry = build_profile_line_geometry(
        sensor_poses=sensor_poses,
        frame=frame,
        x_values=x_values,
    )

    axis = np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    )

    delta_R = rotation_matrix_from_axis_angle(
        axis=axis,
        angle_deg=180.0,
    )

    best_t, best_rms, success = (
        optimize_translation_for_relative_rotation(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=delta_R,
            translation_bound_mm=(
                translation_bound_mm
            ),
        )
    )

    decomposition = (
        relative_transform_best_plane_decomposition(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=delta_R,
            delta_t_mm=best_t,
        )
    )

    return TranslationOptimizedRotationResult(
        rotation_angle_deg=180.0,
        rotation_axis=axis,
        translation_mm=best_t,
        rms_mm=best_rms,
        optimizer_success=success,
        decomposition=decomposition,
    )



def _fibonacci_axes(
    num_axes: int,
) -> np.ndarray:
    """
    Deterministic approximately uniform axes on S^2.

    Cardinal +/- axes are always included so x/y/z branches are never missed.
    """

    if num_axes < 0:
        raise ValueError(
            "num_axes must be non-negative"
        )

    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]

    if num_axes > 0:

        golden_angle = (
            np.pi
            * (3.0 - np.sqrt(5.0))
        )

        for k in range(num_axes):

            z = (
                1.0
                - 2.0
                * (k + 0.5)
                / num_axes
            )

            radius = np.sqrt(
                max(
                    0.0,
                    1.0 - z * z,
                )
            )

            phi = (
                golden_angle
                * k
            )

            axes.append(
                np.array(
                    [
                        radius * np.cos(phi),
                        radius * np.sin(phi),
                        z,
                    ],
                    dtype=float,
                )
            )

    # Remove near duplicates.
    unique_axes = []

    for axis in axes:

        axis = axis / np.linalg.norm(axis)

        if not any(
            np.linalg.norm(
                axis - previous
            ) < 1e-10
            for previous in unique_axes
        ):
            unique_axes.append(
                axis
            )

    return np.asarray(
        unique_axes,
        dtype=float,
    )



def evaluate_global_ambiguity(
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    x_values: np.ndarray,
    min_angle_deg: float = 60.0,
    angle_step_deg: float = 15.0,
    num_fibonacci_axes: int = 12,
    translation_bound_mm: float = 400.0,
    translation_maxiter: int = 250,
) -> tuple[
    GlobalAmbiguityMetrics,
    list[dict],
]:
    """
    Approximate GT-independent global ambiguity search.

    Define

        Phi(Delta R)
          = min_{Delta t, n, l} RMS(Delta R, Delta t, n, l).

    For every sampled Delta R:
      1) Delta t is numerically optimized.
      2) n and l are eliminated exactly by best-plane PCA, equivalently by
         lambda_min of the exact profile scatter matrix.

    The returned min_far_rms_mm approximates

        G_alpha = min_{angle(Delta R, I) >= alpha} Phi(Delta R).

    Larger G_alpha is better global branch separation.

    This function does NOT use T_ef_s_true or T_eval.
    """

    if not (
        0.0 < min_angle_deg <= 180.0
    ):
        raise ValueError(
            "min_angle_deg must be in (0, 180]"
        )

    if angle_step_deg <= 0.0:
        raise ValueError(
            "angle_step_deg must be positive"
        )

    line_geometry = build_profile_line_geometry(
        sensor_poses=sensor_poses,
        frame=frame,
        x_values=x_values,
    )

    identity_rms = (
        relative_transform_best_plane_rms(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=np.eye(3),
            delta_t_mm=np.zeros(3),
        )
    )

    y180_axis = np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    )

    y180_R = rotation_matrix_from_axis_angle(
        axis=y180_axis,
        angle_deg=180.0,
    )

    (
        y180_t,
        y180_rms,
        y180_success,
    ) = optimize_translation_for_relative_rotation(
        sensor_poses=sensor_poses,
        line_geometry=line_geometry,
        delta_R=y180_R,
        translation_bound_mm=(
            translation_bound_mm
        ),
        maxiter=translation_maxiter,
    )

    y180_decomposition = (
        relative_transform_best_plane_decomposition(
            sensor_poses=sensor_poses,
            line_geometry=line_geometry,
            delta_R=y180_R,
            delta_t_mm=y180_t,
        )
    )

    axes = _fibonacci_axes(
        num_axes=num_fibonacci_axes
    )

    angles = np.arange(
        float(min_angle_deg),
        180.0 + 0.5 * float(angle_step_deg),
        float(angle_step_deg),
    )

    angles = angles[
        angles <= 180.0 + 1e-12
    ]

    if (
        len(angles) == 0
        or abs(
            float(angles[-1]) - 180.0
        ) > 1e-12
    ):
        angles = np.concatenate(
            [
                angles,
                np.array([180.0]),
            ]
        )

    sample_rows = []

    best_rms = np.inf
    best_angle = np.nan
    best_axis = None
    best_t = None
    best_decomposition = None

    evaluated = 0

    for axis_id, axis in enumerate(
        axes
    ):

        for angle_deg in angles:

            delta_R = rotation_matrix_from_axis_angle(
                axis=axis,
                angle_deg=float(angle_deg),
            )

            (
                best_delta_t,
                rms_mm,
                optimizer_success,
            ) = optimize_translation_for_relative_rotation(
                sensor_poses=sensor_poses,
                line_geometry=line_geometry,
                delta_R=delta_R,
                translation_bound_mm=(
                    translation_bound_mm
                ),
                maxiter=translation_maxiter,
            )

            decomposition = (
                relative_transform_best_plane_decomposition(
                    sensor_poses=sensor_poses,
                    line_geometry=line_geometry,
                    delta_R=delta_R,
                    delta_t_mm=best_delta_t,
                )
            )

            evaluated += 1

            sample_rows.append(
                {
                    "axis_id": int(axis_id),
                    "axis_x": float(axis[0]),
                    "axis_y": float(axis[1]),
                    "axis_z": float(axis[2]),
                    "angle_deg": float(angle_deg),

                    "best_translation_x_mm": float(
                        best_delta_t[0]
                    ),
                    "best_translation_y_mm": float(
                        best_delta_t[1]
                    ),
                    "best_translation_z_mm": float(
                        best_delta_t[2]
                    ),

                    "best_plane_rms_mm": float(
                        rms_mm
                    ),

                    "center_sse": float(
                        decomposition.center_sse
                    ),
                    "direction_sse": float(
                        decomposition.direction_sse
                    ),
                    "total_sse": float(
                        decomposition.total_sse
                    ),

                    "center_rms_contribution_mm": float(
                        decomposition.center_rms_contribution_mm
                    ),
                    "direction_rms_contribution_mm": float(
                        decomposition.direction_rms_contribution_mm
                    ),

                    "center_fraction": float(
                        decomposition.center_fraction
                    ),
                    "direction_fraction": float(
                        decomposition.direction_fraction
                    ),

                    "best_plane_normal_x": float(
                        decomposition.best_plane_normal[0]
                    ),
                    "best_plane_normal_y": float(
                        decomposition.best_plane_normal[1]
                    ),
                    "best_plane_normal_z": float(
                        decomposition.best_plane_normal[2]
                    ),

                    "lambda_min_center": float(
                        decomposition.lambda_min_center
                    ),
                    "lambda_min_direction": float(
                        decomposition.lambda_min_direction
                    ),

                    "translation_optimizer_success": int(
                        optimizer_success
                    ),
                }
            )

            if rms_mm < best_rms:

                best_rms = float(
                    rms_mm
                )

                best_angle = float(
                    angle_deg
                )

                best_axis = np.asarray(
                    axis,
                    dtype=float,
                ).copy()

                best_t = np.asarray(
                    best_delta_t,
                    dtype=float,
                ).copy()

                best_decomposition = (
                    decomposition
                )

    if (
        best_axis is None
        or best_t is None
        or best_decomposition is None
    ):
        raise RuntimeError(
            "global ambiguity search evaluated no valid rotations"
        )

    metrics = GlobalAmbiguityMetrics(
        identity_rms_mm=float(
            identity_rms
        ),

        y180_rms_mm=float(
            y180_rms
        ),
        y180_translation_mm=np.asarray(
            y180_t,
            dtype=float,
        ),
        y180_optimizer_success=bool(
            y180_success
        ),
        y180_decomposition=(
            y180_decomposition
        ),

        min_far_rms_mm=float(
            best_rms
        ),
        min_far_angle_deg=float(
            best_angle
        ),
        min_far_axis=np.asarray(
            best_axis,
            dtype=float,
        ),
        min_far_translation_mm=np.asarray(
            best_t,
            dtype=float,
        ),
        min_far_decomposition=(
            best_decomposition
        ),

        min_angle_deg=float(
            min_angle_deg
        ),
        angle_step_deg=float(
            angle_step_deg
        ),
        num_axes=int(
            len(axes)
        ),
        evaluated_rotation_count=int(
            evaluated
        ),
    )

    return (
        metrics,
        sample_rows,
    )



def print_ambiguity_decomposition(
    decomposition: AmbiguityDecompositionResult,
    prefix: str = "",
) -> None:

    p = str(prefix)

    print(
        f"{p}best plane normal             :",
        decomposition.best_plane_normal,
    )

    print(
        f"{p}total SSE                     :",
        decomposition.total_sse,
    )

    print(
        f"{p}center SSE                    :",
        decomposition.center_sse,
    )

    print(
        f"{p}direction SSE                 :",
        decomposition.direction_sse,
    )

    print(
        f"{p}total RMS [mm]                :",
        decomposition.total_rms_mm,
    )

    print(
        f"{p}center RMS contribution [mm]  :",
        decomposition.center_rms_contribution_mm,
    )

    print(
        f"{p}direction RMS contribution [mm]:",
        decomposition.direction_rms_contribution_mm,
    )

    print(
        f"{p}center fraction               :",
        decomposition.center_fraction,
    )

    print(
        f"{p}direction fraction            :",
        decomposition.direction_fraction,
    )

    print(
        f"{p}lambda_min(H_center)          :",
        decomposition.lambda_min_center,
    )

    print(
        f"{p}lambda_min(H_direction)       :",
        decomposition.lambda_min_direction,
    )

    print(
        f"{p}eig(H_center)                 :",
        decomposition.eigenvalues_center,
    )

    print(
        f"{p}eig(H_direction)              :",
        decomposition.eigenvalues_direction,
    )

    print(
        f"{p}eig(H_total)                  :",
        decomposition.eigenvalues_total,
    )



def print_global_ambiguity_metrics(
    metrics: GlobalAmbiguityMetrics,
    label: str = "",
) -> None:

    print("\n" + "=" * 72)
    print(
        f"Global ambiguity analysis: {label}"
    )
    print("=" * 72)

    print(
        "identity RMS [mm]         :",
        metrics.identity_rms_mm,
    )

    print("\nTargeted y-axis 180 deg branch")

    print(
        "y180 best RMS [mm]        :",
        metrics.y180_rms_mm,
    )

    print(
        "y180 best Delta t [mm]    :",
        metrics.y180_translation_mm,
    )

    print(
        "y180 optimizer success    :",
        metrics.y180_optimizer_success,
    )

    print("\ny180 center / direction decomposition")

    print_ambiguity_decomposition(
        metrics.y180_decomposition,
        prefix="  ",
    )

    print("\nSampled far-rotation search")

    print(
        "excluded angle [deg]      :",
        metrics.min_angle_deg,
    )

    print(
        "angle step [deg]          :",
        metrics.angle_step_deg,
    )

    print(
        "sampled axes              :",
        metrics.num_axes,
    )

    print(
        "evaluated rotations       :",
        metrics.evaluated_rotation_count,
    )

    print(
        "min far RMS [mm]          :",
        metrics.min_far_rms_mm,
    )

    print(
        "best false angle [deg]    :",
        metrics.min_far_angle_deg,
    )

    print(
        "best false axis           :",
        metrics.min_far_axis,
    )

    print(
        "best false Delta t [mm]   :",
        metrics.min_far_translation_mm,
    )

    print("\nbest far branch center / direction decomposition")

    print_ambiguity_decomposition(
        metrics.min_far_decomposition,
        prefix="  ",
    )
