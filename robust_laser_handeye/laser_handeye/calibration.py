from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Literal
import numpy as np

from .data import LaserScan
from .geometry import fit_plane_pca, scaled_normal_from_plane
from .se3 import make_T, project_to_so3, transform_points


PlaneOffsetMode = Literal["fitted", "joint"]


@dataclass
class CalibrationResult:
    T_ef_s: np.ndarray
    iterations: int
    converged: bool
    plane_rms_history: list[float]
    delta_history: list[float]
    rank_history: list[int]
    cond_history: list[float]
    T_history: list[np.ndarray] = field(default_factory=list)
    plane_offsets: list[float] = field(default_factory=list)
    plane_offset_mode: PlaneOffsetMode = "fitted"


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------

# 현재 handeye 추정값을 사용해 point를 base coordinate로 표현
def reconstruct_points_base(scans: list[LaserScan], T_ef_s: np.ndarray) -> np.ndarray:
    all_pts = []
    for scan in scans:
        points_s = scan.valid_points_s
        if len(points_s) == 0:
            continue
        pts_ef = transform_points(T_ef_s, points_s)
        pts_b = transform_points(scan.T_base_ef, pts_ef)
        all_pts.append(pts_b)
    if not all_pts:
        raise ValueError("no finite laser points are available")
    return np.vstack(all_pts)


# -----------------------------------------------------------------------------
# Linear system: Aw = Y
# scan data ps와 scaled 된 n을 사용해 Aw = Y 선형식 build
def build_linear_system(scans: list[LaserScan], n_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the linear system Aw = Y

    Sensor profile points are assumed to lie on the sensor y=0 plane:

        p_s = [x, 0, z]^T

    Therefore the second column of R_ef_s is not solved directly. The unknown
    vector is

        w = [R1, R3, t]^T

    i.e.

        w = [r11, r21, r31, r13, r23, r33, tx, ty, tz]^T.

    Plane convention:

        n_scaled^T p_b = ||n_scaled||^2
    """
    n = np.asarray(n_scaled, dtype=float).reshape(3)
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    nn = float(n @ n)

    for scan in scans:
        Rbe = scan.T_base_ef[:3, :3]
        tbe = scan.T_base_ef[:3, 3]

        # a = n^T R_base_ef
        a = n @ Rbe
        base_rhs = nn - float(n @ tbe)

        points_s = scan.valid_points_s
        if len(points_s) == 0:
            continue

        # n^T Rbe (x R1 + z R3 + t) = ||n||^2 - n^T tbe
        rows.append(
            np.column_stack(
                [
                    points_s[:, 0, None] * a[None, :],
                    points_s[:, 2, None] * a[None, :],
                    np.broadcast_to(a, (len(points_s), 3)),
                ]
            )
        )
        rhs.append(np.full(len(points_s), base_rhs, dtype=float))

    if not rows:
        raise ValueError("no finite laser points are available")
    return np.vstack(rows), np.concatenate(rhs)


# R1, R3를 사용해 R2를 복원 > 이후 회전행렬을 SO(3) 만족하도록 변형
def solve_rotation_translation_linear(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(A, y, rcond=None)

    r1 = w[0:3]
    r3 = w[3:6]

    # R = [R1 R2 R3], so the right-handed second axis is R2 = R3 x R1.
    r2 = np.cross(r3, r1)
    R_raw = np.column_stack([r1, r2, r3])
    R = project_to_so3(R_raw)

    t = w[6:9]
    return make_T(R, t)


# 각 평면별 스캔 데이터 형식을 정리
# [(plane_id, scans), ...]."""
def _normalize_scan_groups(
    scans_by_plane: Mapping[int, list[LaserScan]] | Sequence[list[LaserScan]],
) -> list[tuple[int, list[LaserScan]]]:
    """Normalize dict/list grouped scan data into [(plane_id, scans), ...]."""
    if isinstance(scans_by_plane, Mapping):
        items = [(int(k), list(v)) for k, v in scans_by_plane.items()]
    else:
        items = [(i, list(v)) for i, v in enumerate(scans_by_plane)]

    items = [(pid, scans) for pid, scans in items if len(scans) > 0]
    if not items:
        raise ValueError("at least one non-empty plane scan group is required")
    return items

# 각 plane의 결과들을 합쳐 Aw = Y 만들기
def _build_grouped_system_from_current_T(
    scans_by_plane: Mapping[int, list[LaserScan]] | Sequence[list[LaserScan]],
    T_ef_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], list[np.ndarray], list[tuple[int, list[LaserScan]]]]:
    groups = _normalize_scan_groups(scans_by_plane)

    A_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    rms_all: list[float] = []
    normals_scaled: list[np.ndarray] = []

    for _plane_id, scans in groups:
        pts_b = reconstruct_points_base(scans, T_ef_s) # 현재  handeye 추정값을 이용해 point들을 base coordinate로 변환
        n_scaled, plane_rms = scaled_normal_from_plane(pts_b) # 평면 추정
        A_j, y_j = build_linear_system(scans, n_scaled) # Aw = Y

        A_all.append(A_j)
        y_all.append(y_j)
        rms_all.append(float(plane_rms))
        normals_scaled.append(n_scaled) # 한개의 행렬로 append

    return np.vstack(A_all), np.concatenate(y_all), rms_all, normals_scaled, groups


def _build_grouped_joint_offset_system_from_current_T(
    scans_by_plane: Mapping[int, list[LaserScan]] | Sequence[list[LaserScan]],
    T_ef_s: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[float],
    list[np.ndarray],
    list[tuple[int, list[LaserScan]]],
]:
    """Build a linear update that estimates each plane offset with hand-eye.

    Holding the currently fitted unit normal ``n_j`` fixed, every profile point
    supplies

        [x a, z a, a, -e_j] [R1, R3, t, l_1, ..., l_m] = -n_j^T t_be,

    where ``a = n_j^T R_be``.  Estimating ``l_j`` in the same least-squares
    problem removes the very slow (and, for one-sided poses, exactly neutral)
    feedback path created by encoding the offset fitted from the previous
    hand-eye iterate into a scaled normal.

    This remains a linear alternating method; it is not the optional six-DOF
    nonlinear refinement used by the example programs.
    """
    groups = _normalize_scan_groups(scans_by_plane)
    n_planes = len(groups)

    blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    rms_all: list[float] = []
    normals_unit: list[np.ndarray] = []

    for plane_index, (_plane_id, scans) in enumerate(groups):
        points_base = reconstruct_points_base(scans, T_ef_s)
        normal, _offset, _centroid, plane_rms = fit_plane_pca(points_base)
        normals_unit.append(normal)
        rms_all.append(float(plane_rms))

        for scan in scans:
            points_s = scan.valid_points_s
            if len(points_s) == 0:
                continue

            R_be = scan.T_base_ef[:3, :3]
            t_be = scan.T_base_ef[:3, 3]
            a = normal @ R_be
            count = len(points_s)
            offset_columns = np.zeros((count, n_planes), dtype=float)
            offset_columns[:, plane_index] = -1.0

            blocks.append(
                np.column_stack(
                    [
                        points_s[:, 0, None] * a[None, :],
                        points_s[:, 2, None] * a[None, :],
                        np.broadcast_to(a, (count, 3)),
                        offset_columns,
                    ]
                )
            )
            rhs_blocks.append(
                np.full(count, -float(normal @ t_be), dtype=float)
            )

    if not blocks:
        raise ValueError("no finite laser points are available")

    return (
        np.vstack(blocks),
        np.concatenate(rhs_blocks),
        rms_all,
        normals_unit,
        groups,
    )


def _solve_column_equilibrated(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares after deterministic column-norm equilibration."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    column_scale = np.linalg.norm(A, axis=0)
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise np.linalg.LinAlgError("linear system contains an empty column")
    solution_scaled, *_ = np.linalg.lstsq(A / column_scale, y, rcond=None)
    return solution_scaled / column_scale


def translation_plane_offset_observability_matrix(
    groups: list[tuple[int, list[LaserScan]]],
    normals_unit: list[np.ndarray],
) -> np.ndarray:
    """Return the scan-level matrix for jointly identifying ``t`` and offsets.

    Point count, profile noise, and profile coordinates cannot repair a missing
    translation direction.  The relevant structural test is therefore built
    once per robot pose from rows ``[n_j^T R_be, -e_j]``.
    """
    if len(groups) != len(normals_unit):
        raise ValueError("one fitted normal is required per plane group")

    n_planes = len(groups)
    rows: list[np.ndarray] = []
    for plane_index, ((_plane_id, scans), normal) in enumerate(
        zip(groups, normals_unit)
    ):
        normal = np.asarray(normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal)
        for scan in scans:
            offset_columns = np.zeros(n_planes, dtype=float)
            offset_columns[plane_index] = -1.0
            rows.append(
                np.concatenate(
                    [normal @ scan.T_base_ef[:3, :3], offset_columns]
                )
            )

    if not rows:
        raise ValueError("at least one scan pose is required")
    return np.vstack(rows)


def _relative_rank(A: np.ndarray, relative_tolerance: float = 1e-10) -> int:
    """Rank after column equilibration, using an explicit relative threshold."""
    A = np.asarray(A, dtype=float)
    scale = np.linalg.norm(A, axis=0)
    if np.any(scale <= 0.0):
        return 0
    singular_values = np.linalg.svd(A / scale, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        return 0
    return int(
        np.sum(singular_values > relative_tolerance * singular_values[0])
    )


def _column_normalized_condition(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    scale = np.linalg.norm(A, axis=0)
    if np.any(scale <= 0.0):
        return float("inf")
    singular_values = np.linalg.svd(A / scale, compute_uv=False)
    if len(singular_values) == 0 or singular_values[-1] <= 0.0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def solve_rotation_translation_joint_offsets_linear(
    A: np.ndarray,
    y: np.ndarray,
    n_planes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``[R1, R3, t, plane offsets]`` and project rotation to SO(3)."""
    if n_planes <= 0:
        raise ValueError("n_planes must be positive")
    if A.shape[1] != 9 + int(n_planes):
        raise ValueError("joint-offset system has an unexpected column count")

    solution = _solve_column_equilibrated(A, y)
    r1 = solution[0:3]
    r3 = solution[3:6]
    r2 = np.cross(r3, r1)
    rotation = project_to_so3(np.column_stack([r1, r2, r3]))
    return make_T(rotation, solution[6:9]), solution[9:].copy()


def refine_translation_and_plane_offsets_with_fixed_rotation(
    groups: list[tuple[int, list[LaserScan]]],
    normals_unit: list[np.ndarray],
    R_ef_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly re-estimate translation and offsets after SO(3) projection."""
    if len(groups) != len(normals_unit):
        raise ValueError("one fitted normal is required per plane group")

    n_planes = len(groups)
    r1 = np.asarray(R_ef_s, dtype=float).reshape(3, 3)[:, 0]
    r3 = np.asarray(R_ef_s, dtype=float).reshape(3, 3)[:, 2]
    blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []

    for plane_index, ((_plane_id, scans), normal) in enumerate(
        zip(groups, normals_unit)
    ):
        normal = np.asarray(normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal)

        for scan in scans:
            points_s = scan.valid_points_s
            if len(points_s) == 0:
                continue

            R_be = scan.T_base_ef[:3, :3]
            t_be = scan.T_base_ef[:3, 3]
            a = normal @ R_be
            count = len(points_s)
            offset_columns = np.zeros((count, n_planes), dtype=float)
            offset_columns[:, plane_index] = -1.0
            blocks.append(
                np.column_stack(
                    [
                        np.broadcast_to(a, (count, 3)),
                        offset_columns,
                    ]
                )
            )
            rhs_blocks.append(
                -float(normal @ t_be)
                - points_s[:, 0] * float(a @ r1)
                - points_s[:, 2] * float(a @ r3)
            )

    if not blocks:
        raise ValueError("no finite laser points are available")

    solution = _solve_column_equilibrated(
        np.vstack(blocks),
        np.concatenate(rhs_blocks),
    )
    return solution[:3].copy(), solution[3:].copy()

# SO(3)만족하도록 고정한 R은 고정하고, translation만 최적화
def refine_translation_with_fixed_rotation_grouped(
    groups: list[tuple[int, list[LaserScan]]],
    normals_scaled: list[np.ndarray],
    R_ef_s: np.ndarray,
) -> np.ndarray:
    """Re-estimate one common translation using all fitted plane equations."""
    rows = []
    rhs = []

    r1 = R_ef_s[:, 0]
    r3 = R_ef_s[:, 2]

    for (_plane_id, scans), n_scaled in zip(groups, normals_scaled):
        n = np.asarray(n_scaled, dtype=float).reshape(3)
        nn = float(n @ n)

        for scan in scans:
            Rbe = scan.T_base_ef[:3, :3]
            tbe = scan.T_base_ef[:3, 3]
            a = n @ Rbe

            for p in scan.valid_points_s:
                x, _, z = p
                y_i = nn - float(n @ tbe) - float(x * (a @ r1)) - float(z * (a @ r3))
                rows.append(a)
                rhs.append(y_i)

    t, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return t

# SO(3)로 fixed 된 rotation로 만든 T를 사용해 R은 고정한 상태로 다시 t를 least-square 최적화
def refine_translation_with_fixed_rotation(
    scans: list[LaserScan],
    n_scaled: np.ndarray,
    R_ef_s: np.ndarray,
) -> np.ndarray:
    return refine_translation_with_fixed_rotation_grouped(
        groups=[(0, scans)],
        normals_scaled=[n_scaled],
        R_ef_s=R_ef_s,
    )


# -----------------------------------------------------------------------------
# Unknown-plane calibration
# -----------------------------------------------------------------------------

def calibrate_planes(
    scans_by_plane: Mapping[int, list[LaserScan]] | Sequence[list[LaserScan]],
    T_init: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    min_rank: int = 9,
    plane_offset_mode: PlaneOffsetMode = "fitted",
    max_translation_offset_condition: float = 1e6,
) -> CalibrationResult:
    """
    Generic unknown-plane calibration for one or more plane groups.
        Step A: reconstruct each plane group with current T and fit a plane.
        Step B: stack all plane constraints and solve the linear LS system.

    ``plane_offset_mode='fitted'`` reproduces the paper's scaled-normal
    iteration: each fitted offset is held fixed during the next hand-eye
    update. ``'joint'`` keeps the fitted normal but estimates the plane offset
    in the same linear system as hand-eye. The latter requires one additional
    independent equation per physical plane and converges much faster for the
    single-plane problem.

    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if plane_offset_mode not in ("fitted", "joint"):
        raise ValueError("plane_offset_mode must be 'fitted' or 'joint'")
    if max_translation_offset_condition <= 1.0:
        raise ValueError("max_translation_offset_condition must be > 1")

    T = np.asarray(T_init, dtype=float).reshape(4, 4).copy()

    plane_rms_history: list[float] = []
    delta_history: list[float] = []
    rank_history: list[int] = []
    cond_history: list[float] = []
    T_history: list[np.ndarray] = []
    plane_offsets: list[float] = []

    converged = False
    for k in range(max_iter):
        if plane_offset_mode == "joint":
            (
                A,
                y,
                rms_all,
                normals_unit,
                groups,
            ) = _build_grouped_joint_offset_system_from_current_T(
                scans_by_plane,
                T,
            )
            required_rank = max(int(min_rank), 9 + len(groups))
            translation_offset_matrix = (
                translation_plane_offset_observability_matrix(
                    groups,
                    normals_unit,
                )
            )
            translation_offset_rank = _relative_rank(
                translation_offset_matrix
            )
            translation_offset_condition = _column_normalized_condition(
                translation_offset_matrix
            )
            required_translation_offset_rank = 3 + len(groups)
            if (
                translation_offset_rank < required_translation_offset_rank
                or translation_offset_condition
                > float(max_translation_offset_condition)
            ):
                failed_checks: list[str] = []
                if translation_offset_rank < required_translation_offset_rank:
                    failed_checks.append(
                        f"rank {translation_offset_rank} < "
                        f"{required_translation_offset_rank}"
                    )
                if (
                    translation_offset_condition
                    > float(max_translation_offset_condition)
                ):
                    failed_checks.append(
                        "normalized condition "
                        f"{translation_offset_condition:.6g} > "
                        f"{max_translation_offset_condition:.6g}"
                    )
                raise np.linalg.LinAlgError(
                    "translation/plane-offset observability check failed ("
                    + "; ".join(failed_checks)
                    + "); vary the theta/"
                    "incidence magnitude or add an off-ring reference pose"
                )
        else:
            (
                A,
                y,
                rms_all,
                normals_scaled,
                groups,
            ) = _build_grouped_system_from_current_T(scans_by_plane, T)
            required_rank = int(min_rank)

        rank = int(np.linalg.matrix_rank(A))
        s = np.linalg.svd(A, compute_uv=False)
        cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        if rank < required_rank:
            raise np.linalg.LinAlgError(
                f"coefficient matrix rank {rank} < {required_rank}; "
                "calibration data lacks hand-eye/plane-offset variation"
            )

        if plane_offset_mode == "joint":
            T_new, plane_offsets_array = (
                solve_rotation_translation_joint_offsets_linear(
                    A,
                    y,
                    n_planes=len(groups),
                )
            )
            translation, plane_offsets_array = (
                refine_translation_and_plane_offsets_with_fixed_rotation(
                    groups=groups,
                    normals_unit=normals_unit,
                    R_ef_s=T_new[:3, :3],
                )
            )
            T_new[:3, 3] = translation
            plane_offsets = [float(value) for value in plane_offsets_array]
        else:
            T_new = solve_rotation_translation_linear(A, y)
            T_new[:3, 3] = refine_translation_with_fixed_rotation_grouped(
                groups=groups,
                normals_scaled=normals_scaled,
                R_ef_s=T_new[:3, :3],
            )

        delta = float(np.linalg.norm(T_new - T))
        plane_rms_history.append(float(np.mean(rms_all)))
        delta_history.append(delta)
        rank_history.append(rank)
        cond_history.append(cond)

        T = T_new
        T_history.append(T.copy())
        if delta < tol:
            converged = True
            break

    # A negative tolerance is the documented fixed-iteration mode. Completing
    # the requested count is a successful solver termination even though no
    # early-stop threshold was evaluated.
    if tol < 0.0:
        converged = True

    return CalibrationResult(
        T_ef_s=T,
        iterations=k + 1,
        converged=converged,
        plane_rms_history=plane_rms_history,
        delta_history=delta_history,
        rank_history=rank_history,
        cond_history=cond_history,
        T_history=T_history,
        plane_offsets=plane_offsets,
        plane_offset_mode=plane_offset_mode,
    )


# step B: single plane wrapper
def calibrate_single_plane(
    scans: list[LaserScan],
    T_init: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    min_rank: int = 9,
    plane_offset_mode: PlaneOffsetMode = "joint",
    max_translation_offset_condition: float = 1e6,
) -> CalibrationResult:
    """
    Unknown single-plane calibration wrapper.

    input:
        scans  - scan profile data from one physical plane
        T_init - initial hand-eye transform
    """
    if len(scans) == 0:
        raise ValueError("at least one scan is required")
    return calibrate_planes(
        {0: scans},
        T_init=T_init,
        max_iter=max_iter,
        tol=tol,
        min_rank=min_rank,
        plane_offset_mode=plane_offset_mode,
        max_translation_offset_condition=max_translation_offset_condition,
    )



























# -----------------------------------------------------------------------------
# Known-plane sanity checks
# -----------------------------------------------------------------------------

def calibrate_with_known_plane(scans: list[LaserScan], plane_n_unit: np.ndarray, plane_l: float) -> np.ndarray:
    """One-shot LS calibration when the calibration plane in robot base is known.

    This is useful for synthetic validation or a fixture whose plane pose was
    measured independently. The plane equation is n_unit^T p = plane_l.
    """
    n_unit = np.asarray(plane_n_unit, dtype=float).reshape(3)
    n_unit = n_unit / np.linalg.norm(n_unit)
    n_scaled = float(plane_l) * n_unit
    A, y = build_linear_system(scans, n_scaled)
    if np.linalg.matrix_rank(A) < 9:
        raise np.linalg.LinAlgError("coefficient matrix is rank deficient")
    T = solve_rotation_translation_linear(A, y)
    T[:3, 3] = refine_translation_with_fixed_rotation(scans, n_scaled, T[:3, :3])
    return T


def calibrate_with_known_planes(scans_by_plane: Mapping[int, tuple[list[LaserScan], np.ndarray, float]]) -> np.ndarray:
    """
    One-shot known-plane LS calibration for multiple plane groups.

    scans_by_plane maps plane_id -> (scans, plane_n_unit, plane_l).
    This is only for synthetic/debug validation.
    """
    A_all = []
    y_all = []
    grouped = []
    normals_scaled = []

    for plane_id, (scans, plane_n_unit, plane_l) in scans_by_plane.items():
        n_unit = np.asarray(plane_n_unit, dtype=float).reshape(3)
        n_unit = n_unit / np.linalg.norm(n_unit)
        n_scaled = float(plane_l) * n_unit
        A_j, y_j = build_linear_system(scans, n_scaled)
        A_all.append(A_j)
        y_all.append(y_j)
        grouped.append((int(plane_id), scans))
        normals_scaled.append(n_scaled)

    A = np.vstack(A_all)
    y = np.concatenate(y_all)
    if np.linalg.matrix_rank(A) < 9:
        raise np.linalg.LinAlgError("coefficient matrix is rank deficient")

    T = solve_rotation_translation_linear(A, y)
    T[:3, 3] = refine_translation_with_fixed_rotation_grouped(grouped, normals_scaled, T[:3, :3])
    return T
