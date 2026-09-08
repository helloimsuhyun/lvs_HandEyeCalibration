from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from .data import PlaneFrame
from .patterns import circular_lines
from .simulation import PoseGeometry, sensor_pose_from_target_line


PoseDesignMethod = Literal["random", "lhs", "circular"]
NormalizedDesignMethod = Literal["random", "lhs"]
PLANE_RELATIVE_POSE_CONVENTION = "plane_normal_azimuth_sensor_v2"


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be finite and non-zero")
    return vector / norm


@dataclass(frozen=True)
class PlaneRelativePoseBounds:
    target_u_mm: tuple[float, float] = (-40.0, 40.0)
    target_v_mm: tuple[float, float] = (-40.0, 40.0)
    distance_mm: tuple[float, float] = (60.0, 150.0)
    tilt_deg: tuple[float, float] = (10.0, 60.0)
    azimuth_deg: tuple[float, float] = (-180.0, 180.0)
    normal_azimuth_sensor_deg: tuple[float, float] = (-180.0, 180.0)

    def __post_init__(self) -> None:
        for name in (
            "target_u_mm",
            "target_v_mm",
            "distance_mm",
            "tilt_deg",
            "azimuth_deg",
            "normal_azimuth_sensor_deg",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.shape != (2,)
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
            ):
                raise ValueError(f"{name} must be an increasing finite pair")
            object.__setattr__(self, name, (float(values[0]), float(values[1])))
        if self.distance_mm[0] <= 0.0:
            raise ValueError("distance range must be positive")
        if self.tilt_deg[0] <= 0.0 or self.tilt_deg[1] >= 89.0:
            raise ValueError(
                "tilt range must lie within (0, 89) degrees because the "
                "projected plane-normal azimuth is undefined at zero tilt"
            )

    @property
    def ordered(self) -> tuple[tuple[float, float], ...]:
        return (
            self.target_u_mm,
            self.target_v_mm,
            self.distance_mm,
            self.tilt_deg,
            self.azimuth_deg,
            self.normal_azimuth_sensor_deg,
        )


@dataclass(frozen=True)
class PlaneRelativePose:
    sample_id: int
    target_u_mm: float
    target_v_mm: float
    distance_mm: float
    tilt_deg: float
    azimuth_deg: float
    normal_azimuth_sensor_deg: float

    @property
    def center_depth_mm(self) -> float:
        return self.distance_mm

    @property
    def view_tilt_deg(self) -> float:
        return self.tilt_deg

    @property
    def view_azimuth_deg(self) -> float:
        return self.azimuth_deg

    @property
    def plane_normal_azimuth_sensor_deg(self) -> float:
        """Azimuth of the plane normal projected into sensor XY."""
        return self.normal_azimuth_sensor_deg


@dataclass(frozen=True)
class DesignedPose:
    sample_id: int
    method: PoseDesignMethod
    T_base_s: np.ndarray
    parameters: dict[str, float | int | str]
    normalized_parameters: np.ndarray | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_base_s, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_base_s must be a finite 4x4 transform")
        object.__setattr__(self, "T_base_s", transform.copy())
        if self.normalized_parameters is not None:
            normalized = np.asarray(self.normalized_parameters, dtype=float).reshape(-1)
            if np.any(normalized < 0.0) or np.any(normalized >= 1.0):
                raise ValueError("normalized parameters must lie in [0, 1)")
            object.__setattr__(self, "normalized_parameters", normalized.copy())


def random_uniform(
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    """Draw independent uniform rows in ``[0, 1)^dimensions``."""
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    return rng.random((int(count), int(dimensions)))


def latin_hypercube(
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    """Draw a randomized Latin hypercube in ``[0, 1)^dimensions``."""
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    values = np.empty((int(count), int(dimensions)), dtype=float)
    for dimension in range(int(dimensions)):
        values[:, dimension] = (
            rng.permutation(int(count)) + rng.random(int(count))
        ) / float(count)
    return values


def latin_hypercube_strata(
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    """Return one permutation of integer LHS strata per dimension."""
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    strata = np.empty((int(count), int(dimensions)), dtype=np.int64)
    for dimension in range(int(dimensions)):
        strata[:, dimension] = rng.permutation(int(count))
    return strata


def audit_latin_hypercube(normalized: np.ndarray) -> None:
    """Raise when rows are not an exact Latin hypercube."""
    values = np.asarray(normalized, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("normalized design must be a non-empty matrix")
    if np.any(values < 0.0) or np.any(values >= 1.0):
        raise ValueError("normalized design escaped [0, 1)")
    count = len(values)
    strata = np.floor(values * count).astype(int)
    expected = np.arange(count, dtype=int)
    for dimension in range(values.shape[1]):
        if not np.array_equal(np.sort(strata[:, dimension]), expected):
            raise ValueError(
                f"dimension {dimension} is not an exact {count}-point LHS"
            )


def nested_sliced_latin_hypercube(
    rng: np.random.Generator,
    *,
    initial_count: int,
    total_count: int,
    dimensions: int,
) -> np.ndarray:
    """Return an ordered design whose prefix and full set are both LHS."""
    if initial_count <= 0 or total_count % initial_count != 0:
        raise ValueError("total_count must be a positive multiple of initial_count")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    refinement = total_count // initial_count
    values = np.empty((total_count, dimensions), dtype=float)
    for dimension in range(dimensions):
        coarse_order = rng.permutation(initial_count)
        sub_orders = np.vstack(
            [rng.permutation(refinement) for _ in range(initial_count)]
        )
        for row in range(initial_count):
            fine = coarse_order[row] * refinement + sub_orders[row, 0]
            values[row, dimension] = (fine + rng.random()) / total_count
        output_row = initial_count
        for layer in range(1, refinement):
            for source_row in rng.permutation(initial_count):
                fine = coarse_order[source_row] * refinement + sub_orders[source_row, layer]
                values[output_row, dimension] = (fine + rng.random()) / total_count
                output_row += 1
    audit_latin_hypercube(values)
    audit_latin_hypercube(values[:initial_count])
    return values


def normalized_design(
    method: NormalizedDesignMethod,
    *,
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    if method == "random":
        return random_uniform(rng, count, dimensions)
    if method == "lhs":
        return latin_hypercube(rng, count, dimensions)
    raise ValueError("normalized design method must be 'random' or 'lhs'")


def plane_relative_pose_from_row(
    row: np.ndarray,
    *,
    sample_id: int,
    bounds: PlaneRelativePoseBounds,
) -> PlaneRelativePose:
    normalized = np.asarray(row, dtype=float).reshape(6)
    if np.any(normalized < 0.0) or np.any(normalized >= 1.0):
        raise ValueError("normalized pose row must lie in [0, 1)")
    scaled = [
        lower + value * (upper - lower)
        for value, (lower, upper) in zip(normalized, bounds.ordered)
    ]
    return PlaneRelativePose(int(sample_id), *map(float, scaled))


def sensor_pose_from_plane_relative(
    frame: PlaneFrame,
    target_center_base_mm: np.ndarray,
    pose: PlaneRelativePose,
) -> np.ndarray:
    """Construct ``T_base_s`` using the canonical plane-relative convention.

    ``normal_azimuth_sensor_deg`` is defined directly from
    ``b = R_base_sensor.T @ frame.n`` as ``atan2(b_y, b_x)``.  Consequently
    ``b = [sin(tilt) cos(alpha), sin(tilt) sin(alpha), -cos(tilt)]`` and view
    azimuth is a remaining orientation degree of freedom that does not change
    ``b``.
    """
    tilt = np.deg2rad(float(pose.tilt_deg))
    azimuth = np.deg2rad(float(pose.azimuth_deg))
    normal_azimuth = np.deg2rad(float(pose.normal_azimuth_sensor_deg))
    z_axis = _normalize(
        np.sin(tilt) * np.cos(azimuth) * frame.u
        + np.sin(tilt) * np.sin(azimuth) * frame.v
        - np.cos(tilt) * frame.n,
        "sensor +Z view direction",
    )
    normal_projection = frame.n - float(frame.n @ z_axis) * z_axis
    normal_projection_norm = float(np.linalg.norm(normal_projection))
    if normal_projection_norm <= 1e-10:
        raise ValueError(
            "plane-normal azimuth in sensor XY is undefined at zero tilt"
        )
    normal_axis = normal_projection / normal_projection_norm
    tangent_axis = _normalize(
        np.cross(z_axis, normal_axis), "projected-normal tangent axis"
    )
    x_axis = _normalize(
        np.cos(normal_azimuth) * normal_axis
        - np.sin(normal_azimuth) * tangent_axis,
        "sensor X from projected-normal azimuth",
    )
    y_axis = _normalize(
        np.sin(normal_azimuth) * normal_axis
        + np.cos(normal_azimuth) * tangent_axis,
        "sensor Y from projected-normal azimuth",
    )
    target = (
        np.asarray(target_center_base_mm, dtype=float).reshape(3)
        + pose.target_u_mm * frame.u
        + pose.target_v_mm * frame.v
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = target - pose.distance_mm * z_axis
    return transform


def generate_plane_relative_design(
    method: NormalizedDesignMethod,
    *,
    rng: np.random.Generator,
    count: int,
    frame: PlaneFrame,
    target_center_base_mm: np.ndarray,
    bounds: PlaneRelativePoseBounds,
) -> list[DesignedPose]:
    rows = normalized_design(method, rng=rng, count=count, dimensions=6)
    output: list[DesignedPose] = []
    for sample_id, row in enumerate(rows):
        pose = plane_relative_pose_from_row(row, sample_id=sample_id, bounds=bounds)
        output.append(
            DesignedPose(
                sample_id=sample_id,
                method=method,
                T_base_s=sensor_pose_from_plane_relative(
                    frame, target_center_base_mm, pose
                ),
                normalized_parameters=row,
                parameters={
                    "target_u_mm": pose.target_u_mm,
                    "target_v_mm": pose.target_v_mm,
                    "distance_mm": pose.distance_mm,
                    "tilt_deg": pose.tilt_deg,
                    "azimuth_deg": pose.azimuth_deg,
                    "normal_azimuth_sensor_deg": pose.normal_azimuth_sensor_deg,
                },
            )
        )
    return output


def generate_circular_design(
    *,
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    radius_mm: float,
    scan_parameters: Sequence[Mapping[str, float]],
    line_count: int = 9,
    branch_mode: Literal["alternating", "positive"] = "alternating",
    pose_geometry: PoseGeometry = "observable_dihedral",
) -> list[DesignedPose]:
    """Generate the existing target-line circular pose family as transforms."""
    if radius_mm <= 0.0 or line_count <= 0:
        raise ValueError("radius and line_count must be positive")
    if branch_mode not in ("alternating", "positive"):
        raise ValueError("branch_mode must be 'alternating' or 'positive'")
    output: list[DesignedPose] = []
    for line_id, (line_p0, line_p1) in enumerate(
        circular_lines(float(radius_mm), n_lines=int(line_count))
    ):
        branch_sign = 1.0 if branch_mode == "positive" or line_id % 2 == 0 else -1.0
        for parameter_id, parameters in enumerate(scan_parameters):
            sample_id = len(output)
            d_mm = float(parameters["d_mm"])
            theta_deg = float(parameters["theta_deg"])
            beta_deg = float(parameters["beta_deg"])
            transform = sensor_pose_from_target_line(
                plane_R=plane_R,
                plane_t=plane_t,
                line_p0=line_p0,
                line_p1=line_p1,
                d_mm=d_mm,
                theta_deg=theta_deg,
                beta_deg=beta_deg,
                branch_sign=branch_sign,
                pose_geometry=pose_geometry,
            )
            output.append(
                DesignedPose(
                    sample_id=sample_id,
                    method="circular",
                    T_base_s=transform,
                    parameters={
                        "line_id": line_id,
                        "parameter_id": parameter_id,
                        "d_mm": d_mm,
                        "theta_deg": theta_deg,
                        "beta_deg": beta_deg,
                        "branch_sign": branch_sign,
                        "pose_geometry": pose_geometry,
                    },
                )
            )
    return output


def generate_pose_design(
    method: PoseDesignMethod,
    **kwargs,
) -> list[DesignedPose]:
    """Small public dispatcher for random, LHS, and circular pose designs."""
    if method in ("random", "lhs"):
        return generate_plane_relative_design(method, **kwargs)
    if method == "circular":
        return generate_circular_design(**kwargs)
    raise ValueError("pose design method must be random, lhs, or circular")


# --------------------------------------------------------------------------------- pose design 
# -- by suhyun

# -------------------------------------------------------------------
# 원주위를 N개로 분할하여, 동일한 각도 간격으로 평면위의 (u,v) 점을 생성

def circular_uv_points(
    radius_mm: float,
    N: int,
    start_angle_deg: float = 0.0,
) -> np.ndarray:

    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be positive")
    if N <= 0:
        raise ValueError("count must be positive")

    start = np.deg2rad(float(start_angle_deg))

    angles = start + 2.0 * np.pi * np.arange(N) / N

    u = radius_mm * np.cos(angles)
    v = radius_mm * np.sin(angles)

    return np.column_stack([u, v]) # (N,2) ndarray local board coordinate [u_i,v_i]

# plane local (u,v) 좌표를 월드상 평면의 point vector로 변환
def plane_uv_to_base_points(
    uv: np.ndarray,
    target_center_base_mm: np.ndarray,
    frame: PlaneFrame,
) -> np.ndarray:
    """
    uv : 보드 local plane 좌표 ndarray (N,2)
    target_center_base_mm : base에서 보드 중심에 대한 vector (x,y,z)
    frame : world에서 표현한 plane normal / offset / u,v base vectors
    """

    uv = np.asarray(uv, dtype=float)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")

    center = np.asarray(
        target_center_base_mm,
        dtype=float,
    ).reshape(3)

    points_base = (
        center[None, :]
        + uv[:, 0, None] * frame.u[None, :]
        + uv[:, 1, None] * frame.v[None, :]
    )

    return points_base

# 평면 위 target base 좌표 > 센서 4x4 transform
def sensor_pose_from_target_point(
    target_point_base_mm: np.ndarray,
    frame: PlaneFrame,
    distance_mm: float,
    tilt_deg: float,
    azimuth_deg: float,
    normal_azimuth_sensor_deg: float,
) -> np.ndarray:
    """
    target_point_base_mm : target의 점 - 센서 +z축이 평면과 만나는 점
    frame : frame의 world normal, offset, u,v base vectors

    distance_mm : target_point_base_mm와 센서 원점과의 거리
    tilt_deg: plane normal과 센서 -z축 사잇각
    azimuth_deg: 센서 +z를 plane에 projection 했을때의 방위각
    normal_azimuth_sensor_deg: : plane normal을 센서 xy plane에 투영한 방위각

    # tilt와 normal_azimuth_sensor_deg는 센서에서 본 법선의 자유도를 완전히 결정
    # tilt가 0이면, normal_azimuth_sensor_deg가 정의되지 않기에 범위는 90 > tilt > 0 

    Returns
    -------
    T_base_s : (4, 4) ndarray
        T_base_s =
            [ R_base_s   t_base_s ]
            [    0          1     ]
    """

    # ---------------------------------------------------------------
    # Input validation

    target = np.asarray(
        target_point_base_mm,
        dtype=float,
    ).reshape(3)

    if not np.all(np.isfinite(target)):
        raise ValueError("target_point_base_mm must be finite")

    distance_mm = float(distance_mm)

    if not np.isfinite(distance_mm) or distance_mm <= 0.0:
        raise ValueError("distance_mm must be positive and finite")

    if not (0.0 < tilt_deg < 90.0):
        raise ValueError("tilt_deg must lie in (0, 90) degrees")

    tilt = np.deg2rad(float(tilt_deg))
    azimuth = np.deg2rad(float(azimuth_deg))
    normal_azimuth = np.deg2rad(
        float(normal_azimuth_sensor_deg)
    )
    

    # ---------------------------------------------------------------
    # 2. Construct sensor +Z direction in base coordinates

    z_axis = _normalize(
        np.sin(tilt) * np.cos(azimuth) * frame.u
        + np.sin(tilt) * np.sin(azimuth) * frame.v
        - np.cos(tilt) * frame.n,
        "sensor +Z view direction",
    )

    # ---------------------------------------------------------------
    # 3. Project plane normal onto sensor XY plane

    normal_projection = (
        frame.n
        - float(frame.n @ z_axis) * z_axis
    )

    normal_axis = _normalize(
        normal_projection,
        "projected plane normal",
    )

    # ---------------------------------------------------------------
    # 4. Construct second reference axis in sensor XY plane
    tangent_axis = _normalize(
        np.cross(z_axis, normal_axis),
        "projected-normal tangent axis",
    )

    # ---------------------------------------------------------------
    # 5. Determine sensor X/Y orientation around sensor +Z
  
    x_axis = _normalize(
        np.cos(normal_azimuth) * normal_axis
        - np.sin(normal_azimuth) * tangent_axis,
        "sensor X axis",
    )

    y_axis = _normalize(
        np.sin(normal_azimuth) * normal_axis
        + np.cos(normal_azimuth) * tangent_axis,
        "sensor Y axis",
    )

    sensor_origin = (
        target
        - distance_mm * z_axis
    )

    T_base_s = np.eye(4, dtype=float)

    T_base_s[:3, :3] = np.column_stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ]
    )

    T_base_s[:3, 3] = sensor_origin

    return T_base_s



__all__ = [
    "DesignedPose",
    "NormalizedDesignMethod",
    "PLANE_RELATIVE_POSE_CONVENTION",
    "PlaneFrame",
    "PlaneRelativePose",
    "PlaneRelativePoseBounds",
    "PoseDesignMethod",
    "audit_latin_hypercube",
    "generate_circular_design",
    "generate_plane_relative_design",
    "generate_pose_design",
    "latin_hypercube",
    "latin_hypercube_strata",
    "nested_sliced_latin_hypercube",
    "normalized_design",
    "plane_relative_pose_from_row",
    "random_uniform",
    "sensor_pose_from_plane_relative",
    "circular_uv_points",
    "plane_uv_to_base_points",
    "sensor_pose_from_target_point",
]
