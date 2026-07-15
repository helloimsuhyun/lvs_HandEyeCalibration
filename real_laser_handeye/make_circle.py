# initial point 를 바탕으로 원형 패턴을 생성
# 원점은 추정 중심점

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("zero-length vector")
    return v / norm


def load_plane_boundary(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "plane_frame" not in data:
        raise KeyError("JSON does not contain 'plane_frame'")

    if "T_base_plane" not in data["plane_frame"]:
        raise KeyError("JSON does not contain 'plane_frame.T_base_plane'")

    if (
        "safe_bounds_uv_mm" not in data
        and "observed_bounds_uv_mm" not in data
    ):
        raise KeyError(
            "JSON must contain 'safe_bounds_uv_mm' "
            "or 'observed_bounds_uv_mm'"
        )

    return data


def read_bounds(data: dict) -> dict[str, float]:
    # margin=0인 기존 코드라면 safe와 observed가 동일하다.
    if "safe_bounds_uv_mm" in data:
        raw = data["safe_bounds_uv_mm"]
    else:
        raw = data["observed_bounds_uv_mm"]

    bounds = {
        "u_min": float(raw["u_min"]),
        "u_max": float(raw["u_max"]),
        "v_min": float(raw["v_min"]),
        "v_max": float(raw["v_max"]),
    }

    if bounds["u_min"] >= bounds["u_max"]:
        raise ValueError("invalid u bounds")

    if bounds["v_min"] >= bounds["v_max"]:
        raise ValueError("invalid v bounds")

    return bounds


def point_inside_bounds(
    point_uv: np.ndarray,
    bounds: dict[str, float],
    tolerance_mm: float = 1e-9,
) -> bool:
    u, v = np.asarray(point_uv, dtype=float).reshape(2)

    return bool(
        bounds["u_min"] - tolerance_mm <= u <= bounds["u_max"] + tolerance_mm
        and bounds["v_min"] - tolerance_mm <= v <= bounds["v_max"] + tolerance_mm
    )


def uv_to_base(
    uv: np.ndarray,
    T_base_plane: np.ndarray,
) -> np.ndarray:
    """
    평면 좌표 [u, v, 0]을 base 좌표로 변환한다.

    T_base_plane:
        plane frame -> base frame
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    T_base_plane = np.asarray(T_base_plane, dtype=float).reshape(4, 4)

    points_plane = np.column_stack(
        [
            uv[:, 0],
            uv[:, 1],
            np.zeros(len(uv)),
        ]
    )

    return (
        points_plane @ T_base_plane[:3, :3].T
        + T_base_plane[:3, 3]
    )


def directional_distance_to_boundary(
    center_uv: np.ndarray,
    direction_uv: np.ndarray,
    bounds: dict[str, float],
) -> float:
    """
    center에서 +direction 방향으로 이동할 때 rectangle boundary까지의 거리.

        p(s) = center + s * direction,  s >= 0

    direction은 단위 벡터라고 가정한다.
    """
    center_uv = np.asarray(center_uv, dtype=float).reshape(2)
    direction_uv = normalize(direction_uv)

    candidates: list[float] = []

    for coordinate, direction, minimum, maximum in (
        (
            center_uv[0],
            direction_uv[0],
            bounds["u_min"],
            bounds["u_max"],
        ),
        (
            center_uv[1],
            direction_uv[1],
            bounds["v_min"],
            bounds["v_max"],
        ),
    ):
        if direction > 1e-12:
            candidates.append((maximum - coordinate) / direction)
        elif direction < -1e-12:
            candidates.append((minimum - coordinate) / direction)

    positive = [
        float(value)
        for value in candidates
        if value >= 0.0
    ]

    if not positive:
        raise RuntimeError("failed to intersect ray with boundary")

    return min(positive)


def maximum_symmetric_half_length(
    center_uv: np.ndarray,
    direction_uv: np.ndarray,
    bounds: dict[str, float],
) -> float:
    """
    공통 중심을 지나는 선분:

        p0 = center - h * direction
        p1 = center + h * direction

    이 rectangle 안에 존재하도록 하는 최대 h를 반환한다.
    """
    positive_distance = directional_distance_to_boundary(
        center_uv,
        direction_uv,
        bounds,
    )

    negative_distance = directional_distance_to_boundary(
        center_uv,
        -np.asarray(direction_uv, dtype=float),
        bounds,
    )

    return float(min(positive_distance, negative_distance))


def generate_radial_line_pattern(
    T_base_plane: np.ndarray,
    bounds: dict[str, float],
    *,
    line_count: int = 9,
    line_half_length_mm: float | None = None,
    start_angle_deg: float = 0.0,
    pattern_center_uv: np.ndarray | None = None,
    auto_length_scale: float = 0.9,
) -> dict:
    """
    공통 중심을 지나는 radial line pattern을 생성한다.

    line i:
        phi_i = start_angle + i * 360 / line_count
        direction_i = [cos(phi_i), sin(phi_i)]

        p0_i = center - h * direction_i
        p1_i = center + h * direction_i

    9개 line이면 각도 간격은 40도이다.
    """
    if line_count < 2:
        raise ValueError("line_count must be at least 2")

    if not 0.0 < auto_length_scale <= 1.0:
        raise ValueError("auto_length_scale must be in (0, 1]")

    T_base_plane = np.asarray(
        T_base_plane,
        dtype=float,
    ).reshape(4, 4)

    if pattern_center_uv is None:
        pattern_center_uv = np.array(
            [
                0.5 * (bounds["u_min"] + bounds["u_max"]),
                0.5 * (bounds["v_min"] + bounds["v_max"]),
            ],
            dtype=float,
        )
    else:
        pattern_center_uv = np.asarray(
            pattern_center_uv,
            dtype=float,
        ).reshape(2)

    if not point_inside_bounds(pattern_center_uv, bounds):
        raise ValueError("pattern center is outside boundary")

    angles_deg = (
        float(start_angle_deg)
        + np.arange(line_count, dtype=float) * (360.0 / line_count)
    )

    directions_uv = []
    maximum_half_lengths = []

    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)

        direction_uv = np.array(
            [
                np.cos(angle_rad),
                np.sin(angle_rad),
            ],
            dtype=float,
        )
        direction_uv = normalize(direction_uv)

        directions_uv.append(direction_uv)

        maximum_half_lengths.append(
            maximum_symmetric_half_length(
                pattern_center_uv,
                direction_uv,
                bounds,
            )
        )

    # 모든 방향에서 동일 길이를 사용하기 위한 공통 최대 half length.
    common_maximum_half_length = float(
        min(maximum_half_lengths)
    )

    if common_maximum_half_length <= 0.0:
        raise ValueError("no positive radial line length fits inside boundary")

    if line_half_length_mm is None:
        used_half_length = (
            float(auto_length_scale)
            * common_maximum_half_length
        )
    else:
        used_half_length = float(line_half_length_mm)

    if used_half_length <= 0.0:
        raise ValueError("line_half_length_mm must be positive")

    if used_half_length > common_maximum_half_length + 1e-9:
        raise ValueError(
            f"line_half_length_mm={used_half_length:.6f} exceeds "
            f"common maximum {common_maximum_half_length:.6f} mm"
        )

    pattern_center_base = uv_to_base(
        pattern_center_uv.reshape(1, 2),
        T_base_plane,
    )[0]

    lines = []

    for line_index, (
        angle_deg,
        direction_uv,
        direction_max_half_length,
    ) in enumerate(
        zip(
            angles_deg,
            directions_uv,
            maximum_half_lengths,
        )
    ):
        p0_uv = (
            pattern_center_uv
            - used_half_length * direction_uv
        )

        p1_uv = (
            pattern_center_uv
            + used_half_length * direction_uv
        )

        if not (
            point_inside_bounds(p0_uv, bounds)
            and point_inside_bounds(p1_uv, bounds)
        ):
            raise RuntimeError(
                f"line {line_index} endpoints are outside boundary"
            )

        endpoints_base = uv_to_base(
            np.vstack([p0_uv, p1_uv]),
            T_base_plane,
        )

        direction_base = (
            T_base_plane[:3, :2] @ direction_uv
        )
        direction_base = normalize(direction_base)

        lines.append(
            {
                "line_index": int(line_index),
                "angle_deg": float(angle_deg),
                "center_uv_mm": pattern_center_uv.tolist(),
                "p0_uv_mm": p0_uv.tolist(),
                "p1_uv_mm": p1_uv.tolist(),
                "direction_uv": direction_uv.tolist(),
                "center_base_mm": pattern_center_base.tolist(),
                "p0_base_mm": endpoints_base[0].tolist(),
                "p1_base_mm": endpoints_base[1].tolist(),
                "direction_base": direction_base.tolist(),
                "maximum_half_length_for_direction_mm": float(
                    direction_max_half_length
                ),
            }
        )

    return {
        "pattern_type": "common_center_radial_lines",
        "line_count": int(line_count),
        "angle_step_deg": float(360.0 / line_count),
        "start_angle_deg": float(start_angle_deg),
        "pattern_center_uv_mm": pattern_center_uv.tolist(),
        "pattern_center_base_mm": pattern_center_base.tolist(),
        "line_half_length_mm": float(used_half_length),
        "line_length_mm": float(2.0 * used_half_length),
        "common_maximum_half_length_mm": float(
            common_maximum_half_length
        ),
        "auto_length_scale": float(auto_length_scale),
        "bounds_uv_mm": bounds,
        "T_base_plane": T_base_plane.tolist(),
        "lines": lines,
    }


def save_plot(
    pattern: dict,
    output_path: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    bounds = pattern["bounds_uv_mm"]

    figure, axis = plt.subplots(figsize=(8, 8))

    rectangle_u = [
        bounds["u_min"],
        bounds["u_max"],
        bounds["u_max"],
        bounds["u_min"],
        bounds["u_min"],
    ]

    rectangle_v = [
        bounds["v_min"],
        bounds["v_min"],
        bounds["v_max"],
        bounds["v_max"],
        bounds["v_min"],
    ]

    axis.plot(
        rectangle_u,
        rectangle_v,
        linewidth=2,
        label="plane boundary",
    )

    for line in pattern["lines"]:
        p0 = np.asarray(line["p0_uv_mm"], dtype=float)
        p1 = np.asarray(line["p1_uv_mm"], dtype=float)

        axis.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            linewidth=2,
        )

        label_position = p1
        axis.text(
            label_position[0],
            label_position[1],
            str(line["line_index"]),
        )

    center = np.asarray(
        pattern["pattern_center_uv_mm"],
        dtype=float,
    )

    axis.scatter(
        [center[0]],
        [center[1]],
        marker="x",
        label="common center",
    )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("plane u [mm]")
    axis.set_ylabel("plane v [mm]")
    axis.set_title("Common-center radial line pattern")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create common-center radial target lines "
            "inside an estimated plane boundary."
        )
    )

    parser.add_argument(
        "--plane-boundary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--line-count",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--line-half-length-mm",
        type=float,
        default=None,
        help=(
            "Half line length. If omitted, a common valid length "
            "is computed automatically."
        ),
    )

    parser.add_argument(
        "--start-angle-deg",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--center-u-mm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--center-v-mm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--auto-length-scale",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radial_pattern.json"),
    )

    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_plane_boundary(args.plane_boundary)
    bounds = read_bounds(data)

    T_base_plane = np.asarray(
        data["plane_frame"]["T_base_plane"],
        dtype=float,
    )

    if (
        args.center_u_mm is None
        and args.center_v_mm is None
    ):
        pattern_center_uv = None

    elif (
        args.center_u_mm is not None
        and args.center_v_mm is not None
    ):
        pattern_center_uv = np.array(
            [
                args.center_u_mm,
                args.center_v_mm,
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            "--center-u-mm and --center-v-mm must be provided together"
        )

    pattern = generate_radial_line_pattern(
        T_base_plane=T_base_plane,
        bounds=bounds,
        line_count=args.line_count,
        line_half_length_mm=args.line_half_length_mm,
        start_angle_deg=args.start_angle_deg,
        pattern_center_uv=pattern_center_uv,
        auto_length_scale=args.auto_length_scale,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            pattern,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.plot is not None:
        args.plot.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        save_plot(pattern, args.plot)

    print(f"saved: {args.output}")
    print(
        "pattern center [uv, mm]:",
        pattern["pattern_center_uv_mm"],
    )
    print(
        "line count:",
        pattern["line_count"],
    )
    print(
        "angle step [deg]:",
        pattern["angle_step_deg"],
    )
    print(
        "line half length [mm]:",
        pattern["line_half_length_mm"],
    )
    print(
        "line full length [mm]:",
        pattern["line_length_mm"],
    )
    print(
        "common maximum half length [mm]:",
        pattern["common_maximum_half_length_mm"],
    )

    for line in pattern["lines"]:
        print(
            f"[{line['line_index']:02d}] "
            f"angle={line['angle_deg']:.3f} deg | "
            f"p0_uv={line['p0_uv_mm']} | "
            f"p1_uv={line['p1_uv_mm']}"
        )


if __name__ == "__main__":
    main()