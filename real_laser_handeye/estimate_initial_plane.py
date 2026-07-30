
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import sys
import time
import uuid

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:  # pragma: no cover
    plt = None
    Poly3DCollection = None

try:
    from .laser_adapter import LaserAdapter
    from .__robot_adapter import RobotAdapter
except ImportError:
    from laser_adapter import LaserAdapter
    from real_laser_handeye.__robot_adapter import RobotAdapter


@dataclass(frozen=True)
class PlaneEstimate:
    centroid_w: np.ndarray
    normal_w: np.ndarray
    offset_w: float
    basis_u_w: np.ndarray
    basis_v_w: np.ndarray
    boundary_uv: np.ndarray
    boundary_w: np.ndarray
    rms_mm: float
    max_abs_mm: float
    singular_values: np.ndarray
    scan_count: int
    point_count: int


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform.copy()


def load_transform(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("T_tcp_sensor", "T_ef_s", "transform"):
                if key in value:
                    value = value[key]
                    break
        return validate_transform(np.asarray(value, dtype=float), str(path))
    try:
        value = np.loadtxt(path, delimiter=",")
    except ValueError:
        value = np.loadtxt(path)
    return validate_transform(value, str(path))


def transform_points(T_a_b: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    points = np.asarray(points_b, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ T_a_b[:3, :3].T + T_a_b[:3, 3]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def next_capture_path(dataset_dir: Path) -> Path:
    ids: list[int] = []
    for path in dataset_dir.glob("capture_*.npz"):
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return dataset_dir / f"capture_{max(ids, default=0) + 1:04d}.npz"


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def filter_profile(points_s: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = np.asarray(points_s, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("laser adapter must return an (N, 3) array")
    valid = np.all(np.isfinite(points), axis=1)
    valid &= np.abs(points[:, 1]) <= args.max_abs_sensor_y_mm
    if args.min_sensor_x_mm is not None:
        valid &= points[:, 0] >= args.min_sensor_x_mm
    if args.max_sensor_x_mm is not None:
        valid &= points[:, 0] <= args.max_sensor_x_mm
    if args.min_sensor_z_mm is not None:
        valid &= points[:, 2] >= args.min_sensor_z_mm
    if args.max_sensor_z_mm is not None:
        valid &= points[:, 2] <= args.max_sensor_z_mm
    result = points[valid]
    if len(result) < args.min_points:
        raise RuntimeError(
            f"profile has {len(result)} valid points; need at least {args.min_points}"
        )
    return result


def capture_once(robot: RobotAdapter, laser: LaserAdapter, args: argparse.Namespace) -> Path:
    T_before = validate_transform(robot.read_T_base_tcp(), "pre-capture TCP")
    points_s = filter_profile(laser.read_profile(timeout_s=args.timeout_s), args)
    profile_timestamp_ns = time.time_ns()
    T_after = validate_transform(robot.read_T_base_tcp(), "post-capture TCP")
    tcp_timestamp_ns = time.time_ns()

    translation_delta = float(np.linalg.norm(T_after[:3, 3] - T_before[:3, 3]))
    rotation_delta = rotation_distance_deg(T_before[:3, :3], T_after[:3, :3])
    if translation_delta > args.max_stationarity_translation_mm:
        raise RuntimeError(f"robot moved {translation_delta:.3f} mm during capture")
    if rotation_delta > args.max_stationarity_rotation_deg:
        raise RuntimeError(f"robot rotated {rotation_delta:.3f} deg during capture")

    T_world_sensor = T_after @ args.T_tcp_sensor
    points_w = transform_points(T_world_sensor, points_s)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    output = next_capture_path(args.dataset_dir)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                T_world_tcp=T_after,
                T_tcp_sensor=args.T_tcp_sensor,
                T_world_sensor=T_world_sensor,
                points_s=points_s,
                points_w=points_w,
                tcp_timestamp_ns=np.int64(tcp_timestamp_ns),
                profile_timestamp_ns=np.int64(profile_timestamp_ns),
                captured_at=np.array(datetime.now(timezone.utc).isoformat()),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    advance_pose = getattr(robot, "advance_pose", None)
    if callable(advance_pose):
        advance_pose()

    print(f"saved {output} ({len(points_s)} points)")
    print(f"world endpoints: {points_w[0]} -> {points_w[-1]}")
    return output


def load_world_points(dataset_dir: Path, T_tcp_sensor: np.ndarray) -> tuple[list[np.ndarray], list[Path]]:
    scans_w: list[np.ndarray] = []
    paths = sorted(dataset_dir.glob("capture_*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as pair:
            if "T_world_tcp" in pair:
                T_world_tcp = validate_transform(pair["T_world_tcp"], str(path))
            elif "T_base_tcp" in pair:
                T_world_tcp = validate_transform(pair["T_base_tcp"], str(path))
            else:
                raise ValueError(f"{path} has no T_world_tcp/T_base_tcp")

            if "points_s" not in pair:
                raise ValueError(f"{path} has no points_s")
            points_s = np.asarray(pair["points_s"], dtype=float)

        T_world_sensor = T_world_tcp @ T_tcp_sensor
        scans_w.append(transform_points(T_world_sensor, points_s))
    return scans_w, paths


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_2d(points_uv: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points_uv, dtype=float), axis=0)
    if len(points) < 3:
        raise RuntimeError("at least three non-collinear projected points are required")
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise RuntimeError("projected points are nearly collinear; vary robot pose more")
    return hull


def polygon_area(points_uv: np.ndarray) -> float:
    x = points_uv[:, 0]
    y = points_uv[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def fit_plane_and_boundary(
    scans_w: list[np.ndarray],
    *,
    normal_hint_w: np.ndarray | None = None,
) -> PlaneEstimate:
    if len(scans_w) < 2:
        raise RuntimeError(
            "one laser profile is only a 3D line; capture at least two non-collinear scans"
        )
    points_w = np.vstack(scans_w)
    if len(points_w) < 3:
        raise RuntimeError("not enough world points")

    centroid = np.mean(points_w, axis=0)
    centered = points_w - centroid
    _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1] <= 1e-9:
        raise RuntimeError("all captured profiles are collinear; change sensor pose")

    basis_u = Vt[0].copy()
    basis_v = Vt[1].copy()
    normal = Vt[2].copy()

    if np.dot(np.cross(basis_u, basis_v), normal) < 0.0:
        basis_v *= -1.0
    if normal_hint_w is not None:
        hint = np.asarray(normal_hint_w, dtype=float)
        norm = float(np.linalg.norm(hint))
        if norm <= 0.0:
            raise ValueError("normal hint must be nonzero")
        hint /= norm
        if np.dot(normal, hint) < 0.0:
            normal *= -1.0
            basis_v *= -1.0
    elif normal[2] < 0.0:
        normal *= -1.0
        basis_v *= -1.0

    signed_distances = centered @ normal
    projected = points_w - signed_distances[:, None] * normal[None, :]
    projected_centered = projected - centroid
    points_uv = np.column_stack((projected_centered @ basis_u, projected_centered @ basis_v))
    boundary_uv = convex_hull_2d(points_uv)
    boundary_w = (
        centroid[None, :]
        + boundary_uv[:, [0]] * basis_u[None, :]
        + boundary_uv[:, [1]] * basis_v[None, :]
    )

    return PlaneEstimate(
        centroid_w=centroid,
        normal_w=normal,
        offset_w=float(np.dot(normal, centroid)),
        basis_u_w=basis_u,
        basis_v_w=basis_v,
        boundary_uv=boundary_uv,
        boundary_w=boundary_w,
        rms_mm=float(np.sqrt(np.mean(signed_distances**2))),
        max_abs_mm=float(np.max(np.abs(signed_distances))),
        singular_values=singular_values,
        scan_count=len(scans_w),
        point_count=len(points_w),
    )


def set_axes_equal_3d(ax, all_points: np.ndarray) -> None:
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_world_3d(
    scans_w: list[np.ndarray],
    estimate: PlaneEstimate,
    output_path: Path,
    *,
    show_plot: bool,
    axis_length_mm: float = 50.0,
) -> None:
    if plt is None or Poly3DCollection is None:
        raise RuntimeError("matplotlib is required for 3D plotting")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for idx, scan in enumerate(scans_w, start=1):
        scan = np.asarray(scan, dtype=float)
        ax.plot(scan[:, 0], scan[:, 1], scan[:, 2], linewidth=1.5, label=f"scan {idx}")

    all_points_w = np.vstack(scans_w)
    ax.scatter(
        all_points_w[:, 0],
        all_points_w[:, 1],
        all_points_w[:, 2],
        s=4,
        alpha=0.25,
        label="all world points",
    )

    boundary_closed = np.vstack([estimate.boundary_w, estimate.boundary_w[0]])
    ax.plot(
        boundary_closed[:, 0],
        boundary_closed[:, 1],
        boundary_closed[:, 2],
        linewidth=2.5,
        label="plane boundary",
    )

    polygon = Poly3DCollection([estimate.boundary_w], alpha=0.25)
    ax.add_collection3d(polygon)

    c = estimate.centroid_w
    ax.scatter([c[0]], [c[1]], [c[2]], s=60, marker="o", label="plane centroid")

    ax.quiver(
        c[0], c[1], c[2],
        estimate.basis_u_w[0], estimate.basis_u_w[1], estimate.basis_u_w[2],
        length=axis_length_mm, normalize=True, linewidth=2.0,
    )
    ax.quiver(
        c[0], c[1], c[2],
        estimate.basis_v_w[0], estimate.basis_v_w[1], estimate.basis_v_w[2],
        length=axis_length_mm, normalize=True, linewidth=2.0,
    )
    ax.quiver(
        c[0], c[1], c[2],
        estimate.normal_w[0], estimate.normal_w[1], estimate.normal_w[2],
        length=axis_length_mm, normalize=True, linewidth=2.5,
    )

    label_u = c + axis_length_mm * estimate.basis_u_w
    label_v = c + axis_length_mm * estimate.basis_v_w
    label_n = c + axis_length_mm * estimate.normal_w
    ax.text(label_u[0], label_u[1], label_u[2], "u")
    ax.text(label_v[0], label_v[1], label_v[2], "v")
    ax.text(label_n[0], label_n[1], label_n[2], "n")

    axes_reference = np.vstack(
        [
            all_points_w,
            estimate.boundary_w,
            c[None, :],
            label_u[None, :],
            label_v[None, :],
            label_n[None, :],
        ]
    )
    set_axes_equal_3d(ax, axes_reference)

    ax.set_xlabel("World X [mm]")
    ax.set_ylabel("World Y [mm]")
    ax.set_zlabel("World Z [mm]")
    ax.set_title(
        "Estimated plane and observed profile lines\n"
        f"scans={estimate.scan_count}, points={estimate.point_count}, "
        f"RMS={estimate.rms_mm:.4f} mm"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    print(f"saved 3D plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)



def save_estimate(estimate: PlaneEstimate, output_dir: Path, capture_paths: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "plane_boundary_world.csv", estimate.boundary_w, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
    np.savetxt(output_dir / "plane_boundary_uv.csv", estimate.boundary_uv, delimiter=",", header="u_mm,v_mm", comments="")

    frame = np.eye(4)
    frame[:3, 0] = estimate.basis_u_w
    frame[:3, 1] = estimate.basis_v_w
    frame[:3, 2] = estimate.normal_w
    frame[:3, 3] = estimate.centroid_w
    np.savetxt(output_dir / "T_world_plane.csv", frame, delimiter=",", fmt="%.12g")

    atomic_json(
        output_dir / "plane_estimate.json",
        {
            "coordinate_frame": "world/base",
            "plane_equation": "normal_w dot p_w = offset_w",
            "normal_w": estimate.normal_w.tolist(),
            "offset_w_mm": estimate.offset_w,
            "centroid_w_mm": estimate.centroid_w.tolist(),
            "basis_u_w": estimate.basis_u_w.tolist(),
            "basis_v_w": estimate.basis_v_w.tolist(),
            "rms_point_to_plane_mm": estimate.rms_mm,
            "max_abs_point_to_plane_mm": estimate.max_abs_mm,
            "singular_values": estimate.singular_values.tolist(),
            "scan_count": estimate.scan_count,
            "point_count": estimate.point_count,
            "boundary_vertex_count": len(estimate.boundary_w),
            "boundary_area_mm2": polygon_area(estimate.boundary_uv),
            "boundary_world_csv": "plane_boundary_world.csv",
            "boundary_uv_csv": "plane_boundary_uv.csv",
            "T_world_plane_csv": "T_world_plane.csv",
            "world_plot_png": "plane_estimate_3d.png",
            "captures": [str(path) for path in capture_paths],
            "estimated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def estimate_from_dataset(args: argparse.Namespace) -> PlaneEstimate:
    scans_w, paths = load_world_points(args.dataset_dir, args.T_tcp_sensor)
    normal_hint = None
    if args.normal_hint_world is not None:
        normal_hint = np.asarray(args.normal_hint_world, dtype=float)
    estimate = fit_plane_and_boundary(scans_w, normal_hint_w=normal_hint)
    save_estimate(estimate, args.estimate_dir, paths)
    plot_world_3d(
        scans_w,
        estimate,
        args.estimate_dir / "plane_estimate_3d.png",
        show_plot=args.show_plot,
        axis_length_mm=args.plot_axis_length_mm,
    )

    print("\nPlane estimate")
    print(f"  scans / points : {estimate.scan_count} / {estimate.point_count}")
    print(f"  normal_w       : {np.array2string(estimate.normal_w, precision=8)}")
    print(f"  offset_w [mm]  : {estimate.offset_w:.6f}")
    print(f"  RMS [mm]       : {estimate.rms_mm:.6f}")
    print(f"  max abs [mm]   : {estimate.max_abs_mm:.6f}")
    print(f"  hull vertices  : {len(estimate.boundary_w)}")
    print(f"  hull area [mm2]: {polygon_area(estimate.boundary_uv):.3f}")
    print(f"saved estimate to {args.estimate_dir}")
    return estimate


def connect_hardware(args: argparse.Namespace) -> tuple[RobotAdapter, LaserAdapter]:
    robot = RobotAdapter(args.robot_host, args.robot_port)
    laser = LaserAdapter(
        ip=args.laser_ip,
        control_port=args.laser_control_port,
        high_speed_port=args.laser_high_speed_port,
        batch_profiles=args.batch_profiles,
        aggregate=args.aggregate,
    )
    robot.connect()
    try:
        laser.connect()
    except BaseException:
        robot.close()
        raise
    return robot, laser


def run_capture(args: argparse.Namespace) -> None:
    robot, laser = connect_hardware(args)
    try:
        capture_once(robot, laser, args)
    finally:
        laser.close()
        robot.close()


def run_session(args: argparse.Namespace) -> None:
    robot, laser = connect_hardware(args)
    print("Initial plane capture session")
    print("c + Enter: capture | p + Enter: estimate plane/boundary/3D plot | q + Enter: quit")
    print("> ", end="", flush=True)
    try:
        running = True
        while running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            command = sys.stdin.readline().strip().lower()
            try:
                if command in ("", "c", "capture"):
                    capture_once(robot, laser, args)
                elif command in ("p", "plane", "estimate"):
                    estimate_from_dataset(args)
                elif command in ("q", "quit", "exit"):
                    running = False
                else:
                    print("unknown command: c, p, q")
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
            if running:
                print("> ", end="", flush=True)
    finally:
        laser.close()
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture initial laser profiles and estimate a world-frame plane boundary"
    )
    parser.add_argument("command", nargs="?", choices=("session", "capture", "estimate"), default="session")
    parser.add_argument("--dataset-dir", type=Path, default=Path("runs/real/initial_plane/dataset"))
    parser.add_argument("--estimate-dir", type=Path, default=Path("runs/real/initial_plane/estimate"))
    parser.add_argument(
        "--handeye",
        type=Path,
        default=Path("real_laser_handeye/initial_T_tcp_sensor.json"),
        help="Current T_tcp_sensor transform",
    )

    parser.add_argument("--robot-host", default="192.168.0.10")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--laser-ip", default="192.168.1.1")
    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--batch-profiles", type=int, default=5)
    parser.add_argument("--aggregate", choices=("median", "latest"), default="median")
    parser.add_argument("--timeout-s", type=float, default=3.0)

    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--max-abs-sensor-y-mm", type=float, default=0.1)
    parser.add_argument("--min-sensor-x-mm", type=float)
    parser.add_argument("--max-sensor-x-mm", type=float)
    parser.add_argument("--min-sensor-z-mm", type=float)
    parser.add_argument("--max-sensor-z-mm", type=float)
    parser.add_argument("--max-stationarity-translation-mm", type=float, default=0.2)
    parser.add_argument("--max-stationarity-rotation-deg", type=float, default=0.2)
    parser.add_argument(
        "--normal-hint-world",
        type=float,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Optional vector used only to choose the plane normal sign",
    )
    parser.add_argument(
        "--plot-axis-length-mm",
        type=float,
        default=50.0,
        help="Axis length for the plotted plane frame",
    )
    parser.set_defaults(show_plot=True)
    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        help="Show the world-coordinate 3D plot after estimation (default)",
    )
    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help="Only save the 3D plot PNG without opening a window",
    )

    args = parser.parse_args()
    args.T_tcp_sensor = load_transform(args.handeye)
    return args


def main() -> None:
    args = parse_args()
    if args.command == "capture":
        run_capture(args)
    elif args.command == "estimate":
        estimate_from_dataset(args)
    else:
        run_session(args)


if __name__ == "__main__":
    main()