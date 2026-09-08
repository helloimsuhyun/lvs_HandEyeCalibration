#!/usr/bin/env python3
"""
Visualize dense PCL Harris3D response over a CAD surface as a heatmap.

Normal convention:
  CAD triangle normals are sampled directly from the mesh and preserved through
  voxel downsampling. PCA/radius-based point-cloud normal estimation is NOT used.
  This assumes the STL triangle winding is consistently outward.

Visualization:
  cyan wire sphere = exact Harris neighborhood radius around the selected point
  white points     = sampled CAD points inside that radius
  magenta marker   = selected Harris-response point
  green arrows     = sampled CAD triangle-normal directions

Key point:
  pcl.harris3d(..., nonmax=False)
returns a Harris response for every processed CAD point rather than only
non-maximum-suppressed keypoints.

Example
-------
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/harris_prior_scan_registration.py \
  /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
  --mesh-unit mm \
  --scan-length-mm 30 \
  --init-error-translation-mm 6 \
  --init-error-rotation-deg 5 \
  --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise RuntimeError("Open3D is required: pip install open3d") from exc

try:
    import pclpybridge as pcl
except ImportError as exc:
    raise RuntimeError(
        "pclpybridge is required. Install it in the same Python environment."
    ) from exc


def positive_float(v: str) -> float:
    x = float(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def positive_int(v: str) -> int:
    x = int(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def percentile_0_100(v: str) -> float:
    x = float(v)
    if not 0.0 <= x <= 100.0:
        raise argparse.ArgumentTypeError("must be in [0, 100]")
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show dense PCL Harris3D response heatmap on a CAD surface."
    )
    p.add_argument("cad", type=Path)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")

    p.add_argument(
        "--sample-points",
        type=positive_int,
        default=120000,
        help="Uniform mesh samples before voxel downsampling.",
    )
    p.add_argument(
        "--voxel-mm",
        type=positive_float,
        default=0.5,
        help="PCL VoxelGrid leaf size [mm].",
    )
    p.add_argument(
        "--harris-radius-mm",
        type=positive_float,
        default=3.0,
        help="PCL Harris3D neighborhood radius [mm].",
    )
    p.add_argument(
        "--method",
        choices=("HARRIS", "NOBLE", "LOWE", "TOMASI", "CURVATURE"),
        default="HARRIS",
    )
    p.add_argument("--threads", type=int, default=0)

    # Visualization only: raw response itself is not changed.
    p.add_argument(
        "--clip-low",
        type=percentile_0_100,
        default=1.0,
        help="Lower percentile used only for heatmap color normalization.",
    )
    p.add_argument(
        "--clip-high",
        type=percentile_0_100,
        default=99.0,
        help="Upper percentile used only for heatmap color normalization.",
    )
    p.add_argument(
        "--point-size",
        type=positive_float,
        default=4.0,
        help="Open3D heatmap point size.",
    )
    p.add_argument(
        "--colormap",
        default="turbo",
        help="Matplotlib colormap name, e.g. turbo, viridis, plasma, jet.",
    )

    p.add_argument(
        "--harris-viz-rank",
        type=positive_int,
        default=1,
        help=(
            "Visualize the Harris neighborhood around this response rank "
            "(1 = strongest response point)."
        ),
    )
    p.add_argument(
        "--normal-count",
        type=positive_int,
        default=250,
        help="Approximate number of CAD normal arrows to draw.",
    )
    p.add_argument(
        "--normal-length-mm",
        type=positive_float,
        default=3.0,
        help="Displayed CAD normal arrow length [mm].",
    )
    p.add_argument(
        "--no-radius-viz",
        action="store_true",
        help="Do not draw the Harris-radius sphere/neighborhood overlay.",
    )
    p.add_argument(
        "--no-normal-viz",
        action="store_true",
        help="Do not draw CAD normal arrows.",
    )

    p.add_argument("--show", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_centered_mesh(path: Path, mesh_unit: str):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"failed to load mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    center_raw = np.asarray(bbox.get_center(), dtype=np.float64)
    extent_raw = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag_raw = float(np.linalg.norm(extent_raw))

    if mesh_unit == "m":
        scale = 1.0
        unit_label = "m"
    elif mesh_unit == "mm":
        scale = 0.001
        unit_label = "mm"
    else:
        # Same convention as the previous Harris3D CAD script.
        scale = 0.001 if diag_raw > 10.0 else 1.0
        unit_label = "mm(auto)" if scale == 0.001 else "m(auto)"

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices[:] = (vertices - center_raw) * scale

    # Use CAD triangle winding as the surface-normal convention.
    # NOTE: this assumes the STL triangle winding is consistently outward.
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    return mesh, {
        "input_unit": unit_label,
        "diameter_mm": diag_raw * scale * 1000.0,
        "center_raw": center_raw,
        "extent_raw": extent_raw,
    }


def robust_normalize(
    response: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> tuple[np.ndarray, float, float]:
    r = np.asarray(response, dtype=np.float64).reshape(-1)

    finite_mask = np.isfinite(r)
    if not np.any(finite_mask):
        raise RuntimeError("Harris3D returned no finite responses.")

    finite = r[finite_mask]

    lo = float(np.percentile(finite, clip_low))
    hi = float(np.percentile(finite, clip_high))

    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))

    if hi <= lo:
        q = np.zeros_like(r, dtype=np.float64)
        q[finite_mask] = 0.5
        return q, lo, hi

    q = np.zeros_like(r, dtype=np.float64)
    q[finite_mask] = np.clip(
        (r[finite_mask] - lo) / (hi - lo),
        0.0,
        1.0,
    )
    return q, lo, hi


def response_to_colors(q: np.ndarray, colormap: str) -> np.ndarray:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: pip install matplotlib") from exc

    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = cmap(np.asarray(q, dtype=np.float64))
    return np.asarray(rgba[:, :3], dtype=np.float64)


def save_results(
    output_dir: Path,
    points: np.ndarray,
    response: np.ndarray,
    colors: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_cloud = o3d.geometry.PointCloud()
    heatmap_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    heatmap_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    ply_path = output_dir / "harris3d_response_heatmap.ply"
    o3d.io.write_point_cloud(str(ply_path), heatmap_cloud)

    csv_path = output_dir / "harris3d_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x_mm", "y_mm", "z_mm", "harris_response"])
        for idx, (p, r) in enumerate(zip(points, response)):
            w.writerow(
                [
                    idx,
                    float(p[0] * 1000.0),
                    float(p[1] * 1000.0),
                    float(p[2] * 1000.0),
                    float(r),
                ]
            )

    print(f"  saved: {ply_path.resolve()}")
    print(f"  saved: {csv_path.resolve()}")


def plot_response_distribution(
    output_dir: Path,
    response: np.ndarray,
    clip_lo_value: float,
    clip_hi_value: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -> skip response plots")
        return

    r = np.asarray(response, dtype=np.float64)
    finite = r[np.isfinite(r)]
    if len(finite) == 0:
        return

    # Histogram
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.hist(finite, bins=100)
    ax.axvline(clip_lo_value, linestyle="--", linewidth=1.2, label="heatmap lower clip")
    ax.axvline(clip_hi_value, linestyle="--", linewidth=1.2, label="heatmap upper clip")
    ax.set_xlabel("Harris3D response")
    ax.set_ylabel("Point count")
    ax.set_title("Dense Harris3D response distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "harris3d_response_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path.resolve()}")

    # Sorted response curve
    ordered = np.sort(finite)[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(np.arange(1, len(ordered) + 1), ordered, linewidth=1.2)
    ax.set_xlabel("Response rank")
    ax.set_ylabel("Harris3D response")
    ax.set_title("Dense Harris3D response by rank")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_dir / "harris3d_response_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved: {path.resolve()}")


def make_wire_sphere(
    center: np.ndarray,
    radius_m: float,
) -> o3d.geometry.LineSet:
    """Create a wireframe sphere showing the exact Harris search radius."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(radius_m),
        resolution=20,
    )
    sphere.translate(np.asarray(center, dtype=np.float64))
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wire.paint_uniform_color((0.0, 1.0, 1.0))
    return wire


def make_normal_arrows(
    points: np.ndarray,
    normals: np.ndarray,
    normal_count: int,
    normal_length_m: float,
) -> o3d.geometry.LineSet:
    """
    Draw lightweight 3D arrows for CAD normals.

    Each arrow consists of:
      - one shaft from p to p + L*n
      - two short arrow-head wings near the tip
    """
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)

    if len(pts) == 0 or len(nrm) != len(pts):
        raise RuntimeError("Cannot draw normals: invalid point/normal arrays.")

    count = max(1, min(int(normal_count), len(pts)))
    sample_idx = np.linspace(
        0,
        len(pts) - 1,
        num=count,
        dtype=np.int64,
    )

    vertices = []
    lines = []

    length = float(normal_length_m)
    head_len = 0.28 * length
    head_width = 0.12 * length

    for idx in sample_idx:
        p = pts[idx]
        n = nrm[idx]
        nn = float(np.linalg.norm(n))
        if not np.isfinite(nn) or nn <= 1e-12:
            continue
        n = n / nn

        tip = p + length * n

        # Stable perpendicular direction for the arrow head.
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(ref, n))) > 0.90:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        side = np.cross(n, ref)
        side_norm = float(np.linalg.norm(side))
        if side_norm <= 1e-12:
            continue
        side /= side_norm

        back = tip - head_len * n
        wing1 = back + head_width * side
        wing2 = back - head_width * side

        base_i = len(vertices)
        vertices.extend([p, tip, wing1, wing2])
        lines.extend([
            [base_i + 0, base_i + 1],  # shaft
            [base_i + 1, base_i + 2],  # head
            [base_i + 1, base_i + 3],  # head
        ])

    arrow_set = o3d.geometry.LineSet()
    arrow_set.points = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )
    arrow_set.lines = o3d.utility.Vector2iVector(
        np.asarray(lines, dtype=np.int32)
    )
    arrow_set.paint_uniform_color((0.2, 1.0, 0.2))
    return arrow_set


def show_heatmap(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    response: np.ndarray,
    point_size: float,
    harris_radius_m: float,
    harris_viz_rank: int,
    normal_count: int,
    normal_length_m: float,
    show_radius_viz: bool,
    show_normal_viz: bool,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64).reshape(-1)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64)
    )

    geoms = [cloud]

    if show_radius_viz:
        finite_idx = np.flatnonzero(np.isfinite(response))
        if len(finite_idx) == 0:
            raise RuntimeError("No finite Harris response for radius visualization.")

        # Sort only finite responses, strongest first.
        ranked = finite_idx[
            np.argsort(response[finite_idx])[::-1]
        ]
        rank0 = min(max(int(harris_viz_rank), 1), len(ranked)) - 1
        center_idx = int(ranked[rank0])
        center = points[center_idx]

        dist = np.linalg.norm(points - center[None, :], axis=1)
        neighbor_mask = dist <= float(harris_radius_m)
        neighbor_points = points[neighbor_mask]

        # Exact radius sphere.
        geoms.append(make_wire_sphere(center, harris_radius_m))

        # Points actually lying inside the radius.
        neighborhood = o3d.geometry.PointCloud()
        neighborhood.points = o3d.utility.Vector3dVector(neighbor_points)
        neighborhood.paint_uniform_color((1.0, 1.0, 1.0))
        geoms.append(neighborhood)

        # Mark the selected center with a small sphere.
        center_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(float(harris_radius_m) * 0.08, 0.00025),
            resolution=12,
        )
        center_marker.translate(center)
        center_marker.paint_uniform_color((1.0, 0.0, 1.0))
        geoms.append(center_marker)

        print(
            "  Harris-radius visualization:"
            f" rank={rank0 + 1},"
            f" response={response[center_idx]:.9g},"
            f" radius={harris_radius_m * 1000.0:.3f} mm,"
            f" neighbors={int(np.count_nonzero(neighbor_mask))}"
        )
        print(
            "    cyan wire sphere = exact Harris search radius, "
            "white points = points inside radius, magenta = center"
        )

    if show_normal_viz:
        arrows = make_normal_arrows(
            points,
            normals,
            normal_count=normal_count,
            normal_length_m=normal_length_m,
        )
        geoms.append(arrows)
        print(
            f"  Normal visualization: ~{min(normal_count, len(points))} arrows, "
            f"length={normal_length_m * 1000.0:.3f} mm"
        )
        print("    green arrows = CAD triangle-normal direction")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        "Harris3D heatmap | cyan=radius | green=CAD normals",
        width=1440,
        height=900,
    )

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray((0.025, 0.025, 0.03))
    opt.point_size = float(point_size)
    opt.line_width = 2.0

    vis.run()
    vis.destroy_window()


def print_stats(response: np.ndarray) -> None:
    r = np.asarray(response, dtype=np.float64)
    finite = r[np.isfinite(r)]

    percentiles = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    values = np.percentile(finite, percentiles)

    print("  Harris response statistics:")
    for p, v in zip(percentiles, values):
        print(f"    P{p:>5g}: {v:.9g}")


def main() -> int:
    args = parse_args()

    if not args.cad.is_file():
        raise RuntimeError(f"CAD file does not exist: {args.cad}")
    if args.clip_low >= args.clip_high:
        raise RuntimeError("--clip-low must be smaller than --clip-high")
    if args.threads < 0:
        raise RuntimeError("--threads must be >= 0")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"{args.cad.stem}_harris3d_heatmap")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load + center CAD")
    mesh, meta = load_centered_mesh(args.cad, args.mesh_unit)
    print(
        f"  unit={meta['input_unit']}, "
        f"diameter={meta['diameter_mm']:.3f} mm"
    )

    print("[2/4] Sample CAD surface with triangle normals + voxel downsample")

    # Sample positions and CAD triangle normals together.
    # Unlike the old version, normals are NOT re-estimated from the point cloud.
    dense = mesh.sample_points_uniformly(
        number_of_points=args.sample_points,
        use_triangle_normal=True,
    )

    if not dense.has_normals():
        raise RuntimeError(
            "Open3D sampling did not return CAD normals. "
            "Check Open3D version / mesh triangle normals."
        )

    # Open3D voxel downsampling keeps point attributes such as normals.
    down = dense.voxel_down_sample(
        voxel_size=args.voxel_mm / 1000.0,
    )

    points = np.asarray(down.points, dtype=np.float32)
    normals = np.asarray(down.normals, dtype=np.float32)

    if len(points) == 0:
        raise RuntimeError("Voxel downsampling produced an empty cloud.")
    if len(normals) != len(points):
        raise RuntimeError(
            "Point/normal count mismatch after voxel downsampling: "
            f"points={len(points)}, normals={len(normals)}"
        )

    # Voxel averaging can slightly change normal magnitude, so renormalize.
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = np.isfinite(normal_norm[:, 0]) & (normal_norm[:, 0] > 1e-12)
    if not np.all(valid):
        bad = int(np.count_nonzero(~valid))
        raise RuntimeError(f"Found {bad} invalid CAD normals after voxel downsampling.")

    normals = (normals / normal_norm).astype(np.float32)

    print(
        f"  dense={len(dense.points):,} -> voxel={len(points):,} "
        f"(leaf={args.voxel_mm:.3f} mm)"
    )
    print("  normals=CAD triangle normals (no PCA normal estimation)")

    print("[3/4] Dense Harris3D response using CAD surface normals")

    # IMPORTANT:
    # nonmax=False => PCL returns Harris response for each input point.
    # Thresholding/refinement are only meaningful with nonmax=True.
    response_points, response = pcl.harris3d(
        points,
        radius=args.harris_radius_mm / 1000.0,
        threshold=0.0,
        nonmax=False,
        refine=False,
        method=args.method,
        normals=normals,
        threads=args.threads,
    )

    response_points = np.asarray(response_points, dtype=np.float32)
    response = np.asarray(response, dtype=np.float64).reshape(-1)

    if len(response_points) != len(response):
        raise RuntimeError(
            "pcl.harris3d returned mismatched point/response lengths: "
            f"{len(response_points)} vs {len(response)}"
        )

    # With nonmax=False, PCL should return one response per input point.
    if len(response_points) != len(points):
        print(
            "  WARNING: expected one Harris response per input point, "
            f"but got input={len(points):,}, output={len(response_points):,}."
        )

    print(
        f"  method={args.method}, radius={args.harris_radius_mm:.3f} mm, "
        f"responses={len(response):,}"
    )
    print_stats(response)

    print("[4/4] Build heatmap")
    q, clip_lo_value, clip_hi_value = robust_normalize(
        response,
        args.clip_low,
        args.clip_high,
    )
    colors = response_to_colors(q, args.colormap)

    print(
        f"  color normalization: "
        f"P{args.clip_low:g}={clip_lo_value:.9g} -> "
        f"P{args.clip_high:g}={clip_hi_value:.9g}"
    )
    print("  low response = blue/purple, high response = yellow/red")

    save_results(
        output_dir,
        response_points,
        response,
        colors,
    )
    plot_response_distribution(
        output_dir,
        response,
        clip_lo_value,
        clip_hi_value,
    )

    if args.show:
        if len(response_points) != len(normals):
            raise RuntimeError(
                "Cannot visualize CAD normals because Harris response point count "
                "does not match the CAD normal count."
            )

        show_heatmap(
            response_points,
            normals,
            colors,
            response,
            args.point_size,
            args.harris_radius_mm / 1000.0,
            args.harris_viz_rank,
            args.normal_count,
            args.normal_length_mm / 1000.0,
            show_radius_viz=not args.no_radius_viz,
            show_normal_viz=not args.no_normal_viz,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())