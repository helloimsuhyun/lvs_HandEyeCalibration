from __future__ import annotations

from typing import Any

import numpy as np

from .ui_model import DashboardSnapshot


def render_profile_axes(
    live_axis: Any, previous_axis: Any, snapshot: DashboardSnapshot
) -> None:
    live_axis.clear()
    previous_axis.clear()
    live = snapshot.live_profile_s
    previous = snapshot.previous_profile_s
    if len(live):
        live_axis.plot(live[:, 0], live[:, 2], color="#00c2ff", linewidth=1.2)
    if len(previous):
        previous_axis.plot(
            previous[:, 0], previous[:, 2], color="#ffb000", linewidth=1.2
        )
    live_axis.set_title(f"Live profile · seq {snapshot.live_sequence}")
    previous_axis.set_title(
        "Previous capture"
        if snapshot.previous_scan_id is None
        else f"Previous capture · scan {snapshot.previous_scan_id}"
    )
    for axis in (live_axis, previous_axis):
        axis.set_xlabel("Sensor X [mm]")
        axis.set_ylabel("Sensor Z [mm]")
        axis.grid(True, alpha=0.25)
    live_axis.figure.tight_layout()


def _set_axes_equal(axis: Any, points: np.ndarray) -> None:
    finite = points[np.all(np.isfinite(points), axis=1)]
    if not len(finite):
        return
    minimum = np.min(finite, axis=0)
    maximum = np.max(finite, axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(1.0, 0.55 * float(np.max(maximum - minimum)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _box_vertices(bounds: np.ndarray) -> np.ndarray:
    minimum, maximum = np.asarray(bounds, dtype=float).reshape(2, 3)
    return np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )


def _draw_box(
    axis: Any,
    bounds: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float,
    alpha: float,
) -> None:
    if not np.all(np.isfinite(bounds)):
        return
    vertices = _box_vertices(bounds)
    first = True
    for start in range(8):
        for bit in (1, 2, 4):
            end = start ^ bit
            if end <= start:
                continue
            axis.plot(
                vertices[[start, end], 0],
                vertices[[start, end], 1],
                vertices[[start, end], 2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label if first else None,
            )
            first = False


def _draw_frame(axis: Any, T: np.ndarray, *, scale: float, label: str) -> None:
    if not np.all(np.isfinite(T)):
        return
    origin = T[:3, 3]
    colors = ("#e53935", "#43a047", "#1e88e5")
    for index, color in enumerate(colors):
        direction = T[:3, index] * float(scale)
        axis.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=1.8,
            arrow_length_ratio=0.18,
            label=label if index == 0 else None,
        )


def render_scene_axis(axis: Any, snapshot: DashboardSnapshot) -> None:
    axis.clear()
    _draw_box(
        axis,
        snapshot.workspace_bounds_base,
        color="#607d8b",
        label="configured TCP workspace",
        linewidth=0.7,
        alpha=0.22,
    )
    for index, bounds in enumerate(snapshot.no_go_bounds_base):
        _draw_box(
            axis,
            bounds,
            color="#d32f2f",
            label="no-go boxes" if index == 0 else "",
            linewidth=1.8,
            alpha=0.8,
        )
    planned = snapshot.planned_tcp_positions_base
    completed = snapshot.completed_mask
    if len(planned):
        axis.scatter(
            planned[~completed, 0],
            planned[~completed, 1],
            planned[~completed, 2],
            s=9,
            color="#758195",
            alpha=0.45,
            label="planned TCP",
        )
        if np.any(completed):
            done = planned[completed]
            axis.scatter(
                done[:, 0], done[:, 1], done[:, 2],
                s=17, color="#2ecc71", label="captured TCP"
            )
    corners = snapshot.bootstrap_plane_corners_base
    axis.plot(
        corners[:, 0], corners[:, 1], corners[:, 2],
        color="#4285f4", linewidth=2.0, label="bootstrap boundary"
    )
    points = snapshot.accumulated_points_base
    if len(points):
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=2, color="#ff9f43", alpha=0.35, label="accumulated profiles"
        )
    estimate = snapshot.plane_estimate
    if estimate.available:
        normal = np.asarray(estimate.normal_base)
        u = corners[1] - corners[0]
        u = u - float(u @ normal) * normal
        if np.linalg.norm(u) < 1e-9:
            u = corners[2] - corners[1]
            u = u - float(u @ normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        span = max(
            10.0,
            0.55 * float(np.max(np.linalg.norm(corners - np.mean(corners, axis=0), axis=1))),
        )
        grid = np.linspace(-span, span, 8)
        U, V = np.meshgrid(grid, grid)
        surface = (
            estimate.center_base_mm[None, None, :]
            + U[:, :, None] * u[None, None, :]
            + V[:, :, None] * v[None, None, :]
        )
        axis.plot_surface(
            surface[:, :, 0], surface[:, :, 1], surface[:, :, 2],
            color="#9b59b6", alpha=0.18, linewidth=0
        )
    if snapshot.next_scan_id is not None:
        target = snapshot.next_T_base_tcp[:3, 3]
        approach = snapshot.next_T_base_tcp_approach[:3, 3]
        sensor = snapshot.next_T_base_sensor_nominal[:3, 3]
        transit = snapshot.safe_transit_T_base_tcp[:3, 3]
        if np.all(np.isfinite(transit)):
            axis.plot(
                [transit[0], approach[0], target[0]],
                [transit[1], approach[1], target[1]],
                [transit[2], approach[2], target[2]],
                color="#ff3b30",
                linestyle="--",
                linewidth=2.1,
                label="next safe route",
            )
            axis.scatter(
                [transit[0]],
                [transit[1]],
                [transit[2]],
                marker="s",
                s=45,
                color="#8e44ad",
                label="safe transit",
            )
        axis.plot(
            [approach[0], target[0]],
            [approach[1], target[1]],
            [approach[2], target[2]],
            color="#ff3b30", linewidth=3.0, label="next approach → TCP"
        )
        axis.scatter(
            [target[0]], [target[1]], [target[2]],
            marker="*", s=130, color="#ff3b30", label="next TCP"
        )
        axis.scatter(
            [sensor[0]], [sensor[1]], [sensor[2]],
            marker="^", s=70, color="#00c2ff", label="nominal sensor"
        )
        scale = max(12.0, 0.08 * float(np.linalg.norm(corners[2] - corners[0])))
        _draw_frame(
            axis,
            snapshot.next_T_base_tcp,
            scale=scale,
            label="next TCP XYZ axes",
        )
        _draw_frame(
            axis,
            snapshot.next_T_base_sensor_nominal,
            scale=0.8 * scale,
            label="next sensor XYZ axes",
        )
    all_points = [planned, corners, points]
    if snapshot.next_scan_id is not None:
        all_points.extend(
            [
                snapshot.next_T_base_tcp[:3, 3][None, :],
                snapshot.next_T_base_tcp_approach[:3, 3][None, :],
                snapshot.safe_transit_T_base_tcp[:3, 3][None, :],
            ]
        )
    finite_sets = [
        item
        for item in all_points
        if len(item) and np.any(np.all(np.isfinite(item), axis=1))
    ]
    _set_axes_equal(axis, np.vstack(finite_sets))
    axis.set_xlabel("Base X [mm]")
    axis.set_ylabel("Base Y [mm]")
    axis.set_zlabel("Base Z [mm]")
    axis.set_title(
        f"Plan {snapshot.completed_count}/{snapshot.total_count} · "
        f"plane: {estimate.status}"
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)


def render_calibration_axis(axis: Any, diagnostics: dict[str, Any] | None) -> None:
    axis.clear()
    axis.set_title("Calibration convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Value (log scale)")
    axis.grid(True, alpha=0.25)
    if not diagnostics:
        axis.text(
            0.5, 0.5, "Calibration has not been run",
            ha="center", va="center", transform=axis.transAxes
        )
        return
    plane = np.asarray(diagnostics.get("delta_history", []), dtype=float)
    if len(plane):
        axis.semilogy(
            np.arange(1, len(plane) + 1),
            np.maximum(plane, 1e-16),
            marker="o", markersize=3, label="||T(k)-T(k-1)||"
        )
    conditions = np.asarray(diagnostics.get("condition_history", []), dtype=float)
    if len(conditions):
        axis.semilogy(
            np.arange(1, len(conditions) + 1),
            np.maximum(conditions, 1e-16),
            label="linear condition"
        )
    axis.legend(loc="best")
