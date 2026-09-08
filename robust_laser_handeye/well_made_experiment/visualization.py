from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from robust_laser_handeye.laser_handeye.data import (
    LaserScan,
    PlaneFrame,
)


def transform_points(
    T: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    points = np.asarray(points, dtype=float).reshape(-1, 3)

    R = T[:3, :3]
    t = T[:3, 3]

    return (R @ points.T).T + t


def plot_single_plane_scans(
    frame: PlaneFrame,
    board_center: np.ndarray,
    sensor_poses: list[np.ndarray],
    scans: list[LaserScan],
    plane_size_mm: float = 300.0,
    axis_length_mm: float = 40.0,
) -> None:

    board_center = np.asarray(
        board_center,
        dtype=float,
    ).reshape(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --------------------------------------------------
    # Plane
    # --------------------------------------------------
    grid = np.linspace(
        -plane_size_mm / 2.0,
        plane_size_mm / 2.0,
        15,
    )

    uu, vv = np.meshgrid(grid, grid)

    plane_points = (
        board_center[None, None, :]
        + uu[:, :, None] * frame.u
        + vv[:, :, None] * frame.v
    )

    ax.plot_surface(
        plane_points[:, :, 0],
        plane_points[:, :, 1],
        plane_points[:, :, 2],
        alpha=0.2,
    )

    # --------------------------------------------------
    # Sensor poses + profiles
    # --------------------------------------------------
    for T_base_s, scan in zip(sensor_poses, scans):

        T_base_s = np.asarray(
            T_base_s,
            dtype=float,
        ).reshape(4, 4)

        origin = T_base_s[:3, 3]
        R = T_base_s[:3, :3]

        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        # sensor X axis - red
        ax.quiver(
            origin[0], origin[1], origin[2],
            x_axis[0], x_axis[1], x_axis[2],
            length=axis_length_mm,
            color="r",
        )

        # sensor Y axis - green
        ax.quiver(
            origin[0], origin[1], origin[2],
            y_axis[0], y_axis[1], y_axis[2],
            length=axis_length_mm,
            color="g",
        )

        # sensor Z axis - blue
        ax.quiver(
            origin[0], origin[1], origin[2],
            z_axis[0], z_axis[1], z_axis[2],
            length=axis_length_mm,
            color="b",
        )

        # profile: sensor -> base
        profile_base = transform_points(
            T_base_s,
            scan.points_s,
        )

        ax.plot(
            profile_base[:, 0],
            profile_base[:, 1],
            profile_base[:, 2],
            color="m",
        )

    # --------------------------------------------------
    # Board center + plane normal
    # --------------------------------------------------
    ax.scatter(
        board_center[0],
        board_center[1],
        board_center[2],
        s=40,
        color="k",
    )

    ax.quiver(
        board_center[0],
        board_center[1],
        board_center[2],
        frame.n[0],
        frame.n[1],
        frame.n[2],
        length=axis_length_mm,
        color="k",
    )

    ax.set_xlabel("Base X [mm]")
    ax.set_ylabel("Base Y [mm]")
    ax.set_zlabel("Base Z [mm]")

    ax.set_box_aspect((1, 1, 1))

    plt.show()