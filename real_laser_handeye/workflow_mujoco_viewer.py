from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from handeye_mujoco import HandEyeSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated native MuJoCo trajectory viewer"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--title", default="MuJoCo trajectory preview")
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=60.0,
        help="viewer refresh rate",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help=(
            "preview-only playback multiplier; this does not change the "
            "saved MoveIt trajectory timing"
        ),
    )
    return parser.parse_args()


def _load_preview(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load timed .npz previews, with legacy .npy fallback."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "qpos" not in loaded:
                raise ValueError("timed preview .npz is missing 'qpos'")
            qpos = np.asarray(loaded["qpos"], dtype=float)
            times = (
                None
                if "time_s" not in loaded
                else np.asarray(loaded["time_s"], dtype=float).reshape(-1)
            )
        finally:
            loaded.close()
        return qpos, times
    return np.asarray(loaded, dtype=float), None


def _validate_timed_preview(
    qpos: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=float)
    times = np.asarray(times, dtype=float).reshape(-1)
    if qpos.ndim != 2 or len(qpos) == 0:
        raise ValueError(
            f"trajectory must be a non-empty 2D array, got {qpos.shape}"
        )
    if times.shape != (len(qpos),):
        raise ValueError(
            "time_s must contain exactly one timestamp per qpos sample"
        )
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(times)):
        raise ValueError("trajectory/time_s contains non-finite values")
    if np.any(times < 0.0):
        raise ValueError("time_s contains negative values")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    return qpos, times - float(times[0])


def _interpolate_qpos(
    qpos: np.ndarray,
    times: np.ndarray,
    target_s: float,
) -> np.ndarray:
    if target_s <= 0.0 or len(qpos) == 1:
        return qpos[0]
    if target_s >= float(times[-1]):
        return qpos[-1]

    right = int(np.searchsorted(times, target_s, side="right"))
    left = right - 1
    t0 = float(times[left])
    t1 = float(times[right])
    alpha = (float(target_s) - t0) / (t1 - t0)
    return (1.0 - alpha) * qpos[left] + alpha * qpos[right]


def _scaled_playback_time(
    elapsed_wall_s: float,
    duration_s: float,
    playback_speed: float,
) -> float:
    """Map wall time to preview time without modifying trajectory timestamps."""
    return min(max(0.0, elapsed_wall_s) * playback_speed, duration_s)


def _configure_viewer_visuals(viewer, model) -> None:
    """Match the project's intended MuJoCo visualization policy.

    The UR/RB MuJoCo files contain both pretty visual meshes and simplified
    collision geoms.  Collision geoms live in geom group 1 when visual meshes
    named ``robot_visual_*`` are present.  Keep every other group visible so
    the generated calibration plane/board and sensor debug geometry remain
    visible.
    """
    import mujoco

    viewer.opt.geomgroup[:] = 1

    has_robot_visuals = any(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(
            "robot_visual_"
        )
        for index in range(model.ngeom)
    )
    if has_robot_visuals:
        # Hide the simplified collision/bounding geoms, not the visual meshes.
        viewer.opt.geomgroup[1] = 0

    # Start from the same named overview camera used by the in-app renderer.
    # The user can still change camera controls in the native viewer afterward.
    overview_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "overview",
    )
    if overview_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = int(overview_id)

    plane_like: list[str] = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        lowered = name.lower()
        if "plane" in lowered or "board" in lowered or "calibration" in lowered:
            plane_like.append(f"{name or '<unnamed>'}(group={int(model.geom_group[index])})")

    print(
        "MuJoCo visual policy: "
        f"robot_visuals={'yes' if has_robot_visuals else 'no'}, "
        f"geomgroup={np.asarray(viewer.opt.geomgroup, dtype=int).tolist()}, "
        f"overview_camera={'yes' if overview_id >= 0 else 'no'}"
    )
    if plane_like:
        print("Plane/board geoms: " + ", ".join(plane_like))


def _view_timed(
    simulation: HandEyeSimulation,
    qpos: np.ndarray,
    times: np.ndarray,
    *,
    title: str,
    refresh_hz: float,
    playback_speed: float,
) -> None:
    if not np.isfinite(refresh_hz) or refresh_hz <= 0.0:
        raise ValueError("--refresh-hz must be positive and finite")
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive and finite")

    import mujoco
    import mujoco.viewer

    duration_s = float(times[-1])
    print(
        f"{title}: {len(qpos)} MoveIt samples, "
        f"trajectory duration={duration_s:.3f} s, "
        f"preview={playback_speed:g}x "
        f"({duration_s / playback_speed:.3f} s wall time)"
    )

    simulation.reset_home()
    simulation.set_joint_positions(qpos[0])
    mujoco.mj_forward(simulation.model, simulation.data)

    frame_period_s = 1.0 / refresh_hz
    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
    ) as viewer:
        _configure_viewer_visuals(viewer, simulation.model)
        viewer.sync()

        started = time.monotonic()
        finished_reported = False

        while viewer.is_running():
            loop_started = time.monotonic()
            elapsed_s = loop_started - started
            playback_s = _scaled_playback_time(
                elapsed_s,
                duration_s,
                playback_speed,
            )
            simulation.set_joint_positions(
                _interpolate_qpos(qpos, times, playback_s)
            )
            mujoco.mj_forward(simulation.model, simulation.data)
            viewer.sync()

            if playback_s >= duration_s and not finished_reported:
                print(
                    f"{title}: playback complete; holding final pose "
                    "until the viewer is closed"
                )
                finished_reported = True

            remaining_sleep = frame_period_s - (time.monotonic() - loop_started)
            if remaining_sleep > 0.0:
                time.sleep(remaining_sleep)


def main() -> None:
    args = parse_args()
    trajectory, times = _load_preview(args.trajectory)
    if trajectory.ndim != 2 or len(trajectory) == 0:
        raise ValueError(
            f"trajectory must be a non-empty 2D array, got {trajectory.shape}"
        )

    simulation = HandEyeSimulation(args.model)

    # Legacy files keep the project's original viewer path, which already has
    # the established visual/camera behavior.
    if times is None:
        print(
            f"{args.title}: {len(trajectory)} legacy frames; "
            "no MoveIt timing metadata, using legacy viewer speed"
        )
        simulation.view_trajectory_interactive(trajectory)
        return

    qpos, times = _validate_timed_preview(trajectory, times)
    _view_timed(
        simulation,
        qpos,
        times,
        title=args.title,
        refresh_hz=float(args.refresh_hz),
        playback_speed=float(args.playback_speed),
    )


if __name__ == "__main__":
    main()
