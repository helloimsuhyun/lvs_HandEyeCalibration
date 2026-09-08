from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from .builder import BuildError, build_model
from .simulation import HandEyeSimulation


def _build(args: argparse.Namespace) -> int:
    result = build_model(args.config, force_cad=args.force_cad)
    print(f"model: {result.model_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"sensor parent: {result.parent_link}")
    low, high = result.sensor_bounds_m
    print(f"sensor bounds [m]: min={low.tolist()} max={high.tolist()}")
    print(f"sensor mesh cache: {'hit' if result.sensor_mesh_cached else 'rebuilt'}")
    return 0


def _check(args: argparse.Namespace) -> int:
    simulation = HandEyeSimulation(args.model)
    simulation.reset_home()
    report = simulation.collision_report()
    print(f"joints: {list(simulation.joint_names)}")
    print("T_world_sensor:")
    print(np.array2string(simulation.sensor_pose_world(), precision=6, suppress_small=True))
    print(f"contacts: {len(report.contacts)}")
    for contact in report.contacts:
        print(f"  {contact.geom1} <-> {contact.geom2}: {contact.distance_m:.6f} m")
    print(f"collision_free: {report.collision_free}")
    return 0 if report.collision_free else 2


def _info(args: argparse.Namespace) -> int:
    model = mujoco.MjModel.from_xml_path(str(Path(args.model).resolve()))
    payload = {
        "nq": model.nq,
        "nv": model.nv,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "ngeom": model.ngeom,
        "joints": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _view(args: argparse.Namespace) -> int:
    try:
        import mujoco.viewer
    except ImportError as error:
        raise BuildError(f"MuJoCo passive viewer is unavailable: {error}") from error
    simulation = HandEyeSimulation(args.model)
    simulation.reset_home()
    if args.collision_only:
        for geom_id in range(simulation.model.ngeom):
            group = int(simulation.model.geom_group[geom_id])
            if group == 1:
                # Robot collision meshes.
                simulation.model.geom_rgba[geom_id] = [0.2, 0.55, 1.0, 1.0]
            elif group == 3:
                # Sensor collision proxy.
                name = mujoco.mj_id2name(
                    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                ) or ""
                if "cable" in name or "connector" in name:
                    simulation.model.geom_rgba[geom_id] = [1.0, 0.55, 0.02, 1.0]
                else:
                    simulation.model.geom_rgba[geom_id] = [1.0, 0.08, 0.03, 1.0]
                # MuJoCo disables higher geom groups in some viewer presets.
                # Move the proxy to always-visible group 0 for this session.
                simulation.model.geom_group[geom_id] = 0
            elif group == 2:
                # Hide the sensor CAD so its proxy cannot be occluded.
                simulation.model.geom_rgba[geom_id, 3] = 0.0
    mujoco.viewer.launch(simulation.model, simulation.data)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handeye-mujoco",
        description="Build collision-aware MuJoCo from robot URDF, sensor CAD, and T_tcp_sensor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build/rebuild the MJCF model")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--force-cad", action="store_true", help="ignore converted CAD cache")
    build.set_defaults(func=_build)

    check = subparsers.add_parser("check", help="check home pose and sensor transform")
    check.add_argument("--model", required=True, type=Path)
    check.set_defaults(func=_check)

    info = subparsers.add_parser("info", help="print compiled model summary")
    info.add_argument("--model", required=True, type=Path)
    info.set_defaults(func=_info)

    view = subparsers.add_parser("view", help="open the interactive MuJoCo viewer")
    view.add_argument("--model", required=True, type=Path)
    view.add_argument(
        "--collision-only",
        action="store_true",
        help="show robot collision meshes in blue and the sensor proxy in red",
    )
    view.set_defaults(func=_view)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (BuildError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
