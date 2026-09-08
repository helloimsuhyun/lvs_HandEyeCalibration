#!/usr/bin/env python3
"""Generate reusable, noise-free laser hand-eye calibration datasets.

This command intentionally stops at ideal acquisition generation.  Measurement
noise, pose-readback noise, calibration, and Monte Carlo analysis belong to a
separate experiment stage so every method can consume the same raw profiles.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from laser_handeye.calibration_dataset.generators import (
    GenerationSeeds,
    SinglePlaneCircularGenerationConfig,
    ThreePlaneGenerationConfig,
    generate_single_plane_circular_dataset,
    generate_three_plane_dataset,
)
from laser_handeye.calibration_dataset import (
    logical_dataset_sha256,
    save_calibration_dataset,
)


COLLECTION_SCHEMA = "laser_handeye.calibration_dataset_collection"
COLLECTION_SCHEMA_VERSION = 1


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _add_common_options(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=1,
        help="number of independently seeded ideal acquisitions to generate",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=7,
        help="non-negative master seed used to derive per-trial random streams",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty collection directory (existing files are never overwritten)",
    )


def _add_single_plane_circular_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser)
    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--heights-mm",
        type=float,
        nargs="+",
        default=(60.0, 90.0, 120.0),
    )
    parser.add_argument("--theta-deg", type=float, nargs="+", default=(30.0,))
    parser.add_argument(
        "--beta-deg",
        type=float,
        nargs="+",
        default=(60.0, 90.0, 120.0),
    )
    parser.add_argument(
        "--pose-geometry",
        choices=("paper_incidence", "observable_dihedral"),
        default="paper_incidence",
    )
    parser.add_argument(
        "--projection-branch-mode",
        choices=("alternating", "positive"),
        default="alternating",
    )
    parser.add_argument(
        "--reference-line-ids",
        type=int,
        nargs="*",
        default=(1, 2, 5, 6),
        metavar="LINE_ID",
        help="line IDs 0..8; pass the option with no values to disable the reference ring",
    )
    parser.add_argument(
        "--reference-heights-mm",
        type=float,
        nargs="+",
        default=(60.0, 90.0, 120.0),
    )
    parser.add_argument("--reference-theta-deg", type=float, default=60.0)
    parser.add_argument(
        "--reference-beta-deg",
        type=float,
        nargs="+",
        default=(60.0, 90.0, 120.0),
    )
    parser.add_argument(
        "--plane-angle-range-deg",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-5.0, 5.0),
    )
    parser.add_argument("--plane-min-abs-angle-deg", type=float, default=1.0)
    parser.add_argument(
        "--plane-xy-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-100.0, 100.0),
    )
    parser.add_argument(
        "--plane-z-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(400.0, 550.0),
    )
    parser.add_argument(
        "--check-reachability",
        action="store_true",
        help="enable the simulator's simple workspace-box filter (not robot IK)",
    )


def _add_three_plane_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser)
    parser.add_argument("--poses-per-plane", type=_positive_int, default=35)
    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument(
        "--plane-distance-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(650.0, 1000.0),
    )
    parser.add_argument("--plane-min-axis-angle-deg", type=float, default=1.0)
    parser.add_argument("--tangent-range-mm", type=float, default=220.0)
    parser.add_argument(
        "--profile-depth-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(60.0, 150.0),
    )
    parser.add_argument("--min-view-dot", type=float, default=0.0)
    parser.add_argument(
        "--max-trials-per-plane",
        type=_positive_int,
        default=50_000,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate immutable ideal/raw calibration acquisitions. Noise and "
            "calibration are deliberately deferred to a separate experiment CLI."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    circular = subparsers.add_parser(
        "single-plane-circular",
        help="single-plane nine-line circular optimal-pattern acquisition",
    )
    _add_single_plane_circular_options(circular)

    three_plane = subparsers.add_parser(
        "three-plane",
        help="random 6-DoF profiles on three mutually orthogonal planes",
    )
    _add_three_plane_options(three_plane)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _single_plane_circular_config(
    args: argparse.Namespace,
) -> SinglePlaneCircularGenerationConfig:
    return SinglePlaneCircularGenerationConfig(
        profile_points=args.profile_points,
        profile_half_width_mm=args.profile_half_width_mm,
        radius_mm=args.radius_mm,
        heights_mm=tuple(args.heights_mm),
        theta_deg=tuple(args.theta_deg),
        beta_deg=tuple(args.beta_deg),
        pose_geometry=args.pose_geometry,
        projection_branch_mode=args.projection_branch_mode,
        reference_line_ids=tuple(args.reference_line_ids),
        reference_heights_mm=tuple(args.reference_heights_mm),
        reference_theta_deg=args.reference_theta_deg,
        reference_beta_deg=tuple(args.reference_beta_deg),
        plane_angle_range_deg=tuple(args.plane_angle_range_deg),
        plane_min_abs_angle_deg=args.plane_min_abs_angle_deg,
        plane_xy_range_mm=tuple(args.plane_xy_range_mm),
        plane_z_range_mm=tuple(args.plane_z_range_mm),
        check_reachability=args.check_reachability,
    )


def _three_plane_config(args: argparse.Namespace) -> ThreePlaneGenerationConfig:
    return ThreePlaneGenerationConfig(
        poses_per_plane=args.poses_per_plane,
        profile_points=args.profile_points,
        profile_half_width_mm=args.profile_half_width_mm,
        plane_distance_range_mm=tuple(args.plane_distance_range_mm),
        plane_min_axis_angle_deg=args.plane_min_axis_angle_deg,
        tangent_range_mm=args.tangent_range_mm,
        profile_depth_range_mm=tuple(args.profile_depth_range_mm),
        min_view_dot=args.min_view_dot,
        max_trials_per_plane=args.max_trials_per_plane,
    )


def _mode_components(
    args: argparse.Namespace,
) -> tuple[Any, Callable[[Any, GenerationSeeds], Any]]:
    if args.mode == "single-plane-circular":
        return _single_plane_circular_config(args), generate_single_plane_circular_dataset
    if args.mode == "three-plane":
        return _three_plane_config(args), generate_three_plane_dataset
    raise ValueError(f"unsupported generation mode: {args.mode}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _collection_config(config: Any) -> dict[str, Any]:
    payload = _jsonable(asdict(config))
    return payload


def _prepare_output_directory(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(
                f"output path exists and is not a directory: {output_dir}"
            )
        if next(output_dir.iterdir(), None) is not None:
            raise FileExistsError(
                f"output directory is not empty; refusing to overwrite: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "trials").mkdir(exist_ok=False)
    return output_dir


def _write_collection(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, generator = _mode_components(args)
    output_dir = _prepare_output_directory(args.output_dir)

    acquisition_modes = {
        "single-plane-circular": "single_plane_circular",
        "three-plane": "three_plane_random",
    }
    trial_entries: list[dict[str, Any]] = []
    collection = {
        "schema": COLLECTION_SCHEMA,
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "command_mode": args.mode,
        "acquisition_mode": acquisition_modes[args.mode],
        "profile_state": "ideal",
        "requested_trials": args.trials,
        "completed_trials": 0,
        "config": _collection_config(config),
        "trials": trial_entries,
    }
    collection_path = output_dir / "collection.json"
    _write_collection(collection_path, collection)

    for trial_index in range(args.trials):
        seeds = GenerationSeeds.derive(args.seed, trial_index)
        dataset = generator(config, seeds)
        relative_path = Path("trials") / f"trial_{trial_index:06d}"
        trial_path = output_dir / relative_path
        save_calibration_dataset(dataset, trial_path)
        trial_entries.append(
            {
                "trial_index": trial_index,
                "relative_path": relative_path.as_posix(),
                "logical_dataset_sha256": logical_dataset_sha256(dataset),
                "generation_seeds": asdict(seeds),
            }
        )
        collection["completed_trials"] = len(trial_entries)
        _write_collection(collection_path, collection)
        print(
            f"[{args.mode}] saved trial {trial_index + 1}/{args.trials}: "
            f"{relative_path.as_posix()}"
        )

    collection["status"] = "complete"
    _write_collection(collection_path, collection)
    print(f"saved collection manifest: {collection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
