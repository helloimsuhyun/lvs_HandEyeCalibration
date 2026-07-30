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
    TranslationCompositeGenerationConfig,
    generate_single_plane_circular_dataset,
    generate_three_plane_dataset,
    generate_translation_composite_dataset,
)
from laser_handeye.calibration_dataset import (
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.tan2025.simulation import PaperSimulationConfig


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
    *,
    default_handeye_preset: str,
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
    parser.add_argument(
        "--handeye-preset",
        choices=("random", "tan2025"),
        default=default_handeye_preset,
        help="ground-truth hand-eye transform source",
    )


def _add_translation_composite_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser, default_handeye_preset="tan2025")
    parser.add_argument("--translation-poses", type=_positive_int, default=36)
    parser.add_argument("--composite-poses", type=_positive_int, default=30)
    parser.add_argument("--profile-points", type=_positive_int, default=640)
    parser.add_argument("--scan-angle-deg", type=float, default=21.4)
    parser.add_argument("--sensor-z-min-mm", type=float, default=190.0)
    parser.add_argument("--sensor-z-max-mm", type=float, default=290.0)
    parser.add_argument("--nominal-sensor-distance-mm", type=float, default=240.0)
    parser.add_argument(
        "--plane-mode",
        choices=("fixed", "random"),
        default="fixed",
        help=(
            "fixed uses --plane-center-base-mm/--plane-normal-base; random "
            "resamples one plane per trial and shares it across both groups"
        ),
    )
    parser.add_argument(
        "--plane-center-base-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(500.0, 0.0, 300.0),
        help="base-frame point on the plane used by --plane-mode fixed",
    )
    parser.add_argument(
        "--plane-normal-base",
        type=float,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(0.337918, 0.1050427, -0.935296),
        help="plane normal used by --plane-mode fixed",
    )
    parser.add_argument(
        "--plane-angle-range-deg",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-30.0, 30.0),
        help="Euler XYZ sampling interval used by --plane-mode random",
    )
    parser.add_argument(
        "--plane-min-axis-angle-deg",
        type=float,
        default=1.0,
        help="minimum acute angle from every signed base axis in random mode",
    )
    parser.add_argument(
        "--plane-distance-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(350.0, 600.0),
        help="positive plane offset interval along the sampled normal",
    )
    parser.add_argument(
        "--plane-tangent-range-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-100.0, 100.0),
        help="two plane-tangent center-coordinate interval used in random mode",
    )
    parser.add_argument("--translation-tangent-span-mm", type=float, default=55.0)
    parser.add_argument("--translation-normal-span-mm", type=float, default=22.0)
    parser.add_argument("--composite-target-span-mm", type=float, default=55.0)
    parser.add_argument("--composite-tilt-span-deg", type=float, default=30.0)
    parser.add_argument("--composite-roll-span-deg", type=float, default=170.0)
    parser.add_argument("--composite-distance-span-mm", type=float, default=12.0)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.8)


def _add_single_plane_circular_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser, default_handeye_preset="random")
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
    _add_common_options(parser, default_handeye_preset="random")
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

    translation = subparsers.add_parser(
        "translation-composite",
        help="separate pure-translation and composite motion groups",
    )
    _add_translation_composite_options(translation)

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


def _translation_composite_config(
    args: argparse.Namespace,
) -> TranslationCompositeGenerationConfig:
    simulation = PaperSimulationConfig(
        num_translation_poses=args.translation_poses,
        num_composite_poses=args.composite_poses,
        num_profile_points=args.profile_points,
        scan_angle_deg=args.scan_angle_deg,
        sensor_z_min_mm=args.sensor_z_min_mm,
        sensor_z_max_mm=args.sensor_z_max_mm,
        nominal_sensor_distance_mm=args.nominal_sensor_distance_mm,
        plane_center_base_mm=tuple(args.plane_center_base_mm),
        plane_normal_base=np.asarray(args.plane_normal_base, dtype=float),
        translation_tangent_span_mm=args.translation_tangent_span_mm,
        translation_normal_span_mm=args.translation_normal_span_mm,
        composite_target_span_mm=args.composite_target_span_mm,
        composite_tilt_span_deg=args.composite_tilt_span_deg,
        composite_roll_span_deg=args.composite_roll_span_deg,
        composite_distance_span_mm=args.composite_distance_span_mm,
        minimum_valid_fraction=args.minimum_valid_fraction,
    )
    return TranslationCompositeGenerationConfig(
        simulation=simulation,
        handeye_preset=args.handeye_preset,
        plane_mode=args.plane_mode,
        plane_angle_range_deg=tuple(args.plane_angle_range_deg),
        plane_min_axis_angle_deg=args.plane_min_axis_angle_deg,
        plane_distance_range_mm=tuple(args.plane_distance_range_mm),
        plane_tangent_range_mm=tuple(args.plane_tangent_range_mm),
    )


def _single_plane_circular_config(
    args: argparse.Namespace,
) -> SinglePlaneCircularGenerationConfig:
    return SinglePlaneCircularGenerationConfig(
        handeye_preset=args.handeye_preset,
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
        handeye_preset=args.handeye_preset,
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
    if args.mode == "translation-composite":
        return _translation_composite_config(args), generate_translation_composite_dataset
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
    if isinstance(config, TranslationCompositeGenerationConfig):
        # The generator replaces this dataclass default independently in each
        # trial from handeye_preset, so persisting it as effective GT would be
        # misleading.  Every trial manifest stores its actual truth transform.
        simulation = payload.get("simulation", {})
        if isinstance(simulation, dict):
            simulation.pop("T_ef_s_true", None)
            simulation["T_ef_s_true_source"] = "handeye_preset_per_trial"
            if config.plane_mode == "random":
                simulation.pop("plane_center_base_mm", None)
                simulation.pop("plane_normal_base", None)
                simulation["plane_source"] = "random_plane_config_per_trial"
            else:
                simulation["plane_source"] = "fixed_simulation_config"
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
        "translation-composite": "translation_composite",
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
