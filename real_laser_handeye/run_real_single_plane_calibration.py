from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from .planning import (
    build_single_plane_plan,
    load_json,
    load_transform,
    save_json,
    save_plan_csv,
    validate_plan_identity,
    validate_plan_runtime_safety,
)
from .safety import LIVE_ACKNOWLEDGEMENT
from .workflow import (
    calibrate_dataset,
    capture_bootstrap_plane,
    collect_plan,
    load_runtime_config,
    make_laser,
    make_robot,
    validate_bootstrap_boundary_quality,
    validate_plan_bootstrap_quality,
)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plane-boundary", type=Path, required=True)
    parser.add_argument("--handeye", type=Path, required=True, help="initial T_tcp_sensor CSV")
    parser.add_argument("--heights-mm", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--theta-deg", type=float, nargs="+", default=[30])
    parser.add_argument("--beta-deg", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--reference-scans", type=int, default=24)
    parser.add_argument("--reference-theta-deg", type=float, nargs="+", default=[60])
    parser.add_argument("--reference-heights-mm", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--reference-beta-deg", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--pattern-radius-mm", type=float, default=None)
    parser.add_argument("--pattern-radius-scale", type=float, default=0.8)
    parser.add_argument("--pattern-center-uv-mm", type=float, nargs=2, default=None)
    parser.add_argument(
        "--pose-geometry",
        choices=["paper_incidence", "observable_dihedral"],
        default="paper_incidence",
    )
    parser.add_argument(
        "--allow-unobservable",
        action="store_true",
        help="diagnostic plan only; calibration still rejects rank-deficient data",
    )


def _make_plan(args: argparse.Namespace, output_path: Path) -> dict:
    _config, safety, capture = load_runtime_config(args.config)
    boundary = load_json(args.plane_boundary)
    handeye = load_transform(args.handeye)
    validate_bootstrap_boundary_quality(boundary, capture, handeye)
    plan = build_single_plane_plan(
        plane_boundary=boundary,
        T_tcp_sensor_init=handeye,
        safety=safety,
        heights_mm=args.heights_mm,
        theta_deg=args.theta_deg,
        beta_deg=args.beta_deg,
        reference_scan_count=args.reference_scans,
        reference_theta_deg=args.reference_theta_deg,
        reference_heights_mm=args.reference_heights_mm,
        reference_beta_deg=args.reference_beta_deg,
        pattern_radius_mm=args.pattern_radius_mm,
        pattern_radius_scale=args.pattern_radius_scale,
        pattern_center_uv_mm=(
            None
            if args.pattern_center_uv_mm is None
            else np.asarray(args.pattern_center_uv_mm, dtype=float)
        ),
        pose_geometry=args.pose_geometry,
        allow_unobservable=args.allow_unobservable,
    )
    save_json(output_path, plan)
    csv_path = output_path.with_suffix(".csv")
    save_plan_csv(csv_path, plan)
    print(f"saved plan: {output_path}")
    print(f"saved review CSV (extrinsic XYZ Euler): {csv_path}")
    print(
        f"poses: main={plan['main_scan_count']}, "
        f"reference={plan['reference_scan_count']}, total={len(plan['entries'])}"
    )
    report = plan["observability"]
    print(
        "translation/offset observability: "
        f"rank={report['rank']}/{report['required_rank']}, "
        f"condition={report['column_normalized_condition']:.6g}"
    )
    return plan


def _connect_hardware(config: dict):
    robot = make_robot(config)
    laser = make_laser(config, robot=robot)
    try:
        robot.connect()
        laser.connect()
    except BaseException:
        try:
            laser.close()
        finally:
            robot.close()
        raise
    return robot, laser


def _close_hardware(robot, laser) -> None:
    try:
        laser.close()
    finally:
        robot.close()


def _execute_collection(args: argparse.Namespace, plan: dict, dataset_dir: Path) -> dict:
    config, safety, capture = load_runtime_config(args.config)
    validate_plan_identity(plan)
    validate_plan_runtime_safety(plan, safety)
    validate_plan_bootstrap_quality(plan, capture)
    if not args.execute:
        print("DRY RUN: no robot or laser connection was opened; no motion was sent.")
        print(
            "After reviewing the plan and controller simulation, enable live_enabled "
            "and run with --execute --acknowledge-risk " + LIVE_ACKNOWLEDGEMENT
        )
        return {}
    safety.assert_live_unlocked(args.acknowledge_risk)
    robot, laser = _connect_hardware(config)
    try:
        return collect_plan(
            robot=robot,
            laser=laser,
            plan=plan,
            output_dir=dataset_dir,
            safety=safety,
            capture=capture,
        )
    finally:
        _close_hardware(robot, laser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe real-robot single-plane laser hand-eye workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="capture four manually jogged profiles and estimate plane bounds"
    )
    bootstrap.add_argument("--config", type=Path, required=True)
    bootstrap.add_argument("--handeye", type=Path, required=True)
    bootstrap.add_argument("--output-dir", type=Path, required=True)
    bootstrap.add_argument("--margin-mm", type=float, default=20.0)

    plan = subparsers.add_parser("plan", help="generate and statically validate the 105-pose plan")
    _add_plan_arguments(plan)
    plan.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="dry-run or execute an existing plan")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--dataset-dir", type=Path, required=True)
    collect.add_argument("--execute", action="store_true")
    collect.add_argument("--acknowledge-risk", default=None)

    calibrate = subparsers.add_parser("calibrate", help="run joint linear-offset calibration")
    calibrate.add_argument("--dataset-dir", type=Path, required=True)
    calibrate.add_argument("--handeye", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--max-iter", type=int, default=30)
    calibrate.add_argument("--tol", type=float, default=1e-9)
    calibrate.add_argument("--max-translation-offset-condition", type=float, default=1e6)
    calibrate.add_argument(
        "--linear-multistart", action=argparse.BooleanOptionalAction, default=True
    )
    calibrate.add_argument("--linear-multistart-threshold-mm", type=float, default=1.0)
    calibrate.add_argument("--linear-multistart-angle-deg", type=float, default=30.0)
    calibrate.add_argument("--max-final-plane-rms-mm", type=float, default=2.0)
    calibrate.add_argument("--allow-partial", action="store_true")

    run = subparsers.add_parser(
        "run", help="plan, optionally collect, and calibrate in one command"
    )
    _add_plan_arguments(run)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--acknowledge-risk", default=None)
    run.add_argument("--max-iter", type=int, default=30)
    run.add_argument("--tol", type=float, default=1e-9)
    run.add_argument(
        "--linear-multistart", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--linear-multistart-threshold-mm", type=float, default=1.0)
    run.add_argument("--linear-multistart-angle-deg", type=float, default=30.0)
    run.add_argument("--max-final-plane-rms-mm", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        config, safety, capture = load_runtime_config(args.config)
        robot, laser = _connect_hardware(config)
        try:
            boundary = capture_bootstrap_plane(
                robot=robot,
                laser=laser,
                T_tcp_sensor_init=load_transform(args.handeye),
                output_dir=args.output_dir,
                safety=safety,
                capture=capture,
                margin_mm=args.margin_mm,
            )
        finally:
            _close_hardware(robot, laser)
        print(f"saved plane boundary: {args.output_dir / 'plane_boundary.json'}")
        print(f"bootstrap plane RMS: {boundary['plane']['rms_error_mm']:.4f} mm")
        return
    if args.command == "plan":
        _make_plan(args, args.output)
        return
    if args.command == "collect":
        _execute_collection(args, load_json(args.plan), args.dataset_dir)
        return
    if args.command == "calibrate":
        result = calibrate_dataset(
            dataset_dir=args.dataset_dir,
            T_tcp_sensor_init=load_transform(args.handeye),
            output_transform=args.output,
            max_iter=args.max_iter,
            tol=args.tol,
            max_translation_offset_condition=args.max_translation_offset_condition,
            linear_multistart=args.linear_multistart,
            linear_multistart_threshold_mm=args.linear_multistart_threshold_mm,
            linear_multistart_angle_deg=args.linear_multistart_angle_deg,
            max_final_plane_rms_mm=args.max_final_plane_rms_mm,
            allow_partial=args.allow_partial,
        )
        print(f"saved calibration: {args.output}")
        print(
            f"converged={result['converged']} iterations={result['iterations']} "
            f"final plane RMS={result['final_plane_rms_mm']:.4f} mm"
        )
        return
    if args.command == "run":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.work_dir / "motion_plan.json"
        plan = _make_plan(args, plan_path)
        _execute_collection(args, plan, args.work_dir / "dataset")
        if not args.execute:
            return
        output = args.work_dir / "T_tcp_sensor_calibrated.csv"
        max_final_plane_rms_mm = args.max_final_plane_rms_mm
        if max_final_plane_rms_mm is None:
            _config, _safety, capture = load_runtime_config(args.config)
            max_final_plane_rms_mm = capture.max_final_plane_rms_mm
        result = calibrate_dataset(
            dataset_dir=args.work_dir / "dataset",
            T_tcp_sensor_init=load_transform(args.handeye),
            output_transform=output,
            max_iter=args.max_iter,
            tol=args.tol,
            linear_multistart=args.linear_multistart,
            linear_multistart_threshold_mm=args.linear_multistart_threshold_mm,
            linear_multistart_angle_deg=args.linear_multistart_angle_deg,
            max_final_plane_rms_mm=max_final_plane_rms_mm,
        )
        print(f"saved calibration: {output}")
        print(
            f"converged={result['converged']} iterations={result['iterations']} "
            f"final plane RMS={result['final_plane_rms_mm']:.4f} mm"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("aborted by operator", file=sys.stderr)
        raise SystemExit(130)
