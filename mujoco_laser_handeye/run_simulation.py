from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .compat.planning import (
    build_single_plane_plan,
    load_json,
    load_transform,
    save_json,
    save_plan_csv,
)
from .compat.safety import LIVE_ACKNOWLEDGEMENT
from .compat.workflow import (
    calibrate_dataset,
    collect_plan,
    load_runtime_config,
    make_laser,
    make_robot,
    validate_bootstrap_boundary_quality,
)

from .adapters import MujocoKeyenceLaser, MujocoRB5Robot
from .bootstrap import capture_automatic_mujoco_bootstrap
from .validation import filter_collision_free_plan, preflight_motion_plan


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "rb5_keyence_single_plane.json"
DEFAULT_HANDEYE = PACKAGE_DIR / "config" / "initial_T_tcp_sensor.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RB5-850E + Keyence single-plane hand-eye calibration in MuJoCo"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--plane-boundary",
        type=Path,
        default=None,
        help=(
            "use an existing boundary instead of capturing the default automatic "
            "four-view MuJoCo bootstrap"
        ),
    )
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument(
        "--true-handeye",
        type=Path,
        default=None,
        help="optional GT override; by default use robot.kwargs.T_tcp_sensor",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("runs/mujoco_rb5_auto_bootstrap")
    )
    parser.add_argument("--heights-mm", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--theta-deg", type=float, nargs="+", default=[30])
    parser.add_argument("--beta-deg", type=float, nargs="+", default=[60, 90, 120])
    parser.add_argument("--reference-scans", type=int, default=24)
    parser.add_argument("--reference-theta-deg", type=float, nargs="+", default=[60])
    parser.add_argument(
        "--reference-heights-mm", type=float, nargs="+", default=[60, 90, 120]
    )
    parser.add_argument(
        "--reference-beta-deg", type=float, nargs="+", default=[60, 90, 120]
    )
    parser.add_argument("--pattern-radius-scale", type=float, default=0.8)
    parser.add_argument("--bootstrap-margin-mm", type=float, default=20.0)
    parser.add_argument(
        "--pose-geometry",
        choices=["paper_incidence", "observable_dihedral"],
        default="paper_incidence",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="open the interactive MuJoCo viewer"
    )
    parser.add_argument(
        "--realtime-scale",
        type=float,
        default=None,
        help="motion playback scale; 0 is as fast as possible",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="capture/finalize the four bootstrap views and stop",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="write plans/reports but do not capture or calibrate",
    )
    parser.add_argument(
        "--no-filter-unsafe",
        action="store_true",
        help="diagnostic mode: preflight all 105 theoretical poses without filtering",
    )
    parser.add_argument("--min-safe-scans", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-final-plane-rms-mm", type=float, default=None)
    return parser.parse_args()


def _progress_filter(index: int, total: int, passed: bool) -> None:
    if index == 1 or index == total or index % 10 == 0:
        print(
            f"[MuJoCo filter {index}/{total}] last={'PASS' if passed else 'REJECT'}"
        )


def _progress_bootstrap(index: int, total: int) -> None:
    print(f"[MuJoCo bootstrap {index}/{total}] captured")


def _progress_preflight(index: int, total: int) -> None:
    if index == 1 or index == total or index % 10 == 0:
        print(f"[MuJoCo preflight {index}/{total}] PASS")


def _handeye_errors(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    rotation = float(
        np.degrees(
            Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude()
        )
    )
    return translation, rotation


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config, safety, capture = load_runtime_config(args.config)
    config.setdefault("robot", {}).setdefault("kwargs", {})["viewer"] = args.viewer
    if args.realtime_scale is not None:
        config["robot"]["kwargs"]["realtime_scale"] = args.realtime_scale
    handeye = load_transform(args.handeye)
    safety.assert_live_unlocked(LIVE_ACKNOWLEDGEMENT)

    robot = make_robot(config)
    if not isinstance(robot, MujocoRB5Robot):
        raise TypeError("configured robot is not MujocoRB5Robot")
    laser = make_laser(config, robot=robot)
    if not isinstance(laser, MujocoKeyenceLaser):
        raise TypeError("configured laser is not MujocoKeyenceLaser")
    robot.connect()
    laser.connect()
    try:
        if args.plane_boundary is None:
            boundary_path = args.work_dir / "bootstrap" / "plane_boundary.json"
            boundary, bootstrap_report = capture_automatic_mujoco_bootstrap(
                robot=robot,
                laser=laser,
                T_tcp_sensor_init=handeye,
                output_dir=boundary_path.parent,
                safety=safety,
                capture=capture,
                margin_mm=args.bootstrap_margin_mm,
                on_progress=_progress_bootstrap,
            )
            diagnostics = bootstrap_report["simulation_gt_diagnostics"]
            print(
                "bootstrap: "
                f"RMS={bootstrap_report['plane_rms_mm']:.4f} mm, "
                f"normal-GT={diagnostics['normal_error_deg']:.4f} deg, "
                "estimated boundary=" + str(boundary_path)
            )
        else:
            boundary_path = args.plane_boundary
            boundary = load_json(boundary_path)
            print(f"bootstrap: using existing boundary {boundary_path}")
        validate_bootstrap_boundary_quality(boundary, capture, handeye)
        if args.bootstrap_only:
            return

        theoretical_plan = build_single_plane_plan(
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
            pattern_radius_scale=args.pattern_radius_scale,
            pose_geometry=args.pose_geometry,
        )
        theoretical_path = args.work_dir / "motion_plan_theoretical.json"
        save_json(theoretical_path, theoretical_plan)
        save_plan_csv(theoretical_path.with_suffix(".csv"), theoretical_plan)
        print(
            f"theoretical plan: {len(theoretical_plan['entries'])} scans, "
            f"rank={theoretical_plan['observability']['rank']}/4"
        )

        if args.no_filter_unsafe:
            plan = theoretical_plan
        else:
            plan, filter_report = filter_collision_free_plan(
                robot,
                theoretical_plan,
                safety,
                min_safe_scans=args.min_safe_scans,
                report_path=args.work_dir / "collision_filter_report.json",
                on_progress=_progress_filter,
            )
            print(
                "collision filter: "
                f"accepted={filter_report['accepted_scan_count']}, "
                f"rejected={filter_report['rejected_scan_count']}, "
                f"rank={plan['observability']['rank']}/4, "
                f"condition={plan['observability']['column_normalized_condition']:.4g}"
            )
        plan_path = args.work_dir / "motion_plan_filtered.json"
        save_json(plan_path, plan)
        save_plan_csv(plan_path.with_suffix(".csv"), plan)
        preflight = preflight_motion_plan(
            robot,
            plan,
            safety,
            report_path=args.work_dir / "preflight_report.json",
            on_progress=_progress_preflight,
        )
        print(
            f"preflight: {preflight['status']} "
            f"({preflight['checked_scans']}/{preflight['total_scans']} scans)"
        )
        if args.preflight_only:
            return

        started = time.time()
        manifest = collect_plan(
            robot=robot,
            laser=laser,
            plan=plan,
            output_dir=args.work_dir / "dataset",
            safety=safety,
            capture=capture,
        )
        print(
            f"capture complete: {len(manifest['scans'])} scans in "
            f"{time.time() - started:.2f} s"
        )
    finally:
        laser.close()
        robot.close()

    output = args.work_dir / "T_tcp_sensor_calibrated.csv"
    max_rms = (
        capture.max_final_plane_rms_mm
        if args.max_final_plane_rms_mm is None
        else args.max_final_plane_rms_mm
    )
    diagnostics = calibrate_dataset(
        dataset_dir=args.work_dir / "dataset",
        T_tcp_sensor_init=handeye,
        output_transform=output,
        max_iter=args.max_iter,
        tol=args.tol,
        max_final_plane_rms_mm=max_rms,
    )
    summary = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "scan_count": len(plan["entries"]),
        "converged": bool(diagnostics["converged"]),
        "iterations": int(diagnostics["iterations"]),
        "final_plane_rms_mm": float(diagnostics["final_plane_rms_mm"]),
        "calibrated_T_tcp_sensor": np.loadtxt(output, delimiter=",").tolist(),
        "bootstrap_plane_boundary": str(boundary_path),
    }
    truth = (
        robot.T_tcp_sensor_true
        if args.true_handeye is None
        else load_transform(args.true_handeye)
    )
    estimate = np.asarray(summary["calibrated_T_tcp_sensor"], dtype=float)
    translation_error, rotation_error = _handeye_errors(estimate, truth)
    summary["ground_truth_T_tcp_sensor"] = truth.tolist()
    summary["translation_error_mm"] = translation_error
    summary["rotation_error_deg"] = rotation_error
    save_json(args.work_dir / "simulation_summary.json", summary)
    print(
        f"calibration: converged={summary['converged']}, "
        f"RMS={summary['final_plane_rms_mm']:.5f} mm"
    )
    if "translation_error_mm" in summary:
        print(
            f"ground-truth error: {summary['translation_error_mm']:.5f} mm, "
            f"{summary['rotation_error_deg']:.5f} deg"
        )
    print(f"result: {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("MuJoCo simulation aborted", file=sys.stderr)
        raise SystemExit(130)
