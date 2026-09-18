"""
Temporary integration check.

Flow:
    1) Pick CAD waypoints
    2) Generate scan path / sensor trajectory
    3) Inspect the original sensor-motion playback GUI
    4) Simulate laser-profile measurements and visualize them

Example:
python main.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \

"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from surface_scan_path import (
    ScanPlannerConfig,
    SurfaceScanPlanner,
    pick_waypoint_segments_fresh_processes,
    visualize_sensor_motion_gui,
)
from laser_profile_simulator import (
    LaserProfileConfig,
    LaserProfileSimulator,
    visualize_simulation,
)


# -----------------------------------------------------------------------------
# Temporary defaults for integration checking
# -----------------------------------------------------------------------------

PATH_STEP_MM = 1.0
SENSOR_STANDOFF_MM = 80.0
ORIENTATION_SMOOTH_WINDOW = 31
AUTO_BREAK_ANGLE_DEG = 45.0

PICKER_POINT_SIZE = 2.0
PICKER_SAMPLE_POINTS = 100_000

# Original scan-path playback GUI defaults
SENSOR_AXIS_LENGTH_MM = 10.0
GUI_POSE_RATE_HZ = 12.0

# Laser-profile simulation display
PROFILE_STRIDE = 10
SIM_POINT_SIZE = 2.0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pick a CAD scan path, inspect the sensor motion, "
            "then simulate laser-profile measurements."
        )
    )
    p.add_argument("cad", type=Path, help="CAD mesh, e.g. part.stl")
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional output for the simulated scan points/metadata.",
    )
    p.add_argument(
        "--save-ply",
        type=Path,
        default=None,
        help="Optional output point cloud.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load CAD + prepare path planner
    # ------------------------------------------------------------------
    planner_config = ScanPlannerConfig(
        mesh_unit=args.mesh_unit,
        path_step_mm=PATH_STEP_MM,
        sensor_standoff_mm=SENSOR_STANDOFF_MM,
        orientation_smooth_window=ORIENTATION_SMOOTH_WINDOW,
        auto_break_angle_deg=AUTO_BREAK_ANGLE_DEG,
    )

    planner = SurfaceScanPlanner(args.cad, planner_config)

    print("\n[1/4] PICK SCAN PATH")
    print("  Pick >= 2 waypoints for each scan segment.")
    print("  Finish an empty new segment to end waypoint input.")

    picker_args = SimpleNamespace(
        cad=args.cad,
        mesh_unit=planner_config.mesh_unit,
        weld_tolerance_mm=planner_config.weld_tolerance_mm,
        picker_point_size=PICKER_POINT_SIZE,
        picker_sample_points=PICKER_SAMPLE_POINTS,
    )

    picked_segments = pick_waypoint_segments_fresh_processes(
        planner.mesh,
        picker_args,
    )

    waypoint_segments = [seg["waypoint_points"] for seg in picked_segments]

    # ------------------------------------------------------------------
    # 2. Generate sensor trajectory from the picked CAD waypoints
    # ------------------------------------------------------------------
    print("\n[2/4] GENERATE SCAN PATH")

    plan = planner.plan_segments(waypoint_segments)
    plan.validate(print_report=True)

    # ------------------------------------------------------------------
    # 3. Original scan-path sensor-motion playback GUI
    # ------------------------------------------------------------------
    print("\n[3/4] INSPECT SENSOR MOTION")
    print("  Close the playback window when inspection is finished.")
    print("  Laser-profile simulation starts after the window closes.")

    flat = plan.flat

    visualize_sensor_motion_gui(
        mesh=planner.mesh,
        waypoint_points=flat["waypoint_points"],
        surface_points=flat["surface_points"],
        sensor_points=flat["sensor_points"],
        frames=flat["frames"],
        arc_length_m=flat["arc_length_m"],
        axis_length_mm=SENSOR_AXIS_LENGTH_MM,
        initial_pose_rate_hz=GUI_POSE_RATE_HZ,
        segment_ids=flat["segment_ids"],
        waypoint_segment_ids=flat["waypoint_segment_ids"],
        waypoint_local_ids=flat["waypoint_local_ids"],
    )

    # ------------------------------------------------------------------
    # 4. Simulate laser profiles along that trajectory
    # ------------------------------------------------------------------
    print("\n[4/4] SIMULATE LASER PROFILE")

    sensor_config = LaserProfileConfig.lj_v7080()

    simulator = LaserProfileSimulator(
        planner.mesh,
        sensor_config,
    )

    result = simulator.simulate_plan(plan)
    result.print_summary()

    if args.save_npz is not None:
        result.save_npz(args.save_npz)
        print(f"Saved NPZ : {args.save_npz}")

    if args.save_ply is not None:
        result.save_ply(args.save_ply)
        print(f"Saved PLY : {args.save_ply}")

    visualize_simulation(
        planner.mesh,
        plan.segments,
        result,
        profile_stride=PROFILE_STRIDE,
        point_size=SIM_POINT_SIZE,
    )


if __name__ == "__main__":
    main()