from __future__ import annotations

import argparse

from .calibration_ui import CalibrationDashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Real laser hand-eye calibration UI")
    parser.add_argument(
        "--mock-demo",
        action="store_true",
        help="launch with the bundled in-memory robot/laser scene",
    )
    parser.add_argument("--config", default="real_laser_handeye/real_config.json")
    parser.add_argument("--handeye", default="initial_T_tcp_sensor.csv")
    parser.add_argument("--plane-boundary", default="runs/bootstrap/plane_boundary.json")
    parser.add_argument("--plan", default="runs/motion_plan.json")
    parser.add_argument("--dataset-dir", default="runs/dataset")
    parser.add_argument("--output", default="runs/T_tcp_sensor_calibrated.csv")
    args = parser.parse_args()
    if args.mock_demo:
        args.config = "real_laser_handeye/mock_demo/config.json"
        args.handeye = "real_laser_handeye/mock_demo/initial_T_tcp_sensor.csv"
        args.plane_boundary = "real_laser_handeye/mock_demo/plane_boundary.json"
        args.plan = "runs/mock_ui/motion_plan.json"
        args.dataset_dir = "runs/mock_ui/dataset"
        args.output = "runs/mock_ui/T_tcp_sensor_calibrated.csv"
    app = CalibrationDashboard(
        config_path=args.config,
        handeye_path=args.handeye,
        plane_boundary_path=args.plane_boundary,
        plan_path=args.plan,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
