# Robust 2D Laser Sensor Hand-Eye Calibration Scaffold

This is a modular Python scaffold for reproducing the single-plane 2D laser hand-eye calibration workflow.

## Core modules

- `laser_handeye/se3.py`: SE(3) utilities, SO(3) projection.
- `laser_handeye/data.py`: `LaserScan` data container.
- `laser_handeye/geometry.py`: PCA plane fitting and paper-style scaled normal.
- `laser_handeye/calibration.py`: iterative single-plane least-squares calibration.
- `laser_handeye/patterns.py`: circular line pattern and scan parameter grid.
- `laser_handeye/simulation.py`: synthetic profile generator for verification.
- `laser_handeye/adapters.py`: abstract robot/laser interfaces for real hardware extension.

## Quick run

```bash
cd robust_laser_handeye
PYTHONPATH=. python examples/run_synthetic_demo.py
```

## Real hardware extension

Implement `RobotAdapter` and `LaserAdapter`, then collect `LaserScan` objects with synchronized robot flange pose and sensor profile points. Feed those scans to `calibrate_single_plane(scans, T_init)`.
