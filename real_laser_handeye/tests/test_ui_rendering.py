from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.figure import Figure
import numpy as np

from real_laser_handeye.hardware import ProfileSample
from real_laser_handeye.tests.test_ui_session import make_scene
from real_laser_handeye.ui_model import build_dashboard_snapshot
from real_laser_handeye.ui_plot import (
    render_calibration_axis,
    render_profile_axes,
    render_scene_axis,
)


def test_headless_rendering_does_not_load_vendor_sdk_or_tk_root(tmp_path):
    _plane, _init, _transit, _safety, _capture, plan, _robot, _laser = make_scene()
    points = np.column_stack(
        [np.linspace(-5, 5, 30), np.zeros(30), np.linspace(10, 11, 30)]
    )
    snapshot = build_dashboard_snapshot(
        plan=plan,
        dataset_dir=tmp_path,
        live_sequence=3,
        live_sample=ProfileSample(points, 1),
    )
    profile_figure = Figure(figsize=(6, 5))
    live_axis = profile_figure.add_subplot(211)
    previous_axis = profile_figure.add_subplot(212)
    render_profile_axes(live_axis, previous_axis, snapshot)
    profile_figure.canvas.draw()
    np.testing.assert_allclose(live_axis.lines[0].get_xdata(), points[:, 0])

    scene_figure = Figure(figsize=(6, 5))
    scene_axis = scene_figure.add_subplot(111, projection="3d")
    render_scene_axis(scene_axis, snapshot)
    scene_figure.canvas.draw()

    calibration_figure = Figure(figsize=(6, 4))
    calibration_axis = calibration_figure.add_subplot(111)
    render_calibration_axis(
        calibration_axis,
        {"delta_history": [1.0, 0.1], "condition_history": [10.0, 8.0]},
    )
    calibration_figure.canvas.draw()
