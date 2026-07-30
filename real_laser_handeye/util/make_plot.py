from __future__ import annotations

from pathlib import Path
from typing import Sequence
import uuid

import numpy as np


def save_plane_rms_plot(
    plane_rms_history: Sequence[float] | np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Plane RMS convergence",
) -> Path:
    """Save iteration-wise plane RMS values as a PNG image.

    Iteration 0 represents the first RMS value reported by the calibration
    solver. Non-finite values are shown as gaps so that a rejected calibration
    can still leave a diagnostic plot.
    """
    values = np.asarray(plane_rms_history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("plane_rms_history must be a non-empty 1-D sequence")

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("plane RMS plot output must use a .png extension")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use the non-GUI Agg canvas so calibration plotting does not interfere
    # with the PyQtGraph live monitor.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9.0, 5.0), dpi=140, constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)

    iterations = np.arange(values.size, dtype=int)
    plot_values = values.copy()
    plot_values[~np.isfinite(plot_values)] = np.nan
    marker = "o" if values.size <= 100 else None
    axis.plot(
        iterations,
        plot_values,
        color="#0072B2",
        linewidth=1.8,
        marker=marker,
        markersize=3.5,
    )

    if np.isfinite(values[0]):
        axis.scatter([0], [values[0]], color="#E69F00", s=36, zorder=3)
    if np.isfinite(values[-1]):
        axis.scatter(
            [values.size - 1],
            [values[-1]],
            color="#D55E00",
            s=36,
            zorder=3,
        )
        axis.annotate(
            f"final = {values[-1]:.6g} mm",
            xy=(values.size - 1, values[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )

    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Plane RMS [mm]")
    axis.set_xlim(left=0)
    axis.grid(True, alpha=0.3)

    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.tmp"
    )
    try:
        figure.savefig(temporary, format="png")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
        figure.clear()
    return output
