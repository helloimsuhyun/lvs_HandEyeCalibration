#!/usr/bin/env python3
"""Compatibility entry point for the canonical line-count plotter.

New code should call:

    python3 main/make_plot/plot_fixed_noise_line_count_boxplots.py

All command-line arguments are forwarded unchanged.
"""

from __future__ import annotations

import runpy
from pathlib import Path


PLOTTER = (
    Path(__file__).resolve().parent
    / "make_plot"
    / "plot_fixed_noise_line_count_boxplots.py"
)


if __name__ == "__main__":
    runpy.run_path(str(PLOTTER), run_name="__main__")
