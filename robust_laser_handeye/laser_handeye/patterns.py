from __future__ import annotations
import numpy as np


def circular_lines(radius: float, n_lines: int = 9) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return line endpoints in a local 2D plane coordinate system.

    Each line starts at the circle center and ends at equally spaced points on
    the circumference. The paper uses 9 lines, i.e. 40 degrees apart.
    """
    lines = []
    for i in range(n_lines):
        a = 2.0 * np.pi * i / n_lines
        p0 = np.array([0.0, 0.0])
        p1 = radius * np.array([np.cos(a), np.sin(a)])
        lines.append((p0, p1))
    return lines


def scan_parameter_grid(
    heights_mm=(60.0, 90.0, 120.0),
    projection_deg=(30.0,),
    tilt_deg=(60.0, 90.0, 120.0),
) -> list[dict]:
    """Default optimized scan-parameter grid from the paper's simulation study."""
    return [
        {'d_mm': float(d), 'theta_deg': float(th), 'beta_deg': float(be)}
        for d in heights_mm for th in projection_deg for be in tilt_deg
    ]
