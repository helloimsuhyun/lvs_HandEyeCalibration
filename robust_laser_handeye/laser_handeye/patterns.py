from __future__ import annotations
import numpy as np

# 2D plane cordinate에서 원형 패턴을 생성
# return plane coordinate line
def circular_lines(radius: float, n_lines: int = 9) -> list[tuple[np.ndarray, np.ndarray]]:

    lines = []
    for i in range(n_lines):
        a = 2.0 * np.pi * i / n_lines
        p0 = np.array([0.0, 0.0])
        p1 = radius * np.array([np.cos(a), np.sin(a)])
        lines.append((p0, p1))
    return lines


def scan_parameter_grid(
    heights_mm=None,
    projection_deg=None,
    tilt_deg=None,
) -> list[dict]:
    """scan-parameter grid.
        d 1 step 10mm
        theta 1 step 5 deg
        beta 1 step 10 deg
    """
    if heights_mm is None:
        heights_mm = np.arange(60.0, 120.0 + 1e-9, 10.0)
    if projection_deg is None:
        projection_deg = np.arange(0.0, 40.0 + 1e-9, 5.0)
    if tilt_deg is None:
        tilt_deg = np.arange(60.0, 120.0 + 1e-9, 10.0)

    return [
        {"d_mm": float(d), "theta_deg": float(th), "beta_deg": float(be)}
        for d in heights_mm for th in projection_deg for be in tilt_deg
    ]


def reduced_scan_parameter_grid() -> list[dict]:
    """Reduced grid used in the later data-reduction study."""
    return scan_parameter_grid(
        heights_mm=(60.0, 90.0, 120.0),
        projection_deg=(30.0,),
        tilt_deg=(60.0, 90.0, 120.0),
    )
