from __future__ import annotations

import numpy as np


def keyence_raw_to_points(
    *,
    raw_z: np.ndarray,
    x_count: int,
    profile_count: int,
    x_start_raw: int,
    x_pitch_raw: int,
    z_unit_raw: int,
    aggregate: str = "median",
) -> np.ndarray:
    """Decode LJ-X simple-array profiles to [x, 0, z] millimetres.

    Keyence X values use 0.01 micrometre units.  Height samples are unsigned
    values offset by 32768 and scaled by the reported Z unit (also 0.01 um).
    Invalid raw zero samples are excluded.  A multi-line batch is reduced with
    a per-X median by default.
    """
    x_count = int(x_count)
    profile_count = int(profile_count)
    raw = np.asarray(raw_z, dtype=np.int32).reshape(-1)
    if x_count <= 0 or profile_count <= 0 or len(raw) < x_count * profile_count:
        return np.empty((0, 3), dtype=float)
    raw = raw[: x_count * profile_count].reshape(profile_count, x_count)
    decoded = (raw.astype(float) - 32768.0) * (float(z_unit_raw) / 100.0) / 1000.0
    decoded[raw == 0] = np.nan
    if aggregate == "latest":
        z_mm = decoded[-1]
    elif aggregate == "median":
        z_mm = np.full(x_count, np.nan, dtype=float)
        columns_with_data = np.any(raw != 0, axis=0)
        z_mm[columns_with_data] = np.nanmedian(
            decoded[:, columns_with_data], axis=0
        )
    else:
        raise ValueError("aggregate must be 'median' or 'latest'")
    x_index = np.arange(x_count, dtype=float)
    x_mm = (float(x_start_raw) + float(x_pitch_raw) * x_index) / 100.0 / 1000.0
    valid = np.isfinite(z_mm)
    return np.column_stack([x_mm[valid], np.zeros(np.count_nonzero(valid)), z_mm[valid]])
