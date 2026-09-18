#!/usr/bin/env python3
"""Numerically verify the tool0/flange/sensor transform contract."""

import math
import numpy as np


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def T(R, p):
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = p
    return out


# URDF fixed joint flange -> tool0: rpy=(+pi/2, 0, +pi/2)
T_f_t = T(rz(math.pi / 2) @ ry(0) @ rx(math.pi / 2), [0, 0, 0])

# Application JSON: tool0 -> sensor.  Compared with the original mount, the
# sensor is rotated pi about wrist_3/tool0 +Z.
T_t_s = T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0.151])

# Xacro: flange -> sensor, rpy=(-pi/2, 0, +pi/2)
T_f_s_xacro = T(rz(math.pi / 2) @ ry(0) @ rx(-math.pi / 2), [0.151, 0, 0])

T_f_s_composed = T_f_t @ T_t_s
np.testing.assert_allclose(T_f_s_composed, T_f_s_xacro, atol=1e-10)

T_s_p = T(np.eye(3), [0, 0, 0.08])
T_t_p = T_t_s @ T_s_p
np.testing.assert_allclose(T_t_p[:3, 3], [0, 0, 0.071], atol=1e-12)

print("OK: ^flange T_sensor matches ^flange T_tool0 @ ^tool0 T_sensor")
print("OK: physical origin is +71 mm along tool0 +Z")
print(T_f_s_composed)
