from __future__ import annotations

import ctypes
import time

import numpy as np

from .hardware import LaserInterface, ProfileSample
from .profile_conversion import keyence_raw_to_points


class KeyenceLaserAdapter(LaserInterface):
    """Lazy-loaded Keyence LJ-X adapter using the bundled Linux SDK wrapper."""

    def __init__(
        self,
        ip: str = "192.168.1.1",
        control_port: int = 24691,
        high_speed_port: int = 24692,
        device_id: int = 0,
        batch_profiles: int = 5,
        aggregate: str = "median",
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.ip = ip
        self.control_port = int(control_port)
        self.high_speed_port = int(high_speed_port)
        self.device_id = int(device_id)
        self.batch_profiles = int(batch_profiles)
        self.aggregate = aggregate
        self.connect_timeout_s = float(connect_timeout_s)
        self._module = None
        self._wrapper = None
        self._sensor = None
        self._z_unit = None

    def connect(self) -> None:
        if self._sensor is not None:
            return
        try:
            octets = tuple(int(value) for value in self.ip.split("."))
        except ValueError as exc:
            raise ValueError(f"invalid Keyence IP address: {self.ip}") from exc
        if len(octets) != 4:
            raise ValueError(f"invalid Keyence IP address: {self.ip}")
        # Importing loads libljxacom.so, so it is intentionally deferred until
        # the user actually requests hardware access.
        from . import LJXAwrap, keyence

        sensor = keyence.Keyence(
            line_width=self.batch_profiles,
            ip_address=octets,
            control_port=self.control_port,
            high_speed_port=self.high_speed_port,
            device_id=self.device_id,
            timeout_sec=self.connect_timeout_s,
        )
        sensor.setup()
        z_unit = ctypes.c_ushort()
        result = LJXAwrap.LJX8IF_GetZUnitSimpleArray(sensor.deviceId, z_unit)
        if result != 0:
            sensor.close()
            raise RuntimeError(f"Keyence Z-unit query failed: {hex(result)}")
        self._module = keyence
        self._wrapper = LJXAwrap
        self._sensor = sensor
        self._z_unit = int(z_unit.value)

    def close(self) -> None:
        if self._sensor is not None:
            self._sensor.close()
        self._sensor = None
        self._module = None
        self._wrapper = None
        self._z_unit = None

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        if self._sensor is None or self._module is None or self._z_unit is None:
            raise RuntimeError("Keyence adapter is not connected")
        # Discard the previously completed batch and wait for a fresh one.
        self._module.image_available = False
        deadline = time.monotonic() + float(timeout_s)
        while not self._module.image_available:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for a fresh Keyence profile")
            time.sleep(0.001)
        profile_count = int(self._module.ysize_acquired)
        raw = np.asarray(self._module.z_val, dtype=np.int32).copy()
        timestamp_ns = (
            time.time_ns()
            if self._module.image_time_ns is None
            else int(self._module.image_time_ns)
        )
        self._module.image_available = False
        points = keyence_raw_to_points(
            raw_z=raw,
            x_count=int(self._sensor.xsize),
            profile_count=profile_count,
            x_start_raw=int(self._sensor.profinfo.lXStart),
            x_pitch_raw=int(self._sensor.profinfo.lXPitch),
            z_unit_raw=self._z_unit,
            aggregate=self.aggregate,
        )
        if len(points) == 0:
            raise RuntimeError("Keyence returned no valid profile points")
        return ProfileSample(points_s=points, timestamp_ns=timestamp_ns)
