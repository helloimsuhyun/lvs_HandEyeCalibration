from __future__ import annotations

import ctypes
from pathlib import Path
import threading
import time

import numpy as np


class _EthernetConfig(ctypes.Structure):
    _fields_ = [
        ("ip", ctypes.c_ubyte * 4),
        ("port", ctypes.c_ushort),
        ("reserve", ctypes.c_ubyte * 2),
    ]


class _ProfileInfo(ctypes.Structure):
    _fields_ = [
        ("profile_count", ctypes.c_ubyte),
        ("reserve1", ctypes.c_ubyte),
        ("luminance_output", ctypes.c_ubyte),
        ("reserve2", ctypes.c_ubyte),
        ("point_count", ctypes.c_ushort),
        ("reserve3", ctypes.c_ubyte * 2),
        ("x_start", ctypes.c_int),
        ("x_pitch", ctypes.c_int),
    ]


class _ProfileHeader(ctypes.Structure):
    _fields_ = [
        ("reserve", ctypes.c_uint),
        ("trigger_count", ctypes.c_uint),
        ("encoder_count", ctypes.c_int),
        ("reserve2", ctypes.c_uint * 3),
    ]


class _PreStartRequest(ctypes.Structure):
    _fields_ = [
        ("send_position", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 3),
    ]


_Callback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(_ProfileHeader),
    ctypes.POINTER(ctypes.c_ushort),
    ctypes.POINTER(ctypes.c_ushort),
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
)


class LaserAdapter:
    """Minimal Keyence LJ-X profile adapter using the bundled Linux library."""

    def __init__(
        self,
        ip: str = "192.168.1.1",
        control_port: int = 24691,
        high_speed_port: int = 24692,
        device_id: int = 0,
        batch_profiles: int = 5,
        aggregate: str = "median",
    ) -> None:
        self.ip = ip
        self.control_port = int(control_port)
        self.high_speed_port = int(high_speed_port)
        self.device_id = int(device_id)
        self.batch_profiles = int(batch_profiles)
        self.aggregate = aggregate
        self._dll = None
        self._callback = None
        self._profile_info = _ProfileInfo()
        self._z_unit = 0
        self._raw = np.empty(0, dtype=np.uint16)
        self._profile_count = 0
        self._last_profile_monotonic: float | None = None
        self._last_callback_monotonic: float | None = None
        self._callback_count = 0
        self._accepted_callback_count = 0
        self._dropped_callback_count = 0
        self._last_notify: int | None = None
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._opened = False
        self._high_speed_initialized = False
        self._measuring = False

    @staticmethod
    def _bind(dll: ctypes.CDLL, name: str, argtypes: list[object]):
        function = getattr(dll, name)
        function.restype = ctypes.c_int
        function.argtypes = argtypes
        return function

    def connect(self) -> None:
        if self._opened:
            return
        octets = tuple(int(value) for value in self.ip.split("."))
        if len(octets) != 4 or any(value < 0 or value > 255 for value in octets):
            raise ValueError(f"invalid Keyence IP address: {self.ip}")
        library_path = Path(__file__).parent / "keyence" / "libljxacom.so"
        self._dll = ctypes.cdll.LoadLibrary(str(library_path))
        dll = self._dll

        ethernet_open = self._bind(
            dll,
            "LJX8IF_EthernetOpen",
            [ctypes.c_int, ctypes.POINTER(_EthernetConfig)],
        )
        initialize = self._bind(
            dll,
            "LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray",
            [
                ctypes.c_int,
                ctypes.POINTER(_EthernetConfig),
                ctypes.c_ushort,
                _Callback,
                ctypes.c_uint,
                ctypes.c_uint,
            ],
        )
        pre_start = self._bind(
            dll,
            "LJX8IF_PreStartHighSpeedDataCommunication",
            [
                ctypes.c_int,
                ctypes.POINTER(_PreStartRequest),
                ctypes.POINTER(_ProfileInfo),
            ],
        )
        start_high_speed = self._bind(
            dll,
            "LJX8IF_StartHighSpeedDataCommunication",
            [ctypes.c_int],
        )
        start_measure = self._bind(
            dll, "LJX8IF_StartMeasure", [ctypes.c_int]
        )
        get_z_unit = self._bind(
            dll,
            "LJX8IF_GetZUnitSimpleArray",
            [ctypes.c_int, ctypes.POINTER(ctypes.c_ushort)],
        )

        ethernet = _EthernetConfig()
        for index, value in enumerate(octets):
            ethernet.ip[index] = value
        ethernet.port = self.control_port
        result = ethernet_open(self.device_id, ctypes.byref(ethernet))
        if result != 0:
            raise ConnectionError(f"Keyence Ethernet open failed: {hex(result)}")
        self._opened = True

        try:
            self._callback = _Callback(self._on_profiles)
            result = initialize(
                self.device_id,
                ctypes.byref(ethernet),
                self.high_speed_port,
                self._callback,
                self.batch_profiles,
                0,
            )
            if result != 0:
                raise RuntimeError(
                    f"Keyence high-speed initialization failed: {hex(result)}"
                )
            self._high_speed_initialized = True

            request = _PreStartRequest()
            request.send_position = 2
            result = pre_start(
                self.device_id,
                ctypes.byref(request),
                ctypes.byref(self._profile_info),
            )
            if result != 0:
                raise RuntimeError(f"Keyence pre-start failed: {hex(result)}")
            if self._profile_info.point_count <= 0:
                raise RuntimeError("Keyence reported zero profile points")

            z_unit = ctypes.c_ushort()
            result = get_z_unit(self.device_id, ctypes.byref(z_unit))
            if result != 0:
                raise RuntimeError(f"Keyence Z-unit query failed: {hex(result)}")
            self._z_unit = int(z_unit.value)

            with self._lock:
                self._raw = np.empty(0, dtype=np.uint16)
                self._profile_count = 0
                self._last_profile_monotonic = None
                self._last_callback_monotonic = None
                self._callback_count = 0
                self._accepted_callback_count = 0
                self._dropped_callback_count = 0
                self._last_notify = None
            self._frame_ready.clear()
            result = start_high_speed(self.device_id)
            if result != 0:
                raise RuntimeError(f"Keyence high-speed start failed: {hex(result)}")
            result = start_measure(self.device_id)
            if result not in (0, 0x8080):
                raise RuntimeError(f"Keyence measurement start failed: {hex(result)}")

            if result == 0x8080:
                print(
                    "Keyence StartMeasure returned 0x8080; "
                    "continuing with the existing measurement state"
                )
            self._measuring = True
        except BaseException:
            self.close()
            raise

    def _on_profiles(
        self,
        _headers,
        heights,
        _luminance,
        _luminance_enabled,
        point_count,
        profile_count,
        notify,
        _user,
    ) -> None:
        received_at = time.monotonic()
        notify_value = int(notify)
        if notify_value not in (0, 0x10000) or profile_count == 0:
            with self._lock:
                self._callback_count += 1
                self._dropped_callback_count += 1
                self._last_callback_monotonic = received_at
                self._last_notify = notify_value
            return None
        total = int(point_count) * int(profile_count)
        raw = np.ctypeslib.as_array(heights, shape=(total,)).copy()
        with self._lock:
            self._callback_count += 1
            self._accepted_callback_count += 1
            self._last_callback_monotonic = received_at
            self._last_notify = notify_value
            self._raw = raw
            self._profile_count = int(profile_count)
            self._last_profile_monotonic = received_at
        self._frame_ready.set()
        return None

    def read_profile(self, *, timeout_s: float) -> np.ndarray:
        if not self._measuring:
            raise RuntimeError("Keyence adapter is not connected")
        self._frame_ready.clear()
        if not self._frame_ready.wait(float(timeout_s)):
            raise TimeoutError("timed out waiting for a fresh Keyence profile")
        with self._lock:
            raw = self._raw.copy()
            profile_count = self._profile_count
        return self._decode_profile(raw, profile_count)

    def read_latest_profile(
        self, *, max_age_s: float | None = None
    ) -> np.ndarray | None:
        """Return a recent callback profile immediately, or ``None``."""
        sample = self.read_latest_profile_sample(max_age_s=max_age_s)
        return None if sample is None else sample[2]

    def read_latest_profile_sample(
        self, *, max_age_s: float | None = None
    ) -> tuple[int, float, np.ndarray] | None:
        """Return ``(callback_id, received_at, points)`` without waiting."""
        if not self._measuring:
            raise RuntimeError("Keyence adapter is not connected")
        if max_age_s is not None and max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        now = time.monotonic()
        with self._lock:
            if (
                self._profile_count <= 0
                or self._raw.size == 0
                or self._last_profile_monotonic is None
            ):
                return None
            if (
                max_age_s is not None
                and now - self._last_profile_monotonic > max_age_s
            ):
                return None
            raw = self._raw.copy()
            profile_count = self._profile_count
            callback_id = self._accepted_callback_count
            received_at = self._last_profile_monotonic
        return (
            callback_id,
            received_at,
            self._decode_profile(raw, profile_count),
        )

    def _decode_profile(
        self, raw: np.ndarray, profile_count: int
    ) -> np.ndarray:
        x_count = int(self._profile_info.point_count)
        required = x_count * int(profile_count)
        if x_count <= 0 or profile_count <= 0 or raw.size < required:
            raise RuntimeError("Keyence returned an incomplete profile buffer")
        raw = raw[: x_count * profile_count].reshape(profile_count, x_count)
        decoded = (
            (raw.astype(float) - 32768.0)
            * (float(self._z_unit) / 100.0)
            / 1000.0
        )
        decoded[raw == 0] = np.nan
        if self.aggregate == "latest":
            z_mm = decoded[-1]
        elif self.aggregate == "median":
            z_mm = np.full(x_count, np.nan)
            valid_columns = np.any(raw != 0, axis=0)
            z_mm[valid_columns] = np.nanmedian(
                decoded[:, valid_columns], axis=0
            )
        else:
            raise ValueError("aggregate must be 'median' or 'latest'")
        index = np.arange(x_count, dtype=float)
        x_mm = (
            (float(self._profile_info.x_start) + self._profile_info.x_pitch * index)
            / 100.0
            / 1000.0
        )
        valid = np.isfinite(z_mm)
        points = np.column_stack(
            [x_mm[valid], np.zeros(np.count_nonzero(valid)), z_mm[valid]]
        )
        if len(points) == 0:
            raise RuntimeError("Keyence returned no valid profile points")
        return points

    def read_diagnostics(self, *, max_errors: int = 16) -> dict[str, object]:
        """Read controller state without clearing or resetting any errors."""
        if not self._opened or self._dll is None:
            raise RuntimeError("Keyence adapter is not connected")
        if not 1 <= int(max_errors) <= 255:
            raise ValueError("max_errors must be between 1 and 255")

        get_error = self._bind(
            self._dll,
            "LJX8IF_GetError",
            [
                ctypes.c_int,
                ctypes.c_ubyte,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.POINTER(ctypes.c_ushort),
            ],
        )
        get_attention_status = self._bind(
            self._dll,
            "LJX8IF_GetAttentionStatus",
            [ctypes.c_int, ctypes.POINTER(ctypes.c_ushort)],
        )
        get_trigger_and_pulse_count = self._bind(
            self._dll,
            "LJX8IF_GetTriggerAndPulseCount",
            [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_int),
            ],
        )

        error_count = ctypes.c_ubyte()
        error_codes_buffer = (ctypes.c_ushort * int(max_errors))()
        get_error_result = int(
            get_error(
                self.device_id,
                int(max_errors),
                ctypes.byref(error_count),
                error_codes_buffer,
            )
        )

        attention_status = ctypes.c_ushort()
        get_attention_result = int(
            get_attention_status(
                self.device_id,
                ctypes.byref(attention_status),
            )
        )

        trigger_count = ctypes.c_uint()
        encoder_count = ctypes.c_int()
        get_count_result = int(
            get_trigger_and_pulse_count(
                self.device_id,
                ctypes.byref(trigger_count),
                ctypes.byref(encoder_count),
            )
        )

        now = time.monotonic()
        with self._lock:
            callback_age_s = (
                None
                if self._last_callback_monotonic is None
                else max(0.0, now - self._last_callback_monotonic)
            )
            profile_callback_age_s = (
                None
                if self._last_profile_monotonic is None
                else max(0.0, now - self._last_profile_monotonic)
            )
            callback_count = self._callback_count
            accepted_callback_count = self._accepted_callback_count
            dropped_callback_count = self._dropped_callback_count
            last_notify = self._last_notify
            latest_buffer_has_data = bool(np.any(self._raw))

        return {
            "get_error_result": get_error_result,
            "error_codes": (
                [
                    int(error_codes_buffer[index])
                    for index in range(min(int(error_count.value), int(max_errors)))
                ]
                if get_error_result == 0
                else None
            ),
            "get_attention_result": get_attention_result,
            "attention_status": (
                int(attention_status.value) if get_attention_result == 0 else None
            ),
            "get_count_result": get_count_result,
            "trigger_count": (
                int(trigger_count.value) if get_count_result == 0 else None
            ),
            "encoder_count": (
                int(encoder_count.value) if get_count_result == 0 else None
            ),
            "callback_count": callback_count,
            "accepted_callback_count": accepted_callback_count,
            "dropped_callback_count": dropped_callback_count,
            "last_notify": last_notify,
            "callback_age_s": callback_age_s,
            "profile_callback_age_s": profile_callback_age_s,
            "latest_buffer_has_data": latest_buffer_has_data,
        }

    def close(self) -> None:
        if self._dll is None:
            return
        if self._measuring:
            self._bind(self._dll, "LJX8IF_StopMeasure", [ctypes.c_int])(
                self.device_id
            )
            self._measuring = False
        if self._high_speed_initialized:
            self._bind(
                self._dll,
                "LJX8IF_StopHighSpeedDataCommunication",
                [ctypes.c_int],
            )(self.device_id)
            self._bind(
                self._dll,
                "LJX8IF_FinalizeHighSpeedDataCommunication",
                [ctypes.c_int],
            )(self.device_id)
            self._high_speed_initialized = False
        if self._opened:
            self._bind(
                self._dll, "LJX8IF_CommunicationClose", [ctypes.c_int]
            )(self.device_id)
            self._opened = False
        self._dll = None
        self._callback = None
