# 트리거가 오면 현재 프로파일 데이터와 로봇의 TCP transform을 패키징 (input : TCP, 트리거)
# 실시간 프로파일 데이터를 일정 주파수로 시각화
# 제일 최근의 프로파일 캡처 데이터와 TCP transform은 변수로 저장하고 있는다.

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import LJXAwrap, keyence
    from .keyence import Keyence
except ImportError:  # Direct execution
    import LJXAwrap
    import keyence
    from keyence import Keyence


# ============================================================
# 설정
# ============================================================

VISUALIZE_HZ = 10.0
SAVE_DIR = Path("captures")


# ============================================================
# 최신 데이터 저장 변수
# ============================================================

data_lock = threading.Lock()

latest_profile = None
latest_profile_time_ns = None

latest_T_base_tcp = None
latest_tcp_time_ns = None

latest_capture = None
capture_count = 0

stop_event = threading.Event()


# ============================================================
# 센서 데이터 변환
# ============================================================

def get_z_unit(device_id: int) -> int:
    """Keyence 센서의 Z 단위값을 읽는다."""
    z_unit = ctypes.c_ushort()
    LJXAwrap.LJX8IF_GetZUnitSimpleArray(device_id, z_unit)
    return z_unit.value


def raw_profile_to_points(
    raw_z: np.ndarray,
    xsize: int,
    ysize: int,
    profinfo,
    z_unit_value: int,
) -> np.ndarray:
    """
    Keyence raw profile을 N x 3 [x, 0, z] 점으로 변환한다.

    여러 프로파일 라인이 들어온 경우 가장 최근 라인만 사용한다.
    """
    raw_z = np.asarray(raw_z, dtype=np.int32)

    if xsize <= 0 or ysize <= 0:
        return np.empty((0, 3))

    if len(raw_z) != xsize * ysize:
        return np.empty((0, 3))

    # 가장 최근 프로파일 한 줄
    raw_z = raw_z.reshape(ysize, xsize)[-1]

    # raw 값이 0인 점은 무효 측정
    valid = raw_z != 0

    if not np.any(valid):
        return np.empty((0, 3))

    x_index = np.arange(xsize)

    # X 좌표 [mm]
    x_mm = (
        profinfo.lXStart
        + profinfo.lXPitch * x_index
    ) / 100.0 / 1000.0

    # Z 좌표 [mm]
    z_mm = (
        raw_z.astype(float) - 32768.0
    ) * (z_unit_value / 100.0) / 1000.0

    return np.column_stack(
        [
            x_mm[valid],
            np.zeros(np.count_nonzero(valid)),
            z_mm[valid],
        ]
    )


# ============================================================
# TCP 입력
# ============================================================

def update_tcp(T_base_tcp: np.ndarray) -> None:
    """
    로봇에서 받은 최신 TCP transform을 저장한다.

    로봇 통신 callback에서 이 함수를 호출하면 된다.
    """
    global latest_T_base_tcp
    global latest_tcp_time_ns

    T_base_tcp = np.asarray(T_base_tcp, dtype=float)

    if T_base_tcp.shape != (4, 4):
        raise ValueError("T_base_tcp must be 4x4")

    with data_lock:
        latest_T_base_tcp = T_base_tcp.copy()
        latest_tcp_time_ns = time.time_ns()


# ============================================================
# 트리거 입력
# ============================================================

def capture() -> dict:
    """
    최신 프로파일과 최신 TCP transform을 하나로 패키징한다.

    트리거 callback에서 이 함수를 호출하면 된다.
    """
    global latest_capture
    global capture_count

    with data_lock:
        if latest_profile is None:
            raise RuntimeError("profile data is not available")

        if latest_T_base_tcp is None:
            raise RuntimeError("TCP transform is not available")

        capture_count += 1

        latest_capture = {
            "index": capture_count,
            "profile": latest_profile.copy(),
            "T_base_tcp": latest_T_base_tcp.copy(),
            "profile_time_ns": latest_profile_time_ns,
            "tcp_time_ns": latest_tcp_time_ns,
            "trigger_time_ns": time.time_ns(),
        }

    save_capture(latest_capture)

    print(
        f"[CAPTURE] index={capture_count}, "
        f"points={len(latest_capture['profile'])}"
    )

    return latest_capture


def save_capture(data: dict) -> None:
    """캡처한 프로파일과 TCP transform을 파일로 저장한다."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    index = data["index"]

    profile_path = SAVE_DIR / f"profile_{index:04d}.csv"
    tcp_path = SAVE_DIR / f"tcp_{index:04d}.csv"
    npz_path = SAVE_DIR / f"capture_{index:04d}.npz"

    np.savetxt(
        profile_path,
        data["profile"],
        delimiter=",",
        header="x_s_mm,y_s_mm,z_s_mm",
        comments="",
    )

    np.savetxt(
        tcp_path,
        data["T_base_tcp"],
        delimiter=",",
    )

    np.savez(
        npz_path,
        profile=data["profile"],
        T_base_tcp=data["T_base_tcp"],
        profile_time_ns=data["profile_time_ns"],
        tcp_time_ns=data["tcp_time_ns"],
        trigger_time_ns=data["trigger_time_ns"],
    )


# ============================================================
# 센서 프로파일 갱신
# ============================================================

def profile_update_loop(
    sensor: Keyence,
    z_unit_value: int,
) -> None:
    """Keyence 센서에서 최신 프로파일을 계속 읽어 저장한다."""
    global latest_profile
    global latest_profile_time_ns

    keyence.image_available = False

    while not stop_event.is_set():
        if not keyence.image_available:
            time.sleep(0.001)
            continue

        # SDK 공유 버퍼를 빠르게 복사
        ysize = int(keyence.ysize_acquired)
        raw_z = np.asarray(keyence.z_val, dtype=np.int32).copy()

        keyence.image_available = False

        points = raw_profile_to_points(
            raw_z=raw_z,
            xsize=sensor.xsize,
            ysize=ysize,
            profinfo=sensor.profinfo,
            z_unit_value=z_unit_value,
        )

        if len(points) == 0:
            continue

        with data_lock:
            latest_profile = points
            latest_profile_time_ns = time.time_ns()


# ============================================================
# 실시간 시각화
# ============================================================

def visualization_loop() -> None:
    """최신 프로파일을 일정 주파수로 표시한다."""
    plt.ion()

    fig, ax = plt.subplots()
    line, = ax.plot([], [], ".")

    ax.set_xlabel("Sensor X [mm]")
    ax.set_ylabel("Sensor Z [mm]")
    ax.grid(True)

    period = 1.0 / VISUALIZE_HZ

    while not stop_event.is_set():
        with data_lock:
            points = (
                None
                if latest_profile is None
                else latest_profile.copy()
            )

        if points is not None and len(points) > 0:
            line.set_data(
                points[:, 0],
                points[:, 2],
            )

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        time.sleep(period)

    plt.close(fig)


# ============================================================
# 실행
# ============================================================

def main() -> None:
    sensor = Keyence()
    sensor.setup()

    z_unit_value = get_z_unit(sensor.deviceId)

    profile_thread = threading.Thread(
        target=profile_update_loop,
        args=(sensor, z_unit_value),
        daemon=True,
    )

    visualization_thread = threading.Thread(
        target=visualization_loop,
        daemon=True,
    )

    profile_thread.start()
    visualization_thread.start()

    # 실제 로봇 TCP가 들어오기 전 테스트용 초기값
    update_tcp(np.eye(4))

    print("Enter: capture")
    print("q: quit")

    try:
        while True:
            command = input().strip().lower()

            if command == "q":
                break

            # Enter가 트리거 역할
            capture()

    finally:
        stop_event.set()

        profile_thread.join(timeout=1.0)
        visualization_thread.join(timeout=1.0)

        sensor.close()


if __name__ == "__main__":
    main()
