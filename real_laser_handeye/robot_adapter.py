# RB robot adaptor 
# made by gpt

from __future__ import annotations

import math
import time

import numpy as np
import rbpodo as rb
from scipy.spatial.transform import Rotation


class RobotAdapter:
    """
    Rainbow Robotics 로봇 어댑터.

    제공 기능
    ---------
    1. 현재 TCP pose 읽기
    2. T_base_tcp 4x4 변환행렬 반환
    3. Cartesian linear move_l 명령
    4. 실제 TCP 기반 목표 도착 확인

    TCP pose 형식
    -------------
    [x, y, z, rx, ry, rz]

    단위
    ----
    x, y, z    : mm
    rx, ry, rz : deg

    반환되는 4x4 변환행렬의 translation도 mm 단위이다.
    """

    def __init__(self, host: str, port: int | None = None) -> None:
        self.host = host
        self.port = port

        # 로봇 명령 전송용
        self.client: rb.Cobot | None = None

        # 로봇 상태/TCP 좌표 수신용
        self.data_client: rb.CobotData | None = None

        # 로봇 명령 응답 및 오류 확인용
        self.response_collector: rb.ResponseCollector | None = None

    def connect(
        self,
        *,
        operation_mode: str | None = None,
        speed_bar: float | None = None,
    ) -> None:
        """
        로봇 명령 채널과 상태 데이터 채널을 생성한다.

        Parameters
        ----------
        operation_mode
            None, "real", "simulation" 중 하나.
            None이면 로봇의 현재 operation mode를 변경하지 않는다.
        speed_bar
            0.0~1.0 범위의 로봇 속도 배율.
            None이면 현재 speed bar를 변경하지 않는다.

        Notes
        -----
        rbpodo의 Cobot/CobotData는 생성 시 로봇 IP에 연결된다.
        port는 rbpodo에서 내부적으로 관리하므로 여기서는 사용하지 않는다.
        """
        try:
            self.client = rb.Cobot(self.host)
            self.data_client = rb.CobotData(self.host)
            self.response_collector = rb.ResponseCollector()

            # 실제 로봇과 통신 가능한지 상태 데이터를 한 번 읽어 확인
            state = self.data_client.request_data()
            if state is None:
                raise ConnectionError(
                    f"로봇 상태 데이터를 받지 못했습니다: host={self.host}"
                )

            if operation_mode is not None:
                mode_text = operation_mode.strip().lower()
                if mode_text == "real":
                    mode = rb.OperationMode.Real
                elif mode_text in ("simulation", "sim"):
                    mode = rb.OperationMode.Simulation
                else:
                    raise ValueError(
                        "operation_mode는 None, 'real', 'simulation' 중 하나여야 합니다."
                    )

                self.client.set_operation_mode(
                    self.response_collector,
                    mode,
                )

            if speed_bar is not None:
                speed_bar = float(speed_bar)
                if not 0.0 < speed_bar <= 1.0:
                    raise ValueError("speed_bar는 0보다 크고 1 이하여야 합니다.")

                self.client.set_speed_bar(
                    self.response_collector,
                    speed_bar,
                )

            if operation_mode is not None or speed_bar is not None:
                self.client.flush(self.response_collector)
                self._throw_if_robot_error()

        except Exception:
            self.close()
            raise

    def _require_connected(self) -> None:
        if self.client is None:
            raise RuntimeError(
                "로봇 명령 채널이 연결되지 않았습니다. connect()를 먼저 호출하세요."
            )
        if self.data_client is None:
            raise RuntimeError(
                "로봇 상태 채널이 연결되지 않았습니다. connect()를 먼저 호출하세요."
            )
        if self.response_collector is None:
            raise RuntimeError(
                "ResponseCollector가 초기화되지 않았습니다. connect()를 먼저 호출하세요."
            )

    def _throw_if_robot_error(self) -> None:
        if self.response_collector is None:
            raise RuntimeError("ResponseCollector가 초기화되지 않았습니다.")
        self.response_collector.error().throw_if_not_empty()

    @staticmethod
    def _validate_pose_vec(pose_vec: np.ndarray, name: str) -> np.ndarray:
        pose = np.asarray(pose_vec, dtype=np.float64).reshape(-1)

        if pose.size != 6:
            raise ValueError(
                f"{name}은 [x, y, z, rx, ry, rz] 형식의 6개 값이어야 합니다."
            )
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"{name}에 유효하지 않은 값이 있습니다: {pose}")

        return pose.copy()

    @staticmethod
    def _angle_difference_deg(
        current_deg: np.ndarray,
        target_deg: np.ndarray,
    ) -> np.ndarray:
        current = np.asarray(current_deg, dtype=np.float64)
        target = np.asarray(target_deg, dtype=np.float64)
        return (current - target + 180.0) % 360.0 - 180.0

    def _read_state(self):
        if self.data_client is None:
            raise RuntimeError(
                "RobotAdapter가 연결되지 않았습니다. connect()를 먼저 호출하세요."
            )

        state = self.data_client.request_data()

        if state is None:
            raise RuntimeError("로봇으로부터 현재 상태 데이터를 받지 못했습니다.")
        if not hasattr(state, "sdata"):
            raise RuntimeError("수신된 로봇 상태에 sdata 필드가 없습니다.")

        return state

    @staticmethod
    def _extract_tcp_pose(state) -> np.ndarray:
        if hasattr(state.sdata, "tcp"):
            values = state.sdata.tcp
        elif hasattr(state.sdata, "tcp_pos"):
            values = state.sdata.tcp_pos
        elif hasattr(state.sdata, "cur_pos"):
            values = state.sdata.cur_pos
        else:
            raise RuntimeError(
                "TCP pose 필드를 찾지 못했습니다. "
                "확인한 필드: tcp, tcp_pos, cur_pos"
            )

        values = np.asarray(values, dtype=np.float64).reshape(-1)

        if values.size < 6:
            raise RuntimeError(
                f"TCP 데이터 크기가 올바르지 않습니다: size={values.size}"
            )

        pose = values[:6].copy()

        if not np.all(np.isfinite(pose)):
            raise RuntimeError(
                f"TCP 데이터에 유효하지 않은 값이 있습니다: {pose}"
            )

        return pose

    def read_tcp_pose_vec_mm(self) -> np.ndarray:
        """
        현재 TCP를 [x, y, z, rx, ry, rz] 형식으로 반환한다.

        Returns
        -------
        np.ndarray
            x, y, z는 mm, rx, ry, rz는 deg.
        """
        state = self._read_state()
        return self._extract_tcp_pose(state)

    def read_T_base_tcp(self) -> np.ndarray:
        """
        현재 TCP pose를 읽어 T_base_tcp로 변환한다.

        Returns
        -------
        np.ndarray
            shape (4, 4)의 homogeneous transformation matrix.
            translation 단위는 mm이다.
        """
        pose = self.read_tcp_pose_vec_mm()
        xyz_mm = pose[:3]
        rpy_deg = pose[3:6]

        transform = np.eye(4, dtype=np.float64)

        # scipy의 소문자 "xyz"는 extrinsic XYZ:
        # R = Rz(rz) @ Ry(ry) @ Rx(rx)
        transform[:3, :3] = Rotation.from_euler(
            "xyz",
            rpy_deg,
            degrees=True,
        ).as_matrix()

        transform[:3, 3] = xyz_mm
        return transform

    def wait_until_tcp_reached(
        self,
        target_pose_vec_mm: np.ndarray,
        *,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        timeout_s: float = 30.0,
        stable_count: int = 5,
        poll_interval_s: float = 0.05,
    ) -> np.ndarray:
        """
        현재 TCP가 목표 pose 허용 범위에 안정적으로 도달할 때까지 기다린다.

        Returns
        -------
        np.ndarray
            실제 도착 TCP pose [x, y, z, rx, ry, rz].
        """
        target = self._validate_pose_vec(
            target_pose_vec_mm,
            "target_pose_vec_mm",
        )

        if position_tolerance_mm <= 0:
            raise ValueError("position_tolerance_mm은 양수여야 합니다.")
        if rotation_tolerance_deg <= 0:
            raise ValueError("rotation_tolerance_deg는 양수여야 합니다.")
        if timeout_s <= 0:
            raise ValueError("timeout_s는 양수여야 합니다.")
        if stable_count <= 0:
            raise ValueError("stable_count는 1 이상이어야 합니다.")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s는 양수여야 합니다.")

        started_at = time.monotonic()
        current_stable_count = 0

        last_pose: np.ndarray | None = None
        last_position_error = math.inf
        last_rotation_error = math.inf

        while time.monotonic() - started_at <= timeout_s:
            current = self.read_tcp_pose_vec_mm()
            last_pose = current

            position_error = float(
                np.linalg.norm(current[:3] - target[:3])
            )

            rotation_error_vector = self._angle_difference_deg(
                current[3:6],
                target[3:6],
            )
            rotation_error = float(
                np.linalg.norm(rotation_error_vector)
            )

            last_position_error = position_error
            last_rotation_error = rotation_error

            if (
                position_error <= position_tolerance_mm
                and rotation_error <= rotation_tolerance_deg
            ):
                current_stable_count += 1
            else:
                current_stable_count = 0

            if current_stable_count >= stable_count:
                return current.copy()

            time.sleep(poll_interval_s)

        raise TimeoutError(
            "목표 TCP pose에 도달하지 못했습니다. "
            f"target={target.tolist()}, "
            f"current={None if last_pose is None else last_pose.tolist()}, "
            f"position_error={last_position_error:.3f} mm, "
            f"rotation_error={last_rotation_error:.3f} deg"
        )

    def move_l(
        self,
        target_pose_vec_mm: np.ndarray,
        *,
        speed_mm_s: float = 80.0,
        accel_mm_s2: float = 80.0,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        timeout_s: float = 30.0,
        stable_count: int = 5,
        poll_interval_s: float = 0.05,
    ) -> np.ndarray:
        """
        Cartesian linear motion을 수행하고 실제 TCP 도착까지 대기한다.

        Parameters
        ----------
        target_pose_vec_mm
            [x, y, z, rx, ry, rz]
            translation은 mm, rotation은 deg.
        speed_mm_s
            rbpodo move_l 속도 인자.
        accel_mm_s2
            rbpodo move_l 가속도 인자.
        position_tolerance_mm
            위치 도착 허용 오차.
        rotation_tolerance_deg
            자세 도착 허용 오차.
        timeout_s
            도착 확인 제한 시간.
        stable_count
            허용 오차 내에 연속으로 들어와야 하는 횟수.
        poll_interval_s
            TCP 상태 확인 주기.

        Returns
        -------
        np.ndarray
            실제 도착 TCP pose [x, y, z, rx, ry, rz].
        """
        self._require_connected()

        target = self._validate_pose_vec(
            target_pose_vec_mm,
            "target_pose_vec_mm",
        )

        if speed_mm_s <= 0:
            raise ValueError("speed_mm_s는 양수여야 합니다.")
        if accel_mm_s2 <= 0:
            raise ValueError("accel_mm_s2는 양수여야 합니다.")

        assert self.client is not None
        assert self.response_collector is not None

        # 이전 명령에서 남은 오류가 있는지 확인
        self._throw_if_robot_error()

        self.client.move_l(
            self.response_collector,
            target,
            float(speed_mm_s),
            float(accel_mm_s2),
        )

        self._throw_if_robot_error()

        arrived_pose = self.wait_until_tcp_reached(
            target,
            position_tolerance_mm=position_tolerance_mm,
            rotation_tolerance_deg=rotation_tolerance_deg,
            timeout_s=timeout_s,
            stable_count=stable_count,
            poll_interval_s=poll_interval_s,
        )

        self._throw_if_robot_error()
        return arrived_pose

    def stop(self) -> None:
        """
        가능한 경우 현재 로봇 이동을 정지한다.

        rbpodo 버전에 따라 stop 계열 API가 다를 수 있으므로,
        지원되는 메서드를 순서대로 확인한다.
        """
        self._require_connected()

        assert self.client is not None
        assert self.response_collector is not None

        stop_methods = (
            "task_stop",
            "stop",
            "halt",
        )

        for method_name in stop_methods:
            method = getattr(self.client, method_name, None)
            if callable(method):
                try:
                    method(self.response_collector)
                except TypeError:
                    method()
                self._throw_if_robot_error()
                return

        raise NotImplementedError(
            "현재 rbpodo 버전에서 지원되는 stop 메서드를 찾지 못했습니다."
        )

    def close(self) -> None:
        """
        rbpodo 객체 참조를 해제한다.

        현재 사용 중인 rbpodo 인터페이스에는 명시적인 close()가 없으므로
        객체 참조를 제거하여 정리한다.
        """
        self.data_client = None
        self.response_collector = None
        self.client = None