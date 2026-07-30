# Mock RB robot adapter
# 실제 로봇 없이 랜덤 TCP pose를 반환하는 테스트용 어댑터

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class RobotAdapter:
    """
    실제 Rainbow Robotics 로봇 대신 랜덤한 TCP pose를 반환하는 Mock 어댑터.

    기존 실제 RobotAdapter와 동일한 클래스명과 메서드를 사용한다.

    TCP pose 형식:
        [x, y, z, rx, ry, rz]

    단위:
        x, y, z    : mm
        rx, ry, rz : deg

    반환 좌표계:
        T_base_tcp

    반환되는 4x4 변환행렬의 translation 단위도 mm이다.
    """

    def __init__(
        self,
        host: str,
        port: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.host = host
        self.port = port

        # 실제 어댑터와 속성 이름을 맞춤
        self.client = None
        self.data_client = None
        self.response_collector = None

        self.connected = False
        self.rng = np.random.default_rng(seed)

        # 랜덤 TCP 생성 범위
        self.xyz_min_mm = np.array(
            [-500.0, -500.0, 100.0],
            dtype=np.float64,
        )
        self.xyz_max_mm = np.array(
            [500.0, 500.0, 800.0],
            dtype=np.float64,
        )

        self.rpy_min_deg = np.array(
            [-180.0, -90.0, -180.0],
            dtype=np.float64,
        )
        self.rpy_max_deg = np.array(
            [180.0, 90.0, 180.0],
            dtype=np.float64,
        )

        self._current_pose: np.ndarray | None = None

    def connect(self) -> None:
        """
        Mock 로봇을 연결 상태로 만든다.
        실제 네트워크 연결은 수행하지 않는다.
        """
        self.connected = True

        # 실제 객체가 존재하는 것처럼 표시하기 위한 더미 값
        self.client = object()
        self.data_client = object()
        self.response_collector = object()

        self._current_pose = self._generate_random_tcp_pose()

        print(
            f"[MOCK ROBOT] connected: "
            f"host={self.host}, port={self.port}"
        )

    def _check_connected(self) -> None:
        if not self.connected:
            raise RuntimeError(
                "RobotAdapter가 연결되지 않았습니다. "
                "connect()를 먼저 호출하세요."
            )

    def _generate_random_tcp_pose(self) -> np.ndarray:
        """
        설정된 범위에서 랜덤 TCP pose를 생성한다.

        Returns
        -------
        np.ndarray
            [x, y, z, rx, ry, rz]
        """
        xyz_mm = self.rng.uniform(
            low=self.xyz_min_mm,
            high=self.xyz_max_mm,
        )

        rpy_deg = self.rng.uniform(
            low=self.rpy_min_deg,
            high=self.rpy_max_deg,
        )

        return np.concatenate(
            [xyz_mm, rpy_deg]
        ).astype(np.float64)

    def read_tcp_pose_vec_mm(self) -> np.ndarray:
        """
        현재 Mock TCP pose를 반환한다.

        한 번의 캡처에서 전/후 TCP가 동일하도록 읽기만으로는 자세를
        변경하지 않는다. 캡처 저장 후 ``advance_pose()``가 다음 자세를
        생성한다.

        Returns
        -------
        np.ndarray
            [x, y, z, rx, ry, rz]
            x, y, z는 mm
            rx, ry, rz는 deg
        """
        self._check_connected()
        if self._current_pose is None:
            self._current_pose = self._generate_random_tcp_pose()
        return self._current_pose.copy()

    def advance_pose(self) -> None:
        """다음 Mock 캡처에서 사용할 랜덤 TCP pose를 생성한다."""
        self._check_connected()
        self._current_pose = self._generate_random_tcp_pose()

    def read_T_base_tcp(self) -> np.ndarray:
        """
        랜덤 TCP pose를 T_base_tcp 변환행렬로 반환한다.

        Returns
        -------
        np.ndarray
            shape (4, 4)의 homogeneous transformation matrix.
            translation 단위는 mm.
        """
        values = self.read_tcp_pose_vec_mm()

        xyz_mm = values[:3]
        rpy_deg = values[3:6]

        transform = np.eye(4, dtype=np.float64)

        # Extrinsic XYZ Euler angles
        # R = Rz(rz) @ Ry(ry) @ Rx(rx)
        transform[:3, :3] = Rotation.from_euler(
            "xyz",
            rpy_deg,
            degrees=True,
        ).as_matrix()

        transform[:3, 3] = xyz_mm

        return transform

    def close(self) -> None:
        """
        Mock 연결을 종료한다.
        """
        self.connected = False
        self._current_pose = None

        self.data_client = None
        self.response_collector = None
        self.client = None

        print("[MOCK ROBOT] disconnected")
