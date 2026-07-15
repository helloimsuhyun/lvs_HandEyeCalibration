from __future__ import annotations

import numpy as np

from .hardware import RobotInterface, transform_from_xyz_rpy_deg, xyz_rpy_deg_from_transform


class RobotAdapter(RobotInterface):
    """Copy this file and replace the five small vendor-specific blocks.

    The workflow deliberately does not guess a robot brand or Euler convention.
    This template assumes the controller exchanges [x, y, z, rx, ry, rz] in
    millimetres/degrees with extrinsic XYZ angles.  Change the two conversion
    calls if the controller uses a rotation vector, quaternion, ZYX Euler, or
    metres.  The configured TCP must stay unchanged for bootstrap and capture.
    """

    def __init__(self, host: str, port: int | None = None) -> None:
        self.host = host
        self.port = port
        self.client = None

    def connect(self) -> None:
        # TODO: self.client = VendorRobotClient(self.host, self.port)
        # TODO: enable/read-only-check remote mode, servo state and safety state.
        raise NotImplementedError("implement the vendor robot connection")

    def current_T_base_tcp(self) -> np.ndarray:
        # TODO: values = self.client.get_current_tcp_xyzrpy_deg()
        values = None
        if values is None:
            raise NotImplementedError("read the current TCP from the robot")
        return transform_from_xyz_rpy_deg(values)

    def move_tcp(
        self,
        T_base_tcp: np.ndarray,
        *,
        linear_speed_mm_s: float,
        angular_speed_deg_s: float,
        timeout_s: float,
    ) -> None:
        values = xyz_rpy_deg_from_transform(T_base_tcp)
        # TODO: call the vendor's blocking, collision-monitored Cartesian-linear
        # TCP command. Do not substitute an unreviewed MoveJ trajectory.
        # self.client.move_tcp(
        #     values,
        #     linear_speed_mm_s=linear_speed_mm_s,
        #     angular_speed_deg_s=angular_speed_deg_s,
        #     timeout_s=timeout_s,
        # )
        del values, linear_speed_mm_s, angular_speed_deg_s, timeout_s
        raise NotImplementedError("send a blocking TCP motion command")

    def stop(self) -> None:
        # TODO: if self.client is not None: self.client.controlled_stop()
        pass

    def supports_independent_stop(self) -> bool:
        # Change to True only after stop() can interrupt a concurrently blocking
        # move_tcp() call, preferably over an independent controller channel.
        return False

    def close(self) -> None:
        # TODO: close the vendor client/socket.
        self.client = None

    def controller_path_is_safe(self, poses: list[np.ndarray]) -> bool | None:
        # TODO (required by the example config): call controller IK + joint-limit
        # + full-link scene collision validation for the same linear motion mode.
        # poses includes the current start pose and every commanded endpoint.
        del poses
        return None
