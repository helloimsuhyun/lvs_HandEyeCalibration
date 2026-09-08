from __future__ import annotations

import math
import threading
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.time import Time

    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from control_msgs.msg import JointTolerance
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectoryPoint
    from tf2_ros import Buffer, TransformListener
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "ROS 2 Python interfaces are unavailable. "
        "Source /opt/ros/humble/setup.bash and the robot driver workspace first."
    ) from exc


class ROS2JointTrajectoryRobotAdapter:
    """Minimal ROS 2 robot adapter used by the hand-eye workflow.

    The robot driver is expected to already be running.

    Common ROS interfaces:
      - /joint_states
      - TF: BASE_FRAME -> TCP_FRAME
      - FollowJointTrajectory action

    Units exposed to the hand-eye workflow:
      - joints: degrees
      - T_base_tcp translation: millimetres
    """

    JOINT_NAMES: tuple[str, ...] = ()
    BASE_FRAME: str = ""
    TCP_FRAME: str = ""
    JOINT_STATE_TOPIC: str = "/joint_states"
    TRAJECTORY_ACTION: str = "/joint_trajectory_controller/follow_joint_trajectory"
    TRAJECTORY_ACTIONS: tuple[str, ...] = ()
    JOINT_STATE_MAX_AGE_S: float = 1.0

    def __init__(self, host: str = "", port: int | None = None) -> None:
        # Kept only for compatibility with existing WorkflowController(host, port).
        self.host = host
        self.port = port

        if len(self.JOINT_NAMES) != 6:
            raise RuntimeError(
                f"{type(self).__name__}.JOINT_NAMES must contain exactly 6 joints"
            )
        if not self.BASE_FRAME or not self.TCP_FRAME:
            raise RuntimeError(
                f"{type(self).__name__} must define BASE_FRAME and TCP_FRAME"
            )

        self._node: Node | None = None
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._action_client: ActionClient | None = None
        self._tf_buffer: Buffer | None = None
        self._tf_listener: TransformListener | None = None
        self.trajectory_action = self.TRAJECTORY_ACTION

        self._owns_rclpy = False
        self._connected = False

        self._joint_lock = threading.Lock()
        self._joint_positions_rad: np.ndarray | None = None
        self._joint_stamp_monotonic = 0.0

        self._goal_lock = threading.Lock()
        self._active_goal = None

    # ------------------------------------------------------------------
    # ROS lifecycle
    # ------------------------------------------------------------------

    def connect(self, *, startup_timeout_s: float = 15.0, **_ignored) -> None:
        """Connect to an already-running ROS 2 robot driver."""
        if self._connected:
            return

        try:
            with self._joint_lock:
                self._joint_positions_rad = None
                self._joint_stamp_monotonic = 0.0

            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True

            node_name = f"laser_handeye_{type(self).__name__.lower()}"
            self._node = Node(node_name)

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(
                self._tf_buffer,
                self._node,
                spin_thread=False,
            )

            self._node.create_subscription(
                JointState,
                self.JOINT_STATE_TOPIC,
                self._joint_state_callback,
                20,
            )

            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name=f"{node_name}_spin",
                daemon=True,
            )
            self._spin_thread.start()

            startup_timeout_s = float(startup_timeout_s)
            if startup_timeout_s <= 0:
                raise ValueError("startup_timeout_s must be positive")
            deadline = time.monotonic() + startup_timeout_s
            candidates = self.TRAJECTORY_ACTIONS or (self.TRAJECTORY_ACTION,)
            for action_name in candidates:
                candidate = ActionClient(
                    self._node,
                    FollowJointTrajectory,
                    action_name,
                )
                remaining = max(0.1, deadline - time.monotonic())
                per_candidate = max(0.1, remaining / max(1, len(candidates)))
                if candidate.wait_for_server(timeout_sec=per_candidate):
                    self._action_client = candidate
                    self.trajectory_action = action_name
                    break
                candidate.destroy()
            if self._action_client is None:
                raise ConnectionError(
                    "FollowJointTrajectory action server is unavailable; tried "
                    f"{list(candidates)}. "
                    "Start the robot ROS 2 driver/controller first."
                )

            remaining = max(0.1, deadline - time.monotonic())
            self._wait_for_joint_state(timeout_s=min(remaining, startup_timeout_s / 2.0))
            remaining = max(0.1, deadline - time.monotonic())
            self._wait_for_tf(timeout_s=remaining)
            self._connected = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._connected = False

        try:
            self.stop()
        except Exception:
            pass

        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass

        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                pass

        if self._action_client is not None:
            try:
                self._action_client.destroy()
            except Exception:
                pass

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass

        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        self._tf_listener = None
        self._tf_buffer = None
        self._action_client = None
        self._executor = None
        self._node = None
        self._spin_thread = None

        # Do not shutdown a ROS context initialized by MoveIt / another adapter.
        if self._owns_rclpy and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        self._owns_rclpy = False

    def _require_connected(self) -> None:
        if not self._connected or self._node is None:
            raise RuntimeError("RobotAdapter is not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Joint states
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in self.JOINT_NAMES):
            return
        try:
            q = np.asarray(
                [msg.position[index[name]] for name in self.JOINT_NAMES],
                dtype=float,
            )
        except (IndexError, TypeError, ValueError):
            return
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return
        with self._joint_lock:
            self._joint_positions_rad = q
            self._joint_stamp_monotonic = time.monotonic()

    def _wait_for_joint_state(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._joint_lock:
                ready = self._joint_positions_rad is not None
            if ready:
                return
            time.sleep(0.02)
        raise TimeoutError(
            f"No complete joint state received on {self.JOINT_STATE_TOPIC}; "
            f"expected joints={list(self.JOINT_NAMES)}"
        )

    def read_joint_positions_deg(
        self,
        *,
        prefer_reference: bool = False,
    ) -> np.ndarray:
        del prefer_reference  # ROS /joint_states is the measured state.
        self._require_connected()
        with self._joint_lock:
            if self._joint_positions_rad is None:
                raise RuntimeError("No joint state has been received")
            q = self._joint_positions_rad.copy()
            age_s = time.monotonic() - self._joint_stamp_monotonic
        if age_s > self.JOINT_STATE_MAX_AGE_S:
            raise RuntimeError(
                f"Joint state is stale ({age_s:.2f} s old) on "
                f"{self.JOINT_STATE_TOPIC}"
            )
        return np.rad2deg(q)

    # Compatibility alias if another caller uses this spelling.
    def read_joints_deg(self) -> np.ndarray:
        return self.read_joint_positions_deg()

    def read_joint_positions(self) -> np.ndarray:
        """Return joints in degrees, matching this adapter's public contract."""
        return self.read_joint_positions_deg()

    def read_state_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return measured joints and TCP for one GUI/workflow sampling cycle.

        ROS joint states and TF are separate streams, so this cannot make them
        perfectly timestamp-identical.  It does keep the public workflow on one
        adapter call and lets robot-specific adapters override it with a truly
        atomic controller snapshot when available.
        """
        joints = self.read_joint_positions_deg()
        transform = self.read_T_base_tcp()
        return joints, transform

    # ------------------------------------------------------------------
    # TF -> T_base_tcp
    # ------------------------------------------------------------------

    def _wait_for_tf(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._lookup_transform(timeout_s=0.2)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        raise TimeoutError(
            f"TF unavailable: {self.BASE_FRAME} -> {self.TCP_FRAME}. "
            f"Last error: {last_error}"
        )

    def _lookup_transform(self, timeout_s: float = 0.5):
        if self._tf_buffer is None:
            raise RuntimeError("TF buffer is not initialized")
        return self._tf_buffer.lookup_transform(
            self.BASE_FRAME,
            self.TCP_FRAME,
            Time(),
            timeout=Duration(seconds=float(timeout_s)),
        )

    def read_T_base_tcp(self) -> np.ndarray:
        self._require_connected()
        stamped = self._lookup_transform(timeout_s=0.5)
        t = stamped.transform.translation
        q = stamped.transform.rotation

        T = np.eye(4, dtype=float)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        # ROS TF translation is metres; workflow uses millimetres.
        T[:3, 3] = 1000.0 * np.asarray([t.x, t.y, t.z], dtype=float)
        return T

    # ------------------------------------------------------------------
    # Joint motion
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_joint_deg(value: Sequence[float]) -> np.ndarray:
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(
                "target_joint_deg must contain six finite joint angles"
            )
        return q.copy()

    @staticmethod
    def _angular_error_deg(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (current - target + 180.0) % 360.0 - 180.0

    @staticmethod
    def _duration_msg(seconds: float):
        # builtin_interfaces/Duration without importing another symbol.
        whole = int(math.floor(seconds))
        nano = int(round((seconds - whole) * 1e9))
        if nano >= 1_000_000_000:
            whole += 1
            nano -= 1_000_000_000
        return whole, nano

    def _wait_future(self, future, timeout_s: float, description: str):
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        if not event.wait(timeout_s):
            raise TimeoutError(f"Timed out waiting for {description}")
        return future.result()

    @staticmethod
    def _duration_seconds(duration) -> float:
        return float(duration.sec) + 1e-9 * float(duration.nanosec)

    @staticmethod
    def _validate_timed_trajectory(
        positions_rad: np.ndarray,
        time_from_start_s: np.ndarray,
        velocities_rad_s: np.ndarray | None,
        accelerations_rad_s2: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        positions = np.asarray(positions_rad, dtype=float)
        times = np.asarray(time_from_start_s, dtype=float).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) == 0:
            raise ValueError("positions_rad must have shape (N, 6) with N >= 1")
        if times.shape != (len(positions),):
            raise ValueError("time_from_start_s must contain one value per trajectory point")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(times)):
            raise ValueError("trajectory positions/times must be finite")
        if np.any(times < 0.0):
            raise ValueError("time_from_start_s must be non-negative")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("time_from_start_s must be strictly increasing")
        if times[-1] <= 0.0:
            raise ValueError("trajectory duration must be positive")

        def optional_array(value, label):
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            if array.size == 0:
                return None
            if array.shape != positions.shape:
                raise ValueError(f"{label} must have shape {positions.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{label} must be finite")
            return array.copy()

        return (
            positions.copy(),
            times.copy(),
            optional_array(velocities_rad_s, "velocities_rad_s"),
            optional_array(accelerations_rad_s2, "accelerations_rad_s2"),
        )

    def execute_joint_trajectory(
        self,
        positions_rad: np.ndarray,
        time_from_start_s: np.ndarray,
        *,
        velocities_rad_s: np.ndarray | None = None,
        accelerations_rad_s2: np.ndarray | None = None,
        tolerance_deg: float = 0.5,
        path_tolerance_deg: float | None = None,
        timeout_s: float = 60.0,
        stable_count: int = 3,
        poll_interval_s: float = 0.05,
        progress_callback=None,
    ) -> np.ndarray:
        """Execute one complete MoveIt trajectory as one ROS action goal.

        ``time_from_start_s`` and optional velocity/acceleration arrays are the
        values produced by MoveIt time parameterization.  The ROS trajectory
        controller owns interpolation and tracking between waypoints.
        """
        self._require_connected()
        if self._action_client is None:
            raise RuntimeError("Trajectory action client is unavailable")
        if tolerance_deg <= 0.0 or timeout_s <= 0.0:
            raise ValueError("tolerance_deg and timeout_s must be positive")
        if path_tolerance_deg is not None and path_tolerance_deg <= 0.0:
            raise ValueError("path_tolerance_deg must be positive when specified")

        positions, times, velocities, accelerations = self._validate_timed_trajectory(
            positions_rad,
            time_from_start_s,
            velocities_rad_s,
            accelerations_rad_s2,
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.JOINT_NAMES)
        for index, q in enumerate(positions):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            if velocities is not None:
                point.velocities = velocities[index].tolist()
            if accelerations is not None:
                point.accelerations = accelerations[index].tolist()
            sec, nanosec = self._duration_msg(float(times[index]))
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            goal.trajectory.points.append(point)

        # Explicitly enforce the same endpoint tolerance in the ROS trajectory
        # controller. An empty goal_tolerance delegates to controller defaults,
        # which can otherwise be looser than the workflow's measured-state gate.
        tolerance_rad = float(np.deg2rad(tolerance_deg))
        goal.goal_tolerance = []
        for name in self.JOINT_NAMES:
            tolerance = JointTolerance()
            tolerance.name = name
            tolerance.position = tolerance_rad
            # Zero means "unspecified/default" for velocity/acceleration; only
            # endpoint position accuracy is tightened here.
            tolerance.velocity = 0.0
            tolerance.acceleration = 0.0
            goal.goal_tolerance.append(tolerance)

        if path_tolerance_deg is not None:
            path_tolerance_rad = float(np.deg2rad(path_tolerance_deg))
            goal.path_tolerance = []
            for name in self.JOINT_NAMES:
                tolerance = JointTolerance()
                tolerance.name = name
                tolerance.position = path_tolerance_rad
                tolerance.velocity = 0.0
                tolerance.acceleration = 0.0
                goal.path_tolerance.append(tolerance)

        total_time = float(times[-1])
        started_at = time.monotonic()

        def on_feedback(message) -> None:
            if progress_callback is None:
                return
            try:
                feedback = message.feedback
                trajectory_time = self._duration_seconds(feedback.desired.time_from_start)
                if trajectory_time <= 0.0:
                    trajectory_time = min(total_time, time.monotonic() - started_at)
                fraction = float(np.clip(trajectory_time / total_time, 0.0, 1.0))
                completed = int(
                    np.clip(
                        np.searchsorted(times, trajectory_time, side="right") - 1,
                        0,
                        max(0, len(times) - 1),
                    )
                )
                progress_callback(
                    {
                        "fraction": fraction,
                        "trajectory_time_s": trajectory_time,
                        "remaining_s": max(0.0, total_time - trajectory_time),
                        "completed_segments": completed,
                        "total_segments": max(0, len(times) - 1),
                    }
                )
            except Exception:
                # Feedback is diagnostic only; never interrupt robot execution.
                pass

        send_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=on_feedback,
        )
        goal_handle = self._wait_future(
            send_future,
            min(5.0, timeout_s),
            "trajectory goal acceptance",
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(
                f"Trajectory goal rejected by {self.trajectory_action}"
            )

        with self._goal_lock:
            self._active_goal = goal_handle

        result_future = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(
                result_future,
                timeout_s,
                "trajectory execution",
            )
        except TimeoutError:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, 2.0, "timed-out trajectory cancellation")
            finally:
                raise
        finally:
            with self._goal_lock:
                self._active_goal = None

        if wrapped is None:
            raise RuntimeError("Trajectory controller returned no result")
        if int(wrapped.status) != int(GoalStatus.STATUS_SUCCEEDED):
            raise RuntimeError(
                "FollowJointTrajectory action did not succeed: "
                f"status={wrapped.status}"
            )
        result = wrapped.result
        if int(result.error_code) != 0:
            raise RuntimeError(
                "FollowJointTrajectory failed: "
                f"error_code={result.error_code}, "
                f"error_string={result.error_string!r}"
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "fraction": 1.0,
                    "trajectory_time_s": total_time,
                    "remaining_s": 0.0,
                    "completed_segments": max(0, len(times) - 1),
                    "total_segments": max(0, len(times) - 1),
                }
            )

        final_target_deg = np.rad2deg(positions[-1])
        return self.wait_until_joint_reached(
            final_target_deg,
            tolerance_deg=tolerance_deg,
            timeout_s=max(2.0, min(15.0, timeout_s)),
            stable_count=stable_count,
            poll_interval_s=poll_interval_s,
        )

    def move_j(
        self,
        target_joint_deg: Sequence[float],
        *,
        speed_deg_s: float = 10.0,
        accel_deg_s2: float = 20.0,
        tolerance_deg: float = 0.5,
        timeout_s: float = 45.0,
        stable_count: int = 3,
        poll_interval_s: float = 0.05,
        prefer_reference: bool = False,
    ) -> np.ndarray:
        del prefer_reference
        self._require_connected()
        if self._action_client is None:
            raise RuntimeError("Trajectory action client is unavailable")
        if speed_deg_s <= 0 or accel_deg_s2 <= 0:
            raise ValueError("speed_deg_s and accel_deg_s2 must be positive")

        target_deg = self._validate_joint_deg(target_joint_deg)
        current_deg = self.read_joint_positions_deg()
        # MoveIt waypoints are explicit joint coordinates.  Do not wrap a
        # 360-degree difference to zero when computing a safe duration.
        max_delta = float(np.max(np.abs(target_deg - current_deg)))

        # Conservative trapezoidal/triangular timing for the slowest joint.
        # A single-point controller trajectory has no explicit acceleration
        # profile, so time_from_start must be long enough for both limits.
        speed = float(speed_deg_s)
        accel = float(accel_deg_s2)
        accel_distance = speed * speed / accel
        if max_delta <= accel_distance:
            minimum_duration = 2.0 * math.sqrt(max_delta / accel)
        else:
            minimum_duration = 2.0 * speed / accel + (max_delta - accel_distance) / speed
        duration_s = max(0.6, 1.20 * minimum_duration)
        if duration_s >= timeout_s:
            raise ValueError(
                "timeout_s is too short for the requested joint speed/acceleration: "
                f"need > {duration_s:.2f} s, got {timeout_s:.2f} s"
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = np.deg2rad(target_deg).tolist()
        sec, nanosec = self._duration_msg(duration_s)
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = nanosec
        goal.trajectory.points = [point]

        send_future = self._action_client.send_goal_async(goal)
        goal_handle = self._wait_future(
            send_future,
            min(5.0, timeout_s),
            "trajectory goal acceptance",
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(
                f"Trajectory goal rejected by {self.trajectory_action}"
            )

        with self._goal_lock:
            self._active_goal = goal_handle

        result_future = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(
                result_future,
                timeout_s,
                "trajectory execution",
            )
        except TimeoutError:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, 2.0, "timed-out trajectory cancellation")
            finally:
                raise
        finally:
            with self._goal_lock:
                self._active_goal = None

        if wrapped is None:
            raise RuntimeError("Trajectory controller returned no result")
        result = wrapped.result
        # FollowJointTrajectory.SUCCESSFUL == 0
        if int(result.error_code) != 0:
            raise RuntimeError(
                "FollowJointTrajectory failed: "
                f"error_code={result.error_code}, "
                f"error_string={result.error_string!r}"
            )

        return self.wait_until_joint_reached(
            target_deg,
            tolerance_deg=tolerance_deg,
            timeout_s=max(2.0, timeout_s - duration_s),
            stable_count=stable_count,
            poll_interval_s=poll_interval_s,
        )

    def wait_until_joint_reached(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float = 0.5,
        timeout_s: float = 15.0,
        stable_count: int = 3,
        poll_interval_s: float = 0.05,
        prefer_reference: bool = False,
    ) -> np.ndarray:
        del prefer_reference
        target = self._validate_joint_deg(target_joint_deg)
        if tolerance_deg <= 0 or timeout_s <= 0:
            raise ValueError("tolerance_deg and timeout_s must be positive")

        deadline = time.monotonic() + timeout_s
        count = 0
        last = None
        last_error = math.inf
        while time.monotonic() < deadline:
            current = self.read_joint_positions_deg()
            last = current
            # These are explicit limited-joint coordinates from MoveIt.  A
            # one-turn difference is not interchangeable with the requested
            # winding, even when the end-effector orientation is equivalent.
            err = np.abs(current - target)
            last_error = float(np.max(err))
            count = count + 1 if last_error <= tolerance_deg else 0
            if count >= stable_count:
                return current
            time.sleep(poll_interval_s)

        raise TimeoutError(
            "Joint target was not reached: "
            f"max_error={last_error:.3f} deg, "
            f"target={target.tolist()}, "
            f"current={None if last is None else last.tolist()}"
        )

    def stop(self) -> None:
        """Cancel the active FollowJointTrajectory goal, if any."""
        with self._goal_lock:
            goal_handle = self._active_goal
        if goal_handle is None:
            return
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_future(cancel_future, 2.0, "trajectory cancellation")
        finally:
            with self._goal_lock:
                self._active_goal = None

    # ------------------------------------------------------------------
    # Cartesian compatibility
    # ------------------------------------------------------------------

    def move_l(self, *args, **kwargs):
        raise NotImplementedError(
            "Direct Cartesian move_l is intentionally not implemented in the "
            "minimal ROS adapter. This project should generate collision-checked "
            "motion with MoveIt and execute its complete timed joint trajectory."
        )
