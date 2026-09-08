"""MoveIt 2 service planner for joint-space and Cartesian paths.

Goal
----
Plan directly from the current robot joints to a desired
``sensor_physical_origin`` pose:

    current robot joints
        -> collision-aware MoveIt /compute_ik (seeded by current joints)
        -> one joint goal near the seed (TRAC-IK Distance)
        -> OMPL RRTConnect
        -> collision-free joint trajectory

The target ``sensor_physical_origin`` pose is first converted to one
collision-free joint goal by MoveIt's IK service. OMPL then plans only in
joint space from the supplied start state to that fixed goal.

Collision geometry comes from the MoveIt robot model (RB5 + sensor URDF/SRDF).
World collision objects added here are the measured calibration plane and floor.

The IK request uses the supplied start joints as its seed and keeps
``avoid_collisions=True`` so the configured TRAC-IK Distance mode can prefer a
nearby collision-free branch before RRTConnect runs.

Drop-in API for the existing workflow.py:

    with MoveItServicePlanner(settings) as moveit:
        moveit.apply_scene(T_base_plane_mm, clearance_mm=10.0)
        result = moveit.plan_to_pose(joint_names, q_start_rad, T_base_physical_mm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class MoveItUnavailableError(RuntimeError):
    """Raised when the configured MoveIt environment cannot be used."""


@dataclass(frozen=True)
class MoveItPath:
    success: bool
    reason: str
    joint_names: tuple[str, ...]
    positions_rad: np.ndarray
    time_from_start_s: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=float)
    )
    velocities_rad_s: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    accelerations_rad_s2: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    error_code: int = 0
    planning_time_s: float | None = None


class MoveItServicePlanner:
    """Minimal Python wrapper around MoveIt's own IK + OMPL services."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)
        self.node = None
        self.launch_process: subprocess.Popen | None = None
        self._owns_rclpy = False
        self.ik_client = None
        self.plan_client = None
        self.cartesian_client = None
        self.scene_client = None
        self.state_validity_client = None
        self._import_ros()

    # ------------------------------------------------------------------
    # Basic configuration
    # ------------------------------------------------------------------
    @property
    def planning_frame(self) -> str:
        return str(self.settings.get("planning_frame", "link0"))

    @property
    def group_name(self) -> str:
        return str(self.settings.get("group_name", "rb5_arm"))

    @property
    def tip_link(self) -> str:
        return str(
            self.settings.get(
                "tip_link",
                self.settings.get("physical_link", "sensor_physical_origin"),
            )
        )

    @property
    def pipeline_id(self) -> str:
        return str(self.settings.get("pipeline_id", "ompl"))

    @property
    def planner_id(self) -> str:
        return str(self.settings.get("planner_id", "RRTConnectkConfigDefault"))

    # ------------------------------------------------------------------
    # ROS / MoveIt lifecycle
    # ------------------------------------------------------------------
    def _import_ros(self) -> None:
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from geometry_msgs.msg import Pose, PoseStamped
            from moveit_msgs.msg import (
                CollisionObject,
                Constraints,
                JointConstraint,
                PlanningScene,
                RobotState,
            )
            from moveit_msgs.srv import (
                ApplyPlanningScene,
                GetCartesianPath,
                GetMotionPlan,
                GetPositionIK,
                GetStateValidity,
            )
            from sensor_msgs.msg import JointState
            from shape_msgs.msg import SolidPrimitive
        except ImportError as error:
            raise MoveItUnavailableError(
                "ROS 2 / MoveIt Python message interfaces are unavailable. "
                "Source /opt/ros/humble/setup.bash and the MoveIt workspace first."
            ) from error

        self.ros = locals()

    def _sourced_shell(self, command: str, *, timeout_s: float = 15.0) -> subprocess.CompletedProcess:
        """Run a ROS-aware shell command in the exact MoveIt overlay used here."""
        workspace_setup = Path(
            str(self.settings.get("workspace_setup", ""))
        ).expanduser().resolve()
        ros_setup = Path(
            str(self.settings.get("ros_setup", "/opt/ros/humble/setup.bash"))
        ).expanduser()

        if not workspace_setup.exists():
            raise MoveItUnavailableError(
                f"MoveIt workspace setup does not exist: {workspace_setup}"
            )
        if not ros_setup.exists():
            raise MoveItUnavailableError(f"ROS setup does not exist: {ros_setup}")

        shell = (
            f"source {shlex.quote(str(ros_setup))} && "
            f"source {shlex.quote(str(workspace_setup))} && "
            f"{command}"
        )
        return subprocess.run(
            ["bash", "-lc", shell],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )

    def _move_group_nodes(self) -> tuple[str, ...]:
        """Return active move_group node names in the current ROS domain."""
        result = self._sourced_shell("ros2 node list 2>/dev/null || true")
        return tuple(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().rstrip("/").endswith("move_group")
        )

    def _existing_move_group_matches(self) -> tuple[bool, str]:
        """Check whether the currently running /move_group matches this planner.

        A matching MoveGroup is intentionally reused.  This matters for SIM/REAL
        controllers that already launched the selected robot's MoveGroup during
        Connect.  Only a foreign/stale robot model is removed.
        """
        existing = self._move_group_nodes()
        if not existing:
            return False, "no active move_group"

        try:
            robot_description = self._ros_param_text("robot_description")
            semantic = self._ros_param_text("robot_description_semantic")
        except Exception as exc:
            return False, f"cannot inspect active move_group: {exc}"

        expected_robot = str(self.settings.get("expected_robot_name", "")).strip()
        if expected_robot:
            actual_match = re.search(
                r"<robot\b[^>]*\bname=[\"']([^\"']+)[\"']",
                robot_description,
            )
            actual_robot = actual_match.group(1) if actual_match else "unknown"
            if actual_robot != expected_robot:
                return False, (
                    f"robot model mismatch: expected {expected_robot!r}, "
                    f"got {actual_robot!r}"
                )

        group_pattern = re.compile(
            r"<group\b[^>]*\bname=[\"']"
            + re.escape(self.group_name)
            + r"[\"']"
        )
        if group_pattern.search(semantic) is None:
            return False, f"planning group {self.group_name!r} is missing"

        link_pattern = re.compile(
            r"<link\b[^>]*\bname=[\"']"
            + re.escape(self.planning_frame)
            + r"[\"']"
        )
        if link_pattern.search(robot_description) is None:
            return False, f"planning frame {self.planning_frame!r} is missing"

        tip_pattern = re.compile(
            r"<link\b[^>]*\bname=[\"']"
            + re.escape(self.tip_link)
            + r"[\"']"
        )
        if tip_pattern.search(robot_description) is None:
            return False, f"tip link {self.tip_link!r} is missing"

        return True, "active move_group matches selected robot"

    def _terminate_foreign_move_group(self) -> None:
        """Remove a wrong robot's MoveGroup from this ROS domain."""
        existing = self._move_group_nodes()
        if not existing:
            return

        cleanup = r"""
pkill -INT -f '[m]oveit_ros_move_group/move_group' 2>/dev/null || true
pkill -INT -f '[r]os2 launch rb5_laser_moveit_config move_group.launch.py' 2>/dev/null || true
pkill -INT -f '[r]os2 launch ur5e_laser_moveit_config move_group.launch.py' 2>/dev/null || true
sleep 0.8
pkill -TERM -f '[m]oveit_ros_move_group/move_group' 2>/dev/null || true
pkill -TERM -f '[r]os2 launch rb5_laser_moveit_config move_group.launch.py' 2>/dev/null || true
pkill -TERM -f '[r]os2 launch ur5e_laser_moveit_config move_group.launch.py' 2>/dev/null || true
"""
        self._sourced_shell(cleanup, timeout_s=8.0)

        deadline = time.monotonic() + float(
            self.settings.get("stale_shutdown_timeout_s", 5.0)
        )
        while time.monotonic() < deadline:
            if not self._move_group_nodes():
                return
            time.sleep(0.2)

        remaining = self._move_group_nodes()
        raise MoveItUnavailableError(
            "wrong/stale move_group is active and could not be stopped: "
            + ", ".join(remaining)
        )

    def _ros_param_text(self, parameter: str) -> str:
        result = self._sourced_shell(
            f"ros2 param get /move_group {shlex.quote(parameter)}",
            timeout_s=float(self.settings.get("startup_timeout_s", 25.0)),
        )
        if result.returncode != 0:
            raise MoveItUnavailableError(
                f"cannot read /move_group parameter {parameter!r}: "
                f"{result.stdout.strip()}"
            )
        return result.stdout

    def _verify_loaded_model(self) -> None:
        """Fail fast if MoveIt loaded the wrong robot model/configuration."""
        robot_description = self._ros_param_text("robot_description")
        semantic = self._ros_param_text("robot_description_semantic")

        expected_robot = str(self.settings.get("expected_robot_name", "")).strip()
        if expected_robot:
            robot_pattern = re.compile(
                r"<robot\b[^>]*\bname=[\"']"
                + re.escape(expected_robot)
                + r"[\"']"
            )
            if robot_pattern.search(robot_description) is None:
                actual = re.search(
                    r"<robot\b[^>]*\bname=[\"']([^\"']+)[\"']",
                    robot_description,
                )
                actual_name = actual.group(1) if actual else "unknown"
                raise MoveItUnavailableError(
                    "wrong MoveIt robot model is active: "
                    f"expected {expected_robot!r}, got {actual_name!r}. "
                    "A stale/foreign move_group was detected or the launch package "
                    "does not match the selected robot."
                )

        group_pattern = re.compile(
            r"<group\b[^>]*\bname=[\"']"
            + re.escape(self.group_name)
            + r"[\"']"
        )
        if group_pattern.search(semantic) is None:
            raise MoveItUnavailableError(
                "MoveIt SRDF does not contain the configured planning group "
                f"{self.group_name!r}"
            )

        # Verify frame/link names before any PlanningScene or IK request is sent.
        link_pattern = re.compile(
            r"<link\b[^>]*\bname=[\"']"
            + re.escape(self.planning_frame)
            + r"[\"']"
        )
        frame_in_semantic = (
            f'base_link="{self.planning_frame}"' in semantic
            or f"base_link='{self.planning_frame}'" in semantic
            or f'child_link="{self.planning_frame}"' in semantic
            or f"child_link='{self.planning_frame}'" in semantic
        )
        if link_pattern.search(robot_description) is None and not frame_in_semantic:
            raise MoveItUnavailableError(
                "configured MoveIt planning_frame is not present in the loaded model: "
                f"{self.planning_frame!r}"
            )

        tip_pattern = re.compile(
            r"<link\b[^>]*\bname=[\"']"
            + re.escape(self.tip_link)
            + r"[\"']"
        )
        if tip_pattern.search(robot_description) is None:
            raise MoveItUnavailableError(
                "configured MoveIt tip_link is not present in the loaded model: "
                f"{self.tip_link!r}"
            )

    def _launch_move_group(self) -> None:
        if not bool(self.settings.get("auto_launch", True)):
            return

        # A controller may already have launched the selected robot's MoveGroup
        # during Connect. Reuse it if and only if model/group/frame/tip match.
        # This prevents us from killing our own correct MoveGroup at Plan time.
        existing = self._move_group_nodes()
        if existing:
            matches, detail = self._existing_move_group_matches()
            if matches:
                self.launch_process = None
                return

            if not bool(self.settings.get("exclusive_move_group", True)):
                raise MoveItUnavailableError(
                    "a foreign move_group is already active: " + detail
                )
            self._terminate_foreign_move_group()

        workspace_setup = Path(
            str(self.settings.get("workspace_setup", ""))
        ).expanduser().resolve()
        if not workspace_setup.exists():
            raise MoveItUnavailableError(
                f"MoveIt workspace setup does not exist: {workspace_setup}"
            )

        ros_setup = Path(
            str(self.settings.get("ros_setup", "/opt/ros/humble/setup.bash"))
        ).expanduser()
        if not ros_setup.exists():
            raise MoveItUnavailableError(f"ROS setup does not exist: {ros_setup}")

        launch_package = str(
            self.settings.get("launch_package", "rb5_laser_moveit_config")
        )
        launch_file = str(self.settings.get("launch_file", "move_group.launch.py"))

        launch_arguments = []
        for key, value in dict(self.settings.get("launch_arguments", {})).items():
            if not str(key).replace("_", "").isalnum():
                raise MoveItUnavailableError(
                    f"invalid MoveIt launch argument name: {key!r}"
                )
            if isinstance(value, bool):
                value = "true" if value else "false"
            launch_arguments.append(
                shlex.quote(f"{key}:={value}")
            )
        suffix = "" if not launch_arguments else " " + " ".join(launch_arguments)

        command = (
            f"source {shlex.quote(str(ros_setup))} && "
            f"source {shlex.quote(str(workspace_setup))} && "
            f"exec ros2 launch {shlex.quote(launch_package)} "
            f"{shlex.quote(launch_file)}{suffix}"
        )

        self.launch_process = subprocess.Popen(
            ["bash", "-c", command],
            env=os.environ.copy(),
            start_new_session=True,
        )

    def __enter__(self) -> "MoveItServicePlanner":
        rclpy = self.ros["rclpy"]
        try:
            self._launch_move_group()

            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True

            self.node = rclpy.create_node(f"handeye_moveit_{os.getpid()}")

            self.ik_client = self.node.create_client(
                self.ros["GetPositionIK"], "/compute_ik"
            )
            self.plan_client = self.node.create_client(
                self.ros["GetMotionPlan"], "/plan_kinematic_path"
            )
            self.cartesian_client = self.node.create_client(
                self.ros["GetCartesianPath"], "/compute_cartesian_path"
            )
            self.scene_client = self.node.create_client(
                self.ros["ApplyPlanningScene"], "/apply_planning_scene"
            )
            self.state_validity_client = self.node.create_client(
                self.ros["GetStateValidity"], "/check_state_validity"
            )

            startup_timeout = float(self.settings.get("startup_timeout_s", 25.0))
            for name, client in (
                ("/compute_ik", self.ik_client),
                ("/plan_kinematic_path", self.plan_client),
                ("/compute_cartesian_path", self.cartesian_client),
                ("/apply_planning_scene", self.scene_client),
                ("/check_state_validity", self.state_validity_client),
            ):
                if not client.wait_for_service(timeout_sec=startup_timeout):
                    detail = ""
                    if (
                        self.launch_process is not None
                        and self.launch_process.poll() is not None
                    ):
                        detail = f" move_group exited with code {self.launch_process.returncode}."
                    raise MoveItUnavailableError(
                        f"MoveIt service {name} did not start.{detail}"
                    )

            # Do not send a single IK/PlanningScene request until the launched
            # model, SRDF group, frame, and tip are confirmed to match config.
            self._verify_loaded_model()
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

        if self._owns_rclpy:
            self.ros["rclpy"].shutdown()
            self._owns_rclpy = False

        if self.launch_process is not None:
            if self.launch_process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.launch_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.launch_process.wait(timeout=2.0)
            self.launch_process = None

    def __exit__(self, *_args) -> None:
        self.close()

    def _call(self, client, request, timeout_s: float):
        if self.node is None:
            raise RuntimeError("MoveItServicePlanner is not active")

        service_name = getattr(client, "srv_name", "<unknown>")

        future = client.call_async(request)
        self.ros["rclpy"].spin_until_future_complete(
            self.node,
            future,
            timeout_sec=float(timeout_s),
        )

        if not future.done() or future.result() is None:
            raise TimeoutError(
                f"MoveIt service {service_name} timed out "
                f"after {float(timeout_s):.1f} s"
            )

        return future.result()

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------
    def _robot_state(self, names: tuple[str, ...], positions: np.ndarray):
        positions = np.asarray(positions, dtype=float).reshape(-1)
        if len(names) != len(positions):
            raise ValueError("joint name count does not match joint position count")
        if not np.all(np.isfinite(positions)):
            raise ValueError("joint positions contain non-finite values")

        state = self.ros["RobotState"]()
        state.joint_state = self.ros["JointState"]()
        state.joint_state.name = list(names)
        state.joint_state.position = positions.tolist()
        return state

    def _pose_mm(self, transform_mm: np.ndarray):
        transform = np.asarray(transform_mm, dtype=float).copy()
        if transform.shape != (4, 4):
            raise ValueError("transform must have shape (4, 4)")
        if not np.all(np.isfinite(transform)):
            raise ValueError("transform contains non-finite values")

        pose = self.ros["Pose"]()
        pose.position.x = float(transform[0, 3] * 1e-3)
        pose.position.y = float(transform[1, 3] * 1e-3)
        pose.position.z = float(transform[2, 3] * 1e-3)
        qx, qy, qz, qw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose

    @staticmethod
    def _duration_message(DurationType, seconds: float):
        seconds = max(0.0, float(seconds))
        whole = int(seconds)
        nanos = int(round((seconds - whole) * 1e9))
        if nanos >= 1_000_000_000:
            whole += 1
            nanos -= 1_000_000_000
        return DurationType(sec=whole, nanosec=nanos)

    @staticmethod
    def _duration_to_seconds(duration) -> float:
        return float(duration.sec) + float(duration.nanosec) * 1e-9

    @staticmethod
    def _empty_path(
        names: tuple[str, ...],
        *,
        reason: str,
        error_code: int,
        planning_time_s: float | None = None,
    ) -> MoveItPath:
        return MoveItPath(
            success=False,
            reason=reason,
            joint_names=names,
            positions_rad=np.empty((0, len(names)), dtype=float),
            error_code=error_code,
            planning_time_s=planning_time_s,
        )

    # ------------------------------------------------------------------
    # Planning scene: one large thin plane box
    # ------------------------------------------------------------------
    def apply_scene(
        self,
        T_world_plane_mm: np.ndarray,
        *,
        clearance_mm: float,
        plane_size_xy_mm: np.ndarray | tuple[float, float] | list[float] | None = None,
    ) -> None:
        """Add the calibration plane and floor as world collision objects.

        The plane frame's +Z/-Z axis is its normal.  A large thin box is used
        instead of a half-space.  ``clearance_mm`` simply thickens the plane so
        paths keep a small distance from it on either side.
        """
        transform = np.asarray(T_world_plane_mm, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("plane transform must be a finite 4x4 matrix")

        if plane_size_xy_mm is None:
            size = self.settings.get("plane_size_xy_m", [4.0, 4.0])
            if np.isscalar(size):
                sx = sy = float(size)
            else:
                values = np.asarray(size, dtype=float).reshape(-1)
                if values.size != 2:
                    raise ValueError("plane_size_xy_m must be scalar or [x, y]")
                sx, sy = map(float, values)
        else:
            values_mm = np.asarray(plane_size_xy_mm, dtype=float).reshape(-1)
            if values_mm.size != 2 or not np.all(np.isfinite(values_mm)):
                raise ValueError("plane_size_xy_mm must contain two finite values")
            if np.any(values_mm <= 0.0):
                raise ValueError("plane_size_xy_mm values must be positive")
            sx, sy = map(float, values_mm * 1e-3)

        physical_thickness = float(self.settings.get("plane_thickness_m", 0.01))
        thickness = physical_thickness + 2.0 * max(0.0, float(clearance_mm)) * 1e-3
        if sx <= 0 or sy <= 0 or thickness <= 0:
            raise ValueError("plane collision dimensions must be positive")

        plane_obj = self.ros["CollisionObject"]()
        plane_obj.id = "calibration_plane"
        plane_obj.header.frame_id = self.planning_frame

        plane_primitive = self.ros["SolidPrimitive"]()
        plane_primitive.type = self.ros["SolidPrimitive"].BOX
        plane_primitive.dimensions = [sx, sy, thickness]

        plane_obj.primitives = [plane_primitive]
        plane_obj.primitive_poses = [self._pose_mm(transform)]
        plane_obj.operation = self.ros["CollisionObject"].ADD

        floor_obj = self.ros["CollisionObject"]()
        floor_obj.id = "floor"
        floor_obj.header.frame_id = self.planning_frame

        floor_size = np.asarray(
            self.settings.get("floor_size_xyz_m", [4.0, 4.0, 0.10]),
            dtype=float,
        ).reshape(-1)
        if floor_size.size != 3 or not np.all(np.isfinite(floor_size)):
            raise ValueError("floor_size_xyz_m must contain three finite values")
        if np.any(floor_size <= 0.0):
            raise ValueError("floor_size_xyz_m values must be positive")

        floor_surface_z_m = float(self.settings.get("floor_surface_z_m", 0.0))
        floor_pose = np.eye(4)
        floor_pose[2, 3] = (floor_surface_z_m - 0.5 * floor_size[2]) * 1e3

        floor_primitive = self.ros["SolidPrimitive"]()
        floor_primitive.type = self.ros["SolidPrimitive"].BOX
        floor_primitive.dimensions = floor_size.tolist()

        floor_obj.primitives = [floor_primitive]
        floor_obj.primitive_poses = [self._pose_mm(floor_pose)]
        floor_obj.operation = self.ros["CollisionObject"].ADD

        scene = self.ros["PlanningScene"]()
        scene.is_diff = True
        scene.world.collision_objects = [plane_obj, floor_obj]

        request = self.ros["ApplyPlanningScene"].Request()
        request.scene = scene
        response = self._call(
            self.scene_client,
            request,
            float(self.settings.get("scene_timeout_s", 5.0)),
        )
        if not response.success:
            raise RuntimeError("MoveIt rejected the calibration-plane scene")

    # ------------------------------------------------------------------
    # 1) Pose -> MoveIt IK
    # ------------------------------------------------------------------
    def _compute_ik(
        self,
        names: tuple[str, ...],
        start_rad: np.ndarray,
        target_mm: np.ndarray,
        *,
        avoid_collisions: bool = True,
        timeout_s: float | None = None,
    ) -> tuple[bool, str, np.ndarray, int]:
        request = self.ros["GetPositionIK"].Request()
        ik = request.ik_request
        ik.group_name = self.group_name
        ik.ik_link_name = self.tip_link
        ik.robot_state = self._robot_state(names, start_rad)
        ik.avoid_collisions = bool(avoid_collisions)

        stamped = self.ros["PoseStamped"]()
        stamped.header.frame_id = self.planning_frame
        stamped.pose = self._pose_mm(target_mm)
        ik.pose_stamped = stamped

        ik_timeout_s = float(
            self.settings.get("ik_timeout_s", 0.25)
            if timeout_s is None
            else timeout_s
        )
        if not np.isfinite(ik_timeout_s) or ik_timeout_s <= 0.0:
            raise ValueError("MoveIt IK timeout must be positive and finite")
        ik.timeout = self._duration_message(self.ros["Duration"], ik_timeout_s)

        response = self._call(
            self.ik_client,
            request,
            float(self.settings.get("ik_service_timeout_s", max(2.0, ik_timeout_s + 1.0))),
        )

        error_code = int(response.error_code.val)
        if error_code != 1:
            return (
                False,
                f"MoveIt IK failed (error {error_code})",
                np.empty((0,), dtype=float),
                error_code,
            )

        solution_names = tuple(str(n) for n in response.solution.joint_state.name)
        solution_positions = np.asarray(response.solution.joint_state.position, dtype=float)
        index = {name: i for i, name in enumerate(solution_names)}
        missing = [name for name in names if name not in index]
        if missing:
            return (
                False,
                "MoveIt IK omitted joints: " + ", ".join(missing),
                np.empty((0,), dtype=float),
                error_code,
            )

        q_goal = np.asarray([solution_positions[index[name]] for name in names], dtype=float)
        return True, "MoveIt IK succeeded", q_goal, error_code

    # ------------------------------------------------------------------
    # 2) q_start -> fixed joint goal with OMPL RRTConnect
    # ------------------------------------------------------------------
    def _plan_joint_goal(
        self,
        names: tuple[str, ...],
        start_rad: np.ndarray,
        goal_rad: np.ndarray,
    ) -> MoveItPath:
        request = self.ros["GetMotionPlan"].Request()
        motion = request.motion_plan_request

        motion.group_name = self.group_name
        motion.pipeline_id = self.pipeline_id
        motion.planner_id = self.planner_id
        motion.num_planning_attempts = int(self.settings.get("planning_attempts", 1))
        motion.allowed_planning_time = float(self.settings.get("planning_time_s", 5.0))
        motion.max_velocity_scaling_factor = float(
            self.settings.get("velocity_scaling", 0.15)
        )
        motion.max_acceleration_scaling_factor = float(
            self.settings.get("acceleration_scaling", 0.15)
        )
        motion.start_state = self._robot_state(names, start_rad)

        workspace_min = np.asarray(
            self.settings.get("workspace_min_m", [-1.5, -1.5, -0.2]),
            dtype=float,
        ).reshape(3)
        workspace_max = np.asarray(
            self.settings.get("workspace_max_m", [1.5, 1.5, 1.5]),
            dtype=float,
        ).reshape(3)
        motion.workspace_parameters.header.frame_id = self.planning_frame
        motion.workspace_parameters.min_corner.x = float(workspace_min[0])
        motion.workspace_parameters.min_corner.y = float(workspace_min[1])
        motion.workspace_parameters.min_corner.z = float(workspace_min[2])
        motion.workspace_parameters.max_corner.x = float(workspace_max[0])
        motion.workspace_parameters.max_corner.y = float(workspace_max[1])
        motion.workspace_parameters.max_corner.z = float(workspace_max[2])

        tolerance = float(self.settings.get("joint_goal_tolerance_rad", 1e-3))
        constraints = self.ros["Constraints"]()
        for name, value in zip(names, goal_rad):
            joint = self.ros["JointConstraint"]()
            joint.joint_name = str(name)
            joint.position = float(value)
            joint.tolerance_above = tolerance
            joint.tolerance_below = tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        motion.goal_constraints = [constraints]

        response = self._call(
            self.plan_client,
            request,
            float(
                self.settings.get(
                    "planning_service_timeout_s",
                    motion.allowed_planning_time + 5.0,
                )
            ),
        ).motion_plan_response

        return self._path_from_motion_plan_response(
            names,
            response,
            success_reason=f"MoveIt IK + {self.pipeline_id}/{self.planner_id}",
            failure_reason=f"{self.pipeline_id}/{self.planner_id} joint-space planning failed",
        )

    def _path_from_motion_plan_response(
        self,
        names: tuple[str, ...],
        response,
        *,
        success_reason: str,
        failure_reason: str,
    ) -> MoveItPath:
        """Convert a GetMotionPlan response without changing its timing."""
        error_code = int(response.error_code.val)
        planning_time = float(response.planning_time)
        if error_code != 1:
            return self._empty_path(
                names,
                reason=f"{failure_reason} (MoveIt error {error_code})",
                error_code=error_code,
                planning_time_s=planning_time,
            )

        trajectory = response.trajectory.joint_trajectory
        traj_names = tuple(str(name) for name in trajectory.joint_names)
        index = {name: i for i, name in enumerate(traj_names)}
        missing = [name for name in names if name not in index]
        if missing:
            return self._empty_path(
                names,
                reason="MoveIt trajectory omitted joints: " + ", ".join(missing),
                error_code=error_code,
                planning_time_s=planning_time,
            )

        points = list(trajectory.points)
        if not points:
            return self._empty_path(
                names,
                reason="MoveIt returned an empty trajectory",
                error_code=error_code,
                planning_time_s=planning_time,
            )

        positions = np.asarray(
            [[float(p.positions[index[name]]) for name in names] for p in points],
            dtype=float,
        )
        times = np.asarray(
            [self._duration_to_seconds(p.time_from_start) for p in points],
            dtype=float,
        )

        def optional_field(field_name: str) -> np.ndarray:
            rows = []
            for p in points:
                values = getattr(p, field_name)
                if len(values) == 0:
                    return np.empty((len(points), 0), dtype=float)
                rows.append([float(values[index[name]]) for name in names])
            return np.asarray(rows, dtype=float)

        return MoveItPath(
            success=True,
            reason=success_reason,
            joint_names=names,
            positions_rad=positions,
            time_from_start_s=times,
            velocities_rad_s=optional_field("velocities"),
            accelerations_rad_s2=optional_field("accelerations"),
            error_code=error_code,
            planning_time_s=planning_time,
        )

    def _path_from_cartesian_response(
        self,
        names: tuple[str, ...],
        response,
        *,
        requested_fraction: float,
    ) -> MoveItPath:
        """Convert /compute_cartesian_path output to the workflow trajectory type.

        MoveIt Humble's CartesianPathService performs TOTG time
        parameterization before returning ``response.solution``.  We preserve
        those timestamps, velocities, and accelerations exactly as returned.
        """
        error_code = int(response.error_code.val)
        fraction = float(response.fraction)
        if error_code != 1:
            return self._empty_path(
                names,
                reason=(
                    "MoveIt Cartesian path failed "
                    f"(error {error_code}, fraction {fraction:.6f})"
                ),
                error_code=error_code,
            )

        if fraction < requested_fraction:
            return self._empty_path(
                names,
                reason=(
                    "MoveIt Cartesian path incomplete: "
                    f"fraction={fraction:.6f} < required={requested_fraction:.6f}"
                ),
                error_code=error_code,
            )

        trajectory = response.solution.joint_trajectory
        traj_names = tuple(str(name) for name in trajectory.joint_names)
        index = {name: i for i, name in enumerate(traj_names)}
        missing = [name for name in names if name not in index]
        if missing:
            return self._empty_path(
                names,
                reason="Cartesian trajectory omitted joints: " + ", ".join(missing),
                error_code=error_code,
            )

        points = list(trajectory.points)
        if not points:
            return self._empty_path(
                names,
                reason="MoveIt Cartesian path returned an empty trajectory",
                error_code=error_code,
            )

        positions = np.asarray(
            [[float(p.positions[index[name]]) for name in names] for p in points],
            dtype=float,
        )
        times = np.asarray(
            [self._duration_to_seconds(p.time_from_start) for p in points],
            dtype=float,
        )

        def optional_field(field_name: str) -> np.ndarray:
            rows = []
            for p in points:
                values = getattr(p, field_name)
                if len(values) == 0:
                    return np.empty((len(points), 0), dtype=float)
                rows.append([float(values[index[name]]) for name in names])
            return np.asarray(rows, dtype=float)

        return MoveItPath(
            success=True,
            reason=f"MoveIt Cartesian path succeeded (fraction={fraction:.6f})",
            joint_names=names,
            positions_rad=positions,
            time_from_start_s=times,
            velocities_rad_s=optional_field("velocities"),
            accelerations_rad_s2=optional_field("accelerations"),
            error_code=error_code,
            planning_time_s=None,
        )

    def _diagnose_cartesian_failure(
        self,
        names: tuple[str, ...],
        start_rad: np.ndarray,
        start_pose_mm: np.ndarray | None,
        target_mm: np.ndarray,
        reported_fraction: float,
        *,
        debug_label: str | None = None,
    ) -> str:
        """Replay a failed Cartesian segment with MoveIt IK/state checks.

        This is diagnostic only and is never used as an executable path.

        For each straight-line sample:
          1. collision-aware IK from the previous successful joint seed,
          2. if that fails, collision-off IK,
          3. if collision-off IK succeeds, /check_state_validity identifies
             whether the candidate is colliding.

        ``compute_cartesian_path`` does not report the exact failing collision
        contact, so this replay is intended to distinguish the common causes:
        collision, kinematic IK failure, or seed/search instability.
        """
        label = str(debug_label or "Cartesian path")
        fraction = float(np.clip(reported_fraction, 0.0, 1.0))

        if start_pose_mm is None:
            message = (
                f"{label}: fraction={fraction:.6f}; exact replay unavailable "
                "(start Cartesian pose was not supplied)"
            )
            print(f"[CARTESIAN DIAG] {message}")
            return message

        start_pose = np.asarray(start_pose_mm, dtype=float)
        target = np.asarray(target_mm, dtype=float)
        if (
            start_pose.shape != (4, 4)
            or target.shape != (4, 4)
            or not np.all(np.isfinite(start_pose))
            or not np.all(np.isfinite(target))
        ):
            message = (
                f"{label}: fraction={fraction:.6f}; invalid diagnostic poses"
            )
            print(f"[CARTESIAN DIAG] {message}")
            return message

        translation_distance_mm = float(
            np.linalg.norm(target[:3, 3] - start_pose[:3, 3])
        )
        diagnostic_step_m = float(
            self.settings.get(
                "cartesian_diagnostic_step_m",
                self.settings.get("cartesian_step_m", 0.010),
            )
        )
        if not np.isfinite(diagnostic_step_m) or diagnostic_step_m <= 0.0:
            diagnostic_step_m = 0.010
        diagnostic_step_mm = diagnostic_step_m * 1e3
        segment_count = max(
            2,
            int(np.ceil(max(translation_distance_mm, 1e-9) / diagnostic_step_mm)),
        )

        # Keep the replay bounded. It runs only on failed Cartesian requests.
        segment_count = min(
            segment_count,
            int(self.settings.get("cartesian_diagnostic_max_samples", 25)),
        )

        start_R = start_pose[:3, :3]
        target_R = target[:3, :3]
        relative_rotation = Rotation.from_matrix(start_R.T @ target_R)

        previous_q = np.asarray(start_rad, dtype=float).copy()
        last_ok_alpha = 0.0

        print(
            f"[CARTESIAN DIAG] {label}: MoveIt reported fraction={fraction:.6f}; "
            f"replaying {segment_count} samples over {translation_distance_mm:.1f} mm"
        )

        for sample_index in range(1, segment_count + 1):
            alpha = float(sample_index / segment_count)
            sample_pose = np.eye(4)
            sample_pose[:3, 3] = (
                (1.0 - alpha) * start_pose[:3, 3]
                + alpha * target[:3, 3]
            )
            sample_pose[:3, :3] = (
                start_R
                @ Rotation.from_rotvec(
                    alpha * relative_rotation.as_rotvec()
                ).as_matrix()
            )

            try:
                aware_ok, aware_reason, aware_q, aware_error = self._compute_ik(
                    names,
                    previous_q,
                    sample_pose,
                    avoid_collisions=True,
                )
            except Exception as error:
                message = (
                    f"{label}: diagnostic collision-aware IK call failed at "
                    f"{alpha * 100.0:.1f}%: {type(error).__name__}: {error}"
                )
                print(f"[CARTESIAN DIAG] {message}")
                return message

            if aware_ok:
                previous_q = np.asarray(aware_q, dtype=float).copy()
                last_ok_alpha = alpha
                continue

            # Collision-aware IK failed. Try the same pose without collision
            # filtering so we can distinguish kinematics from collision.
            try:
                kin_ok, _kin_reason, kin_q, kin_error = self._compute_ik(
                    names,
                    previous_q,
                    sample_pose,
                    avoid_collisions=False,
                )
            except Exception as error:
                message = (
                    f"{label}: first replay failure at {alpha * 100.0:.1f}% "
                    f"(last OK {last_ok_alpha * 100.0:.1f}%); "
                    f"collision-aware IK error={aware_error}; "
                    f"collision-off diagnostic call failed: "
                    f"{type(error).__name__}: {error}"
                )
                print(f"[CARTESIAN DIAG] {message}")
                return message

            distance_from_start_mm = alpha * translation_distance_mm
            remaining_mm = max(
                0.0, translation_distance_mm - distance_from_start_mm
            )

            if not kin_ok:
                message = (
                    f"{label}: KINEMATIC_IK_FAILURE near {alpha * 100.0:.1f}% "
                    f"({distance_from_start_mm:.1f} mm from start, "
                    f"{remaining_mm:.1f} mm remaining; last OK "
                    f"{last_ok_alpha * 100.0:.1f}%); "
                    f"collision-aware error={aware_error}, "
                    f"collision-off error={kin_error}"
                )
                print(f"[CARTESIAN DIAG] {message}")
                return message

            try:
                valid, pairs, detail = self._check_state_validity(
                    names,
                    kin_q,
                )
            except Exception as error:
                message = (
                    f"{label}: first replay failure near {alpha * 100.0:.1f}% "
                    f"({distance_from_start_mm:.1f} mm from start); "
                    "collision-off IK succeeded but state-validity check failed: "
                    f"{type(error).__name__}: {error}"
                )
                print(f"[CARTESIAN DIAG] {message}")
                return message

            if not valid:
                contact_text = (
                    ", ".join(f"{a}<->{b}" for a, b in pairs)
                    if pairs
                    else detail
                )
                message = (
                    f"{label}: COLLISION near {alpha * 100.0:.1f}% "
                    f"({distance_from_start_mm:.1f} mm from start, "
                    f"{remaining_mm:.1f} mm remaining; last OK "
                    f"{last_ok_alpha * 100.0:.1f}%): {contact_text}"
                )
                print(f"[CARTESIAN DIAG] {message}")
                return message

            message = (
                f"{label}: IK_SEARCH_OR_BRANCH_FAILURE near "
                f"{alpha * 100.0:.1f}% ({distance_from_start_mm:.1f} mm from "
                f"start; last OK {last_ok_alpha * 100.0:.1f}%); "
                "collision-off IK found a state-valid solution"
            )
            print(f"[CARTESIAN DIAG] {message}")
            return message

        message = (
            f"{label}: replay reached 100% with collision-aware IK even though "
            f"compute_cartesian_path reported fraction={fraction:.6f}; "
            "likely internal IK seed/branch-search behavior rather than a "
            "deterministic collision"
        )
        print(f"[CARTESIAN DIAG] {message}")
        return message

    def _plan_cartesian_goal(
        self,
        names: tuple[str, ...],
        start_rad: np.ndarray,
        target_mm: np.ndarray,
        *,
        start_pose_mm: np.ndarray | None = None,
        debug_label: str | None = None,
    ) -> MoveItPath:
        """Plan a collision-checked straight Cartesian tip path.

        This uses MoveIt's native ``/compute_cartesian_path`` capability, not
        the Pilz industrial motion planner.  The request starts from the
        supplied joint state, follows a straight Cartesian interpolation to the
        target pose, rejects incomplete paths, and asks MoveIt to avoid
        collisions against the current PlanningScene.

        MoveIt Humble time-parameterizes the returned joint trajectory with
        TimeOptimalTrajectoryGeneration (TOTG) inside the Cartesian-path
        service, using the velocity/acceleration scaling factors below.
        """
        request = self.ros["GetCartesianPath"].Request()
        request.header.frame_id = self.planning_frame
        request.start_state = self._robot_state(names, start_rad)
        request.group_name = self.group_name
        request.link_name = self.tip_link
        request.waypoints = [self._pose_mm(target_mm)]

        max_step = float(self.settings.get("cartesian_step_m", 0.010))
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("cartesian_step_m must be positive and finite")
        request.max_step = max_step

        jump_threshold = float(
            self.settings.get("cartesian_jump_threshold", 0.0)
        )
        if not np.isfinite(jump_threshold) or jump_threshold < 0.0:
            raise ValueError(
                "cartesian_jump_threshold must be finite and >= 0"
            )
        request.jump_threshold = jump_threshold
        request.avoid_collisions = True

        velocity_scaling = float(
            self.settings.get(
                "cartesian_velocity_scaling",
                self.settings.get("velocity_scaling", 0.15),
            )
        )
        acceleration_scaling = float(
            self.settings.get(
                "cartesian_acceleration_scaling",
                self.settings.get("acceleration_scaling", 0.15),
            )
        )
        if not 0.0 < velocity_scaling <= 1.0:
            raise ValueError("cartesian_velocity_scaling must be in (0, 1]")
        if not 0.0 < acceleration_scaling <= 1.0:
            raise ValueError("cartesian_acceleration_scaling must be in (0, 1]")

        # GetCartesianPath changed across MoveIt 2 package revisions.
        # Some Humble installations expose velocity/acceleration scaling fields
        # in the service request and some older message definitions do not.
        # Support both without binding the workflow to one moveit_msgs revision.
        request_has_scaling = (
            hasattr(request, "max_velocity_scaling_factor")
            and hasattr(request, "max_acceleration_scaling_factor")
        )
        if request_has_scaling:
            request.max_velocity_scaling_factor = velocity_scaling
            request.max_acceleration_scaling_factor = acceleration_scaling

        timeout_s = float(
            self.settings.get("cartesian_service_timeout_s", 10.0)
        )
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError(
                "cartesian_service_timeout_s must be positive and finite"
            )

        response = self._call(
            self.cartesian_client,
            request,
            timeout_s,
        )

        fraction_threshold = float(
            self.settings.get("cartesian_fraction_threshold", 0.999999)
        )
        if not 0.0 < fraction_threshold <= 1.0:
            raise ValueError(
                "cartesian_fraction_threshold must be in (0, 1]"
            )

        error_code = int(response.error_code.val)
        reported_fraction = float(response.fraction)

        if error_code == 1 and reported_fraction < fraction_threshold:
            diagnostic = self._diagnose_cartesian_failure(
                names,
                start_rad,
                start_pose_mm,
                target_mm,
                reported_fraction,
                debug_label=debug_label,
            )
            return self._empty_path(
                names,
                reason=(
                    "MoveIt Cartesian path incomplete: "
                    f"fraction={reported_fraction:.6f} < "
                    f"required={fraction_threshold:.6f}; {diagnostic}"
                ),
                error_code=error_code,
            )

        path = self._path_from_cartesian_response(
            names,
            response,
            requested_fraction=fraction_threshold,
        )

        # Older GetCartesianPath requests cannot pass scaling factors into
        # move_group. Their Cartesian service still returns a timed trajectory,
        # so conservatively stretch that timing here. This keeps the existing
        # FollowJointTrajectory execution path unchanged.
        if path.success and not request_has_scaling:
            velocity_factor = velocity_scaling
            acceleration_factor = acceleration_scaling

            # Stretch time enough to satisfy both requested velocity and
            # acceleration reductions:
            #   v' = v / stretch
            #   a' = a / stretch^2
            stretch = max(
                1.0 / velocity_factor,
                1.0 / np.sqrt(acceleration_factor),
            )

            times = np.asarray(path.time_from_start_s, dtype=float)
            velocities = np.asarray(path.velocities_rad_s, dtype=float)
            accelerations = np.asarray(path.accelerations_rad_s2, dtype=float)

            if times.size:
                times = times * stretch
            if velocities.size:
                velocities = velocities / stretch
            if accelerations.size:
                accelerations = accelerations / (stretch * stretch)

            path = MoveItPath(
                success=path.success,
                reason=(
                    path.reason
                    + "; request has no Cartesian scaling fields, "
                    + f"timing stretched x{stretch:.3f}"
                ),
                joint_names=path.joint_names,
                positions_rad=path.positions_rad.copy(),
                time_from_start_s=times.copy(),
                velocities_rad_s=velocities.copy(),
                accelerations_rad_s2=accelerations.copy(),
                error_code=path.error_code,
                planning_time_s=path.planning_time_s,
            )

        return path

    # ------------------------------------------------------------------
    # Collision diagnostics
    # ------------------------------------------------------------------
    def _check_state_validity(
        self,
        names: tuple[str, ...],
        positions_rad: np.ndarray,
    ) -> tuple[bool, tuple[tuple[str, str], ...], str]:
        """Ask MoveIt whether a joint state is valid in the current scene.

        This uses MoveIt's own planning scene and collision checker, so the
        reported result is consistent with the collision-aware IK / OMPL
        pipeline.  Contact pairs are returned when MoveIt provides them.
        """
        request = self.ros["GetStateValidity"].Request()
        request.robot_state = self._robot_state(names, positions_rad)
        request.group_name = self.group_name

        response = self._call(
            self.state_validity_client,
            request,
            float(self.settings.get("state_validity_timeout_s", 3.0)),
        )

        valid = bool(response.valid)

        pairs: list[tuple[str, str]] = []
        for contact in getattr(response, "contacts", []):
            body1 = str(getattr(contact, "contact_body_1", "")).strip()
            body2 = str(getattr(contact, "contact_body_2", "")).strip()
            if not body1 and not body2:
                continue
            pair = (body1 or "?", body2 or "?")
            reverse = (pair[1], pair[0])
            if pair not in pairs and reverse not in pairs:
                pairs.append(pair)

        if valid:
            detail = "MoveIt state validity: valid"
        elif pairs:
            detail = "MoveIt collision: " + ", ".join(
                f"{a} <-> {b}" for a, b in pairs
            )
        else:
            detail = (
                "MoveIt state validity: invalid "
                "(no contact pair returned; check collision/joint constraints)"
            )

        return valid, tuple(pairs), detail

    def _diagnose_collision_aware_ik_failure(
        self,
        names: tuple[str, ...],
        start_rad: np.ndarray,
        target_mm: np.ndarray,
        original_reason: str,
        original_error: int,
    ) -> str:
        """Differentiate no-IK from collision-rejected IK.

        Diagnostic procedure:
          1. Collision-aware MoveIt IK already failed.
          2. Ask the same MoveIt IK solver for a kinematic solution with
             collision checking disabled.
          3. If no solution exists even then, this is a genuine kinematic
             IK failure (or solver/seed issue).
          4. If a solution exists, pass that state to MoveIt's own
             /check_state_validity service and report collision contacts.

        The collision-off IK solution is *never* used for motion planning.
        It is diagnostic only.
        """
        try:
            kin_ok, kin_reason, q_candidate, kin_error = self._compute_ik(
                names,
                start_rad,
                target_mm,
                avoid_collisions=False,
            )
        except Exception as error:
            return (
                f"{original_reason}; collision diagnostic failed: "
                f"{type(error).__name__}: {error}"
            )

        if not kin_ok:
            return (
                f"{original_reason}; collision-off IK also failed "
                f"(error {kin_error}) -> no kinematic IK solution found"
            )

        try:
            valid, pairs, detail = self._check_state_validity(
                names,
                q_candidate,
            )
        except Exception as error:
            return (
                f"{original_reason}; collision-off IK succeeded, but state "
                f"validity diagnostic failed: {type(error).__name__}: {error}"
            )

        if not valid:
            return (
                f"{original_reason}; collision-off IK succeeds -> "
                f"collision-aware IK rejected candidate; {detail}"
            )

        return (
            f"{original_reason}; collision-off IK succeeds and sampled "
            f"candidate is state-valid. Likely IK seed/solver search issue "
            f"rather than a deterministic collision."
        )

    # ------------------------------------------------------------------
    # Public IK API
    # ------------------------------------------------------------------
    def solve_pose_ik(
        self,
        names,
        seed_rad,
        target_mm,
        *,
        avoid_collisions: bool = True,
        diagnose_failure: bool = False,
    ) -> tuple[bool, str, np.ndarray, int]:
        """Solve only the MoveIt IK problem for a pose target.

        This deliberately does *not* run OMPL/RRTConnect.  It is useful when
        a valid collision-free joint configuration itself is needed before a
        path start state exists, e.g. the automatically generated stable pose
        in the simulation workflow.
        """
        names = tuple(str(name) for name in names)
        seed = np.asarray(seed_rad, dtype=float).reshape(-1)
        target = np.asarray(target_mm, dtype=float)

        if not names or len(names) != len(seed):
            raise ValueError("joint names/IK seed state mismatch")
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_mm must be a finite 4x4 transform")

        ok, reason, solution, error_code = self._compute_ik(
            names,
            seed,
            target,
            avoid_collisions=avoid_collisions,
        )
        if not ok and avoid_collisions and diagnose_failure:
            reason = self._diagnose_collision_aware_ik_failure(
                names,
                seed,
                target,
                reason,
                error_code,
            )
        return ok, reason, solution, error_code

    # ------------------------------------------------------------------
    # Public planning API used by workflow.py
    # ------------------------------------------------------------------
    def plan_to_pose(
        self,
        names,
        start_rad,
        target_mm,
    ) -> MoveItPath:
        names = tuple(str(name) for name in names)
        start = np.asarray(start_rad, dtype=float).reshape(-1)
        target = np.asarray(target_mm, dtype=float)

        if not names or len(names) != len(start):
            raise ValueError("joint names/start joint state mismatch")
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_mm must be a finite 4x4 transform")

        ok, reason, q_goal, ik_error = self._compute_ik(
            names,
            start,
            target,
            avoid_collisions=True,
        )
        if not ok:
            diagnostic_reason = self._diagnose_collision_aware_ik_failure(
                names,
                start,
                target,
                reason,
                ik_error,
            )
            return self._empty_path(
                names,
                reason=diagnostic_reason,
                error_code=ik_error,
            )

        # All six RB5 planning joints are bounded revolute joints in the
        # MoveIt URDF, so use a direct difference. In particular, +179 deg
        # and -179 deg are not treated as a two-degree motion.
        start_deg = np.degrees(start)
        goal_deg = np.degrees(q_goal)
        delta_deg = goal_deg - start_deg
        print("[IK]")
        print(f"start_deg = {np.round(start_deg, 3).tolist()}")
        print(f"goal_deg  = {np.round(goal_deg, 3).tolist()}")
        print(f"delta_deg = {np.round(delta_deg, 3).tolist()}")
        print(f"max_abs_delta_deg = {float(np.max(np.abs(delta_deg))):.3f}")
        print(f"sum_abs_delta_deg = {float(np.sum(np.abs(delta_deg))):.3f}")

        return self._plan_joint_goal(names, start, q_goal)

    def plan_to_joint_positions(
        self,
        names,
        start_rad,
        goal_rad,
    ) -> MoveItPath:
        """Plan a collision-checked path between two fixed joint states."""
        names = tuple(str(name) for name in names)
        start = np.asarray(start_rad, dtype=float).reshape(-1)
        goal = np.asarray(goal_rad, dtype=float).reshape(-1)
        if not names or len(names) != len(start) or len(names) != len(goal):
            raise ValueError("joint names/start/goal joint state mismatch")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("start_rad and goal_rad must be finite")
        return self._plan_joint_goal(names, start, goal)

    def plan_cartesian_to_pose(
        self,
        names,
        start_rad,
        target_mm,
        *,
        start_pose_mm=None,
        debug_label: str | None = None,
    ) -> MoveItPath:
        """Plan a straight collision-checked Cartesian tip path with MoveIt."""
        names = tuple(str(name) for name in names)
        start = np.asarray(start_rad, dtype=float).reshape(-1)
        target = np.asarray(target_mm, dtype=float)
        if not names or len(names) != len(start):
            raise ValueError("joint names/start joint state mismatch")
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_mm must be a finite 4x4 transform")
        start_pose = None
        if start_pose_mm is not None:
            start_pose = np.asarray(start_pose_mm, dtype=float)
            if start_pose.shape != (4, 4) or not np.all(np.isfinite(start_pose)):
                raise ValueError(
                    "start_pose_mm must be None or a finite 4x4 transform"
                )
        return self._plan_cartesian_goal(
            names,
            start,
            target,
            start_pose_mm=start_pose,
            debug_label=debug_label,
        )

    def plan_rrtconnect_to_pose(self, names, start_rad, target_mm) -> MoveItPath:
        """Backward-compatible alias; planner selection now comes from settings."""
        return self.plan_to_pose(names, start_rad, target_mm)


# Keep the previous import name working.
MoveItPlanner = MoveItServicePlanner