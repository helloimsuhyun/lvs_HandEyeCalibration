from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any


class ROS2DriverLauncher:
    """Own a robot driver's ROS 2 launch process for one workflow session."""

    def __init__(self, config) -> None:
        self.config = config
        self.settings: dict[str, Any] = dict(
            config.values.get("equipment", {}).get("driver", {})
        )
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_path: Path | None = None
        self.calibration_path: Path | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("auto_launch", False))

    def _path(self, raw: str | Path) -> Path:
        value = Path(str(raw)).expanduser()
        if not value.is_absolute():
            value = self.config.source_path.parent / value
        return value.resolve()

    def _shell_prefix(self) -> list[str]:
        sources = []
        # Source the robot overlay last.  This repo contains a lightweight
        # visualization-only package also named rbpodo_description; the real
        # RB driver must resolve the official package from rbpodo_ros2_ws.
        for key in ("ros_setup", "project_workspace_setup", "robot_workspace_setup"):
            raw = self.settings.get(key)
            if not raw:
                continue
            path = self._path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"ROS setup file not found ({key}): {path}")
            sources.append(f"source {shlex.quote(str(path))}")
        return sources

    def _command(self, package: str, launch_file: str, arguments: dict[str, Any]) -> list[str]:
        parts = ["ros2", "launch", package, launch_file]
        for key, value in arguments.items():
            if value is None:
                continue
            if not str(key).replace("_", "").isalnum():
                raise ValueError(f"invalid ROS launch argument name: {key!r}")
            if isinstance(value, bool):
                value = "true" if value else "false"
            parts.append(f"{key}:={value}")
        command = " ".join(shlex.quote(str(part)) for part in parts)
        setup = self._shell_prefix()
        return ["bash", "-lc", " && ".join([*setup, f"exec {command}"])]

    @staticmethod
    def _stop_process(process: subprocess.Popen, grace_s: float = 8.0) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=grace_s)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def _prepare_ur_calibration(self, robot_ip: str, log_dir: Path) -> Path | None:
        calibration = self.settings.get("calibration")
        if not isinstance(calibration, dict) or not calibration.get("auto_extract", False):
            return None

        safe_ip = robot_ip.replace(".", "_").replace(":", "_")
        raw_target = calibration.get("target_file")
        target = (
            self._path(raw_target)
            if raw_target
            else log_dir / f"ur5e_{safe_ip}_kinematics.yaml"
        )
        if target.is_file() and target.stat().st_size > 0:
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        arguments = dict(calibration.get("launch_arguments", {}))
        arguments.update(robot_ip=robot_ip, target_filename=str(target))
        command = self._command(
            str(calibration.get("launch_package", "ur_calibration")),
            str(calibration.get("launch_file", "calibration_correction.launch.py")),
            arguments,
        )
        calibration_log = log_dir / "ur_calibration.log"
        with calibration_log.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "ROS_LOG_DIR": str(log_dir / "ros_logs")},
            )
            try:
                return_code = process.wait(
                    timeout=float(calibration.get("timeout_s", 45.0))
                )
            except subprocess.TimeoutExpired as error:
                self._stop_process(process)
                raise TimeoutError(
                    "UR factory calibration extraction timed out; check robot IP and network. "
                    f"Log: {calibration_log}"
                ) from error
        if return_code != 0 or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(
                "UR factory calibration extraction failed. "
                f"exit={return_code}, log={calibration_log}"
            )
        return target

    def start(self, robot_ip: str) -> None:
        if not self.enabled:
            return
        if self.process is not None and self.process.poll() is None:
            return
        address = ipaddress.ip_address(robot_ip)
        if address.version != 4:
            raise ValueError("robot_ip must be an IPv4 address")

        log_dir = self.config.path("session_root") / "ros_driver"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "ros_logs").mkdir(parents=True, exist_ok=True)

        arguments = dict(self.settings.get("launch_arguments", {}))
        arguments["robot_ip"] = robot_ip
        calibration_path = self._prepare_ur_calibration(robot_ip, log_dir)
        self.calibration_path = calibration_path
        if calibration_path is not None:
            arguments["kinematics_params_file"] = str(calibration_path)

        command = self._command(
            str(self.settings["launch_package"]),
            str(self.settings["launch_file"]),
            arguments,
        )
        self.log_path = log_dir / "driver.log"
        self.log_file = self.log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "ROS_LOG_DIR": str(log_dir / "ros_logs")},
        )
        time.sleep(float(self.settings.get("process_alive_check_s", 1.0)))
        self.raise_if_exited()

    def raise_if_exited(self) -> None:
        if self.process is None:
            return
        return_code = self.process.poll()
        if return_code is None:
            return
        tail = ""
        if self.log_path is not None and self.log_path.is_file():
            tail = "\n".join(
                self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            )
        raise RuntimeError(
            f"ROS 2 robot driver exited early (exit={return_code}). "
            f"Log: {self.log_path}\n{tail}"
        )

    def close(self) -> None:
        if self.process is not None:
            self._stop_process(self.process)
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = None
        self.calibration_path = None
