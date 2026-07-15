from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .hardware import ProfileSample
from .planning import (
    build_single_plane_plan,
    load_json,
    load_transform,
    save_json,
    save_plan_csv,
    validate_plan_identity,
    validate_plan_runtime_safety,
)
from .profile_broker import ProfileBroker
from .safety import LIVE_ACKNOWLEDGEMENT
from .ui_model import DashboardSnapshot, build_dashboard_snapshot
from .ui_plot import render_calibration_axis, render_profile_axes, render_scene_axis
from .workflow import (
    ScanCaptureSession,
    calibrate_dataset,
    capture_bootstrap_once,
    finalize_bootstrap_plane,
    load_runtime_config,
    make_laser,
    make_robot,
    validate_bootstrap_boundary_quality,
    validate_plan_bootstrap_quality,
)


class CalibrationDashboard(tk.Tk):
    """Threaded desktop dashboard for the real calibration workflow."""

    def __init__(
        self,
        *,
        config_path: str = "real_laser_handeye/real_config.json",
        handeye_path: str = "initial_T_tcp_sensor.csv",
        plane_boundary_path: str = "runs/bootstrap/plane_boundary.json",
        plan_path: str = "runs/motion_plan.json",
        dataset_dir: str = "runs/dataset",
        output_path: str = "runs/T_tcp_sensor_calibrated.csv",
    ) -> None:
        super().__init__()
        self.title("Real 2D Laser Hand–Eye Calibration")
        self.geometry("1540x940")
        self.minsize(1180, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.config_path = tk.StringVar(value=config_path)
        self.handeye_path = tk.StringVar(value=handeye_path)
        self.boundary_path = tk.StringVar(value=plane_boundary_path)
        self.plan_path = tk.StringVar(value=plan_path)
        self.dataset_dir = tk.StringVar(value=dataset_dir)
        self.output_path = tk.StringVar(value=output_path)
        self.heights = tk.StringVar(value="60 90 120")
        self.thetas = tk.StringVar(value="30")
        self.betas = tk.StringVar(value="60 90 120")
        self.reference_count = tk.StringVar(value="24")
        self.reference_theta = tk.StringVar(value="60")
        self.bootstrap_margin = tk.StringVar(value="20")
        self.acknowledgement = tk.StringVar(value="")
        self.stage_var = tk.StringVar(value="DISCONNECTED")
        self.progress_var = tk.StringVar(value="Plan not loaded")
        self.profile_var = tk.StringVar(value="Laser preview disconnected")

        self._events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2048)
        self._job_lock = threading.Lock()
        self._job_active = False
        self._active_job_label: str | None = None
        self._job_done = threading.Event()
        self._job_done.set()
        self._closing = False
        self._closing_requested = False
        self._close_wait_notice_shown = False
        self._stop_thread: threading.Thread | None = None
        self._auto_pause = threading.Event()
        self._auto_pause.set()
        self._last_live_sequence = -1
        self._last_live_error: str | None = None
        self._snapshot_running = False
        self._snapshot_pending = False
        self._selected_previous_scan: int | None = None

        self.config_data: dict[str, Any] | None = None
        self.safety = None
        self.capture_config = None
        self.robot = None
        self.laser = None
        self.profile_broker: ProfileBroker | None = None
        self.plan: dict[str, Any] | None = None
        self.session: ScanCaptureSession | None = None
        self.snapshot: DashboardSnapshot | None = None
        self.calibration_diagnostics: dict[str, Any] | None = None
        self.calibrated_T: np.ndarray | None = None

        self._build_layout()
        self.after(80, self._drain_events)
        self.after(100, self._refresh_live_profile)

    # ------------------------------------------------------------------ UI
    def _build_layout(self) -> None:
        paths = ttk.LabelFrame(self, text="Files and scan parameters")
        paths.pack(fill="x", padx=8, pady=(8, 4))
        rows = [
            ("Config", self.config_path, "file"),
            ("Initial T_tcp_sensor", self.handeye_path, "file"),
            ("Plane boundary", self.boundary_path, "file"),
            ("Motion plan", self.plan_path, "file"),
            ("Dataset directory", self.dataset_dir, "dir"),
            ("Calibration output", self.output_path, "save"),
        ]
        for row, (label, variable, browse_type) in enumerate(rows):
            ttk.Label(paths, text=label, width=20).grid(
                row=row // 2, column=(row % 2) * 3, sticky="w", padx=5, pady=3
            )
            ttk.Entry(paths, textvariable=variable, width=49).grid(
                row=row // 2, column=(row % 2) * 3 + 1, sticky="ew", padx=3
            )
            ttk.Button(
                paths,
                text="…",
                width=3,
                command=lambda v=variable, t=browse_type: self._browse(v, t),
            ).grid(row=row // 2, column=(row % 2) * 3 + 2, padx=(0, 7))
        paths.columnconfigure(1, weight=1)
        paths.columnconfigure(4, weight=1)

        parameter_row = ttk.Frame(paths)
        parameter_row.grid(row=3, column=0, columnspan=6, sticky="ew", pady=4)
        for label, variable, width in (
            ("d [mm]", self.heights, 17),
            ("theta [deg]", self.thetas, 12),
            ("beta [deg]", self.betas, 17),
            ("theta60 refs", self.reference_count, 7),
            ("reference theta", self.reference_theta, 10),
            ("bootstrap margin", self.bootstrap_margin, 8),
        ):
            ttk.Label(parameter_row, text=label).pack(side="left", padx=(6, 2))
            ttk.Entry(parameter_row, textvariable=variable, width=width).pack(
                side="left"
            )

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=4)
        ttk.Button(toolbar, text="Load plan", command=self._load_plan).pack(side="left")
        ttk.Button(toolbar, text="Generate 105-pose plan", command=self._generate_plan).pack(
            side="left", padx=4
        )
        self.connect_button = ttk.Button(
            toolbar, text="Connect preview", command=self._connect_hardware
        )
        self.connect_button.pack(side="left", padx=(14, 4))
        ttk.Button(toolbar, text="Disconnect", command=self._disconnect_hardware).pack(
            side="left"
        )
        ttk.Label(toolbar, text="Live acknowledgement:").pack(side="left", padx=(18, 3))
        ttk.Entry(toolbar, textvariable=self.acknowledgement, width=31).pack(side="left")
        tk.Button(
            toolbar,
            text="SOFTWARE STOP",
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            command=self._emergency_stop,
        ).pack(side="right")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)
        monitor = ttk.Frame(notebook)
        scene = ttk.Frame(notebook)
        plan_tab = ttk.Frame(notebook)
        calibration = ttk.Frame(notebook)
        notebook.add(monitor, text="Live / Capture")
        notebook.add(scene, text="Accumulated plane / 3D plan")
        notebook.add(plan_tab, text="Capture plan")
        notebook.add(calibration, text="Calibration")

        self._build_monitor_tab(monitor)
        self._build_scene_tab(scene)
        self._build_plan_tab(plan_tab)
        self._build_calibration_tab(calibration)

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8, pady=(0, 7))
        ttk.Label(status, textvariable=self.stage_var, width=19).pack(side="left")
        self.progress_value = tk.DoubleVar(value=0.0)
        ttk.Progressbar(status, variable=self.progress_value, maximum=100).pack(
            side="left", fill="x", expand=True, padx=5
        )
        ttk.Label(status, textvariable=self.progress_var).pack(side="left", padx=8)
        tk.Label(
            status,
            text="Software stop은 물리 E-stop을 대체하지 않습니다",
            fg="#c62828",
        ).pack(side="right")

    def _build_monitor_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)
        figure = Figure(figsize=(7.5, 6.5), dpi=100)
        self.live_axis = figure.add_subplot(211)
        self.previous_axis = figure.add_subplot(212)
        self.profile_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.profile_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(parent, padding=8)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        ttk.Label(side, textvariable=self.profile_var).grid(row=0, column=0, sticky="w")
        self.next_text = tk.Text(side, height=17, width=56, font=("TkFixedFont", 10))
        self.next_text.grid(row=1, column=0, sticky="nsew", pady=6)
        side.rowconfigure(1, weight=1)

        bootstrap = ttk.LabelFrame(side, text="Manual bootstrap (robot motion disabled)")
        bootstrap.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Button(
            bootstrap, text="Capture current bootstrap view", command=self._capture_bootstrap
        ).pack(side="left", padx=4, pady=4)
        ttk.Button(
            bootstrap, text="Finalize 4 views / plane", command=self._finalize_bootstrap
        ).pack(side="left", padx=4)
        self.bootstrap_status = ttk.Label(bootstrap, text="0/4")
        self.bootstrap_status.pack(side="right", padx=8)

        controls = ttk.LabelFrame(side, text="Reviewed-plan collection")
        controls.grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Button(controls, text="Capture next 1", command=self._capture_next).pack(
            side="left", padx=4, pady=4
        )
        ttk.Button(controls, text="Start automatic", command=self._start_auto).pack(
            side="left", padx=4
        )
        ttk.Button(controls, text="Pause after safe return", command=self._pause_auto).pack(
            side="left", padx=4
        )

    def _build_scene_tab(self, parent: ttk.Frame) -> None:
        figure = Figure(figsize=(10, 7), dpi=100)
        self.scene_axis = figure.add_subplot(111, projection="3d")
        self.scene_canvas = FigureCanvasTkAgg(figure, master=parent)
        toolbar = NavigationToolbar2Tk(self.scene_canvas, parent, pack_toolbar=False)
        toolbar.pack(side="top", fill="x")
        self.scene_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_plan_tab(self, parent: ttk.Frame) -> None:
        columns = ("status", "id", "line", "d", "theta", "beta", "reference")
        self.plan_tree = ttk.Treeview(parent, columns=columns, show="headings")
        widths = (95, 65, 65, 80, 80, 80, 90)
        for column, width in zip(columns, widths):
            self.plan_tree.heading(column, text=column)
            self.plan_tree.column(column, width=width, anchor="center")
        self.plan_tree.tag_configure("done", background="#d8f3dc")
        self.plan_tree.tag_configure("next", background="#ffe5d9")
        self.plan_tree.bind("<<TreeviewSelect>>", self._select_previous_capture)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.plan_tree.yview)
        self.plan_tree.configure(yscrollcommand=scrollbar.set)
        self.plan_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_calibration_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)
        left = ttk.Frame(parent, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        ttk.Button(
            left,
            text="Run joint-linear calibration",
            command=self._run_calibration,
        ).pack(fill="x", pady=4)
        ttk.Button(left, text="Refresh dataset", command=self._refresh_snapshot).pack(
            fill="x", pady=4
        )
        self.calibration_text = tk.Text(left, width=54, font=("TkFixedFont", 10))
        self.calibration_text.pack(fill="both", expand=True, pady=5)
        figure = Figure(figsize=(8, 6), dpi=100)
        self.calibration_axis = figure.add_subplot(111)
        self.calibration_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.calibration_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        render_calibration_axis(self.calibration_axis, None)

    # --------------------------------------------------------------- helpers
    def _browse(self, variable: tk.StringVar, kind: str) -> None:
        with self._job_lock:
            if self._job_active:
                messagebox.showwarning(
                    "Busy", "Pause/finish the current operation before changing paths"
                )
                return
        if kind == "dir":
            value = filedialog.askdirectory(initialdir=str(Path(variable.get()).parent))
        elif kind == "save":
            value = filedialog.asksaveasfilename(initialfile=Path(variable.get()).name)
        else:
            value = filedialog.askopenfilename(initialdir=str(Path(variable.get()).parent))
        if value:
            variable.set(value)

    @staticmethod
    def _numbers(value: str) -> list[float]:
        result = [float(item) for item in value.replace(",", " ").split()]
        if not result:
            raise ValueError("parameter list cannot be empty")
        return result

    def _submit(
        self,
        label: str,
        function: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
    ) -> None:
        if self._closing_requested:
            return
        with self._job_lock:
            if self._job_active:
                messagebox.showwarning("Busy", "Another hardware/solver operation is running")
                return
            self._job_active = True
            self._active_job_label = label
            self._job_done.clear()
        self.stage_var.set(label)

        def worker() -> None:
            try:
                result = function()
                self._events.put(("success", (label, result, on_success)))
            except BaseException as exc:
                self._events.put(
                    (
                        "error",
                        (label, exc, traceback.format_exc()),
                    )
                )

        threading.Thread(target=worker, name=f"ui-{label}", daemon=True).start()

    def _workflow_event(self, event: dict[str, Any]) -> None:
        try:
            self._events.put_nowait(("workflow", event))
        except queue.Full:
            # Waypoint progress is lossy by design; safety-relevant terminal
            # events block briefly until Tk drains the queue.
            if event.get("stage") != "MOVING":
                self._events.put(("workflow", event), timeout=1.0)

    def _drain_events(self) -> None:
        if self._closing:
            return
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "workflow":
                    self._handle_workflow_event(payload)
                elif kind == "success":
                    label, result, callback = payload
                    try:
                        # During shutdown CONNECTING must transfer newly-created
                        # handles to the close path, while DISCONNECTING must
                        # clear handles it already closed. Other UI callbacks
                        # are deliberately skipped once close was requested.
                        if callback is not None and (
                            not self._closing_requested
                            or label in {"CONNECTING", "DISCONNECTING"}
                        ):
                            callback(result)
                        if self.stage_var.get() == label:
                            self.stage_var.set(
                                "CONNECTED_IDLE" if self.robot else "READY"
                            )
                    except Exception as exc:
                        self.stage_var.set("FAULT")
                        self.progress_var.set(f"{label} callback: {exc}")
                        print(traceback.format_exc())
                        if not self._closing_requested:
                            messagebox.showerror(label, str(exc))
                    finally:
                        with self._job_lock:
                            self._job_active = False
                            self._active_job_label = None
                        self._job_done.set()
                elif kind == "error":
                    label, error, detail = payload
                    with self._job_lock:
                        self._job_active = False
                        self._active_job_label = None
                    self._job_done.set()
                    self._auto_pause.set()
                    self.stage_var.set("FAULT")
                    self.progress_var.set(f"{label}: {error}")
                    print(detail)
                    if not self._closing_requested:
                        messagebox.showerror(label, str(error))
                        self._refresh_snapshot()
                elif kind == "snapshot":
                    (
                        requested_plan_id,
                        requested_dataset,
                        snapshot,
                        error,
                        detail,
                    ) = payload
                    self._snapshot_running = False
                    active_plan = (
                        self.session.plan if self.session is not None else self.plan
                    )
                    active_plan_id = (
                        None if active_plan is None else str(active_plan.get("plan_id", ""))
                    )
                    active_dataset = str(self._active_dataset_path())
                    if (
                        not self._closing_requested
                        and requested_plan_id == active_plan_id
                        and requested_dataset == active_dataset
                    ):
                        if error is not None:
                            self.progress_var.set(f"Dashboard data error: {error}")
                            if detail:
                                print(detail)
                        else:
                            assert snapshot is not None
                            # A broker frame can arrive while disk/SVD work is
                            # running. Preserve the newest profile when the
                            # completed snapshot is published on Tk's thread.
                            if self.profile_broker is not None:
                                sequence, sample, broker_error = (
                                    self.profile_broker.latest()
                                )
                                effective_error = self._effective_profile_error(
                                    sample, broker_error
                                )
                                if (
                                    sequence != snapshot.live_sequence
                                    or effective_error != snapshot.live_error
                                ):
                                    points = (
                                        np.empty((0, 3))
                                        if sample is None
                                        else np.asarray(sample.points_s, dtype=float).copy()
                                    )
                                    points.setflags(write=False)
                                    snapshot = replace(
                                        snapshot,
                                        live_sequence=sequence,
                                        live_profile_s=points,
                                        live_error=effective_error,
                                    )
                            self.snapshot = snapshot
                            self._render_snapshot(full=True)
                    if self._snapshot_pending and not self._closing_requested:
                        self._snapshot_pending = False
                        self.after_idle(self._refresh_snapshot)
        except queue.Empty:
            pass
        self.after(80, self._drain_events)

    def _handle_workflow_event(self, event: dict[str, Any]) -> None:
        stage = str(event.get("stage", "WORKING"))
        self.stage_var.set(stage)
        if stage == "MOVING":
            self.progress_var.set(
                f"{event.get('label')} · waypoint "
                f"{int(event.get('waypoint_index', 0)) + 1}/{event.get('waypoint_count', '?')}"
            )
        elif stage in ("SCAN_COMPLETE", "COMPLETE", "SAFE_IDLE"):
            self.progress_var.set(
                f"captured {event.get('completed', 0)}/{event.get('total', '?')}"
            )
            self._refresh_snapshot()
        elif stage == "FAULT":
            self.progress_var.set(str(event.get("message", "workflow fault")))

    # -------------------------------------------------------- plan / hardware
    def _load_plan(self) -> None:
        with self._job_lock:
            if self._job_active:
                messagebox.showwarning(
                    "Busy", "Pause/finish the current hardware operation first"
                )
                return
        try:
            plan = load_json(self.plan_path.get())
            validate_plan_identity(plan)
            _config, safety, capture = load_runtime_config(self.config_path.get())
            validate_plan_runtime_safety(plan, safety)
            validate_plan_bootstrap_quality(plan, capture)
            self.plan = plan
            self.session = None
            self._clear_calibration_result()
            self._populate_plan_tree()
            self._refresh_snapshot()
            self.stage_var.set("PLAN_READY")
        except Exception as exc:
            messagebox.showerror("Load plan", str(exc))

    def _generate_plan(self) -> None:
        try:
            config_path = self.config_path.get()
            boundary_path = self.boundary_path.get()
            handeye_path = self.handeye_path.get()
            output_path = Path(self.plan_path.get())
            heights = self._numbers(self.heights.get())
            thetas = self._numbers(self.thetas.get())
            betas = self._numbers(self.betas.get())
            reference_count = int(self.reference_count.get())
            reference_theta = self._numbers(self.reference_theta.get())
        except ValueError as exc:
            messagebox.showerror("Plan parameters", str(exc))
            return

        def work() -> dict[str, Any]:
            _config, safety, capture = load_runtime_config(config_path)
            boundary = load_json(boundary_path)
            handeye = load_transform(handeye_path)
            validate_bootstrap_boundary_quality(boundary, capture, handeye)
            plan = build_single_plane_plan(
                plane_boundary=boundary,
                T_tcp_sensor_init=handeye,
                safety=safety,
                heights_mm=heights,
                theta_deg=thetas,
                beta_deg=betas,
                reference_scan_count=reference_count,
                reference_theta_deg=reference_theta,
                reference_heights_mm=heights,
                reference_beta_deg=betas,
                pose_geometry="paper_incidence",
            )
            save_json(output_path, plan)
            save_plan_csv(output_path.with_suffix(".csv"), plan)
            return plan

        def success(plan: dict[str, Any]) -> None:
            self.plan = plan
            self.session = None
            self._clear_calibration_result()
            self._populate_plan_tree()
            self._refresh_snapshot()
            self.stage_var.set("PLAN_READY")
            messagebox.showinfo(
                "Plan generated",
                f"{len(plan['entries'])} poses · rank "
                f"{plan['observability']['rank']}/4\n{self.plan_path.get()}",
            )

        self._submit("PLANNING", work, success)

    def _connect_hardware(self) -> None:
        if self.robot is not None:
            messagebox.showinfo("Hardware", "Hardware is already connected")
            return

        config_path = self.config_path.get()

        def work():
            config, safety, capture = load_runtime_config(config_path)
            robot = make_robot(config)
            laser = make_laser(config, robot=robot)
            try:
                robot.connect()
                laser.connect()
                broker = ProfileBroker(laser, acquisition_timeout_s=capture.timeout_s)
                broker.start()
            except BaseException:
                try:
                    laser.close()
                finally:
                    robot.close()
                raise
            return config, safety, capture, robot, laser, broker

        def success(result) -> None:
            (
                self.config_data,
                self.safety,
                self.capture_config,
                self.robot,
                self.laser,
                self.profile_broker,
            ) = result
            self.stage_var.set("CONNECTED_IDLE")
            self.profile_var.set("Waiting for fresh laser profile…")
            self._update_bootstrap_count()

        self._submit("CONNECTING", work, success)

    def _disconnect_hardware(self) -> None:
        if self.robot is None:
            return
        with self._job_lock:
            if self._job_active:
                messagebox.showwarning(
                    "Busy",
                    "Current motion/capture must finish or be software-stopped before disconnect",
                )
                return
        self._auto_pause.set()
        robot, laser, broker = self.robot, self.laser, self.profile_broker
        session = self.session
        prior_stop_thread = self._stop_thread
        stop_wait_s = (
            30.0
            if self.safety is None
            else max(1.0, float(self.safety.motion_timeout_s))
        )

        def work() -> None:
            if (
                prior_stop_thread is not None
                and prior_stop_thread is not threading.current_thread()
                and prior_stop_thread.is_alive()
            ):
                prior_stop_thread.join(timeout=stop_wait_s)
                if prior_stop_thread.is_alive():
                    raise TimeoutError(
                        "controlled-stop call is still active; hardware was not closed"
                    )
            if session is not None:
                session.request_stop()
            else:
                robot.stop()
            if broker is not None:
                broker.stop()
            try:
                if laser is not None:
                    laser.close()
            finally:
                robot.close()

        def success(_result) -> None:
            self.session = None
            self.robot = self.laser = self.profile_broker = None
            self.stage_var.set("DISCONNECTED")
            self.profile_var.set("Laser preview disconnected")

        self._submit("DISCONNECTING", work, success)

    def _ensure_live_session(self) -> ScanCaptureSession:
        if self.robot is None or self.profile_broker is None:
            raise RuntimeError("connect robot and laser first")
        if self.plan is None:
            raise RuntimeError("load or generate a reviewed motion plan first")
        assert self.safety is not None and self.capture_config is not None
        if not self.capture_config.return_to_safe_between_scans:
            raise RuntimeError(
                "UI safe-pause mode requires capture.return_to_safe_between_scans=true"
            )
        acknowledgement = self.acknowledgement.get()
        dataset_dir = self.dataset_dir.get()
        self.safety.assert_live_unlocked(acknowledgement)
        if self.session is None:
            self.session = ScanCaptureSession(
                robot=self.robot,
                profile_source=self.profile_broker,
                plan=self.plan,
                output_dir=dataset_dir,
                safety=self.safety,
                capture=self.capture_config,
                on_event=self._workflow_event,
            )
        return self.session

    # ------------------------------------------------------------- bootstrap
    def _bootstrap_indices(self) -> set[int]:
        manifest = Path(self.boundary_path.get()).parent / "bootstrap_manifest.json"
        if not manifest.exists():
            return set()
        try:
            return {
                int(item["index"])
                for item in load_json(manifest).get("captures", [])
            }
        except Exception:
            return set()

    def _update_bootstrap_count(self) -> None:
        completed = self._bootstrap_indices()
        self.bootstrap_status.configure(text=f"{len(completed)}/4")

    def _capture_bootstrap(self) -> None:
        if self.robot is None or self.profile_broker is None:
            messagebox.showerror("Bootstrap", "Connect hardware preview first")
            return
        missing = [index for index in range(1, 5) if index not in self._bootstrap_indices()]
        if not missing:
            messagebox.showinfo("Bootstrap", "All four views are already captured")
            return
        index = missing[0]
        if not messagebox.askokcancel(
            "Manual bootstrap",
            f"Teach pendant로 view {index}/4 위치에 정지했고 충돌을 확인했습니까?\n"
            "이 버튼은 로봇 이동 명령을 보내지 않습니다.",
        ):
            return
        assert self.safety is not None and self.capture_config is not None
        output_dir = Path(self.boundary_path.get()).parent
        robot = self.robot
        broker = self.profile_broker
        safety = self.safety
        capture_config = self.capture_config
        self._submit(
            f"BOOTSTRAP_CAPTURE_{index}",
            lambda: capture_bootstrap_once(
                robot=robot,
                profile_source=broker,
                output_dir=output_dir,
                index=index,
                safety=safety,
                capture=capture_config,
            ),
            lambda _record: self._update_bootstrap_count(),
        )

    def _finalize_bootstrap(self) -> None:
        try:
            output_dir = Path(self.boundary_path.get()).parent
            handeye_path = self.handeye_path.get()
            config_path = self.config_path.get()
            connected_capture_config = self.capture_config
            margin_mm = float(self.bootstrap_margin.get())
        except ValueError as exc:
            messagebox.showerror("Bootstrap margin", str(exc))
            return

        def work():
            capture_config = connected_capture_config
            if capture_config is None:
                _config, _safety, capture_config = load_runtime_config(config_path)
            return finalize_bootstrap_plane(
                output_dir=output_dir,
                T_tcp_sensor_init=load_transform(handeye_path),
                margin_mm=margin_mm,
                max_plane_rms_mm=capture_config.max_bootstrap_plane_rms_mm,
                min_span_mm=capture_config.min_bootstrap_span_mm,
                min_sensor_distance_mm=(
                    capture_config.min_bootstrap_sensor_plane_distance_mm
                ),
            )

        def success(boundary):
            self.boundary_path.set(
                str(Path(self.boundary_path.get()).parent / "plane_boundary.json")
            )
            messagebox.showinfo(
                "Bootstrap plane",
                f"RMS = {boundary['plane']['rms_error_mm']:.4f} mm\n"
                f"saved: {self.boundary_path.get()}",
            )

        self._submit("BOOTSTRAP_FIT", work, success)

    # --------------------------------------------------------------- capture
    def _capture_next(self) -> None:
        try:
            session = self._ensure_live_session()
        except Exception as exc:
            messagebox.showerror("Capture next", str(exc))
            return

        def work():
            return session.capture_next()

        def success(record):
            self._refresh_snapshot()
            if record is None:
                messagebox.showinfo("Collection", "Reviewed capture plan is complete")

        self._submit("CAPTURE_NEXT", work, success)

    def _start_auto(self) -> None:
        try:
            session = self._ensure_live_session()
        except Exception as exc:
            messagebox.showerror("Automatic collection", str(exc))
            return
        self._auto_pause.clear()

        def work():
            while not self._auto_pause.is_set():
                record = session.capture_next()
                if record is None:
                    session.finalize()
                    break
            return session.completed_count, session.total_count

        def success(result):
            self._auto_pause.set()
            self._refresh_snapshot()
            completed, total = result
            self.stage_var.set("COMPLETE" if completed == total else "PAUSED_SAFE")

        self._submit("AUTO_COLLECTION", work, success)

    def _pause_auto(self) -> None:
        self._auto_pause.set()
        self.progress_var.set("Pause requested: current scan will retreat to safe transit first")

    def _emergency_stop(self) -> None:
        self._auto_pause.set()
        self.stage_var.set("STOPPING")
        self.progress_var.set("Software stop requested; verify physical robot state")
        if self._stop_thread is not None and self._stop_thread.is_alive():
            return
        session, robot = self.session, self.robot

        def stop() -> None:
            try:
                if session is not None:
                    session.request_stop()
                elif robot is not None:
                    robot.stop()
            except Exception as exc:
                self._events.put(("workflow", {"stage": "FAULT", "message": str(exc)}))

        self._stop_thread = threading.Thread(
            target=stop, name="ui-software-stop", daemon=True
        )
        self._stop_thread.start()

    # ----------------------------------------------------------- calibration
    def _run_calibration(self) -> None:
        output = Path(self.output_path.get())
        dataset_dir = self._active_dataset_path()
        handeye_path = self.handeye_path.get()
        config_path = self.config_path.get()
        connected_capture_config = self.capture_config
        active_plan = self.session.plan if self.session is not None else self.plan
        if active_plan is None:
            messagebox.showerror("Calibration", "Load the reviewed motion plan first")
            return
        expected_plan_id = str(active_plan["plan_id"])

        def work():
            capture_config = connected_capture_config
            if capture_config is None:
                _config, _safety, capture_config = load_runtime_config(config_path)
            diagnostics = calibrate_dataset(
                dataset_dir=dataset_dir,
                T_tcp_sensor_init=load_transform(handeye_path),
                output_transform=output,
                max_iter=30,
                tol=1e-9,
                max_final_plane_rms_mm=capture_config.max_final_plane_rms_mm,
                allow_partial=False,
                expected_plan_id=expected_plan_id,
            )
            return diagnostics, np.asarray(diagnostics["T_tcp_sensor"], dtype=float)

        def success(result):
            self.calibration_diagnostics, self.calibrated_T = result
            if not self.calibration_diagnostics.get("accepted", False):
                self.calibrated_T = None
                self.stage_var.set("CALIBRATION_REJECTED")
                messagebox.showerror(
                    "Calibration", "Solver result was not accepted; transform was not activated"
                )
                return
            self.stage_var.set("CALIBRATED")
            self._render_calibration_result()
            self._refresh_snapshot()

        self._submit("CALIBRATING", work, success)

    def _render_calibration_result(self) -> None:
        diagnostics = self.calibration_diagnostics
        self.calibration_text.delete("1.0", "end")
        if diagnostics is None:
            return
        T = np.asarray(diagnostics["T_tcp_sensor"], dtype=float)
        report = diagnostics["observability"]
        text = (
            f"solver: {diagnostics['solver']}\n"
            f"nonlinear refinement: {diagnostics['nonlinear_refinement']}\n"
            f"converged: {diagnostics['converged']}\n"
            f"iterations: {diagnostics['iterations']}\n"
            f"scans / points: {diagnostics['scan_count']} / {diagnostics['point_count']}\n"
            f"observability: rank {report['rank']}/{report['required_rank']}\n"
            f"normalized condition: {report['column_normalized_condition']:.6g}\n"
            f"initial plane RMS: {diagnostics['initial_plane_rms_mm']:.6f} mm\n"
            f"final plane RMS: {diagnostics['final_plane_rms_mm']:.6f} mm\n"
            f"acceptance RMS limit: {diagnostics['max_final_plane_rms_mm']:.6f} mm\n"
            f"linear multistart used: {diagnostics['linear_multistart_used']}\n\n"
            "T_tcp_sensor [mm]\n"
            + np.array2string(T, precision=8, suppress_small=True)
        )
        self.calibration_text.insert("1.0", text)
        render_calibration_axis(self.calibration_axis, diagnostics)
        self.calibration_canvas.draw_idle()

    def _clear_calibration_result(self) -> None:
        self.calibration_diagnostics = None
        self.calibrated_T = None
        if hasattr(self, "calibration_text"):
            self.calibration_text.delete("1.0", "end")
            render_calibration_axis(self.calibration_axis, None)
            self.calibration_canvas.draw_idle()

    # ----------------------------------------------------------- visualization
    def _populate_plan_tree(self) -> None:
        self.plan_tree.delete(*self.plan_tree.get_children())
        if self.plan is None:
            return
        for entry in self.plan["entries"]:
            scan_id = int(entry["scan_id"])
            self.plan_tree.insert(
                "",
                "end",
                iid=str(scan_id),
                values=(
                    "pending",
                    scan_id,
                    entry["line_id"],
                    f"{entry['d_mm']:.0f}",
                    f"{entry['theta_deg']:.0f}",
                    f"{entry['beta_deg']:.0f}",
                    "yes" if entry["reference_pose"] else "no",
                ),
            )

    def _effective_profile_error(
        self, sample: ProfileSample | None, error: str | None
    ) -> str | None:
        if error:
            return error
        if sample is None:
            return None
        maximum_age_ms = (
            1000.0
            if self.capture_config is None
            else float(self.capture_config.max_profile_age_ms)
        )
        age_ms = (time.time_ns() - int(sample.timestamp_ns)) / 1e6
        if age_ms > maximum_age_ms:
            return f"STALE: latest profile is older than {maximum_age_ms:.0f} ms"
        if age_ms < -10.0:
            return "ERROR: latest profile timestamp is in the future"
        return None

    def _refresh_snapshot(self) -> None:
        if self._closing_requested:
            return
        active_plan = self.session.plan if self.session is not None else self.plan
        if active_plan is None:
            return
        if self._snapshot_running:
            self._snapshot_pending = True
            return
        active_dataset = self._active_dataset_path()
        sequence, sample, error = (0, None, None)
        if self.profile_broker is not None:
            sequence, sample, error = self.profile_broker.latest()
        error = self._effective_profile_error(sample, error)
        requested_plan_id = str(active_plan.get("plan_id", ""))
        requested_dataset = str(active_dataset)
        calibrated_T = (
            None if self.calibrated_T is None else self.calibrated_T.copy()
        )
        handeye_source = "calibrated" if calibrated_T is not None else "initial"
        stage = self.stage_var.get()
        self._snapshot_running = True

        def worker() -> None:
            try:
                snapshot = build_dashboard_snapshot(
                    plan=active_plan,
                    dataset_dir=active_dataset,
                    live_sequence=sequence,
                    live_sample=sample,
                    live_error=error,
                    T_tcp_sensor=calibrated_T,
                    handeye_source=handeye_source,
                    stage=stage,
                )
                payload = (
                    requested_plan_id,
                    requested_dataset,
                    snapshot,
                    None,
                    None,
                )
            except Exception as exc:
                payload = (
                    requested_plan_id,
                    requested_dataset,
                    None,
                    exc,
                    traceback.format_exc(),
                )
            self._events.put(("snapshot", payload))

        threading.Thread(
            target=worker, name="ui-dashboard-snapshot", daemon=True
        ).start()

    def _refresh_live_profile(self) -> None:
        if self._closing:
            return
        broker = self.profile_broker
        if broker is not None:
            sequence, sample, error = broker.latest()
            effective_error = self._effective_profile_error(sample, error)
            changed = (
                sequence != self._last_live_sequence
                or effective_error != self._last_live_error
            )
            if changed:
                self._last_live_sequence = sequence
                self._last_live_error = effective_error
                if self.snapshot is not None:
                    points = np.empty((0, 3)) if sample is None else sample.points_s
                    points = np.asarray(points, dtype=float).copy()
                    points.setflags(write=False)
                    self.snapshot = replace(
                        self.snapshot,
                        live_sequence=sequence,
                        live_profile_s=points,
                        live_error=effective_error,
                    )
                    render_profile_axes(self.live_axis, self.previous_axis, self.snapshot)
                    self.profile_canvas.draw_idle()
                elif sample is not None:
                    self.live_axis.clear()
                    self.live_axis.plot(sample.points_s[:, 0], sample.points_s[:, 2])
                    self.live_axis.set_xlabel("Sensor X [mm]")
                    self.live_axis.set_ylabel("Sensor Z [mm]")
                    self.live_axis.grid(True, alpha=0.25)
                    self.profile_canvas.draw_idle()
            age_ms = (
                None
                if sample is None
                else (time.time_ns() - int(sample.timestamp_ns)) / 1e6
            )
            if effective_error:
                suffix = effective_error
                if age_ms is not None:
                    suffix += f" · last frame age {age_ms:.0f} ms"
            else:
                suffix = f"{0 if sample is None else len(sample.points_s)} points"
                if age_ms is not None:
                    suffix += f" · age {age_ms:.0f} ms"
            self.profile_var.set(f"Live seq {sequence} · {suffix}")
        self.after(100, self._refresh_live_profile)

    def _render_snapshot(self, *, full: bool) -> None:
        snapshot = self.snapshot
        if snapshot is None:
            return
        render_profile_axes(self.live_axis, self.previous_axis, snapshot)
        self.profile_canvas.draw_idle()
        if full:
            render_scene_axis(self.scene_axis, snapshot)
            self.scene_canvas.draw_idle()
            self._update_next_text(snapshot)
            self._update_plan_status(snapshot)
        percent = 100.0 * snapshot.completed_count / max(1, snapshot.total_count)
        self.progress_value.set(percent)
        if self.stage_var.get() not in {
            "MOVING",
            "SETTLING",
            "CAPTURING",
            "STOPPING",
        }:
            self.progress_var.set(
                f"captured {snapshot.completed_count}/{snapshot.total_count} · "
                f"rank {snapshot.observability_rank}/4 · "
                f"condition {snapshot.observability_condition:.3g}"
            )

    def _update_next_text(self, snapshot: DashboardSnapshot) -> None:
        self.next_text.delete("1.0", "end")
        estimate = snapshot.plane_estimate
        if snapshot.next_scan_id is None:
            next_text = "CAPTURE PLAN COMPLETE\n"
        else:
            values = snapshot.next_tcp_xyzrpy_deg
            next_text = (
                f"NEXT SCAN {snapshot.next_scan_id}/{snapshot.total_count - 1}\n"
                f"line={snapshot.next_line_id}  reference={snapshot.next_reference_pose}\n"
                f"d={snapshot.next_d_mm:.1f} mm  theta={snapshot.next_theta_deg:.1f}°  "
                f"beta={snapshot.next_beta_deg:.1f}°\n\n"
                "Target TCP [x y z rx ry rz]\n"
                + np.array2string(values, precision=4, suppress_small=True)
                + "\n\nSafe route XYZ [safe transit → approach → target]\n"
                + np.array2string(
                    np.vstack(
                        [
                            snapshot.safe_transit_T_base_tcp[:3, 3],
                            snapshot.next_T_base_tcp_approach[:3, 3],
                            snapshot.next_T_base_tcp[:3, 3],
                        ]
                    ),
                    precision=4,
                    suppress_small=True,
                )
                + "\n\nT_base_tcp\n"
                + np.array2string(snapshot.next_T_base_tcp, precision=5, suppress_small=True)
                + "\n"
            )
        plane_text = (
            f"\nACCUMULATED PLANE\n{estimate.status}\n"
            f"points={estimate.point_count}, distinct lines={estimate.distinct_line_count}\n"
        )
        if estimate.available:
            plane_text += (
                f"normal={np.array2string(estimate.normal_base, precision=6)}\n"
                f"offset={estimate.offset_mm:.4f} mm, RMS={estimate.rms_mm:.4f} mm\n"
            )
        if snapshot.data_errors:
            plane_text += "\nDATA WARNINGS\n" + "\n".join(snapshot.data_errors)
        self.next_text.insert("1.0", next_text + plane_text)

    def _update_plan_status(self, snapshot: DashboardSnapshot) -> None:
        completed = set(snapshot.completed_ids)
        for item in self.plan_tree.get_children():
            scan_id = int(item)
            values = list(self.plan_tree.item(item, "values"))
            if scan_id in completed:
                values[0] = "captured"
                tags = ("done",)
            elif scan_id == snapshot.next_scan_id:
                values[0] = "next"
                tags = ("next",)
            else:
                values[0] = "pending"
                tags = ()
            self.plan_tree.item(item, values=values, tags=tags)
        if snapshot.next_scan_id is not None and self.plan_tree.exists(str(snapshot.next_scan_id)):
            self.plan_tree.see(str(snapshot.next_scan_id))

    def _select_previous_capture(self, _event=None) -> None:
        if self.snapshot is None:
            return
        selection = self.plan_tree.selection()
        if not selection:
            return
        scan_id = int(selection[0])
        active_dataset = self._active_dataset_path()
        manifest_path = active_dataset / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = load_json(manifest_path)
            active_plan = self.session.plan if self.session is not None else self.plan
            if active_plan is None or manifest.get("plan_id") != active_plan.get("plan_id"):
                raise ValueError("dataset manifest belongs to another motion plan")
            record = next(
                item for item in manifest.get("scans", []) if int(item["scan_id"]) == scan_id
            )
            from .initial_point import load_profile

            points = load_profile(active_dataset / record["profile_file"])
            points.setflags(write=False)
            self.snapshot = replace(
                self.snapshot,
                previous_scan_id=scan_id,
                previous_profile_s=points,
            )
            render_profile_axes(self.live_axis, self.previous_axis, self.snapshot)
            self.profile_canvas.draw_idle()
        except (StopIteration, OSError, ValueError) as exc:
            self.progress_var.set(f"Cannot display scan {scan_id}: {exc}")

    def _active_dataset_path(self) -> Path:
        return (
            self.session.output_dir
            if self.session is not None
            else Path(self.dataset_dir.get())
        )

    # ---------------------------------------------------------------- close
    def _on_close(self) -> None:
        if self._closing_requested:
            return
        if not messagebox.askokcancel(
            "Exit",
            "실행 중인 동작을 software stop하고 UI를 종료합니다. 계속합니까?",
        ):
            return
        self._closing_requested = True
        self._auto_pause.set()
        self.stage_var.set("STOPPING")
        self.progress_var.set(
            "Waiting for motion worker to return before closing hardware…"
        )
        with self._job_lock:
            active_job_label = self._active_job_label
        session, robot = self.session, self.robot

        def stop() -> None:
            try:
                if session is not None:
                    session.request_stop()
                elif robot is not None:
                    robot.stop()
            except Exception as exc:
                print(f"close stop error: {exc}")

        stop_is_active = (
            self._stop_thread is not None and self._stop_thread.is_alive()
        )
        # CONNECTING sends no motion, and DISCONNECTING already owns the
        # stop→broker→laser→robot shutdown sequence. Starting another vendor
        # stop call here could overlap close() on a non-reentrant SDK.
        if not stop_is_active and active_job_label not in {
            "CONNECTING",
            "DISCONNECTING",
        }:
            self._stop_thread = threading.Thread(
                target=stop, name="ui-close-stop", daemon=True
            )
            self._stop_thread.start()
        self._close_requested_at = time.monotonic()
        self.after(100, self._finish_close_when_safe)

    def _finish_close_when_safe(self) -> None:
        worker_done = self._job_done.is_set()
        stop_done = self._stop_thread is None or not self._stop_thread.is_alive()
        if not (worker_done and stop_done):
            if (
                not self._close_wait_notice_shown
                and time.monotonic() - self._close_requested_at > 10.0
            ):
                self._close_wait_notice_shown = True
                self.progress_var.set(
                    "Still waiting for controller stop. Use the physical E-stop if required; "
                    "hardware connections will not be closed underneath an active move."
                )
            self.after(100, self._finish_close_when_safe)
            return
        if self.profile_broker is not None:
            try:
                self.profile_broker.stop(timeout_s=0.2)
            except Exception as exc:
                self.progress_var.set(
                    "Waiting for laser acquisition thread to stop before closing hardware…"
                )
                if not isinstance(exc, TimeoutError):
                    print(f"profile broker stop error: {exc}")
                self.after(100, self._finish_close_when_safe)
                return
        self._closing = True
        try:
            if self.laser is not None:
                self.laser.close()
        except Exception as exc:
            print(f"laser close error: {exc}")
        try:
            if self.robot is not None:
                self.robot.close()
        except Exception as exc:
            print(f"robot close error: {exc}")
        self.destroy()
