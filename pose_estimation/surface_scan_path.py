#!/usr/bin/env python3
"""
Manual CAD waypoints -> multiple independent surface-following scan segments.

Example
-------
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/surface_scan_path.py \
    /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --mesh-unit auto \
    --path-step-mm 1.0 \
    --sensor-standoff-mm 80 \
    --normal-radius-mm 3 \
    --orientation-smooth-window 35 \
    --gui \
    --save-json scan_path.json


"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import heapq
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise RuntimeError("Open3D is required: pip install open3d") from exc


EPS = 1.0e-12

# If the previously selected CAD triangle is still almost as close to the
# current path point as the newly reported nearest triangle, keep the previous
# face. This prevents A/B/A/B face flicker on triangle boundaries.
FACE_CONTINUITY_TOLERANCE_MM = 0.05


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise RuntimeError("Cannot normalize zero-length vector.")
    return v / n


def positive_float(v: str) -> float:
    x = float(v)
    if x <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def nonnegative_float(v: str) -> float:
    x = float(v)
    if x < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def positive_int(v: str) -> int:
    x = int(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def odd_positive_int(v: str) -> int:
    x = int(v)
    if x <= 0 or x % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return x


def weld_triangle_mesh(mesh, tolerance_m: float):
    """
    Weld nearly coincident STL vertices without depending on Open3D cleanup APIs.

    STL files are often triangle soups: two triangles can touch geometrically while
    still owning different vertex indices. A pure triangle-edge graph then appears
    disconnected. Quantizing vertices with a very small tolerance repairs that
    topological duplication while preserving the CAD geometry for path planning.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    T = np.asarray(mesh.triangles, dtype=np.int64)
    if len(V) == 0 or len(T) == 0:
        raise RuntimeError("Cannot weld an empty mesh.")

    tol = float(tolerance_m)
    if tol <= 0.0:
        out = o3d.geometry.TriangleMesh(mesh)
        out.compute_triangle_normals()
        out.compute_vertex_normals()
        return out, {
            "vertices_before": len(V),
            "vertices_after": len(V),
            "triangles_before": len(T),
            "triangles_after": len(T),
        }

    # Shift before quantization to keep integer magnitudes small even when the CAD
    # origin is far from the part. This does NOT change the final CAD coordinates.
    origin = np.min(V, axis=0)
    keys = np.rint((V - origin[None, :]) / tol).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_new = int(np.max(inverse)) + 1

    counts = np.bincount(inverse, minlength=n_new).astype(np.float64)
    V_new = np.zeros((n_new, 3), dtype=np.float64)
    for axis in range(3):
        V_new[:, axis] = np.bincount(
            inverse, weights=V[:, axis], minlength=n_new
        ) / np.maximum(counts, 1.0)

    T_new = inverse[T]

    # Remove triangles collapsed by welding.
    nondeg = (T_new[:, 0] != T_new[:, 1]) & (T_new[:, 1] != T_new[:, 2]) & (T_new[:, 2] != T_new[:, 0])
    T_new = T_new[nondeg]
    if len(T_new) == 0:
        raise RuntimeError(
            "All triangles collapsed during vertex welding. Reduce --weld-tolerance-mm."
        )

    # Remove duplicate triangles while preserving the winding of the first copy.
    canonical = np.sort(T_new, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    T_new = T_new[np.sort(first)]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(V_new)
    out.triangles = o3d.utility.Vector3iVector(T_new.astype(np.int32))
    out.compute_triangle_normals()
    out.compute_vertex_normals()

    return out, {
        "vertices_before": len(V),
        "vertices_after": len(V_new),
        "triangles_before": len(T),
        "triangles_after": len(T_new),
    }


def load_mesh_preserve_frame(path: Path, mesh_unit: str, weld_tolerance_mm: float):
    """Load mesh, convert to meters, preserve CAD origin, then weld STL seams."""
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Failed to load triangle mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent_raw = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag_raw = float(np.linalg.norm(extent_raw))

    if mesh_unit == "m":
        scale = 1.0
        unit_label = "m"
    elif mesh_unit == "mm":
        scale = 0.001
        unit_label = "mm"
    else:
        scale = 0.001 if diag_raw > 10.0 else 1.0
        unit_label = "mm(auto)" if scale == 0.001 else "m(auto)"

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices[:] *= scale

    mesh, weld_info = weld_triangle_mesh(
        mesh, tolerance_m=float(weld_tolerance_mm) / 1000.0
    )

    bbox_m = mesh.get_axis_aligned_bounding_box()
    center_m = np.asarray(bbox_m.get_center(), dtype=np.float64)
    extent_m = np.asarray(bbox_m.get_extent(), dtype=np.float64)

    return mesh, {
        "input_unit": unit_label,
        "scale_to_m": scale,
        "diameter_mm": float(np.linalg.norm(extent_m)) * 1000.0,
        "cad_center_m": center_m,
        "weld_tolerance_mm": float(weld_tolerance_mm),
        **weld_info,
    }


# -----------------------------------------------------------------------------
# Waypoint picking
# -----------------------------------------------------------------------------


def _camera_center_from_visualizer(vis):
    """Return the current Open3D camera center in CAD/world coordinates.

    This is used only to disambiguate front/back waypoint picks. If the current
    Open3D build cannot provide camera parameters, return None and keep the
    original picked point unchanged rather than breaking the picker.
    """
    try:
        view = vis.get_view_control()
        params = view.convert_to_pinhole_camera_parameters()
        E = np.asarray(params.extrinsic, dtype=np.float64)
        R = E[:3, :3]
        t = E[:3, 3]
        return -R.T @ t
    except Exception as exc:
        print(f"  [WARN] Could not read picker camera pose; visible-surface correction disabled: {exc}")
        return None


def _correct_picks_to_first_visible_surface(mesh, clicked_points: np.ndarray, camera_center):
    """Move each picked sample to the first CAD surface hit from the camera.

    VisualizerWithEditing selects from a dense point cloud, so a rear-side sample
    can occasionally be picked through the visible surface. The selected sample
    still gives the intended viewing ray. Ray-casting that direction against the
    triangle mesh and taking only the first hit resolves the ambiguity.

    This function does NOT change path planning: the corrected hit is still
    snapped to the nearest welded mesh vertex by the existing code afterwards.
    """
    P = np.asarray(clicked_points, dtype=np.float64).reshape(-1, 3)
    if len(P) == 0 or camera_center is None:
        return P.copy(), np.zeros(len(P), dtype=np.float64)

    cam = np.asarray(camera_center, dtype=np.float64).reshape(3)
    directions = P - cam[None, :]
    lengths = np.linalg.norm(directions, axis=1)
    valid_dir = np.isfinite(lengths) & (lengths > EPS)
    directions[valid_dir] /= lengths[valid_dir, None]

    corrected = P.copy()
    correction_m = np.zeros(len(P), dtype=np.float64)

    try:
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tensor_mesh)

        valid_ids = np.flatnonzero(valid_dir)
        if len(valid_ids) == 0:
            return corrected, correction_m

        origins = np.repeat(cam[None, :], len(valid_ids), axis=0)
        rays_np = np.hstack([origins, directions[valid_ids]]).astype(np.float32)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        t_hit = np.asarray(ans["t_hit"].numpy(), dtype=np.float64).reshape(-1)

        for local_i, point_i in enumerate(valid_ids):
            t = float(t_hit[local_i])
            if np.isfinite(t) and t >= 0.0:
                hit = cam + t * directions[point_i]
                corrected[point_i] = hit
                correction_m[point_i] = float(np.linalg.norm(hit - P[point_i]))

    except Exception as exc:
        print(f"  [WARN] Visible-surface ray correction unavailable; keeping raw picks: {exc}")
        return P.copy(), np.zeros(len(P), dtype=np.float64)

    return corrected, correction_m



def _pick_visible_surface_points_gui(
    mesh,
    previous_points: np.ndarray | None = None,
    segment_id: int = 0,
    marker_radius_mm: float = 1.2,
) -> np.ndarray:
    """Interactive CAD picker with *live* visible-surface waypoint markers.

    Unlike VisualizerWithEditing, this picker never selects from a dense point
    cloud behind the mesh.  Shift+LeftClick reads the rendered CAD depth at the
    clicked pixel and unprojects that pixel to the first visible CAD surface.
    The waypoint marker is added immediately at that 3-D point.

    Controls
    --------
      Shift + Left click  : add visible-surface waypoint
      Shift + Right click : undo the latest waypoint
      Q or Finish button  : finish this segment
      Mouse drag / wheel  : rotate / zoom the camera normally

    Closing/finishing with zero points is valid; the parent process uses that to
    terminate multi-segment waypoint input.
    """
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering

    previous = (
        np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
        if previous_points is not None
        else np.empty((0, 3), dtype=np.float64)
    )

    app = gui.Application.instance
    app.initialize()
    window = app.create_window(
        f"Pick scan segment S{int(segment_id)} | live visible-surface picking",
        1500,
        920,
    )

    em = float(window.theme.font_size)
    panel_width = int(max(330, 22 * em))

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.96, 0.97, 0.98, 1.0])
    scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    panel = gui.Vert(
        0.45 * em,
        gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em),
    )
    title = gui.Label(f"Scan segment S{int(segment_id)}")
    instructions = gui.Label(
        "Shift + Left click : add waypoint\n"
        "Shift + Right click: undo\n"
        "Q / Finish         : finish this segment\n"
        "Drag / wheel       : rotate / zoom\n\n"
        "Markers are placed on the first visible CAD surface."
    )
    count_label = gui.Label("Current waypoints: 0")
    coord_label = gui.Label("Last waypoint: -")
    status_label = gui.Label("Ready")

    controls = gui.Horiz(0.35 * em)
    undo_btn = gui.Button("Undo")
    finish_btn = gui.Button("Finish segment")
    controls.add_child(undo_btn)
    controls.add_child(finish_btn)

    panel.add_child(title)
    panel.add_child(instructions)
    panel.add_child(count_label)
    panel.add_child(coord_label)
    panel.add_child(status_label)
    panel.add_child(controls)

    window.add_child(scene_widget)
    window.add_child(panel)

    # Render the actual triangle mesh, not a sampled point cloud.  Therefore the
    # depth buffer corresponds to the front-most visible CAD surface pixel.
    cad = o3d.geometry.TriangleMesh(mesh)
    if not cad.has_vertex_normals():
        cad.compute_vertex_normals()
    cad.paint_uniform_color([0.66, 0.66, 0.69])

    cad_mat = rendering.MaterialRecord()
    cad_mat.shader = "defaultLit"
    cad_mat.base_color = [0.66, 0.66, 0.69, 1.0]
    scene_widget.scene.add_geometry("CAD", cad, cad_mat)

    prev_mat = rendering.MaterialRecord()
    prev_mat.shader = "defaultLit"
    prev_mat.base_color = [0.15, 0.90, 0.25, 1.0]
    current_mat = rendering.MaterialRecord()
    current_mat.shader = "defaultLit"
    current_mat.base_color = [1.0, 0.68, 0.05, 1.0]

    marker_radius_m = max(float(marker_radius_mm), 0.1) / 1000.0

    # Previous scan strokes remain visible as green spheres for context.
    for i, p in enumerate(previous):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius_m)
        sphere.translate(p)
        sphere.compute_vertex_normals()
        scene_widget.scene.add_geometry(f"Previous waypoint {i}", sphere, prev_mat)

    bounds = cad.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    scene_widget.setup_camera(55.0, bounds, center)

    state = {
        "points": [],
        "marker_names": [],
        "labels": [],
        "closed": False,
    }

    def update_text():
        count_label.text = f"Current waypoints: {len(state['points'])}"
        if state["points"]:
            p = 1000.0 * np.asarray(state["points"][-1])
            coord_label.text = (
                f"Last waypoint [mm]: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}"
            )
        else:
            coord_label.text = "Last waypoint: -"
        scene_widget.force_redraw()

    def add_waypoint(world):
        if state["closed"]:
            return
        p = np.asarray(world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            status_label.text = "Ignored invalid pick"
            return

        idx = len(state["points"])
        state["points"].append(p.copy())
        name = f"Current waypoint {idx}"
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius_m)
        sphere.translate(p)
        sphere.compute_vertex_normals()
        scene_widget.scene.add_geometry(name, sphere, current_mat)
        state["marker_names"].append(name)

        try:
            label = scene_widget.add_3d_label(p, f"W{idx}")
        except Exception:
            label = None
        state["labels"].append(label)

        status_label.text = f"Added W{idx} on visible CAD surface"
        update_text()

    def undo_waypoint():
        if not state["points"]:
            status_label.text = "Nothing to undo"
            return

        idx = len(state["points"]) - 1
        state["points"].pop()
        name = state["marker_names"].pop()
        try:
            scene_widget.scene.remove_geometry(name)
        except Exception:
            pass

        label = state["labels"].pop()
        if label is not None:
            try:
                scene_widget.remove_3d_label(label)
            except Exception:
                pass

        status_label.text = f"Removed W{idx}"
        update_text()

    def finish_picker():
        # Do NOT call Application.quit() here. Some Open3D/Linux builds do not
        # reliably return from app.run() when quit/close is requested inside a
        # key callback. Instead, set a Python-side flag. The explicit
        # run_one_tick() loop below observes this flag and exits deterministically.
        if state["closed"]:
            return
        state["closed"] = True
        status_label.text = "Finishing segment..."
        print(f"  [PICKER] Finish requested for S{int(segment_id)}; "
              f"waypoints={len(state['points'])}", flush=True)

    def on_mouse(event):
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if not event.is_modifier_down(gui.KeyModifier.SHIFT):
            return gui.Widget.EventCallbackResult.IGNORED

        if event.is_button_down(gui.MouseButton.RIGHT):
            undo_waypoint()
            return gui.Widget.EventCallbackResult.HANDLED

        if not event.is_button_down(gui.MouseButton.LEFT):
            return gui.Widget.EventCallbackResult.IGNORED

        # Mouse-event coordinates are absolute window coordinates. Convert them
        # to the SceneWidget's local pixel coordinates before indexing depth.
        x = int(round(event.x - scene_widget.frame.x))
        y = int(round(event.y - scene_widget.frame.y))
        width = max(1, int(scene_widget.frame.width))
        height = max(1, int(scene_widget.frame.height))

        if x < 0 or y < 0 or x >= width or y >= height:
            return gui.Widget.EventCallbackResult.HANDLED

        status_label.text = "Picking visible surface..."

        def depth_callback(depth_image):
            depth_np = np.asarray(depth_image)
            if (
                y < 0 or x < 0
                or y >= depth_np.shape[0]
                or x >= depth_np.shape[1]
            ):
                world = None
            else:
                depth = float(depth_np[y, x])
                if not np.isfinite(depth) or depth >= 0.999999:
                    world = None
                else:
                    world = np.asarray(
                        scene_widget.scene.camera.unproject(
                            x, y, depth, width, height
                        ),
                        dtype=np.float64,
                    )

            def commit_pick():
                if state["closed"]:
                    return
                if world is None:
                    status_label.text = "No CAD surface at clicked pixel"
                    return
                add_waypoint(world)

            gui.Application.instance.post_to_main_thread(window, commit_pick)

        # Rendered depth is a z-buffer: the returned depth is necessarily the
        # first visible surface at this screen pixel, so a rear face cannot win.
        scene_widget.scene.scene.render_to_depth_image(depth_callback)
        return gui.Widget.EventCallbackResult.HANDLED

    def _is_q_key(event) -> bool:
        try:
            if event.type != gui.KeyEvent.Type.DOWN:
                return False
        except Exception:
            pass

        key = event.key
        q_key = getattr(gui.KeyName, "Q", None)
        if q_key is not None and key == q_key:
            return True
        try:
            return int(key) in (ord("Q"), ord("q"))
        except Exception:
            return False

    def on_scene_key(event):
        # Keep the SceneWidget handler as a fallback when it owns keyboard focus.
        if _is_q_key(event):
            finish_picker()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def on_window_key(event):
        # Window-level interception sees the key before it is dispatched to the
        # focused widget. This remains active even after using a side-panel button.
        if _is_q_key(event):
            finish_picker()
            return True
        return False

    def on_close():
        # Closing the OS window is treated exactly like Finish/Q.
        if not state["closed"]:
            state["closed"] = True
            print(f"  [PICKER] Window close requested for S{int(segment_id)}; "
                  f"waypoints={len(state['points'])}", flush=True)
        return True

    def on_layout(_layout_context):
        r = window.content_rect
        scene_widget.frame = gui.Rect(
            r.x,
            r.y,
            max(1, r.width - panel_width),
            r.height,
        )
        panel.frame = gui.Rect(
            r.get_right() - panel_width,
            r.y,
            panel_width,
            r.height,
        )

    scene_widget.set_on_mouse(on_mouse)
    scene_widget.set_on_key(on_scene_key)
    window.set_on_key(on_window_key)
    window.set_on_close(on_close)
    undo_btn.set_on_clicked(undo_waypoint)
    finish_btn.set_on_clicked(finish_picker)
    window.set_on_layout(on_layout)

    # Make the 3-D view own keyboard focus initially. Window.set_on_key() still
    # intercepts Q globally, but explicit focus also keeps SceneWidget shortcuts
    # reliable on older Open3D builds.
    try:
        window.set_focus_widget(scene_widget)
    except Exception:
        pass

    # IMPORTANT: do not use app.run() here. With some Linux/Open3D combinations
    # a close/quit request issued from a key callback leaves app.run() blocked.
    # Driving the event loop one tick at a time lets Python observe state["closed"]
    # and return to run_picker_worker() deterministically, which then writes the
    # handoff JSON used by the parent process to open S1 / generate the path.
    while not state["closed"]:
        alive = app.run_one_tick()
        if not alive:
            break

    try:
        window.close()
    except Exception:
        pass

    points = np.asarray(state["points"], dtype=np.float64).reshape(-1, 3)
    print(f"  [PICKER] Returning S{int(segment_id)} with {len(points)} waypoint(s)",
          flush=True)
    return points


def pick_waypoints(
    mesh,
    point_size: float = 2.0,
    sample_points: int = 100000,
):
    """Pick ordered waypoints directly on the first visible CAD surface.

    ``sample_points`` is retained for backward API compatibility but is no longer
    used by the live picker.  The picker renders the triangle mesh itself and
    unprojects the clicked depth pixel, so the marker appears immediately on the
    visible/front-most surface rather than on a rear point-cloud sample.
    """
    del sample_points  # compatibility-only argument

    V = np.asarray(mesh.vertices, dtype=np.float64)
    if len(V) < 2:
        raise RuntimeError("Mesh has too few vertices.")

    clicked = _pick_visible_surface_points_gui(
        mesh=mesh,
        previous_points=None,
        segment_id=0,
        marker_radius_mm=max(0.8, 0.55 * float(point_size)),
    )
    if len(clicked) < 2:
        raise RuntimeError(f"Need at least 2 waypoints, but got {len(clicked)}.")

    # Keep the established topology-safe planning behavior: live visible-surface
    # picks are still snapped to welded mesh vertices before Dijkstra planning.
    snapped_ids = []
    snap_distances = []
    for pnt in clicked:
        d2 = np.sum((V - pnt[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        snapped_ids.append(idx)
        snap_distances.append(float(np.sqrt(d2[idx])))

    cleaned_ids = [snapped_ids[0]]
    cleaned_d = [snap_distances[0]]
    for idx, sd in zip(snapped_ids[1:], snap_distances[1:]):
        if idx != cleaned_ids[-1]:
            cleaned_ids.append(idx)
            cleaned_d.append(sd)

    if len(cleaned_ids) < 2:
        raise RuntimeError(
            "Waypoint selection collapsed to fewer than 2 unique mesh vertices. "
            "Choose points farther apart or use a finer CAD mesh."
        )

    for i, (idx, sd) in enumerate(zip(cleaned_ids, cleaned_d)):
        print(
            f"  W{i}: visible-surface pick -> mesh vertex {idx}, "
            f"vertex-snap={sd * 1000.0:.4f} mm"
        )

    ids = np.asarray(cleaned_ids, dtype=np.int64)
    return ids, V[ids]


def pick_waypoint_segments(
    mesh,
    point_size: float = 2.0,
    sample_points: int = 100000,
):
    """
    Pick one or more INDEPENDENT scan segments.

    Each picker session defines one scan segment.  Waypoints inside a session are
    connected in order, but the last waypoint of one session is NEVER connected to
    the first waypoint of the next session.

    Workflow
    --------
      1) Pick S0 waypoints -> Q
      2) the next picker opens automatically for S1
      3) repeat as needed
      4) press Q with NO points in a fresh picker to finish all input

    This is intentionally simple and robust with Open3D VisualizerWithEditing,
    which has no native "break polyline here" interaction.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    if len(V) < 2:
        raise RuntimeError("Mesh has too few vertices.")

    n_samples = max(int(sample_points), len(V))
    sampled = mesh.sample_points_uniformly(
        number_of_points=n_samples,
        use_triangle_normal=True,
    )
    if sampled.is_empty():
        raise RuntimeError("Failed to sample CAD surface for waypoint picking.")

    S = np.asarray(sampled.points, dtype=np.float64).copy()
    previous_sample_ids: list[int] = []
    segments: list[dict] = []

    while True:
        seg_idx = len(segments)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(S)
        colors = np.repeat(
            np.asarray([[0.72, 0.72, 0.75]], dtype=np.float64),
            len(S),
            axis=0,
        )
        if previous_sample_ids:
            colors[np.asarray(previous_sample_ids, dtype=np.int64)] = np.asarray(
                [0.15, 0.95, 0.25], dtype=np.float64
            )
        cloud.colors = o3d.utility.Vector3dVector(colors)

        print(f"\n[PICK SCAN SEGMENT S{seg_idx}]")
        print("  Shift + LEFT CLICK : add waypoint to THIS segment")
        print("  Shift + RIGHT CLICK: undo")
        print("  Q                   : finish THIS segment")
        print("  Pick at least TWO points in scan order.")
        if segments:
            print("  Green points are waypoints selected in previous segments.")
        print(f"  Picker surface samples: {len(S):,}")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name=f"Pick scan segment S{seg_idx} | Shift+LeftClick -> Q",
            width=1440,
            height=900,
        )
        vis.add_geometry(cloud)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.025, 0.025, 0.03])
        opt.point_size = float(point_size)
        vis.run()
        picked_sample_ids = [int(i) for i in vis.get_picked_points()]
        camera_center = _camera_center_from_visualizer(vis)
        vis.destroy_window()

        if len(picked_sample_ids) == 0:
            if segments:
                print("  No points selected -> finishing segment input.")
                break
            raise RuntimeError("No scan segment was selected.")

        if len(picked_sample_ids) < 2:
            print(
                f"  [WARN] S{seg_idx} needs at least 2 points; "
                f"got {len(picked_sample_ids)}. Please pick this segment again."
            )
            continue

        raw_clicked = S[np.asarray(picked_sample_ids, dtype=np.int64)]
        clicked, visible_correction_m = _correct_picks_to_first_visible_surface(
            mesh, raw_clicked, camera_center
        )

        snapped_ids = []
        snap_distances = []
        for p in clicked:
            d2 = np.sum((V - p[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            snapped_ids.append(idx)
            snap_distances.append(float(np.sqrt(d2[idx])))

        cleaned_ids = [snapped_ids[0]]
        cleaned_samples = [picked_sample_ids[0]]
        cleaned_d = [snap_distances[0]]
        cleaned_visible_d = [visible_correction_m[0]]
        for idx, sample_id, sd, vd in zip(
            snapped_ids[1:],
            picked_sample_ids[1:],
            snap_distances[1:],
            visible_correction_m[1:],
        ):
            if idx != cleaned_ids[-1]:
                cleaned_ids.append(idx)
                cleaned_samples.append(int(sample_id))
                cleaned_d.append(sd)
                cleaned_visible_d.append(vd)

        if len(cleaned_ids) < 2:
            print(
                f"  [WARN] S{seg_idx} collapsed to fewer than 2 unique mesh vertices. "
                "Pick points farther apart."
            )
            continue

        ids = np.asarray(cleaned_ids, dtype=np.int64)
        points = V[ids]
        segments.append(
            {
                "segment_id": seg_idx,
                "waypoint_vertex_ids": ids,
                "waypoint_points": points,
                "picked_sample_ids": np.asarray(cleaned_samples, dtype=np.int64),
            }
        )
        previous_sample_ids.extend(cleaned_samples)

        for local_i, (idx, sd, vd) in enumerate(
            zip(cleaned_ids, cleaned_d, cleaned_visible_d)
        ):
            print(
                f"  S{seg_idx}-W{local_i}: mesh vertex {idx}, "
                f"visible-correction={vd * 1000.0:.4f} mm, "
                f"vertex-snap={sd * 1000.0:.4f} mm"
            )

        print(
            f"  S{seg_idx} saved. The next picker window defines S{seg_idx + 1}."
        )
        print("  To finish all segment input, press Q in the next window WITHOUT picking points.")

    return segments


def _write_json_atomic(path: Path, data):
    """Write a small JSON handoff file used by picker subprocesses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_picker_worker(args):
    """Open one fresh live visible-surface picker and return picked CAD points.

    Each scan segment still gets its own subprocess, preserving the previous
    Open3D/GLFW lifecycle workaround.  The only behavioral change is picking:
    markers now appear immediately on the front-most visible CAD surface.
    """
    mesh, _info = load_mesh_preserve_frame(
        args.cad, args.mesh_unit, args.weld_tolerance_mm
    )

    previous_points = np.empty((0, 3), dtype=np.float64)
    if args.picker_worker_previous_json is not None:
        prev_path = Path(args.picker_worker_previous_json)
        if prev_path.exists():
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            previous_points = np.asarray(
                prev_data.get("previous_waypoints_m", []), dtype=np.float64
            ).reshape(-1, 3)

    sid = int(args.picker_worker_segment_id)
    print(f"\n[PICK SCAN SEGMENT S{sid}]")
    print("  Shift + LEFT CLICK : add waypoint on FIRST VISIBLE CAD surface")
    print("  Shift + RIGHT CLICK: undo")
    print("  Q / Finish button  : finish THIS segment")
    print("  Mouse drag / wheel : rotate / zoom")
    if len(previous_points):
        print("  Green spheres       : waypoints from previous segments")
    print("  Orange spheres      : current segment waypoints (shown immediately)")
    print("  Empty fresh segment + Finish/Q = finish all segment input")

    clicked = _pick_visible_surface_points_gui(
        mesh=mesh,
        previous_points=previous_points,
        segment_id=sid,
        marker_radius_mm=max(0.8, 0.55 * float(args.picker_point_size)),
    )

    output_path = Path(args.picker_worker_output)
    _write_json_atomic(
        output_path,
        {
            "segment_id": sid,
            "clicked_points_m": clicked.tolist(),
            "picked_count": int(len(clicked)),
            "picker_mode": "live_depth_first_visible_surface",
        },
    )
    print(f"  [PICKER] Handoff saved: {output_path}", flush=True)


def _launch_one_segment_picker(
    args,
    segment_id: int,
    previous_waypoints: np.ndarray,
) -> np.ndarray:
    """Launch one fresh picker process and return clicked 3-D CAD points."""
    script = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="surface_scan_picker_") as td:
        td = Path(td)
        output_json = td / "picked.json"
        previous_json = td / "previous.json"
        _write_json_atomic(
            previous_json,
            {
                "previous_waypoints_m": np.asarray(
                    previous_waypoints, dtype=np.float64
                ).reshape(-1, 3).tolist()
            },
        )

        cmd = [
            sys.executable,
            str(script),
            str(args.cad),
            "--mesh-unit", str(args.mesh_unit),
            "--weld-tolerance-mm", str(args.weld_tolerance_mm),
            "--picker-point-size", str(args.picker_point_size),
            "--picker-sample-points", str(args.picker_sample_points),
            "--picker-worker-output", str(output_json),
            "--picker-worker-segment-id", str(int(segment_id)),
            "--picker-worker-previous-json", str(previous_json),
        ]

        print(f"\n[OPEN PICKER S{segment_id} IN FRESH PROCESS]")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Segment picker S{segment_id} exited with code {result.returncode}."
            )
        if not output_json.exists():
            raise RuntimeError(
                f"Segment picker S{segment_id} closed without returning picked points."
            )

        data = json.loads(output_json.read_text(encoding="utf-8"))
        return np.asarray(data.get("clicked_points_m", []), dtype=np.float64).reshape(-1, 3)


def pick_waypoint_segments_fresh_processes(mesh, args):
    """
    Pick multiple independent scan strokes, one fresh Open3D process per stroke.

    Each worker uses live depth-buffer visible-surface picking. This preserves the
    simple workflow (S0 -> Q, S1 -> Q, ...), while avoiding
    the black second-picker window observed when multiple legacy Open3D picker
    windows are created sequentially inside one process.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    if len(V) < 2:
        raise RuntimeError("Mesh has too few vertices.")

    segments: list[dict] = []
    previous_waypoints: list[np.ndarray] = []

    while True:
        sid = len(segments)
        clicked = _launch_one_segment_picker(
            args,
            segment_id=sid,
            previous_waypoints=(
                np.vstack(previous_waypoints)
                if previous_waypoints
                else np.empty((0, 3), dtype=np.float64)
            ),
        )

        if len(clicked) == 0:
            if not segments:
                raise RuntimeError("No scan segment was selected.")
            print(f"  S{sid}: no points selected -> finishing segment input.")
            break

        if len(clicked) < 2:
            print(
                f"  [WARN] S{sid} needs at least 2 waypoints; got {len(clicked)}. "
                "Reopening the same segment in a fresh picker."
            )
            continue

        snapped_ids = []
        snap_distances = []
        for p in clicked:
            d2 = np.sum((V - p[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            snapped_ids.append(idx)
            snap_distances.append(float(np.sqrt(d2[idx])))

        cleaned_ids = [snapped_ids[0]]
        cleaned_d = [snap_distances[0]]
        for idx, sd in zip(snapped_ids[1:], snap_distances[1:]):
            if idx != cleaned_ids[-1]:
                cleaned_ids.append(idx)
                cleaned_d.append(sd)

        if len(cleaned_ids) < 2:
            print(
                f"  [WARN] S{sid} collapsed to fewer than 2 unique mesh vertices. "
                "Pick points farther apart."
            )
            continue

        ids = np.asarray(cleaned_ids, dtype=np.int64)
        points = V[ids]
        spec = {
            "segment_id": sid,
            "waypoint_vertex_ids": ids,
            "waypoint_points": points,
        }
        segments.append(spec)
        previous_waypoints.append(points.copy())

        for local_i, (idx, sd) in enumerate(zip(cleaned_ids, cleaned_d)):
            print(
                f"  S{sid}-W{local_i}: mesh vertex {idx}, "
                f"snap={sd * 1000.0:.4f} mm"
            )
        print(f"  S{sid} saved. Next picker = S{sid + 1}.")
        print(
            f"  To finish after S{sid}, press Q with NO points in the fresh S{sid + 1} picker."
        )

    return segments


# -----------------------------------------------------------------------------
# Mesh graph / surface path
# -----------------------------------------------------------------------------


def build_mesh_edge_graph(vertices: np.ndarray, triangles: np.ndarray):
    """Build undirected weighted adjacency from triangle edges."""
    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64)
    n = len(V)
    neighbors: list[dict[int, float]] = [dict() for _ in range(n)]

    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            u = int(u)
            v = int(v)
            if u == v:
                continue
            w = float(np.linalg.norm(V[u] - V[v]))
            if w <= EPS:
                continue
            old = neighbors[u].get(v)
            if old is None or w < old:
                neighbors[u][v] = w
                neighbors[v][u] = w

    return [list(d.items()) for d in neighbors]


def graph_component_labels(adjacency):
    """Return connected-component labels for a mesh adjacency list."""
    n = len(adjacency)
    labels = np.full(n, -1, dtype=np.int64)
    sizes = []
    comp = 0
    for seed in range(n):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = comp
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v, _ in adjacency[u]:
                if labels[v] < 0:
                    labels[v] = comp
                    stack.append(v)
        sizes.append(size)
        comp += 1
    return labels, np.asarray(sizes, dtype=np.int64)


def dijkstra_vertex_path(adjacency, start: int, goal: int) -> np.ndarray:
    """Shortest path on mesh-edge graph from start vertex to goal vertex."""
    start = int(start)
    goal = int(goal)
    if start == goal:
        return np.asarray([start], dtype=np.int64)

    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)

    dist[start] = 0.0
    heap = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == goal:
            break

        for v, w in adjacency[u]:
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[goal]):
        raise RuntimeError(
            f"No surface path between mesh vertices {start} and {goal}. "
            "After STL vertex welding they are still in different mesh components. "
            "If these surfaces are truly connected in CAD, try a slightly larger "
            "--weld-tolerance-mm (for example 0.01)."
        )

    rev = [goal]
    cur = goal
    while cur != start:
        cur = int(prev[cur])
        if cur < 0:
            raise RuntimeError("Internal shortest-path reconstruction failure.")
        rev.append(cur)
    rev.reverse()
    return np.asarray(rev, dtype=np.int64)


def concatenate_waypoint_paths(adjacency, waypoint_vertex_ids: np.ndarray) -> np.ndarray:
    all_ids: list[int] = []
    ids = [int(x) for x in waypoint_vertex_ids]

    for i in range(len(ids) - 1):
        seg = dijkstra_vertex_path(adjacency, ids[i], ids[i + 1])
        if i > 0:
            seg = seg[1:]  # avoid duplicate junction vertex
        all_ids.extend(int(x) for x in seg)

    if len(all_ids) < 2:
        raise RuntimeError("Generated surface path is too short.")
    return np.asarray(all_ids, dtype=np.int64)


# -----------------------------------------------------------------------------
# Arc-length resampling + normal interpolation
# -----------------------------------------------------------------------------


def remove_zero_length_segments(points: np.ndarray, normals: np.ndarray):
    P = np.asarray(points, dtype=np.float64)
    N = np.asarray(normals, dtype=np.float64)
    if len(P) != len(N):
        raise ValueError("points/normals length mismatch")

    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1.0e-10:
            keep.append(i)

    if len(keep) < 2:
        raise RuntimeError("Surface path has zero total length.")
    keep = np.asarray(keep, dtype=np.int64)
    return P[keep], N[keep]


def resample_polyline_with_normals(
    points: np.ndarray,
    normals: np.ndarray,
    step_mm: float,
):
    """Uniform arc-length resampling with linear normal interpolation."""
    P, N = remove_zero_length_segments(points, normals)

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    step = float(step_mm) / 1000.0

    count = max(2, int(np.floor(total / step)) + 1)
    sample_s = np.arange(count, dtype=np.float64) * step
    if sample_s[-1] < total - 1.0e-10:
        sample_s = np.concatenate([sample_s, [total]])
    else:
        sample_s[-1] = total

    out_P = np.empty((len(sample_s), 3), dtype=np.float64)
    out_N = np.empty((len(sample_s), 3), dtype=np.float64)

    j = 0
    for k, sk in enumerate(sample_s):
        while j < len(seg) - 1 and sk > s[j + 1]:
            j += 1
        ds = float(s[j + 1] - s[j])
        a = 0.0 if ds <= EPS else float((sk - s[j]) / ds)
        a = min(max(a, 0.0), 1.0)

        out_P[k] = (1.0 - a) * P[j] + a * P[j + 1]
        n = (1.0 - a) * N[j] + a * N[j + 1]
        if np.linalg.norm(n) <= EPS:
            n = N[j]
        out_N[k] = normalize(n)

    return out_P, out_N, sample_s



def resample_polyline(points: np.ndarray, step_mm: float):
    """Uniform arc-length resampling of a 3-D polyline."""
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("points must be N x 3 with N >= 2")

    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1.0e-10:
            keep.append(i)
    P = P[np.asarray(keep, dtype=np.int64)]
    if len(P) < 2:
        raise RuntimeError("Surface path has zero total length.")

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    step = float(step_mm) / 1000.0
    if step <= 0.0:
        raise ValueError("step_mm must be positive")

    count = max(2, int(np.floor(total / step)) + 1)
    sample_s = np.arange(count, dtype=np.float64) * step
    if sample_s[-1] < total - 1.0e-10:
        sample_s = np.concatenate([sample_s, [total]])
    else:
        sample_s[-1] = total

    out = np.empty((len(sample_s), 3), dtype=np.float64)
    j = 0
    for k, sk in enumerate(sample_s):
        while j < len(seg) - 1 and sk > s[j + 1]:
            j += 1
        ds = float(s[j + 1] - s[j])
        a = 0.0 if ds <= EPS else float((sk - s[j]) / ds)
        a = min(max(a, 0.0), 1.0)
        out[k] = (1.0 - a) * P[j] + a * P[j + 1]

    return out, sample_s


def build_mesh_face_normal_query(mesh):
    """Build nearest-triangle queries directly on the original CAD mesh."""
    legacy = o3d.geometry.TriangleMesh(mesh)
    legacy.compute_triangle_normals()

    V = np.asarray(legacy.vertices, dtype=np.float64).copy()
    T = np.asarray(legacy.triangles, dtype=np.int64).copy()
    triangle_normals = np.asarray(legacy.triangle_normals, dtype=np.float64).copy()

    if len(triangle_normals) != len(T):
        raise RuntimeError("Triangle normals are unavailable.")

    mag = np.linalg.norm(triangle_normals, axis=1)
    if np.any(~np.isfinite(mag)) or np.any(mag <= EPS):
        raise RuntimeError("CAD contains invalid triangle normals.")
    triangle_normals /= mag[:, None]

    triangle_vertices = V[T]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene, triangle_normals, triangle_vertices


def _point_triangle_distance_sq(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """Squared Euclidean distance from point p to triangle (a,b,c)."""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.dot(ap, ap))

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.dot(bp, bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, EPS)
        q = a + v * ab
        diff = p - q
        return float(np.dot(diff, diff))

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.dot(cp, cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, EPS)
        q = a + w * ac
        diff = p - q
        return float(np.dot(diff, diff))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / max(denom, EPS)
        q = b + w * (c - b)
        diff = p - q
        return float(np.dot(diff, diff))

    denom = va + vb + vc
    if abs(denom) <= EPS:
        # Degenerate fallback; CAD cleanup should normally prevent this.
        return min(
            float(np.dot(p - a, p - a)),
            float(np.dot(p - b, p - b)),
            float(np.dot(p - c, p - c)),
        )

    inv = 1.0 / denom
    v = vb * inv
    w = vc * inv
    q = a + ab * v + ac * w
    diff = p - q
    return float(np.dot(diff, diff))


def mesh_face_normals_at_path_points(
    path_points: np.ndarray,
    raycast_scene,
    triangle_normals: np.ndarray,
    triangle_vertices: np.ndarray,
    flip_all: bool = False,
    continuity_tolerance_mm: float = FACE_CONTINUITY_TOLERANCE_MM,
):
    """
    Use nearest CAD triangle normals with a small face-continuity hysteresis.

    For each new path point, Open3D first reports the geometrically nearest face.
    If that face differs from the previously selected face, the previous face is
    retained when it is still within `continuity_tolerance_mm` of the best face.

    This suppresses numerical A/B/A/B switching when the path lies on or very
    near a shared triangle edge, while still allowing a real face transition
    once the previous triangle is clearly farther away.
    """
    P = np.asarray(path_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) == 0:
        raise ValueError("path_points must be a non-empty N x 3 array")

    query = o3d.core.Tensor(P.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ans = raycast_scene.compute_closest_points(query)

    nearest_ids = np.asarray(
        ans["primitive_ids"].numpy(), dtype=np.int64
    ).reshape(-1)
    closest_points = np.asarray(
        ans["points"].numpy(), dtype=np.float64
    ).reshape(-1, 3)

    TN = np.asarray(triangle_normals, dtype=np.float64)
    TV = np.asarray(triangle_vertices, dtype=np.float64)

    if np.any(nearest_ids < 0) or np.any(nearest_ids >= len(TN)):
        raise RuntimeError("Nearest-triangle query returned an invalid face id.")
    if TV.shape != (len(TN), 3, 3):
        raise ValueError("triangle_vertices must have shape (num_faces, 3, 3)")

    best_distance = np.linalg.norm(P - closest_points, axis=1)
    tolerance_m = max(float(continuity_tolerance_mm), 0.0) / 1000.0

    face_ids = nearest_ids.copy()
    switch_count = 0
    held_count = 0

    for i in range(1, len(P)):
        candidate = int(nearest_ids[i])
        previous = int(face_ids[i - 1])

        if candidate == previous:
            face_ids[i] = previous
            continue

        a, b, c = TV[previous]
        previous_distance = np.sqrt(
            max(_point_triangle_distance_sq(P[i], a, b, c), 0.0)
        )

        # Hysteresis: when both faces are effectively tied, keep the previous one.
        if previous_distance <= float(best_distance[i]) + tolerance_m:
            face_ids[i] = previous
            held_count += 1
        else:
            face_ids[i] = candidate
            switch_count += 1

    normals = TN[face_ids].copy()
    if flip_all:
        normals *= -1.0

    normals = orient_normals_consistently(normals, flip_all=False)
    return normals, face_ids, switch_count, held_count




# -----------------------------------------------------------------------------
# Normal conditioning and frame generation
# -----------------------------------------------------------------------------


def orient_normals_consistently(normals: np.ndarray, flip_all: bool = False):
    """Preserve mesh-normal orientation and remove only local +/- sign flips."""
    N = np.asarray(normals, dtype=np.float64).copy()
    if flip_all:
        N *= -1.0

    N[0] = normalize(N[0])
    for i in range(1, len(N)):
        N[i] = normalize(N[i])
        if float(np.dot(N[i - 1], N[i])) < 0.0:
            N[i] *= -1.0
    return N


def sample_cad_normal_field(mesh, sample_points: int):
    """
    Uniformly sample the CAD surface together with triangle normals.

    Triangle normals are intentional here: around a sharp edge we want the local
    neighborhood to contain the distinct face-normal modes instead of using an
    already vertex-averaged normal field.
    """
    n = max(1000, int(sample_points))
    cloud = mesh.sample_points_uniformly(
        number_of_points=n,
        use_triangle_normal=True,
    )
    if cloud.is_empty():
        raise RuntimeError("Failed to sample CAD surface for local-normal averaging.")

    P = np.asarray(cloud.points, dtype=np.float64).copy()
    N = np.asarray(cloud.normals, dtype=np.float64).copy()
    if len(P) == 0 or len(N) != len(P):
        raise RuntimeError("CAD normal-field sampling did not return valid normals.")

    mag = np.linalg.norm(N, axis=1)
    valid = np.isfinite(mag) & (mag > EPS)
    if not np.any(valid):
        raise RuntimeError("CAD normal-field sampling returned only invalid normals.")

    P = P[valid]
    N = N[valid] / mag[valid, None]

    field = o3d.geometry.PointCloud()
    field.points = o3d.utility.Vector3dVector(P)
    field.normals = o3d.utility.Vector3dVector(N)
    tree = o3d.geometry.KDTreeFlann(field)
    return P, N, tree


def spatial_average_cad_normals(
    path_points: np.ndarray,
    reference_normals: np.ndarray,
    cad_points: np.ndarray,
    cad_normals: np.ndarray,
    cad_tree,
    radius_mm: float,
    flip_all: bool = False,
):
    """
    Average CAD normals in a Euclidean ball around each path point.

    This is deliberately different from a moving average along the path.  A path
    point close to an edge can therefore use normals from both adjacent faces even
    when the path itself remains entirely on one face.

    A Gaussian distance weight favors the closest surface while still allowing
    nearby faces to influence the result.  If opposing surfaces cancel almost
    completely (for example on a very thin wall), the interpolated path normal is
    used as a safe fallback.
    """
    P = np.asarray(path_points, dtype=np.float64)
    R = np.asarray(reference_normals, dtype=np.float64)
    C = np.asarray(cad_points, dtype=np.float64)
    CN = np.asarray(cad_normals, dtype=np.float64)

    if len(P) != len(R):
        raise ValueError("path_points/reference_normals length mismatch")

    radius_m = float(radius_mm) / 1000.0
    if radius_m <= 0.0:
        raise ValueError("radius_mm must be positive")

    normals_field = CN.copy()
    refs = R.copy()
    if flip_all:
        normals_field *= -1.0
        refs *= -1.0

    refs = orient_normals_consistently(refs, flip_all=False)
    out = np.empty_like(refs)

    # sigma = radius / 2 -> samples at the radius boundary still contribute,
    # but much less than samples close to the path point.
    sigma = max(0.5 * radius_m, 1.0e-9)
    sigma2 = sigma * sigma

    neighbor_counts = np.zeros(len(P), dtype=np.int64)

    for i, p in enumerate(P):
        k, idx, d2 = cad_tree.search_radius_vector_3d(p, radius_m)
        neighbor_counts[i] = int(k)
        ref = normalize(refs[i])

        if k <= 0:
            out[i] = ref
            continue

        ids = np.asarray(idx, dtype=np.int64)
        d2 = np.asarray(d2, dtype=np.float64)
        local_N = normals_field[ids].copy()

        # Align every sampled face normal to the current path-surface normal
        # before averaging. STL triangle soups can contain inconsistent winding;
        # without this step two geometrically compatible normals may cancel.
        sign = local_N @ ref
        local_N[sign < 0.0] *= -1.0

        w = np.exp(-0.5 * d2 / sigma2)
        s = np.sum(w[:, None] * local_N, axis=0)
        support = float(np.sum(w))

        # If nearby opposite-facing surfaces nearly cancel, do not manufacture an
        # unstable bisector. Fall back to the path's own surface normal instead.
        if support <= EPS or np.linalg.norm(s) < 0.10 * support:
            candidate = ref
        else:
            candidate = normalize(s)

        # Preserve the local outward/inward convention of the path normal.
        if float(np.dot(candidate, ref)) < 0.0:
            candidate *= -1.0
        out[i] = candidate

    out = orient_normals_consistently(out, flip_all=False)
    return out, neighbor_counts


def moving_average_path_normals(normals: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth already-local-averaged normals along path order.

    This is the *second* normal averaging stage:
      1) spatial_average_cad_normals(): geometry-aware local CAD neighborhood
      2) moving_average_path_normals(): trajectory continuity along the path

    A centered triangular window is used instead of a flat box filter so the
    current pose remains dominant. Every neighboring normal is sign-aligned to
    the center normal before accumulation, then the result is normalized.
    Surface positions are never altered.
    """
    N = np.asarray(normals, dtype=np.float64)
    if len(N) == 0:
        return N.copy()

    window = int(window)
    if window <= 1:
        return orient_normals_consistently(N, flip_all=False)
    if window % 2 == 0:
        raise ValueError("orientation smooth window must be odd")

    src = orient_normals_consistently(N, flip_all=False)
    out = np.empty_like(src)
    half = window // 2

    for i in range(len(src)):
        lo = max(0, i - half)
        hi = min(len(src), i + half + 1)
        chunk = src[lo:hi].copy()
        ref = normalize(src[i])

        # Defensive sign alignment before averaging.
        chunk[chunk @ ref < 0.0] *= -1.0

        # Triangular weights: for window=5 -> [1,2,3,2,1], cropped at ends.
        ids = np.arange(lo, hi)
        w = (half + 1 - np.abs(ids - i)).astype(np.float64)
        s = np.sum(w[:, None] * chunk, axis=0)

        if np.linalg.norm(s) <= EPS:
            out[i] = ref
        else:
            out[i] = normalize(s)

    return orient_normals_consistently(out, flip_all=False)


def max_adjacent_normal_angle_deg(normals: np.ndarray) -> float:
    """Largest angle between consecutive normal vectors, for diagnostics."""
    N = np.asarray(normals, dtype=np.float64)
    if len(N) < 2:
        return 0.0
    dots = np.sum(N[:-1] * N[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(dots))))


def arbitrary_tangent_from_normal(normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    axes = np.eye(3)
    ref = axes[int(np.argmin(np.abs(axes @ n)))]
    t = ref - np.dot(ref, n) * n
    return normalize(t)


def estimate_path_tangents(points: np.ndarray, normals: np.ndarray, span: int = 3) -> np.ndarray:
    """Estimate scan-motion tangent and project it onto the local tangent plane."""
    P = np.asarray(points, dtype=np.float64)
    N = np.asarray(normals, dtype=np.float64)
    n_pts = len(P)
    span = max(1, int(span))

    T = np.empty_like(P)
    for i in range(n_pts):
        lo = max(0, i - span)
        hi = min(n_pts - 1, i + span)

        if hi == lo:
            raw = np.zeros(3)
        else:
            raw = P[hi] - P[lo]

        # Ensure forward path sign near endpoints / degenerate neighborhoods.
        if np.linalg.norm(raw) <= EPS:
            if i < n_pts - 1:
                raw = P[i + 1] - P[i]
            elif i > 0:
                raw = P[i] - P[i - 1]

        n = normalize(N[i])
        t = raw - np.dot(raw, n) * n

        if np.linalg.norm(t) <= 1.0e-9:
            # Use an immediate forward/backward segment if wide-span projection
            # happens to become degenerate.
            if i < n_pts - 1:
                raw2 = P[i + 1] - P[i]
            else:
                raw2 = P[i] - P[i - 1]
            t = raw2 - np.dot(raw2, n) * n

        if np.linalg.norm(t) <= 1.0e-9:
            t = arbitrary_tangent_from_normal(n)

        t = normalize(t)

        # Keep +Y aligned with traversal direction.
        if i < n_pts - 1:
            forward = P[i + 1] - P[i]
        else:
            forward = P[i] - P[i - 1]
        if np.dot(t, forward) < 0.0:
            t *= -1.0

        T[i] = t

    return T


def make_sensor_frames(normals: np.ndarray, tangents: np.ndarray):
    """
    Frame convention:
      +Y : scan-motion direction
      +Z : CAD surface-normal direction
      +X : laser profile direction
    Viewing direction is approximately -Z.
    """
    N = np.asarray(normals, dtype=np.float64)
    T = np.asarray(tangents, dtype=np.float64)
    frames = np.empty((len(N), 3, 3), dtype=np.float64)

    for i, (n_raw, t_raw) in enumerate(zip(N, T)):
        z = normalize(n_raw)
        y = t_raw - np.dot(t_raw, z) * z
        if np.linalg.norm(y) <= 1.0e-9:
            y = arbitrary_tangent_from_normal(z)
        y = normalize(y)

        x = normalize(np.cross(y, z))
        y = normalize(np.cross(z, x))

        R = np.column_stack((x, y, z))
        if np.linalg.det(R) < 0.0:
            x *= -1.0
            R = np.column_stack((x, y, z))

        frames[i] = R

    return frames


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def make_sphere(center, radius_mm, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_mm) / 1000.0)
    s.translate(np.asarray(center, dtype=np.float64))
    s.paint_uniform_color(np.asarray(color, dtype=np.float64))
    s.compute_vertex_normals()
    return s


def make_polyline(points: np.ndarray, color):
    P = np.asarray(points, dtype=np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P)
    if len(P) >= 2:
        lines = np.column_stack(
            [np.arange(len(P) - 1, dtype=np.int32), np.arange(1, len(P), dtype=np.int32)]
        )
    else:
        lines = np.empty((0, 2), dtype=np.int32)
    ls.lines = o3d.utility.Vector2iVector(lines)
    if len(lines) > 0:
        ls.colors = o3d.utility.Vector3dVector(
            np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(lines), axis=0)
        )
    return ls



def make_segmented_polyline(points: np.ndarray, segment_ids: np.ndarray, color):
    """Polyline that NEVER draws a bridge between different scan segments."""
    P = np.asarray(points, dtype=np.float64)
    S = np.asarray(segment_ids, dtype=np.int64)
    if len(P) != len(S):
        raise ValueError("points/segment_ids length mismatch")

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P)
    if len(P) >= 2:
        valid = S[:-1] == S[1:]
        a = np.arange(len(P) - 1, dtype=np.int32)[valid]
        b = a + 1
        lines = np.column_stack([a, b]).astype(np.int32)
    else:
        lines = np.empty((0, 2), dtype=np.int32)

    ls.lines = o3d.utility.Vector2iVector(lines)
    if len(lines) > 0:
        ls.colors = o3d.utility.Vector3dVector(
            np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(lines), axis=0)
        )
    return ls


def make_frame_lines(origin: np.ndarray, R: np.ndarray, axis_length_mm: float = 10.0):
    """
    Sensor frame visualization.

    X = red   : laser profile direction
    Y = green : scan-motion direction
    Z = blue  : outward CAD surface normal
    """
    o = np.asarray(origin, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = float(axis_length_mm) / 1000.0

    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    pts = np.vstack([o, o + L * x, o, o + L * y, o, o + L * z])
    lines = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 1.0],
        ],
        dtype=np.float64,
    )

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_segment(p0: np.ndarray, p1: np.ndarray, color):
    """Create one colored 3-D line segment."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(
        np.asarray([p0, p1], dtype=np.float64)
    )
    ls.lines = o3d.utility.Vector2iVector(
        np.asarray([[0, 1]], dtype=np.int32)
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.asarray([color], dtype=np.float64)
    )
    return ls


def frame_display_indices(n_points: int, frame_stride: int) -> list[int]:
    """Indices at which frames/view rays are shown in the preview."""
    if n_points <= 0:
        return []

    stride = max(1, int(frame_stride))
    ids = list(range(0, n_points, stride))
    if ids[-1] != n_points - 1:
        ids.append(n_points - 1)
    return ids


def build_visualization_geometries(
    mesh,
    waypoint_points: np.ndarray,
    surface_points: np.ndarray,
    sensor_points: np.ndarray,
    frames: np.ndarray,
    frame_stride: int,
    axis_length_mm: float,
):
    """
    Build preview geometries shared by the modern O3DVisualizer and legacy fallback.

    Color convention
    ----------------
    Gray    : CAD
    Red     : surface scan path
    Blue    : sensor-origin trajectory after standoff
    Green   : first / last manual waypoint
    Yellow  : intermediate manual waypoints
    Cyan    : sensor viewing segment along -Z, from sensor origin to CAD surface
    RGB     : sensor X/Y/Z axes
    """
    cad = o3d.geometry.TriangleMesh(mesh)
    cad.paint_uniform_color([0.60, 0.60, 0.63])

    geoms: list[tuple[str, object]] = [("CAD", cad)]
    geoms.append(("Surface scan path", make_polyline(surface_points, [1.0, 0.05, 0.05])))
    geoms.append(("Sensor trajectory", make_polyline(sensor_points, [0.05, 0.35, 1.0])))

    for i, p in enumerate(np.asarray(waypoint_points, dtype=np.float64)):
        is_endpoint = i in (0, len(waypoint_points) - 1)
        color = [0.10, 1.0, 0.15] if is_endpoint else [1.0, 0.75, 0.0]
        geoms.append((f"Waypoint W{i}", make_sphere(p, 1.8, color)))

    ids = frame_display_indices(len(sensor_points), frame_stride)
    for i in ids:
        geoms.append(
            (
                f"Sensor frame {i}",
                make_frame_lines(sensor_points[i], frames[i], axis_length_mm),
            )
        )

        # Sensor +Z is the outward surface normal.  Therefore the actual viewing
        # direction is -Z.  With the current standoff construction, the
        # corresponding surface sample is exactly the natural ray endpoint.
        geoms.append(
            (
                f"View ray {i}",
                make_segment(
                    sensor_points[i],
                    surface_points[i],
                    [0.0, 0.85, 0.90],
                ),
            )
        )

    return geoms, ids


def visualize_scan_path(
    mesh,
    waypoint_points: np.ndarray,
    surface_points: np.ndarray,
    sensor_points: np.ndarray,
    frames: np.ndarray,
    frame_stride: int,
    axis_length_mm: float,
):
    """
    Visualize the generated scan path.

    The preferred viewer is O3DVisualizer because it supports 3-D waypoint
    labels (W0, W1, ...).  If that GUI is unavailable, the function falls back
    to the legacy draw_geometries viewer; all geometry remains visible, but the
    3-D text labels are omitted.
    """
    geoms, ids = build_visualization_geometries(
        mesh=mesh,
        waypoint_points=waypoint_points,
        surface_points=surface_points,
        sensor_points=sensor_points,
        frames=frames,
        frame_stride=frame_stride,
        axis_length_mm=axis_length_mm,
    )

    print("\n[VIEW]")
    print("  Gray CAD                 : original CAD mesh")
    print("  Red path                 : generated surface scan path")
    print("  Blue path                : sensor-origin trajectory after standoff")
    print("  Green spheres            : W0 and final waypoint")
    print("  Yellow spheres           : intermediate manual waypoints")
    print("  W0, W1, ... labels       : manual waypoint order")
    print("  Frame X/Y/Z              : red / green / blue")
    print("  Cyan segment             : viewing direction along -Z toward CAD")
    print(f"  Frames/view rays shown   : every {max(1, int(frame_stride))} path samples")
    print("  +X = laser profile / +Y = scan direction / +Z = CAD normal / view = -Z")

    # Modern Open3D GUI: waypoint text labels make the preview much easier to
    # interpret.  Keep a legacy fallback because some Open3D builds omit the GUI.
    try:
        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer(
            "CAD surface path + sensor SE(3) trajectory",
            1440,
            900,
        )
        vis.show_settings = True

        for name, geom in geoms:
            vis.add_geometry(name, geom)

        # Put labels slightly away from the CAD so they do not z-fight with the
        # mesh.  A fixed small offset is sufficient because it is display-only.
        label_offset = max(float(axis_length_mm) * 0.35, 2.0) / 1000.0
        surf = np.asarray(surface_points, dtype=np.float64)
        wp = np.asarray(waypoint_points, dtype=np.float64)
        for i, p in enumerate(wp):
            nearest = int(np.argmin(np.sum((surf - p[None, :]) ** 2, axis=1)))
            outward = frames[nearest, :, 2]
            label_pos = p + label_offset * outward
            vis.add_3d_label(label_pos, f"W{i}")

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
        return

    except Exception as exc:
        print(
            "  [WARN] O3DVisualizer labels unavailable; "
            f"using legacy viewer instead: {exc}"
        )

    o3d.visualization.draw_geometries(
        [geom for _, geom in geoms],
        window_name="CAD surface path + sensor SE(3) trajectory",
        width=1440,
        height=900,
        mesh_show_back_face=True,
    )


# -----------------------------------------------------------------------------
# Interactive sensor-motion GUI
# -----------------------------------------------------------------------------


def _gui_material(shader: str, base_color, line_width: float = 2.0):
    """Create a small Open3D rendering material for the custom GUI."""
    rendering = o3d.visualization.rendering
    mat = rendering.MaterialRecord()
    mat.shader = shader
    mat.base_color = [float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0]
    if shader == "unlitLine":
        mat.line_width = float(line_width)
    return mat


def _axis_geometry(origin: np.ndarray, R: np.ndarray, axis_length_mm: float):
    """Return separate X/Y/Z line geometries so the GUI can color them clearly."""
    o = np.asarray(origin, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = float(axis_length_mm) / 1000.0
    return {
        "Current +X (laser)": make_segment(o, o + L * R[:, 0], [1.0, 0.0, 0.0]),
        "Current +Y (motion)": make_segment(o, o + L * R[:, 1], [0.0, 1.0, 0.0]),
        "Current +Z (normal)": make_segment(o, o + L * R[:, 2], [0.0, 0.35, 1.0]),
    }


def visualize_sensor_motion_gui(
    mesh,
    waypoint_points: np.ndarray,
    surface_points: np.ndarray,
    sensor_points: np.ndarray,
    frames: np.ndarray,
    arc_length_m: np.ndarray,
    axis_length_mm: float,
    initial_pose_rate_hz: float = 12.0,
    segment_ids: np.ndarray | None = None,
    waypoint_segment_ids: np.ndarray | None = None,
    waypoint_local_ids: np.ndarray | None = None,
):
    """
    Interactive GUI for inspecting how the sensor frame moves along the path.

    Static geometry
    ---------------
    gray   : CAD mesh
    red    : surface scan path
    blue   : sensor-origin trajectory
    green/yellow spheres : manual waypoints

    Dynamic geometry at the selected pose
    -------------------------------------
    red axis   : sensor +X = laser profile direction
    green axis : sensor +Y = scan-motion direction
    blue axis  : sensor +Z = outward CAD normal
    cyan ray   : actual viewing direction, sensor -> surface = -Z
    magenta line at the surface : laser-profile direction projected through the
                                  current surface sample (display only)
    white sphere : current sensor origin

    The pose can be scrubbed with a slider or played as an animation.
    """
    import time

    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering

    surface_points = np.asarray(surface_points, dtype=np.float64)
    sensor_points = np.asarray(sensor_points, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    waypoint_points = np.asarray(waypoint_points, dtype=np.float64)
    arc_length_m = np.asarray(arc_length_m, dtype=np.float64)

    n = len(surface_points)
    if segment_ids is None:
        segment_ids = np.zeros(n, dtype=np.int64)
    else:
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if len(segment_ids) != n:
        raise RuntimeError("segment_ids length mismatch")

    if waypoint_segment_ids is None:
        waypoint_segment_ids = np.zeros(len(waypoint_points), dtype=np.int64)
    else:
        waypoint_segment_ids = np.asarray(waypoint_segment_ids, dtype=np.int64)
    if waypoint_local_ids is None:
        waypoint_local_ids = np.arange(len(waypoint_points), dtype=np.int64)
    else:
        waypoint_local_ids = np.asarray(waypoint_local_ids, dtype=np.int64)

    if len(waypoint_segment_ids) != len(waypoint_points) or len(waypoint_local_ids) != len(waypoint_points):
        raise RuntimeError("waypoint segment metadata length mismatch")
    if n == 0:
        raise RuntimeError("Cannot open sensor-motion GUI: path is empty.")
    if len(sensor_points) != n or len(frames) != n:
        raise RuntimeError("surface_points/sensor_points/frames length mismatch")

    app = gui.Application.instance
    app.initialize()
    window = app.create_window("Sensor frame / viewing-direction playback", 1500, 920)

    em = float(window.theme.font_size)
    panel_width = int(max(330, 22 * em))

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.96, 0.97, 0.98, 1.0])

    panel = gui.Vert(0.45 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))

    title = gui.Label("Sensor pose playback")
    panel.add_child(title)

    pose_label = gui.Label("")
    arc_label = gui.Label("")
    sensor_pos_label = gui.Label("")
    view_label = gui.Label("")
    axis_label = gui.Label("")
    panel.add_child(pose_label)
    panel.add_child(arc_label)
    panel.add_child(sensor_pos_label)
    panel.add_child(view_label)
    panel.add_child(axis_label)

    panel.add_child(gui.Label("Pose index"))
    slider = gui.Slider(gui.Slider.INT)
    slider.set_limits(0, max(0, n - 1))
    slider.int_value = 0
    panel.add_child(slider)

    controls = gui.Horiz(0.35 * em)
    prev_btn = gui.Button("◀ Prev")
    play_btn = gui.Button("▶ Play")
    next_btn = gui.Button("Next ▶")
    controls.add_child(prev_btn)
    controls.add_child(play_btn)
    controls.add_child(next_btn)
    panel.add_child(controls)

    panel.add_child(gui.Label("Playback speed"))
    speed_box = gui.Combobox()
    speed_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    for value in speed_values:
        speed_box.add_item(f"{value:g}×")
    speed_box.selected_index = 2
    panel.add_child(speed_box)

    loop_check = gui.Checkbox("Loop playback")
    loop_check.checked = True
    panel.add_child(loop_check)

    show_view_check = gui.Checkbox("Show -Z viewing ray")
    show_view_check.checked = True
    panel.add_child(show_view_check)

    show_profile_check = gui.Checkbox("Show laser-profile direction")
    show_profile_check.checked = True
    panel.add_child(show_profile_check)

    legend = gui.Label(
        "Axes\n"
        "  X red   = laser profile\n"
        "  Y green = scan motion\n"
        "  Z blue  = CAD outward normal\n"
        "  cyan    = viewing direction (-Z)\n"
        "\nDrag: rotate | wheel: zoom | Shift+drag: pan"
    )
    panel.add_child(legend)

    window.add_child(scene_widget)
    window.add_child(panel)

    # ---------------------------- static scene ----------------------------
    cad = o3d.geometry.TriangleMesh(mesh)
    cad.paint_uniform_color([0.66, 0.66, 0.68])
    cad_mat = _gui_material("defaultLit", [0.66, 0.66, 0.68])
    path_mat = _gui_material("unlitLine", [1.0, 0.05, 0.05], line_width=3.0)
    sensor_path_mat = _gui_material("unlitLine", [0.05, 0.35, 1.0], line_width=2.5)
    waypoint_end_mat = _gui_material("defaultLit", [0.10, 1.0, 0.15])
    waypoint_mid_mat = _gui_material("defaultLit", [1.0, 0.75, 0.0])

    scene_widget.scene.add_geometry("CAD", cad, cad_mat)
    scene_widget.scene.add_geometry(
        "Surface scan paths",
        make_segmented_polyline(surface_points, segment_ids, [1.0, 0.05, 0.05]),
        path_mat,
    )
    scene_widget.scene.add_geometry(
        "Sensor trajectories",
        make_segmented_polyline(sensor_points, segment_ids, [0.05, 0.35, 1.0]),
        sensor_path_mat,
    )

    # Endpoints are evaluated inside each independent segment.
    segment_wp_counts = {
        int(s): int(np.sum(waypoint_segment_ids == s))
        for s in np.unique(waypoint_segment_ids)
    }
    for j, wp in enumerate(waypoint_points):
        sid = int(waypoint_segment_ids[j])
        lid = int(waypoint_local_ids[j])
        endpoint = lid in (0, segment_wp_counts[sid] - 1)
        sphere = make_sphere(
            wp,
            1.8,
            [0.10, 1.0, 0.15] if endpoint else [1.0, 0.75, 0.0],
        )
        label = f"S{sid}-W{lid}"
        scene_widget.scene.add_geometry(
            f"Waypoint {label}",
            sphere,
            waypoint_end_mat if endpoint else waypoint_mid_mat,
        )
        try:
            scene_widget.add_3d_label(wp, label)
        except Exception:
            pass

    # Camera bounds include both CAD and standoff trajectory.
    all_pts = np.vstack([np.asarray(mesh.vertices), surface_points, sensor_points])
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(all_pts)
    )
    center = bbox.get_center()
    extent = bbox.get_extent()
    if float(np.linalg.norm(extent)) <= EPS:
        extent = np.ones(3) * 0.1
        bbox = o3d.geometry.AxisAlignedBoundingBox(center - 0.5 * extent, center + 0.5 * extent)
    scene_widget.setup_camera(55.0, bbox, center)

    # --------------------------- dynamic scene ----------------------------
    dynamic_names = [
        "Current +X (laser)",
        "Current +Y (motion)",
        "Current +Z (normal)",
        "Current view -Z",
        "Current laser profile",
        "Current sensor origin",
        "Current surface point",
    ]

    axis_mats = {
        "Current +X (laser)": _gui_material("unlitLine", [1.0, 0.0, 0.0], 4.0),
        "Current +Y (motion)": _gui_material("unlitLine", [0.0, 1.0, 0.0], 4.0),
        "Current +Z (normal)": _gui_material("unlitLine", [0.0, 0.35, 1.0], 4.0),
    }
    view_mat = _gui_material("unlitLine", [0.0, 0.85, 0.90], 4.0)
    profile_mat = _gui_material("unlitLine", [0.95, 0.1, 0.85], 3.0)
    current_sensor_mat = _gui_material("defaultLit", [0.95, 0.95, 0.95])
    current_surface_mat = _gui_material("defaultLit", [1.0, 0.15, 0.15])

    state = {
        "idx": 0,
        "playing": False,
        "speed": 1.0,
        "last_tick": time.perf_counter(),
        "accum": 0.0,
        "pose_rate_hz": max(float(initial_pose_rate_hz), 0.1),
    }

    def remove_dynamic():
        for name in dynamic_names:
            try:
                if scene_widget.scene.has_geometry(name):
                    scene_widget.scene.remove_geometry(name)
            except Exception:
                try:
                    scene_widget.scene.remove_geometry(name)
                except Exception:
                    pass

    def update_pose(index: int, update_slider: bool = True):
        i = int(np.clip(index, 0, n - 1))
        state["idx"] = i
        remove_dynamic()

        p_sens = sensor_points[i]
        p_surf = surface_points[i]
        R = frames[i]

        for name, geom in _axis_geometry(p_sens, R, axis_length_mm).items():
            scene_widget.scene.add_geometry(name, geom, axis_mats[name])

        if show_view_check.checked:
            scene_widget.scene.add_geometry(
                "Current view -Z",
                make_segment(p_sens, p_surf, [0.0, 0.85, 0.90]),
                view_mat,
            )

        if show_profile_check.checked:
            # Display-only laser-profile direction: a line through the current
            # surface point along sensor +X. It helps visualize how the profile
            # plane sweeps while +Y advances along the scan path.
            half = 0.75 * float(axis_length_mm) / 1000.0
            x = R[:, 0]
            scene_widget.scene.add_geometry(
                "Current laser profile",
                make_segment(p_surf - half * x, p_surf + half * x, [0.95, 0.1, 0.85]),
                profile_mat,
            )

        scene_widget.scene.add_geometry(
            "Current sensor origin",
            make_sphere(p_sens, max(0.9, 0.10 * float(axis_length_mm)), [0.95, 0.95, 0.95]),
            current_sensor_mat,
        )
        scene_widget.scene.add_geometry(
            "Current surface point",
            make_sphere(p_surf, max(0.7, 0.075 * float(axis_length_mm)), [1.0, 0.15, 0.15]),
            current_surface_mat,
        )

        if update_slider:
            slider.int_value = i

        arc_mm = float(arc_length_m[i] * 1000.0) if i < len(arc_length_m) else 0.0
        pos_mm = 1000.0 * p_sens
        view = -R[:, 2]
        x, y, z = R[:, 0], R[:, 1], R[:, 2]
        sid = int(segment_ids[i])
        seg_pose_ids = np.flatnonzero(segment_ids == sid)
        local_pose = int(np.searchsorted(seg_pose_ids, i)) + 1
        pose_label.text = (
            f"Segment S{sid} | Pose {local_pose} / {len(seg_pose_ids)} "
            f"(global {i + 1}/{n})"
        )
        arc_label.text = f"Segment arc length: {arc_mm:.2f} mm"
        sensor_pos_label.text = (
            f"Sensor xyz [mm]: {pos_mm[0]:.2f}, {pos_mm[1]:.2f}, {pos_mm[2]:.2f}"
        )
        view_label.text = (
            f"View -Z: [{view[0]:+.3f}, {view[1]:+.3f}, {view[2]:+.3f}]"
        )
        axis_label.text = (
            f"X [{x[0]:+.2f},{x[1]:+.2f},{x[2]:+.2f}]\n"
            f"Y [{y[0]:+.2f},{y[1]:+.2f},{y[2]:+.2f}]\n"
            f"Z [{z[0]:+.2f},{z[1]:+.2f},{z[2]:+.2f}]"
        )
        scene_widget.force_redraw()

    def set_playing(playing: bool):
        state["playing"] = bool(playing)
        state["last_tick"] = time.perf_counter()
        state["accum"] = 0.0
        play_btn.text = "⏸ Pause" if state["playing"] else "▶ Play"

    def on_slider(value):
        set_playing(False)
        update_pose(int(round(value)), update_slider=False)

    def on_prev():
        set_playing(False)
        update_pose(state["idx"] - 1)

    def on_next():
        set_playing(False)
        update_pose(state["idx"] + 1)

    def on_play():
        set_playing(not state["playing"])

    def on_speed(text, index):
        idx = int(np.clip(index, 0, len(speed_values) - 1))
        state["speed"] = float(speed_values[idx])

    def on_view_checked(_checked):
        update_pose(state["idx"], update_slider=False)

    def on_profile_checked(_checked):
        update_pose(state["idx"], update_slider=False)

    slider.set_on_value_changed(on_slider)
    prev_btn.set_on_clicked(on_prev)
    next_btn.set_on_clicked(on_next)
    play_btn.set_on_clicked(on_play)
    speed_box.set_on_selection_changed(on_speed)
    show_view_check.set_on_checked(on_view_checked)
    show_profile_check.set_on_checked(on_profile_checked)

    def on_tick():
        now = time.perf_counter()
        dt = max(0.0, now - state["last_tick"])
        state["last_tick"] = now

        if not state["playing"]:
            return False

        state["accum"] += dt * state["pose_rate_hz"] * state["speed"]
        steps = int(state["accum"])
        if steps <= 0:
            return False
        state["accum"] -= steps

        nxt = state["idx"] + steps
        if nxt >= n:
            if loop_check.checked:
                nxt %= n
            else:
                nxt = n - 1
                set_playing(False)
        update_pose(nxt)
        return True

    window.set_on_tick_event(on_tick)

    def on_layout(layout_context):
        r = window.content_rect
        scene_widget.frame = gui.Rect(r.x, r.y, max(1, r.width - panel_width), r.height)
        panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    window.set_on_layout(on_layout)

    update_pose(0)
    app.run()





def build_mesh_path_topology(triangles: np.ndarray):
    """
    Build the topology needed to track which CAD face strip the Dijkstra
    mesh-edge path actually follows.

    Returns
    -------
    edge_to_faces
        (min(v0,v1), max(v0,v1)) -> tuple(face ids incident to that mesh edge)
    vertex_face_neighbors
        For each mesh vertex, face-to-face adjacency around that vertex.
        Two faces are neighbors here only when they share a mesh edge that
        contains the vertex.
    """
    T = np.asarray(triangles, dtype=np.int64)
    if T.ndim != 2 or T.shape[1] != 3 or len(T) == 0:
        raise ValueError("triangles must be a non-empty M x 3 array")

    edge_to_faces_mut: dict[tuple[int, int], list[int]] = {}
    n_vertices = int(np.max(T)) + 1

    for fid, (a, b, c) in enumerate(T):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            edge_to_faces_mut.setdefault(key, []).append(int(fid))

    edge_to_faces = {
        key: tuple(faces)
        for key, faces in edge_to_faces_mut.items()
    }

    vertex_face_neighbors: list[dict[int, set[int]]] = [
        {} for _ in range(n_vertices)
    ]

    for (u, v), faces in edge_to_faces.items():
        # Boundary edge: one face. Manifold edge: normally two.
        # Non-manifold edges (>2) are retained rather than silently discarded.
        for vertex in (u, v):
            nbr_map = vertex_face_neighbors[vertex]
            for f in faces:
                nbr_map.setdefault(int(f), set())
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    f = int(faces[i])
                    g = int(faces[j])
                    nbr_map[f].add(g)
                    nbr_map[g].add(f)

    return edge_to_faces, vertex_face_neighbors


def _face_normal_angle_deg(
    face_a: int,
    face_b: int,
    triangle_normals: np.ndarray,
) -> float:
    if int(face_a) == int(face_b):
        return 0.0
    N = np.asarray(triangle_normals, dtype=np.float64)
    d = float(np.dot(N[int(face_a)], N[int(face_b)]))
    return float(np.degrees(np.arccos(np.clip(d, -1.0, 1.0))))


def _faces_connected_around_vertex(
    vertex_id: int,
    face_a: int,
    face_b: int,
    vertex_face_neighbors,
) -> bool:
    """
    True when two faces belong to the same local triangle fan around vertex_id.

    This is stricter than "both faces contain the same vertex": a broken or
    non-manifold fan can therefore trigger an automatic scan break.
    """
    face_a = int(face_a)
    face_b = int(face_b)
    if face_a == face_b:
        return True

    v = int(vertex_id)
    if v < 0 or v >= len(vertex_face_neighbors):
        return False

    graph = vertex_face_neighbors[v]
    if face_a not in graph or face_b not in graph:
        return False

    stack = [face_a]
    seen = {face_a}
    while stack:
        f = stack.pop()
        for g in graph.get(f, ()):
            g = int(g)
            if g == face_b:
                return True
            if g not in seen:
                seen.add(g)
                stack.append(g)
    return False


def _track_one_topology_face_block(
    graph_path_ids: np.ndarray,
    edge_start: int,
    edge_to_faces,
    vertex_face_neighbors,
    triangle_normals: np.ndarray,
    sharp_break_angle_deg: float,
    face_switch_penalty_deg: float = 2.0,
):
    """
    Dynamic-programming face tracking from edge_start until continuity fails.

    Each Dijkstra mesh edge normally has one or two incident CAD triangles.
    We choose the face sequence with the smallest normal-direction change,
    while allowing a face change only through the same local triangle fan.

    A block ends when:
      - no topologically connected candidate remains, or
      - every topologically connected transition exceeds sharp_break_angle_deg.

    Returns
    -------
    block : dict
        edge_start, edge_end, edge_face_ids, break_reason_after
    next_edge : int
        First graph-edge index of the next block.
    """
    path = np.asarray(graph_path_ids, dtype=np.int64)
    N = np.asarray(triangle_normals, dtype=np.float64)
    n_edges = len(path) - 1
    if edge_start < 0 or edge_start >= n_edges:
        raise ValueError("edge_start out of range")

    def candidates(edge_i: int):
        u = int(path[edge_i])
        v = int(path[edge_i + 1])
        key = (u, v) if u < v else (v, u)
        faces = edge_to_faces.get(key, ())
        if not faces:
            raise RuntimeError(
                f"Dijkstra edge ({u}, {v}) has no incident CAD triangle."
            )
        return tuple(int(f) for f in faces)

    start_candidates = candidates(edge_start)
    # dp[face] = total cost. backmaps[k][face] = previous face.
    dp = {f: 0.0 for f in start_candidates}
    backmaps: list[dict[int, int | None]] = [
        {f: None for f in start_candidates}
    ]

    edge_end = edge_start
    break_reason_after = None
    threshold = float(sharp_break_angle_deg)
    threshold_enabled = threshold > 0.0
    switch_penalty = float(face_switch_penalty_deg) ** 2

    for edge_i in range(edge_start + 1, n_edges):
        shared_vertex = int(path[edge_i])
        next_candidates = candidates(edge_i)

        new_dp: dict[int, float] = {}
        new_back: dict[int, int] = {}
        any_topology_transition = False
        any_angle_allowed_transition = False

        for g in next_candidates:
            best_cost = np.inf
            best_prev = None

            for f, base_cost in dp.items():
                if not _faces_connected_around_vertex(
                    shared_vertex,
                    f,
                    g,
                    vertex_face_neighbors,
                ):
                    continue

                any_topology_transition = True
                angle = _face_normal_angle_deg(f, g, N)

                if threshold_enabled and angle > threshold:
                    continue

                any_angle_allowed_transition = True
                cost = (
                    float(base_cost)
                    + angle * angle
                    + (switch_penalty if int(f) != int(g) else 0.0)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_prev = int(f)

            if best_prev is not None:
                new_dp[int(g)] = float(best_cost)
                new_back[int(g)] = int(best_prev)

        if not new_dp:
            if not any_topology_transition:
                break_reason_after = "topology_discontinuity"
            elif not any_angle_allowed_transition:
                break_reason_after = "sharp_normal_transition"
            else:
                break_reason_after = "face_tracking_failure"
            break

        dp = new_dp
        backmaps.append(new_back)
        edge_end = edge_i

    # Backtrack the minimum-cost face sequence for this block.
    final_face = min(dp, key=dp.get)
    seq = [int(final_face)]
    for local_i in range(len(backmaps) - 1, 0, -1):
        prev = backmaps[local_i][seq[-1]]
        seq.append(int(prev))
    seq.reverse()

    if len(seq) != edge_end - edge_start + 1:
        raise RuntimeError("Internal topology-face backtracking length mismatch.")

    return {
        "edge_start": int(edge_start),
        "edge_end": int(edge_end),
        "edge_face_ids": np.asarray(seq, dtype=np.int64),
        "break_reason_after": break_reason_after,
    }, int(edge_end + 1)


def track_topology_face_blocks(
    graph_path_ids: np.ndarray,
    edge_to_faces,
    vertex_face_neighbors,
    triangle_normals: np.ndarray,
    sharp_break_angle_deg: float = 45.0,
):
    """
    Track continuous CAD face strips along a Dijkstra mesh-edge path.

    The result is one or more graph-edge blocks. A new block means the robot
    must reposition; normal smoothing is never performed across that break.
    """
    path = np.asarray(graph_path_ids, dtype=np.int64)
    if len(path) < 2:
        raise RuntimeError("Graph path must contain at least two vertices.")

    blocks = []
    edge_start = 0
    reason_before = None
    n_edges = len(path) - 1

    while edge_start < n_edges:
        block, next_edge = _track_one_topology_face_block(
            graph_path_ids=path,
            edge_start=edge_start,
            edge_to_faces=edge_to_faces,
            vertex_face_neighbors=vertex_face_neighbors,
            triangle_normals=triangle_normals,
            sharp_break_angle_deg=sharp_break_angle_deg,
        )
        block["break_reason_before"] = reason_before
        blocks.append(block)
        reason_before = block["break_reason_after"]

        if next_edge <= edge_start:
            raise RuntimeError("Topology tracker made no forward progress.")
        edge_start = next_edge

    return blocks


def resample_polyline_with_edge_ids(
    points: np.ndarray,
    step_mm: float,
):
    """
    Uniform arc-length resampling while retaining the source graph-edge index
    for every generated pose.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("points must be N x 3 with N >= 2")

    # Dijkstra mesh edges should already have positive lengths. Keep a strict
    # check here because edge->face metadata cannot survive arbitrary point removal.
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if np.any(seg <= 1.0e-10):
        raise RuntimeError(
            "Topology path contains a zero-length mesh edge after welding."
        )

    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    step = float(step_mm) / 1000.0
    if step <= 0.0:
        raise ValueError("step_mm must be positive")

    count = max(2, int(np.floor(total / step)) + 1)
    sample_s = np.arange(count, dtype=np.float64) * step
    if sample_s[-1] < total - 1.0e-10:
        sample_s = np.concatenate([sample_s, [total]])
    else:
        sample_s[-1] = total

    out = np.empty((len(sample_s), 3), dtype=np.float64)
    source_edge = np.empty(len(sample_s), dtype=np.int64)

    j = 0
    for k, sk in enumerate(sample_s):
        while j < len(seg) - 1 and sk > s[j + 1]:
            j += 1
        ds = float(s[j + 1] - s[j])
        a = float((sk - s[j]) / ds)
        a = min(max(a, 0.0), 1.0)
        out[k] = (1.0 - a) * P[j] + a * P[j + 1]
        source_edge[k] = int(j)

    return out, sample_s, source_edge


def _waypoint_graph_positions(
    graph_path_ids: np.ndarray,
    waypoint_vertex_ids: np.ndarray,
) -> np.ndarray:
    """Find the ordered occurrence of each manual waypoint on the Dijkstra path."""
    path = np.asarray(graph_path_ids, dtype=np.int64)
    wps = np.asarray(waypoint_vertex_ids, dtype=np.int64)

    positions = []
    search_from = 0
    for wid in wps:
        matches = np.flatnonzero(path[search_from:] == int(wid))
        if len(matches) == 0:
            raise RuntimeError(
                f"Manual waypoint vertex {int(wid)} was not found in its Dijkstra path."
            )
        pos = int(search_from + matches[0])
        positions.append(pos)
        search_from = pos

    return np.asarray(positions, dtype=np.int64)


def _block_waypoints(
    graph_path_ids: np.ndarray,
    waypoint_vertex_ids: np.ndarray,
    waypoint_graph_positions: np.ndarray,
    edge_start: int,
    edge_end: int,
    vertices: np.ndarray,
):
    """
    Keep manual waypoints that belong to this auto-generated block and add the
    automatic break endpoints when necessary.
    """
    path = np.asarray(graph_path_ids, dtype=np.int64)
    wp_ids = np.asarray(waypoint_vertex_ids, dtype=np.int64)
    wp_pos = np.asarray(waypoint_graph_positions, dtype=np.int64)
    V = np.asarray(vertices, dtype=np.float64)

    v_start_pos = int(edge_start)
    v_end_pos = int(edge_end + 1)
    ids = [int(path[v_start_pos])]

    for wid, pos in zip(wp_ids, wp_pos):
        pos = int(pos)
        if v_start_pos < pos < v_end_pos:
            ids.append(int(wid))

    end_id = int(path[v_end_pos])
    if end_id != ids[-1]:
        ids.append(end_id)

    # One graph edge always has two distinct vertices, so this should be >=2.
    if len(ids) < 2:
        raise RuntimeError("Auto-split block collapsed to fewer than two endpoints.")

    ids_arr = np.asarray(ids, dtype=np.int64)
    return ids_arr, V[ids_arr]


def renumber_scan_segments(segments: list[dict]):
    """Assign final contiguous S0, S1, ... ids after automatic splitting."""
    for sid, seg in enumerate(segments):
        seg["segment_id"] = int(sid)
    return segments



def optimize_standoff_distances(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    nominal_mm: float,
    min_mm: float,
    max_mm: float,
    curvature_weight: float = 1.0,
    nominal_weight: float = 0.05,
    delta_weight: float = 0.20,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    Optimize one scalar standoff d_i per path pose.

        sensor_i = surface_i + d_i * normal_i

    Objective:
      1) make the sensor-origin trajectory smooth,
      2) prefer the nominal working distance,
      3) avoid abrupt changes in standoff.

    Box constraint:
        min_mm <= d_i <= max_mm

    The problem is convex quadratic. It is solved with NumPy-only projected
    gradient descent so no scipy/cvxpy dependency is added.
    """
    P = np.asarray(surface_points, dtype=np.float64)
    N = np.asarray(surface_normals, dtype=np.float64)

    if P.ndim != 2 or P.shape[1] != 3 or N.shape != P.shape:
        raise ValueError("surface_points/surface_normals must both be N x 3")
    if len(P) < 2:
        raise RuntimeError("Need at least two path poses for standoff optimization.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(N)):
        raise ValueError("surface_points/surface_normals contain NaN or Inf")

    d0 = float(nominal_mm) / 1000.0
    dmin = float(min_mm) / 1000.0
    dmax = float(max_mm) / 1000.0

    if dmin < 0.0 or dmax < 0.0 or dmin > dmax:
        raise ValueError("Invalid standoff bounds.")
    if not (dmin <= d0 <= dmax):
        raise ValueError("Nominal standoff must lie inside [min, max].")

    n = len(P)
    if abs(dmax - dmin) <= 1.0e-12:
        return np.full(n, dmin, dtype=np.float64)

    wc = float(curvature_weight)
    wn = float(nominal_weight)
    wd = float(delta_weight)
    if wc < 0.0 or wn <= 0.0 or wd < 0.0:
        raise ValueError(
            "standoff weights require curvature>=0, nominal>0, delta>=0"
        )

    # Sensor-path second-difference term:
    #
    # S_{i+2} - 2*S_{i+1} + S_i
    # = c_i + B_i d
    #
    # Stack xyz coordinates into B @ d + c.
    if n >= 3:
        B = np.zeros((3 * (n - 2), n), dtype=np.float64)
        c = np.zeros(3 * (n - 2), dtype=np.float64)

        row = 0
        for i in range(n - 2):
            for axis in range(3):
                B[row, i] = N[i, axis]
                B[row, i + 1] = -2.0 * N[i + 1, axis]
                B[row, i + 2] = N[i + 2, axis]
                c[row] = (
                    P[i, axis]
                    - 2.0 * P[i + 1, axis]
                    + P[i + 2, axis]
                )
                row += 1

        BtB = B.T @ B
        Btc = B.T @ c
    else:
        BtB = np.zeros((n, n), dtype=np.float64)
        Btc = np.zeros(n, dtype=np.float64)

    # Penalize d_{i+1} - d_i.
    D = np.zeros((max(0, n - 1), n), dtype=np.float64)
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    DtD = D.T @ D if len(D) else np.zeros((n, n), dtype=np.float64)

    # f(d) = d^T Q d + 2 b^T d + constant.
    Q = wc * BtB + wn * np.eye(n) + wd * DtD
    b = wc * Btc - wn * d0 * np.ones(n, dtype=np.float64)

    # Good initial point: unconstrained quadratic optimum, then clip.
    try:
        d = np.linalg.solve(Q, -b)
    except np.linalg.LinAlgError:
        d = np.full(n, d0, dtype=np.float64)
    d = np.clip(d, dmin, dmax)

    # Projected-gradient refinement for exact box constraints.
    L = 2.0 * float(np.max(np.sum(np.abs(Q), axis=1)))
    if not np.isfinite(L) or L <= EPS:
        return np.clip(d, dmin, dmax)

    step = 1.0 / L
    for _ in range(max(1, int(max_iterations))):
        grad = 2.0 * (Q @ d + b)
        nxt = np.clip(d - step * grad, dmin, dmax)
        if float(np.max(np.abs(nxt - d))) < 1.0e-10:
            d = nxt
            break
        d = nxt

    return d



def build_scan_segments(
    segment_spec: dict,
    adjacency,
    vertices: np.ndarray,
    edge_to_faces,
    vertex_face_neighbors,
    triangle_normals: np.ndarray,
    path_step_mm: float,
    orientation_smooth_window: int,
    tangent_span: int,
    sensor_standoff_mm: float,
    flip_normals: bool,
    auto_break_angle_deg: float = 45.0,
    standoff_min_mm: float | None = None,
    standoff_max_mm: float | None = None,
    standoff_nominal_mm: float | None = None,
):
    """
    Generate one manual scan stroke, automatically splitting it whenever the
    tracked CAD face strip loses topological continuity or crosses a sharp
    normal transition.

    The Dijkstra surface path itself is unchanged.
    """
    source_sid = int(segment_spec["segment_id"])
    waypoint_ids = np.asarray(segment_spec["waypoint_vertex_ids"], dtype=np.int64)

    graph_path_ids = concatenate_waypoint_paths(adjacency, waypoint_ids)
    graph_positions = _waypoint_graph_positions(graph_path_ids, waypoint_ids)

    topology_blocks = track_topology_face_blocks(
        graph_path_ids=graph_path_ids,
        edge_to_faces=edge_to_faces,
        vertex_face_neighbors=vertex_face_neighbors,
        triangle_normals=triangle_normals,
        sharp_break_angle_deg=auto_break_angle_deg,
    )

    nominal_mm = (
        float(sensor_standoff_mm)
        if standoff_nominal_mm is None
        else float(standoff_nominal_mm)
    )

    out_segments: list[dict] = []

    for split_idx, block in enumerate(topology_blocks):
        edge_start = int(block["edge_start"])
        edge_end = int(block["edge_end"])
        graph_ids_sub = graph_path_ids[edge_start:edge_end + 2]
        graph_points_sub = np.asarray(vertices, dtype=np.float64)[graph_ids_sub]

        surface_points, arc_length_m, source_edge = resample_polyline_with_edge_ids(
            graph_points_sub,
            step_mm=path_step_mm,
        )

        edge_face_ids = np.asarray(block["edge_face_ids"], dtype=np.int64)
        pose_face_ids = edge_face_ids[source_edge]
        raw_face_N = np.asarray(triangle_normals, dtype=np.float64)[pose_face_ids].copy()

        if flip_normals:
            raw_face_N *= -1.0
        raw_face_N = orient_normals_consistently(raw_face_N, flip_all=False)

        # Crucially, smoothing restarts after every automatic break.
        surface_N = moving_average_path_normals(
            raw_face_N,
            window=orientation_smooth_window,
        )

        tangents = estimate_path_tangents(
            surface_points,
            surface_N,
            span=tangent_span,
        )
        frames = make_sensor_frames(surface_N, tangents)

        if standoff_min_mm is None and standoff_max_mm is None:
            standoff_m = np.full(
                len(surface_points),
                nominal_mm / 1000.0,
                dtype=np.float64,
            )
        else:
            if standoff_min_mm is None or standoff_max_mm is None:
                raise ValueError(
                    "standoff_min_mm and standoff_max_mm must be supplied together"
                )
            standoff_m = optimize_standoff_distances(
                surface_points=surface_points,
                surface_normals=surface_N,
                nominal_mm=nominal_mm,
                min_mm=float(standoff_min_mm),
                max_mm=float(standoff_max_mm),
            )

        sensor_points = (
            surface_points
            + standoff_m[:, None] * frames[:, :, 2]
        )

        block_wp_ids, block_wp_points = _block_waypoints(
            graph_path_ids=graph_path_ids,
            waypoint_vertex_ids=waypoint_ids,
            waypoint_graph_positions=graph_positions,
            edge_start=edge_start,
            edge_end=edge_end,
            vertices=vertices,
        )

        edge_switch_count = int(
            np.count_nonzero(edge_face_ids[1:] != edge_face_ids[:-1])
        ) if len(edge_face_ids) > 1 else 0

        out_segments.append({
            # Final S-id is assigned after all manual strokes are generated.
            "segment_id": -1,
            "source_manual_segment_id": source_sid,
            "auto_split_index": int(split_idx),
            "break_reason_before": block.get("break_reason_before"),
            "break_reason_after": block.get("break_reason_after"),
            "waypoint_vertex_ids": block_wp_ids,
            "waypoint_points": block_wp_points,
            "graph_path_ids": graph_ids_sub,
            "graph_edge_start": edge_start,
            "graph_edge_end": edge_end,
            "surface_points": surface_points,
            "raw_face_normals": raw_face_N,
            "surface_normals": surface_N,
            "triangle_face_ids": pose_face_ids,
            "tracked_edge_face_ids": edge_face_ids,
            "face_switch_count": edge_switch_count,
            "tangents": tangents,
            "frames": frames,
            "sensor_points": sensor_points,
            "standoff_m": standoff_m,
            "arc_length_m": arc_length_m,
        })

    if not out_segments:
        raise RuntimeError("Topology tracking produced no scan segment.")

    return out_segments


def flatten_scan_segments(segments: list[dict]):
    """Flatten independent segments for playback while preserving break metadata."""
    if not segments:
        raise RuntimeError("No generated scan segments.")

    surface_points = np.concatenate([s["surface_points"] for s in segments], axis=0)
    surface_normals = np.concatenate([s["surface_normals"] for s in segments], axis=0)
    tangents = np.concatenate([s["tangents"] for s in segments], axis=0)
    sensor_points = np.concatenate([s["sensor_points"] for s in segments], axis=0)
    frames = np.concatenate([s["frames"] for s in segments], axis=0)
    arc_length_m = np.concatenate([s["arc_length_m"] for s in segments], axis=0)
    segment_ids = np.concatenate(
        [np.full(len(s["surface_points"]), int(s["segment_id"]), dtype=np.int64) for s in segments]
    )

    waypoint_points = np.concatenate([s["waypoint_points"] for s in segments], axis=0)
    waypoint_segment_ids = np.concatenate(
        [np.full(len(s["waypoint_points"]), int(s["segment_id"]), dtype=np.int64) for s in segments]
    )
    waypoint_local_ids = np.concatenate(
        [np.arange(len(s["waypoint_points"]), dtype=np.int64) for s in segments]
    )

    return {
        "surface_points": surface_points,
        "surface_normals": surface_normals,
        "tangents": tangents,
        "sensor_points": sensor_points,
        "frames": frames,
        "arc_length_m": arc_length_m,
        "segment_ids": segment_ids,
        "waypoint_points": waypoint_points,
        "waypoint_segment_ids": waypoint_segment_ids,
        "waypoint_local_ids": waypoint_local_ids,
    }


# -----------------------------------------------------------------------------
# Import-friendly API + diagnostics (does not change the path algorithm)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanPlannerConfig:
    """Same planning parameters as the CLI. Distance values are in millimetres."""

    mesh_unit: str = "auto"
    weld_tolerance_mm: float = 0.001
    path_step_mm: float = 1.0
    sensor_standoff_mm: float = 20.0
    standoff_min_mm: float | None = None
    standoff_max_mm: float | None = None
    standoff_nominal_mm: float | None = None
    flip_normals: bool = False
    normal_radius_mm: float = 3.0
    normal_field_samples: int = 120000
    orientation_smooth_window: int = 5
    tangent_span: int = 3
    auto_break_angle_deg: float = 45.0

    def validate(self):
        if self.mesh_unit not in {"auto", "m", "mm"}:
            raise ValueError("mesh_unit must be auto, m, or mm")
        if self.weld_tolerance_mm < 0 or self.sensor_standoff_mm < 0:
            raise ValueError("weld_tolerance_mm and sensor_standoff_mm must be >= 0")

        one_bound = (self.standoff_min_mm is None) != (self.standoff_max_mm is None)
        if one_bound:
            raise ValueError(
                "standoff_min_mm and standoff_max_mm must be supplied together"
            )
        if self.standoff_min_mm is not None:
            dmin = float(self.standoff_min_mm)
            dmax = float(self.standoff_max_mm)
            dnom = (
                float(self.sensor_standoff_mm)
                if self.standoff_nominal_mm is None
                else float(self.standoff_nominal_mm)
            )
            if dmin < 0.0 or dmax < 0.0 or dmin > dmax:
                raise ValueError("invalid standoff range")
            if not (dmin <= dnom <= dmax):
                raise ValueError(
                    "nominal standoff must lie inside the standoff range"
                )

        if self.path_step_mm <= 0 or self.normal_radius_mm <= 0:
            raise ValueError("path_step_mm and normal_radius_mm must be > 0")
        if self.normal_field_samples <= 0 or self.tangent_span <= 0:
            raise ValueError("normal_field_samples and tangent_span must be > 0")
        if self.orientation_smooth_window <= 0 or self.orientation_smooth_window % 2 == 0:
            raise ValueError("orientation_smooth_window must be a positive odd integer")
        if self.auto_break_angle_deg < 0.0 or self.auto_break_angle_deg > 180.0:
            raise ValueError("auto_break_angle_deg must be in [0, 180]")


def _max_position_step_mm(points: np.ndarray) -> float:
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1)) * 1000.0)


def _max_rotation_step_deg(frames: np.ndarray) -> float:
    R = np.asarray(frames, dtype=np.float64)
    if len(R) < 2:
        return 0.0
    rel = np.einsum("nij,njk->nik", np.transpose(R[:-1], (0, 2, 1)), R[1:])
    c = (np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5
    return float(np.degrees(np.max(np.arccos(np.clip(c, -1.0, 1.0)))))


def validate_scan_segments(segments: list[dict], print_report: bool = True) -> dict:
    """Inspect a generated path only. Nothing is smoothed, rejected, or modified."""
    reports = []
    ok = bool(segments)

    for seg in segments:
        P = np.asarray(seg["sensor_points"], dtype=np.float64)
        N = np.asarray(seg["surface_normals"], dtype=np.float64)
        R = np.asarray(seg["frames"], dtype=np.float64)

        finite = bool(np.all(np.isfinite(P)) and np.all(np.isfinite(N)) and np.all(np.isfinite(R)))
        if len(R):
            RtR = np.einsum("nij,njk->nik", np.transpose(R, (0, 2, 1)), R)
            orth_err = float(np.max(np.linalg.norm(RtR - np.eye(3), axis=(1, 2))))
            det_err = float(np.max(np.abs(np.linalg.det(R) - 1.0)))
        else:
            orth_err = det_err = float("inf")
        frame_ok = bool(orth_err <= 1.0e-6 and det_err <= 1.0e-6)
        ok = ok and finite and frame_ok

        reports.append({
            "segment_id": int(seg["segment_id"]),
            "pose_count": int(len(P)),
            "path_length_mm": float(seg["arc_length_m"][-1] * 1000.0),
            "max_position_step_mm": _max_position_step_mm(P),
            "max_normal_step_deg": max_adjacent_normal_angle_deg(N),
            "max_rotation_step_deg": _max_rotation_step_deg(R),
            "all_finite": finite,
            "rotation_frames_valid": frame_ok,
        })

    result = {
        "ok": bool(ok),
        "segment_count": len(reports),
        "pose_count": int(sum(r["pose_count"] for r in reports)),
        "segments": reports,
    }

    if print_report:
        print("\n[SCAN PLAN VALIDATION]")
        print(f"  status      : {'OK' if result['ok'] else 'INVALID'}")
        for r in reports:
            print(
                f"  S{r['segment_id']}: poses={r['pose_count']}, "
                f"length={r['path_length_mm']:.3f} mm, "
                f"pos-step={r['max_position_step_mm']:.3f} mm, "
                f"normal-step={r['max_normal_step_deg']:.2f} deg, "
                f"rotation-step={r['max_rotation_step_deg']:.2f} deg"
            )
            if not r["all_finite"]:
                print("      [ERROR] NaN/Inf detected")
            if not r["rotation_frames_valid"]:
                print("      [ERROR] invalid rotation matrix detected")

    return result


class ScanPlan:
    """Small result wrapper. Existing segment dictionaries remain untouched."""

    def __init__(self, segments: list[dict]):
        self.segments = segments
        self.flat = flatten_scan_segments(segments)

    def validate(self, print_report: bool = True) -> dict:
        return validate_scan_segments(self.segments, print_report=print_report)

    def pose_matrices(self) -> np.ndarray:
        P = self.flat["sensor_points"]
        R = self.flat["frames"]
        T = np.repeat(np.eye(4)[None, :, :], len(P), axis=0)
        T[:, :3, :3] = R
        T[:, :3, 3] = P
        return T


class SurfaceScanPlanner:
    """Reusable wrapper around the existing Dijkstra -> normal -> frame pipeline."""

    def __init__(self, cad_path: str | Path, config: ScanPlannerConfig | None = None):
        self.cad_path = Path(cad_path)
        self.config = config or ScanPlannerConfig()
        self.config.validate()

        self.mesh, self.mesh_info = load_mesh_preserve_frame(
            self.cad_path, self.config.mesh_unit, self.config.weld_tolerance_mm
        )
        self.V = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.VN = np.asarray(self.mesh.vertex_normals, dtype=np.float64)
        self.T = np.asarray(self.mesh.triangles, dtype=np.int64)
        self.adjacency = build_mesh_edge_graph(self.V, self.T)
        self.component_labels, self.component_sizes = graph_component_labels(self.adjacency)
        self.mesh.compute_triangle_normals()
        self.triangle_normals = np.asarray(
            self.mesh.triangle_normals, dtype=np.float64
        ).copy()
        mag = np.linalg.norm(self.triangle_normals, axis=1)
        if np.any(~np.isfinite(mag)) or np.any(mag <= EPS):
            raise RuntimeError("CAD contains invalid triangle normals.")
        self.triangle_normals /= mag[:, None]

        self.edge_to_faces, self.vertex_face_neighbors = build_mesh_path_topology(
            self.T
        )

    def _make_spec(self, segment_id: int, waypoints_m) -> dict:
        points = np.asarray(list(waypoints_m), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("waypoints_m must be an N x 3 array with N >= 2")
        if not np.all(np.isfinite(points)):
            raise ValueError("waypoints_m contains NaN or Inf")

        # Same behaviour as the current GUI picker: nearest welded mesh vertex.
        ids = []
        for p in points:
            d2 = np.sum((self.V - p[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            if not ids or idx != ids[-1]:
                ids.append(idx)
        if len(ids) < 2:
            raise RuntimeError("Waypoints collapsed to fewer than 2 unique mesh vertices")

        ids = np.asarray(ids, dtype=np.int64)
        comps = self.component_labels[ids]
        if np.any(comps != comps[0]):
            raise RuntimeError("Waypoints lie on disconnected mesh components")

        return {
            "segment_id": int(segment_id),
            "waypoint_vertex_ids": ids,
            "waypoint_points": self.V[ids],
        }

    def _generate(self, spec: dict) -> list[dict]:
        c = self.config
        return build_scan_segments(
            segment_spec=spec,
            adjacency=self.adjacency,
            vertices=self.V,
            edge_to_faces=self.edge_to_faces,
            vertex_face_neighbors=self.vertex_face_neighbors,
            triangle_normals=self.triangle_normals,
            path_step_mm=c.path_step_mm,
            orientation_smooth_window=c.orientation_smooth_window,
            tangent_span=c.tangent_span,
            sensor_standoff_mm=c.sensor_standoff_mm,
            flip_normals=c.flip_normals,
            auto_break_angle_deg=c.auto_break_angle_deg,
            standoff_min_mm=c.standoff_min_mm,
            standoff_max_mm=c.standoff_max_mm,
            standoff_nominal_mm=c.standoff_nominal_mm,
        )

    def plan(self, waypoints_m) -> ScanPlan:
        """Generate one manual stroke, with automatic topology/sharp-edge splits."""
        segments = self._generate(self._make_spec(0, waypoints_m))
        renumber_scan_segments(segments)
        return ScanPlan(segments)

    def plan_segments(self, waypoint_segments_m) -> ScanPlan:
        """Generate manual strokes; each stroke may be split into safe subsegments."""
        segments: list[dict] = []
        for sid, waypoints in enumerate(waypoint_segments_m):
            segments.extend(self._generate(self._make_spec(sid, waypoints)))
        if not segments:
            raise ValueError("At least one waypoint segment is required")
        renumber_scan_segments(segments)
        return ScanPlan(segments)


def visualize_segmented_scan_path(
    mesh,
    segments: list[dict],
    frame_stride: int,
    axis_length_mm: float,
):
    """Static overview with true gaps between independent scan segments."""
    cad = o3d.geometry.TriangleMesh(mesh)
    cad.paint_uniform_color([0.60, 0.60, 0.63])
    geoms: list[tuple[str, object]] = [("CAD", cad)]

    for seg in segments:
        sid = int(seg["segment_id"])
        surf = seg["surface_points"]
        sens = seg["sensor_points"]
        frames = seg["frames"]
        wps = seg["waypoint_points"]

        geoms.append((f"S{sid} surface path", make_polyline(surf, [1.0, 0.05, 0.05])))
        geoms.append((f"S{sid} sensor trajectory", make_polyline(sens, [0.05, 0.35, 1.0])))

        for j, wp in enumerate(wps):
            endpoint = j in (0, len(wps) - 1)
            color = [0.10, 1.0, 0.15] if endpoint else [1.0, 0.75, 0.0]
            geoms.append((f"S{sid}-W{j}", make_sphere(wp, 1.8, color)))

        for i in frame_display_indices(len(sens), frame_stride):
            geoms.append((f"S{sid} frame {i}", make_frame_lines(sens[i], frames[i], axis_length_mm)))
            geoms.append((f"S{sid} view {i}", make_segment(sens[i], surf[i], [0.0, 0.85, 0.90])))

    print("\n[SEGMENTED VIEW]")
    print("  Red   : independent surface scan segments (no bridge)")
    print("  Blue  : independent sensor trajectories (no bridge)")
    print("  Labels: S0-W0, S0-W1, S1-W0, ...")
    print("  Cyan  : viewing direction (-Z)")

    try:
        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(
            "Independent CAD scan segments",
            1440,
            900,
        )
        vis.show_settings = True
        for name, geom in geoms:
            vis.add_geometry(name, geom)
        for seg in segments:
            sid = int(seg["segment_id"])
            for j, wp in enumerate(seg["waypoint_points"]):
                vis.add_3d_label(wp, f"S{sid}-W{j}")
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
        return
    except Exception as exc:
        print(f"  [WARN] modern viewer unavailable, using legacy viewer: {exc}")

    o3d.visualization.draw_geometries(
        [g for _, g in geoms],
        window_name="Independent CAD scan segments",
        width=1440,
        height=900,
        mesh_show_back_face=True,
    )


def export_json_segments(path: Path, cad_path: Path, segments: list[dict]):
    """Export independent scans. A downstream robot must reposition between segments."""
    out_segments = []
    for seg in segments:
        sid = int(seg["segment_id"])
        poses = []
        prev_q = None
        for i, (p_surf, p_sens, n, t, R) in enumerate(
            zip(
                seg["surface_points"],
                seg["sensor_points"],
                seg["surface_normals"],
                seg["tangents"],
                seg["frames"],
            )
        ):
            q = rotation_matrix_to_quaternion_xyzw(R)
            if prev_q is not None and np.dot(prev_q, q) < 0.0:
                q *= -1.0
            prev_q = q.copy()
            poses.append(
                {
                    "index": i,
                    "arc_length_mm": float(seg["arc_length_m"][i] * 1000.0),
                    "surface_position_m": p_surf.tolist(),
                    "sensor_position_m": p_sens.tolist(),
                    "standoff_mm": float(np.linalg.norm(p_sens - p_surf) * 1000.0),
                    "normal_z": n.tolist(),
                    "scan_tangent_y": t.tolist(),
                    "quaternion_xyzw": q.tolist(),
                    "R_cad_sensor": R.tolist(),
                }
            )

        out_segments.append(
            {
                "segment_id": sid,
                "source_manual_segment_id": int(seg.get("source_manual_segment_id", sid)),
                "auto_split_index": int(seg.get("auto_split_index", 0)),
                "break_reason_before": seg.get("break_reason_before"),
                "break_reason_after": seg.get("break_reason_after"),
                "reposition_required_before": bool(sid > 0),
                "waypoint_vertex_ids": [int(x) for x in seg["waypoint_vertex_ids"]],
                "waypoints_m": np.asarray(seg["waypoint_points"], dtype=float).tolist(),
                "surface_path_length_mm": float(seg["arc_length_m"][-1] * 1000.0),
                "poses": poses,
            }
        )

    data = {
        "cad": str(cad_path),
        "independent_scan_segments": True,
        "segment_count": len(out_segments),
        "between_segments": "reposition; do not execute a linear scan bridge",
        "frame_convention": {
            "+X": "laser profile direction",
            "+Y": "scan motion direction",
            "+Z": "CAD surface-normal direction",
            "viewing_direction": "-Z",
        },
        "segments": out_segments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_csv_segments(path: Path, segments: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment_id",
                "local_index",
                "arc_length_mm",
                "x_m", "y_m", "z_m",
                "qx", "qy", "qz", "qw",
            ]
        )
        for seg in segments:
            sid = int(seg["segment_id"])
            prev_q = None
            for i, (p, R) in enumerate(zip(seg["sensor_points"], seg["frames"])):
                q = rotation_matrix_to_quaternion_xyzw(R)
                if prev_q is not None and np.dot(prev_q, q) < 0.0:
                    q *= -1.0
                prev_q = q.copy()
                writer.writerow(
                    [
                        sid,
                        i,
                        float(seg["arc_length_m"][i] * 1000.0),
                        float(p[0]), float(p[1]), float(p[2]),
                        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    ]
                )


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), EPS)
    return q


def export_json(
    path: Path,
    cad_path: Path,
    waypoint_vertex_ids: np.ndarray,
    waypoint_points: np.ndarray,
    arc_length_m: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    tangents: np.ndarray,
    sensor_points: np.ndarray,
    frames: np.ndarray,
):
    poses = []
    prev_q = None

    for i in range(len(sensor_points)):
        q = rotation_matrix_to_quaternion_xyzw(frames[i])
        # Quaternion q and -q represent the same rotation. Keep exported signs
        # continuous to make downstream interpolation safer.
        if prev_q is not None and np.dot(prev_q, q) < 0.0:
            q *= -1.0
        prev_q = q.copy()

        poses.append(
            {
                "index": i,
                "arc_length_mm": float(arc_length_m[i] * 1000.0),
                "surface_position_m": surface_points[i].tolist(),
                "sensor_position_m": sensor_points[i].tolist(),
                "normal_z": surface_normals[i].tolist(),
                "scan_tangent_y": tangents[i].tolist(),
                "quaternion_xyzw": q.tolist(),
                "R_cad_sensor": frames[i].tolist(),
            }
        )

    data = {
        "cad": str(cad_path),
        "frame_convention": {
            "+X": "laser profile direction",
            "+Y": "scan motion direction",
            "+Z": "CAD surface-normal direction",
            "viewing_direction": "-Z",
        },
        "waypoint_vertex_ids": [int(x) for x in waypoint_vertex_ids],
        "waypoints_m": np.asarray(waypoint_points, dtype=float).tolist(),
        "poses": poses,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_csv(
    path: Path,
    arc_length_m: np.ndarray,
    sensor_points: np.ndarray,
    frames: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "arc_length_mm",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
        )

        prev_q = None
        for i, (p, R) in enumerate(zip(sensor_points, frames)):
            q = rotation_matrix_to_quaternion_xyzw(R)
            if prev_q is not None and np.dot(prev_q, q) < 0.0:
                q *= -1.0
            prev_q = q.copy()
            writer.writerow(
                [
                    i,
                    float(arc_length_m[i] * 1000.0),
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                ]
            )



# -----------------------------------------------------------------------------
# Isolated GUI playback
# -----------------------------------------------------------------------------


def load_flattened_playback_json(path: Path):
    """Load segmented scan JSON into the flat arrays expected by the GUI."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        raise RuntimeError(f"Playback JSON contains no scan segments: {path}")

    surface_points = []
    sensor_points = []
    frames = []
    arc_length_m = []
    segment_ids = []
    waypoint_points = []
    waypoint_segment_ids = []
    waypoint_local_ids = []

    for seg in segments:
        sid = int(seg["segment_id"])
        wps = np.asarray(seg.get("waypoints_m", []), dtype=np.float64)
        for lid, wp in enumerate(wps):
            waypoint_points.append(wp)
            waypoint_segment_ids.append(sid)
            waypoint_local_ids.append(lid)

        poses = seg.get("poses", [])
        if len(poses) < 2:
            raise RuntimeError(f"Segment S{sid} has fewer than two poses in playback JSON.")
        for pose in poses:
            surface_points.append(np.asarray(pose["surface_position_m"], dtype=np.float64))
            sensor_points.append(np.asarray(pose["sensor_position_m"], dtype=np.float64))
            frames.append(np.asarray(pose["R_cad_sensor"], dtype=np.float64))
            arc_length_m.append(float(pose["arc_length_mm"]) / 1000.0)
            segment_ids.append(sid)

    return {
        "surface_points": np.asarray(surface_points, dtype=np.float64),
        "sensor_points": np.asarray(sensor_points, dtype=np.float64),
        "frames": np.asarray(frames, dtype=np.float64),
        "arc_length_m": np.asarray(arc_length_m, dtype=np.float64),
        "segment_ids": np.asarray(segment_ids, dtype=np.int64),
        "waypoint_points": np.asarray(waypoint_points, dtype=np.float64),
        "waypoint_segment_ids": np.asarray(waypoint_segment_ids, dtype=np.int64),
        "waypoint_local_ids": np.asarray(waypoint_local_ids, dtype=np.int64),
    }


def launch_gui_in_fresh_process(args, json_path: Path):
    """
    Start playback in a fresh Python/Open3D process.

    VisualizerWithEditing uses OpenGL/GLFW, while the playback UI uses
    Open3D's gui.Application/Filament renderer. On some Linux/Open3D builds,
    opening/closing multiple picker windows before starting gui.Application in
    the same process makes app.run() return immediately. A fresh process gives
    the playback GUI its own clean graphics context and is much more robust.
    """
    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        str(args.cad),
        "--mesh-unit", str(args.mesh_unit),
        "--weld-tolerance-mm", str(args.weld_tolerance_mm),
        "--sensor-axis-mm", str(args.sensor_axis_mm),
        "--gui-pose-rate-hz", str(args.gui_pose_rate_hz),
        "--playback-json", str(json_path),
    ]
    print("\n[LAUNCH GUI]")
    print("  Starting playback in a fresh Open3D process...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Playback GUI process exited with code {result.returncode}.")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pick ordered CAD waypoints and generate a surface-following scan path "
            "with automatic sensor orientation."
        )
    )

    p.add_argument("cad", type=Path)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument(
        "--weld-tolerance-mm",
        type=nonnegative_float,
        default=0.001,
        help=(
            "Merge nearly coincident STL triangle vertices before path planning. "
            "Increase slightly (e.g. 0.01 mm) only if a visually continuous STL "
            "still reports disconnected components."
        ),
    )
    p.add_argument("--picker-point-size", type=positive_float, default=2.0)
    p.add_argument(
        "--picker-sample-points",
        type=positive_int,
        default=100000,
        help="Dense CAD surface samples shown in the waypoint picker.",
    )

    p.add_argument(
        "--path-step-mm",
        type=positive_float,
        default=1.0,
        help="Arc-length spacing of generated path poses.",
    )
    p.add_argument(
        "--sensor-standoff-mm",
        type=nonnegative_float,
        default=20.0,
        help=(
            "Fixed standoff in mm. When variable standoff is enabled, this is "
            "also the nominal distance unless --standoff-nominal-mm is supplied."
        ),
    )
    p.add_argument(
        "--standoff-min-mm",
        type=nonnegative_float,
        default=None,
        help=(
            "Minimum allowed sensor standoff. Supply together with "
            "--standoff-max-mm to enable variable-standoff optimization."
        ),
    )
    p.add_argument(
        "--standoff-max-mm",
        type=nonnegative_float,
        default=None,
        help=(
            "Maximum allowed sensor standoff. Supply together with "
            "--standoff-min-mm to enable variable-standoff optimization."
        ),
    )
    p.add_argument(
        "--standoff-nominal-mm",
        type=nonnegative_float,
        default=None,
        help=(
            "Preferred working distance inside the allowed range. Defaults to "
            "--sensor-standoff-mm."
        ),
    )
    p.add_argument(
        "--flip-normals",
        action="store_true",
        help="Flip all CAD normals if the STL winding points the sensor to the wrong side.",
    )
    p.add_argument(
        "--normal-radius-mm",
        type=positive_float,
        default=3.0,
        help=(
            "Compatibility option; ignored in mesh-face-normal mode. "
            "Normals are taken directly from the nearest CAD triangle."
        ),
    )
    p.add_argument(
        "--normal-field-samples",
        type=positive_int,
        default=120000,
        help="Compatibility option; ignored in mesh-face-normal mode.",
    )
    p.add_argument(
        "--orientation-smooth-window",
        type=odd_positive_int,
        default=5,
        help=(
            "Centered moving-average window applied to the local-average normals "
            "along the generated path. 1 disables path-wise smoothing; use odd "
            "values such as 3, 5, or 7."
        ),
    )
    p.add_argument(
        "--tangent-span",
        type=positive_int,
        default=3,
        help="Half-span in samples used for central-difference scan tangent estimation.",
    )
    p.add_argument(
        "--auto-break-angle-deg",
        type=nonnegative_float,
        default=45.0,
        help=(
            "Automatically split a scan when every topologically continuous "
            "face transition exceeds this normal-angle threshold. "
            "Default: 45 deg. Set 0 to disable angle-based breaks; topology "
            "discontinuities still split."
        ),
    )

    p.add_argument(
        "--frame-stride",
        type=positive_int,
        default=10,
        help="Show one sensor frame every N generated path samples.",
    )
    p.add_argument(
        "--sensor-axis-mm",
        type=positive_float,
        default=10.0,
        help="Displayed sensor-frame axis length.",
    )

    p.add_argument("--save-json", type=Path, default=None)
    p.add_argument("--save-csv", type=Path, default=None)
    p.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Open an interactive playback GUI with a pose slider, Play/Pause, "
            "sensor X/Y/Z axes, and the -Z viewing ray moving along the path."
        ),
    )
    p.add_argument(
        "--gui-pose-rate-hz",
        type=positive_float,
        default=12.0,
        help="Base GUI playback rate in generated poses per second.",
    )
    p.add_argument("--show", action="store_true", help="Open the static path overview viewer.")
    p.add_argument(
        "--playback-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    p.add_argument("--picker-worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--picker-worker-segment-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--picker-worker-previous-json", type=Path, default=None, help=argparse.SUPPRESS)

    return p.parse_args()


def main():
    args = parse_args()

    # Internal single-picker mode. Each segment picker gets its own process so
    # GLFW/OpenGL is initialized exactly once in that process.
    if args.picker_worker_output is not None:
        run_picker_worker(args)
        # Open3D/Filament may leave native GUI/render threads alive even after
        # the picker window and Python event loop have returned.  This worker is
        # intentionally a one-shot subprocess, so after the handoff JSON has
        # been written successfully, terminate the process immediately instead
        # of waiting for those native threads to unwind.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)

    # Internal playback-only mode. This branch intentionally does not create
    # VisualizerWithEditing picker windows, so gui.Application starts with a
    # clean graphics context.
    if args.playback_json is not None:
        mesh, _info = load_mesh_preserve_frame(
            args.cad, args.mesh_unit, args.weld_tolerance_mm
        )
        flat = load_flattened_playback_json(args.playback_json)
        print(f"[PLAYBACK] Loaded {len(flat['surface_points'])} poses from {args.playback_json}")
        visualize_sensor_motion_gui(
            mesh=mesh,
            waypoint_points=flat["waypoint_points"],
            surface_points=flat["surface_points"],
            sensor_points=flat["sensor_points"],
            frames=flat["frames"],
            arc_length_m=flat["arc_length_m"],
            axis_length_mm=args.sensor_axis_mm,
            initial_pose_rate_hz=args.gui_pose_rate_hz,
            segment_ids=flat["segment_ids"],
            waypoint_segment_ids=flat["waypoint_segment_ids"],
            waypoint_local_ids=flat["waypoint_local_ids"],
        )
        return

    mesh, info = load_mesh_preserve_frame(
        args.cad, args.mesh_unit, args.weld_tolerance_mm
    )
    V = np.asarray(mesh.vertices, dtype=np.float64)
    VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    T = np.asarray(mesh.triangles, dtype=np.int64)

    if len(VN) != len(V):
        raise RuntimeError("Mesh vertex normals are unavailable.")

    # One fresh subprocess = one independent scan stroke. This avoids the
    # black S1/S2 VisualizerWithEditing window seen on some Linux/Open3D builds.
    segment_specs = pick_waypoint_segments_fresh_processes(mesh, args)

    print("\n[BUILD SURFACE GRAPH]")
    adjacency = build_mesh_edge_graph(V, T)
    component_labels, component_sizes = graph_component_labels(adjacency)
    print(
        f"  connected components: {len(component_sizes)} | "
        f"largest={int(np.max(component_sizes)) if len(component_sizes) else 0} vertices"
    )

    for spec in segment_specs:
        sid = int(spec["segment_id"])
        comps = component_labels[spec["waypoint_vertex_ids"]]
        print(
            f"  S{sid} waypoint components: "
            + ", ".join(str(int(x)) for x in comps)
        )
        if np.any(comps != comps[0]):
            raise RuntimeError(
                f"S{sid} contains waypoints on disconnected mesh components. "
                "Increase --weld-tolerance-mm only if the CAD surfaces are truly connected."
            )

    print("[BUILD CAD PATH TOPOLOGY]")
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64).copy()
    tri_mag = np.linalg.norm(triangle_normals, axis=1)
    if np.any(~np.isfinite(tri_mag)) or np.any(tri_mag <= EPS):
        raise RuntimeError("CAD contains invalid triangle normals.")
    triangle_normals /= tri_mag[:, None]

    edge_to_faces, vertex_face_neighbors = build_mesh_path_topology(T)
    nonmanifold_edges = sum(1 for faces in edge_to_faces.values() if len(faces) > 2)
    boundary_edges = sum(1 for faces in edge_to_faces.values() if len(faces) == 1)

    print(f"  triangle faces : {len(triangle_normals):,}")
    print(f"  mesh edges     : {len(edge_to_faces):,}")
    print(f"  boundary edges : {boundary_edges:,}")
    print(f"  nonmanifold    : {nonmanifold_edges:,}")
    print("  normal source  : face strip attached to the actual Dijkstra mesh edges")

    generated_segments: list[dict] = []
    for spec in segment_specs:
        source_sid = int(spec["segment_id"])
        print(f"[GENERATE MANUAL SEGMENT S{source_sid}]")
        parts = build_scan_segments(
            segment_spec=spec,
            adjacency=adjacency,
            vertices=V,
            edge_to_faces=edge_to_faces,
            vertex_face_neighbors=vertex_face_neighbors,
            triangle_normals=triangle_normals,
            path_step_mm=args.path_step_mm,
            orientation_smooth_window=args.orientation_smooth_window,
            tangent_span=args.tangent_span,
            sensor_standoff_mm=args.sensor_standoff_mm,
            flip_normals=args.flip_normals,
            auto_break_angle_deg=args.auto_break_angle_deg,
            standoff_min_mm=args.standoff_min_mm,
            standoff_max_mm=args.standoff_max_mm,
            standoff_nominal_mm=args.standoff_nominal_mm,
        )
        generated_segments.extend(parts)

        if len(parts) == 1:
            print("  topology tracking: continuous -> no automatic split")
        else:
            print(f"  topology tracking: {len(parts)} scan blocks")
            for j, part in enumerate(parts):
                print(
                    f"    block {j}: graph edges "
                    f"{part['graph_edge_start']}..{part['graph_edge_end']} | "
                    f"break-before={part['break_reason_before']}"
                )

    renumber_scan_segments(generated_segments)

    flat = flatten_scan_segments(generated_segments)
    validate_scan_segments(generated_segments, print_report=True)

    print("\n" + "=" * 100)
    print("MANUAL WAYPOINTS -> INDEPENDENT SURFACE SCAN SEGMENTS")
    print("=" * 100)
    print(f"CAD unit interpretation : {info['input_unit']}")
    print(f"CAD diameter            : {info['diameter_mm']:.3f} mm")
    print(f"Connected components    : {len(component_sizes)}")
    print(f"Scan segments           : {len(generated_segments)}")
    print(f"Path pose spacing       : {args.path_step_mm:.3f} mm")
    if args.standoff_min_mm is None:
        print(f"Sensor standoff         : fixed {args.sensor_standoff_mm:.3f} mm")
    else:
        nominal_print = (
            args.sensor_standoff_mm
            if args.standoff_nominal_mm is None
            else args.standoff_nominal_mm
        )
        print(
            f"Sensor standoff         : optimized "
            f"[{args.standoff_min_mm:.3f}, {args.standoff_max_mm:.3f}] mm "
            f"(nominal {nominal_print:.3f} mm)"
        )
    print("Normal source           : Dijkstra-edge topology face tracking")
    print(f"Orientation smooth win  : {args.orientation_smooth_window}")
    print(f"Auto-break angle        : {args.auto_break_angle_deg:.3f} deg (0=disabled)")
    print("Between segments        : DISCONNECTED; robot reposition required")
    print("Frame convention        : +X laser / +Y motion / +Z CAD normal / view=-Z")

    total_poses = 0
    total_scan_length_mm = 0.0
    for seg in generated_segments:
        sid = int(seg["segment_id"])
        nposes = len(seg["surface_points"])
        length_mm = float(seg["arc_length_m"][-1] * 1000.0)
        total_poses += nposes
        total_scan_length_mm += length_mm
        face_ids = np.asarray(seg["triangle_face_ids"], dtype=np.int64)
        print(
            f"  S{sid}: waypoints={len(seg['waypoint_points'])}, "
            f"poses={nposes}, length={length_mm:.3f} mm, "
            f"face-normal-step={max_adjacent_normal_angle_deg(seg['raw_face_normals']):.2f}° "
            f"-> smoothed={max_adjacent_normal_angle_deg(seg['surface_normals']):.2f}°"
        )
        print(
            f"      CAD triangle faces used: {len(np.unique(face_ids))} unique | "
            f"tracked face switches={seg['face_switch_count']}"
        )
        print(
            f"      source manual S{seg['source_manual_segment_id']} | "
            f"auto-block={seg['auto_split_index']} | "
            f"break-before={seg['break_reason_before']} | "
            f"break-after={seg['break_reason_after']}"
        )
        standoff_mm = 1000.0 * np.asarray(seg["standoff_m"], dtype=np.float64)
        print(
            f"      standoff: min={np.min(standoff_mm):.3f} mm | "
            f"median={np.median(standoff_mm):.3f} mm | "
            f"max={np.max(standoff_mm):.3f} mm"
        )
        for j, (idx, p) in enumerate(
            zip(seg["waypoint_vertex_ids"], seg["waypoint_points"])
        ):
            print(
                f"      S{sid}-W{j}: vertex={int(idx):6d} | "
                + np.array2string(p * 1000.0, precision=3, suppress_small=True)
                + " mm"
            )

    print(f"Total scan poses        : {total_poses}")
    print(f"Total scan length       : {total_scan_length_mm:.3f} mm (gaps excluded)")

    if args.save_json is not None:
        export_json_segments(args.save_json, args.cad, generated_segments)
        print(f"Saved JSON              : {args.save_json}")

    if args.save_csv is not None:
        export_csv_segments(args.save_csv, generated_segments)
        print(f"Saved CSV               : {args.save_csv}")

    if args.gui:
        # Always launch playback in a fresh process. This prevents the Linux
        # Open3D/Filament GUI from inheriting stale graphics state from the
        # one-or-more VisualizerWithEditing picker windows used above.
        temporary_json = None
        if args.save_json is not None:
            gui_json = args.save_json.resolve()
        else:
            tmp = tempfile.NamedTemporaryFile(
                prefix="surface_scan_playback_", suffix=".json", delete=False
            )
            tmp.close()
            temporary_json = Path(tmp.name)
            export_json_segments(temporary_json, args.cad, generated_segments)
            gui_json = temporary_json

        try:
            launch_gui_in_fresh_process(args, gui_json)
        finally:
            if temporary_json is not None:
                try:
                    temporary_json.unlink(missing_ok=True)
                except Exception:
                    pass
    elif args.show:
        visualize_segmented_scan_path(
            mesh=mesh,
            segments=generated_segments,
            frame_stride=args.frame_stride,
            axis_length_mm=args.sensor_axis_mm,
        )


if __name__ == "__main__":
    main()