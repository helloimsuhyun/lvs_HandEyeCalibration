from __future__ import annotations

import copy
import ctypes
import ctypes.util
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .mesh import MeshError, prepare_sensor_mesh
from .transforms import (
    format_vector,
    matrix_to_quaternion,
    validate_transform,
    xyz_rpy_transform,
)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    model_path: Path
    manifest_path: Path
    parent_link: str
    sensor_bounds_m: tuple[np.ndarray, np.ndarray]
    sensor_mesh_cached: bool


def _resolve(base: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_package_root(base: Path, raw_path: str) -> Path:
    """Resolve a normal path or ``ros2://<package>`` share directory."""
    raw = str(raw_path)
    if raw.startswith("ros2://"):
        package = raw[len("ros2://") :].strip("/")
        if not package:
            raise BuildError("ros2:// package root requires a package name")
        try:
            prefix = subprocess.check_output(
                ["ros2", "pkg", "prefix", package],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise BuildError(
                f"Could not resolve ROS package {package!r}. "
                "Source ROS first and make sure the package is installed."
            ) from error
        share = Path(prefix) / "share" / package
        if not share.is_dir():
            raise BuildError(f"ROS package share directory does not exist: {share}")
        return share.resolve()
    return _resolve(base, raw)


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    with config_path.open("rt", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise BuildError("Configuration root must be a mapping")
    for section in ("robot", "sensor", "handeye", "output"):
        if section not in config:
            raise BuildError(f"Missing required configuration section: {section}")
    return config, config_path.parent


def _numbers(text: str | None, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != length:
        raise BuildError(f"Expected {length} numbers, received {text!r}")
    return values


def _joint_origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    if origin is None:
        return np.eye(4)
    return xyz_rpy_transform(
        _numbers(origin.get("xyz"), 3, (0.0, 0.0, 0.0)),
        _numbers(origin.get("rpy"), 3, (0.0, 0.0, 0.0)),
    )


def _find_attachment(urdf_root: ET.Element, tcp_link: str) -> tuple[str, np.ndarray]:
    """Collapse fixed joints from tcp_link to its nearest movable-body ancestor."""
    child_joints: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_joints[child.get("link", "")] = joint
    links = {link.get("name") for link in urdf_root.findall("link")}
    if tcp_link not in links:
        raise BuildError(f"TCP link {tcp_link!r} does not exist in URDF")

    current = tcp_link
    ancestor_to_tcp = np.eye(4)
    while current in child_joints:
        joint = child_joints[current]
        joint_type = joint.get("type", "fixed")
        if joint_type != "fixed":
            return current, ancestor_to_tcp
        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            raise BuildError(f"Fixed joint {joint.get('name')} has no parent link")
        ancestor_to_tcp = _joint_origin(joint) @ ancestor_to_tcp
        current = parent.get("link", "")
    return current, ancestor_to_tcp


def _fixed_transform_from_ancestor(
    urdf_root: ET.Element, ancestor: str, target_link: str
) -> np.ndarray:
    """Return ``^ancestor T_target`` along a fixed-joint-only chain."""
    if target_link == ancestor:
        return np.eye(4)
    child_joints: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_joints[child.get("link", "")] = joint
    current = target_link
    ancestor_to_target = np.eye(4)
    while current != ancestor:
        joint = child_joints.get(current)
        if joint is None:
            raise BuildError(
                f"Frame {target_link!r} is not a fixed descendant of {ancestor!r}"
            )
        if joint.get("type", "fixed") != "fixed":
            raise BuildError(
                f"Frame {target_link!r} reaches movable joint {joint.get('name')!r} "
                f"before ancestor {ancestor!r}"
            )
        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            raise BuildError(f"Joint {joint.get('name')!r} has no parent link")
        ancestor_to_target = _joint_origin(joint) @ ancestor_to_target
        current = parent.get("link", "")
    return ancestor_to_target


def _find_root_link(urdf_root: ET.Element) -> str:
    links = {link.get("name") for link in urdf_root.findall("link") if link.get("name")}
    children = {
        child.get("link")
        for joint in urdf_root.findall("joint")
        for child in [joint.find("child")]
        if child is not None and child.get("link")
    }
    roots = sorted(links - children)
    if len(roots) != 1:
        raise BuildError(f"URDF must have exactly one root link; found {roots}")
    return roots[0]


def _adjacent_link_pairs(urdf_root: ET.Element) -> list[tuple[str, str]]:
    pairs = []
    for joint in urdf_root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None and parent.get("link") and child.get("link"):
            pairs.append((parent.get("link", ""), child.get("link", "")))
    return pairs


def _collapse_fixed_link_name(
    link: str,
    body_names: set[str | None],
    adjacent_pairs: list[tuple[str, str]],
) -> str | None:
    """Map a URDF link to the MuJoCo body that owns it after fixed collapse."""
    parent_by_child = {child: parent for parent, child in adjacent_pairs}
    seen: set[str] = set()
    current = link
    while current not in body_names:
        if current in seen or current not in parent_by_child:
            return None
        seen.add(current)
        current = parent_by_child[current]
    return current


def _package_path(uri: str, package_roots: dict[str, Path]) -> Path:
    remainder = uri[len("package://") :]
    package, separator, relative = remainder.partition("/")
    if not separator or package not in package_roots:
        known = ", ".join(sorted(package_roots)) or "none"
        raise BuildError(f"Cannot resolve {uri!r}; configured package roots: {known}")
    return package_roots[package] / relative


def _convert_collada_to_obj(source: Path, output: Path) -> None:
    """Convert a ROS visual COLLADA mesh into a MuJoCo-readable OBJ.

    MuJoCo does not decode DAE files. Ubuntu's ROS/RViz installation already
    provides Assimp, so use its stable C API directly instead of adding a
    Python-only mesh dependency to the runtime environment.
    """
    library_name = ctypes.util.find_library("assimp")
    if not library_name:
        raise BuildError(
            "A COLLADA robot visual mesh was requested, but libassimp is not "
            "installed. Install libassimp5 (or disable render_visual_meshes)."
        )
    try:
        library = ctypes.CDLL(library_name)
    except OSError as error:
        raise BuildError(f"Could not load Assimp library {library_name}: {error}") from error
    library.aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    library.aiImportFile.restype = ctypes.c_void_p
    library.aiExportScene.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    library.aiExportScene.restype = ctypes.c_int
    library.aiReleaseImport.argtypes = [ctypes.c_void_p]
    library.aiGetErrorString.restype = ctypes.c_char_p

    # Triangulate, join duplicate vertices, and bake scene-node transforms;
    # OBJ cannot represent the DAE node hierarchy by itself.
    processing_flags = 0x8 | 0x2 | 0x100
    scene = library.aiImportFile(str(source).encode(), processing_flags)
    if not scene:
        detail = library.aiGetErrorString().decode(errors="replace")
        raise BuildError(f"Could not import COLLADA mesh {source}: {detail}")
    try:
        result = library.aiExportScene(
            scene, b"obj", str(output).encode(), 0
        )
        if result != 0:
            detail = library.aiGetErrorString().decode(errors="replace")
            raise BuildError(f"Could not convert COLLADA mesh {source}: {detail}")
    finally:
        library.aiReleaseImport(scene)

    # Assimp exports COLLADA's Z_UP content into OBJ's conventional Y_UP
    # coordinates. MuJoCo treats OBJ coordinates as Z_UP, so rotate +90 deg
    # about X to retain the same link-local frame as the UR collision STLs.
    converted_lines: list[str] = []
    for line in output.read_text(encoding="utf-8").splitlines(keepends=True):
        # A single MuJoCo mesh asset reads only one OBJ object/group. Merge the
        # material-separated DAE submeshes while retaining their shared indices.
        if line.startswith(("o ", "g ", "usemtl ", "mtllib ")):
            continue
        prefix, separator, values = line.partition("  ")
        if separator and prefix in {"v", "vn"}:
            xyz = values.split()
            if len(xyz) >= 3:
                x, y, z = (float(value) for value in xyz[:3])
                suffix = " " + " ".join(xyz[3:]) if len(xyz) > 3 else ""
                newline = "\n" if line.endswith("\n") else ""
                line = f"{prefix}  {x:.10g} {-z:.10g} {y:.10g}{suffix}{newline}"
        converted_lines.append(line)
    output.write_text("".join(converted_lines), encoding="utf-8")


def _prepare_urdf(
    source: Path,
    output: Path,
    package_roots: dict[str, Path],
    use_collision_meshes: bool,
    keep_visual_meshes: bool = False,
) -> tuple[ET.Element, Path]:
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "robot":
        raise BuildError(f"Expected a URDF <robot>, received <{root.tag}>")

    if use_collision_meshes:
        for link in root.findall("link"):
            if not link.findall("collision"):
                for visual in link.findall("visual"):
                    collision = copy.deepcopy(visual)
                    collision.tag = "collision"
                    link.append(collision)
            if not keep_visual_meshes:
                for visual in list(link.findall("visual")):
                    link.remove(visual)
    for removable in ("gazebo", "transmission", "ros2_control"):
        for element in list(root.findall(removable)):
            root.remove(element)

    mesh_directory = output.parent / "meshes" / "robot"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, Path] = {}
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            source_mesh = _package_path(filename, package_roots)
        elif filename.startswith("file://"):
            source_mesh = Path(filename[len("file://") :])
        else:
            raw = Path(filename).expanduser()
            source_mesh = raw if raw.is_absolute() else source.parent / raw
        source_mesh = source_mesh.resolve()
        if not source_mesh.is_file():
            raise BuildError(f"URDF mesh does not exist: {source_mesh}")
        if source_mesh not in copied:
            is_collada = source_mesh.suffix.lower() == ".dae"
            candidate = mesh_directory / (
                f"{source_mesh.stem}.obj" if is_collada else source_mesh.name
            )
            index = 1
            while (
                not is_collada
                and candidate.exists()
                and candidate.read_bytes() != source_mesh.read_bytes()
            ):
                candidate = mesh_directory / f"{source_mesh.stem}_{index}{source_mesh.suffix}"
                index += 1
            if is_collada:
                _convert_collada_to_obj(source_mesh, candidate)
            elif not is_collada and not candidate.exists():
                shutil.copy2(source_mesh, candidate)
            copied[source_mesh] = candidate
        mesh.set("filename", copied[source_mesh].relative_to(output.parent).as_posix())

    tree.write(output, encoding="utf-8", xml_declaration=True)
    return root, output


def _add_robot_visual_meshes(
    mjcf_root: ET.Element,
    urdf_root: ET.Element,
    adjacent_pairs: list[tuple[str, str]],
    rgba: str,
) -> int:
    """Add URDF visual meshes that MuJoCo's URDF importer discards.

    Collision meshes remain the physical geoms.  These added geoms are purely
    visual and are attached to the body that owns each possibly-collapsed URDF
    link using the full fixed-joint transform.
    """
    asset = mjcf_root.find("asset")
    if asset is None:
        raise BuildError("Converted MJCF has no asset section")
    bodies = {
        body.get("name"): body
        for body in mjcf_root.iter("body")
        if body.get("name")
    }
    body_names = set(bodies)
    count = 0
    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        body_name = _collapse_fixed_link_name(
            link_name, body_names, adjacent_pairs
        )
        if body_name is None:
            continue
        body = bodies[body_name]
        body_to_link = _fixed_transform_from_ancestor(
            urdf_root, body_name, link_name
        )
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            transform = body_to_link @ _joint_origin(visual)
            asset_name = f"robot_visual_mesh_{count}"
            geom_name = f"robot_visual_{count}_{link_name}"
            mesh_attributes = {
                "name": asset_name,
                "file": mesh.get("filename", ""),
            }
            if mesh.get("scale"):
                mesh_attributes["scale"] = mesh.get("scale", "")
            ET.SubElement(asset, "mesh", **mesh_attributes)
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="mesh",
                mesh=asset_name,
                pos=format_vector(transform[:3, 3]),
                quat=format_vector(matrix_to_quaternion(transform[:3, :3])),
                rgba=rgba,
                contype="0",
                conaffinity="0",
                group="0",
            )
            count += 1
    return count


def _load_transform_entry(
    config: dict[str, Any], base: Path, key: str, label: str
) -> np.ndarray:
    transform_path = _resolve(base, config["path"])
    if not transform_path.is_file():
        raise BuildError(f"Hand-eye transform file does not exist: {transform_path}")
    suffix = transform_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(transform_path.read_text(encoding="utf-8"))
        if key not in data:
            raise BuildError(f"Transform JSON has no {key!r} key")
        matrix = np.asarray(data[key], dtype=float)
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(transform_path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".npy":
        matrix = np.load(transform_path)
    else:
        raise BuildError("Hand-eye transform must be JSON, CSV, TXT, or NPY")
    try:
        matrix = validate_transform(matrix, label)
    except ValueError as error:
        raise BuildError(str(error)) from error
    units = str(config.get("translation_units", "m")).lower()
    scale = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3}.get(units)
    if scale is None:
        raise BuildError(f"Unsupported hand-eye translation_units: {units}")
    matrix = matrix.copy()
    matrix[:3, 3] *= scale
    return matrix


def _load_handeye(config: dict[str, Any], base: Path) -> np.ndarray:
    return _load_transform_entry(
        config, base, str(config.get("key", "T_tcp_sensor")), "T_tcp_sensor"
    )


def _load_sensor_physical(config: dict[str, Any], base: Path) -> np.ndarray | None:
    key = config.get("physical_key")
    if not key:
        return None
    return _load_transform_entry(config, base, str(key), "T_sensor_physical")


def _sensor_cad_transform(sensor: dict[str, Any], mesh_scale: float) -> np.ndarray:
    """Return T_sensor_cad: native CAD coordinates to measurement-frame metres."""
    frame = sensor.get("frame")
    if not frame:
        return np.eye(4)
    try:
        origin_cad = np.asarray(frame["origin_in_cad"], dtype=float).reshape(3)
        axes_cad_sensor = np.column_stack(
            [
                np.asarray(frame["x_axis_in_cad"], dtype=float).reshape(3),
                np.asarray(frame["y_axis_in_cad"], dtype=float).reshape(3),
                np.asarray(frame["z_axis_in_cad"], dtype=float).reshape(3),
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BuildError(
            "sensor.frame requires origin_in_cad and x/y/z_axis_in_cad 3-vectors"
        ) from error
    # Columns above express sensor axes in CAD. Its transpose maps CAD vectors
    # into the sensor frame.
    cad_to_sensor_rotation = axes_cad_sensor.T
    transform = np.eye(4)
    transform[:3, :3] = cad_to_sensor_rotation
    transform[:3, 3] = -cad_to_sensor_rotation @ (origin_cad * mesh_scale)
    try:
        return validate_transform(transform, "T_sensor_cad")
    except ValueError as error:
        raise BuildError(str(error)) from error


def _transform_bounds(
    low_native: np.ndarray,
    high_native: np.ndarray,
    mesh_scale: float,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [x, y, z]
            for x in (low_native[0], high_native[0])
            for y in (low_native[1], high_native[1])
            for z in (low_native[2], high_native[2])
        ],
        dtype=float,
    )
    points = (transform[:3, :3] @ (corners * mesh_scale).T).T + transform[:3, 3]
    return points.min(axis=0), points.max(axis=0)


def _add_environment(worldbody: ET.Element, config: dict[str, Any]) -> None:
    if not config.get("enabled", True):
        return
    ET.SubElement(
        worldbody,
        "geom",
        name="environment_floor",
        type="plane",
        size="2 2 0.05",
        rgba="0.22 0.25 0.28 1",
        friction="0.8 0.02 0.002",
        condim="4",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos="1 -1 2.2",
        dir="-0.4 0.4 -1",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="overview",
        pos="1.65 -1.65 1.35",
        xyaxes="0.707 0.707 0 -0.35 0.35 0.866",
    )


def _add_frame_triad(
    parent: ET.Element,
    *,
    name: str,
    transform: np.ndarray,
    origin_rgba: str,
    axis_length_m: float,
    axis_radius_m: float = 0.0032,
) -> None:
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    ET.SubElement(
        parent,
        "site",
        name=f"{name}_origin_preview",
        type="sphere",
        pos=format_vector(origin),
        size="0.008",
        rgba=origin_rgba,
    )
    colors = (
        ("x", "1.0 0.05 0.05 1.0"),
        ("y", "0.05 1.0 0.05 1.0"),
        ("z", "0.10 0.35 1.0 1.0"),
    )
    for axis_index, (axis_name, color) in enumerate(colors):
        tip = origin + rotation @ (np.eye(3)[axis_index] * axis_length_m)
        ET.SubElement(
            parent,
            "site",
            name=f"{name}_axis_{axis_name}_preview",
            type="cylinder",
            fromto=format_vector(np.concatenate([origin, tip])),
            size=str(axis_radius_m),
            rgba=color,
        )


def _add_sensor(
    root: ET.Element,
    parent_link: str,
    transform: np.ndarray,
    sensor_name: str,
    sensor_mesh: Path,
    mesh_scale: float,
    low_native: np.ndarray,
    high_native: np.ndarray,
    collision: dict[str, Any],
    safety_margin: float,
    sensor_to_cad: np.ndarray,
    *,
    frame_previews: dict[str, np.ndarray] | None = None,
    sensor_to_physical: np.ndarray | None = None,
    axis_length_m: float = 0.075,
) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise BuildError("Converted MJCF is missing asset/worldbody")
    ET.SubElement(
        asset,
        "mesh",
        name=f"{sensor_name}_visual_mesh",
        file=sensor_mesh.relative_to(sensor_mesh.parents[2]).as_posix(),
        scale=format_vector([mesh_scale] * 3),
    )

    if parent_link == "world":
        parent = worldbody
    else:
        parent = next((body for body in root.iter("body") if body.get("name") == parent_link), None)
        if parent is None:
            raise BuildError(
                f"Attachment body {parent_link!r} was not preserved by MuJoCo's URDF importer"
            )
    previews = frame_previews or {}
    preview_colors = {
        "flange": "1.0 0.45 0.05 1.0",
        "tool0": "0.05 0.90 1.0 1.0",
    }
    for preview_name, preview_transform in previews.items():
        _add_frame_triad(
            parent,
            name=preview_name,
            transform=preview_transform,
            origin_rgba=preview_colors.get(preview_name, "0.95 0.95 0.95 1.0"),
            axis_length_m=axis_length_m,
        )
    if "tool0" in previews:
        ET.SubElement(
            parent,
            "site",
            name="tool0_to_sensor_offset_preview",
            type="cylinder",
            fromto=format_vector(
                np.concatenate([previews["tool0"][:3, 3], transform[:3, 3]])
            ),
            size="0.0022",
            rgba="0.20 0.20 0.20 0.75",
        )

    body = ET.SubElement(
        parent,
        "body",
        name=sensor_name,
        pos=format_vector(transform[:3, 3]),
        quat=format_vector(matrix_to_quaternion(transform[:3, :3])),
    )
    ET.SubElement(body, "site", name="sensor_origin", type="sphere", size="0.009", rgba="1.0 0.05 0.85 1")
    sensor_identity = np.eye(4)
    _add_frame_triad(
        body,
        name="sensor",
        transform=sensor_identity,
        origin_rgba="1.0 0.05 0.85 0.01",
        axis_length_m=axis_length_m,
        axis_radius_m=0.0028,
    )
    if sensor_to_physical is not None:
        physical = sensor_to_physical
        _add_frame_triad(
            body,
            name="physical",
            transform=physical,
            origin_rgba="1.0 0.90 0.05 1.0",
            axis_length_m=axis_length_m,
            axis_radius_m=0.0036,
        )
        ET.SubElement(
            body,
            "site",
            name="sensor_to_physical_offset_preview",
            type="cylinder",
            fromto=format_vector(np.concatenate([np.zeros(3), physical[:3, 3]])),
            size="0.0025",
            rgba="1.0 0.75 0.15 0.85",
        )
    ET.SubElement(
        body,
        "geom",
        name=f"{sensor_name}_visual",
        type="mesh",
        mesh=f"{sensor_name}_visual_mesh",
        pos=format_vector(sensor_to_cad[:3, 3]),
        quat=format_vector(matrix_to_quaternion(sensor_to_cad[:3, :3])),
        rgba="0.12 0.18 0.24 1",
        contype="0",
        conaffinity="0",
        group="2",
    )

    collision_type = str(collision.get("type", "box")).lower()
    padding = float(collision.get("padding_m", 0.003))
    center_native = (low_native + high_native) / 2.0
    center = sensor_to_cad[:3, :3] @ (center_native * mesh_scale) + sensor_to_cad[:3, 3]
    mesh_quat = format_vector(matrix_to_quaternion(sensor_to_cad[:3, :3]))
    if collision_type == "box":
        half_size = (high_native - low_native) * mesh_scale / 2.0 + padding
        ET.SubElement(
            body,
            "geom",
            name=f"{sensor_name}_collision",
            type="box",
            pos=format_vector(center),
            size=format_vector(half_size),
            quat=mesh_quat,
            rgba="0.2 0.7 1 0.18",
            group="3",
            margin=str(safety_margin),
        )
    elif collision_type == "mesh":
        ET.SubElement(
            body,
            "geom",
            name=f"{sensor_name}_collision",
            type="mesh",
            mesh=f"{sensor_name}_visual_mesh",
            pos=format_vector(sensor_to_cad[:3, 3]),
            quat=mesh_quat,
            rgba="0.2 0.7 1 0.18",
            group="3",
            margin=str(safety_margin),
        )
    elif collision_type in {"boxes", "primitives"}:
        boxes = collision.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            raise BuildError(
                f"sensor collision type {collision_type!r} requires a non-empty boxes list"
            )
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise BuildError("Each sensor collision box must be a mapping")
            if "center_mm" in box and "half_size_mm" in box:
                box_center = np.asarray(box["center_mm"], dtype=float) * 1e-3
                half_size = np.asarray(box["half_size_mm"], dtype=float) * 1e-3
            elif "center_m" in box and "half_size_m" in box:
                box_center = np.asarray(box["center_m"], dtype=float)
                half_size = np.asarray(box["half_size_m"], dtype=float)
            else:
                raise BuildError(
                    "Collision box needs center_mm/half_size_mm or center_m/half_size_m"
                )
            if box_center.shape != (3,) or half_size.shape != (3,) or np.any(half_size <= 0):
                raise BuildError("Collision box center and positive half-size must be 3-vectors")
            rpy = np.asarray(box.get("rpy", [0.0, 0.0, 0.0]), dtype=float)
            if rpy.shape != (3,):
                raise BuildError("Collision box rpy must be a 3-vector in radians")
            box_rotation = xyz_rpy_transform([0, 0, 0], rpy)[:3, :3]
            ET.SubElement(
                body,
                "geom",
                name=str(box.get("name", f"{sensor_name}_collision_{index}")),
                type="box",
                pos=format_vector(box_center),
                size=format_vector(half_size + padding),
                quat=format_vector(matrix_to_quaternion(box_rotation)),
                rgba="0.2 0.7 1 0.18",
                group="3",
                margin=str(safety_margin),
            )
        if collision_type == "primitives":
            capsules = collision.get("capsules", [])
            if not isinstance(capsules, list):
                raise BuildError("sensor collision capsules must be a list")
            for index, capsule in enumerate(capsules):
                if not isinstance(capsule, dict):
                    raise BuildError("Each sensor collision capsule must be a mapping")
                if "from_mm" in capsule and "to_mm" in capsule and "radius_mm" in capsule:
                    point1 = np.asarray(capsule["from_mm"], dtype=float) * 1e-3
                    point2 = np.asarray(capsule["to_mm"], dtype=float) * 1e-3
                    radius = float(capsule["radius_mm"]) * 1e-3
                elif "from_m" in capsule and "to_m" in capsule and "radius_m" in capsule:
                    point1 = np.asarray(capsule["from_m"], dtype=float)
                    point2 = np.asarray(capsule["to_m"], dtype=float)
                    radius = float(capsule["radius_m"])
                else:
                    raise BuildError(
                        "Collision capsule needs from_mm/to_mm/radius_mm or metre equivalents"
                    )
                if (
                    point1.shape != (3,)
                    or point2.shape != (3,)
                    or not np.all(np.isfinite(point1))
                    or not np.all(np.isfinite(point2))
                    or not np.isfinite(radius)
                    or radius <= 0
                    or np.linalg.norm(point2 - point1) <= 0
                ):
                    raise BuildError("Collision capsule endpoints and radius are invalid")
                ET.SubElement(
                    body,
                    "geom",
                    name=str(capsule.get("name", f"{sensor_name}_capsule_{index}")),
                    type="capsule",
                    fromto=format_vector(np.concatenate([point1, point2])),
                    size=format_vector([radius + padding]),
                    rgba="1 0.55 0.05 0.35",
                    group="3",
                    margin=str(safety_margin),
                )
    else:
        raise BuildError(f"Unsupported sensor collision type: {collision_type}")


def _decorate_mjcf(
    mjcf_path: Path,
    robot: dict[str, Any],
    scene: dict[str, Any],
    sensor: dict[str, Any],
    sensor_mesh: Path,
    mesh_scale: float,
    low: np.ndarray,
    high: np.ndarray,
    parent_link: str,
    parent_to_sensor: np.ndarray,
    root_link: str,
    adjacent_pairs: list[tuple[str, str]],
    sensor_to_cad: np.ndarray,
    frame_previews: dict[str, np.ndarray] | None,
    sensor_to_physical: np.ndarray | None,
    urdf_root: ET.Element,
) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(scene.get("timestep", 0.002)))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.7 0.7 0.7")
    ET.SubElement(visual, "rgba", contactpoint="1 0.2 0.2 1", contactforce="1 0.4 0.1 1")

    safety_margin = float(scene.get("safety_margin_m", 0.005))
    for index, geom in enumerate(root.iter("geom")):
        if not geom.get("name"):
            geom.set("name", f"robot_geom_{index}")
        if geom.get("type") == "mesh":
            geom.set("rgba", "0.68 0.72 0.78 1")
            geom.set("margin", str(safety_margin))
            geom.set("group", "1")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise BuildError("Converted MJCF has no worldbody")
    # MuJoCo's URDF importer places the root-link geoms directly in world, which
    # makes the first joint body a sibling and defeats parent-child contact
    # filtering. Restore a named, fixed root body around the imported robot tree.
    if not any(body.get("name") == root_link for body in root.iter("body")):
        root_body = ET.Element("body", name=root_link)
        imported_geoms = list(worldbody.findall("geom"))
        imported_bodies = list(worldbody.findall("body"))
        for element in imported_geoms + imported_bodies:
            worldbody.remove(element)
            root_body.append(element)
        worldbody.insert(0, root_body)
    if bool(robot.get("render_visual_meshes", False)):
        count = _add_robot_visual_meshes(
            root,
            urdf_root,
            adjacent_pairs,
            str(robot.get("visual_rgba", "0.72 0.76 0.82 1")),
        )
        if count == 0:
            raise BuildError(
                "render_visual_meshes is enabled, but the robot URDF has no visual meshes"
            )
    _add_environment(worldbody, scene.get("floor", {}))
    _add_sensor(
        root,
        parent_link,
        parent_to_sensor,
        str(sensor.get("name", "laser_sensor")),
        sensor_mesh,
        mesh_scale,
        low,
        high,
        sensor.get("collision", {}),
        safety_margin,
        sensor_to_cad,
        frame_previews=frame_previews,
        sensor_to_physical=sensor_to_physical,
        axis_length_m=float(scene.get("frame_axis_length_m", 0.075)),
    )
    body_names = {body.get("name") for body in root.iter("body")}
    exclusions: list[tuple[str, str]] = []
    for parent, child in adjacent_pairs:
        parent_body = _collapse_fixed_link_name(parent, body_names, adjacent_pairs)
        child_body = _collapse_fixed_link_name(child, body_names, adjacent_pairs)
        pair = (parent_body, child_body)
        if (
            parent_body is not None
            and child_body is not None
            and parent_body != child_body
            and pair not in exclusions
        ):
            exclusions.append(pair)
    sensor_name = str(sensor.get("name", "laser_sensor"))
    if parent_link in body_names and sensor_name in body_names:
        exclusions.append((parent_link, sensor_name))
    if exclusions:
        contact = ET.SubElement(root, "contact")
        for parent, child in exclusions:
            ET.SubElement(contact, "exclude", body1=parent, body2=child)

    actuator = ET.SubElement(root, "actuator")
    for joint in root.iter("joint"):
        if joint.get("name"):
            ET.SubElement(
                actuator,
                "position",
                name=f"{joint.get('name')}_position",
                joint=joint.get("name", ""),
                kp=str(robot.get("position_kp", 100.0)),
                kv=str(robot.get("position_kv", 20.0)),
            )

    home = robot.get("home_qpos")
    if home is not None:
        keyframe = ET.SubElement(root, "keyframe")
        ET.SubElement(keyframe, "key", name="home", qpos=format_vector(home), ctrl=format_vector(home))

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def build_model(config_path: str | Path, force_cad: bool = False) -> BuildResult:
    config, base = load_config(Path(config_path))
    robot, sensor, handeye, output_config = (
        config["robot"], config["sensor"], config["handeye"], config["output"]
    )
    urdf_path = _resolve(base, robot["urdf"])
    cad_path = _resolve(base, sensor["cad"])
    if not urdf_path.is_file():
        raise BuildError(f"Robot URDF does not exist: {urdf_path}")
    if not cad_path.is_file():
        raise BuildError(f"Sensor CAD does not exist: {cad_path}")

    output_directory = _resolve(base, output_config.get("directory", "../build/default"))
    output_directory.mkdir(parents=True, exist_ok=True)
    package_roots = {
        name: _resolve_package_root(base, path)
        for name, path in robot.get("package_roots", {}).items()
    }
    prepared_urdf = output_directory / "robot_prepared.urdf"
    urdf_root, _ = _prepare_urdf(
        urdf_path,
        prepared_urdf,
        package_roots,
        bool(robot.get("use_collision_meshes", True)),
        bool(robot.get("render_visual_meshes", False)),
    )

    tcp_link = str(robot.get("tcp_link", "tcp"))
    attachment_link, attachment_to_tcp = _find_attachment(urdf_root, tcp_link)
    root_link = _find_root_link(urdf_root)
    adjacent_pairs = _adjacent_link_pairs(urdf_root)
    tcp_to_sensor = _load_handeye(handeye, base)
    sensor_to_physical = _load_sensor_physical(handeye, base)
    attachment_to_sensor = attachment_to_tcp @ tcp_to_sensor

    frame_previews: dict[str, np.ndarray] = {}
    for raw_frame in config.get("scene", {}).get("preview_frames", []):
        frame_name = str(raw_frame)
        frame_previews[frame_name] = _fixed_transform_from_ancestor(
            urdf_root, attachment_link, frame_name
        )

    preconverted = sensor.get("preconverted_mesh")
    preconverted_path = _resolve(base, preconverted) if preconverted else None
    preconverted_hash = sensor.get("preconverted_source_sha256")
    if preconverted_path is not None and not preconverted_hash:
        raise BuildError(
            "preconverted_mesh requires preconverted_source_sha256 to prevent stale CAD reuse"
        )
    if cad_path.suffix.lower() == ".obj" and preconverted_path is None:
        sensor_mesh_suffix = ".obj"
    elif preconverted_path is not None:
        sensor_mesh_suffix = preconverted_path.suffix.lower()
    else:
        sensor_mesh_suffix = ".stl"
    if sensor_mesh_suffix not in {".stl", ".obj"}:
        raise BuildError("preconverted_mesh must be STL or OBJ")
    sensor_mesh = output_directory / "meshes" / "sensor" / f"sensor{sensor_mesh_suffix}"
    try:
        low, high, face_count, cached = prepare_sensor_mesh(
            cad_path,
            sensor_mesh,
            sensor.get("tessellation", {}),
            force=force_cad,
            preconverted=preconverted_path,
            preconverted_source_sha256=str(preconverted_hash) if preconverted_hash else None,
        )
    except MeshError as error:
        raise BuildError(str(error)) from error
    units = str(sensor.get("cad_units", "mm")).lower()
    mesh_scale = {"m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3}.get(units)
    if mesh_scale is None:
        raise BuildError(f"Unsupported sensor cad_units: {units}")
    sensor_to_cad = _sensor_cad_transform(sensor, mesh_scale)

    try:
        model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
        mjcf_path = output_directory / "model.xml"
        mujoco.mj_saveLastXML(str(mjcf_path), model)
    except Exception as error:
        raise BuildError(f"MuJoCo could not import prepared URDF: {error}") from error

    _decorate_mjcf(
        mjcf_path,
        robot,
        config.get("scene", {}),
        sensor,
        sensor_mesh,
        mesh_scale,
        low,
        high,
        attachment_link,
        attachment_to_sensor,
        root_link,
        adjacent_pairs,
        sensor_to_cad,
        frame_previews,
        sensor_to_physical,
        urdf_root,
    )
    try:
        compiled = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as error:
        raise BuildError(f"Generated MJCF failed final compilation: {error}") from error

    sensor_low_m, sensor_high_m = _transform_bounds(low, high, mesh_scale, sensor_to_cad)
    manifest = {
        "model": str(mjcf_path),
        "robot_urdf": str(urdf_path),
        "sensor_cad": str(cad_path),
        "tcp_link": tcp_link,
        "attachment_link": attachment_link,
        "T_attachment_tcp_m": attachment_to_tcp.tolist(),
        "T_tcp_sensor_m": tcp_to_sensor.tolist(),
        "T_attachment_sensor_m": attachment_to_sensor.tolist(),
        "T_sensor_physical_m": sensor_to_physical.tolist() if sensor_to_physical is not None else None,
        "preview_frames": {name: value.tolist() for name, value in frame_previews.items()},
        "T_sensor_cad_m": sensor_to_cad.tolist(),
        "sensor_bounds_in_sensor_frame_m": [sensor_low_m.tolist(), sensor_high_m.tolist()],
        "sensor_mesh_faces": face_count,
        "sensor_mesh_cached": cached,
        "mujoco": {
            "nq": compiled.nq,
            "nv": compiled.nv,
            "nbody": compiled.nbody,
            "njnt": compiled.njnt,
            "ngeom": compiled.ngeom,
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return BuildResult(
        model_path=mjcf_path,
        manifest_path=manifest_path,
        parent_link=attachment_link,
        sensor_bounds_m=(sensor_low_m, sensor_high_m),
        sensor_mesh_cached=cached,
    )
