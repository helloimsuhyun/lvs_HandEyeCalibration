from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class MeshError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read binary or ASCII STL bounds without another mesh dependency."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(84)
        if len(header) >= 84:
            triangles = struct.unpack("<I", header[80:84])[0]
            if 84 + triangles * 50 == size:
                low = np.full(3, np.inf)
                high = np.full(3, -np.inf)
                for _ in range(triangles):
                    record = stream.read(50)
                    vertices = np.frombuffer(record, dtype="<f4", count=9, offset=12).reshape(3, 3)
                    low = np.minimum(low, vertices.min(axis=0))
                    high = np.maximum(high, vertices.max(axis=0))
                return low, high, triangles
    vertices = []
    with path.open("rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            tokens = line.strip().split()
            if len(tokens) == 4 and tokens[0].lower() == "vertex":
                vertices.append([float(value) for value in tokens[1:]])
    if not vertices:
        raise MeshError(f"Could not read STL vertices: {path}")
    points = np.asarray(vertices)
    return points.min(axis=0), points.max(axis=0), len(vertices) // 3


def obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    vertices = []
    faces = 0
    with path.open("rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces += 1
    if not vertices:
        raise MeshError(f"Could not read OBJ vertices: {path}")
    points = np.asarray(vertices)
    return points.min(axis=0), points.max(axis=0), faces


def mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    if path.suffix.lower() == ".stl":
        return stl_bounds(path)
    if path.suffix.lower() == ".obj":
        return obj_bounds(path)
    raise MeshError(f"Bounds are supported for STL/OBJ only, received: {path.suffix}")


def _convert_with_cadquery(source: Path, target: Path, options: dict[str, Any]) -> None:
    try:
        from cadquery import exporters, importers
    except ImportError as error:
        raise MeshError("CadQuery is not installed") from error
    suffix = source.suffix.lower()
    if suffix in {".step", ".stp"}:
        workplane = importers.importStep(str(source))
    else:
        raise MeshError(f"CadQuery conversion does not support {suffix}")
    exporters.export(
        workplane,
        str(target),
        tolerance=float(options.get("linear_deflection", 0.25)),
        angularTolerance=float(options.get("angular_deflection", 0.15)),
    )


def _convert_with_freecad(source: Path, target: Path, options: dict[str, Any]) -> None:
    executable = shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
    if not executable:
        raise MeshError("FreeCADCmd is not installed")
    script = target.with_suffix(".freecad.py")
    script.write_text(
        "import Mesh, Part\n"
        f"shape = Part.read({str(source)!r})\n"
        f"mesh = Mesh.Mesh()\nmesh.addFacets(shape.tessellate({float(options.get('linear_deflection', 0.25))})[1])\n"
        f"mesh.write({str(target)!r})\n",
        encoding="utf-8",
    )
    try:
        subprocess.run([executable, str(script)], check=True)
    finally:
        script.unlink(missing_ok=True)


def prepare_sensor_mesh(
    source: Path,
    target: Path,
    options: dict[str, Any],
    force: bool = False,
    preconverted: Path | None = None,
    preconverted_source_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Copy/convert CAD and return native-unit bounds plus cache status."""
    source_hash = sha256(source)
    metadata_path = target.with_suffix(target.suffix + ".json")
    if not force and target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") == source_hash:
            low, high, count = mesh_bounds(target)
            return low, high, count, True

    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    seeded = False
    if not force and preconverted is not None and preconverted_source_sha256 == source_hash:
        if not preconverted.is_file():
            raise MeshError(f"Configured preconverted mesh does not exist: {preconverted}")
        shutil.copy2(preconverted, target)
        seeded = True
    elif suffix in {".stl", ".obj"}:
        shutil.copy2(source, target)
    elif suffix in {".step", ".stp"}:
        errors = []
        for converter in (_convert_with_cadquery, _convert_with_freecad):
            try:
                converter(source, target, options)
                break
            except MeshError as error:
                errors.append(str(error))
        else:
            raise MeshError(
                "STEP conversion needs CadQuery (`pip install -e '.[cad]'`) or FreeCADCmd. "
                + " / ".join(errors)
            )
    else:
        raise MeshError(f"Unsupported sensor CAD format: {suffix}; use STEP, STL, or OBJ")

    low, high, count = mesh_bounds(target)
    metadata_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": source_hash,
                "bounds_native": [low.tolist(), high.tolist()],
                "face_count": count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return low, high, count, seeded
