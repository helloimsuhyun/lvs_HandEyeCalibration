from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator, Mapping
import uuid

import numpy as np

from ..data import LaserScan
from .models import (
    AcquisitionGroup,
    CalibrationDataset,
    CalibrationTruth,
    PlaneTruth,
)


SCHEMA_NAME = "laser_handeye_calibration_dataset"
SCHEMA_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "scans.npz"

_REQUIRED_ARRAYS = {
    "scan_id",
    "plane_id",
    "scan_group_id",
    "sequence_index_in_group",
    "T_base_ef",
    "profile_offsets",
    "points_s",
    "channel_ids",
    "valid_mask",
    "scan_meta_json",
}

_CONVENTIONS = {
    "length_unit": "mm",
    "angle_unit": "deg",
    "transform": "T_A_B maps coordinates expressed in frame B into frame A",
    "robot_pose": "T_base_ef",
    "profile_points": "points_s are expressed in the sensor frame",
    "plane_equation": "normal_base dot point_base = offset_mm",
    "profile_storage": "packed points with profile_offsets",
}


def _json_compatible(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError(f"{path} contains a non-finite float")
        return converted
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), path)
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} dictionary keys must be strings")
            result[key] = _json_compatible(item, f"{path}.{key}")
        return result
    raise ValueError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _group_dict(group: AcquisitionGroup) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "name": group.name,
        "acquisition_role": group.acquisition_role,
        "motion_kind": group.motion_kind,
        "plane_id": group.plane_id,
        "include_in_calibration": group.include_in_calibration,
        "metadata": _json_compatible(group.metadata, f"group[{group.group_id}].metadata"),
    }


def _dataset_counts(dataset: CalibrationDataset) -> dict[str, Any]:
    group_counts = {
        group.name: int(
            np.count_nonzero(dataset.scan_group_ids == group.group_id)
        )
        for group in dataset.groups
    }
    plane_counts: dict[str, int] = {}
    for scan in dataset.scans:
        key = str(scan.plane_id)
        plane_counts[key] = plane_counts.get(key, 0) + 1
    return {
        "scans": len(dataset.scans),
        "point_records": dataset.num_points,
        "groups": group_counts,
        "planes": plane_counts,
    }


def _truth_summary(dataset: CalibrationDataset) -> dict[str, Any]:
    truth = dataset.truth
    return {
        "present": truth is not None,
        "has_handeye": bool(truth is not None and truth.T_ef_s_true is not None),
        "plane_count": 0 if truth is None else len(truth.planes),
        "has_true_robot_poses": bool(
            truth is not None and truth.T_base_ef_true is not None
        ),
        "has_commanded_robot_poses": bool(
            truth is not None and truth.T_base_ef_commanded is not None
        ),
    }


def _semantic_manifest(dataset: CalibrationDataset) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "conventions": _CONVENTIONS,
        "acquisition_mode": dataset.acquisition_mode,
        "source": dataset.source,
        "profile_state": dataset.profile_state,
        "channel_id_semantics": dataset.channel_id_semantics,
        "groups": [
            _group_dict(group)
            for group in dataset.groups
        ],
        "metadata": _json_compatible(dataset.metadata),
        "truth_metadata": (
            {}
            if dataset.truth is None
            else _json_compatible(dataset.truth.metadata, "truth.metadata")
        ),
        "counts": _dataset_counts(dataset),
        "truth": _truth_summary(dataset),
    }


def _scan_meta_json(scan: LaserScan, scan_index: int) -> str:
    metadata = dict(scan.meta)
    metadata.pop("channel_ids", None)
    return _canonical_json(
        _json_compatible(metadata, f"scans[{scan_index}].meta")
    )


def _pack_dataset(dataset: CalibrationDataset) -> dict[str, np.ndarray]:
    scan_count = len(dataset.scans)
    offsets = np.empty(scan_count + 1, dtype=np.int64)
    offsets[0] = 0
    point_blocks: list[np.ndarray] = []
    channel_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    meta_strings: list[str] = []

    for index, scan in enumerate(dataset.scans):
        points = np.asarray(scan.points_s, dtype=np.float64)
        channel_ids = np.asarray(scan.meta["channel_ids"], dtype=np.int64)
        point_blocks.append(points.copy())
        channel_blocks.append(channel_ids.copy())
        valid_blocks.append(np.all(np.isfinite(points), axis=1))
        offsets[index + 1] = offsets[index] + scan.num_points
        meta_strings.append(_scan_meta_json(scan, index))

    arrays: dict[str, np.ndarray] = {
        "scan_id": np.asarray(
            [int(scan.scan_id) for scan in dataset.scans],
            dtype=np.int64,
        ),
        "plane_id": np.asarray(
            [int(scan.plane_id) for scan in dataset.scans],
            dtype=np.int32,
        ),
        "scan_group_id": np.asarray(dataset.scan_group_ids, dtype=np.str_),
        "sequence_index_in_group": np.asarray(
            dataset.sequence_indices,
            dtype=np.int32,
        ),
        "T_base_ef": np.stack(
            [np.asarray(scan.T_base_ef, dtype=np.float64) for scan in dataset.scans],
            axis=0,
        ),
        "profile_offsets": offsets,
        "points_s": (
            np.concatenate(point_blocks, axis=0)
            if point_blocks
            else np.empty((0, 3), dtype=np.float64)
        ),
        "channel_ids": (
            np.concatenate(channel_blocks)
            if channel_blocks
            else np.empty(0, dtype=np.int64)
        ),
        "valid_mask": (
            np.concatenate(valid_blocks)
            if valid_blocks
            else np.empty(0, dtype=bool)
        ),
        "scan_meta_json": np.asarray(meta_strings, dtype=np.str_),
    }

    truth = dataset.truth
    if truth is None:
        return arrays
    if truth.T_ef_s_true is not None:
        arrays["T_ef_s_true"] = np.asarray(truth.T_ef_s_true, dtype=np.float64)
    if truth.T_base_ef_true is not None:
        arrays["T_base_ef_true"] = np.asarray(
            truth.T_base_ef_true,
            dtype=np.float64,
        )
    if truth.T_base_ef_commanded is not None:
        arrays["T_base_ef_commanded"] = np.asarray(
            truth.T_base_ef_commanded,
            dtype=np.float64,
        )
    if truth.planes:
        arrays["truth_plane_id"] = np.asarray(
            [plane.plane_id for plane in truth.planes],
            dtype=np.int32,
        )
        arrays["truth_plane_normal_base"] = np.stack(
            [plane.normal_base for plane in truth.planes],
            axis=0,
        ).astype(np.float64, copy=False)
        arrays["truth_plane_offset_mm"] = np.asarray(
            [plane.offset_mm for plane in truth.planes],
            dtype=np.float64,
        )
        has_frame = np.asarray(
            [plane.T_base_plane is not None for plane in truth.planes],
            dtype=bool,
        )
        frames = np.full((len(truth.planes), 4, 4), np.nan, dtype=np.float64)
        for index, plane in enumerate(truth.planes):
            if plane.T_base_plane is not None:
                frames[index] = plane.T_base_plane
        arrays["truth_plane_has_frame"] = has_frame
        arrays["truth_T_base_plane"] = frames
        arrays["truth_plane_meta_json"] = np.asarray(
            [
                _canonical_json(
                    _json_compatible(
                        plane.metadata,
                        f"truth.planes[{index}].metadata",
                    )
                )
                for index, plane in enumerate(truth.planes)
            ],
            dtype=np.str_,
        )
    return arrays


def _hash_array(digest, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"array {name!r} has forbidden object dtype")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")

    if array.dtype.kind in ("U", "S"):
        digest.update(b"utf8\0")
        for item in array.reshape(-1):
            text = str(item).encode("utf-8")
            digest.update(len(text).to_bytes(8, "little"))
            digest.update(text)
        return

    dtype = array.dtype
    if dtype.byteorder == ">" or (
        dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    canonical = np.ascontiguousarray(array)
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))


def _logical_hash_from_parts(
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> str:
    digest = sha256()
    digest.update(_canonical_json(semantic_manifest).encode("utf-8"))
    digest.update(b"\0arrays\0")
    for name in sorted(arrays):
        _hash_array(digest, name, arrays[name])
        digest.update(b"\0")
    return digest.hexdigest()


def logical_dataset_sha256(dataset: CalibrationDataset) -> str:
    """Hash logical arrays and all schema-relevant group/frame semantics."""
    if not isinstance(dataset, CalibrationDataset):
        raise TypeError("dataset must be a CalibrationDataset")
    return _logical_hash_from_parts(
        _pack_dataset(dataset),
        _semantic_manifest(dataset),
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_calibration_dataset(
    dataset: CalibrationDataset,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one portable dataset and commit it by a directory rename.

    A complete sibling staging directory is created first. On overwrite, the
    previous two-file dataset is moved to a recovery directory before the new
    directory is installed, so interruption cannot pair an old manifest with a
    new payload.
    """
    if not isinstance(dataset, CalibrationDataset):
        raise TypeError("dataset must be a CalibrationDataset")
    root = Path(directory)
    if root.is_symlink():
        raise ValueError(f"dataset directory must not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"dataset path is not a directory: {root}")

    existing_entries = list(root.iterdir()) if root.exists() else []
    if existing_entries and not overwrite:
        raise FileExistsError(
            f"dataset target already exists: {root}; pass overwrite=True"
        )
    allowed_names = {MANIFEST_FILENAME, PAYLOAD_FILENAME}
    unexpected = [
        path for path in existing_entries if path.name not in allowed_names
    ]
    if unexpected:
        raise FileExistsError(
            "refusing to replace a directory containing non-dataset files: "
            + ", ".join(str(path) for path in unexpected)
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = root.parent / f".{root.name}.{token}.staging"
    backup = root.parent / f".{root.name}.{token}.backup"
    stage.mkdir(exist_ok=False)
    stage_manifest = stage / MANIFEST_FILENAME
    stage_payload = stage / PAYLOAD_FILENAME

    try:
        arrays = _pack_dataset(dataset)
        semantic = _semantic_manifest(dataset)
        logical_hash = _logical_hash_from_parts(arrays, semantic)
        _atomic_npz(stage_payload, arrays)
        byte_hash = _file_sha256(stage_payload)

        manifest = {
            **semantic,
            "payload": {
                "file": PAYLOAD_FILENAME,
                "byte_sha256": byte_hash,
                "logical_sha256": logical_hash,
            },
        }
        _atomic_json(stage_manifest, manifest)

        if root.exists():
            os.replace(root, backup)
        try:
            os.replace(stage, root)
        except BaseException:
            if backup.exists() and not root.exists():
                os.replace(backup, root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return root / MANIFEST_FILENAME


def _manifest_semantic_subset(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_name",
        "schema_version",
        "conventions",
        "acquisition_mode",
        "source",
        "profile_state",
        "channel_id_semantics",
        "groups",
        "metadata",
        "truth_metadata",
        "counts",
        "truth",
    )
    missing = [key for key in keys if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required fields: {missing}")
    return {key: manifest[key] for key in keys}


def _validate_schema(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"unsupported dataset schema: {manifest.get('schema_name')!r}"
        )
    version = manifest.get("schema_version")
    if not isinstance(version, str):
        raise ValueError("schema_version must be a string")
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid schema_version: {version!r}")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {version}; expected {SCHEMA_VERSION}"
        )


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except ValueError as exception:
        raise ValueError(f"invalid or unsafe NPZ payload: {path}") from exception
    object_arrays = [name for name, value in arrays.items() if value.dtype.hasobject]
    if object_arrays:
        raise ValueError(f"object arrays are forbidden: {object_arrays}")
    missing = sorted(_REQUIRED_ARRAYS - arrays.keys())
    if missing:
        raise ValueError(f"dataset payload is missing required arrays: {missing}")
    return arrays


def _require_shape(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    value = arrays[name]
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        expected_text = "(" + ", ".join(
            "N" if item is None else str(item) for item in shape
        ) + ")"
        raise ValueError(f"{name} must have shape {expected_text}, got {value.shape}")
    return value


def _unpack_dataset(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> CalibrationDataset:
    scan_ids = _require_shape(arrays, "scan_id", (None,))
    scan_count = len(scan_ids)
    plane_ids = _require_shape(arrays, "plane_id", (scan_count,))
    group_ids = _require_shape(arrays, "scan_group_id", (scan_count,))
    sequence = _require_shape(
        arrays,
        "sequence_index_in_group",
        (scan_count,),
    )
    transforms = _require_shape(arrays, "T_base_ef", (scan_count, 4, 4))
    offsets = _require_shape(arrays, "profile_offsets", (scan_count + 1,))
    points = _require_shape(arrays, "points_s", (None, 3))
    channel_ids = _require_shape(arrays, "channel_ids", (len(points),))
    valid_mask = _require_shape(arrays, "valid_mask", (len(points),))
    scan_meta = _require_shape(arrays, "scan_meta_json", (scan_count,))

    for name, value in (
        ("scan_id", scan_ids),
        ("plane_id", plane_ids),
        ("scan_group_id", group_ids),
        ("sequence_index_in_group", sequence),
        ("profile_offsets", offsets),
        ("channel_ids", channel_ids),
    ):
        if name == "scan_group_id":
            continue
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must have integer dtype")
    if group_ids.dtype.kind not in ("U", "S"):
        raise ValueError("scan_group_id must have a Unicode/string dtype")
    if valid_mask.dtype != np.dtype(bool):
        raise ValueError("valid_mask must have boolean dtype")
    if offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
        raise ValueError("profile_offsets must start at 0, be monotonic, and end at P")
    expected_valid = np.all(np.isfinite(points), axis=1)
    if not np.array_equal(valid_mask, expected_valid):
        raise ValueError("valid_mask does not match finite rows in points_s")
    if np.any(np.isinf(points)):
        raise ValueError("points_s may contain finite values or NaNs, not infinities")

    scans: list[LaserScan] = []
    for index in range(scan_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        try:
            metadata = json.loads(str(scan_meta[index]))
        except json.JSONDecodeError as exception:
            raise ValueError(f"scan_meta_json[{index}] is invalid JSON") from exception
        if not isinstance(metadata, dict):
            raise ValueError(f"scan_meta_json[{index}] must encode an object")
        metadata["channel_ids"] = channel_ids[start:stop].astype(
            np.int64,
            copy=True,
        )
        scans.append(
            LaserScan(
                T_base_ef=transforms[index],
                points_s=points[start:stop],
                plane_id=int(plane_ids[index]),
                scan_id=int(scan_ids[index]),
                meta=metadata,
            )
        )

    raw_groups = manifest["groups"]
    if not isinstance(raw_groups, list):
        raise ValueError("manifest groups must be a list")
    groups = []
    for index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            raise ValueError(f"manifest groups[{index}] must be an object")
        try:
            groups.append(
                AcquisitionGroup(
                    group_id=group["group_id"],
                    name=group["name"],
                    acquisition_role=group["acquisition_role"],
                    motion_kind=group["motion_kind"],
                    plane_id=group.get("plane_id"),
                    include_in_calibration=group["include_in_calibration"],
                    metadata=group.get("metadata", {}),
                )
            )
        except KeyError as exception:
            raise ValueError(
                f"manifest groups[{index}] is missing {exception.args[0]!r}"
            ) from exception

    truth_present = any(
        name in arrays
        for name in (
            "T_ef_s_true",
            "T_base_ef_true",
            "T_base_ef_commanded",
            "truth_plane_id",
        )
    ) or bool(manifest.get("truth", {}).get("present", False))
    truth: CalibrationTruth | None = None
    if truth_present:
        planes: list[PlaneTruth] = []
        plane_keys = {
            "truth_plane_id",
            "truth_plane_normal_base",
            "truth_plane_offset_mm",
            "truth_plane_has_frame",
            "truth_T_base_plane",
            "truth_plane_meta_json",
        }
        present_plane_keys = plane_keys & arrays.keys()
        if present_plane_keys and present_plane_keys != plane_keys:
            missing = sorted(plane_keys - arrays.keys())
            raise ValueError(f"incomplete plane truth arrays; missing {missing}")
        if present_plane_keys:
            truth_plane_ids = _require_shape(arrays, "truth_plane_id", (None,))
            plane_count = len(truth_plane_ids)
            normals = _require_shape(
                arrays,
                "truth_plane_normal_base",
                (plane_count, 3),
            )
            plane_offsets = _require_shape(
                arrays,
                "truth_plane_offset_mm",
                (plane_count,),
            )
            has_frames = _require_shape(
                arrays,
                "truth_plane_has_frame",
                (plane_count,),
            )
            plane_frames = _require_shape(
                arrays,
                "truth_T_base_plane",
                (plane_count, 4, 4),
            )
            plane_meta = _require_shape(
                arrays,
                "truth_plane_meta_json",
                (plane_count,),
            )
            if has_frames.dtype != np.dtype(bool):
                raise ValueError("truth_plane_has_frame must have boolean dtype")
            for index in range(plane_count):
                try:
                    metadata = json.loads(str(plane_meta[index]))
                except json.JSONDecodeError as exception:
                    raise ValueError(
                        f"truth_plane_meta_json[{index}] is invalid JSON"
                    ) from exception
                planes.append(
                    PlaneTruth(
                        plane_id=int(truth_plane_ids[index]),
                        normal_base=normals[index],
                        offset_mm=float(plane_offsets[index]),
                        T_base_plane=(
                            plane_frames[index] if has_frames[index] else None
                        ),
                        metadata=metadata,
                    )
                )
        truth = CalibrationTruth(
            T_ef_s_true=arrays.get("T_ef_s_true"),
            planes=planes,
            T_base_ef_true=arrays.get("T_base_ef_true"),
            T_base_ef_commanded=arrays.get("T_base_ef_commanded"),
            metadata=manifest.get("truth_metadata", {}),
        )

    return CalibrationDataset(
        scans=scans,
        scan_group_ids=group_ids,
        sequence_indices=sequence,
        groups=groups,
        acquisition_mode=manifest["acquisition_mode"],
        source=manifest["source"],
        profile_state=manifest["profile_state"],
        truth=truth,
        metadata=manifest.get("metadata", {}),
        channel_id_semantics=manifest["channel_id_semantics"],
    )


def load_calibration_dataset(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> CalibrationDataset:
    """Load and validate one portable dataset without enabling pickle."""
    root = Path(directory)
    manifest_path = root / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    except json.JSONDecodeError as exception:
        raise ValueError(f"invalid dataset manifest JSON: {manifest_path}") from exception
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must contain a JSON object")
    _validate_schema(manifest)
    semantic = _manifest_semantic_subset(manifest)

    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("manifest payload must be an object")
    filename = payload.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("manifest payload.file must be a plain filename")
    payload_path = root / filename
    if verify_hashes:
        expected_file_hash = payload.get("byte_sha256")
        if not isinstance(expected_file_hash, str):
            raise ValueError("manifest payload.byte_sha256 is missing")
        actual_file_hash = _file_sha256(payload_path)
        if actual_file_hash != expected_file_hash:
            raise ValueError(
                "dataset payload byte SHA-256 mismatch; file may be corrupted"
            )

    arrays = _load_arrays(payload_path)
    actual_logical_hash = _logical_hash_from_parts(arrays, semantic)
    expected_logical_hash = payload.get("logical_sha256")
    if verify_hashes:
        if not isinstance(expected_logical_hash, str):
            raise ValueError("manifest payload.logical_sha256 is missing")
        if actual_logical_hash != expected_logical_hash:
            raise ValueError(
                "dataset logical SHA-256 mismatch; arrays or semantics changed"
            )

    dataset = _unpack_dataset(arrays, manifest)
    if verify_hashes and logical_dataset_sha256(dataset) != actual_logical_hash:
        raise ValueError("dataset changed during schema reconstruction")
    return dataset


def iter_calibration_dataset_directories(root: str | Path) -> Iterator[Path]:
    """Yield dataset directories below ``root`` in deterministic path order."""
    path = Path(root)
    if (path / MANIFEST_FILENAME).is_file():
        yield path
        return
    if not path.exists():
        return
    manifests = sorted(
        path.rglob(MANIFEST_FILENAME),
        key=lambda item: item.as_posix(),
    )
    for manifest in manifests:
        yield manifest.parent


def iter_calibration_datasets(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> Iterator[CalibrationDataset]:
    """Load every portable dataset below ``root`` in deterministic order."""
    for directory in iter_calibration_dataset_directories(root):
        yield load_calibration_dataset(directory, verify_hashes=verify_hashes)


__all__ = [
    "MANIFEST_FILENAME",
    "PAYLOAD_FILENAME",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "iter_calibration_dataset_directories",
    "iter_calibration_datasets",
    "load_calibration_dataset",
    "logical_dataset_sha256",
    "save_calibration_dataset",
]
