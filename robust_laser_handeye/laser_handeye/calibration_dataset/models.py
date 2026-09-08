from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..data import LaserScan, ScansByPlane


_STABLE_CHANNEL_SEMANTICS = "stable_sensor_channel"
_PER_SCAN_CHANNEL_SEMANTICS = "per_scan_ordinal"


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _copy_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-6,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} rotation must lie in SO(3)")
    return transform.copy()


def _copy_transform_stack(
    value: np.ndarray | Sequence[np.ndarray] | None,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    transforms = np.asarray(value, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4)")
    return np.stack(
        [_copy_transform(transform, f"{name}[{index}]") for index, transform in enumerate(transforms)],
        axis=0,
    )


def _validated_channel_ids(scan: LaserScan, scan_index: int) -> np.ndarray:
    raw_ids = scan.meta.get("channel_ids")
    if raw_ids is None:
        raise ValueError(
            f"scan {scan_index} has no channel_ids; a portable capture must "
            "preserve original sensor-channel indices explicitly"
        )
    channel_ids = np.asarray(raw_ids)
    if channel_ids.ndim != 1 or len(channel_ids) != scan.num_points:
        raise ValueError(
            f"scan {scan_index} channel_ids must have one entry per point"
        )
    if not np.issubdtype(channel_ids.dtype, np.integer):
        if not np.all(np.isfinite(channel_ids)) or not np.all(
            np.equal(channel_ids, np.round(channel_ids))
        ):
            raise ValueError(f"scan {scan_index} channel_ids must be integers")
        channel_ids = np.round(channel_ids).astype(np.int64)
    else:
        channel_ids = channel_ids.astype(np.int64, copy=False)
    if np.any(channel_ids < 0):
        raise ValueError(f"scan {scan_index} channel_ids must be non-negative")
    if len(np.unique(channel_ids)) != len(channel_ids):
        raise ValueError(
            f"scan {scan_index} channel_ids must be unique within the profile"
        )
    return channel_ids.copy()


def _copy_scan(scan: LaserScan, *, scan_index: int) -> LaserScan:
    if not isinstance(scan, LaserScan):
        raise TypeError(f"scans[{scan_index}] must be a LaserScan")
    points = np.asarray(scan.points_s, dtype=np.float64)
    if np.any(np.isinf(points)):
        raise ValueError(
            f"scans[{scan_index}].points_s may contain finite values or NaNs, not infinities"
        )
    channel_ids = _validated_channel_ids(scan, scan_index)
    metadata = deepcopy(scan.meta)
    metadata["channel_ids"] = channel_ids
    return LaserScan(
        T_base_ef=_copy_transform(
            scan.T_base_ef,
            f"scans[{scan_index}].T_base_ef",
        ),
        points_s=points.copy(),
        plane_id=scan.plane_id,
        scan_id=scan_index if scan.scan_id is None else scan.scan_id,
        meta=metadata,
    )


@dataclass(frozen=True)
class AcquisitionGroup:
    """One non-overlapping acquisition subset within a capture.

    ``acquisition_role`` describes why a group was acquired (for example,
    ``calibration``, ``reference`` or ``bootstrap``). ``motion_kind`` describes
    its kinematics (for example, ``pure_translation``, ``composite``,
    ``circular`` or ``general_6dof``). Keeping these concepts separate lets a
    portable dataset serve several calibration algorithms without encoding one
    solver in the file format.
    """

    group_id: str
    acquisition_role: str
    motion_kind: str
    plane_id: int | None = None
    include_in_calibration: bool = True
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        group_id = _nonempty_text(self.group_id, "group_id")
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(
            self,
            "name",
            group_id if self.name is None else _nonempty_text(self.name, "name"),
        )
        object.__setattr__(
            self,
            "acquisition_role",
            _nonempty_text(self.acquisition_role, "acquisition_role"),
        )
        object.__setattr__(
            self,
            "motion_kind",
            _nonempty_text(self.motion_kind, "motion_kind"),
        )
        if self.plane_id is not None:
            object.__setattr__(
                self,
                "plane_id",
                _integer(self.plane_id, "plane_id"),
            )
        if not isinstance(self.include_in_calibration, (bool, np.bool_)):
            raise ValueError("include_in_calibration must be boolean")
        object.__setattr__(
            self,
            "include_in_calibration",
            bool(self.include_in_calibration),
        )
        if not isinstance(self.metadata, dict):
            raise ValueError("AcquisitionGroup.metadata must be a dictionary")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    @property
    def id(self) -> str:
        return self.group_id


@dataclass(frozen=True)
class PlaneTruth:
    """Optional signed ground truth for one physical calibration plane."""

    plane_id: int
    normal_base: np.ndarray
    offset_mm: float
    T_base_plane: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_id", _integer(self.plane_id, "plane_id"))
        normal = np.asarray(self.normal_base, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("normal_base must contain three finite values")
        norm = float(np.linalg.norm(normal))
        if norm <= np.finfo(float).eps:
            raise ValueError("normal_base must be nonzero")
        offset = float(self.offset_mm)
        if not np.isfinite(offset):
            raise ValueError("offset_mm must be finite")
        # Preserve the orientation and sign while making the equation unit-normal.
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
            normal = normal / norm
            offset = offset / norm
        object.__setattr__(self, "normal_base", normal.copy())
        object.__setattr__(self, "offset_mm", offset)
        if self.T_base_plane is not None:
            plane_transform = _copy_transform(
                self.T_base_plane,
                "T_base_plane",
            )
            if not np.allclose(
                plane_transform[:3, 2],
                normal,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "T_base_plane normal axis must match normal_base"
                )
            frame_offset = float(normal @ plane_transform[:3, 3])
            if not np.isclose(
                frame_offset,
                offset,
                rtol=1e-9,
                atol=1e-6,
            ):
                raise ValueError(
                    "T_base_plane origin must lie on the signed plane equation"
                )
            object.__setattr__(
                self,
                "T_base_plane",
                plane_transform,
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("PlaneTruth.metadata must be a dictionary")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass
class CalibrationTruth:
    """Simulation/evaluation-only values; calibration inputs never require them."""

    T_ef_s_true: np.ndarray | None = None
    planes: Sequence[PlaneTruth] = field(default_factory=tuple)
    T_base_ef_true: np.ndarray | Sequence[np.ndarray] | None = None
    T_base_ef_commanded: np.ndarray | Sequence[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.T_ef_s_true is not None:
            self.T_ef_s_true = _copy_transform(self.T_ef_s_true, "T_ef_s_true")
        self.planes = tuple(self.planes)
        if any(not isinstance(plane, PlaneTruth) for plane in self.planes):
            raise TypeError("planes must contain only PlaneTruth values")
        plane_ids = [plane.plane_id for plane in self.planes]
        if len(set(plane_ids)) != len(plane_ids):
            raise ValueError("PlaneTruth.plane_id values must be unique")
        self.T_base_ef_true = _copy_transform_stack(
            self.T_base_ef_true,
            "T_base_ef_true",
        )
        self.T_base_ef_commanded = _copy_transform_stack(
            self.T_base_ef_commanded,
            "T_base_ef_commanded",
        )
        if not isinstance(self.metadata, dict):
            raise ValueError("CalibrationTruth.metadata must be a dictionary")
        self.metadata = deepcopy(self.metadata)


@dataclass
class CalibrationDataset:
    """Algorithm-neutral laser hand-eye capture with explicit acquisition groups.

    Every profile carries explicit channel IDs. Fixed-width profiles should
    retain invalid sensor channels as NaN rows; truly ragged profiles may keep
    only returned channels, provided their original channel IDs are retained.
    """

    scans: Sequence[LaserScan]
    scan_group_ids: Sequence[str]
    groups: Sequence[AcquisitionGroup]
    acquisition_mode: str
    source: str
    profile_state: str
    sequence_indices: Sequence[int] | None = None
    truth: CalibrationTruth | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    channel_id_semantics: str = _STABLE_CHANNEL_SEMANTICS

    def __post_init__(self) -> None:
        self.acquisition_mode = _nonempty_text(
            self.acquisition_mode,
            "acquisition_mode",
        )
        self.source = _nonempty_text(self.source, "source")
        self.profile_state = _nonempty_text(self.profile_state, "profile_state")
        if self.channel_id_semantics not in (
            _STABLE_CHANNEL_SEMANTICS,
            _PER_SCAN_CHANNEL_SEMANTICS,
        ):
            raise ValueError(
                "channel_id_semantics must be 'stable_sensor_channel' or "
                "'per_scan_ordinal'"
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("CalibrationDataset.metadata must be a dictionary")
        self.metadata = deepcopy(self.metadata)

        self.groups = tuple(self.groups)
        if not self.groups:
            raise ValueError("groups must not be empty")
        if any(not isinstance(group, AcquisitionGroup) for group in self.groups):
            raise TypeError("groups must contain only AcquisitionGroup values")
        group_ids = [group.group_id for group in self.groups]
        group_names = [group.name for group in self.groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("AcquisitionGroup.group_id values must be unique")
        if len(set(group_names)) != len(group_names):
            raise ValueError("AcquisitionGroup.name values must be unique")
        known_group_ids = set(group_ids)

        self.scans = [
            _copy_scan(scan, scan_index=index)
            for index, scan in enumerate(self.scans)
        ]
        if not self.scans:
            raise ValueError("scans must not be empty")
        scan_ids = [scan.scan_id for scan in self.scans]
        if len(set(scan_ids)) != len(scan_ids):
            raise ValueError("scan_id values must be unique within a dataset")

        raw_group_ids = list(self.scan_group_ids)
        if len(raw_group_ids) != len(self.scans):
            raise ValueError("scan_group_ids must have one entry per scan")
        self.scan_group_ids = np.asarray(
            [
                _nonempty_text(value, f"scan_group_ids[{index}]")
                for index, value in enumerate(raw_group_ids)
            ],
            dtype=np.str_,
        )
        unknown = sorted(set(map(str, self.scan_group_ids)) - known_group_ids)
        if unknown:
            raise ValueError(f"scan_group_ids reference unknown groups: {unknown}")

        if self.sequence_indices is None:
            counters = {group_id: 0 for group_id in known_group_ids}
            sequence_indices: list[int] = []
            for group_id in map(str, self.scan_group_ids):
                sequence_indices.append(counters[group_id])
                counters[group_id] += 1
        else:
            sequence_indices = [
                _integer(value, f"sequence_indices[{index}]")
                for index, value in enumerate(self.sequence_indices)
            ]
            if len(sequence_indices) != len(self.scans):
                raise ValueError("sequence_indices must have one entry per scan")
        seen: set[tuple[str, int]] = set()
        for group_id, sequence_index in zip(
            map(str, self.scan_group_ids),
            sequence_indices,
        ):
            key = (group_id, sequence_index)
            if key in seen:
                raise ValueError(
                    "(scan_group_id, sequence_index) pairs must be unique"
                )
            seen.add(key)
        self.sequence_indices = np.asarray(sequence_indices, dtype=np.int32)

        if self.truth is not None and not isinstance(self.truth, CalibrationTruth):
            raise TypeError("truth must be CalibrationTruth or None")
        if self.truth is not None:
            for name in ("T_base_ef_true", "T_base_ef_commanded"):
                transforms = getattr(self.truth, name)
                if transforms is not None and len(transforms) != len(self.scans):
                    raise ValueError(f"truth.{name} must have one transform per scan")

        groups_by_id = self.groups_by_id
        for index, scan in enumerate(self.scans):
            group = groups_by_id[str(self.scan_group_ids[index])]
            if group.plane_id is not None and scan.plane_id != group.plane_id:
                raise ValueError(
                    f"scan {index} plane_id={scan.plane_id} disagrees with "
                    f"group {group.group_id!r} plane_id={group.plane_id}"
                )

    @property
    def groups_by_id(self) -> dict[str, AcquisitionGroup]:
        return {group.group_id: group for group in self.groups}

    @property
    def groups_by_name(self) -> dict[str, AcquisitionGroup]:
        return {group.name: group for group in self.groups}

    @property
    def num_points(self) -> int:
        return int(sum(scan.num_points for scan in self.scans))

    def group(self, value: int | str) -> AcquisitionGroup:
        group_id = _nonempty_text(str(value), "group")
        try:
            return self.groups_by_id[group_id]
        except KeyError as exception:
            try:
                return self.groups_by_name[group_id]
            except KeyError:
                raise KeyError(
                    f"unknown acquisition group id or name: {group_id!r}"
                ) from exception

    def iter_group_scans(
        self,
        group: int | str,
    ) -> Iterator[LaserScan]:
        group_id = self.group(group).group_id
        indices = [
            index
            for index, candidate in enumerate(map(str, self.scan_group_ids))
            if candidate == group_id
        ]
        indices.sort(key=lambda index: int(self.sequence_indices[index]))
        for index in indices:
            yield _copy_scan(self.scans[index], scan_index=index)

    def _selected_group_ids(
        self,
        groups: Iterable[int | str] | None,
        *,
        include_excluded: bool,
    ) -> set[str]:
        if groups is None:
            return {
                group.group_id
                for group in self.groups
                if include_excluded or group.include_in_calibration
            }
        selected = {self.group(value).group_id for value in groups}
        if include_excluded:
            return selected
        included = {
            group.group_id
            for group in self.groups
            if group.include_in_calibration
        }
        return selected & included

    def to_scans_by_plane(
        self,
        *,
        groups: Iterable[int | str] | None = None,
        include_excluded: bool = False,
    ) -> ScansByPlane:
        """Convert selected acquisition groups to the iterative-solver input."""
        selected = self._selected_group_ids(
            groups,
            include_excluded=include_excluded,
        )
        result: ScansByPlane = {}
        # The group declaration order and each group's explicit sequence index
        # are the portable acquisition order, even if asynchronously captured
        # scans were merged into the file in another row order.
        for group in self.groups:
            if group.group_id not in selected:
                continue
            for scan in self.iter_group_scans(group.group_id):
                result.setdefault(scan.plane_id, []).append(scan)
        if not result:
            raise ValueError("no scans were selected for calibration")
        return result

__all__ = [
    "AcquisitionGroup",
    "CalibrationDataset",
    "CalibrationTruth",
    "PlaneTruth",
]
