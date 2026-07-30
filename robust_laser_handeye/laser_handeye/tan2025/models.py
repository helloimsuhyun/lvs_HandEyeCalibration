from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..data import LaserScan


def _copy_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return transform.copy()


@dataclass
class Tan2025Dataset:
    """The two motion groups required by the Tan et al. closed-form method.

    ``translation_scans`` must have the same tool orientation.  Their profile
    samples also need a stable ray/channel correspondence because paper
    equations (17)--(23) use ``d[0, j] - d[i, j]``.

    A simulator should store integer channel IDs in
    ``scan.meta["channel_ids"]``.  A real sensor should do the same, or keep a
    fixed-width profile and represent invalid returns with NaNs.  If channel
    IDs are omitted, equal row indices are treated as corresponding channels.
    ``composite_scans`` may have independent valid-point masks because only a
    PCA line and its centroid are used from each scan.
    """

    translation_scans: Sequence[LaserScan]
    composite_scans: Sequence[LaserScan]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.translation_scans = list(self.translation_scans)
        self.composite_scans = list(self.composite_scans)
        self.metadata = dict(self.metadata)
        if not self.translation_scans:
            raise ValueError("translation_scans must not be empty")
        if not self.composite_scans:
            raise ValueError("composite_scans must not be empty")

    @classmethod
    def from_tagged_scans(
        cls,
        scans: Sequence[LaserScan],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "Tan2025Dataset":
        """Partition backend scans using ``meta['motion_kind']`` tags."""
        translation: list[LaserScan] = []
        composite: list[LaserScan] = []
        for index, scan in enumerate(scans):
            motion_kind = scan.meta.get("motion_kind")
            if motion_kind == "translation":
                translation.append(scan)
            elif motion_kind == "composite":
                composite.append(scan)
            else:
                raise ValueError(
                    f"scan {index} has motion_kind={motion_kind!r}; expected "
                    "'translation' or 'composite'"
                )
        return cls(translation, composite, metadata=metadata or {})

    @property
    def all_scans(self) -> list[LaserScan]:
        return [*self.translation_scans, *self.composite_scans]

    @property
    def num_points(self) -> int:
        return int(sum(scan.num_points for scan in self.all_scans))

    def aligned_translation_points(self) -> list[np.ndarray]:
        """Return translation profiles aligned on their common channel IDs."""
        scans = self.translation_scans
        raw_ids = [scan.meta.get("channel_ids") for scan in scans]
        have_ids = [item is not None for item in raw_ids]

        if any(have_ids) and not all(have_ids):
            raise ValueError(
                "translation profiles must either all provide channel_ids or "
                "all rely on row-index correspondence"
            )

        if not any(have_ids):
            lengths = {scan.num_points for scan in scans}
            if len(lengths) != 1:
                raise ValueError(
                    "translation profiles have different lengths; preserve "
                    "channel_ids or fixed-width NaN rows during capture"
                )
            stacked = np.stack([scan.points_s for scan in scans], axis=0)
            common_valid = np.all(np.isfinite(stacked), axis=(0, 2))
            aligned = [points[common_valid].copy() for points in stacked]
        else:
            id_arrays: list[np.ndarray] = []
            lookup: list[dict[int, int]] = []
            for scan, raw in zip(scans, raw_ids):
                ids = np.asarray(raw)
                if ids.ndim != 1 or len(ids) != scan.num_points:
                    raise ValueError(
                        "each channel_ids array must have one entry per point"
                    )
                if not np.issubdtype(ids.dtype, np.integer):
                    if not np.all(np.equal(ids, np.round(ids))):
                        raise ValueError("channel_ids must be integers")
                    ids = np.round(ids).astype(np.int64)
                else:
                    ids = ids.astype(np.int64, copy=False)
                if len(np.unique(ids)) != len(ids):
                    raise ValueError("channel_ids must be unique within a profile")
                id_arrays.append(ids)
                lookup.append({int(channel): i for i, channel in enumerate(ids)})

            common = set(int(value) for value in id_arrays[0])
            for ids in id_arrays[1:]:
                common.intersection_update(int(value) for value in ids)
            common_ids = sorted(common)
            aligned = []
            for scan, positions in zip(scans, lookup):
                indices = [positions[channel] for channel in common_ids]
                aligned.append(scan.points_s[indices].copy())

            if aligned:
                stacked = np.stack(aligned, axis=0)
                common_valid = np.all(np.isfinite(stacked), axis=(0, 2))
                aligned = [points[common_valid].copy() for points in stacked]

        if not aligned or len(aligned[0]) < 3:
            raise ValueError(
                "translation profiles need at least three common finite channels"
            )
        return aligned

    def aligned_translation_pair(
        self,
        first_index: int,
        second_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two translation profiles without a global channel intersection.

        Paper equations (17)--(23) compare the reference profile with one pose
        at a time, while the centroid constraint compares one pose pair at a
        time.  Keeping the intersection local prevents independent dropouts
        from eliminating a channel merely because it is absent in an unrelated
        third scan.
        """
        count = len(self.translation_scans)
        if not 0 <= first_index < count or not 0 <= second_index < count:
            raise IndexError("translation scan index is out of range")
        first = self.translation_scans[first_index]
        second = self.translation_scans[second_index]
        first_raw_ids = first.meta.get("channel_ids")
        second_raw_ids = second.meta.get("channel_ids")

        if (first_raw_ids is None) != (second_raw_ids is None):
            raise ValueError(
                "paired translation profiles must both provide channel_ids or "
                "both rely on row-index correspondence"
            )
        if first_raw_ids is None:
            if first.num_points != second.num_points:
                raise ValueError(
                    "paired translation profiles have different lengths; "
                    "preserve channel_ids or fixed-width NaN rows during capture"
                )
            first_points = first.points_s
            second_points = second.points_s
        else:
            first_ids = self._validated_channel_ids(first, first_raw_ids)
            second_ids = self._validated_channel_ids(second, second_raw_ids)
            first_lookup = {
                int(channel): index for index, channel in enumerate(first_ids)
            }
            second_lookup = {
                int(channel): index for index, channel in enumerate(second_ids)
            }
            common_ids = sorted(set(first_lookup).intersection(second_lookup))
            first_points = first.points_s[
                [first_lookup[channel] for channel in common_ids]
            ]
            second_points = second.points_s[
                [second_lookup[channel] for channel in common_ids]
            ]

        common_valid = np.all(np.isfinite(first_points), axis=1)
        common_valid &= np.all(np.isfinite(second_points), axis=1)
        first_aligned = first_points[common_valid].copy()
        second_aligned = second_points[common_valid].copy()
        if len(first_aligned) < 3:
            raise ValueError(
                f"translation scan pair ({first_index}, {second_index}) needs "
                "at least three common finite channels"
            )
        return first_aligned, second_aligned

    def translation_channel_maps(self) -> list[dict[int, np.ndarray]]:
        """Return one ``channel_id -> point`` map for each translation scan."""
        raw_ids = [
            scan.meta.get("channel_ids") for scan in self.translation_scans
        ]
        have_ids = [item is not None for item in raw_ids]
        if any(have_ids) and not all(have_ids):
            raise ValueError(
                "translation profiles must either all provide channel_ids or "
                "all rely on row-index correspondence"
            )
        if not any(have_ids):
            lengths = {scan.num_points for scan in self.translation_scans}
            if len(lengths) != 1:
                raise ValueError(
                    "translation profiles have different lengths; preserve "
                    "channel_ids or fixed-width NaN rows during capture"
                )
        maps: list[dict[int, np.ndarray]] = []
        for scan, raw in zip(self.translation_scans, raw_ids):
            if raw is None:
                ids = np.arange(scan.num_points, dtype=np.int64)
            else:
                ids = self._validated_channel_ids(scan, raw)
            maps.append(
                {
                    int(channel): scan.points_s[index]
                    for index, channel in enumerate(ids)
                }
            )
        return maps

    @staticmethod
    def _validated_channel_ids(
        scan: LaserScan,
        raw_ids: Any,
    ) -> np.ndarray:
        ids = np.asarray(raw_ids)
        if ids.ndim != 1 or len(ids) != scan.num_points:
            raise ValueError("each channel_ids array must have one entry per point")
        if not np.issubdtype(ids.dtype, np.integer):
            if not np.all(np.equal(ids, np.round(ids))):
                raise ValueError("channel_ids must be integers")
            ids = np.round(ids).astype(np.int64)
        else:
            ids = ids.astype(np.int64, copy=False)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("channel_ids must be unique within a profile")
        return ids


@dataclass
class Tan2025GroundTruth:
    """Ground truth kept on the simulation/evaluation side of the API."""

    T_ef_s: np.ndarray
    plane_normal_base: np.ndarray
    plane_offset_mm: float
    true_translation_poses: Sequence[np.ndarray] = field(default_factory=tuple)
    true_composite_poses: Sequence[np.ndarray] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.T_ef_s = _copy_transform(self.T_ef_s, "T_ef_s")
        normal = np.asarray(self.plane_normal_base, dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            raise ValueError("plane_normal_base must be nonzero and finite")
        self.plane_normal_base = normal / norm
        self.plane_offset_mm = float(self.plane_offset_mm)
        if not np.isfinite(self.plane_offset_mm):
            raise ValueError("plane_offset_mm must be finite")
        self.true_translation_poses = [
            _copy_transform(value, "true_translation_pose")
            for value in self.true_translation_poses
        ]
        self.true_composite_poses = [
            _copy_transform(value, "true_composite_pose")
            for value in self.true_composite_poses
        ]
