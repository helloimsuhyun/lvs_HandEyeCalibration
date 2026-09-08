from __future__ import annotations

import json

import numpy as np
import pytest

from laser_handeye.calibration_dataset import (
    AcquisitionGroup,
    CalibrationDataset,
    CalibrationTruth,
    PlaneTruth,
    load_calibration_dataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.data import LaserScan


def _transform(x: float) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[0, 3] = x
    return value


def _scan(
    scan_id: int,
    points: np.ndarray,
    channel_ids: np.ndarray,
    *,
    plane_id: int = 0,
) -> LaserScan:
    return LaserScan(
        T_base_ef=_transform(float(scan_id)),
        points_s=np.asarray(points, dtype=float),
        plane_id=plane_id,
        scan_id=scan_id,
        meta={
            "channel_ids": np.asarray(channel_ids, dtype=np.int64),
            "pattern": {"line_id": scan_id},
        },
    )


def _dataset(*, channel_id_semantics: str = "stable_sensor_channel"):
    scans = [
        _scan(
            10,
            np.array([[0.0, 0.0, 200.0], [np.nan, np.nan, np.nan], [2.0, 0.0, 201.0]]),
            np.array([100, 101, 102]),
        ),
        _scan(
            20,
            np.array([[-1.0, 0.0, 210.0], [1.0, 0.0, 211.0]]),
            np.array([100, 102]),
        ),
        _scan(
            30,
            np.array([[-2.0, 0.0, 220.0], [2.0, 0.0, 222.0]]),
            np.array([10, 11]),
        ),
    ]
    groups = [
        AcquisitionGroup(
            group_id="primary_ring",
            acquisition_role="calibration",
            motion_kind="circular",
            plane_id=0,
        ),
        AcquisitionGroup(
            group_id="reference_ring",
            acquisition_role="reference",
            motion_kind="circular_reference",
            plane_id=0,
        ),
        AcquisitionGroup(
            group_id="bootstrap",
            acquisition_role="bootstrap",
            motion_kind="raster",
            plane_id=0,
            include_in_calibration=False,
        ),
    ]
    truth = CalibrationTruth(
        T_ef_s_true=np.eye(4),
        planes=[
            PlaneTruth(
                plane_id=0,
                normal_base=np.array([0.0, 0.0, 1.0]),
                offset_mm=-123.5,
                metadata={"signed": True},
            )
        ],
        T_base_ef_true=np.stack([scan.T_base_ef for scan in scans]),
        metadata={"evaluation_only": True},
    )
    return CalibrationDataset(
        scans=scans,
        scan_group_ids=["primary_ring", "reference_ring", "bootstrap"],
        sequence_indices=[0, 0, 0],
        groups=groups,
        acquisition_mode="single_plane_circular",
        source="simulation",
        profile_state="ideal",
        truth=truth,
        metadata={"seed": np.int64(7)},
        channel_id_semantics=channel_id_semantics,
    )


def test_round_trip_preserves_ragged_profiles_channels_truth_and_adapters(tmp_path):
    dataset = _dataset()
    before_hash = logical_dataset_sha256(dataset)

    manifest_path = save_calibration_dataset(dataset, tmp_path / "trial")
    assert manifest_path.name == "manifest.json"
    assert (manifest_path.parent / "scans.npz").is_file()
    with np.load(manifest_path.parent / "scans.npz", allow_pickle=False) as archive:
        assert archive["profile_offsets"].tolist() == [0, 3, 5, 7]
        assert archive["scan_group_id"].dtype.kind == "U"
        assert all(not archive[name].dtype.hasobject for name in archive.files)

    loaded = load_calibration_dataset(manifest_path.parent)
    assert logical_dataset_sha256(loaded) == before_hash
    assert [scan.num_points for scan in loaded.scans] == [3, 2, 2]
    assert np.array_equal(
        loaded.scans[0].points_s,
        dataset.scans[0].points_s,
        equal_nan=True,
    )
    assert loaded.scans[0].meta["channel_ids"].tolist() == [100, 101, 102]
    assert loaded.scans[1].meta["channel_ids"].tolist() == [100, 102]
    assert loaded.truth is not None
    assert loaded.truth.planes[0].offset_mm == -123.5

    grouped = loaded.to_scans_by_plane()
    assert list(grouped) == [0]
    assert [scan.scan_id for scan in grouped[0]] == [10, 20]
    bootstrap = loaded.to_scans_by_plane(
        groups=["bootstrap"],
        include_excluded=True,
    )
    assert [scan.scan_id for scan in bootstrap[0]] == [30]

def test_file_hash_detects_payload_corruption(tmp_path):
    manifest = save_calibration_dataset(_dataset(), tmp_path / "trial")
    payload = manifest.parent / "scans.npz"
    with payload.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="byte SHA-256 mismatch"):
        load_calibration_dataset(manifest.parent)


def test_logical_hash_detects_manifest_semantic_change(tmp_path):
    manifest_path = save_calibration_dataset(_dataset(), tmp_path / "trial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["groups"][0]["acquisition_role"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="logical SHA-256 mismatch"):
        load_calibration_dataset(manifest_path.parent)


def test_logical_hash_covers_declared_counts(tmp_path):
    manifest_path = save_calibration_dataset(_dataset(), tmp_path / "trial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["scans"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="logical SHA-256 mismatch"):
        load_calibration_dataset(manifest_path.parent)


def test_save_refuses_orphan_payload_without_explicit_overwrite(tmp_path):
    directory = tmp_path / "trial"
    directory.mkdir()
    (directory / "scans.npz").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="dataset target already exists"):
        save_calibration_dataset(_dataset(), directory)


def test_explicit_overwrite_replaces_complete_dataset_directory(tmp_path):
    directory = tmp_path / "trial"
    first = _dataset()
    save_calibration_dataset(first, directory)
    replacement = _dataset()
    replacement.metadata["revision"] = 2
    expected_hash = logical_dataset_sha256(replacement)
    save_calibration_dataset(replacement, directory, overwrite=True)
    assert logical_dataset_sha256(load_calibration_dataset(directory)) == expected_hash
    assert not list(tmp_path.glob(".trial.*.staging"))
    assert not list(tmp_path.glob(".trial.*.backup"))


def test_missing_channel_ids_are_rejected():
    scan = LaserScan(
        T_base_ef=np.eye(4),
        points_s=np.array([[0.0, 0.0, 1.0]]),
    )
    group = AcquisitionGroup(
        group_id="general",
        acquisition_role="calibration",
        motion_kind="general_6dof",
        plane_id=0,
    )
    with pytest.raises(ValueError, match="has no channel_ids"):
        CalibrationDataset(
            scans=[scan],
            scan_group_ids=["general"],
            groups=[group],
            acquisition_mode="general",
            source="real",
            profile_state="measured",
        )


def test_adapters_follow_group_declaration_and_sequence_order_after_round_trip(
    tmp_path,
):
    source = _dataset()
    dataset = CalibrationDataset(
        scans=[source.scans[0], source.scans[1], source.scans[2]],
        scan_group_ids=["primary_ring", "reference_ring", "primary_ring"],
        sequence_indices=[1, 0, 0],
        groups=source.groups,
        acquisition_mode=source.acquisition_mode,
        source=source.source,
        profile_state=source.profile_state,
    )
    assert [scan.scan_id for scan in dataset.to_scans_by_plane()[0]] == [30, 10, 20]
    loaded_path = save_calibration_dataset(dataset, tmp_path / "ordered")
    loaded = load_calibration_dataset(loaded_path.parent)
    assert [group.group_id for group in loaded.groups] == [
        "primary_ring",
        "reference_ring",
        "bootstrap",
    ]
    assert [scan.scan_id for scan in loaded.to_scans_by_plane()[0]] == [30, 10, 20]
