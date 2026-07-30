from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from laser_handeye.calibration_dataset import load_calibration_dataset


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = PACKAGE_ROOT / "examples" / "generate_calibration_dataset.py"
LEGACY_GENERATOR = PACKAGE_ROOT / "examples" / "generate_tan2025_dataset.py"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PACKAGE_ROOT)
        if not existing
        else os.pathsep.join((str(PACKAGE_ROOT), existing))
    )
    return environment


def test_generic_cli_writes_reloadable_incremental_collection(tmp_path) -> None:
    output = tmp_path / "collection"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "translation-composite",
            "--trials",
            "1",
            "--seed",
            "13",
            "--translation-poses",
            "4",
            "--composite-poses",
            "6",
            "--profile-points",
            "16",
            "--plane-mode",
            "random",
            "--plane-angle-range-deg",
            "-20",
            "20",
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    collection = json.loads(
        (output / "collection.json").read_text(encoding="utf-8")
    )
    assert collection["status"] == "complete"
    assert collection["acquisition_mode"] == "translation_composite"
    assert collection["requested_trials"] == 1
    assert collection["completed_trials"] == 1
    trial = output / collection["trials"][0]["relative_path"]
    dataset = load_calibration_dataset(trial)
    assert len(dataset.scans) == 10
    assert collection["config"]["plane_mode"] == "random"
    assert (
        collection["config"]["simulation"]["plane_source"]
        == "random_plane_config_per_trial"
    )
    assert "plane_center_base_mm" not in collection["config"]["simulation"]
    assert dataset.truth.metadata["plane"]["plane_mode"] == "random"


def test_legacy_generator_name_exposes_all_generic_modes() -> None:
    completed = subprocess.run(
        [sys.executable, str(LEGACY_GENERATOR), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "translation-composite" in completed.stdout
    assert "single-plane-circular" in completed.stdout
    assert "three-plane" in completed.stdout
