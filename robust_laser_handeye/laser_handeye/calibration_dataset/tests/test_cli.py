from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from laser_handeye.calibration_dataset import load_calibration_dataset


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = PACKAGE_ROOT / "examples" / "generate_calibration_dataset.py"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PACKAGE_ROOT)
        if not existing
        else os.pathsep.join((str(PACKAGE_ROOT), existing))
    )
    return environment


def test_generic_cli_writes_reloadable_circular_collection(tmp_path) -> None:
    output = tmp_path / "collection"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "single-plane-circular",
            "--trials", "1",
            "--seed", "13",
            "--profile-points", "8",
            "--heights-mm", "80",
            "--theta-deg", "30",
            "--beta-deg", "90",
            "--reference-line-ids",
            "--output-dir", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    collection = json.loads((output / "collection.json").read_text())
    assert collection["status"] == "complete"
    assert collection["acquisition_mode"] == "single_plane_circular"
    trial = output / collection["trials"][0]["relative_path"]
    assert len(load_calibration_dataset(trial).scans) == 9


def test_generator_help_exposes_supported_modes() -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "single-plane-circular" in completed.stdout
    assert "three-plane" in completed.stdout
