from __future__ import annotations

from copy import deepcopy

import pytest

from main.run_nonlinear_refinement_ablation import (
    ARMS,
    _paired_metric,
    _validate_condition_pairing,
)


def _paired_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        rows.append(
            {
                "arm": arm,
                "trial_index": 0,
                "pair_trial_index": 0,
                "relative_path": "trials/trial_000000",
                "logical_dataset_sha256": "dataset",
                "noise_seed_key": "1:0:0",
                "initialization_seed_key": "1:0:1",
                "initial_estimate_sha256": "initial",
                "alternating_estimate_sha256": "alternating",
                "error_type": "",
                "optimizer_success": True,
                "outlier": False,
                "translation_error_mm": 1.0,
                "rotation_error_deg": 0.1,
            }
        )
    return rows


def test_pairing_audit_rejects_missing_and_mismatched_arms() -> None:
    rows = _paired_rows()
    assert _validate_condition_pairing(rows)["validated"]

    with pytest.raises(ValueError, match="exactly one row"):
        _validate_condition_pairing(rows[:-1])

    mismatched = deepcopy(rows)
    mismatched[0]["noise_seed_key"] = "different"
    with pytest.raises(ValueError, match="mismatched noise_seed_key"):
        _validate_condition_pairing(mismatched)


def test_paired_metric_retains_failure_denominators() -> None:
    reference = [
        {
            "error_type": "",
            "optimizer_success": True,
            "translation_error_mm": 2.0,
        },
        {
            "error_type": "",
            "optimizer_success": True,
            "translation_error_mm": 1.0,
        },
    ]
    candidate = [
        {
            "error_type": "",
            "optimizer_success": True,
            "translation_error_mm": 1.0,
        },
        {
            "error_type": "RuntimeError",
            "optimizer_success": False,
            "translation_error_mm": float("nan"),
        },
    ]

    summary = _paired_metric(
        reference,
        candidate,
        field="translation_error_mm",
        tie_tolerance=1e-4,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert summary["planned_pairs"] == 2
    assert summary["complete_case_pairs"] == 1
    assert summary["candidate_exception_only"] == 1
    assert summary["paired_trials"] == 1
    assert summary["win_fraction"] == 1.0
    lower, upper = summary["win_fraction_wilson_95_ci"]
    assert 0.0 <= lower <= upper <= 1.0
