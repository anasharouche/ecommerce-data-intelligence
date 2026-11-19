"""Tests for the preregistered Phase 5.4.3 historical campaign."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.backtesting import monthly_backtest_folds
from src.ml.historical_calibration import (
    CAMPAIGN_VERSION,
    build_expanded_historical_partitions,
    deterministic_topk_metrics,
    fit_expanded_calibration_strategies,
)
from src.ml.run_phase543 import run_phase543, validate_artifact_manifest
from tests.ml.test_backtesting import _temporal_dataset


def _linear_data() -> tuple[pd.DataFrame, pd.Series]:
    values = np.linspace(-3.0, 3.0, 100)
    return pd.DataFrame({"score": values}), pd.Series((values > 0).astype("int8"))


class HistoricalCalibrationTests(unittest.TestCase):
    def test_manifest_validation_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "result.json"
            artifact.write_text("{}", encoding="utf-8")
            import hashlib

            manifest = {
                "artifacts": {
                    "result": {
                        "filename": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                }
            }
            manifest_path = root / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_artifact_manifest(manifest_path)
            artifact.write_text("{\"changed\": true}", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "divergente"):
                validate_artifact_manifest(manifest_path)

    def test_expanded_window_is_disjoint_and_mature(self) -> None:
        partitions = build_expanded_historical_partitions(
            _temporal_dataset(), monthly_backtest_folds()[0]
        )
        self.assertEqual(partitions.training_as_of, pd.Timestamp("2017-10-01"))
        self.assertFalse(set(partitions.train.order_id) & set(partitions.calibration.order_id))
        self.assertFalse(
            set(partitions.calibration.order_id) & set(partitions.evaluation.order_id)
        )
        self.assertTrue(
            partitions.train.label_observed_at.lt(partitions.training_as_of).all()
        )
        self.assertTrue(
            partitions.calibration.label_observed_at
            .lt(partitions.fold.evaluation_start)
            .all()
        )

    def test_expanded_sigmoid_is_monotone_and_decreasing_one_is_rejected(self) -> None:
        x_train, y_train = _linear_data()
        accepted = fit_expanded_calibration_strategies(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_train,
            y_calibration=y_train,
        )
        self.assertTrue(accepted.diagnostics["sigmoid_expanded"]["admissible"])
        self.assertTrue(accepted.diagnostics["isotonic_expanded"]["rank_preserved"])
        rejected = fit_expanded_calibration_strategies(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_train,
            y_calibration=1 - y_train,
        )
        self.assertFalse(rejected.diagnostics["sigmoid_expanded"]["admissible"])
        self.assertTrue(rejected.diagnostics["uncalibrated_raw"]["admissible"])

    def test_topk_budget_metrics_and_ties_are_deterministic(self) -> None:
        identifiers = np.array([f"id_{value:02d}" for value in range(10)])
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        scores = np.full(10, 0.5)
        first = deterministic_topk_metrics(y, scores, identifiers, capacity=0.2)
        permutation = np.array([8, 4, 2, 9, 1, 7, 0, 6, 5, 3])
        second = deterministic_topk_metrics(
            y[permutation], scores[permutation], identifiers[permutation], capacity=0.2
        )
        self.assertEqual(first, second)
        self.assertEqual(first["alert_count"], 2)
        self.assertEqual(first["detected_delays"], 2)
        self.assertEqual(first["false_alerts"], 0)
        self.assertEqual(first["missed_delays"], 0)
        self.assertEqual(first["precision_at_k"], 1.0)
        self.assertEqual(first["recall_at_k"], 1.0)
        self.assertEqual(first["lift_at_k"], 5.0)
        self.assertEqual(first["rows_tied_at_boundary"], 10)
        self.assertFalse(first["budget_expanded_for_ties"])

    def test_topk_uses_ceil_and_one_alert_for_small_batch(self) -> None:
        result = deterministic_topk_metrics(
            np.array([0, 1, 0]),
            np.array([0.1, 0.9, 0.2]),
            np.array(["a", "b", "c"]),
            capacity=0.05,
        )
        self.assertEqual(result["alert_count"], 1)
        self.assertAlmostEqual(result["alert_rate"], 1 / 3)

    def test_stop_rule_refuses_a_second_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "experiment_config.json").write_text(
                json.dumps({"campaign_version": CAMPAIGN_VERSION}), encoding="utf-8"
            )
            (output / "existing_result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "réexécution interdite"):
                run_phase543(input_dir=output / "missing", output_dir=output)


if __name__ == "__main__":
    unittest.main()
