"""Tests for the versioned monotone calibration policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.backtesting import monthly_backtest_folds
from src.ml.calibration_policy import (
    build_policy_fold_partitions,
    fit_monotone_calibration_policy,
    ranking_diagnostics,
)
from src.ml.correct_calibration import _select_experiment
from tests.ml.test_backtesting import _temporal_dataset


def _linearly_separable() -> tuple[pd.DataFrame, pd.Series]:
    values = np.linspace(-3.0, 3.0, 80)
    return pd.DataFrame({"score": values}), pd.Series((values > 0).astype("int8"))


class CalibrationPolicyTests(unittest.TestCase):
    def test_candidate_is_rejected_when_temporal_support_is_insufficient(self) -> None:
        comparison = [
            {
                "experiment": "synthetic",
                "contract": "strict_v1",
                "model": "logistic_regression",
                "mean_raw_average_precision": 0.2,
                "mean_raw_roc_auc": 0.65,
                "minimum_raw_roc_auc": 0.6,
                "minimum_alert_rate": 0.05,
                "maximum_alert_rate": 0.8,
                "all_evaluation_rankings_preserved": True,
                "partition_support_sufficient": False,
            }
        ]
        selection = _select_experiment(comparison)
        self.assertFalse(selection["admissible"])
        self.assertIn("moins de 30 retards", selection["reason"])

    def test_increasing_sigmoid_is_selected_and_reproducible(self) -> None:
        x_train, y_train = _linearly_separable()
        x_calibration = pd.DataFrame({"score": np.linspace(-2.5, 2.5, 40)})
        y_calibration = pd.Series(
            (x_calibration["score"].to_numpy() > 0).astype("int8")
        )
        first = fit_monotone_calibration_policy(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration_fit=x_calibration,
            y_calibration_fit=y_calibration,
        )
        second = fit_monotone_calibration_policy(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration_fit=x_calibration,
            y_calibration_fit=y_calibration,
        )
        self.assertEqual(first.selected_method, "sigmoid")
        self.assertTrue(first.diagnostics["attempts"]["sigmoid"]["increasing"])
        self.assertEqual(first.estimator.classes_.tolist(), [0, 1])
        np.testing.assert_allclose(
            first.estimator.predict_proba(x_calibration),
            second.estimator.predict_proba(x_calibration),
        )

    def test_decreasing_sigmoid_is_rejected_with_explicit_raw_fallback(self) -> None:
        x_train, y_train = _linearly_separable()
        x_calibration = pd.DataFrame({"score": np.linspace(-2.5, 2.5, 40)})
        y_calibration = pd.Series(
            (x_calibration["score"].to_numpy() < 0).astype("int8")
        )
        result = fit_monotone_calibration_policy(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration_fit=x_calibration,
            y_calibration_fit=y_calibration,
        )
        self.assertFalse(result.diagnostics["attempts"]["sigmoid"]["admissible"])
        self.assertEqual(result.selected_method, "raw")
        self.assertEqual(result.calibration_status, "uncalibrated_raw_fallback")

    def test_isotonic_is_monotone_and_ties_are_reported(self) -> None:
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        isotonic = np.array([0.05, 0.2, 0.2, 0.7, 0.7])
        diagnostic = ranking_diagnostics(raw, isotonic)
        self.assertTrue(diagnostic["rank_preserved"])
        self.assertEqual(diagnostic["calibrated_unique_scores"], 3)
        self.assertEqual(diagnostic["calibrated_tied_rows"], 2)
        self.assertGreater(diagnostic["largest_calibrated_tie_fraction"], 0)

    def test_temporal_subpartitions_are_disjoint_and_labels_are_mature(self) -> None:
        partitions = build_policy_fold_partitions(
            _temporal_dataset(), monthly_backtest_folds()[0]
        )
        frames = (
            partitions.train,
            partitions.calibration_fit,
            partitions.threshold_selection,
            partitions.evaluation,
        )
        identifiers = [set(frame["order_id"]) for frame in frames]
        for index, left in enumerate(identifiers):
            for right in identifiers[index + 1 :]:
                self.assertFalse(left & right)
        self.assertTrue(
            partitions.calibration_fit["label_observed_at"]
            .lt(partitions.calibration_split_at)
            .all()
        )
        self.assertTrue(
            partitions.threshold_selection["label_observed_at"]
            .lt(partitions.fold.evaluation_start)
            .all()
        )

    def test_policy_artifact_round_trip(self) -> None:
        x_train, y_train = _linearly_separable()
        result = fit_monotone_calibration_policy(
            LogisticRegression(random_state=42),
            x_train=x_train,
            y_train=y_train,
            x_calibration_fit=x_train,
            y_calibration_fit=y_train,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.joblib"
            joblib.dump(result.estimator, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(
                result.estimator.predict_proba(x_train),
                loaded.predict_proba(x_train),
            )


if __name__ == "__main__":
    unittest.main()
