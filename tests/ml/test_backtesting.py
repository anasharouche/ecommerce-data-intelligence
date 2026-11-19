"""Tests for pre-test access guards, temporal folds and calibration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.ml.backtesting import (
    BacktestError,
    assert_calibration_ranking_preserved,
    assert_phase54_safe_path,
    build_fold_partitions,
    build_phase54_pipeline,
    evaluate_calibration_variants,
    fit_calibration_variants,
    monthly_backtest_folds,
)
from src.ml.contract import STRICT_FEATURE_COLUMNS
from src.ml.drift import numeric_population_stability_index
from src.ml.training import feature_matrix, target_vector


def _temporal_dataset() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", "2018-01-31", freq="2D")
    rows = len(dates)
    index = np.arange(rows)
    delivered = dates + pd.to_timedelta(5 + index % 9, unit="D")
    frame = pd.DataFrame(
        {
            "order_id": [f"order_{value:04d}" for value in index],
            "purchase_timestamp": dates,
            "label_observed_at": delivered,
            "purchase_hour": (index % 24).astype("int8"),
            "purchase_day_of_week": dates.day_of_week.astype("int8"),
            "purchase_month": dates.month.astype("int8"),
            "purchase_quarter": dates.quarter.astype("int8"),
            "purchase_is_weekend": dates.day_of_week >= 5,
            "customer_state": np.where(index % 2, "SP", "RJ"),
            "customer_latitude": -23.0 + (index % 4) * 0.1,
            "customer_longitude": -46.0 + (index % 5) * 0.1,
            "customer_has_gps": True,
            "is_delayed": ((index % 7 == 0) | (index % 11 == 0)).astype("int8"),
        }
    )
    return frame


class BacktestingTests(unittest.TestCase):
    def test_calibration_ranking_guard_and_positive_class(self) -> None:
        raw = np.array([0.1, 0.2, 0.2, 0.8])
        assert_calibration_ranking_preserved(
            raw,
            np.array([0.05, 0.15, 0.15, 0.9]),
            method="synthetic_monotone",
        )
        with self.assertRaisesRegex(BacktestError, "inverse le classement"):
            assert_calibration_ranking_preserved(
                raw,
                np.array([0.9, 0.8, 0.8, 0.1]),
                method="synthetic_reversed",
            )

    def test_final_test_path_is_rejected_before_read(self) -> None:
        with patch("pandas.read_parquet") as read:
            with self.assertRaises(BacktestError):
                assert_phase54_safe_path(Path("strict_v1_test.parquet"))
            read.assert_not_called()

    def test_folds_are_reproducible_disjoint_and_mature(self) -> None:
        fold = monthly_backtest_folds()[0]
        first = build_fold_partitions(_temporal_dataset(), fold)
        second = build_fold_partitions(_temporal_dataset(), fold)
        self.assertEqual(first.train["order_id"].tolist(), second.train["order_id"].tolist())
        self.assertFalse(set(first.train.order_id) & set(first.calibration.order_id))
        self.assertFalse(set(first.calibration.order_id) & set(first.evaluation.order_id))
        self.assertTrue(first.train.label_observed_at.lt(fold.training_as_of).all())
        self.assertTrue(
            first.calibration.label_observed_at.lt(fold.evaluation_start).all()
        )
        self.assertLess(fold.calibration_as_of, fold.evaluation_end)

    def test_calibration_threshold_and_trusted_round_trip(self) -> None:
        partitions = build_fold_partitions(
            _temporal_dataset(), monthly_backtest_folds()[0]
        )
        x_train = feature_matrix(partitions.train)
        x_calibration = feature_matrix(partitions.calibration)
        x_evaluation = feature_matrix(partitions.evaluation)
        variants = fit_calibration_variants(
            build_phase54_pipeline("strict_v1", "logistic_regression"),
            x_train=x_train,
            y_train=target_vector(partitions.train),
            x_calibration=x_calibration,
            y_calibration=target_vector(partitions.calibration),
        )
        np.testing.assert_array_equal(variants["sigmoid"].classes_, [0, 1])
        first = evaluate_calibration_variants(
            variants,
            x_calibration=x_calibration,
            y_calibration=target_vector(partitions.calibration),
            x_evaluation=x_evaluation,
            y_evaluation=target_vector(partitions.evaluation),
            historical_threshold=0.45956620240681,
        )
        second = evaluate_calibration_variants(
            variants,
            x_calibration=x_calibration,
            y_calibration=target_vector(partitions.calibration),
            x_evaluation=x_evaluation,
            y_evaluation=target_vector(partitions.evaluation),
            historical_threshold=0.45956620240681,
        )
        self.assertTrue(
            first["sigmoid"]["threshold_selected_before_evaluation"]
        )
        self.assertEqual(
            first["sigmoid"]["candidate_threshold"],
            second["sigmoid"]["candidate_threshold"],
        )
        np.testing.assert_allclose(
            first["sigmoid"]["evaluation_probabilities"],
            second["sigmoid"]["evaluation_probabilities"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trusted_calibrated.joblib"
            joblib.dump(variants["sigmoid"], path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(
                variants["sigmoid"].predict_proba(x_evaluation)[:, 1],
                loaded.predict_proba(x_evaluation)[:, 1],
            )

    def test_numeric_drift_is_reproducible(self) -> None:
        reference = pd.Series(np.arange(100, dtype=float))
        current = pd.Series(np.arange(100, dtype=float) + 20)
        first = numeric_population_stability_index(reference, current)
        second = numeric_population_stability_index(reference, current)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)
        self.assertEqual(len(STRICT_FEATURE_COLUMNS), 9)


if __name__ == "__main__":
    unittest.main()
