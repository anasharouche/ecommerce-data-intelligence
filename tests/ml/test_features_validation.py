"""Tests for strict features and leakage validation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.ml.contract import STRICT_FEATURE_COLUMNS
from src.ml.dataset import build_supervised_population
from src.ml.features import build_strict_features, model_matrix
from src.ml.validation import ValidationError, validate_strict_dataset
from tests.ml.helpers import synthetic_order_source


class FeatureTests(unittest.TestCase):
    """Validate deterministic feature construction."""

    def setUp(self) -> None:
        population = build_supervised_population(synthetic_order_source())
        self.dataset = build_strict_features(population.supervised_source)

    def test_features_are_deterministic_and_leak_free(self) -> None:
        population = build_supervised_population(synthetic_order_source())
        second = build_strict_features(population.supervised_source)
        pd.testing.assert_frame_equal(self.dataset, second)

        matrix = model_matrix(self.dataset)
        self.assertEqual(tuple(matrix.columns), STRICT_FEATURE_COLUMNS)
        self.assertNotIn("order_id", matrix.columns)
        self.assertNotIn("purchase_timestamp", matrix.columns)
        self.assertNotIn("label_observed_at", matrix.columns)
        self.assertNotIn("is_delayed", matrix.columns)

    def test_calendar_and_missing_gps(self) -> None:
        indexed = self.dataset.set_index("order_id")
        self.assertEqual(int(indexed.loc["on_time", "purchase_hour"]), 10)
        self.assertEqual(indexed.loc["on_time", "customer_state"], "SP")
        self.assertFalse(bool(indexed.loc["delayed", "customer_has_gps"]))
        self.assertTrue(pd.isna(indexed.loc["delayed", "customer_latitude"]))
        self.assertTrue(pd.isna(indexed.loc["delayed", "customer_longitude"]))

    def test_duplicate_order_is_rejected(self) -> None:
        duplicate = pd.concat([self.dataset, self.dataset.iloc[[0]]])
        with self.assertRaises(ValidationError):
            validate_strict_dataset(duplicate)


if __name__ == "__main__":
    unittest.main()
