"""Tests for temporal partitions and label availability."""

from __future__ import annotations

import unittest

from src.ml.dataset import build_supervised_population
from src.ml.features import build_strict_features
from src.ml.splits import labels_available_as_of, split_temporally
from src.ml.validation import validate_split_order
from tests.ml.helpers import synthetic_order_source


class TemporalSplitTests(unittest.TestCase):
    """Validate boundaries, overlap prevention, and as-of filtering."""

    def setUp(self) -> None:
        population = build_supervised_population(synthetic_order_source())
        self.dataset = build_strict_features(population.supervised_source)

    def test_exact_boundaries_and_no_overlap(self) -> None:
        splits = split_temporally(self.dataset)
        validate_split_order(splits)

        self.assertEqual(splits["train"]["order_id"].tolist(), ["on_time"])
        self.assertEqual(splits["validation"]["order_id"].tolist(), ["delayed"])
        self.assertEqual(
            splits["test"]["order_id"].tolist(), ["test_boundary"]
        )
        all_ids = set().union(
            *(set(frame["order_id"]) for frame in splits.values())
        )
        self.assertEqual(all_ids, set(self.dataset["order_id"]))

    def test_as_of_date_uses_label_observation_time(self) -> None:
        available = labels_available_as_of(self.dataset, "2018-01-10")
        self.assertEqual(available["order_id"].tolist(), ["on_time"])
        self.assertTrue(
            available["label_observed_at"].lt("2018-01-10").all()
        )


if __name__ == "__main__":
    unittest.main()
