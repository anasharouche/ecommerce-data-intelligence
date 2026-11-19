"""Tests for target calculation and eligibility rules."""

from __future__ import annotations

import unittest

from src.ml.dataset import build_supervised_population
from tests.ml.helpers import synthetic_order_source


class DatasetPopulationTests(unittest.TestCase):
    """Validate target semantics and exclusion reasons."""

    def test_population_and_calendar_day_target(self) -> None:
        result = build_supervised_population(synthetic_order_source())
        eligible = result.supervised_source.set_index("order_id")

        self.assertEqual(len(eligible), 3)
        self.assertEqual(int(eligible.loc["on_time", "is_delayed"]), 0)
        self.assertEqual(int(eligible.loc["delayed", "is_delayed"]), 1)
        self.assertEqual(int(eligible.loc["test_boundary", "is_delayed"]), 0)

    def test_unknown_and_invalid_targets_are_excluded(self) -> None:
        result = build_supervised_population(synthetic_order_source())
        reasons = result.exclusions.set_index("order_id")["exclusion_reason"]

        self.assertEqual(reasons["canceled"], "status_not_delivered")
        self.assertEqual(
            reasons["missing_delivery"], "missing_delivered_timestamp"
        )
        self.assertEqual(
            reasons["missing_estimate"],
            "missing_estimated_delivery_timestamp",
        )
        self.assertEqual(
            reasons["delivery_before_purchase"],
            "delivery_before_purchase",
        )


if __name__ == "__main__":
    unittest.main()
