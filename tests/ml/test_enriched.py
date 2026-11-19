"""Tests for enriched order aggregation, distance and cartesian protection."""

from __future__ import annotations

import unittest

import pandas as pd

from src.ml.dataset import DatasetError
from src.ml.enriched import aggregate_order_items, haversine_km


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "order_a",
                "order_item_id": 1,
                "seller_id": "seller_1",
                "price": 100.0,
                "freight_value": 10.0,
                "seller_state": "SP",
                "seller_latitude": 0.0,
                "seller_longitude": 0.0,
                "seller_flag_has_gps": True,
                "customer_latitude": 0.0,
                "customer_longitude": 0.0,
                "customer_flag_has_gps": True,
                "product_category": "books",
                "product_weight_g": 1000.0,
                "product_length_cm": 10.0,
                "product_height_cm": 10.0,
                "product_width_cm": 10.0,
            },
            {
                "order_id": "order_a",
                "order_item_id": 2,
                "seller_id": "seller_2",
                "price": 50.0,
                "freight_value": 5.0,
                "seller_state": "SP",
                "seller_latitude": 1.0,
                "seller_longitude": 0.0,
                "seller_flag_has_gps": True,
                "customer_latitude": 0.0,
                "customer_longitude": 0.0,
                "customer_flag_has_gps": True,
                "product_category": "toys",
                "product_weight_g": 500.0,
                "product_length_cm": 5.0,
                "product_height_cm": 5.0,
                "product_width_cm": 5.0,
            },
            {
                "order_id": "order_a",
                "order_item_id": 3,
                "seller_id": "seller_2",
                "price": 25.0,
                "freight_value": 5.0,
                "seller_state": "SP",
                "seller_latitude": None,
                "seller_longitude": None,
                "seller_flag_has_gps": False,
                "customer_latitude": 0.0,
                "customer_longitude": 0.0,
                "customer_flag_has_gps": True,
                "product_category": "books",
                "product_weight_g": None,
                "product_length_cm": None,
                "product_height_cm": None,
                "product_width_cm": None,
            },
        ]
    )


class EnrichedFeatureTests(unittest.TestCase):
    def test_haversine_units(self) -> None:
        distance = haversine_km([0.0], [0.0], [1.0], [0.0])[0]
        self.assertAlmostEqual(distance, 111.195, places=2)

    def test_multiseller_and_missing_coordinate_aggregation(self) -> None:
        aggregate = aggregate_order_items(_items()).set_index("order_id").loc["order_a"]
        self.assertEqual(int(aggregate["item_count"]), 3)
        self.assertEqual(int(aggregate["seller_count"]), 2)
        self.assertEqual(int(aggregate["seller_state_count"]), 1)
        self.assertTrue(bool(aggregate["is_multi_seller"]))
        self.assertAlmostEqual(float(aggregate["distance_coverage_ratio"]), 2 / 3)
        self.assertTrue(bool(aggregate["distance_incomplete"]))
        self.assertEqual(aggregate["dominant_category"], "books")
        self.assertAlmostEqual(float(aggregate["total_product_weight_g"]), 1500.0)
        self.assertAlmostEqual(float(aggregate["total_product_volume_cm3"]), 1125.0)
        self.assertAlmostEqual(float(aggregate["freight_ratio"]), 20.0 / 175.0)

    def test_duplicate_item_rejects_cartesian_product(self) -> None:
        duplicated = pd.concat([_items(), _items().iloc[[0]]], ignore_index=True)
        with self.assertRaises(DatasetError):
            aggregate_order_items(duplicated)


if __name__ == "__main__":
    unittest.main()
