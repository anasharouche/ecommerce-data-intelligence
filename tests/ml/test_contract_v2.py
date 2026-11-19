"""Tests for preserving strict V1 and versioning conditional V2."""

from __future__ import annotations

import unittest

from src.ml.contract import FEATURE_SCHEMA_VERSION, STRICT_FEATURE_COLUMNS
from src.ml.contract_v2 import (
    ENRICHED_FEATURE_COLUMNS,
    ENRICHED_FEATURE_SCHEMA_VERSION,
    enriched_feature_contract_as_dict,
)
from src.ml.validation import FORBIDDEN_FEATURE_COLUMNS


class ContractV2Tests(unittest.TestCase):
    def test_strict_v1_is_unchanged_and_v2_is_versioned(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, "strict_v1.0.0")
        self.assertEqual(
            STRICT_FEATURE_COLUMNS,
            (
                "purchase_hour",
                "purchase_day_of_week",
                "purchase_month",
                "purchase_quarter",
                "purchase_is_weekend",
                "customer_state",
                "customer_latitude",
                "customer_longitude",
                "customer_has_gps",
            ),
        )
        self.assertEqual(ENRICHED_FEATURE_SCHEMA_VERSION, "enriched_v2.0.0")
        self.assertEqual(
            ENRICHED_FEATURE_COLUMNS[: len(STRICT_FEATURE_COLUMNS)],
            STRICT_FEATURE_COLUMNS,
        )

    def test_v2_is_conditional_and_contains_no_forbidden_feature(self) -> None:
        contract = enriched_feature_contract_as_dict()
        self.assertEqual(
            contract["operational_status"],
            "conditional_not_production_validated",
        )
        self.assertTrue(contract["conditional_features"])
        self.assertTrue(
            all(
                item["decision"] == "conditional"
                for item in contract["conditional_features"]
            )
        )
        self.assertFalse(
            set(ENRICHED_FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS)
        )


if __name__ == "__main__":
    unittest.main()
