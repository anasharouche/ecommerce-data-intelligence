"""Synthetic tests for the frozen Phase 5.5 final-holdout protocol."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.backtesting import BacktestError
from src.ml.contract import FEATURE_SCHEMA_VERSION, STRICT_FEATURE_COLUMNS
from src.ml.final_holdout import (
    CANDIDATE_DIR,
    CANDIDATE_VERSION,
    _aggregate_monthly_top10,
    _monthly_top10,
    build_evaluation_protocol,
    deterministic_topk_mask,
    evaluate_holdout_once,
    run_preflight,
    safe_probability_metrics,
    score_frozen_candidate,
    validate_strict_dtypes,
    verify_frozen_protocol,
)


class _NoFitPredictor:
    """Minimal predictor whose training methods must never be called."""

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit is forbidden")

    def fit_transform(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit_transform is forbidden")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.linspace(0.1, 0.9, len(features))
        return np.column_stack((1.0 - positive, positive))


def _preflight_stub() -> dict[str, object]:
    return {
        "candidate_pipeline_sha256": "a" * 64,
        "candidate_version": CANDIDATE_VERSION,
        "feature_contract": FEATURE_SCHEMA_VERSION,
        "feature_order": list(STRICT_FEATURE_COLUMNS),
    }


def _write_protocol(root: Path) -> None:
    protocol = build_evaluation_protocol(_preflight_stub())
    payload = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    (root / "evaluation_protocol.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "evaluation_protocol.sha256").write_text(
        f"{digest}  evaluation_protocol.json\n", encoding="utf-8"
    )


class FinalHoldoutTests(unittest.TestCase):
    def test_actual_candidate_preflight_is_reproducible_and_contractual(self) -> None:
        report = run_preflight(CANDIDATE_DIR)
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["feature_contract"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(report["feature_order"], list(STRICT_FEATURE_COLUMNS))
        self.assertEqual(report["classes"], [0, 1])
        self.assertTrue(report["checks"]["reload_reproducible"])
        self.assertTrue(
            report["checks"]["missing_values_supported_by_frozen_pipeline"]
        )

    def test_scoring_does_not_call_training_methods_or_change_state(self) -> None:
        predictor = _NoFitPredictor()
        frame = pd.DataFrame({"feature": [1, 2, 3]})
        probability = score_frozen_candidate(predictor, frame)
        np.testing.assert_allclose(probability, [0.1, 0.5, 0.9])

    def test_top10_uses_ceil_and_stable_order_id_ties(self) -> None:
        scores = np.full(11, 0.5)
        order_ids = np.array([f"id_{value:02d}" for value in range(10, -1, -1)])
        first = deterministic_topk_mask(scores, order_ids)
        self.assertEqual(int(first.sum()), 2)
        self.assertEqual(set(order_ids[first]), {"id_00", "id_01"})
        permutation = np.array([7, 0, 10, 3, 8, 1, 6, 4, 2, 9, 5])
        second = deterministic_topk_mask(scores[permutation], order_ids[permutation])
        self.assertEqual(set(order_ids[permutation][second]), {"id_00", "id_01"})

    def test_monthly_top10_aggregation_concatenates_exact_budgets(self) -> None:
        rows: list[dict[str, object]] = []
        scores: list[float] = []
        for month in range(5, 9):
            for row in range(11):
                rows.append(
                    {
                        "order_id": f"2018-{month:02d}-{row:02d}",
                        "purchase_timestamp": pd.Timestamp(2018, month, 1),
                        "is_delayed": int(row < 2),
                    }
                )
                scores.append(float(11 - row))
        frame = pd.DataFrame(rows)
        monthly, alerts = _monthly_top10(frame, np.asarray(scores))
        aggregate = _aggregate_monthly_top10(
            frame["is_delayed"].to_numpy(dtype=np.int8), alerts, monthly
        )
        self.assertEqual(int(alerts.sum()), 8)
        self.assertEqual(aggregate["monthly_batch_totals"]["detected_delays"], 8)
        self.assertEqual(aggregate["monthly_batch_totals"]["precision_at_10_percent"], 1.0)

    def test_metrics_handle_single_class_without_invented_ranking_values(self) -> None:
        metrics = safe_probability_metrics(
            np.zeros(4, dtype=np.int8), np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertIsNone(metrics["average_precision"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertFalse(metrics["ranking_metrics_defined"])
        self.assertGreater(metrics["brier_score"], 0.0)

    def test_monthly_top10_with_no_positive_keeps_the_frozen_budget(self) -> None:
        frame = pd.DataFrame(
            {
                "order_id": [f"id_{row:02d}" for row in range(10)],
                "purchase_timestamp": [pd.Timestamp("2018-05-01")] * 10,
                "is_delayed": np.zeros(10, dtype=np.int8),
            }
        )
        monthly, alerts = _monthly_top10(frame, np.linspace(0.0, 0.9, 10))
        aggregate = _aggregate_monthly_top10(
            frame["is_delayed"].to_numpy(), alerts, monthly
        )
        self.assertEqual(int(alerts.sum()), 1)
        self.assertEqual(aggregate["monthly_batch_totals"]["recall_at_10_percent"], 0.0)
        self.assertEqual(aggregate["monthly_batch_totals"]["lift_at_10_percent"], 0.0)

    def test_strict_dtype_contract_rejects_coercion(self) -> None:
        expected = {
            "order_id": pd.Series(["a"], dtype="string"),
            "purchase_timestamp": pd.Series(
                [pd.Timestamp("2018-05-01")], dtype="datetime64[us]"
            ),
            "label_observed_at": pd.Series(
                [pd.Timestamp("2018-05-02")], dtype="datetime64[us]"
            ),
            "purchase_hour": pd.Series([1], dtype="int8"),
            "purchase_day_of_week": pd.Series([1], dtype="int8"),
            "purchase_month": pd.Series([5], dtype="int8"),
            "purchase_quarter": pd.Series([2], dtype="int8"),
            "purchase_is_weekend": pd.Series([False], dtype="bool"),
            "customer_state": pd.Series(["SP"], dtype="string"),
            "customer_latitude": pd.Series([-23.5], dtype="Float64"),
            "customer_longitude": pd.Series([-46.6], dtype="Float64"),
            "customer_has_gps": pd.Series([True], dtype="bool"),
            "is_delayed": pd.Series([0], dtype="int8"),
        }
        frame = pd.DataFrame(expected)
        self.assertEqual(validate_strict_dtypes(frame)["purchase_hour"], "int8")
        frame["purchase_hour"] = frame["purchase_hour"].astype("int64")
        with self.assertRaisesRegex(BacktestError, "Types strict_v1"):
            validate_strict_dtypes(frame)

    def test_protocol_hash_detects_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_protocol(root)
            expected = verify_frozen_protocol(root)
            self.assertEqual(len(expected), 64)
            with (root / "evaluation_protocol.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(BacktestError, "modifi"):
                verify_frozen_protocol(root)

    def test_second_evaluation_is_refused_before_any_holdout_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_protocol(root)
            (root / "preflight_report.json").write_text(
                json.dumps({"all_checks_passed": True}), encoding="utf-8"
            )
            (root / "evaluation_opened.json").write_text("{}", encoding="utf-8")
            with patch("pandas.read_parquet") as read:
                with self.assertRaisesRegex(BacktestError, "holdout Phase 5.5"):
                    evaluate_holdout_once(
                        input_dir=root,
                        candidate_dir=root / "missing",
                        output_dir=root,
                    )
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
