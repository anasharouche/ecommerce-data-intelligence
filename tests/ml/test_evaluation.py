"""Unit tests for probability metrics and threshold selection."""

from __future__ import annotations

import unittest

import numpy as np

from src.ml.evaluation import (
    best_f1_threshold,
    calibration_bins,
    evaluate_probabilities,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self) -> None:
        target = np.array([0, 0, 1, 1], dtype=np.int8)
        probability = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = evaluate_probabilities(target, probability, threshold=0.5)

        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
        )

    def test_threshold_and_calibration_are_valid(self) -> None:
        target = np.array([0, 0, 0, 1, 1], dtype=np.int8)
        probability = np.array([0.05, 0.2, 0.6, 0.4, 0.8])
        threshold, metrics = best_f1_threshold(target, probability)

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreater(metrics["f1"], 0.0)
        bins = calibration_bins(target, probability, bins=3)
        self.assertEqual(sum(item["rows"] for item in bins), len(target))


if __name__ == "__main__":
    unittest.main()
