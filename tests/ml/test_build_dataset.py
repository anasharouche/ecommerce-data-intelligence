"""Tests for in-memory builds and safe artifact export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ml.build_dataset import build_artifacts, export_artifacts
from tests.ml.helpers import synthetic_order_source


class BuildPipelineTests(unittest.TestCase):
    """Validate summaries, Parquet exports, and overwrite protection."""

    def test_build_summary_and_export(self) -> None:
        artifacts = build_artifacts(
            synthetic_order_source(), as_of_date="2018-01-10"
        )
        self.assertEqual(artifacts.summary["eligible_rows"], 3)
        self.assertEqual(artifacts.summary["delayed_rows"], 1)
        self.assertEqual(artifacts.summary["strict_feature_count"], 9)
        self.assertEqual(len(artifacts.available_as_of), 1)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            files = export_artifacts(artifacts, output, overwrite=False)
            self.assertTrue(files["dataset"].is_file())
            self.assertEqual(len(pd.read_parquet(files["dataset"])), 3)

            with files["summary"].open(encoding="utf-8") as stream:
                summary = json.load(stream)
            self.assertEqual(summary["eligible_rows"], 3)

            with self.assertRaises(FileExistsError):
                export_artifacts(artifacts, output, overwrite=False)


if __name__ == "__main__":
    unittest.main()
