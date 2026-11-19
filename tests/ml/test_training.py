"""Unit tests for sklearn preprocessing, models, and trusted artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.contract import STRICT_FEATURE_COLUMNS
from src.ml.training import (
    GPS_FEATURES,
    build_model_pipeline,
    feature_matrix,
    model_definitions,
)


def _synthetic_features(rows: int = 180) -> tuple[pd.DataFrame, pd.Series]:
    index = np.arange(rows)
    state = np.where(index % 3 == 0, "SP", np.where(index % 3 == 1, "RJ", "MG"))
    latitude = -23.0 + (index % 7) * 0.1
    longitude = -46.0 + (index % 5) * 0.1
    latitude = latitude.astype(float)
    longitude = longitude.astype(float)
    latitude[index % 17 == 0] = np.nan
    longitude[index % 17 == 0] = np.nan
    frame = pd.DataFrame(
        {
            "purchase_hour": (index % 24).astype("int8"),
            "purchase_day_of_week": (index % 7).astype("int8"),
            "purchase_month": ((index % 12) + 1).astype("int8"),
            "purchase_quarter": ((index % 4) + 1).astype("int8"),
            "purchase_is_weekend": (index % 7 >= 5),
            "customer_state": pd.Series(state, dtype="string"),
            "customer_latitude": pd.Series(latitude, dtype="Float64"),
            "customer_longitude": pd.Series(longitude, dtype="Float64"),
            "customer_has_gps": ~(index % 17 == 0),
        }
    )
    target = pd.Series(
        ((index % 11 == 0) | ((state == "RJ") & (index % 8 == 0))).astype("int8")
    )
    return frame, target


class TrainingPipelineTests(unittest.TestCase):
    def test_all_models_train_and_produce_probabilities(self) -> None:
        features, target = _synthetic_features()
        train_x, validation_x = features.iloc[:150], features.iloc[150:].copy()
        train_y = target.iloc[:150]
        validation_x.loc[:, "customer_state"] = "ZZ"

        for definition in model_definitions().values():
            with self.subTest(model=definition.name):
                pipeline = build_model_pipeline(definition)
                pipeline.fit(train_x, train_y)
                probability = pipeline.predict_proba(validation_x)[:, 1]
                self.assertEqual(len(probability), len(validation_x))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_unknown_category_is_not_fitted_from_validation(self) -> None:
        features, target = _synthetic_features()
        definition = model_definitions()["logistic_regression"]
        pipeline = build_model_pipeline(definition)
        pipeline.fit(features.iloc[:150], target.iloc[:150])

        encoder = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .named_steps["one_hot"]
        )
        categories = set(encoder.categories_[0])
        self.assertNotIn("ZZ", categories)

        unknown = features.iloc[[151]].copy()
        unknown.loc[:, "customer_state"] = "ZZ"
        probability = pipeline.predict_proba(unknown)[:, 1]
        self.assertEqual(len(probability), 1)

    def test_ablation_pipeline_does_not_mutate_full_pipeline(self) -> None:
        features, target = _synthetic_features()
        definition = model_definitions()["logistic_regression"]
        full_pipeline = build_model_pipeline(definition).fit(features, target)
        expected = full_pipeline.predict_proba(features)[:, 1]

        ablation_features = tuple(
            name for name in STRICT_FEATURE_COLUMNS if name not in GPS_FEATURES
        )
        build_model_pipeline(
            definition, feature_columns=ablation_features
        ).fit(features.loc[:, list(ablation_features)], target)

        actual = full_pipeline.predict_proba(features)[:, 1]
        np.testing.assert_allclose(expected, actual)

    def test_random_forest_is_reproducible(self) -> None:
        features, target = _synthetic_features()
        definition_one = model_definitions(42)["random_forest"]
        definition_two = model_definitions(42)["random_forest"]
        first = build_model_pipeline(definition_one).fit(features, target)
        second = build_model_pipeline(definition_two).fit(features, target)
        np.testing.assert_allclose(
            first.predict_proba(features)[:, 1],
            second.predict_proba(features)[:, 1],
        )

    def test_trusted_joblib_round_trip(self) -> None:
        features, target = _synthetic_features()
        pipeline = build_model_pipeline(
            model_definitions()["logistic_regression"]
        ).fit(features, target)
        expected = pipeline.predict_proba(features.iloc[:10])[:, 1]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trusted_test_pipeline.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
            actual = loaded.predict_proba(features.iloc[:10])[:, 1]
        np.testing.assert_allclose(expected, actual)

    def test_feature_matrix_order_matches_contract(self) -> None:
        features, _ = _synthetic_features()
        matrix = feature_matrix(features)
        self.assertEqual(tuple(matrix.columns), STRICT_FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
