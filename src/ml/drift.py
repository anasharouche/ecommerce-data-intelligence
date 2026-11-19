"""Descriptive feature and score drift indicators for temporal backtests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _psi(reference_rates: np.ndarray, current_rates: np.ndarray) -> float:
    epsilon = 1e-6
    reference = np.clip(reference_rates.astype(float), epsilon, None)
    current = np.clip(current_rates.astype(float), epsilon, None)
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def numeric_population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI with train-derived quantile edges and explicit missing bin."""

    ref = pd.to_numeric(pd.Series(reference), errors="coerce")
    cur = pd.to_numeric(pd.Series(current), errors="coerce")
    non_null = ref.dropna().to_numpy(dtype=float)
    if non_null.size == 0:
        return 0.0 if cur.isna().all() else float("inf")
    quantiles = np.unique(
        np.quantile(non_null, np.linspace(0.0, 1.0, bins + 1))
    )
    if quantiles.size <= 1:
        ref_categories = np.where(ref.isna(), "<missing>", "<constant>")
        cur_categories = np.where(cur.isna(), "<missing>", "<constant>")
        categories = ["<constant>", "<missing>"]
        return _psi(
            pd.Series(ref_categories).value_counts(normalize=True).reindex(
                categories, fill_value=0
            ).to_numpy(),
            pd.Series(cur_categories).value_counts(normalize=True).reindex(
                categories, fill_value=0
            ).to_numpy(),
        )
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    ref_bins = pd.cut(ref, bins=quantiles, include_lowest=True).astype("string")
    cur_bins = pd.cut(cur, bins=quantiles, include_lowest=True).astype("string")
    ref_bins = ref_bins.fillna("<missing>")
    cur_bins = cur_bins.fillna("<missing>")
    categories = sorted(set(ref_bins) | set(cur_bins))
    return _psi(
        ref_bins.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(),
        cur_bins.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(),
    )


def categorical_population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> float:
    """Compute PSI for categorical values with unseen and missing categories."""

    ref = pd.Series(reference, dtype="string").fillna("<missing>")
    cur = pd.Series(current, dtype="string").fillna("<missing>")
    categories = sorted(set(ref) | set(cur))
    return _psi(
        ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(),
        cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(),
    )


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    boolean_features: tuple[str, ...],
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return PSI and missing-rate changes without claiming causal drift."""

    feature_psi: dict[str, float] = {}
    for feature in numeric_features:
        feature_psi[feature] = numeric_population_stability_index(
            reference[feature], current[feature]
        )
    for feature in categorical_features + boolean_features:
        feature_psi[feature] = categorical_population_stability_index(
            reference[feature], current[feature]
        )
    missing = {
        feature: {
            "reference_rate": float(reference[feature].isna().mean()),
            "current_rate": float(current[feature].isna().mean()),
            "absolute_delta": float(
                abs(current[feature].isna().mean() - reference[feature].isna().mean())
            ),
        }
        for feature in numeric_features + categorical_features + boolean_features
    }
    score_psi = None
    if reference_scores is not None and current_scores is not None:
        score_psi = numeric_population_stability_index(
            reference_scores, current_scores
        )
    return {
        "psi_definition": (
            "Population Stability Index using reference-derived deciles for "
            "numeric values and complete category support for categorical values."
        ),
        "monitoring_hypotheses_not_business_standards": {
            "low": "PSI < 0.10",
            "review": "0.10 <= PSI < 0.25",
            "high": "PSI >= 0.25",
        },
        "feature_psi": feature_psi,
        "maximum_feature_psi": float(max(feature_psi.values())),
        "maximum_feature": max(feature_psi, key=feature_psi.get),
        "missing_rates": missing,
        "score_psi": score_psi,
    }
