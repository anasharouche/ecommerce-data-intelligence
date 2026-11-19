"""Chronological dataset splits and label-availability filters."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    """Raised when temporal split constraints are violated."""


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Exclusive upper bounds for chronological partitions."""

    train_end: pd.Timestamp = pd.Timestamp("2018-01-01")
    validation_end: pd.Timestamp = pd.Timestamp("2018-05-01")
    test_end: pd.Timestamp = pd.Timestamp("2018-09-01")

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end < self.test_end:
            raise SplitError("Les frontières temporelles sont incohérentes.")


def split_temporally(
    dataset: pd.DataFrame,
    config: TemporalSplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a complete supervised dataset by purchase time."""

    config = config or TemporalSplitConfig()
    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    if purchase.isna().any():
        raise SplitError("purchase_timestamp contient une valeur invalide.")

    outside = purchase.ge(config.test_end)
    if outside.any():
        raise SplitError(
            f"{int(outside.sum())} ligne(s) sont postérieures à test_end."
        )

    masks = {
        "train": purchase.lt(config.train_end),
        "validation": purchase.ge(config.train_end)
        & purchase.lt(config.validation_end),
        "test": purchase.ge(config.validation_end)
        & purchase.lt(config.test_end),
    }
    splits = {
        name: dataset.loc[mask].sort_values(
            ["purchase_timestamp", "order_id"]
        ).reset_index(drop=True)
        for name, mask in masks.items()
    }
    _validate_split_coverage(dataset, splits)
    return splits


def _validate_split_coverage(
    dataset: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Assert full coverage and pairwise-disjoint order identifiers."""

    all_ids: list[str] = []
    for frame in splits.values():
        all_ids.extend(frame["order_id"].astype(str).tolist())

    if len(all_ids) != len(dataset):
        raise SplitError("Les partitions ne couvrent pas toutes les lignes.")
    if len(all_ids) != len(set(all_ids)):
        raise SplitError("Chevauchement d'order_id entre les partitions.")
    if set(all_ids) != set(dataset["order_id"].astype(str)):
        raise SplitError("Les partitions ne couvrent pas tous les order_id.")


def labels_available_as_of(
    dataset: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return rows whose label was observable strictly before a cutoff."""

    cutoff = pd.Timestamp(as_of_date)
    if pd.isna(cutoff):
        raise SplitError("as_of_date est invalide.")
    if cutoff.tzinfo is not None:
        raise SplitError(
            "as_of_date doit être sans fuseau, comme les timestamps source."
        )

    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    if observed.isna().any() or purchase.isna().any():
        raise SplitError("Timestamp invalide pour le filtre as_of_date.")

    available = dataset.loc[observed.lt(cutoff)].copy()
    if not available["purchase_timestamp"].lt(cutoff).all():
        raise SplitError("Un label est disponible avant la création associée.")

    return available.sort_values(
        ["purchase_timestamp", "order_id"]
    ).reset_index(drop=True)
