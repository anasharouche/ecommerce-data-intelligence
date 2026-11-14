
import logging
from pathlib import Path

import pandas as pd

from config.paths import RAW_DATA_DIR, OLIST_FILES


logger = logging.getLogger(__name__)


def extract_csv(file_path: Path) -> pd.DataFrame:
    """
    Lit un fichier CSV et retourne un DataFrame Pandas.

    Args:
        file_path: Chemin du fichier CSV.

    Returns:
        DataFrame contenant les données du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est vide.
    """

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    logger.info("Lecture du fichier : %s", file_path.name)

    try:
        dataframe = pd.read_csv(
            file_path,
            encoding="utf-8",
            low_memory=False
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Le fichier est vide : {file_path.name}"
        ) from exc

    logger.info(
        "%s chargé : %s lignes, %s colonnes",
        file_path.name,
        dataframe.shape[0],
        dataframe.shape[1]
    )

    return dataframe


def extract_all() -> dict[str, pd.DataFrame]:
    """
    Charge les neuf fichiers CSV du dataset Olist.

    Returns:
        Dictionnaire contenant les DataFrames,
        indexés par leurs noms logiques.
    """

    datasets = {}

    for table_name, filename in OLIST_FILES.items():

        file_path = RAW_DATA_DIR / filename

        datasets[table_name] = extract_csv(file_path)

    return datasets


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    datasets = extract_all()

    print("\n=== RÉSUMÉ DE L'EXTRACTION ===")

    for name, dataframe in datasets.items():
        print(
            f"{name:<22} "
            f"{dataframe.shape[0]:>10,} lignes | "
            f"{dataframe.shape[1]:>2} colonnes"
        )
