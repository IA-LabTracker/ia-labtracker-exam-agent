from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import logger

_COLUMN_MAP: dict[str, str] = {
    "tema": "tema",
    "theme": "tema",
    "classificacao": "classificacao",
    "classificação": "classificacao",
    "classification": "classificacao",
    "equivalencia": "equivalencia",
    "equivalência": "equivalencia",
    "equivalence": "equivalencia",
    "num_questoes": "num_questoes",
    "num_questões": "num_questoes",
    "num_questions": "num_questoes",
    "questoes": "num_questoes",
    "questões": "num_questoes",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        _COLUMN_MAP.get(
            c.strip().lower().replace(" ", "_"), c.strip().lower().replace(" ", "_")
        )
        for c in df.columns
    ]
    return df


def read_excel(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    logger.info("[read_excel] reading file: %s", path.name)

    df = pd.read_excel(path, engine="openpyxl")
    logger.debug("[read_excel] raw columns from Excel: %s", list(df.columns))
    df = _normalize_columns(df)
    logger.debug("[read_excel] normalized columns: %s", list(df.columns))

    if "tema" not in df.columns:
        raise ValueError(
            f"Excel file {path.name} must have a 'tema' (or 'theme') column. "
            f"Found columns: {list(df.columns)}"
        )

    for col in ("classificacao", "equivalencia", "num_questoes"):
        if col not in df.columns:
            logger.debug("[read_excel] column '%s' not found, adding as None", col)
            df[col] = None

    logger.info("[read_excel] dropping rows with missing 'tema'")
    df = df.dropna(subset=["tema"])
    records = df.to_dict(orient="records")
    logger.info(
        "[read_excel] successfully read %d rows from %s", len(records), path.name
    )
    return records
