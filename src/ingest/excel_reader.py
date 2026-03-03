from typing import List, Dict
import pandas as pd


def read_excel(path: str) -> List[Dict]:
    """Load input Excel and return list of rows with expected fields.

    Input workbook must have columns: tema, classificacao, equivalencia,
    num_questoes. Extra columns are ignored. Column names are normalized to
    lowercase.
    """
    df = pd.read_excel(path)
    df = df.rename(columns=lambda s: s.strip().lower())
    rows: List[Dict] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "tema": r.get("tema"),
                "classificacao": r.get("classificacao"),
                "equivalencia": r.get("equivalencia"),
                "num_questoes": int(r.get("num_questoes", 0)),
            }
        )
    return rows
