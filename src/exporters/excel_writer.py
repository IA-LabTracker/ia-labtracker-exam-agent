from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill

from src.aggregator.consolidate import ReconciledRow
from src.utils.logging import logger

FILL_MAP = {
    "#EF4444": PatternFill(start_color="EF4444", end_color="EF4444", fill_type="solid"),
    "#F97316": PatternFill(start_color="F97316", end_color="F97316", fill_type="solid"),
    "#EAB308": PatternFill(start_color="EAB308", end_color="EAB308", fill_type="solid"),
    "#22C55E": PatternFill(start_color="22C55E", end_color="22C55E", fill_type="solid"),
}

WHITE_FONT = Font(color="FFFFFF", bold=True)
BLACK_FONT = Font(color="000000")

COLUMNS = [
    "input_tema",
    "input_classificacao",
    "input_equivalencia",
    "normalized_tema",
    "normalized_subtema",
    "similarity",
    "num_questions",
    "cor",
    "cor_hex",
    "matched_ids",
    "priority_score",
    "notes",
]


def write_excel(
    rows: list[ReconciledRow],
    output_path: str | Path,
    also_csv: bool = False,
) -> Path:
    output_path = Path(output_path)
    records = []
    for r in rows:
        d = asdict(r)
        d["matched_ids"] = ",".join(str(i) for i in d["matched_ids"])
        records.append(d)

    df = pd.DataFrame(records)
    df = df[[c for c in COLUMNS if c in df.columns]]
    df.to_excel(output_path, index=False, engine="openpyxl")

    _apply_color_formatting(output_path, df)

    logger.info("Wrote %d rows to %s", len(df), output_path)

    if also_csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info("Also wrote CSV: %s", csv_path)

    return output_path


def _apply_color_formatting(path: Path, df: pd.DataFrame) -> None:
    from openpyxl import load_workbook

    if "cor_hex" not in df.columns:
        return

    wb = load_workbook(path)
    ws = wb.active
    cor_hex_col = list(df.columns).index("cor_hex") + 1

    for row_idx in range(2, len(df) + 2):
        hex_value = ws.cell(row=row_idx, column=cor_hex_col).value
        if not hex_value:
            continue

        fill = FILL_MAP.get(hex_value)
        if not fill:
            continue

        font = WHITE_FONT if hex_value in ("#EF4444", "#F97316") else BLACK_FONT

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.fill = fill
            cell.font = font

    wb.save(path)
