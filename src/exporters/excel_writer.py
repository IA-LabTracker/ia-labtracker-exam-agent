from __future__ import annotations

from copy import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from src.aggregator.consolidate import ReconciledRow
from src.utils.logging import logger

COLUMNS = [
    "input_tema",
    "input_equivalencia",
    "normalized_tema",
    "normalized_subtema",
    "num_questions",
    "notes",
]

COLUMN_HEADERS = {
    "input_tema": "Tema (entrada)",
    "input_equivalencia": "Equivalência",
    "normalized_tema": "Tema",
    "normalized_subtema": "Subtema",
    "num_questions": "Qtd. Questões",
    "notes": "Observações",
}

ROW_COLORS = {
    "#EF4444": "00FECACA",
    "#F97316": "00FED7AA",
    "#22C55E": "00DCFCE7",
    "#3B82F6": "00DBEAFE",
}

BADGE_COLORS = {
    "#EF4444": ("00EF4444", "00FFFFFF"),
    "#F97316": ("00F97316", "00FFFFFF"),
    "#22C55E": ("0022C55E", "00000000"),
    "#3B82F6": ("003B82F6", "00FFFFFF"),
}


def _make_border() -> Border:
    side = Side(style="thin", color="00000000")
    return Border(left=side, right=side, top=side, bottom=side)


def _make_header_fill() -> PatternFill:
    return PatternFill(start_color="001F1F1F", end_color="001F1F1F", fill_type="solid")


def _make_fill(argb: str) -> PatternFill:
    return PatternFill(start_color=argb, end_color=argb, fill_type="solid")


def write_excel(
    rows: list[ReconciledRow],
    output_path: str | Path,
    also_csv: bool = False,
) -> Path:
    output_path = Path(output_path)
    records = []
    cor_hex_list = []
    for r in rows:
        d = asdict(r)
        cor_hex_list.append(d.pop("cor_hex", "#22C55E"))
        d["num_questions"] = f"{d['num_questions']} questões"
        records.append(d)

    df = pd.DataFrame(records)
    df = df[[c for c in COLUMNS if c in df.columns]]
    df.to_excel(output_path, index=False, engine="openpyxl")

    _apply_styling(output_path, df, cor_hex_list)

    logger.info("Wrote %d rows to %s", len(df), output_path)

    if also_csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info("Also wrote CSV: %s", csv_path)

    return output_path


def _apply_styling(path: Path, df: pd.DataFrame, cor_hex_list: list[str]) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb.active

    num_cols = ws.max_column
    num_rows = ws.max_row

    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    header_font = Font(color="00FFFFFF", bold=True, size=11)
    data_font = Font(size=11, color="00000000")

    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        col_name = df.columns[col_idx - 1] if col_idx <= len(df.columns) else ""
        cell.value = COLUMN_HEADERS.get(col_name, cell.value)
        cell.fill = _make_header_fill()
        cell.font = copy(header_font)
        cell.alignment = copy(center)
        cell.border = _make_border()

    num_q_col = (
        (list(df.columns).index("num_questions") + 1)
        if "num_questions" in df.columns
        else None
    )

    for row_idx in range(2, num_rows + 1):
        hex_value = (
            cor_hex_list[row_idx - 2] if (row_idx - 2) < len(cor_hex_list) else None
        )
        row_argb = ROW_COLORS.get(hex_value) if hex_value else None
        badge = BADGE_COLORS.get(hex_value) if hex_value else None

        for col_idx in range(1, num_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = _make_border()
            cell.alignment = copy(center)
            cell.font = copy(data_font)

            if col_idx == num_q_col and badge:
                cell.fill = _make_fill(badge[0])
                cell.font = Font(color=badge[1], bold=True, size=11)
            elif row_argb:
                cell.fill = _make_fill(row_argb)

    ws.row_dimensions[1].height = 30
    for row_idx in range(2, num_rows + 1):
        ws.row_dimensions[row_idx].height = 25

    for col_idx in range(1, num_cols + 1):
        col_letter = ws.cell(row=1, column=col_idx).column_letter
        max_len = 0
        for row_idx in range(1, num_rows + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 15), 55)

    wb.save(path)
