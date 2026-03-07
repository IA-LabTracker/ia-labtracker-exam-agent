from __future__ import annotations

from copy import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from src.aggregator.consolidate import ReconciledRow, ReverseRow
from src.utils.logging import logger

COLUMNS = [
    "input_tema",
    "input_equivalencia",
    "normalized_tema",
    "normalized_subtema",
    "num_questions",
    "match_label",
    "match_score",
    "notes",
]

COLUMN_HEADERS = {
    "input_tema": "Tema (entrada)",
    "input_equivalencia": "Equivalência",
    "normalized_tema": "Tema",
    "normalized_subtema": "Subtema",
    "num_questions": "Qtd. Questões",
    "match_label": "Confiança",
    "match_score": "Score",
    "notes": "Observações",
}

REVERSE_COLUMNS = [
    "db_tema",
    "db_subtema",
    "db_num_questions",
    "matched_input",
    "similarity_score",
    "coverage_status",
    "notes",
]

REVERSE_HEADERS = {
    "db_tema": "Tema (banco)",
    "db_subtema": "Subtema (banco)",
    "db_num_questions": "Qtd. Questões",
    "matched_input": "Entrada Correspondente",
    "similarity_score": "Similaridade",
    "coverage_status": "Status Cobertura",
    "notes": "Observações",
}

ROW_COLORS = {
    "#EF4444": "00FECACA",  # vermelho → fundo rosa claro
    "#F97316": "00FED7AA",  # laranja → fundo pêssego
    "#EAB308": "00FEF9C3",  # amarelo → fundo amarelo claro
    "#22C55E": "00DCFCE7",  # verde → fundo verde claro
    "#3B82F6": "00DBEAFE",  # azul → fundo azul claro
}

BADGE_COLORS = {
    "#EF4444": ("00EF4444", "00FFFFFF"),  # vermelho
    "#F97316": ("00F97316", "00FFFFFF"),  # laranja
    "#EAB308": ("00EAB308", "00000000"),  # amarelo
    "#22C55E": ("0022C55E", "00000000"),  # verde
    "#3B82F6": ("003B82F6", "00FFFFFF"),  # azul
}

COVERAGE_COLORS = {
    "coberto": "00DCFCE7",  # green
    "parcial": "00FEF9C3",  # yellow
    "não coberto": "00FECACA",  # red
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
    reverse_rows: list[ReverseRow] | None = None,
) -> Path:
    output_path = Path(output_path)
    records = []
    cor_hex_list = []
    for r in rows:
        d = asdict(r)
        cor_hex_list.append(d.pop("cor_hex", "#22C55E"))
        d.pop("match_method", None)
        d["num_questions"] = f"{d['num_questions']} questões"
        d["match_score"] = f"{d.get('match_score', 0):.0%}"
        records.append(d)

    df = pd.DataFrame(records)
    df = df[[c for c in COLUMNS if c in df.columns]]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranking")

        if reverse_rows:
            rev_records = []
            rev_cor_list = []
            for rr in reverse_rows:
                d = asdict(rr)
                rev_cor_list.append(d.pop("db_cor_hex", "#22C55E"))
                d["db_num_questions"] = f"{d['db_num_questions']} questões"
                d["similarity_score"] = f"{d['similarity_score']:.2%}"
                rev_records.append(d)
            df_rev = pd.DataFrame(rev_records)
            df_rev = df_rev[[c for c in REVERSE_COLUMNS if c in df_rev.columns]]
            df_rev.to_excel(writer, index=False, sheet_name="Cobertura Reversa")

    _apply_styling(output_path, df, cor_hex_list, reverse_rows)

    logger.info("Wrote %d rows to %s", len(df), output_path)

    if also_csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info("Also wrote CSV: %s", csv_path)

    return output_path


def _apply_styling(
    path: Path,
    df: pd.DataFrame,
    cor_hex_list: list[str],
    reverse_rows: list[ReverseRow] | None = None,
) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)

    # --- Style "Ranking" sheet ---
    ws = wb["Ranking"]
    _style_ranking_sheet(ws, df, cor_hex_list)

    # --- Style "Cobertura Reversa" sheet ---
    if reverse_rows and "Cobertura Reversa" in wb.sheetnames:
        ws_rev = wb["Cobertura Reversa"]
        _style_reverse_sheet(ws_rev, reverse_rows)

    wb.save(path)


def _style_ranking_sheet(ws, df: pd.DataFrame, cor_hex_list: list[str]) -> None:
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
    confidence_col = (
        (list(df.columns).index("match_label") + 1)
        if "match_label" in df.columns
        else None
    )
    score_col = (
        (list(df.columns).index("match_score") + 1)
        if "match_score" in df.columns
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
            elif col_idx == confidence_col or col_idx == score_col:
                # Style confidence/score cells based on label content
                val = str(cell.value or "")
                if "Quente" in val or col_idx == score_col:
                    cell.font = Font(size=11, bold=True, color="00000000")
                _apply_confidence_fill(cell, val, row_argb)
            elif row_argb:
                cell.fill = _make_fill(row_argb)

    ws.row_dimensions[1].height = 30
    for row_idx in range(2, num_rows + 1):
        ws.row_dimensions[row_idx].height = 25

    _auto_fit_columns(ws, num_cols, num_rows)


def _style_reverse_sheet(ws, reverse_rows: list[ReverseRow]) -> None:
    num_cols = ws.max_column
    num_rows = ws.max_row

    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    header_font = Font(color="00FFFFFF", bold=True, size=11)
    data_font = Font(size=11, color="00000000")

    # Headers
    rev_col_names = [c for c in REVERSE_COLUMNS]
    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        col_name = rev_col_names[col_idx - 1] if col_idx <= len(rev_col_names) else ""
        cell.value = REVERSE_HEADERS.get(col_name, cell.value)
        cell.fill = _make_header_fill()
        cell.font = copy(header_font)
        cell.alignment = copy(center)
        cell.border = _make_border()

    # Find special column indices
    num_q_col = None
    status_col = None
    sim_col = None
    for i, c in enumerate(REVERSE_COLUMNS):
        if c == "db_num_questions":
            num_q_col = i + 1
        elif c == "coverage_status":
            status_col = i + 1
        elif c == "similarity_score":
            sim_col = i + 1

    for row_idx in range(2, num_rows + 1):
        rr_idx = row_idx - 2
        rr = reverse_rows[rr_idx] if rr_idx < len(reverse_rows) else None
        hex_value = rr.db_cor_hex if rr else None
        row_argb = ROW_COLORS.get(hex_value) if hex_value else None
        badge = BADGE_COLORS.get(hex_value) if hex_value else None

        for col_idx in range(1, num_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = _make_border()
            cell.alignment = copy(center)
            cell.font = copy(data_font)

            if col_idx == num_q_col and badge:
                # Manchester badge color on num_questions
                cell.fill = _make_fill(badge[0])
                cell.font = Font(color=badge[1], bold=True, size=11)
            elif col_idx == status_col and rr:
                # Coverage status colored by coverage level
                cov_color = COVERAGE_COLORS.get(rr.coverage_status)
                if cov_color:
                    cell.fill = _make_fill(cov_color)
                cell.font = Font(size=11, bold=True, color="00000000")
            elif col_idx == sim_col and rr:
                cell.font = Font(size=11, bold=True, color="00000000")
                if row_argb:
                    cell.fill = _make_fill(row_argb)
            elif row_argb:
                cell.fill = _make_fill(row_argb)

    ws.row_dimensions[1].height = 30
    for row_idx in range(2, num_rows + 1):
        ws.row_dimensions[row_idx].height = 25

    _auto_fit_columns(ws, num_cols, num_rows)


CONFIDENCE_FILLS = {
    "Quente": "00DCFCE7",  # green
    "Morno": "00FEF9C3",  # yellow
    "Frio": "00DBEAFE",  # light blue
    "Sem match": "00FECACA",  # red
}


def _apply_confidence_fill(cell, label_value: str, fallback_argb: str | None) -> None:
    """Color the confidence cell based on its temperature label."""
    for keyword, argb in CONFIDENCE_FILLS.items():
        if keyword in label_value:
            cell.fill = _make_fill(argb)
            return
    if fallback_argb:
        cell.fill = _make_fill(fallback_argb)


def _auto_fit_columns(ws, num_cols: int, num_rows: int) -> None:
    for col_idx in range(1, num_cols + 1):
        col_letter = ws.cell(row=1, column=col_idx).column_letter
        max_len = 0
        for row_idx in range(1, num_rows + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 15), 55)
