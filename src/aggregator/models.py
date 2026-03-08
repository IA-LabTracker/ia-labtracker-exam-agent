"""Shared data models and constants for the aggregator module."""

from __future__ import annotations

from dataclasses import dataclass


# Fixed institution list — order determines column order in the exported spreadsheet
INSTITUTIONS: list[str] = [
    "AMP-PR",
    "AMRIGS",
    "CERMAM",
    "FAMENE",
    "FAMERP",
    "FELUMA",
    "HCPA",
    "HEVV - HOSPITAL EVANGÉLICO DE VILA VELHA",
    "HIAE",
    "IAMSPE",
    "INTO RJ",
    "PSU - MG",
    "PSU – GO",
    "REVALIDA INEP",
    "SCMSP",
    "SES-DF",
    "SES-PE",
    "SIRIO",
    "SUS BA",
    "SUS SP",
    "UEPA",
    "UERJ",
    "UNESP",
    "UNICAMP",
    "UNIFESP",
    "USP RP",
    "USP SP",
]

MATCH_EXACT = "exato"
MATCH_FTS = "FTS"
MATCH_SEMANTIC = "semântico"
MATCH_LLM = "LLM"
MATCH_NONE = "sem match"

# Minimum acceptable score — rows below this get retried with variants
MIN_ACCEPTABLE_SCORE = 0.50


@dataclass
class MatchInfo:
    """Tracks how a match was found and its confidence score."""

    method: str = MATCH_NONE
    score: float = 0.0
    label: str = ""


def _classify_temperature(method: str, score: float) -> str:
    """Classify match quality as quente/morno/frio based on method + score."""
    if method == MATCH_EXACT:
        return "🔴 Quente (exato)"
    if method == MATCH_FTS:
        if score >= 0.7:
            return "🔴 Quente (FTS)"
        if score >= 0.5:
            return "🟠 Morno (FTS)"
        return "🟡 Frio (FTS)"
    if method == MATCH_SEMANTIC:
        if score >= 0.75:
            return "🔴 Quente (semântico)"
        if score >= 0.6:
            return "🟠 Morno (semântico)"
        if score >= 0.5:
            return "🟡 Frio (semântico)"
        return "🔵 Muito frio (semântico)"
    if method == MATCH_LLM:
        if score >= 0.8:
            return "🔴 Quente (LLM)"
        if score >= 0.6:
            return "🟠 Morno (LLM)"
        return "🟡 Frio (LLM)"
    return "⚪ Sem match"


@dataclass
class ReconciledRow:
    input_tema: str
    input_equivalencia: str | None
    normalized_tema: str
    normalized_subtema: str | None
    # {institution: num_questions} — one entry per university; missing institutions → 0
    questions_by_institution: dict[str, int]
    match_method: str = MATCH_NONE
    match_score: float = 0.0
    match_label: str = "⚪ Sem match"
    # Row-level color derived from the SUM of all institutions
    cor_hex: str = "#22C55E"
    notes: str = ""


@dataclass
class ReverseRow:
    """Row for the reverse-coverage sheet."""

    db_tema: str
    db_subtema: str | None
    db_num_questions: int
    db_cor_hex: str
    matched_input: str | None
    similarity_score: float
    coverage_status: str  # "coberto", "parcial", "não coberto"
    notes: str = ""
