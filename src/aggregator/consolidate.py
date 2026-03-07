from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from src.normalize.normalizer import classify_color, normalize_tema_subtema
from src.retriever.hybrid_retriever import Candidate, retrieve_candidates
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder



@dataclass
class ReconciledRow:
    input_tema: str
    input_equivalencia: str | None
    normalized_tema: str
    normalized_subtema: str | None
    num_questions: int
    cor_hex: str = "#22C55E"
    notes: str = ""


def _build_equivalencia(stat: dict) -> str:
    """Build 'tema | subtema' string from a theme_stats row."""
    tema = stat.get("tema", "")
    subtema = stat.get("subtema")
    if subtema:
        return f"{tema} | {subtema}"
    return tema


def reconcile_row(
    row: dict[str, Any],
    embedder: BaseEmbedder,
    db: DBClient,
) -> ReconciledRow:
    tema_raw = str(row.get("tema", ""))
    equivalencia = row.get("equivalencia")
    logger.debug("[reconcile_row] processing tema='%s'", tema_raw[:50])

    norm_tema, norm_subtema = normalize_tema_subtema(tema_raw)
    logger.debug(
        "[reconcile_row] normalized: tema='%s' subtema='%s'",
        norm_tema[:50],
        norm_subtema[:50] if norm_subtema else "(none)",
    )

    query = norm_tema
    if norm_subtema:
        query += f" {norm_subtema}"
    if equivalencia:
        query += f" {equivalencia}"

    logger.debug("[reconcile_row] search query='%s'", query[:100])
    candidates: list[Candidate] = retrieve_candidates(query, embedder, db)
    logger.debug("[reconcile_row] found %d candidates", len(candidates))

    num_candidates = len(candidates)
    best_candidate = candidates[0] if candidates else None

    final_tema = (
        best_candidate.tema_normalized or norm_tema if best_candidate else norm_tema
    )
    final_subtema = (
        best_candidate.subtema_normalized or norm_subtema
        if best_candidate
        else norm_subtema
    )

    stat = db.get_theme_stat(final_tema, final_subtema)
    if not stat:
        stat = db.find_best_theme_stat(tema_raw)
    if not stat:
        stat = db.find_best_theme_stat(norm_tema)

    notes_parts = []
    equivalencia_out = equivalencia

    if stat:
        cor_hex = stat["cor_hex"]
        num_questions = stat["num_questions"]

        stat_tema = stat.get("tema", "")
        stat_subtema = stat.get("subtema")
        if stat_tema:
            final_tema = stat_tema
        if stat_subtema:
            final_subtema = stat_subtema

        equivalencia_out = _build_equivalencia(stat)

        subtemas = db.get_subtemas_for_tema(stat["tema"])
        if subtemas:
            subtema_names = [s["subtema"] for s in subtemas if s.get("subtema")]
            if subtema_names:
                notes_parts.append(
                    f"Subtemas: {' | '.join(subtema_names)}"
                )

        notes_parts.append(
            f"Fonte: {stat.get('institution', 'N/A')} "
            f"(ranking #{stat.get('ranking', '?')})"
        )
        logger.debug(
            "[reconcile_row] theme_stat match: tema='%s' subtema='%s' num=%d",
            stat.get("tema"),
            stat.get("subtema"),
            num_questions,
        )
    else:
        num_questions = num_candidates
        _, cor_hex = classify_color(num_questions)
        notes_parts.append("No matches found in DB")

    if not candidates:
        notes_parts.append("No matches found in questions DB")
    elif best_candidate:
        notes_parts.append(
            f"Best match: {best_candidate.tema_normalized or '?'}"
            f" (sim={best_candidate.similarity:.2f})"
        )

    notes = "; ".join(notes_parts)

    return ReconciledRow(
        input_tema=tema_raw,
        input_equivalencia=equivalencia_out,
        normalized_tema=final_tema,
        normalized_subtema=final_subtema,
        num_questions=num_questions,
        cor_hex=cor_hex,
        notes=notes,
    )


def reconcile_all(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
    db: DBClient,
) -> list[ReconciledRow]:
    logger.info(
        "[reconcile_all] starting reconciliation of %d input rows", len(input_rows)
    )
    results = []
    for i, row in enumerate(input_rows, start=1):
        logger.info("[reconcile_all] processing row %d / %d", i, len(input_rows))
        try:
            result = reconcile_row(row, embedder, db)
            results.append(result)
        except Exception as exc:
            logger.error(
                "[reconcile_all] error on row %d / %d (tema='%s'): %s",
                i,
                len(input_rows),
                row.get("tema", "(unknown)")[:50],
                exc,
                exc_info=True,
            )
            raise

    results.sort(key=lambda r: r.num_questions, reverse=True)
    logger.info(
        "[reconcile_all] reconciliation complete: %d rows produced", len(results)
    )
    return results
