"""Reconciliation pipeline — orchestrates matching, retry, and reverse coverage.

Split into focused modules:
  - models.py: dataclasses + constants
  - matchers.py: tema/subtema matching logic
  - llm_refinement.py: optional LLM judge step
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from src.normalize.normalizer import classify_color, normalize_tema_subtema
from src.retriever.hybrid_retriever import Candidate, retrieve_candidates
from src.utils.logging import logger

# Re-export public API so existing imports still work
from src.aggregator.models import (  # noqa: F401
    MATCH_EXACT,
    MATCH_FTS,
    MATCH_LLM,
    MATCH_NONE,
    MATCH_SEMANTIC,
    MIN_ACCEPTABLE_SCORE,
    MatchInfo,
    ReconciledRow,
    ReverseRow,
    _classify_temperature,
)
from src.aggregator.matchers import (
    find_stat_tema_only,
    find_stat_with_subtema,
    resolve_tema,
    retry_low_score,
)
from src.aggregator.llm_refinement import apply_llm_judge

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder
    from src.llm.judge import LLMJudge


# ---------------------------------------------------------------------------
# Main reconciliation
# ---------------------------------------------------------------------------


def reconcile_row(
    row: dict[str, Any],
    embedder: BaseEmbedder,
    db: DBClient,
) -> ReconciledRow:
    tema_raw = str(row.get("tema", ""))
    subtema_raw = row.get("subtema")
    if (
        subtema_raw
        and str(subtema_raw).strip()
        and str(subtema_raw).strip().lower() != "nan"
    ):
        subtema_raw = str(subtema_raw).strip()
    else:
        subtema_raw = None
    equivalencia = row.get("equivalencia")
    has_subtema = bool(subtema_raw)
    logger.debug(
        "[reconcile_row] processing tema='%s' subtema='%s'",
        tema_raw[:50],
        (subtema_raw or "")[:50],
    )

    norm_tema, norm_subtema = normalize_tema_subtema(tema_raw, subtema_raw)
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
    candidate_tema = best_candidate.tema_normalized if best_candidate else None
    candidate_subtema = best_candidate.subtema_normalized if best_candidate else None

    # --- Comparison #1: tema vs tema (with semantic fallback) ---
    resolved_tema, tema_match = resolve_tema(
        tema_raw, norm_tema, candidate_tema, equivalencia, db, embedder
    )
    final_tema = resolved_tema or norm_tema
    match_info = tema_match

    # --- Comparison #2: subtema vs subtema (or tema-only stat) ---
    stat: dict | None = None
    num_questions = 0
    if resolved_tema:
        if has_subtema:
            stat, match_info = find_stat_with_subtema(
                resolved_tema,
                subtema_raw,
                norm_subtema,
                candidate_subtema,
                db,
                embedder,
                tema_match,
            )
            if stat:
                num_questions = stat["num_questions"]
        else:
            stat, num_questions = find_stat_tema_only(resolved_tema, db)

    notes_parts = []
    equivalencia_out = equivalencia
    final_subtema = norm_subtema

    if stat:
        _, cor_hex = classify_color(num_questions)

        stat_subtema = stat.get("subtema")
        if stat_subtema:
            final_subtema = stat_subtema

        if has_subtema:
            if stat_subtema:
                equivalencia_out = f"{final_tema} | {stat_subtema}"
            else:
                equivalencia_out = final_tema
        else:
            equivalencia_out = final_tema

        subtemas = db.get_subtemas_for_tema(final_tema)
        if subtemas:
            subtema_names = [s["subtema"] for s in subtemas if s.get("subtema")]
            if subtema_names:
                notes_parts.append(f"Subtemas: {' | '.join(subtema_names)}")

        notes_parts.append(
            f"Fonte: {stat.get('institution', 'N/A')} "
            f"(ranking #{stat.get('ranking', '?')})"
        )
        logger.debug(
            "[reconcile_row] matched: tema='%s' subtema='%s' num=%d",
            final_tema,
            stat_subtema,
            num_questions,
        )
    else:
        num_questions = num_candidates
        _, cor_hex = classify_color(num_questions)
        notes_parts.append("No matches found in DB")

    if not candidates and not stat:
        notes_parts.append("No matches found in questions DB")
    elif best_candidate:
        notes_parts.append(
            f"Best match: {best_candidate.tema_normalized or '?'}"
            f" (sim={best_candidate.similarity:.2f})"
        )

    notes = "; ".join(notes_parts)
    input_display = f"{tema_raw} | {subtema_raw}" if subtema_raw else tema_raw

    return ReconciledRow(
        input_tema=input_display,
        input_equivalencia=equivalencia_out,
        normalized_tema=final_tema,
        normalized_subtema=final_subtema if has_subtema else None,
        num_questions=num_questions,
        match_method=match_info.method,
        match_score=match_info.score,
        match_label=match_info.label
        or _classify_temperature(match_info.method, match_info.score),
        cor_hex=cor_hex,
        notes=notes,
    )


def reconcile_all(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
    db: DBClient,
    llm_judge: LLMJudge | None = None,
) -> list[ReconciledRow]:
    """Reconcile every input row 1:1. No dedup — every input line produces one output line."""
    logger.info(
        "[reconcile_all] starting reconciliation of %d input rows", len(input_rows)
    )
    results = []
    for i, row in enumerate(input_rows, start=1):
        if i % 50 == 0 or i == 1:
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

    # Retry rows with low scores using alternative queries
    for i, r in enumerate(results):
        if r.match_score < MIN_ACCEPTABLE_SCORE and r.match_method != MATCH_NONE:
            retried = retry_low_score(input_rows[i], r, embedder, db)
            if retried and retried.match_score > r.match_score:
                logger.info(
                    "[reconcile_all] retry improved row %d: %.0f%% -> %.0f%%",
                    i + 1,
                    r.match_score * 100,
                    retried.match_score * 100,
                )
                results[i] = retried

    # Optional LLM judge step — validate/improve low-confidence matches
    if llm_judge is not None:
        from src.config import get_settings

        settings = get_settings()
        results = apply_llm_judge(
            results,
            llm_judge,
            db,
            threshold=settings.llm_judge_threshold,
            embedder=embedder,
        )

    # Sort by score descending (highest confidence first) — NO dedup
    results.sort(key=lambda r: r.match_score, reverse=True)
    logger.info(
        "[reconcile_all] reconciliation complete: %d rows in -> %d rows out",
        len(input_rows),
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Reverse coverage: DB themes → check if input covers them
# ---------------------------------------------------------------------------


def reverse_coverage(
    reconciled: list[ReconciledRow],
    embedder: BaseEmbedder,
    db: DBClient,
    coverage_threshold: float = 0.50,
    partial_threshold: float = 0.35,
) -> list[ReverseRow]:
    """For each theme_stat in the DB, check if any reconciled result covers it."""
    logger.info("[reverse_coverage] starting reverse coverage analysis")

    all_stats = db.get_all_theme_stats()
    if not all_stats:
        logger.warning("[reverse_coverage] no theme_stats found in DB")
        return []

    # Build mapping: (normalized_tema, normalized_subtema) → best reconciled row
    matched_map: dict[tuple[str, str | None], ReconciledRow] = {}
    for r in reconciled:
        key = (r.normalized_tema.lower(), (r.normalized_subtema or "").lower() or None)
        existing = matched_map.get(key)
        if existing is None or r.match_score > existing.match_score:
            matched_map[key] = r

    # For semantic fallback: embed unique reconciled temas (dedup for speed)
    seen_texts: dict[str, int] = {}
    reconciled_texts = []
    reconciled_displays = []
    for r in reconciled:
        text = r.normalized_tema
        if r.normalized_subtema:
            text += f" {r.normalized_subtema}"
        if text not in seen_texts:
            seen_texts[text] = len(reconciled_texts)
            reconciled_texts.append(text)
            reconciled_displays.append(r.input_tema)
    reconciled_embeddings = (
        embedder.embed_batch(reconciled_texts) if reconciled_texts else []
    )

    results: list[ReverseRow | None] = []
    unmatched_stats = []
    unmatched_indices = []

    for stat in all_stats:
        stat_tema = stat["tema"].lower()
        stat_subtema = (stat.get("subtema") or "").lower() or None

        matched_row = matched_map.get((stat_tema, stat_subtema))

        if not matched_row and stat_subtema is None:
            for key, row in matched_map.items():
                if key[0] == stat_tema:
                    if matched_row is None or row.match_score > matched_row.match_score:
                        matched_row = row

        if matched_row and matched_row.match_score >= partial_threshold:
            score = matched_row.match_score
            status = "coberto" if score >= coverage_threshold else "parcial"
            results.append(
                ReverseRow(
                    db_tema=stat["tema"],
                    db_subtema=stat.get("subtema"),
                    db_num_questions=stat["num_questions"],
                    db_cor_hex=stat.get("cor_hex", "#22C55E"),
                    matched_input=matched_row.input_tema,
                    similarity_score=score,
                    coverage_status=status,
                    notes=f"Match: {matched_row.input_tema} (score={score:.0%})",
                )
            )
        else:
            unmatched_stats.append(stat)
            unmatched_indices.append(len(results))
            results.append(None)  # placeholder

    # Semantic fallback for unmatched DB entries
    if unmatched_stats and reconciled_embeddings:
        stat_texts = []
        for s in unmatched_stats:
            text = s["tema"]
            if s.get("subtema"):
                text += f" {s['subtema']}"
            stat_texts.append(text)
        stat_embeddings = embedder.embed_batch(stat_texts)

        stat_matrix = np.array(stat_embeddings, dtype=np.float32)
        rec_matrix = np.array(reconciled_embeddings, dtype=np.float32)
        stat_norms = np.linalg.norm(stat_matrix, axis=1, keepdims=True)
        rec_norms = np.linalg.norm(rec_matrix, axis=1, keepdims=True)
        stat_norms[stat_norms == 0] = 1.0
        rec_norms[rec_norms == 0] = 1.0
        sim_matrix = (stat_matrix / stat_norms) @ (rec_matrix / rec_norms).T
        best_indices = np.argmax(sim_matrix, axis=1)
        best_scores = sim_matrix[np.arange(len(unmatched_stats)), best_indices]

        for i, stat in enumerate(unmatched_stats):
            best_score = float(best_scores[i])
            best_display = reconciled_displays[int(best_indices[i])]

            if best_score >= coverage_threshold:
                status = "coberto"
            elif best_score >= partial_threshold:
                status = "parcial"
            else:
                status = "não coberto"

            notes_parts = []
            if best_score >= partial_threshold:
                notes_parts.append(f"Melhor match (semântico): {best_display}")
            notes_parts.append(f"Similaridade: {best_score:.2%}")

            results[unmatched_indices[i]] = ReverseRow(
                db_tema=stat["tema"],
                db_subtema=stat.get("subtema"),
                db_num_questions=stat["num_questions"],
                db_cor_hex=stat.get("cor_hex", "#22C55E"),
                matched_input=best_display if best_score >= partial_threshold else None,
                similarity_score=best_score,
                coverage_status=status,
                notes="; ".join(notes_parts),
            )

    final_results = [r for r in results if r is not None]
    final_results.sort(key=lambda r: r.similarity_score, reverse=True)

    covered = sum(1 for r in final_results if r.coverage_status == "coberto")
    partial = sum(1 for r in final_results if r.coverage_status == "parcial")
    uncovered = sum(1 for r in final_results if r.coverage_status == "não coberto")
    logger.info(
        "[reverse_coverage] done: %d coberto, %d parcial, %d não coberto (total=%d)",
        covered,
        partial,
        uncovered,
        len(final_results),
    )
    return final_results
