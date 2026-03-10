"""Reconciliation pipeline — orchestrates matching, retry, and reverse coverage.

Split into focused modules:
  - models.py: dataclasses + constants
  - matchers.py: tema/subtema matching logic
  - llm_refinement.py: optional LLM judge step
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    _eq_raw = row.get("equivalencia")
    equivalencia = (
        str(_eq_raw).strip()
        if _eq_raw and str(_eq_raw).strip().lower() not in ("nan", "none", "")
        else None
    )
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
    if resolved_tema:
        if has_subtema:
            stat, sub_match = find_stat_with_subtema(
                resolved_tema,
                subtema_raw,
                norm_subtema,
                candidate_subtema,
                db,
                embedder,
            )
            if stat:
                match_info = sub_match
            else:
                # No cross-level: if subtema not found, reset to MATCH_NONE
                # (don't inherit tema-level score for a subtema-level row)
                match_info = MatchInfo()
        else:
            stat, _ = find_stat_tema_only(resolved_tema, db)

    notes_parts = []
    equivalencia_out = equivalencia
    final_subtema = norm_subtema

    if stat:
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
            "[reconcile_row] matched: tema='%s' subtema='%s'",
            final_tema,
            stat_subtema,
        )
    elif resolved_tema:
        # Tema was found in DB but subtema didn't match — build equivalência from what was resolved
        equivalencia_out = resolved_tema

    # --- Fetch per-institution question counts from theme_stats_all ---
    if resolved_tema or stat:
        lookup_tema = final_tema
        # Only pass subtema when it was actually resolved from the DB (stat has it).
        # Using the raw/normalized input subtema would produce no matches in theme_stats_all,
        # causing every institution to show 0 questions.
        lookup_subtema = stat.get("subtema") if (stat and stat.get("subtema")) else None
        questions_by_institution = db.get_questions_by_institution(
            lookup_tema, lookup_subtema
        )
    else:
        questions_by_institution = {}

    total_questions = (
        sum(questions_by_institution.values())
        if questions_by_institution
        else num_candidates
    )
    _, cor_hex = classify_color(total_questions)

    if not stat and not resolved_tema:
        notes_parts.append("Nenhum match encontrado na base de dados")
    elif not stat and resolved_tema:
        notes_parts.append(f"Tema resolvido: {resolved_tema} (sem estatísticas de subtema)")

    if not candidates and not stat:
        notes_parts.append("Nenhum candidato encontrado na busca")
    elif best_candidate:
        notes_parts.append(
            f"Best match: {best_candidate.tema_normalized or '?'}"
            f" (sim={best_candidate.similarity:.2f})"
        )

    notes = "; ".join(notes_parts)
    input_display = f"{tema_raw} | {subtema_raw}" if subtema_raw else tema_raw

    # Safety net: equivalencia must always reflect the resolved match.
    # Handles cases where the stat/resolved_tema paths didn't set it
    # (e.g. NaN from pandas input or no DB match at all).
    if not equivalencia_out and final_tema:
        if final_subtema and has_subtema:
            equivalencia_out = f"{final_tema} | {final_subtema}"
        else:
            equivalencia_out = final_tema

    return ReconciledRow(
        input_tema=input_display,
        input_equivalencia=equivalencia_out,
        normalized_tema=final_tema,
        normalized_subtema=final_subtema if has_subtema else None,
        questions_by_institution=questions_by_institution,
        match_method=match_info.method,
        match_score=match_info.score,
        match_label=match_info.label
        or _classify_temperature(match_info.method, match_info.score),
        cor_hex=cor_hex,
        notes=notes,
    )


_MAX_WORKERS = min(8, (os.cpu_count() or 2) + 2)


def reconcile_all(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
    db: DBClient,
    llm_judge: LLMJudge | None = None,
) -> list[ReconciledRow]:
    """Reconcile every input row 1:1. No dedup — every input line produces one output line.

    Rows are processed concurrently via ThreadPoolExecutor. Each worker shares
    the same DB connection (protected by DBClient._lock) and embedder cache,
    so embedding API calls — the main bottleneck — run in true parallel while
    DB round-trips are serialised automatically.
    """
    n = len(input_rows)
    logger.info("[reconcile_all] starting reconciliation of %d input rows (workers=%d)", n, _MAX_WORKERS)

    results: list[ReconciledRow | None] = [None] * n

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(reconcile_row, row, embedder, db): i
            for i, row in enumerate(input_rows)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            completed += 1
            if completed % 50 == 0 or completed == 1:
                logger.info("[reconcile_all] completed %d / %d", completed, n)
            try:
                results[i] = future.result()
            except Exception as exc:
                logger.error(
                    "[reconcile_all] error on row %d / %d (tema='%s'): %s",
                    i + 1,
                    n,
                    input_rows[i].get("tema", "(unknown)")[:50],
                    exc,
                    exc_info=True,
                )
                raise

    # Retry rows with low scores — also parallelised
    retry_candidates = [
        i for i, r in enumerate(results) if r is not None and r.match_score < MIN_ACCEPTABLE_SCORE
    ]
    if retry_candidates:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(retry_low_score, input_rows[i], results[i], embedder, db): i
                for i in retry_candidates
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    retried = future.result()
                    if retried and retried.match_score > results[i].match_score:
                        logger.info(
                            "[reconcile_all] retry improved row %d: %.0f%% -> %.0f%%",
                            i + 1,
                            results[i].match_score * 100,
                            retried.match_score * 100,
                        )
                        results[i] = retried
                except Exception as exc:
                    logger.warning("[reconcile_all] retry failed for row %d: %s", i + 1, exc)

    # All rows should be populated after parallel processing (exceptions re-raised above)
    final_results: list[ReconciledRow] = [r for r in results if r is not None]

    # Optional LLM judge step — validate/improve low-confidence matches
    if llm_judge is not None:
        from src.config import get_settings

        settings = get_settings()
        final_results = apply_llm_judge(
            final_results,
            llm_judge,
            db,
            threshold=settings.llm_judge_threshold,
            embedder=embedder,
        )

    # Sort by score descending (highest confidence first) — NO dedup
    final_results.sort(key=lambda r: r.match_score, reverse=True)
    logger.info(
        "[reconcile_all] reconciliation complete: %d rows in -> %d rows out",
        len(input_rows),
        len(final_results),
    )
    return final_results


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

        # No cross-level: only match tema-level DB entries with tema-level reconciled rows
        if not matched_row and stat_subtema is None:
            for key, row in matched_map.items():
                if key[0] == stat_tema and key[1] is None:
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
