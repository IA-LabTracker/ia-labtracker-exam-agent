"""LLM judge integration for the reconciliation pipeline.

Applies LLM-as-Judge to validate/improve low-confidence matches.
For each low-confidence row, performs independent DB searches (semantic + FTS)
to find alternative candidates that the initial hybrid search may have missed,
then sends everything to the LLM to pick the best match.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.normalize.normalizer import classify_color
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder
    from src.llm.judge import LLMJudge

from src.aggregator.models import (
    MATCH_LLM,
    MATCH_NONE,
    ReconciledRow,
    _classify_temperature,
)


def _search_alternative_candidates(
    input_tema: str,
    input_subtema: str,
    current_match_tema: str,
    embedder: BaseEmbedder,
    db: DBClient,
) -> list[dict]:
    """Search the DB for alternative candidates using the original input text.

    Does semantic search + FTS using the user's INPUT (not the current match),
    so the LLM sees candidates the hybrid search may have missed.
    Returns deduplicated list of candidate stats.
    """
    seen_keys: set[tuple[str, str | None]] = set()
    candidates: list[dict] = []

    def _add(stat: dict) -> None:
        key = (stat["tema"].lower(), (stat.get("subtema") or "").lower() or None)
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(stat)

    # Strategy 1: Semantic search using the original input text
    search_queries = [input_tema]
    if input_subtema:
        search_queries.append(f"{input_tema} {input_subtema}")
        search_queries.append(input_subtema)

    embeddings = embedder.embed_batch(search_queries)
    for q, emb in zip(search_queries, embeddings):
        results = db.semantic_search_theme_stats(
            query_embedding=emb, query_text=q, top_k=5,
        )
        for r in results:
            _add(r)

    # Strategy 2: FTS search using the input text
    for q in [input_tema, input_subtema]:
        if not q:
            continue
        stat = db.find_best_theme_stat(q)
        if stat:
            _add(stat)

    # Strategy 3: Also search subtemas under potential tema matches
    for c in list(candidates):
        subtemas = db.get_subtemas_for_tema(c["tema"])
        for s in subtemas[:3]:  # top 3 subtemas by num_questions
            _add(s)

    # Strategy 4: If current match tema differs from input, search around it too
    if current_match_tema.lower() != input_tema.lower():
        subtemas = db.get_subtemas_for_tema(current_match_tema)
        for s in subtemas[:3]:
            _add(s)

    return candidates[:10]  # Cap at 10 to keep prompt reasonable


def apply_llm_judge(
    results: list[ReconciledRow],
    llm_judge: LLMJudge,
    db: DBClient,
    threshold: float = 0.60,
    embedder: BaseEmbedder | None = None,
) -> list[ReconciledRow]:
    """Send low-confidence rows to the LLM judge with real DB candidates.

    For each row below threshold:
    1. Searches the DB independently (semantic + FTS) using the original input
    2. Collects real alternative candidates the hybrid search may have missed
    3. Sends current match + alternatives to the LLM
    4. LLM picks the best match or confirms the current one
    """
    review_indices = []
    items = []
    candidates_list = []

    for i, r in enumerate(results):
        if r.match_method == MATCH_NONE or r.match_score >= threshold:
            continue
        review_indices.append(i)

        parts = r.input_tema.split(" | ", 1)
        input_tema = parts[0]
        input_subtema = parts[1] if len(parts) > 1 else ""

        current_match = r.normalized_tema
        if r.normalized_subtema:
            current_match += f" | {r.normalized_subtema}"

        items.append(
            {
                "input_tema": input_tema,
                "input_subtema": input_subtema,
                "current_match": current_match,
                "current_score": r.match_score,
            }
        )

        # Search DB independently for real alternative candidates
        if embedder:
            db_candidates = _search_alternative_candidates(
                input_tema, input_subtema, r.normalized_tema, embedder, db,
            )
        else:
            # Fallback: use FTS + subtemas if no embedder available
            db_candidates = []
            stat = db.find_best_theme_stat(input_tema)
            if stat:
                db_candidates.append(stat)
            subtemas = db.get_subtemas_for_tema(r.normalized_tema)
            db_candidates.extend(subtemas[:5])

        candidates_list.append(db_candidates)

    if not items:
        logger.info(
            "[LLM Judge] no rows below threshold %.0f%% to review", threshold * 100
        )
        return results

    logger.info(
        "[LLM Judge] sending %d low-confidence rows for review (threshold=%.0f%%)",
        len(items),
        threshold * 100,
    )

    verdicts = llm_judge.judge_batch(items, candidates_list)

    improved = 0
    for idx, verdict in zip(review_indices, verdicts):
        r = results[idx]

        if verdict.is_equivalent and verdict.confidence > r.match_score:
            # LLM confirmed the current match with higher confidence — preserve existing per-institution counts
            results[idx] = ReconciledRow(
                input_tema=r.input_tema,
                input_equivalencia=r.input_equivalencia,
                normalized_tema=r.normalized_tema,
                normalized_subtema=r.normalized_subtema,
                questions_by_institution=r.questions_by_institution,
                match_method=MATCH_LLM,
                match_score=verdict.confidence,
                match_label=_classify_temperature(MATCH_LLM, verdict.confidence),
                cor_hex=r.cor_hex,
                notes=f"{r.notes}; LLM: {verdict.reasoning}",
            )
            improved += 1

        elif not verdict.is_equivalent and verdict.suggested_match:
            # LLM found a better match — look it up in DB
            suggested = verdict.suggested_match
            # Try exact match first, then check if it's a "tema | subtema" format
            stat = None
            if " | " in suggested:
                s_parts = [p.strip() for p in suggested.split(" | ", 1)]
                stat = db.get_theme_stat(s_parts[0], s_parts[1])
                if not stat:
                    stat = db.get_theme_stat(s_parts[0], None)
            else:
                stat = db.get_theme_stat(suggested, None)
                if not stat:
                    # Try FTS as fallback to find the suggested match
                    stat = db.find_best_theme_stat(suggested)

            if stat:
                suggested_tema = stat["tema"]
                suggested_subtema = stat.get("subtema")
                equiv = f"{suggested_tema} | {suggested_subtema}" if suggested_subtema else suggested_tema
                qbi = db.get_questions_by_institution(suggested_tema, suggested_subtema)
                total_q = sum(qbi.values()) if qbi else 0
                _, cor_hex = classify_color(total_q)

                results[idx] = ReconciledRow(
                    input_tema=r.input_tema,
                    input_equivalencia=equiv,
                    normalized_tema=suggested_tema,
                    normalized_subtema=suggested_subtema or r.normalized_subtema,
                    questions_by_institution=qbi,
                    match_method=MATCH_LLM,
                    match_score=verdict.confidence,
                    match_label=_classify_temperature(MATCH_LLM, verdict.confidence),
                    cor_hex=cor_hex,
                    notes=f"Match anterior: {r.input_equivalencia or r.normalized_tema} (score={r.match_score:.0%}); LLM sugeriu: {equiv}; {verdict.reasoning}",
                )
                improved += 1

    logger.info("[LLM Judge] improved %d / %d reviewed rows", improved, len(items))
    return results
