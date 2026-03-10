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
    equivalencia: str,
    current_match_tema: str,
    embedder: BaseEmbedder,
    db: DBClient,
) -> list[dict]:
    """Search the DB for alternative candidates using the original input text.

    Does semantic search + FTS using the user's INPUT (not the current match),
    so the LLM sees candidates the hybrid search may have missed.
    Returns deduplicated list of candidate stats, prioritizing specific matches.
    """
    seen_keys: set[tuple[str, str | None]] = set()
    candidates: list[dict] = []

    def _add(stat: dict) -> None:
        key = (stat["tema"].lower(), (stat.get("subtema") or "").lower() or None)
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(stat)

    # Strategy 1: Semantic search using the original input text + equivalencia
    search_queries = [input_tema]
    if input_subtema:
        search_queries.append(f"{input_tema} {input_subtema}")
        search_queries.append(input_subtema)
    if equivalencia and equivalencia.lower() not in (input_tema.lower(), input_subtema.lower() if input_subtema else ""):
        search_queries.append(equivalencia)

    embeddings = embedder.embed_batch(search_queries)
    for q, emb in zip(search_queries, embeddings):
        results = db.semantic_search_theme_stats(
            query_embedding=emb,
            query_text=q,
            top_k=7,
        )
        for r in results:
            _add(r)

    # Strategy 2: FTS search using the input text + equivalencia
    for q in [input_tema, input_subtema, equivalencia]:
        if not q:
            continue
        stat = db.find_best_theme_stat(q)
        if stat:
            _add(stat)

    # Strategy 3: Search subtemas for the top 4 candidate temas.
    # Fetch up to 8 subtemas so the LLM sees more specific options
    # beyond generic "Aspectos Gerais" entries.
    for c in list(candidates)[:4]:
        subtemas = db.get_subtemas_for_tema(c["tema"])
        for s in subtemas[:8]:
            _add(s)

    # Strategy 4: If current match tema differs from input, search around it too
    if current_match_tema and current_match_tema.lower() != input_tema.lower():
        subtemas = db.get_subtemas_for_tema(current_match_tema)
        for s in subtemas[:8]:
            _add(s)

    # No cross-level: filter candidates to match the input level.
    # If input has subtema, only show subtema-level candidates.
    # If input is tema-only, only show tema-level candidates.
    has_subtema = bool(input_subtema and input_subtema.strip())
    if has_subtema:
        candidates = [c for c in candidates if c.get("subtema")]
    else:
        candidates = [c for c in candidates if not c.get("subtema")]

    # Sort by num_questions descending
    candidates.sort(
        key=lambda c: c.get("num_questions", 0),
        reverse=True,
    )

    return candidates[:15]


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
        # Skip only rows that already have a good match (above threshold)
        # MATCH_NONE rows should also be reviewed so the LLM can try to find something
        if r.match_score >= threshold and r.match_method != MATCH_NONE:
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
                "equivalencia": r.input_equivalencia or "",
                "current_match": current_match,
                "current_score": r.match_score,
                "match_method": r.match_method,
            }
        )

        # Search DB independently for real alternative candidates
        if embedder:
            db_candidates = _search_alternative_candidates(
                input_tema,
                input_subtema,
                r.input_equivalencia or "",
                r.normalized_tema,
                embedder,
                db,
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

        if verdict.is_equivalent:
            # LLM confirmed the current match.
            # Use the HIGHER of LLM confidence and current score.
            # But cap LLM confidence: if the LLM says 0.95 but the original method
            # was very weak, trust the LLM but don't inflate beyond what's reasonable.
            new_score = max(verdict.confidence, r.match_score)

            # Ensure normalized_subtema comes from theme_stats, not from input normalization.
            # No cross-level: do NOT fall back to tema-level stat when subtema not found.
            confirmed_subtema = r.normalized_subtema
            if r.normalized_tema:
                db_stat = db.get_theme_stat(r.normalized_tema, r.normalized_subtema)
                if db_stat:
                    confirmed_subtema = db_stat.get("subtema")

            # Derive equivalencia from DB-confirmed data
            confirmed_equiv = r.input_equivalencia
            if not confirmed_equiv and r.normalized_tema:
                confirmed_equiv = (
                    f"{r.normalized_tema} | {confirmed_subtema}"
                    if confirmed_subtema
                    else r.normalized_tema
                )

            # Refresh institution counts using confirmed DB data
            qbi = r.questions_by_institution
            if r.normalized_tema:
                fresh_qbi = db.get_questions_by_institution(
                    r.normalized_tema, confirmed_subtema
                )
                if fresh_qbi:
                    qbi = fresh_qbi

            total_q = sum(qbi.values()) if qbi else 0
            _, cor_hex = classify_color(total_q) if total_q > 0 else ("azul", r.cor_hex)

            results[idx] = ReconciledRow(
                input_tema=r.input_tema,
                input_equivalencia=confirmed_equiv,
                normalized_tema=r.normalized_tema,
                normalized_subtema=confirmed_subtema,
                questions_by_institution=qbi,
                match_method=MATCH_LLM,
                match_score=new_score,
                match_label=_classify_temperature(MATCH_LLM, new_score),
                cor_hex=cor_hex,
                notes=f"{r.notes}; LLM confirmado (conf={verdict.confidence:.0%}): {verdict.reasoning}",
            )
            improved += 1

        elif not verdict.is_equivalent and verdict.suggested_match:
            # LLM found a better match — look it up in DB.
            # Clean the raw string first: LLMs sometimes include prompt annotations
            # like "[MATCH ATUAL]" prefixes or "(N questoes)" suffixes.
            from src.llm.judge import _clean_suggested_match

            suggested = _clean_suggested_match(verdict.suggested_match)
            # Try exact match first, then check if it's a "tema | subtema" format
            # No cross-level: respect the level of the LLM suggestion.
            # If LLM suggests "tema | subtema", only accept subtema-level stat.
            # If LLM suggests tema-only, only accept tema-level stat.
            stat = None
            if " | " in suggested:
                s_parts = [p.strip() for p in suggested.split(" | ", 1)]
                stat = db.get_theme_stat(s_parts[0], s_parts[1])
                # Do NOT fall back to tema-only if subtema not found
            else:
                stat = db.get_theme_stat(suggested, None)
                if not stat:
                    stat = db.find_best_theme_stat(suggested)

            # No cross-level: if input had subtema, only accept stats with subtema
            has_input_subtema = " | " in r.input_tema
            if stat and has_input_subtema and not stat.get("subtema"):
                logger.debug(
                    "[LLM Judge] rejecting tema-only suggestion %r for subtema-level input %r",
                    suggested,
                    r.input_tema,
                )
                stat = None

            if stat:
                suggested_tema = stat["tema"]
                suggested_subtema = stat.get("subtema")
                equiv = (
                    f"{suggested_tema} | {suggested_subtema}"
                    if suggested_subtema
                    else suggested_tema
                )
                qbi = db.get_questions_by_institution(suggested_tema, suggested_subtema)
                total_q = sum(qbi.values()) if qbi else 0
                _, cor_hex = classify_color(total_q)

                results[idx] = ReconciledRow(
                    input_tema=r.input_tema,
                    input_equivalencia=equiv,
                    normalized_tema=suggested_tema,
                    normalized_subtema=suggested_subtema,  # None when tema-only DB entry; never fall back to input subtema
                    questions_by_institution=qbi,
                    match_method=MATCH_LLM,
                    match_score=verdict.confidence,
                    match_label=_classify_temperature(MATCH_LLM, verdict.confidence),
                    cor_hex=cor_hex,
                    notes=f"Match anterior: {r.input_equivalencia or r.normalized_tema} (score={r.match_score:.0%}); LLM sugeriu: {equiv}; {verdict.reasoning}",
                )
                improved += 1
            else:
                logger.warning(
                    "[LLM Judge] suggested_match %r not found in DB for input %r — keeping original row",
                    suggested,
                    r.input_tema,
                )

        elif not verdict.is_equivalent and not verdict.suggested_match:
            # LLM explicitly rejected the current match and has no alternative.
            # Downgrade the score to signal this is a bad match.
            if r.match_score > 0.3:
                downgraded_score = min(r.match_score, 0.30)
                results[idx] = ReconciledRow(
                    input_tema=r.input_tema,
                    input_equivalencia=r.input_equivalencia,
                    normalized_tema=r.normalized_tema,
                    normalized_subtema=r.normalized_subtema,
                    questions_by_institution=r.questions_by_institution,
                    match_method=MATCH_LLM,
                    match_score=downgraded_score,
                    match_label=_classify_temperature(MATCH_LLM, downgraded_score),
                    cor_hex=r.cor_hex,
                    notes=f"{r.notes}; LLM rejeitou match (conf={verdict.confidence:.0%}): {verdict.reasoning}",
                )
                logger.info(
                    "[LLM Judge] REJECTED match for %r: %s → score downgraded %.0f%% → %.0f%%",
                    r.input_tema,
                    verdict.reasoning[:80],
                    r.match_score * 100,
                    downgraded_score * 100,
                )

    logger.info("[LLM Judge] improved %d / %d reviewed rows", improved, len(items))
    return results
