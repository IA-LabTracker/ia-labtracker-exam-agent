"""LLM judge integration for the reconciliation pipeline.

Extracted from consolidate.py to keep modules focused.
Applies LLM-as-Judge to validate/improve low-confidence matches.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.normalize.normalizer import classify_color
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.llm.judge import LLMJudge

from src.aggregator.models import (
    MATCH_LLM,
    MATCH_NONE,
    MatchInfo,
    ReconciledRow,
    _classify_temperature,
)


def apply_llm_judge(
    results: list[ReconciledRow],
    input_rows: list[dict[str, Any]],
    llm_judge: LLMJudge,
    db: DBClient,
    threshold: float = 0.60,
) -> list[ReconciledRow]:
    """Send low-confidence rows to the LLM judge for validation/improvement.

    Only rows with score < threshold and a non-empty match are sent.
    """
    review_indices = []
    items = []
    candidates_list = []

    # Fetch DB stats once (not per row — was the #1 perf bug)
    all_stats = db.get_all_theme_stats()

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

        # Pick top candidates by word overlap (using cached all_stats)
        search_tema = r.normalized_tema.lower()
        search_words = set(search_tema.split())
        scored = []
        for s in all_stats:
            common = search_words & set(s["tema"].lower().split())
            scored.append((len(common), s))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates_list.append([s for _, s in scored[:5]])

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
            results[idx] = ReconciledRow(
                input_tema=r.input_tema,
                input_equivalencia=r.input_equivalencia,
                normalized_tema=r.normalized_tema,
                normalized_subtema=r.normalized_subtema,
                num_questions=r.num_questions,
                match_method=MATCH_LLM,
                match_score=verdict.confidence,
                match_label=_classify_temperature(MATCH_LLM, verdict.confidence),
                cor_hex=r.cor_hex,
                notes=f"{r.notes}; LLM: {verdict.reasoning}",
            )
            improved += 1

        elif not verdict.is_equivalent and verdict.suggested_match:
            suggested = verdict.suggested_match
            stat = db.get_theme_stat(suggested, None)
            if stat:
                num_q = stat["num_questions"]
                _, cor_hex = classify_color(num_q)
                results[idx] = ReconciledRow(
                    input_tema=r.input_tema,
                    input_equivalencia=suggested,
                    normalized_tema=suggested,
                    normalized_subtema=r.normalized_subtema,
                    num_questions=num_q,
                    match_method=MATCH_LLM,
                    match_score=verdict.confidence,
                    match_label=_classify_temperature(MATCH_LLM, verdict.confidence),
                    cor_hex=cor_hex,
                    notes=f"LLM sugeriu: {suggested}; {verdict.reasoning}",
                )
                improved += 1

    logger.info("[LLM Judge] improved %d / %d reviewed rows", improved, len(items))
    return results
