"""Tema and subtema matching strategies.

Extracted from consolidate.py to keep each module under ~350 lines.
Contains all matching logic: exact, FTS, and semantic search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.normalize.normalizer import classify_color
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder

from src.aggregator.models import (
    MATCH_EXACT,
    MATCH_FTS,
    MATCH_NONE,
    MATCH_SEMANTIC,
    MIN_ACCEPTABLE_SCORE,
    MatchInfo,
    ReconciledRow,
    _classify_temperature,
)

# ---------------------------------------------------------------------------
# Step 1 – Resolve TEMA (tema vs tema)
# ---------------------------------------------------------------------------


def resolve_tema(
    tema_raw: str,
    norm_tema: str,
    candidate_tema: str | None,
    equivalencia: str | None,
    db: DBClient,
    embedder: BaseEmbedder | None = None,
) -> tuple[str | None, MatchInfo]:
    """Comparison #1 — tema vs tema.

    Tries exact match, then FTS, then semantic search via embeddings.
    Returns (resolved_tema, match_info).
    """
    for t in [candidate_tema, norm_tema, tema_raw]:
        if not t:
            continue
        if db.get_theme_stat(t, None) or db.get_subtemas_for_tema(t):
            logger.debug("[resolve_tema] exact match for tema='%s'", t)
            info = MatchInfo(MATCH_EXACT, 1.0, _classify_temperature(MATCH_EXACT, 1.0))
            return t, info

    # FTS fallback — try tema_raw, norm_tema, and equivalencia as hints
    for i, q in enumerate([tema_raw, norm_tema, equivalencia]):
        if not q:
            continue
        stat = db.find_best_theme_stat(q)
        if stat:
            resolved = stat.get("tema", "")
            fts_score = 0.80 - (i * 0.10)
            logger.debug(
                "[resolve_tema] FTS match for tema='%s' (query='%s' score=%.2f)",
                resolved,
                q[:50],
                fts_score,
            )
            info = MatchInfo(
                MATCH_FTS, fts_score, _classify_temperature(MATCH_FTS, fts_score)
            )
            return resolved, info

    # Semantic fallback — use embeddings on theme_stats
    if embedder:
        search_text = equivalencia or norm_tema or tema_raw
        resolved, score = _semantic_resolve_tema(search_text, db, embedder)
        if resolved:
            logger.debug(
                "[resolve_tema] semantic match tema='%s' (query='%s' score=%.3f)",
                resolved,
                search_text[:50],
                score,
            )
            info = MatchInfo(
                MATCH_SEMANTIC, score, _classify_temperature(MATCH_SEMANTIC, score)
            )
            return resolved, info

    logger.debug("[resolve_tema] no match found for tema_raw='%s'", tema_raw[:50])
    return None, MatchInfo()


def _semantic_resolve_tema(
    query: str,
    db: DBClient,
    embedder: BaseEmbedder,
    threshold: float = 0.4,
) -> tuple[str | None, float]:
    """Use embedding similarity on theme_stats to find the best match.

    Batch-embeds the query + word chunks in a single call for speed.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    queries = [query]
    for i in range(len(words)):
        chunk = " ".join(words[i:])
        if chunk != query:
            queries.append(chunk)

    # Batch embed all queries at once (single model call)
    embeddings = embedder.embed_batch(queries)

    best_tema = None
    best_score = 0.0

    for q, emb in zip(queries, embeddings):
        results = db.semantic_search_theme_stats(
            query_embedding=emb,
            query_text=q,
            top_k=3,
        )
        if results:
            score = results[0].get("hybrid_score", 0) or 0
            if score > best_score:
                best_tema = results[0]["tema"]
                best_score = score
                if best_score >= MIN_ACCEPTABLE_SCORE:
                    break

    if best_score >= threshold:
        return best_tema, best_score
    return None, 0.0


# ---------------------------------------------------------------------------
# Step 2a – Tema-only: find tema-level stat
# ---------------------------------------------------------------------------


def find_stat_tema_only(resolved_tema: str, db: DBClient) -> tuple[dict | None, int]:
    """For tema-only input: find the tema-level stat.

    Only returns tema-level rows (subtema IS NULL).
    If only subtema-level rows exist, aggregates their num_questions.
    """
    stat = db.get_theme_stat(resolved_tema, None)
    if stat:
        return stat, stat["num_questions"]

    subtemas = db.get_subtemas_for_tema(resolved_tema)
    if subtemas:
        total = sum(s.get("num_questions", 0) for s in subtemas)
        synthetic = {
            **subtemas[0],
            "subtema": None,
            "category": "tema",
            "num_questions": total,
        }
        return synthetic, total

    return None, 0


# ---------------------------------------------------------------------------
# Step 2b – Tema+subtema: find subtema-level stat under resolved tema
# ---------------------------------------------------------------------------


def find_stat_with_subtema(
    resolved_tema: str,
    subtema_raw: str,
    norm_subtema: str | None,
    candidate_subtema: str | None,
    db: DBClient,
    embedder: BaseEmbedder | None = None,
    tema_match: MatchInfo | None = None,
) -> tuple[dict | None, MatchInfo]:
    """Comparison #2 — subtema vs subtema (under the already-resolved tema)."""
    base = tema_match or MatchInfo()

    # Exact subtema matches
    for sub in [candidate_subtema, norm_subtema, subtema_raw]:
        if not sub:
            continue
        stat = db.get_theme_stat(resolved_tema, sub)
        if stat:
            if base.method == MATCH_EXACT:
                info = MatchInfo(
                    MATCH_EXACT, 1.0, _classify_temperature(MATCH_EXACT, 1.0)
                )
            else:
                info = MatchInfo(base.method, base.score, base.label)
            return stat, info

    # FTS — only accept results whose tema matches the resolved tema
    for i, sub in enumerate([subtema_raw, norm_subtema]):
        if not sub:
            continue
        stat = db.find_best_theme_stat(f"{resolved_tema} | {sub}")
        if stat and stat.get("tema", "").lower() == resolved_tema.lower():
            fts_sub_score = 0.75 - (i * 0.10)
            score = min(base.score, fts_sub_score)
            info = MatchInfo(MATCH_FTS, score, _classify_temperature(MATCH_FTS, score))
            return stat, info

    # Semantic fallback — capped at 3 query variants to avoid DB query explosion.
    # Full query + subtema alone + one focused variant (normalized form or longest keyword).
    # Batch-embedded in one call; DB queries run only until a good score is found.
    if embedder and subtema_raw:
        semantic_queries = [
            f"{resolved_tema} {subtema_raw}",
            subtema_raw,
        ]
        if norm_subtema and norm_subtema != subtema_raw:
            semantic_queries.append(norm_subtema)
        else:
            # Use the longest keyword as a targeted fallback (one word, not all words)
            words = sorted(
                [w for w in subtema_raw.split() if len(w) > 3], key=len, reverse=True
            )
            if words:
                semantic_queries.append(f"{resolved_tema} {words[0]}")

        best_result = None
        best_score = 0.0

        # Batch embed all queries at once (single model call)
        embeddings = embedder.embed_batch(semantic_queries)

        for query, embedding in zip(semantic_queries, embeddings):
            results = db.semantic_search_theme_stats(
                query_embedding=embedding,
                query_text=query,
                top_k=5,
            )
            for r in results:
                if r.get("tema", "").lower() == resolved_tema.lower() and r.get(
                    "subtema"
                ):
                    score = r.get("hybrid_score", 0) or 0
                    if score > best_score and score >= 0.35:
                        best_result = r
                        best_score = score
            # Early exit: already found a strong match — skip remaining queries
            if best_score >= MIN_ACCEPTABLE_SCORE:
                break

        if best_result:
            logger.debug(
                "[find_stat_with_subtema] semantic match subtema='%s' score=%.3f",
                best_result["subtema"],
                best_score,
            )
            info = MatchInfo(
                MATCH_SEMANTIC,
                best_score,
                _classify_temperature(MATCH_SEMANTIC, best_score),
            )
            return best_result, info

    return None, MatchInfo()


# ---------------------------------------------------------------------------
# Retry low-score rows with variant queries
# ---------------------------------------------------------------------------


def retry_low_score(
    row: dict,
    original: ReconciledRow,
    embedder: BaseEmbedder,
    db: DBClient,
) -> ReconciledRow | None:
    """Lightweight retry — batch-embeds variant queries and picks best semantic match."""
    tema_raw = str(row.get("tema", "")).strip()
    subtema_raw = row.get("subtema")
    equivalencia = row.get("equivalencia")
    has_subtema = subtema_raw and str(subtema_raw).strip().lower() != "nan"

    variants = []
    if equivalencia and str(equivalencia).strip().lower() != "nan":
        variants.append(str(equivalencia).strip())
    if has_subtema:
        sub = str(subtema_raw).strip()
        variants.append(f"{tema_raw} {sub}")
        for word in sub.split():
            if len(word) > 3:
                variants.append(f"{tema_raw} {word}")
    if not variants:
        return None

    embeddings = embedder.embed_batch(variants)

    best_tema = None
    best_score = original.match_score
    best_query = None

    for query, emb in zip(variants, embeddings):
        results = db.semantic_search_theme_stats(
            query_embedding=emb,
            query_text=query,
            top_k=3,
        )
        if results and (results[0].get("hybrid_score", 0) or 0) > best_score:
            best_score = results[0]["hybrid_score"]
            best_tema = results[0]["tema"]
            best_query = query
            if best_score >= MIN_ACCEPTABLE_SCORE:
                break

    if not best_tema or best_score <= original.match_score:
        return None

    stat, num_q = find_stat_tema_only(best_tema, db)
    if not stat:
        return None

    qbi = db.get_questions_by_institution(best_tema)
    total_q = sum(qbi.values()) if qbi else num_q
    _, cor_hex = classify_color(total_q)
    info = MatchInfo(
        MATCH_SEMANTIC,
        best_score,
        _classify_temperature(MATCH_SEMANTIC, best_score),
    )
    input_display = f"{tema_raw} | {subtema_raw}" if has_subtema else tema_raw

    return ReconciledRow(
        input_tema=input_display,
        input_equivalencia=equivalencia,
        normalized_tema=best_tema,
        normalized_subtema=original.normalized_subtema,
        questions_by_institution=qbi,
        match_method=info.method,
        match_score=info.score,
        match_label=info.label,
        cor_hex=cor_hex,
        notes=f"Retry via '{best_query[:40]}'; "
        f"Fonte: {stat.get('institution', 'N/A')}",
    )
