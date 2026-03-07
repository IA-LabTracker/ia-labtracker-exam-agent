from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from src.normalize.normalizer import classify_color, normalize_tema_subtema
from src.retriever.hybrid_retriever import Candidate, retrieve_candidates
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder


MATCH_EXACT = "exato"
MATCH_FTS = "FTS"
MATCH_SEMANTIC = "semântico"
MATCH_NONE = "sem match"

# Minimum acceptable score — rows below this get retried with variants
MIN_ACCEPTABLE_SCORE = 0.50


@dataclass
class MatchInfo:
    """Tracks how a match was found and its confidence score."""

    method: str = MATCH_NONE  # exato, FTS, semântico, sem match
    score: float = 0.0  # 0.0 - 1.0 raw score
    label: str = ""  # human-readable label (quente/morno/frio)


def _classify_temperature(method: str, score: float) -> str:
    """Classify match quality as quente/morno/frio based on method + score.

    Score ranges (normalized 0-1):
      exact  → always 1.0 (100%)
      FTS    → 0.6-0.8 depending on match quality
      semantic → raw hybrid_score from DB (typically 0.35-0.95)
    """
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
    return "⚪ Sem match"


@dataclass
class ReconciledRow:
    input_tema: str
    input_equivalencia: str | None
    normalized_tema: str
    normalized_subtema: str | None
    num_questions: int
    match_method: str = MATCH_NONE
    match_score: float = 0.0
    match_label: str = "⚪ Sem match"
    cor_hex: str = "#22C55E"
    notes: str = ""


@dataclass
class ReverseRow:
    """Row for the reverse-coverage sheet.

    Each DB theme_stat is checked against the input temas to see
    if the user's input covers it (equivalência → tema direction).
    """

    db_tema: str
    db_subtema: str | None
    db_num_questions: int
    db_cor_hex: str
    matched_input: str | None
    similarity_score: float
    coverage_status: str  # "coberto", "parcial", "não coberto"
    notes: str = ""


# ---------------------------------------------------------------------------
# Step 1 – Resolve TEMA (tema vs tema)
# ---------------------------------------------------------------------------


def _resolve_tema(
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
            logger.debug("[_resolve_tema] exact match for tema='%s'", t)
            info = MatchInfo(MATCH_EXACT, 1.0, _classify_temperature(MATCH_EXACT, 1.0))
            return t, info

    # FTS fallback — try tema_raw, norm_tema, and equivalencia as hints
    for i, q in enumerate([tema_raw, norm_tema, equivalencia]):
        if not q:
            continue
        stat = db.find_best_theme_stat(q)
        if stat:
            resolved = stat.get("tema", "")
            # Score decreases per fallback level: direct=0.80, normalized=0.70, equivalencia=0.60
            fts_score = 0.80 - (i * 0.10)
            logger.debug(
                "[_resolve_tema] FTS match for tema='%s' (query='%s' score=%.2f)",
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
                "[_resolve_tema] semantic match tema='%s' (query='%s' score=%.3f)",
                resolved,
                search_text[:50],
                score,
            )
            info = MatchInfo(
                MATCH_SEMANTIC, score, _classify_temperature(MATCH_SEMANTIC, score)
            )
            return resolved, info

    logger.debug("[_resolve_tema] no match found for tema_raw='%s'", tema_raw[:50])
    return None, MatchInfo()


def _semantic_resolve_tema(
    query: str,
    db: DBClient,
    embedder: BaseEmbedder,
    threshold: float = 0.4,
) -> tuple[str | None, float]:
    """Use embedding similarity on theme_stats to find the best match.

    Tries the original query first; if score is below MIN_ACCEPTABLE_SCORE,
    retries with individual words/chunks from the query to find a better match.

    Returns (tema, score) or (None, 0.0).
    """
    best_tema, best_score = _single_semantic_search(query, db, embedder)

    # If score is acceptable, return immediately
    if best_score >= MIN_ACCEPTABLE_SCORE:
        return best_tema, best_score

    # Retry strategy: try individual meaningful chunks of the query
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    for i in range(len(words)):
        # Try progressively smaller chunks
        chunk = " ".join(words[i:])
        if chunk == query:
            continue
        tema, score = _single_semantic_search(chunk, db, embedder)
        if score > best_score:
            best_tema, best_score = tema, score
        if best_score >= MIN_ACCEPTABLE_SCORE:
            break

    if best_score >= threshold:
        return best_tema, best_score
    return None, 0.0


def _single_semantic_search(
    query: str,
    db: DBClient,
    embedder: BaseEmbedder,
) -> tuple[str | None, float]:
    """Run a single semantic search and return (tema, score)."""
    embedding = embedder.embed(query)
    results = db.semantic_search_theme_stats(
        query_embedding=embedding,
        query_text=query,
        top_k=3,
    )
    if results:
        best = results[0]
        score = best.get("hybrid_score", 0) or 0
        if score > 0:
            logger.debug(
                "[_single_semantic_search] tema='%s' score=%.3f query='%s'",
                best["tema"],
                score,
                query[:50],
            )
            return best["tema"], score
    return None, 0.0


# ---------------------------------------------------------------------------
# Step 2a – Tema-only: find tema-level stat
# ---------------------------------------------------------------------------


def _find_stat_tema_only(resolved_tema: str, db: DBClient) -> tuple[dict | None, int]:
    """For tema-only input: find the tema-level stat.

    Only returns tema-level rows (subtema IS NULL).
    If only subtema-level rows exist, aggregates their num_questions
    but returns the tema name (not a subtema entry).

    Returns (stat, num_questions).
    """
    stat = db.get_theme_stat(resolved_tema, None)
    if stat:
        return stat, stat["num_questions"]

    # No tema-level row — aggregate subtemas count but don't
    # return a subtema entry (tema input must match tema level)
    subtemas = db.get_subtemas_for_tema(resolved_tema)
    if subtemas:
        total = sum(s.get("num_questions", 0) for s in subtemas)
        # Build a synthetic tema-level stat
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


def _find_stat_with_subtema(
    resolved_tema: str,
    subtema_raw: str,
    norm_subtema: str | None,
    candidate_subtema: str | None,
    db: DBClient,
    embedder: BaseEmbedder | None = None,
    tema_match: MatchInfo | None = None,
) -> tuple[dict | None, MatchInfo]:
    """Comparison #2 — subtema vs subtema (under the already-resolved tema).

    Tries exact matches, then FTS, then semantic search, then best subtema.
    Returns (stat, match_info) — match_info combines tema + subtema resolution.
    """
    base = tema_match or MatchInfo()

    # Exact subtema matches
    for sub in [candidate_subtema, norm_subtema, subtema_raw]:
        if not sub:
            continue
        stat = db.get_theme_stat(resolved_tema, sub)
        if stat:
            # If tema was exact and subtema is exact → best possible
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
            # Combined score: tema match quality * subtema FTS quality
            fts_sub_score = 0.75 - (i * 0.10)  # raw=0.75, normalized=0.65
            score = min(base.score, fts_sub_score)
            method = MATCH_FTS
            info = MatchInfo(method, score, _classify_temperature(method, score))
            return stat, info

    # Semantic fallback — multiple subtema-only query strategies via embeddings
    if embedder and subtema_raw:
        # Try multiple queries: combined, subtema-only, significant words
        semantic_queries = [
            f"{resolved_tema} {subtema_raw}",
            subtema_raw,
        ]
        if norm_subtema and norm_subtema != subtema_raw:
            semantic_queries.append(norm_subtema)
        # Add significant words from subtema (>3 chars) combined with tema
        words = [w for w in subtema_raw.split() if len(w) > 3]
        for w in words:
            semantic_queries.append(f"{resolved_tema} {w}")

        best_result = None
        best_score = 0.0

        # Batch embed all queries at once for speed
        embeddings = embedder.embed_batch(semantic_queries)

        for query, embedding in zip(semantic_queries, embeddings):
            results = db.semantic_search_theme_stats(
                query_embedding=embedding,
                query_text=query,
                top_k=5,
            )
            for r in results:
                if r.get("tema", "").lower() == resolved_tema.lower() and r.get("subtema"):
                    score = r.get("hybrid_score", 0) or 0
                    if score > best_score and score >= 0.35:
                        best_result = r
                        best_score = score

        if best_result:
            logger.debug(
                "[_find_stat_with_subtema] semantic match subtema='%s' score=%.3f",
                best_result["subtema"],
                best_score,
            )
            info = MatchInfo(
                MATCH_SEMANTIC,
                best_score,
                _classify_temperature(MATCH_SEMANTIC, best_score),
            )
            return best_result, info

    # No subtema match found at all.
    # DO NOT fallback to tema-level or guess a subtema — subtema must match subtema.
    # Return None so the row shows as unmatched at subtema level.
    return None, MatchInfo()


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
    resolved_tema, tema_match = _resolve_tema(
        tema_raw, norm_tema, candidate_tema, equivalencia, db, embedder
    )
    final_tema = resolved_tema or norm_tema
    match_info = tema_match

    # --- Comparison #2: subtema vs subtema (or tema-only stat) ---
    stat: dict | None = None
    num_questions = 0
    if resolved_tema:
        if has_subtema:
            stat, match_info = _find_stat_with_subtema(
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
            stat, num_questions = _find_stat_tema_only(resolved_tema, db)

    notes_parts = []
    equivalencia_out = equivalencia
    final_subtema = norm_subtema

    if stat:
        _, cor_hex = classify_color(num_questions)

        # final_tema always comes from _resolve_tema (comparison #1),
        # final_subtema comes from the stat (comparison #2)
        stat_subtema = stat.get("subtema")
        if stat_subtema:
            final_subtema = stat_subtema

        # Build equivalencia at the same level as input
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


def _retry_low_score(
    row: dict[str, Any],
    original: ReconciledRow,
    embedder: BaseEmbedder,
    db: DBClient,
) -> ReconciledRow | None:
    """Lightweight retry — only re-does the semantic search with variant queries.

    Instead of calling full reconcile_row (which re-embeds, re-retrieves candidates),
    this batch-embeds all variant queries and picks the best semantic match.
    """
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

    # Batch embed all variants at once (single model call)
    embeddings = embedder.embed_batch(variants)

    best_tema = None
    best_score = original.match_score
    best_query = None

    for query, emb in zip(variants, embeddings):
        results = db.semantic_search_theme_stats(
            query_embedding=emb, query_text=query, top_k=3,
        )
        if results and (results[0].get("hybrid_score", 0) or 0) > best_score:
            best_score = results[0]["hybrid_score"]
            best_tema = results[0]["tema"]
            best_query = query
            if best_score >= MIN_ACCEPTABLE_SCORE:
                break

    if not best_tema or best_score <= original.match_score:
        return None

    # Build a lightweight result using the improved tema match
    stat, num_q = _find_stat_tema_only(best_tema, db)
    if not stat:
        return None

    _, cor_hex = classify_color(num_q)
    info = MatchInfo(
        MATCH_SEMANTIC, best_score,
        _classify_temperature(MATCH_SEMANTIC, best_score),
    )
    input_display = f"{tema_raw} | {subtema_raw}" if has_subtema else tema_raw

    return ReconciledRow(
        input_tema=input_display,
        input_equivalencia=equivalencia,
        normalized_tema=best_tema,
        normalized_subtema=original.normalized_subtema,
        num_questions=num_q,
        match_method=info.method,
        match_score=info.score,
        match_label=info.label,
        cor_hex=cor_hex,
        notes=f"Retry via '{best_query[:40]}'; "
              f"Fonte: {stat.get('institution', 'N/A')}",
    )


def reconcile_all(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
    db: DBClient,
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
            retried = _retry_low_score(input_rows[i], r, embedder, db)
            if retried and retried.match_score > r.match_score:
                logger.info(
                    "[reconcile_all] retry improved row %d: %.0f%% -> %.0f%%",
                    i + 1,
                    r.match_score * 100,
                    retried.match_score * 100,
                )
                results[i] = retried

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
    """For each theme_stat in the DB (equivalência), check if any reconciled
    result from Table 1 covers it.

    This inverts the mapping from reconcile_all:
      Table 1: Manchester (input) → equivalência (DB match)
      Table 2: equivalência (DB) → which Manchester rows matched it?

    Uses the already-computed match scores from reconcile_all, plus a
    semantic fallback for DB entries that weren't directly matched.
    """
    logger.info("[reverse_coverage] starting reverse coverage analysis")

    all_stats = db.get_all_theme_stats()
    if not all_stats:
        logger.warning("[reverse_coverage] no theme_stats found in DB")
        return []

    # Build a mapping: (normalized_tema, normalized_subtema) → best reconciled row
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

    results = []
    unmatched_stats = []
    unmatched_indices = []

    for stat in all_stats:
        stat_tema = stat["tema"].lower()
        stat_subtema = (stat.get("subtema") or "").lower() or None

        # Try exact key match first
        matched_row = matched_map.get((stat_tema, stat_subtema))

        # Also try tema-only match if subtema didn't match
        if not matched_row and stat_subtema is None:
            # Check if any reconciled row matched this tema (ignoring subtema)
            for key, row in matched_map.items():
                if key[0] == stat_tema:
                    if matched_row is None or row.match_score > matched_row.match_score:
                        matched_row = row

        if matched_row and matched_row.match_score >= partial_threshold:
            score = matched_row.match_score
            status = "coberto" if score >= coverage_threshold else "parcial"
            notes = f"Match: {matched_row.input_tema} (score={score:.0%})"
            results.append(
                ReverseRow(
                    db_tema=stat["tema"],
                    db_subtema=stat.get("subtema"),
                    db_num_questions=stat["num_questions"],
                    db_cor_hex=stat.get("cor_hex", "#22C55E"),
                    matched_input=matched_row.input_tema,
                    similarity_score=score,
                    coverage_status=status,
                    notes=notes,
                )
            )
        else:
            # Will try semantic fallback
            unmatched_stats.append(stat)
            unmatched_indices.append(len(results))
            results.append(None)  # placeholder

    # Semantic fallback for DB entries not directly matched by reconcile_all
    if unmatched_stats and reconciled_embeddings:
        stat_texts = []
        for s in unmatched_stats:
            text = s["tema"]
            if s.get("subtema"):
                text += f" {s['subtema']}"
            stat_texts.append(text)
        stat_embeddings = embedder.embed_batch(stat_texts)

        # Vectorized cosine similarity via numpy (much faster than Python loops)
        stat_matrix = np.array(stat_embeddings, dtype=np.float32)
        rec_matrix = np.array(reconciled_embeddings, dtype=np.float32)
        # Normalize rows
        stat_norms = np.linalg.norm(stat_matrix, axis=1, keepdims=True)
        rec_norms = np.linalg.norm(rec_matrix, axis=1, keepdims=True)
        stat_norms[stat_norms == 0] = 1.0
        rec_norms[rec_norms == 0] = 1.0
        # Cosine similarity matrix: (unmatched_stats x reconciled)
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

    # Remove any None placeholders (shouldn't happen, but safety)
    results = [r for r in results if r is not None]

    # Sort by similarity descending (highest first)
    results.sort(key=lambda r: r.similarity_score, reverse=True)

    covered = sum(1 for r in results if r.coverage_status == "coberto")
    partial = sum(1 for r in results if r.coverage_status == "parcial")
    uncovered = sum(1 for r in results if r.coverage_status == "não coberto")
    logger.info(
        "[reverse_coverage] done: %d coberto, %d parcial, %d não coberto (total=%d)",
        covered,
        partial,
        uncovered,
        len(results),
    )
    return results
