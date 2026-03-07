from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

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
                resolved, q[:50], fts_score,
            )
            info = MatchInfo(MATCH_FTS, fts_score, _classify_temperature(MATCH_FTS, fts_score))
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
            info = MatchInfo(MATCH_SEMANTIC, score, _classify_temperature(MATCH_SEMANTIC, score))
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
                best["tema"], score, query[:50],
            )
            return best["tema"], score
    return None, 0.0


# ---------------------------------------------------------------------------
# Step 2a – Tema-only: find tema-level stat
# ---------------------------------------------------------------------------


def _find_stat_tema_only(resolved_tema: str, db: DBClient) -> tuple[dict | None, int]:
    """For tema-only input: find the best tema-level stat.

    Prefers a proper tema-level row (subtema IS NULL). If only subtema-level
    rows exist, returns the one with the most questions BUT aggregates the
    total num_questions across all subtemas.

    Returns (stat, num_questions).
    """
    stat = db.get_theme_stat(resolved_tema, None)
    if stat:
        return stat, stat["num_questions"]

    subtemas = db.get_subtemas_for_tema(resolved_tema)
    if subtemas:
        total = sum(s.get("num_questions", 0) for s in subtemas)
        best = max(subtemas, key=lambda s: s.get("num_questions", 0))
        return best, total

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
                info = MatchInfo(MATCH_EXACT, 1.0, _classify_temperature(MATCH_EXACT, 1.0))
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

    # Semantic fallback — search for tema+subtema via embeddings
    if embedder and subtema_raw:
        query = f"{resolved_tema} {subtema_raw}"
        embedding = embedder.embed(query)
        results = db.semantic_search_theme_stats(
            query_embedding=embedding,
            query_text=query,
            top_k=3,
        )
        for r in results:
            if r.get("tema", "").lower() == resolved_tema.lower() and r.get("subtema"):
                score = r.get("hybrid_score", 0) or 0
                if score >= 0.35:
                    logger.debug(
                        "[_find_stat_with_subtema] semantic match subtema='%s' score=%.3f",
                        r["subtema"],
                        score,
                    )
                    info = MatchInfo(MATCH_SEMANTIC, score, _classify_temperature(MATCH_SEMANTIC, score))
                    return r, info

    # Best subtema under resolved tema (ordered by num_questions DESC)
    # This is a guess — tema matched but subtema didn't, so we pick the top one
    subtemas = db.get_subtemas_for_tema(resolved_tema)
    if subtemas:
        score = base.score * 0.40  # significant penalty: subtema is a guess
        method = MATCH_SEMANTIC
        info = MatchInfo(method, score, _classify_temperature(method, score))
        return subtemas[0], info

    # Last resort: tema-level row (no subtema match at all)
    stat = db.get_theme_stat(resolved_tema, None)
    if stat:
        score = base.score * 0.40
        method = MATCH_SEMANTIC
        info = MatchInfo(method, score, _classify_temperature(method, score))
        return stat, info

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
        match_label=match_info.label or _classify_temperature(match_info.method, match_info.score),
        cor_hex=cor_hex,
        notes=notes,
    )


def _dedup_key(row: ReconciledRow) -> str:
    """Build a deduplication key from normalized tema + subtema."""
    tema = (row.normalized_tema or "").strip().lower()
    subtema = (row.normalized_subtema or "").strip().lower()
    return f"{tema}||{subtema}"


def _retry_low_score(
    row: dict[str, Any],
    embedder: BaseEmbedder,
    db: DBClient,
) -> ReconciledRow | None:
    """Re-attempt reconciliation with alternative query strategies.

    Strategies tried:
    1. Use only the tema (drop subtema noise)
    2. Use only the equivalencia hint
    3. Use tema + each word from subtema separately
    """
    tema_raw = str(row.get("tema", "")).strip()
    subtema_raw = row.get("subtema")
    equivalencia = row.get("equivalencia")

    variants = [tema_raw]
    if equivalencia and str(equivalencia).strip().lower() != "nan":
        variants.append(str(equivalencia).strip())
    if subtema_raw and str(subtema_raw).strip().lower() != "nan":
        # Try tema + individual subtema words
        for word in str(subtema_raw).split():
            if len(word) > 3:
                variants.append(f"{tema_raw} {word}")

    best_result = None
    for variant in variants:
        alt_row = {**row, "tema": variant, "subtema": None, "equivalencia": None}
        try:
            result = reconcile_row(alt_row, embedder, db)
            if best_result is None or result.match_score > best_result.match_score:
                # Preserve original input display
                result.input_tema = (
                    f"{tema_raw} | {subtema_raw}" if subtema_raw else tema_raw
                )
                result.input_equivalencia = result.input_equivalencia or equivalencia
                best_result = result
                if result.match_score >= MIN_ACCEPTABLE_SCORE:
                    break
        except Exception:
            continue

    return best_result


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

    # Retry rows with low scores using alternative queries
    for i, r in enumerate(results):
        if r.match_score < MIN_ACCEPTABLE_SCORE and r.match_method != MATCH_NONE:
            row = input_rows[i]
            retried = _retry_low_score(row, embedder, db)
            if retried and retried.match_score > r.match_score:
                logger.info(
                    "[reconcile_all] retry improved row %d: %.0f%% -> %.0f%%",
                    i + 1, r.match_score * 100, retried.match_score * 100,
                )
                results[i] = retried

    seen: dict[str, ReconciledRow] = {}
    for r in results:
        key = _dedup_key(r)
        if key not in seen or r.match_score > seen[key].match_score:
            seen[key] = r
    deduped = list(seen.values())

    if len(deduped) < len(results):
        logger.info(
            "[reconcile_all] deduplicated %d -> %d rows",
            len(results),
            len(deduped),
        )

    # Sort by score descending (highest confidence first)
    deduped.sort(key=lambda r: r.match_score, reverse=True)
    logger.info(
        "[reconcile_all] reconciliation complete: %d rows produced", len(deduped)
    )
    return deduped


# ---------------------------------------------------------------------------
# Reverse coverage: DB themes → check if input covers them
# ---------------------------------------------------------------------------


def _build_input_embeddings(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
) -> list[tuple[str, list[float]]]:
    """Build (display_text, embedding) pairs for all input rows."""
    texts = []
    displays = []
    for row in input_rows:
        tema = str(row.get("tema", "")).strip()
        subtema = row.get("subtema")
        equiv = row.get("equivalencia")
        parts = [tema]
        if subtema and str(subtema).strip().lower() != "nan":
            parts.append(str(subtema).strip())
        if equiv and str(equiv).strip().lower() != "nan":
            parts.append(str(equiv).strip())
        text = " ".join(parts)
        display = (
            f"{tema} | {subtema}"
            if subtema and str(subtema).strip().lower() != "nan"
            else tema
        )
        texts.append(text)
        displays.append(display)

    embeddings = embedder.embed_batch(texts) if texts else []
    return list(zip(displays, embeddings))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def reverse_coverage(
    input_rows: list[dict[str, Any]],
    embedder: BaseEmbedder,
    db: DBClient,
    coverage_threshold: float = 0.45,
    partial_threshold: float = 0.30,
) -> list[ReverseRow]:
    """For each theme_stat in the DB, check if any input row covers it.

    This is the 'reverse' view: the DB themes are the 'input' and
    the user's spreadsheet entries are the 'equivalence'.
    Uses semantic similarity (embeddings) to find matches.
    """
    logger.info("[reverse_coverage] starting reverse coverage analysis")

    all_stats = db.get_all_theme_stats()
    if not all_stats:
        logger.warning("[reverse_coverage] no theme_stats found in DB")
        return []

    input_pairs = _build_input_embeddings(input_rows, embedder)
    if not input_pairs:
        logger.warning("[reverse_coverage] no input rows to compare")
        return [
            ReverseRow(
                db_tema=s["tema"],
                db_subtema=s.get("subtema"),
                db_num_questions=s["num_questions"],
                db_cor_hex=s.get("cor_hex", "#22C55E"),
                matched_input=None,
                similarity_score=0.0,
                coverage_status="não coberto",
                notes="Nenhuma entrada fornecida",
            )
            for s in all_stats
        ]

    # Build embeddings for each DB theme_stat text
    stat_texts = []
    for s in all_stats:
        text = s["tema"]
        if s.get("subtema"):
            text += f" {s['subtema']}"
        stat_texts.append(text)
    stat_embeddings = embedder.embed_batch(stat_texts)

    results = []
    for stat, stat_emb in zip(all_stats, stat_embeddings):
        best_score = 0.0
        best_input = None

        for display, input_emb in input_pairs:
            sim = _cosine_similarity(stat_emb, input_emb)
            if sim > best_score:
                best_score = sim
                best_input = display

        if best_score >= coverage_threshold:
            status = "coberto"
        elif best_score >= partial_threshold:
            status = "parcial"
        else:
            status = "não coberto"

        notes_parts = []
        if best_input:
            notes_parts.append(f"Melhor match: {best_input}")
        notes_parts.append(f"Similaridade: {best_score:.2%}")

        results.append(
            ReverseRow(
                db_tema=stat["tema"],
                db_subtema=stat.get("subtema"),
                db_num_questions=stat["num_questions"],
                db_cor_hex=stat.get("cor_hex", "#22C55E"),
                matched_input=best_input if best_score >= partial_threshold else None,
                similarity_score=best_score,
                coverage_status=status,
                notes="; ".join(notes_parts),
            )
        )

    status_order = {"não coberto": 0, "parcial": 1, "coberto": 2}
    results.sort(
        key=lambda r: (status_order.get(r.coverage_status, 3), -r.db_num_questions)
    )

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
