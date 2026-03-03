from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import get_settings
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.db.client import DBClient
    from src.embeddings.embedder import BaseEmbedder


@dataclass
class Candidate:
    id: int
    tema_normalized: str | None
    subtema_normalized: str | None
    raw_text: str
    similarity: float
    fts_score: float
    hybrid_score: float


def retrieve_candidates(
    query: str,
    embedder: BaseEmbedder,
    db: DBClient,
    top_k: int | None = None,
    settings=None,
) -> list[Candidate]:
    settings = settings or get_settings()
    top_k = top_k or settings.retriever_top_k

    embedding = embedder.embed(query)
    rows = db.hybrid_search(
        query_embedding=embedding,
        query_text=query,
        top_k=top_k,
        alpha=settings.hybrid_alpha,
        beta=settings.hybrid_beta,
    )

    candidates = []
    for r in rows:
        score = r.get("hybrid_score", 0) or 0
        if score < settings.similarity_threshold:
            continue
        candidates.append(
            Candidate(
                id=r["id"],
                tema_normalized=r.get("tema_normalized"),
                subtema_normalized=r.get("subtema_normalized"),
                raw_text=r.get("raw_text", ""),
                similarity=r.get("similarity", 0) or 0,
                fts_score=r.get("fts_score", 0) or 0,
                hybrid_score=score,
            )
        )

    logger.debug("Query '%s' returned %d candidates", query, len(candidates))
    return candidates
