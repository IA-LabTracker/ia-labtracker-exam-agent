from typing import List, Dict
from src.embeddings.embedder import Embedder
from src.db.client import DBClient, SupabaseClient
from src.config import settings
import logging

logger = logging.getLogger(__name__)


def retrieve_candidates(text: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
    """Return candidate rows from the hybrid_search function.

    The database function signature changed; it now expects the embedding
    first and the textual query second, with optional alpha/beta weights.
    """
    emb = Embedder().embed_single(text)
    try:
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            supa = SupabaseClient()
            resp = supa.rpc(
                "hybrid_search",
                {
                    "query_embedding": emb,
                    "query_text": text,
                    "match_count": top_k,
                    "alpha": alpha,
                },
            )
            return resp if resp else []
        else:
            db = DBClient()
            # note order: embedding then text
            rows = db.fetch(
                "SELECT * FROM hybrid_search(%s::vector, %s, %s, %s)",
                emb,
                text,
                top_k,
                alpha,
            )
            return rows
    except Exception as exc:
        logger.error("hybrid retrieval failed", exc_info=exc)
        return []
