from typing import List, Dict
from src.normalize.normalizer import normalize_tema_subtema
from src.retriever.hybrid_retriever import retrieve_candidates
from src.db.client import DBClient

DEFAULT_WEIGHTS = {"similarity": 0.6, "freq": 0.4}


def consolidate(rows: List[Dict]) -> List[Dict]:
    """Aggregate input Excel rows into a normalized output.

    Each returned dict includes provenance fields used for tracking and
    ranking. The caller can sort by `priority_score` if desired.
    """
    db = DBClient()
    output: List[Dict] = []

    for row in rows:
        tema_norm, sub_norm = normalize_tema_subtema(row.get("tema"))
        query_text = f"{tema_norm} {sub_norm}".strip()
        candidates = retrieve_candidates(query_text)

        provenance = []
        for cand in candidates:
            cnt_rows = db.fetch(
                "SELECT count(*) as cnt FROM questions WHERE id=%s", cand["id"]
            )
            cnt = cnt_rows[0]["cnt"] if cnt_rows else 0
            provenance.append(
                {
                    "id": cand["id"],
                    "tema": cand.get("tema"),
                    "subtema": cand.get("subtema"),
                    "hybrid_score": cand.get("hybrid_score", 0),
                    "fts_score": cand.get("fts_score", 0),
                    "vector_score": cand.get("vector_score", 0),
                    "num_questions": cnt,
                }
            )

        top_similarity = max((p["hybrid_score"] for p in provenance), default=0)
        freq_sum = sum(p["num_questions"] for p in provenance)
        priority_score = (
            top_similarity * DEFAULT_WEIGHTS["similarity"]
            + freq_sum * DEFAULT_WEIGHTS["freq"]
        )

        # attempt to look up ranking/color info from theme_stats
        stats = db.fetch(
            "SELECT ranking, category, percentage, cor, cor_hex"
            " FROM theme_stats WHERE tema=%s AND (subtema IS NULL OR subtema=%s) LIMIT 1",
            tema_norm,
            sub_norm,
        )
        stat = stats[0] if stats else {}

        outrow = {
            "input_tema": row.get("tema"),
            "input_classificacao": row.get("classificacao"),
            "input_equivalencia": row.get("equivalencia"),
            "normalized_tema": tema_norm,
            "normalized_subtema": sub_norm,
            "similarity": top_similarity,
            "num_questions": provenance[0]["num_questions"] if provenance else 0,
            "matched_ids": [p["id"] for p in provenance],
            "notes": "",
            "priority_score": priority_score,
        }
        if stat:
            outrow.update(
                {
                    "ranking": stat.get("ranking"),
                    "category": stat.get("category"),
                    "percentage": stat.get("percentage"),
                    "cor": stat.get("cor"),
                    "cor_hex": stat.get("cor_hex"),
                }
            )
        output.append(outrow)

    return output
