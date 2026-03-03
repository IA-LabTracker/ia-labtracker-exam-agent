from src.aggregator.consolidate import consolidate
import src.aggregator.consolidate as cons


def test_consolidate_simple(monkeypatch):
    rows = [
        {"tema": "Trauma", "classificacao": "A", "equivalencia": "X", "num_questoes": 1}
    ]

    def fake_retrieve(text, top_k=5, alpha=0.5):
        return [{"id": 1, "hybrid_score": 0.9}]

    class FakeDB:
        def __init__(self):
            self.calls = []

        def fetch(self, sql, *args, **kwargs):
            self.calls.append(sql)
            if "theme_stats" in sql:
                return [
                    {
                        "ranking": 7,
                        "category": "tema",
                        "percentage": 2.5,
                        "cor": "verde",
                        "cor_hex": "#22C55E",
                    }
                ]
            return [{"cnt": 3}]

    monkeypatch.setattr(cons, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(cons, "DBClient", lambda: FakeDB())

    out = consolidate(rows)
    assert out[0]["similarity"] == 0.9
    assert out[0]["num_questions"] == 3
    assert out[0]["matched_ids"] == [1]
    assert out[0]["ranking"] == 7
    assert out[0]["cor"] == "verde"
    assert "priority_score" in out[0]
