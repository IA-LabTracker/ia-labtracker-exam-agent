from src.retriever.hybrid_retriever import retrieve_candidates
from unittest.mock import patch


def test_retrieve_calls_embedder(monkeypatch):
    # patch Embedder to return fixed vector
    with patch("src.retriever.hybrid_retriever.Embedder") as FakeEmbed:
        FakeEmbed.return_value.embed_single.return_value = [0.0, 0.1]
        # patch DBClient.fetch to return a dummy row regardless of SQL params
        with patch("src.retriever.hybrid_retriever.DBClient") as FakeDB:
            instance = FakeDB.return_value
            instance.fetch.return_value = [{"id": 42, "hybrid_score": 0.5}]
            res = retrieve_candidates("foo")
            assert res[0]["id"] == 42
            # ensure the embedding call was invoked
            FakeEmbed.return_value.embed_single.assert_called_with("foo")
