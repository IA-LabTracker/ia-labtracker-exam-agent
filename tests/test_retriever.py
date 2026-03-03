from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.retriever.hybrid_retriever import Candidate, retrieve_candidates


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed.return_value = [0.1] * 384
    return emb


@pytest.fixture
def settings():
    return Settings(
        hybrid_alpha=0.7,
        hybrid_beta=0.3,
        similarity_threshold=0.4,
        retriever_top_k=5,
    )


class TestRetrieveCandidates:
    def test_returns_candidates_above_threshold(self, mock_embedder, settings):
        db = MagicMock()
        db.hybrid_search.return_value = [
            {
                "id": 1,
                "tema_normalized": "trauma",
                "subtema_normalized": "abordagem",
                "raw_text": "Question about trauma",
                "similarity": 0.85,
                "fts_score": 0.6,
                "hybrid_score": 0.77,
            },
            {
                "id": 2,
                "tema_normalized": "cardiologia",
                "subtema_normalized": None,
                "raw_text": "Heart question",
                "similarity": 0.3,
                "fts_score": 0.1,
                "hybrid_score": 0.24,
            },
        ]

        results = retrieve_candidates(
            "trauma abordagem", mock_embedder, db, settings=settings
        )

        assert len(results) == 1
        assert results[0].id == 1
        assert isinstance(results[0], Candidate)

    def test_empty_results(self, mock_embedder, settings):
        db = MagicMock()
        db.hybrid_search.return_value = []

        results = retrieve_candidates(
            "nonexistent", mock_embedder, db, settings=settings
        )
        assert results == []

    def test_calls_embedder_and_db(self, mock_embedder, settings):
        db = MagicMock()
        db.hybrid_search.return_value = []

        retrieve_candidates("test query", mock_embedder, db, settings=settings)

        mock_embedder.embed.assert_called_once_with("test query")
        db.hybrid_search.assert_called_once()

    def test_respects_top_k_override(self, mock_embedder, settings):
        db = MagicMock()
        db.hybrid_search.return_value = []

        retrieve_candidates("test", mock_embedder, db, top_k=10, settings=settings)

        call_args = db.hybrid_search.call_args
        assert call_args.kwargs.get("top_k") == 10 or call_args[1].get("top_k") == 10
