from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.aggregator.consolidate import ReconciledRow, reconcile_all, reconcile_row
from src.retriever.hybrid_retriever import Candidate


def _mock_candidates(n: int = 3) -> list[Candidate]:
    return [
        Candidate(
            id=i,
            tema_normalized=f"tema_{i}",
            subtema_normalized=f"subtema_{i}",
            raw_text=f"Question text {i}",
            similarity=0.9 - (i * 0.1),
            fts_score=0.5,
            hybrid_score=0.8 - (i * 0.05),
        )
        for i in range(n)
    ]


def _make_db_mock(theme_stat=None, subtemas=None):
    db = MagicMock()
    db.get_theme_stat.return_value = theme_stat
    db.find_best_theme_stat.return_value = None
    db.get_subtemas_for_tema.return_value = subtemas or []
    return db


class TestReconcileRow:
    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_produces_reconciled_row(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(2)
        embedder = MagicMock()
        db = _make_db_mock()

        row = {
            "tema": "Trauma",
            "equivalencia": "Initial Approach",
        }
        result = reconcile_row(row, embedder, db)

        assert isinstance(result, ReconciledRow)
        assert result.input_tema == "Trauma"
        assert result.num_questions == 2
        assert len(result.matched_ids) == 2
        assert result.similarity > 0

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_no_matches_produces_notes(self, mock_retrieve):
        mock_retrieve.return_value = []
        embedder = MagicMock()
        db = _make_db_mock()

        row = {"tema": "Unknown Topic"}
        result = reconcile_row(row, embedder, db)

        assert result.num_questions == 0
        assert "No matches" in result.notes

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_priority_score_formula(self, mock_retrieve):
        candidates = _mock_candidates(2)
        mock_retrieve.return_value = candidates
        embedder = MagicMock()
        db = _make_db_mock()

        row = {"tema": "Test"}
        result = reconcile_row(row, embedder, db)

        avg_sim = sum(c.similarity for c in candidates) / len(candidates)
        expected = round(0.4 * 2 + 0.6 * avg_sim, 4)
        assert result.priority_score == expected

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_uses_db_color_and_normalized_fields(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        stat = {
            "cor": "vermelho",
            "cor_hex": "#EF4444",
            "num_questions": 9,
            "tema": "Trauma",
            "subtema": "Abordagem inicial",
            "institution": "FAMERP",
            "ranking": 1,
        }
        db = _make_db_mock(theme_stat=stat)

        row = {"tema": "Trauma"}
        result = reconcile_row(row, embedder, db)

        assert result.cor == "vermelho"
        assert result.cor_hex == "#EF4444"
        assert result.num_questions == 9
        assert result.input_equivalencia == "Trauma | Abordagem inicial"
        # normalized fields come from theme_stat
        assert result.normalized_tema == "Trauma"
        assert result.normalized_subtema == "Abordagem inicial"

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_normalized_from_stat_when_no_candidates(self, mock_retrieve):
        """When questions DB is empty, normalized fields should come from theme_stats."""
        mock_retrieve.return_value = []
        embedder = MagicMock()
        db = MagicMock()
        db.get_theme_stat.return_value = None
        db.find_best_theme_stat.return_value = {
            "cor": "vermelho",
            "cor_hex": "#EF4444",
            "num_questions": 7,
            "tema": "Infecções congênitas",
            "subtema": "Sífilis congênita no RN",
            "institution": "FAMERP",
            "ranking": 9,
        }
        db.get_subtemas_for_tema.return_value = []

        row = {"tema": "Infecções Congênitas"}
        result = reconcile_row(row, embedder, db)

        assert result.normalized_tema == "Infecções congênitas"
        assert result.normalized_subtema == "Sífilis congênita no RN"
        assert result.num_questions == 7

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_fallback_color_when_no_stat(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(2)
        embedder = MagicMock()
        db = _make_db_mock()

        row = {"tema": "Test"}
        result = reconcile_row(row, embedder, db)

        assert result.cor == "amarelo"
        assert result.cor_hex == "#EAB308"

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_notes_include_source_info(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        stat = {
            "cor": "vermelho",
            "cor_hex": "#EF4444",
            "num_questions": 7,
            "tema": "Trauma",
            "subtema": None,
            "institution": "FAMERP",
            "ranking": 1,
        }
        db = _make_db_mock(theme_stat=stat)

        row = {"tema": "Trauma"}
        result = reconcile_row(row, embedder, db)

        assert "FAMERP" in result.notes
        assert "ranking" in result.notes


class TestReconcileAll:
    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_preserves_all_rows(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        db = _make_db_mock()

        rows = [{"tema": f"Topic {i}"} for i in range(5)]
        results = reconcile_all(rows, embedder, db)

        assert len(results) == 5

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_sorted_by_priority(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        db = _make_db_mock()

        rows = [{"tema": f"Topic {i}"} for i in range(3)]
        results = reconcile_all(rows, embedder, db)

        scores = [r.priority_score for r in results]
        assert scores == sorted(scores, reverse=True)
