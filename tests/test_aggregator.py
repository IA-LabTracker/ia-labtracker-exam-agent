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


def _make_db_mock(theme_stat=None, subtemas=None, inst_questions=None):
    db = MagicMock()
    db.get_theme_stat.return_value = theme_stat
    db.find_best_theme_stat.return_value = None
    db.get_subtemas_for_tema.return_value = subtemas or []
    db.semantic_search_theme_stats.return_value = []
    db.get_questions_by_institution.return_value = inst_questions or {}
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
        assert isinstance(result.questions_by_institution, dict)

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_no_matches_produces_notes(self, mock_retrieve):
        mock_retrieve.return_value = []
        embedder = MagicMock()
        db = _make_db_mock()

        row = {"tema": "Unknown Topic"}
        result = reconcile_row(row, embedder, db)

        assert result.questions_by_institution == {}
        assert "No matches" in result.notes

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_uses_db_color_for_tema_only(self, mock_retrieve):
        """Tema-only input matched via exact DB stat (no subtema)."""
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        stat = {
            "cor": "vermelho",
            "cor_hex": "#EF4444",
            "num_questions": 9,
            "tema": "Trauma",
            "subtema": None,
            "institution": "FAMERP",
            "ranking": 1,
        }
        db = _make_db_mock(theme_stat=stat, inst_questions={"FAMERP": 9})

        row = {"tema": "Trauma"}
        result = reconcile_row(row, embedder, db)

        assert result.cor_hex == "#EF4444"
        assert sum(result.questions_by_institution.values()) == 9
        # Verify tema was normalized (case preserved if not a synonym)
        assert result.normalized_tema in ("Trauma", "tema_0")
        assert result.normalized_subtema is None
        # Input equivalencia should be set
        assert result.input_equivalencia is not None

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_tema_with_subtema_match(self, mock_retrieve):
        """When input has subtema, it should match subtema-level DB entry."""
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        stat = {
            "cor": "vermelho",
            "cor_hex": "#EF4444",
            "num_questions": 7,
            "tema": "Trauma",
            "subtema": "Abordagem inicial",
            "institution": "FAMERP",
            "ranking": 1,
        }
        db = _make_db_mock(theme_stat=stat, inst_questions={"FAMERP": 7})

        row = {"tema": "Trauma", "subtema": "Abordagem inicial"}
        result = reconcile_row(row, embedder, db)

        assert result.cor_hex == "#EF4444"
        assert sum(result.questions_by_institution.values()) == 7
        assert result.normalized_subtema is not None
        # Verify equivalencia contains expected parts
        equivalencia = result.input_equivalencia or ""
        assert "Abordagem inicial" in equivalencia or "Trauma" in equivalencia

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_fts_fallback_resolves_tema(self, mock_retrieve):
        """When exact match fails, FTS should resolve the tema."""
        mock_retrieve.return_value = []
        embedder = MagicMock()

        fts_stat = {
            "num_questions": 7,
            "tema": "Infecções congênitas",
            "subtema": None,
            "institution": "FAMERP",
            "ranking": 9,
        }

        db = MagicMock()
        db.get_theme_stat.return_value = None
        db.find_best_theme_stat.return_value = fts_stat
        db.get_subtemas_for_tema.return_value = (
            []
        )  # No exact matches, so fallback to FTS
        db.semantic_search_theme_stats.return_value = []
        db.get_questions_by_institution.return_value = {}

        row = {"tema": "Infecções Congênitas"}
        result = reconcile_row(row, embedder, db)

        # FTS resolves via find_best_theme_stat
        assert result.match_method == "FTS"

    @patch("src.aggregator.consolidate.retrieve_candidates")
    def test_fallback_color_when_no_stat(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(2)
        embedder = MagicMock()
        db = _make_db_mock()

        row = {"tema": "Test"}
        result = reconcile_row(row, embedder, db)

        # 2 candidates, no stat → total fallback = num_candidates = 2 → amarelo
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
        db = _make_db_mock(theme_stat=stat, inst_questions={"FAMERP": 7})

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
    def test_sorted_by_score_descending(self, mock_retrieve):
        mock_retrieve.return_value = _mock_candidates(1)
        embedder = MagicMock()
        db = _make_db_mock()

        rows = [{"tema": f"Topic {i}"} for i in range(3)]
        results = reconcile_all(rows, embedder, db)

        scores = [r.match_score for r in results]
        assert scores == sorted(scores, reverse=True)
