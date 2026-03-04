from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.config import get_settings
from src.utils.logging import logger


class DBClient:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or get_settings().database_url
        self._conn: psycopg.Connection | None = None

    def connect(self) -> DBClient:
        self._conn = psycopg.connect(self._dsn, row_factory=dict_row, autocommit=True)
        logger.info("Connected to database")
        return self

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def run_migrations(self) -> None:
        sql_dir = get_settings().sql_dir
        if not sql_dir.exists():
            logger.warning("SQL dir not found: %s", sql_dir)
            return
        files = sorted(sql_dir.glob("*.sql"))
        for f in files:
            logger.info("Running migration: %s", f.name)
            self.conn.execute(f.read_text(encoding="utf-8"))
        logger.info("All migrations applied")

    def upsert_questions(self, questions: list[dict[str, Any]]) -> int:
        count = 0
        with self.conn.transaction():
            for q in questions:
                content = q.get("raw_text", "")
                chash = hashlib.sha256(content.encode()).hexdigest()
                self.conn.execute(
                    """
                    INSERT INTO questions (institution, year, raw_text, tema_normalized, subtema_normalized, source_file, content_hash)
                    VALUES (%(institution)s, %(year)s, %(raw_text)s, %(tema_normalized)s, %(subtema_normalized)s, %(source_file)s, %(content_hash)s)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        tema_normalized = EXCLUDED.tema_normalized,
                        subtema_normalized = EXCLUDED.subtema_normalized
                    """,
                    {**q, "content_hash": chash},
                )
                count += 1
        return count

    def get_questions_without_embeddings(self) -> list[dict]:
        return list(
            self.conn.execute(
                "SELECT id, raw_text, tema_normalized, subtema_normalized FROM questions WHERE embedding IS NULL"
            ).fetchall()
        )

    def update_embedding(self, question_id: int, embedding: list[float]) -> None:
        self.conn.execute(
            "UPDATE questions SET embedding = %s::vector WHERE id = %s",
            (str(embedding), question_id),
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> list[dict]:
        logger.debug(
            "[hybrid_search] executing SQL function with: query_text='%s' top_k=%d alpha=%.2f beta=%.2f",
            query_text[:100],
            top_k,
            alpha,
            beta,
        )
        try:
            rows = self.conn.execute(
                "SELECT * FROM hybrid_search(%s::vector, %s, %s, %s, %s)",
                (str(query_embedding), query_text, top_k, alpha, beta),
            ).fetchall()
            logger.debug("[hybrid_search] query returned %d rows", len(rows))
            return list(rows)
        except Exception as exc:
            logger.error(
                "[hybrid_search] error executing SQL function: %s",
                exc,
                exc_info=True,
            )
            raise

    def file_already_ingested(self, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT id FROM ingest_log WHERE file_hash = %s", (file_hash,)
        ).fetchone()
        return row is not None

    def record_ingest(self, file_name: str, file_hash: str, row_count: int) -> None:
        self.conn.execute(
            "INSERT INTO ingest_log (file_name, file_hash, row_count) VALUES (%s, %s, %s) ON CONFLICT (file_hash) DO NOTHING",
            (file_name, file_hash, row_count),
        )

    def upsert_theme_stats(self, rows: list[dict[str, Any]]) -> int:
        count = 0
        with self.conn.transaction():
            for r in rows:
                self.conn.execute(
                    """
                    INSERT INTO theme_stats (institution, ranking, category, tema, subtema, percentage, num_questions, cor, cor_hex, source_file)
                    VALUES (%(institution)s, %(ranking)s, %(category)s, %(tema)s, %(subtema)s, %(percentage)s, %(num_questions)s, %(cor)s, %(cor_hex)s, %(source_file)s)
                    ON CONFLICT (institution, category, tema, subtema) DO UPDATE SET
                        ranking = EXCLUDED.ranking,
                        percentage = EXCLUDED.percentage,
                        num_questions = EXCLUDED.num_questions,
                        cor = EXCLUDED.cor,
                        cor_hex = EXCLUDED.cor_hex
                    """,
                    r,
                )
                count += 1
        return count

    def get_theme_stat(
        self, tema: str, subtema: str | None = None, institution: str | None = None
    ) -> dict | None:
        if subtema:
            row = self.conn.execute(
                "SELECT * FROM theme_stats WHERE lower(tema) = lower(%s) AND lower(subtema) = lower(%s)"
                + (" AND institution = %s" if institution else "")
                + " LIMIT 1",
                (tema, subtema, institution) if institution else (tema, subtema),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM theme_stats WHERE lower(tema) = lower(%s) AND subtema IS NULL"
                + (" AND institution = %s" if institution else "")
                + " LIMIT 1",
                (tema, institution) if institution else (tema,),
            ).fetchone()
        return dict(row) if row else None

    def find_best_theme_stat(
        self, query: str, institution: str | None = None
    ) -> dict | None:
        """Search theme_stats using FTS for flexible matching.

        Tries exact match first, then FTS, to find the best matching
        theme_stat entry for a given query string.
        """
        if "|" in query:
            parts = [p.strip() for p in query.split("|", 1)]
            exact = self.get_theme_stat(parts[0], parts[1], institution)
            if exact:
                return exact

        exact = self.get_theme_stat(query, None, institution)
        if exact:
            return exact

        base_query = (
            "SELECT * FROM theme_stats "
            "WHERE fts @@ plainto_tsquery('portuguese', %s)"
        )
        params: list = [query]
        if institution:
            base_query += " AND institution = %s"
            params.append(institution)
        base_query += " ORDER BY num_questions DESC LIMIT 1"

        row = self.conn.execute(base_query, params).fetchone()
        return dict(row) if row else None

    def get_subtemas_for_tema(
        self, tema: str, institution: str | None = None
    ) -> list[dict]:
        """Return all subtema-level entries for a given tema."""
        query = (
            "SELECT * FROM theme_stats "
            "WHERE lower(tema) = lower(%s) AND subtema IS NOT NULL"
        )
        params: list = [tema]
        if institution:
            query += " AND institution = %s"
            params.append(institution)
        query += " ORDER BY num_questions DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search_theme_stats_fts(self, query: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM theme_stats WHERE fts @@ plainto_tsquery('portuguese', %s) ORDER BY num_questions DESC LIMIT %s",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
