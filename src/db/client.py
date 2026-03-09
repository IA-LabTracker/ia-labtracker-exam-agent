from __future__ import annotations

import hashlib
import threading
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.config import get_settings
from src.utils.logging import logger


class DBClient:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or get_settings().database_url
        self._conn: psycopg.Connection | None = None
        # Reentrant lock — serialises actual DB network calls across threads.
        # Cache reads are lock-free (GIL-safe in CPython); the lock is only
        # acquired when a real self.conn.execute() is about to happen, using
        # the double-checked locking pattern so cached results bypass it.
        self._lock = threading.RLock()
        # Session-level caches — cleared on close()
        self._theme_stat_cache: dict[
            tuple[str, str | None, str | None], dict | None
        ] = {}
        self._subtemas_cache: dict[tuple[str, str | None], list[dict]] = {}
        self._all_theme_stats_cache: list[dict] | None = None
        self._fts_cache: dict[tuple[str, str | None], dict | None] = {}
        self._semantic_cache: dict[tuple[str, int], list[dict]] = {}
        self._inst_questions_cache: dict[tuple[str, str | None], dict[str, int]] = {}

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
        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear all session-level caches."""
        self._theme_stat_cache.clear()
        self._subtemas_cache.clear()
        self._all_theme_stats_cache = None
        self._fts_cache.clear()
        self._semantic_cache.clear()
        self._inst_questions_cache.clear()

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
        cache_key = (tema.lower(), (subtema or "").lower() or None, institution)
        if cache_key in self._theme_stat_cache:
            return self._theme_stat_cache[cache_key]

        with self._lock:
            if cache_key in self._theme_stat_cache:  # double-check after lock
                return self._theme_stat_cache[cache_key]
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
            result = dict(row) if row else None
            self._theme_stat_cache[cache_key] = result
        return result

    def find_best_theme_stat(
        self, query: str, institution: str | None = None
    ) -> dict | None:
        """Search theme_stats using FTS for flexible matching.

        Tries exact match first, then FTS, to find the best matching
        theme_stat entry for a given query string.
        """
        cache_key = (query.lower(), institution)
        if cache_key in self._fts_cache:
            return self._fts_cache[cache_key]

        with self._lock:
            if cache_key in self._fts_cache:  # double-check after lock
                return self._fts_cache[cache_key]

            if "|" in query:
                parts = [p.strip() for p in query.split("|", 1)]
                exact = self.get_theme_stat(parts[0], parts[1], institution)
                if exact:
                    self._fts_cache[cache_key] = exact
                    return exact

            exact = self.get_theme_stat(query, None, institution)
            if exact:
                self._fts_cache[cache_key] = exact
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
            result = dict(row) if row else None
            self._fts_cache[cache_key] = result
        return result

    def get_subtemas_for_tema(
        self, tema: str, institution: str | None = None
    ) -> list[dict]:
        """Return all subtema-level entries for a given tema."""
        cache_key = (tema.lower(), institution)
        if cache_key in self._subtemas_cache:
            return self._subtemas_cache[cache_key]

        with self._lock:
            if cache_key in self._subtemas_cache:  # double-check after lock
                return self._subtemas_cache[cache_key]
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
            result = [dict(r) for r in rows]
            self._subtemas_cache[cache_key] = result
        return result

    def search_theme_stats_fts(self, query: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM theme_stats WHERE fts @@ plainto_tsquery('portuguese', %s) ORDER BY num_questions DESC LIMIT %s",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- theme_stats embeddings -------------------------------------------------

    def get_theme_stats_without_embeddings(self) -> list[dict]:
        return list(
            self.conn.execute(
                "SELECT id, tema, subtema FROM theme_stats WHERE embedding IS NULL"
            ).fetchall()
        )

    def update_theme_stat_embedding(self, stat_id: int, embedding: list[float]) -> None:
        self.conn.execute(
            "UPDATE theme_stats SET embedding = %s::vector WHERE id = %s",
            (str(embedding), stat_id),
        )

    def update_theme_stat_embeddings_batch(
        self, updates: list[tuple[int, list[float]]]
    ) -> None:
        """Update embeddings for multiple theme_stats rows in a single transaction."""
        if not updates:
            return
        with self._lock:
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE theme_stats SET embedding = %s::vector WHERE id = %s",
                        [(str(emb), stat_id) for stat_id, emb in updates],
                    )

    def semantic_search_theme_stats(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> list[dict]:
        # Cache by (query_text, top_k, alpha, beta) — all parameters affect the result
        cache_key = (query_text.lower().strip(), top_k, alpha, beta)
        if cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]

        with self._lock:
            if cache_key in self._semantic_cache:  # double-check after lock
                return self._semantic_cache[cache_key]
            logger.debug(
                "[semantic_search_theme_stats] query_text='%s' top_k=%d",
                query_text[:100],
                top_k,
            )
            try:
                rows = self.conn.execute(
                    "SELECT * FROM semantic_search_theme_stats(%s::vector, %s, %s, %s, %s)",
                    (str(query_embedding), query_text, top_k, alpha, beta),
                ).fetchall()
                logger.debug("[semantic_search_theme_stats] returned %d rows", len(rows))
                result = [dict(r) for r in rows]
                self._semantic_cache[cache_key] = result
            except Exception as exc:
                logger.error("[semantic_search_theme_stats] error: %s", exc, exc_info=True)
                result = []
                self._semantic_cache[cache_key] = result
        return result

    def _fetch_inst_counts(
        self, table: str, tema: str, subtema: str | None
    ) -> list[dict]:
        """Fetch per-institution question counts from a given table/view by exact tema match."""
        try:
            if subtema:
                rows = self.conn.execute(
                    f"SELECT institution, num_questions FROM {table} "
                    "WHERE lower(tema) = lower(%s) AND lower(subtema) = lower(%s)",
                    (tema, subtema),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    f"SELECT institution, SUM(num_questions) AS num_questions FROM {table} "
                    "WHERE lower(tema) = lower(%s) GROUP BY institution",
                    (tema,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("[_fetch_inst_counts] error querying %s: %s", table, exc)
            return []

    def get_questions_by_institution(
        self, tema: str, subtema: str | None = None
    ) -> dict[str, int]:
        """Return {institution: num_questions} for the given tema/subtema.

        Query order:
          1. theme_stats_all — canonical aggregated view (preferred)
          2. theme_stats     — fallback if view returns nothing
          3. theme_stats FTS — fallback when exact tema name doesn't match (different institutions
                               may name the same topic differently)
        """
        cache_key = (tema.lower(), (subtema or "").lower() or None)
        if cache_key in self._inst_questions_cache:
            return self._inst_questions_cache[cache_key]

        with self._lock:
            if cache_key in self._inst_questions_cache:  # double-check after lock
                return self._inst_questions_cache[cache_key]

            # 1. Try theme_stats_all (canonical table — preferred)
            rows = self._fetch_inst_counts("theme_stats_all", tema, subtema)

            # 2. Fallback: theme_stats directly
            if not rows:
                rows = self._fetch_inst_counts("theme_stats", tema, subtema)

            # 3. If subtema produced no results, retry at tema level (sums all subtemas)
            if not rows and subtema:
                rows = self._fetch_inst_counts("theme_stats_all", tema, None)
                if not rows:
                    rows = self._fetch_inst_counts("theme_stats", tema, None)
                if rows:
                    logger.debug(
                        "[get_questions_by_institution] subtema '%s' not found — fell back to tema-level for '%s'",
                        subtema,
                        tema,
                    )

            # 4. FTS fallback on theme_stats — handles topic name variations across institutions
            if not rows:
                try:
                    fts_rows = self.conn.execute(
                        "SELECT institution, SUM(num_questions) AS num_questions "
                        "FROM theme_stats "
                        "WHERE fts @@ plainto_tsquery('portuguese', %s) "
                        "GROUP BY institution",
                        (tema,),
                    ).fetchall()
                    rows = [dict(r) for r in fts_rows]
                    if rows:
                        logger.debug(
                            "[get_questions_by_institution] FTS fallback found %d institutions for tema='%s'",
                            len(rows),
                            tema,
                        )
                except Exception as exc:
                    logger.warning("[get_questions_by_institution] FTS fallback failed: %s", exc)

            result: dict[str, int] = {r["institution"]: int(r["num_questions"]) for r in rows}
            self._inst_questions_cache[cache_key] = result
            logger.debug(
                "[get_questions_by_institution] tema='%s' subtema='%s' → %d institutions: %s",
                tema,
                subtema or "(none)",
                len(result),
                list(result.keys()),
            )
        return result

    def get_all_theme_stats(self) -> list[dict]:
        """Return one row per unique (tema, subtema) pair from theme_stats_all.

        Aggregates across institutions so reverse_coverage doesn't process the
        same tema+subtema combination multiple times (once per institution).
        Falls back to theme_stats if theme_stats_all is empty.
        """
        if self._all_theme_stats_cache is not None:
            return self._all_theme_stats_cache
        with self._lock:
            if self._all_theme_stats_cache is not None:  # double-check after lock
                return self._all_theme_stats_cache
            rows = self.conn.execute(
                """
                SELECT tema, subtema,
                       SUM(num_questions) AS num_questions,
                       MAX(cor_hex) AS cor_hex
                FROM theme_stats_all
                GROUP BY tema, subtema
                ORDER BY num_questions DESC
                """
            ).fetchall()
            if not rows:
                # Fallback: theme_stats if theme_stats_all is unpopulated
                rows = self.conn.execute(
                    """
                    SELECT tema, subtema,
                           SUM(num_questions) AS num_questions,
                           MAX(cor_hex) AS cor_hex
                    FROM theme_stats
                    GROUP BY tema, subtema
                    ORDER BY num_questions DESC
                    """
                ).fetchall()
            self._all_theme_stats_cache = [dict(r) for r in rows]
        return self._all_theme_stats_cache
