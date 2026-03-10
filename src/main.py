from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.aggregator.consolidate import reconcile_all, reverse_coverage
from src.config import get_settings
from src.db.client import DBClient
from src.embeddings.embedder import Embedder
from src.exporters.excel_writer import write_excel
from src.ingest.excel_reader import read_excel
from src.ingest.pdf_parser import extract_questions, extract_theme_stats, file_hash
from src.utils.logging import logger

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Exam Reconciler", version="0.1.0")

_db: DBClient | None = None
_embedder = None


def get_db() -> DBClient:
    global _db
    if _db is None:
        _db = DBClient().connect()
    return _db


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder.create()
    return _embedder


async def _save_upload(file: UploadFile, suffix: str) -> Path:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    institution: str = "unknown",
    year: int | None = None,
):
    db = get_db()
    embedder = get_embedder()
    tmp_path = await _save_upload(file, ".pdf")

    try:
        fhash = file_hash(tmp_path)
        if db.file_already_ingested(fhash):
            return JSONResponse(
                {"message": f"File '{file.filename}' already ingested", "skipped": True}
            )

        questions = extract_questions(tmp_path, institution=institution, year=year)
        count = db.upsert_questions(questions)

        pending = db.get_questions_without_embeddings()
        if pending:
            texts = [
                f"{q['tema_normalized'] or ''} {q['subtema_normalized'] or ''} {q['raw_text']}"
                for q in pending
            ]
            embeddings = embedder.embed_batch(texts)
            for q, emb in zip(pending, embeddings):
                db.update_embedding(q["id"], emb)

        db.record_ingest(file.filename or "upload.pdf", fhash, count)

        return {"message": "Ingested successfully", "questions_added": count}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/stats")
async def ingest_stats(
    file: UploadFile = File(...),
    institution: str = "unknown",
):
    db = get_db()
    embedder = get_embedder()
    tmp_path = await _save_upload(file, ".pdf")

    try:
        fhash = file_hash(tmp_path)
        if db.file_already_ingested(fhash):
            return JSONResponse(
                {"message": f"File '{file.filename}' already ingested", "skipped": True}
            )

        stats = extract_theme_stats(tmp_path, institution=institution)
        count = db.upsert_theme_stats(stats)

        _ensure_theme_stats_embeddings(db, embedder)

        db.record_ingest(file.filename or "stats.pdf", fhash, count)

        return {"message": "Theme stats ingested", "stats_added": count}
    finally:
        tmp_path.unlink(missing_ok=True)


def _check_embedding_model_changed(db: DBClient) -> bool:
    """Check if the current embedding model differs from what was used to generate stored embeddings.

    Stores the model name in a metadata row. If it changed, all embeddings need regeneration.
    """
    settings = get_settings()
    current_model = f"{settings.embeddings_provider}:{settings.embedding_model}:{settings.embedding_dim}"

    try:
        with db._lock:
            # Use a simple key-value approach via a check on ingest_log or a direct query
            row = db.conn.execute(
                "SELECT file_name FROM ingest_log WHERE file_hash = '__embedding_model__' LIMIT 1"
            ).fetchone()

        if row and row["file_name"] == current_model:
            return False  # Same model, no re-embedding needed

        return True  # Model changed or first time
    except Exception:
        return True  # Be safe — assume changed


def _record_embedding_model(db: DBClient) -> None:
    """Record the current embedding model so we can detect changes."""
    settings = get_settings()
    current_model = f"{settings.embeddings_provider}:{settings.embedding_model}:{settings.embedding_dim}"
    try:
        with db._lock:
            db.conn.execute(
                """INSERT INTO ingest_log (file_name, file_hash, row_count)
                   VALUES (%s, '__embedding_model__', 0)
                   ON CONFLICT (file_hash) DO UPDATE SET file_name = EXCLUDED.file_name""",
                (current_model,),
            )
    except Exception as exc:
        logger.warning("[embeddings] failed to record model metadata: %s", exc)


def _reembed_all_sync(db: DBClient, embedder) -> None:
    """Re-generate ALL embeddings (theme_stats + questions) with the current model."""
    # 1. Re-embed theme_stats
    with db._lock:
        all_stats = db.conn.execute("SELECT id, tema, subtema FROM theme_stats").fetchall()
    stats_list = [dict(r) for r in all_stats]

    if stats_list:
        logger.info("[auto-reembed] re-generating embeddings for %d theme_stats rows", len(stats_list))
        texts = [f"{s['tema']} {s['subtema'] or ''}" for s in stats_list]
        embeddings = embedder.embed_batch(texts)
        updates = [(s["id"], emb) for s, emb in zip(stats_list, embeddings)]
        db.update_theme_stat_embeddings_batch(updates)
        logger.info("[auto-reembed] updated %d theme_stats embeddings", len(updates))

    # 2. Re-embed questions
    with db._lock:
        all_questions = db.conn.execute(
            "SELECT id, raw_text, tema_normalized, subtema_normalized FROM questions"
        ).fetchall()
    questions_list = [dict(r) for r in all_questions]

    if questions_list:
        logger.info("[auto-reembed] re-generating embeddings for %d questions rows", len(questions_list))
        texts = [
            f"{q['tema_normalized'] or ''} {q['subtema_normalized'] or ''} {q['raw_text']}"
            for q in questions_list
        ]
        for i in range(0, len(questions_list), 100):
            chunk_q = questions_list[i : i + 100]
            chunk_t = texts[i : i + 100]
            embs = embedder.embed_batch(chunk_t)
            with db._lock:
                with db.conn.transaction():
                    with db.conn.cursor() as cur:
                        cur.executemany(
                            "UPDATE questions SET embedding = %s::vector WHERE id = %s",
                            [(str(emb), q["id"]) for q, emb in zip(chunk_q, embs)],
                        )
            logger.info("[auto-reembed] questions batch %d-%d done", i, min(i + 100, len(questions_list)))

    db.clear_cache()
    _record_embedding_model(db)
    logger.info("[auto-reembed] complete: %d theme_stats + %d questions", len(stats_list), len(questions_list))


def _ensure_theme_stats_embeddings(db: DBClient, embedder) -> None:
    """Ensure all theme_stats have embeddings. Auto-detects model changes."""
    # Check if the embedding model changed — if so, re-embed EVERYTHING
    if _check_embedding_model_changed(db):
        logger.warning(
            "[embeddings] embedding model changed — re-generating ALL embeddings automatically"
        )
        _reembed_all_sync(db, embedder)
        return

    # Otherwise, just fill in any NULL embeddings (new rows)
    pending = db.get_theme_stats_without_embeddings()
    if not pending:
        return
    logger.info("[embeddings] generating embeddings for %d theme_stats rows with embedding=NULL", len(pending))
    texts = [f"{s['tema']} {s['subtema'] or ''}" for s in pending]
    embeddings = embedder.embed_batch(texts)
    updates = [(s["id"], emb) for s, emb in zip(pending, embeddings)]
    db.update_theme_stat_embeddings_batch(updates)
    logger.info("[embeddings] updated %d theme_stats embeddings", len(updates))


@app.post("/reembed")
async def reembed_all():
    """Re-generate ALL embeddings using the current model.

    MUST be called after switching embedding providers/models (e.g. local → OpenAI)
    because embeddings from different models live in incompatible vector spaces.
    """
    db = get_db()
    embedder = get_embedder()

    # 1. Re-embed theme_stats (all rows, not just NULL)
    with db._lock:
        all_stats = db.conn.execute(
            "SELECT id, tema, subtema FROM theme_stats"
        ).fetchall()
    stats_list = [dict(r) for r in all_stats]

    if stats_list:
        logger.info("[reembed] re-generating embeddings for %d theme_stats rows", len(stats_list))
        texts = [f"{s['tema']} {s['subtema'] or ''}" for s in stats_list]
        embeddings = embedder.embed_batch(texts)
        updates = [(s["id"], emb) for s, emb in zip(stats_list, embeddings)]
        db.update_theme_stat_embeddings_batch(updates)
        logger.info("[reembed] updated %d theme_stats embeddings", len(updates))

    # 2. Re-embed questions (all rows, not just NULL)
    with db._lock:
        all_questions = db.conn.execute(
            "SELECT id, raw_text, tema_normalized, subtema_normalized FROM questions"
        ).fetchall()
    questions_list = [dict(r) for r in all_questions]

    if questions_list:
        logger.info("[reembed] re-generating embeddings for %d questions rows", len(questions_list))
        texts = [
            f"{q['tema_normalized'] or ''} {q['subtema_normalized'] or ''} {q['raw_text']}"
            for q in questions_list
        ]
        # Batch in chunks of 100 to avoid API limits
        for i in range(0, len(questions_list), 100):
            chunk_q = questions_list[i : i + 100]
            chunk_t = texts[i : i + 100]
            embeddings = embedder.embed_batch(chunk_t)
            with db._lock:
                with db.conn.transaction():
                    with db.conn.cursor() as cur:
                        cur.executemany(
                            "UPDATE questions SET embedding = %s::vector WHERE id = %s",
                            [(str(emb), q["id"]) for q, emb in zip(chunk_q, embeddings)],
                        )
            logger.info("[reembed] questions batch %d-%d done", i, min(i + 100, len(questions_list)))

    # Clear all caches since embeddings changed
    db.clear_cache()

    return {
        "message": "Re-embedding complete",
        "theme_stats_updated": len(stats_list),
        "questions_updated": len(questions_list),
    }


def _create_llm_judge():
    """Create an LLMJudge if config allows, else None."""
    from src.llm.judge import LLMJudge

    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("[LLM Judge] no OPENAI_API_KEY set — skipping")
        return None
    return LLMJudge(
        api_key=settings.openai_api_key,
        model=settings.llm_judge_model,
        base_url=settings.llm_judge_base_url or None,
    )


@app.post("/reconcile")
async def reconcile(file: UploadFile = File(...), use_llm: bool = False):
    logger.info("/reconcile called with file: %s (use_llm=%s)", file.filename, use_llm)
    db = get_db()
    embedder = get_embedder()

    tmp_path = await _save_upload(file, ".xlsx")
    logger.info("saved uploaded file to %s", tmp_path)

    try:
        try:
            logger.info("[reconcile] attempting to read Excel from %s", tmp_path.name)
            input_rows = read_excel(tmp_path)
            logger.info("[reconcile] successfully parsed %d rows", len(input_rows))
        except Exception as exc:
            logger.exception("[reconcile] failed to read Excel input %s", tmp_path.name)
            raise HTTPException(
                status_code=400,
                detail=f"could not parse excel file: {exc}",
            )

        # Ensure theme_stats have embeddings
        _ensure_theme_stats_embeddings(db, embedder)

        # Optional LLM judge
        llm_judge = None
        if use_llm:
            llm_judge = _create_llm_judge()
            if llm_judge:
                logger.info("[reconcile] LLM judge enabled")

        logger.info("[reconcile] starting reconciliation pipeline...")
        results = reconcile_all(input_rows, embedder, db, llm_judge=llm_judge)
        logger.info(
            "[reconcile] reconciliation complete: %d rows produced", len(results)
        )

        logger.info("[reconcile] starting reverse coverage analysis...")
        reverse_rows = reverse_coverage(results, embedder, db)
        logger.info("[reconcile] reverse coverage: %d rows produced", len(reverse_rows))

        project_root = Path(__file__).parent.parent
        output_dir = project_root / "tables"
        output_dir.mkdir(exist_ok=True)
        logger.info("[reconcile] output directory: %s", output_dir.absolute())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"ranking_output_{timestamp}.xlsx"
        out_path = output_dir / out_filename

        write_excel(results, out_path, reverse_rows=reverse_rows)
        logger.info(
            "[reconcile] wrote output workbook to: %s",
            out_path.absolute(),
        )

        return FileResponse(
            path=str(out_path),
            filename=out_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.on_event("startup")
async def on_startup():
    settings = get_settings()
    level = settings.log_level.upper()
    try:
        logger.setLevel(level)
        import logging as _logging

        _logging.getLogger().setLevel(level)
    except Exception:
        logger.warning("invalid log level '%s', falling back to INFO", level)

    logger.info("Starting Exam Reconciler API")
    logger.info(
        "Embeddings provider: %s | DB: %s",
        settings.embeddings_provider,
        "supabase" if settings.use_supabase else "direct-postgres",
    )


from fastapi.exceptions import RequestValidationError
from fastapi import status


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(
        "request validation failed: %s | method=%s path=%s content-type=%s",
        exc,
        request.method,
        request.url.path,
        request.headers.get("content-type", "none"),
    )
    for error in exc.errors():
        logger.error("  validation error: %s", error)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        content_type = request.headers.get("content-type", "none")
        logger.info("[REQUEST] %s %s | content-type=%s", method, path, content_type)
        response = await call_next(request)
        return response


app.add_middleware(LoggingMiddleware)
