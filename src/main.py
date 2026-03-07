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


def _ensure_theme_stats_embeddings(db: DBClient, embedder) -> None:
    pending = db.get_theme_stats_without_embeddings()
    if not pending:
        return
    texts = [f"{s['tema']} {s['subtema'] or ''}" for s in pending]
    embeddings = embedder.embed_batch(texts)
    for s, emb in zip(pending, embeddings):
        db.update_theme_stat_embedding(s["id"], emb)
    logger.info("Generated embeddings for %d theme_stats entries", len(pending))


@app.post("/reconcile")
async def reconcile(file: UploadFile = File(...)):
    logger.info("/reconcile called with file: %s", file.filename)
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

        logger.info("[reconcile] starting reconciliation pipeline...")
        results = reconcile_all(input_rows, embedder, db)
        logger.info(
            "[reconcile] reconciliation complete: %d rows produced", len(results)
        )

        logger.info("[reconcile] starting reverse coverage analysis...")
        reverse_rows = reverse_coverage(results, embedder, db)
        logger.info(
            "[reconcile] reverse coverage: %d rows produced", len(reverse_rows)
        )

        project_root = Path(__file__).parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"ranking_output_{timestamp}.xlsx"
        out_path = project_root / out_filename

        write_excel(results, out_path, reverse_rows=reverse_rows)
        logger.info(
            "[reconcile] wrote output workbook to project root: %s",
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
