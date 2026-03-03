from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from src.aggregator.consolidate import reconcile_all
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
    tmp_path = await _save_upload(file, ".pdf")

    try:
        fhash = file_hash(tmp_path)
        if db.file_already_ingested(fhash):
            return JSONResponse(
                {"message": f"File '{file.filename}' already ingested", "skipped": True}
            )

        stats = extract_theme_stats(tmp_path, institution=institution)
        count = db.upsert_theme_stats(stats)
        db.record_ingest(file.filename or "stats.pdf", fhash, count)

        return {"message": "Theme stats ingested", "stats_added": count}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/reconcile")
async def reconcile(file: UploadFile = File(...)):
    db = get_db()
    embedder = get_embedder()
    tmp_path = await _save_upload(file, ".xlsx")

    try:
        input_rows = read_excel(tmp_path)
        results = reconcile_all(input_rows, embedder, db)

        out_path = Path(tempfile.mktemp(suffix=".xlsx"))
        write_excel(results, out_path)

        def cleanup():
            out_path.unlink(missing_ok=True)

        return FileResponse(
            path=str(out_path),
            filename="ranking_output.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=BackgroundTask(cleanup),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Exam Reconciler API")
    settings = get_settings()
    logger.info(
        "Embeddings provider: %s | DB: %s",
        settings.embeddings_provider,
        "supabase" if settings.use_supabase else "direct-postgres",
    )
