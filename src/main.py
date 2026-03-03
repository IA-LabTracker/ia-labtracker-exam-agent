from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import logging
from src.utils.logging import configure_logging
from src.ingest import pdf_parser, excel_reader
from src.aggregator.consolidate import consolidate
from src.exporters.excel_writer import write_excel

configure_logging()
app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/pdf")
def ingest_pdf(file: UploadFile = File(...)):
    # sanitize filename to prevent directory traversal
    name = Path(file.filename).name
    tmp = Path("/tmp")
    tmp.mkdir(exist_ok=True)
    dest = tmp / name
    # optional: simple size limit (e.g. 20 MB)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    count = pdf_parser.ingest_pdf(str(dest))
    logger.info("ingested %d questions from %s", count, name)
    return {"imported": count}


@app.post("/reconcile")
def reconcile(file: UploadFile = File(...)):
    name = Path(file.filename).name
    tmp = Path("/tmp")
    tmp.mkdir(exist_ok=True)
    dest = tmp / name
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    rows = excel_reader.read_excel(str(dest))
    output = consolidate(rows)
    outpath = tmp / f"output_{name}"
    write_excel(output, str(outpath))
    return {"result_file": str(outpath)}
