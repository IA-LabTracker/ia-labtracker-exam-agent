from typing import List, Dict, Optional
import pdfplumber
import re

from src.db.client import DBClient
from src.embeddings.embedder import Embedder


def parse_pdf(path: str) -> List[Dict]:
    """Extract question entries from a PDF file.

    Returned list contains dicts with keys: text, tema, subtema.
    Simple heuristics look for lines prefixed by numbers and for
    preceding "Tema"/"Subtema" markers. Results are best-effort; manual
    review may be required for unstructured PDFs.
    """
    entries: List[Dict] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()
            tema: Optional[str] = None
            subtema: Optional[str] = None
            buffer: List[str] = []
            for line in lines:
                m_tema = re.match(r"^Tema[:\s]+(.+)", line, re.I)
                m_sub = re.match(r"^Subtema[:\s]+(.+)", line, re.I)
                if m_tema:
                    tema = m_tema.group(1).strip()
                    continue
                if m_sub:
                    subtema = m_sub.group(1).strip()
                    continue
                if re.match(r"^\d+\.", line):
                    buffer.append(line)
                elif buffer:
                    buffer[-1] += " " + line
            for q in buffer:
                entries.append({"text": q, "tema": tema, "subtema": subtema})
    return entries


def ingest_pdf(
    path: str, db: Optional[DBClient] = None, embedder: Optional[Embedder] = None
) -> int:
    """Parse and upsert questions from a PDF file into the database.

    Returns the number of new/updated entries processed.
    """
    db = db or DBClient()
    embedder = embedder or Embedder()
    entries = parse_pdf(path)
    to_process: List[Dict] = []
    for e in entries:
        existing = db.fetch(
            "SELECT id, embedding FROM questions WHERE raw_text=%s", e["text"]
        )
        if existing and existing[0].get("embedding"):
            continue
        to_process.append(e)
    if not to_process:
        return 0
    # batch embed
    texts = [e["text"] for e in to_process]
    embeddings = embedder.embed(texts)
    for e, emb in zip(to_process, embeddings):
        existing = db.fetch("SELECT id FROM questions WHERE raw_text=%s", e["text"])
        if existing:
            db.execute(
                "UPDATE questions SET tema_normalized=%s, subtema_normalized=%s, embedding=%s WHERE id=%s",
                e.get("tema"),
                e.get("subtema"),
                emb,
                existing[0]["id"],
            )
        else:
            db.execute(
                "INSERT INTO questions(raw_text, tema_normalized, subtema_normalized, embedding) VALUES (%s,%s,%s,%s)",
                e["text"],
                e.get("tema"),
                e.get("subtema"),
                emb,
            )
    return len(to_process)
