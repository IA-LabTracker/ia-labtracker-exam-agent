from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pdfplumber

from src.normalize.normalizer import classify_color, normalize_tema_subtema
from src.utils.logging import logger


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_questions(
    pdf_path: str | Path,
    institution: str = "unknown",
    year: int | None = None,
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    logger.info("Extracting questions from %s", pdf_path.name)
    entries: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()
            tema: str | None = None
            subtema: str | None = None
            buffer: list[str] = []

            for line in lines:
                m_tema = re.match(r"^Tema[:\s]+(.+)", line, re.IGNORECASE)
                m_sub = re.match(r"^Subtema[:\s]+(.+)", line, re.IGNORECASE)
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
                norm_tema, norm_sub = normalize_tema_subtema(tema or "", subtema)
                entries.append(
                    {
                        "institution": institution,
                        "year": year,
                        "raw_text": q,
                        "tema_normalized": norm_tema,
                        "subtema_normalized": norm_sub,
                        "source_file": pdf_path.name,
                    }
                )

    logger.info("Extracted %d questions from %s", len(entries), pdf_path.name)
    return entries


_METRICS_LINE = re.compile(
    r"(\d+)[ºª°]\s+(.+?)\s*[—\-–]\s*([\d.,]+)%\s*\((\d+)\s*quest[õo]es?\)",
    re.IGNORECASE,
)


def extract_theme_stats(
    pdf_path: str | Path, institution: str = "unknown"
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    logger.info("Extracting theme stats from %s", pdf_path.name)
    results: list[dict[str, Any]] = []

    current_area: str | None = None
    current_tema: str | None = None
    section: str = "area"

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                lower = line.lower()
                if "área" in lower or "area" in lower:
                    section = "area"
                    continue
                if "tema" in lower and "subtema" not in lower:
                    section = "tema"
                    continue
                if "subtema" in lower:
                    section = "subtema"
                    continue

                m = _METRICS_LINE.match(line)
                if not m:
                    continue

                ranking = int(m.group(1))
                name = m.group(2).strip()
                pct = float(m.group(3).replace(",", "."))
                nq = int(m.group(4))
                cor, cor_hex = classify_color(nq)

                if section == "area":
                    current_area = name
                    current_tema = None
                elif section == "tema":
                    current_tema = name

                row: dict[str, Any] = {
                    "institution": institution,
                    "ranking": ranking,
                    "category": section,
                    "tema": current_tema or current_area or name,
                    "subtema": name if section == "subtema" else None,
                    "percentage": pct,
                    "num_questions": nq,
                    "cor": cor,
                    "cor_hex": cor_hex,
                    "source_file": pdf_path.name,
                }

                if section == "area":
                    row["tema"] = name
                    row["subtema"] = None
                elif section == "tema":
                    row["tema"] = name
                    row["subtema"] = None

                results.append(row)

    logger.info("Extracted %d theme stats from %s", len(results), pdf_path.name)
    return results
