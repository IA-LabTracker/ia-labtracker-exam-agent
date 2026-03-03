import re
from typing import Tuple

SYNONYMS = {
    "abordagem inicial": "initial approach",
    "trauma inicio de abordagem": "Trauma | Abordagem inicial",
}


def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return SYNONYMS.get(t, t)


def normalize_tema_subtema(text: str) -> Tuple[str, str]:
    """Return normalized (tema, subtema) split by '|'."""
    if not text:
        return "", ""
    parts = [p.strip() for p in text.split("|")]
    tema = normalize(parts[0])
    sub = normalize(parts[1]) if len(parts) > 1 else ""
    return tema, sub
