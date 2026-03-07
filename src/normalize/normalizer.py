from __future__ import annotations

import re
import unicodedata

SYNONYMS: dict[str, str] = {
    "pcr": "parada cardiorrespiratória",
    "icc": "insuficiência cardíaca congestiva",
    "iam": "infarto agudo do miocárdio",
    "avc": "acidente vascular cerebral",
    "dpoc": "doença pulmonar obstrutiva crônica",
    "itu": "infecção do trato urinário",
    "hiv": "vírus da imunodeficiência humana",
    "dst": "doença sexualmente transmissível",
    "ist": "infecção sexualmente transmissível",
    "has": "hipertensão arterial sistêmica",
    "dm": "diabetes mellitus",
    "dm2": "diabetes mellitus tipo 2",
    "dm1": "diabetes mellitus tipo 1",
    "tce": "traumatismo cranioencefálico",
    "rcp": "ressuscitação cardiopulmonar",
    "bls": "suporte básico de vida",
    "acls": "suporte avançado de vida em cardiologia",
    "atls": "suporte avançado de vida no trauma",
    "sca": "síndrome coronariana aguda",
    "tvp": "trombose venosa profunda",
    "tep": "tromboembolismo pulmonar",
    "ira": "insuficiência renal aguda",
    "irc": "insuficiência renal crônica",
    "pré-eclâmpsia": "pré-eclâmpsia",
    "dheg": "doença hipertensiva específica da gestação",
    "rn": "recém-nascido",
    "sng": "sonda nasogástrica",
    "diabetes": "diabetes mellitus",
}

COLOR_THRESHOLDS = [
    (6, "vermelho", "#EF4444"),
    (4, "laranja", "#F97316"),
    (2, "amarelo", "#EAB308"),
    (1, "verde", "#22C55E"),
    (0, "azul", "#3B82F6"),
]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    nfkd = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in nfkd if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s|]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def apply_synonyms(text: str) -> str:
    key = text.strip().lower()
    return SYNONYMS.get(key, text)


def normalize_tema_subtema(
    tema_raw: str, subtema_raw: str | None = None
) -> tuple[str, str | None]:
    if not tema_raw:
        return "", None

    expanded = apply_synonyms(tema_raw.strip())

    if "|" in expanded:
        parts = [p.strip() for p in expanded.split("|", 1)]
        tema = apply_synonyms(parts[0])
        sub = apply_synonyms(parts[1]) if parts[1] else None
    else:
        tema = expanded
        sub = None

    if subtema_raw:
        sub = apply_synonyms(subtema_raw.strip())

    return tema, sub


def classify_color(num_questions: int) -> tuple[str, str]:
    for threshold, cor, cor_hex in COLOR_THRESHOLDS:
        if num_questions >= threshold:
            return cor, cor_hex
    return "azul", "#3B82F6"
