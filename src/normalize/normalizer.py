from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import NamedTuple

from src.utils.logging import logger


class ColorInfo(NamedTuple):
    """Color classification information."""

    name: str
    hex_code: str
    min_threshold: int


# ============================================================================
# COMPREHENSIVE MEDICAL ABBREVIATIONS & SYNONYMS
# ============================================================================
MEDICAL_ABBREVIATIONS: dict[str, str] = {
    # Cardiologia
    "pcr": "parada cardiorrespiratória",
    "icc": "insuficiência cardíaca congestiva",
    "iam": "infarto agudo do miocárdio",
    "sca": "síndrome coronariana aguda",
    "avc": "acidente vascular cerebral",
    "tvp": "trombose venosa profunda",
    "tep": "tromboembolismo pulmonar",
    "has": "hipertensão arterial sistêmica",
    "rcp": "ressuscitação cardiopulmonar",
    "bls": "suporte básico de vida",
    "acls": "suporte avançado de vida em cardiologia",
    "bradicardia": "bradicardia",
    "taquicardia": "taquicardia",
    "arritmia": "arritmia",
    "miocardiopatia": "miocardiopatia",
    "endocardite": "endocardite",
    # Pulmonar
    "dpoc": "doença pulmonar obstrutiva crônica",
    "asma": "asma",
    "tuberculose": "tuberculose",
    "tb": "tuberculose",
    "pneumonia": "pneumonia",
    "pleurite": "pleurite",
    "edema pulmonar": "edema pulmonar",
    "sindrome da angustia respiratoria": "síndrome do desconforto respiratório agudo",
    "sdra": "síndrome do desconforto respiratório agudo",
    "ards": "síndrome do desconforto respiratório agudo",
    # Renal e Urinário
    "itu": "infecção do trato urinário",
    "ira": "insuficiência renal aguda",
    "irc": "insuficiência renal crônica",
    "drc": "doença renal crônica",
    "glomerulonefrite": "glomerulonefrite",
    "pielonefrite": "pielonefrite",
    "cistite": "cistite",
    "nefrite": "nefrite",
    # Endócrino e Metabólico
    "dm": "diabetes mellitus",
    "dm1": "diabetes mellitus tipo 1",
    "dm2": "diabetes mellitus tipo 2",
    "diabetes": "diabetes mellitus",
    "cetoacidose": "cetoacidose diabética",
    "hipoglicemia": "hipoglicemia",
    "hiperglicemia": "hiperglicemia",
    "tireoidite": "tireoidite",
    "hipertireoidismo": "hipertireoidismo",
    "hipotireoidismo": "hipotireoidismo",
    "bócio": "bócio",
    # Hematologia
    "anemia": "anemia",
    "leucemia": "leucemia",
    "linfoma": "linfoma",
    "trombocitopenia": "trombocitopenia",
    "coagulopatia": "coagulopatia",
    "síndrome hemolítica urêmica": "síndrome hemolítica urêmica",
    "púrpura": "púrpura",
    # Infecções e Imunologia
    "hiv": "vírus da imunodeficiência humana",
    "aids": "síndrome da imunodeficiência adquirida",
    "dst": "doença sexualmente transmissível",
    "ist": "infecção sexualmente transmissível",
    "hepatite": "hepatite",
    "meningite": "meningite",
    "encefalite": "encefalite",
    "sepse": "sepse",
    "choque séptico": "choque séptico",
    "covid": "covid-19",
    "covid19": "covid-19",
    # Trauma e Urgência
    "tce": "traumatismo cranioencefálico",
    "atls": "suporte avançado de vida no trauma",
    "traumatismo": "traumatismo",
    "hemorragia": "hemorragia",
    "choque": "choque",
    "queimadura": "queimadura",
    "poisoning": "intoxicação",
    "toxicologia": "toxicologia",
    # Neurologia
    "ictus": "acidente vascular cerebral",
    "isquemia cerebral": "isquemia cerebral",
    "hemorrhagia cerebral": "hemorragia cerebral",
    "epilepsia": "epilepsia",
    "convulsão": "convulsão",
    "parkinson": "doença de parkinson",
    "alzheimer": "doença de alzheimer",
    "esclerose múltipla": "esclerose múltipla",
    "sme": "esclerose múltipla",
    "síndrome de guillain-barré": "síndrome de guillain-barré",
    "sgb": "síndrome de guillain-barré",
    # Gastroenterologia
    "gastroenterite": "gastroenterite",
    "gastrite": "gastrite",
    "úlcera péptica": "úlcera péptica",
    "refluxo": "refluxo gastroesofágico",
    "cirrose": "cirrose",
    "hepatomegalia": "hepatomegalia",
    "apendicite": "apendicite",
    "obstrução intestinal": "obstrução intestinal",
    "colite": "colite",
    "crohn": "doença de crohn",
    # Obstetrícia e Ginecologia
    "dheg": "doença hipertensiva específica da gestação",
    "pré-eclâmpsia": "pré-eclâmpsia",
    "eclâmpsia": "eclâmpsia",
    "rn": "recém-nascido",
    "parto": "parto",
    "gestação": "gestação",
    "menstruação": "menstruação",
    "endometriose": "endometriose",
    "síndrome dos ovários policísticos": "síndrome dos ovários policísticos",
    "sop": "síndrome dos ovários policísticos",
    "infertilidade": "infertilidade",
    # Pediatria
    "lactente": "lactente",
    "creche": "creche",
    "pré-escolar": "pré-escolar",
    "escolar": "escolar",
    "adolescente": "adolescente",
    "diarréia infantil": "diarréia infantil",
    "cólica": "cólica",
    "dermatite de fralda": "dermatite de fralda",
    # Dermatologia
    "dermatite": "dermatite",
    "eczema": "eczema",
    "psoríase": "psoríase",
    "acne": "acne",
    "rosácea": "rosácea",
    "urticária": "urticária",
    "prurido": "prurido",
    "infecção fúngica": "infecção fúngica",
    "micose": "micose",
    # Reumatologia
    "artrite": "artrite",
    "artrose": "artrose",
    "lúpus": "lúpus eritematoso sistêmico",
    "les": "lúpus eritematoso sistêmico",
    "febre reumática": "febre reumática",
    "gota": "gota",
    # Oncologia
    "câncer": "câncer",
    "tumor": "tumor",
    "neoplasia": "neoplasia",
    "carcinoma": "carcinoma",
    "sarcoma": "sarcoma",
    "melanoma": "melanoma",
    "quimioterapia": "quimioterapia",
    "radioterapia": "radioterapia",
    # Oftalmologia
    "miopia": "miopia",
    "hipermetropia": "hipermetropia",
    "astigmatismo": "astigmatismo",
    "catarata": "catarata",
    "glaucoma": "glaucoma",
    "ceratite": "ceratite",
    "conjuntivite": "conjuntivite",
    "retinite": "retinite",
    # Otorrinolaringologia
    "otite": "otite",
    "sinusite": "sinusite",
    "rinite": "rinite",
    "faringite": "faringite",
    "laringite": "laringite",
    "adenoidite": "adenoidite",
    "tonsilite": "tonsilite",
    # Psiquiatria e Saúde Mental
    "depressão": "depressão",
    "ansiedade": "transtorno de ansiedade",
    "bipolaridade": "transtorno bipolar",
    "esquizofrenia": "esquizofrenia",
    "transtorno de personalidade": "transtorno de personalidade",
    "autismo": "transtorno do espectro autista",
    "tea": "transtorno do espectro autista",
    "tdah": "transtorno do déficit de atenção com hiperatividade",
    "psicose": "psicose",
    # Equipamentos e Técnicas
    "sng": "sonda nasogástrica",
    "svc": "sonda vesical de demora",
    "ventilação mecanica": "ventilação mecânica",
    "traqueostomia": "traqueostomia",
    "linha venosa central": "cateter venoso central",
    "swan": "cateter de swan-ganz",
    "hemograma": "hemograma",
    "urocultura": "urocultura",
    "hemocultura": "hemocultura",
    "ressonância": "ressonância magnética",
    "tc": "tomografia computadorizada",
    "raio x": "radiografia",
    "ekg": "eletrocardiograma",
    "ecg": "eletrocardiograma",
    "ecodopplercardiograma": "ecodopplercardiograma",
}

# Aliases for common variations
MEDICAL_ALIASES: dict[str, str] = {
    "pressão alta": "hipertensão arterial sistêmica",
    "açúcar no sangue": "diabetes mellitus",
    "doença do coração": "doença cardiovascular",
    "infarto": "infarto agudo do miocárdio",
    "derrame cerebral": "acidente vascular cerebral",
    "trombose": "tromboembolismo",
    "crise convulsiva": "convulsão",
    "desfibrilação": "ressuscitação cardiopulmonar",
    "cpap": "pressão positiva contínua das vias aéreas",
    "bipap": "pressão positiva em dois níveis",
}

# Specialty categories for better context
SPECIALTY_MAPPING: dict[str, list[str]] = {
    "cardiologia": [
        "pcr",
        "icc",
        "iam",
        "sca",
        "avc",
        "tvp",
        "tep",
        "has",
        "rcp",
        "bls",
        "acls",
        "bradicardia",
        "taquicardia",
    ],
    "pneumologia": [
        "dpoc",
        "asma",
        "tuberculose",
        "tb",
        "pneumonia",
        "pleurite",
        "edema pulmonar",
        "sdra",
        "ards",
    ],
    "nefrologia": [
        "itu",
        "ira",
        "irc",
        "drc",
        "glomerulonefrite",
        "pielonefrite",
        "cistite",
        "nefrite",
    ],
    "endocrinologia": [
        "dm",
        "dm1",
        "dm2",
        "diabetes",
        "cetoacidose",
        "hipoglicemia",
        "hiperglicemia",
        "tireoidite",
        "hipertireoidismo",
        "hipotireoidismo",
    ],
    "hematologia": ["anemia", "leucemia", "linfoma", "trombocitopenia", "coagulopatia"],
    "infectologia": [
        "hiv",
        "aids",
        "dst",
        "ist",
        "hepatite",
        "meningite",
        "sepse",
        "covid",
        "covid19",
    ],
    "neurologia": [
        "tce",
        "ictus",
        "isquemia cerebral",
        "epilepsia",
        "convulsão",
        "parkinson",
        "alzheimer",
        "esclerose múltipla",
    ],
}

COLOR_THRESHOLDS = [
    ColorInfo("vermelho", "#EF4444", 6),
    ColorInfo("laranja", "#F97316", 4),
    ColorInfo("amarelo", "#EAB308", 2),
    ColorInfo("verde", "#22C55E", 1),
    ColorInfo("azul", "#3B82F6", 0),
]


# ============================================================================
# NORMALIZATION ENGINE
# ============================================================================


class TextNormalizer:
    """Intelligent text normalization with fuzzy matching and context awareness."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._similarity_threshold = 0.85

    def normalize_text(self, text: str, preserve_pipe: bool = True) -> str:
        """
        Normalize text with accent removal, lowercasing, and smart punctuation handling.

        Args:
            text: Raw text to normalize
            preserve_pipe: If True, keeps pipe characters for tema | subtema splits

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Check cache
        cache_key = f"{text}|{preserve_pipe}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Normalize
        normalized = text.strip().lower()

        # Remove accents (NFKD normalization)
        nfkd = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Remove punctuation (except pipe if requested)
        if preserve_pipe:
            normalized = re.sub(r"[^\w\s|]", "", normalized)
        else:
            normalized = re.sub(r"[^\w\s]", "", normalized)

        # Collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Cache and return
        self._cache[cache_key] = normalized
        return normalized

    def find_best_synonym(self, text: str) -> str | None:
        """
        Find the best synonym match using fuzzy matching if exact match fails.

        Args:
            text: Text to find synonym for

        Returns:
            Best match from synonyms, or None if not found
        """
        if not text:
            return None

        key = text.strip().lower()

        # Try exact match first
        if key in MEDICAL_ABBREVIATIONS:
            return MEDICAL_ABBREVIATIONS[key]

        if key in MEDICAL_ALIASES:
            return MEDICAL_ALIASES[key]

        # Try fuzzy match
        best_match = None
        best_ratio = 0.0

        all_keys = list(MEDICAL_ABBREVIATIONS.keys()) + list(MEDICAL_ALIASES.keys())
        for abbr in all_keys:
            ratio = SequenceMatcher(None, key, abbr).ratio()
            if ratio > best_ratio and ratio >= self._similarity_threshold:
                best_ratio = ratio
                best_match = abbr

        if best_match:
            if best_match in MEDICAL_ABBREVIATIONS:
                return MEDICAL_ABBREVIATIONS[best_match]
            else:
                return MEDICAL_ALIASES[best_match]

        return None

    def apply_synonyms(self, text: str) -> str:
        """
        Apply medical synonyms and expansions.

        Maintains backward compatibility: returns synonym if found,
        otherwise returns original text unchanged.

        Args:
            text: Text to expand

        Returns:
            Expanded text with synonyms applied, or original text
        """
        if not text:
            return ""

        key = text.strip().lower()

        # Try exact match first (case-insensitive key)
        if key in MEDICAL_ABBREVIATIONS:
            result = MEDICAL_ABBREVIATIONS[key]
            logger.debug(
                "[apply_synonyms] expanded '%s' → '%s'", text[:50], result[:50]
            )
            return result

        if key in MEDICAL_ALIASES:
            result = MEDICAL_ALIASES[key]
            logger.debug(
                "[apply_synonyms] expanded '%s' → '%s'", text[:50], result[:50]
            )
            return result

        # If no exact match, return original text (preserves case)
        return text

    def normalize_tema_subtema(
        self, tema_raw: str, subtema_raw: str | None = None
    ) -> tuple[str, str | None]:
        """
        Normalize tema and subtema with intelligent parsing.

        Handles:
        - Pipe splits (tema | subtema)
        - Nested pipes
        - Empty or None values
        - Synonym expansion

        Args:
            tema_raw: Theme text (may contain pipe-separated subtema)
            subtema_raw: Optional explicit subtema

        Returns:
            (normalized_tema, normalized_subtema) tuple
        """
        if not tema_raw:
            return "", None

        # Handle pipe-separated format (tema | subtema)
        tema_part = tema_raw.strip()
        subtema_part = None

        if "|" in tema_part:
            parts = [p.strip() for p in tema_part.split("|", 1)]
            tema_part = parts[0]
            subtema_part = parts[1] if len(parts) > 1 and parts[1] else None

        # Expand synonyms for tema
        expanded_tema = self.apply_synonyms(tema_part)

        # Use explicit subtema if provided, otherwise use parsed one
        if subtema_raw and str(subtema_raw).strip().lower() not in ("", "nan", "none"):
            final_subtema = self.apply_synonyms(str(subtema_raw).strip())
        elif subtema_part:
            final_subtema = self.apply_synonyms(subtema_part)
        else:
            final_subtema = None

        logger.debug(
            "[normalize_tema_subtema] '%s' | '%s' → '%s' | '%s'",
            tema_raw[:40],
            subtema_raw or "",
            expanded_tema[:40],
            final_subtema or "",
        )

        return expanded_tema, final_subtema

    def classify_color(self, num_questions: int) -> tuple[str, str]:
        """
        Classify severity color based on question count.

        Args:
            num_questions: Number of questions/occurrences

        Returns:
            (color_name, hex_code) tuple
        """
        if num_questions < 0:
            logger.warning("[classify_color] negative count: %d", num_questions)
            num_questions = 0

        for color_info in COLOR_THRESHOLDS:
            if num_questions >= color_info.min_threshold:
                logger.debug(
                    "[classify_color] %d questions → %s", num_questions, color_info.name
                )
                return color_info.name, color_info.hex_code

        return "azul", "#3B82F6"

    def get_specialty_from_text(self, text: str) -> str | None:
        """
        Detect medical specialty from text content.

        Args:
            text: Text to analyze

        Returns:
            Specialty name if detected, else None
        """
        if not text:
            return None

        normalized = self.normalize_text(text, preserve_pipe=False)

        for specialty, keywords in SPECIALTY_MAPPING.items():
            for keyword in keywords:
                if keyword in normalized:
                    logger.debug(
                        "[get_specialty_from_text] detected '%s' from '%s'",
                        specialty,
                        text[:50],
                    )
                    return specialty

        return None

    def estimate_confidence(self, original: str, normalized: str) -> float:
        """
        Estimate confidence of normalization.

        Args:
            original: Original text
            normalized: Normalized text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not original or not normalized:
            return 0.0

        similarity = SequenceMatcher(None, original.lower(), normalized.lower()).ratio()
        return min(1.0, similarity + 0.1)  # Slight boost for successful normalization


# ============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================================

_normalizer = TextNormalizer()


def normalize_text(text: str) -> str:
    """Convenience function for backward compatibility."""
    return _normalizer.normalize_text(text, preserve_pipe=True)


def apply_synonyms(text: str) -> str:
    """Convenience function for backward compatibility."""
    return _normalizer.apply_synonyms(text)


def normalize_tema_subtema(
    tema_raw: str, subtema_raw: str | None = None
) -> tuple[str, str | None]:
    """Convenience function for backward compatibility."""
    return _normalizer.normalize_tema_subtema(tema_raw, subtema_raw)


def classify_color(num_questions: int) -> tuple[str, str]:
    """Convenience function for backward compatibility."""
    return _normalizer.classify_color(num_questions)
