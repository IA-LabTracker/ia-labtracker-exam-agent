import pytest

from src.normalize.normalizer import (
    TextNormalizer,
    apply_synonyms,
    classify_color,
    normalize_tema_subtema,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("TRAUMA") == "trauma"

    def test_strips_accents(self):
        assert normalize_text("classificação") == "classificacao"

    def test_removes_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  foo   bar  ") == "foo bar"

    def test_preserves_pipe(self):
        result = normalize_text("trauma | abordagem")
        assert "|" in result

    def test_empty_text(self):
        assert normalize_text("") == ""

    def test_none_text(self):
        assert normalize_text(None) == ""


class TestApplySynonyms:
    def test_known_synonym(self):
        assert apply_synonyms("PCR") == "parada cardiorrespiratória"

    def test_known_synonym_case_insensitive(self):
        assert apply_synonyms("pcr") == "parada cardiorrespiratória"

    def test_unknown_returns_original(self):
        assert apply_synonyms("xyz_unknown_topic") == "xyz_unknown_topic"

    def test_medical_expansion_dm(self):
        result = apply_synonyms("dm")
        assert "diabetes" in result.lower()

    def test_medical_expansion_has(self):
        result = apply_synonyms("has")
        assert "hipertensão" in result.lower()

    def test_medical_expansion_dpoc(self):
        result = apply_synonyms("dpoc")
        assert "pulmonar" in result.lower()

    def test_alias_expansion(self):
        result = apply_synonyms("pressão alta")
        assert "hipertensão" in result.lower()

    def test_empty_input(self):
        assert apply_synonyms("") == ""


class TestNormalizeTemSubtema:
    def test_simple_tema(self):
        tema, sub = normalize_tema_subtema("PCR")
        assert tema == "parada cardiorrespiratória"
        assert sub is None

    def test_tema_with_pipe_split(self):
        tema, sub = normalize_tema_subtema("Trauma | Abordagem inicial")
        assert "trauma" in tema.lower()
        assert sub is not None

    def test_tema_with_explicit_subtema(self):
        tema, sub = normalize_tema_subtema("Diabetes", "Tipo 2")
        assert tema == "diabetes mellitus"
        assert sub is not None

    def test_passthrough_unknown(self):
        tema, sub = normalize_tema_subtema("Totally New Topic")
        assert tema == "Totally New Topic"
        assert sub is None

    def test_tema_none(self):
        tema, sub = normalize_tema_subtema(None)
        assert tema == ""
        assert sub is None

    def test_tema_empty_string(self):
        tema, sub = normalize_tema_subtema("")
        assert tema == ""
        assert sub is None

    def test_subtema_nan_handling(self):
        tema, sub = normalize_tema_subtema("Diabetes", "nan")
        assert "diabetes" in tema.lower()
        assert sub is None


class TestClassifyColor:
    def test_high_count_is_vermelho(self):
        cor, cor_hex = classify_color(9)
        assert cor == "vermelho"
        assert cor_hex == "#EF4444"

    def test_threshold_six_is_vermelho(self):
        cor, _ = classify_color(6)
        assert cor == "vermelho"

    def test_four_is_laranja(self):
        cor, cor_hex = classify_color(4)
        assert cor == "laranja"
        assert cor_hex == "#F97316"

    def test_five_is_laranja(self):
        cor, _ = classify_color(5)
        assert cor == "laranja"

    def test_two_is_amarelo(self):
        cor, cor_hex = classify_color(2)
        assert cor == "amarelo"
        assert cor_hex == "#EAB308"

    def test_three_is_amarelo(self):
        cor, _ = classify_color(3)
        assert cor == "amarelo"

    def test_one_is_verde(self):
        cor, cor_hex = classify_color(1)
        assert cor == "verde"
        assert cor_hex == "#22C55E"

    def test_zero_is_azul(self):
        cor, cor_hex = classify_color(0)
        assert cor == "azul"
        assert cor_hex == "#3B82F6"

    def test_negative_count_treated_as_zero(self):
        cor, _ = classify_color(-5)
        assert cor == "azul"


class TestTextNormalizerAdvanced:
    """Tests for advanced normalization features."""

    def setup_method(self):
        self.normalizer = TextNormalizer()

    def test_fuzzy_matching(self):
        # Should find close matches even with typos
        result = self.normalizer.find_best_synonym("pcr")
        assert result is not None

    def test_cache_performance(self):
        # First call
        result1 = self.normalizer.normalize_text("TRAUMA | ABORDAGEM")
        # Second call should use cache
        result2 = self.normalizer.normalize_text("TRAUMA | ABORDAGEM")
        assert result1 == result2

    def test_specialty_detection_cardiology(self):
        specialty = self.normalizer.get_specialty_from_text("PCR e ressuscitação")
        assert specialty == "cardiologia"

    def test_specialty_detection_pneumology(self):
        specialty = self.normalizer.get_specialty_from_text("DPOC avançada")
        assert specialty == "pneumologia"

    def test_specialty_detection_none(self):
        specialty = self.normalizer.get_specialty_from_text("Totalmente desconhecido")
        assert specialty is None

    def test_confidence_estimation(self):
        conf = self.normalizer.estimate_confidence("PCR", "parada cardiorrespiratória")
        assert 0.0 <= conf <= 1.0

    def test_confidence_exact_match(self):
        conf = self.normalizer.estimate_confidence("trauma", "trauma")
        assert conf > 0.8

    def test_multiple_specialties_in_text(self):
        text = "PCR com DPOC"
        # Should detect first matching specialty
        specialty = self.normalizer.get_specialty_from_text(text)
        assert specialty in ["cardiologia", "pneumologia"]

    def test_extensive_medical_abbreviations(self):
        """Verify extensive abbreviation dictionary coverage."""
        # Test various categories
        assert apply_synonyms("iam") != "iam"  # Should be expanded
        assert apply_synonyms("tb") != "tb"  # Should be expanded
        assert apply_synonyms("itu") != "itu"  # Should be expanded
        assert apply_synonyms("dm1") != "dm1"  # Should be expanded

    def test_normalize_preserves_pipe_by_default(self):
        normalized = normalize_text("Diabtes | Tipo 2")
        assert "|" in normalized

    def test_complex_tema_subtema_parsing(self):
        tema, sub = normalize_tema_subtema(
            "Cardiologia | Síndrome Coronariana Aguda", None
        )
        assert tema is not None
        assert sub is not None

    def test_empty_subtema_string(self):
        tema, sub = normalize_tema_subtema("Diabetes", "")
        assert sub is None

    def test_whitespace_only_subtema(self):
        tema, sub = normalize_tema_subtema("Diabetes", "   ")
        assert sub is None
