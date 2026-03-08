import pytest

from src.normalize.normalizer import (
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


class TestApplySynonyms:
    def test_known_synonym(self):
        assert apply_synonyms("PCR") == "parada cardiorrespiratória"

    def test_known_synonym_case_insensitive(self):
        assert apply_synonyms("pcr") == "parada cardiorrespiratória"

    def test_unknown_returns_original(self):
        assert apply_synonyms("xyz_unknown_topic") == "xyz_unknown_topic"


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
