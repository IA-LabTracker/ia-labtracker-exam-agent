from src.normalize.normalizer import normalize, normalize_tema_subtema


def test_basic_normalization():
    assert normalize("Abordagem Inicial!") == "initial approach"
    assert normalize("Hello, World") == "hello world"


def test_tema_subtema():
    t, s = normalize_tema_subtema("Trauma | Abordagem inicial")
    assert t == "trauma"
    assert s == "initial approach"
