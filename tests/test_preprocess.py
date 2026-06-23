"""Tests for the text preprocessing pipeline."""
from news_search.preprocess import Preprocessor, _FALLBACK_STOPWORDS, _load_stopwords


def test_lowercases_and_strips_punctuation():
    pre = Preprocessor()
    toks = pre.tokenize("Hello, WORLD!!! O'Brien & co.")
    assert toks == [t.lower() for t in toks]
    # every surviving token is alphanumeric (punctuation removed before tokenizing)
    assert all(tok.isalnum() for tok in toks)


def test_removes_stopwords():
    pre = Preprocessor()
    toks = pre.tokenize("the cat and the dog")
    assert "the" not in toks and "and" not in toks
    assert any(t.startswith("cat") for t in toks)
    assert any(t.startswith("dog") for t in toks)


def test_stemming_applied():
    pre = Preprocessor()
    assert pre.stem("running") == "run"
    toks = pre.tokenize("running runs runner")
    assert all(t.startswith("run") for t in toks)


def test_empty_whitespace_and_punctuation_only():
    pre = Preprocessor()
    assert pre.tokenize("") == []
    assert pre.tokenize("    ") == []
    assert pre.tokenize("!!!@#$") == []


def test_query_and_document_normalised_identically():
    pre = Preprocessor()
    # Same words, different casing/punctuation -> identical token streams.
    assert pre.tokenize("Climate Change!") == pre.tokenize("climate, change")


def test_stopwords_source_nonempty():
    assert len(_FALLBACK_STOPWORDS) > 50
    assert len(_load_stopwords()) > 0
