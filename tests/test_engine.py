"""Tests for the search engine pipeline.

Run with:  pytest
"""
from pathlib import Path

import pytest

from news_search import SearchEngine, build_index, load_corpus
from news_search.index import InvertedIndex

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_news.jsonl"


@pytest.fixture(scope="module")
def engine() -> SearchEngine:
    docs = load_corpus(SAMPLE)
    return SearchEngine(build_index(docs, verbose=False))


def test_corpus_loads():
    docs = load_corpus(SAMPLE)
    assert len(docs) > 100
    # The overwhelming majority of records have indexable text.
    assert sum(1 for d in docs if d.text) > 0.9 * len(docs)
    assert docs[0].id == 0


def test_index_stats(engine):
    assert engine.index.num_docs > 100
    assert engine.index.vocabulary_size > 500
    # IDF precomputed for every term
    assert all(idf >= 0 for idf in engine.index.idf.values())


def test_basic_search_returns_ranked_results(engine):
    res = engine.search("health", method="tfidf", top_k=10)
    assert res.total_hits > 0
    scores = [r["score"] for r in res.results]
    assert scores == sorted(scores, reverse=True)  # descending
    ranks = [r["rank"] for r in res.results]
    assert ranks == list(range(1, len(ranks) + 1))


def test_or_semantics_not_and(engine):
    """A multi-term query should still return docs even if no single doc has all
    terms (the original AND-only behaviour returned nothing here)."""
    res = engine.search("health technology economy", method="tfidf", top_k=10)
    assert res.total_hits > 0


def test_empty_and_nonsense_queries(engine):
    assert engine.search("").total_hits == 0
    assert engine.search("zzzqqqxyzzy").total_hits == 0


def test_category_filter(engine):
    res = engine.search("the", method="tfidf", top_k=30, category="POLITICS")
    assert all(r["category"] == "POLITICS" for r in res.results)


def test_prf_adds_expansion_terms(engine):
    res = engine.search("health", method="prf", top_k=10)
    # PRF should surface additional terms from the top documents
    assert isinstance(res.expansion_terms, list)
    assert all(t not in res.query.split() for t in res.expansion_terms)


def test_bm25_is_default_and_ranks(engine):
    res = engine.search("health")  # default method
    assert res.method == "bm25"
    assert res.total_hits > 0
    scores = [r["score"] for r in res.results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_avoids_trivial_short_doc_bias(engine):
    """BM25 should not let a 1-2 token headline dominate a multi-term query the
    way normalised-TF TF-IDF does."""
    assert engine.index.avg_doc_len > 0
    res = engine.search("climate change", method="bm25", top_k=5)
    top_lengths = [engine.index.doc_len[r["id"]] for r in res.results]
    # at least one of the top results is a real (non-trivial) document
    assert max(top_lengths) >= 8


def test_unknown_method_raises(engine):
    with pytest.raises(ValueError):
        engine.search("health", method="not-a-method")


def test_index_persistence_roundtrip(engine, tmp_path):
    p = tmp_path / "idx.pkl"
    engine.index.save(p)
    reloaded = InvertedIndex.load(p)
    assert reloaded.num_docs == engine.index.num_docs
    assert reloaded.vocabulary_size == engine.index.vocabulary_size
    res = SearchEngine(reloaded).search("health", top_k=5)
    assert res.total_hits > 0
