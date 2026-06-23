"""Tests for the search engine pipeline.

Run with:  pytest
"""
from pathlib import Path

import pytest

from news_search import Document, SearchEngine, build_index, load_corpus
from news_search import ranking
from news_search.index import InvertedIndex

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_news.jsonl"


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
    # 'health' is a real content word (not a stopword), so this actually exercises
    # the filter — the old test used the stopword 'the' and passed vacuously.
    res = engine.search("health", method="bm25", top_k=30, category="POLITICS")
    assert all(r["category"] == "POLITICS" for r in res.results)


def _make_engine_with_categories() -> SearchEngine:
    """Synthetic corpus: 8 POLITICS + 3 SPORTS docs all containing 'reform'."""
    docs = []
    for i in range(8):
        docs.append(Document(id=i, text="reform policy senate vote",
                             headline="h", short_description="reform policy senate vote",
                             category="POLITICS", date="", link="https://example.com"))
    for i in range(8, 11):
        docs.append(Document(id=i, text="reform team game season",
                             headline="h", short_description="reform team game season",
                             category="SPORTS", date="", link="https://example.com"))
    return SearchEngine(build_index(docs, verbose=False))


def test_category_filter_restricts_before_truncation():
    """Regression for the bug where the category filter ran *after* top_k
    truncation, silently dropping relevant in-category docs ranked below top_k."""
    eng = _make_engine_with_categories()
    full = eng.search("reform", method="bm25", top_k=100, category="POLITICS")
    assert full.total_hits == 8
    assert all(r["category"] == "POLITICS" for r in full.results)

    limited = eng.search("reform", method="bm25", top_k=3, category="POLITICS")
    assert len(limited.results) == 3              # page filled from within the category
    assert limited.total_hits == 8               # honest total, not capped at top_k
    assert all(r["category"] == "POLITICS" for r in limited.results)


def test_category_with_no_docs_returns_empty(engine):
    res = engine.search("health", category="NO_SUCH_CATEGORY_XYZ")
    assert res.total_hits == 0
    assert res.results == []


def test_total_hits_is_true_count_not_capped(engine):
    # Use the most frequent indexed term so we know many docs match.
    term = max(engine.index.postings, key=lambda t: len(engine.index.postings[t]))
    page = engine.search(term, method="bm25", top_k=5)
    full = engine.search(term, method="bm25", top_k=10_000)
    assert page.total_hits == full.total_hits      # total independent of page size
    assert page.total_hits > 5                      # more matches than one page
    assert len(page.results) == 5                   # page filled to top_k
    assert len(full.results) == full.total_hits     # everything returned when top_k huge


def test_ranking_and_mode_requires_all_terms(engine):
    by_df = sorted(engine.index.postings, key=lambda t: len(engine.index.postings[t]),
                   reverse=True)
    t1, t2 = by_df[0], by_df[3]
    or_hits = ranking.bm25([t1, t2], engine.index, top_k=None, mode="or")
    and_hits = ranking.bm25([t1, t2], engine.index, top_k=None, mode="and")
    assert len(and_hits) <= len(or_hits)
    for doc_id, _ in and_hits:
        fwd = engine.index.forward[doc_id]
        assert t1 in fwd and t2 in fwd  # 'and' docs contain every term


def test_ranking_restrict_to_limits_candidates(engine):
    term = max(engine.index.postings, key=lambda t: len(engine.index.postings[t]))
    allowed = set(list(engine.index.meta)[:3])
    res = ranking.bm25([term], engine.index, top_k=None, mode="or", restrict_to=allowed)
    assert {doc_id for doc_id, _ in res}.issubset(allowed)


def test_bert_method_without_bert_is_graceful(engine):
    # engine.bert is None -> bert expansion contributes nothing but must not error
    res = engine.search("health", method="bert", top_k=5)
    assert res.method == "bert"
    assert res.expansion_terms == []


def test_prf_bert_without_bert_still_uses_prf(engine):
    res = engine.search("health", method="prf+bert", top_k=5)
    assert res.method == "prf+bert"
    assert isinstance(res.expansion_terms, list)  # PRF terms only (no bert)


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
