"""Tests for the offline evaluation harness."""
from news_search.evaluate import build_pseudo_qrels, evaluate


def test_build_pseudo_qrels_maps_known_terms(engine):
    qrels = build_pseudo_qrels(["covid vaccine", "election", "zzz nonsense"], engine)
    # 'zzz nonsense' maps to no category and must be excluded.
    assert "zzz nonsense" not in qrels
    # qrels values are sets of doc_ids drawn from the index.
    for q, rel in qrels.items():
        assert isinstance(rel, set)
        assert rel.issubset(set(engine.index.meta))


def test_evaluate_returns_bounded_metrics(engine):
    rows = evaluate(
        engine,
        ["covid vaccine health", "election president vote"],
        methods=["bm25", "tfidf"],
        top_k=10,
    )
    assert {r.method for r in rows} == {"bm25", "tfidf"}
    for r in rows:
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.recall <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert r.avg_ms >= 0.0


def test_evaluate_default_methods(engine):
    rows = evaluate(engine, ["election president"], top_k=5)
    assert [r.method for r in rows] == ["bm25", "tfidf", "prf", "wordnet"]
