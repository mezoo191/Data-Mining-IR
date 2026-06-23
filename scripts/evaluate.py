"""Run offline evaluation (Precision@K / Recall / F1 / latency) across methods.

Uses the prebuilt index at ``artifacts/index.pkl`` if present, otherwise builds
one from the committed sample. Relevance is judged with pseudo-qrels derived from
the dataset's category labels (see ``news_search.evaluate``).

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --index artifacts/index.pkl --top-k 10
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from news_search import SearchEngine, build_index, load_corpus  # noqa: E402
from news_search.index import InvertedIndex  # noqa: E402
from news_search.evaluate import evaluate  # noqa: E402

ALL_METHODS = ["bm25", "tfidf", "prf", "wordnet", "bert", "hybrid"]
BASE_METHODS = ["bm25", "tfidf", "prf", "wordnet"]


def _load_dense(index_path: Path):
    """Load the dense (BERT) retriever sitting next to the index, if present."""
    dense_path = Path(index_path).with_name("dense.pkl")
    if not dense_path.exists():
        return None
    try:
        with dense_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not load {dense_path}: {exc}")
        return None

DEFAULT_QUERIES = [
    "covid vaccine health",
    "election president vote",
    "movie film",
    "stock market money",
    "game sport",
    "travel food recipe",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline IR evaluation across methods.")
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "index.pkl"),
                    help="Prebuilt index to evaluate (falls back to the sample).")
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_news.jsonl"),
                    help="Dataset used when no prebuilt index exists.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of methods to evaluate (default: bm25 tfidf prf wordnet).")
    args = ap.parse_args()

    if Path(args.index).exists():
        print(f"Loading index from {args.index}")
        engine = SearchEngine(InvertedIndex.load(args.index), dense=_load_dense(Path(args.index)))
    else:
        print(f"No index at {args.index}; building from {args.data}")
        engine = SearchEngine(build_index(load_corpus(args.data), verbose=False))

    # Evaluate BERT methods too when embeddings are available.
    methods = args.methods or (ALL_METHODS if engine.dense is not None else BASE_METHODS)
    print(f"Index: {engine.index.num_docs:,} docs | "
          f"BERT {'enabled' if engine.dense is not None else 'disabled'} | methods: {methods}")

    # Warm up the BERT model so its one-time load isn't charged to the first
    # timed query (gives a fair steady-state latency).
    if engine.dense is not None:
        engine.search("warmup query", method="bert", top_k=1)

    rows = evaluate(engine, DEFAULT_QUERIES, methods=methods, top_k=args.top_k)

    print(f"\n{'method':<12}{'P@K':>8}{'Recall':>9}{'F1':>8}{'avg ms':>9}")
    print("-" * 46)
    for r in rows:
        print(f"{r.method:<12}{r.precision:>8.3f}{r.recall:>9.3f}{r.f1:>8.3f}{r.avg_ms:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
