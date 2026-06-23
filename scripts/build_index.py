"""Build (and persist) the inverted index from a dataset file.

Usage
-----
    # Build from the committed sample (default) -> artifacts/index.pkl
    python scripts/build_index.py

    # Build from the full dataset
    python scripts/build_index.py --data data/News_Category_Dataset_v3.json

    # Also build dense BERT document embeddings (heavy, optional)
    python scripts/build_index.py --data data/News_Category_Dataset_v3.json --bert
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from news_search import build_index, load_corpus  # noqa: E402
from news_search.dense import DenseRetriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the search index.")
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_news.jsonl"),
                    help="Path to the dataset (JSON lines).")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "index.pkl"),
                    help="Where to write the pickled index.")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of documents.")
    ap.add_argument("--bert", action="store_true",
                    help="Also build dense BERT document embeddings (saved as dense.pkl).")
    args = ap.parse_args()

    print(f"Loading corpus from {args.data} ...")
    docs = load_corpus(args.data, limit=args.limit)
    print(f"Loaded {len(docs):,} documents.")

    index = build_index(docs)
    index.save(args.out)
    print(f"Saved index -> {args.out}")

    if args.bert:
        dense = DenseRetriever().fit(docs)
        dense_path = Path(args.out).with_name("dense.pkl")
        with dense_path.open("wb") as fh:
            pickle.dump(dense, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved dense retriever -> {dense_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
