"""Inverted index with precomputed statistics.

The original notebook re-tokenised and re-stemmed every candidate document at
*query time*, and recomputed IDF per (term, doc). On 200k+ documents that is
extremely slow. This module fixes that by computing everything **once** at index
build time:

* ``postings[term]``  -> list of ``(doc_id, term_freq)``
* ``doc_len[doc_id]`` -> number of tokens in the document
* ``df[term]``        -> document frequency
* ``idf[term]``       -> ``log(N / df)`` (computed once)
* ``forward[doc_id]`` -> ``{term: freq}`` (used by pseudo-relevance feedback)

At query time, ranking is a cheap lookup over postings — no re-tokenisation.
"""
from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .corpus import Document
from .preprocess import Preprocessor


@dataclass
class InvertedIndex:
    postings: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    doc_len: Dict[int, int] = field(default_factory=dict)
    idf: Dict[str, float] = field(default_factory=dict)
    forward: Dict[int, Dict[str, int]] = field(default_factory=dict)
    meta: Dict[int, dict] = field(default_factory=dict)  # doc_id -> display metadata
    num_docs: int = 0
    avg_doc_len: float = 0.0  # mean document length (for BM25 length normalisation)

    # ------------------------------------------------------------------ #
    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def tfidf(self, term: str, doc_id: int, term_freq: int) -> float:
        """Normalised TF * IDF for a single (term, doc)."""
        dl = self.doc_len.get(doc_id, 0)
        if dl == 0:
            return 0.0
        return (term_freq / dl) * self.idf.get(term, 0.0)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "InvertedIndex":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def build_index(
    docs: List[Document],
    preprocessor: Preprocessor | None = None,
    verbose: bool = True,
) -> InvertedIndex:
    """Build an :class:`InvertedIndex` from a list of documents."""
    pre = preprocessor or Preprocessor()
    index = InvertedIndex(num_docs=len(docs))

    start = time.time()
    # term -> {doc_id: freq}
    raw: Dict[str, Dict[int, int]] = {}

    for doc in docs:
        tokens = pre.tokenize(doc.text)
        index.doc_len[doc.id] = len(tokens)
        index.meta[doc.id] = doc.to_meta()

        term_freqs: Dict[str, int] = {}
        for tok in tokens:
            term_freqs[tok] = term_freqs.get(tok, 0) + 1
        index.forward[doc.id] = term_freqs

        for term, freq in term_freqs.items():
            raw.setdefault(term, {})[doc.id] = freq

        if verbose and doc.id and doc.id % 25_000 == 0:
            print(f"  indexed {doc.id:,} / {len(docs):,} docs...")

    # Freeze postings (sorted by doc_id) and precompute IDF.
    n = max(index.num_docs, 1)
    for term, doc_freqs in raw.items():
        index.postings[term] = sorted(doc_freqs.items())
        index.idf[term] = math.log(n / len(doc_freqs))

    total_len = sum(index.doc_len.values())
    index.avg_doc_len = total_len / n

    if verbose:
        print(
            f"Built index: {index.num_docs:,} docs, "
            f"{index.vocabulary_size:,} terms in {time.time() - start:.1f}s"
        )
    return index
