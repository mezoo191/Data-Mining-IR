"""Ranking over a precomputed :class:`InvertedIndex`.

Two scorers are provided:

* :func:`score` — classic TF-IDF (normalised term frequency times IDF).
* :func:`bm25`  — Okapi BM25, the modern default. BM25 adds term-frequency
  *saturation* (``k1``) and proper *length normalisation* (``b``) relative to the
  average document length. This removes the short-document bias of plain TF-IDF
  (where a one-word headline could outrank a full article) and substantially
  improves ranking quality.

Scoring is *term-at-a-time*: we walk the postings list of each query term and
accumulate the document scores. No document is ever re-tokenised at query time.

Two retrieval modes:
    * ``"or"``  (default) — a document matching *any* query term is a candidate.
                 This avoids the original "AND-only" behaviour that returned
                 zero results whenever a single term was missing.
    * ``"and"`` — a document must contain *all* query terms.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .index import InvertedIndex


def _candidate_docs(terms: Iterable[str], index: InvertedIndex, mode: str) -> Set[int]:
    posting_sets = [
        {doc_id for doc_id, _ in index.postings[t]}
        for t in terms
        if t in index.postings
    ]
    if not posting_sets:
        return set()
    if mode == "and":
        return set.intersection(*posting_sets)
    return set.union(*posting_sets)


def score(
    query_terms: List[str],
    index: InvertedIndex,
    top_k: Optional[int] = 10,
    mode: str = "or",
    restrict_to: Set[int] | None = None,
) -> List[Tuple[int, float]]:
    """Rank documents for ``query_terms`` by summed TF-IDF.

    Args:
        query_terms: already-preprocessed (stemmed) query tokens.
        index: the inverted index.
        top_k: number of results to return; ``None`` returns every scored doc.
        mode: ``"or"`` or ``"and"`` candidate selection.
        restrict_to: optional set of doc_ids to score within (used for re-ranking
            and category filtering).
    """
    if not query_terms:
        return []

    candidates = _candidate_docs(query_terms, index, mode)
    if restrict_to is not None:
        candidates &= restrict_to
    if not candidates:
        return []

    scores: Dict[int, float] = {}
    for term in query_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        idf = index.idf.get(term, 0.0)
        if idf == 0.0:
            continue
        for doc_id, tf in postings:
            if doc_id not in candidates:
                continue
            dl = index.doc_len.get(doc_id, 0)
            if dl:
                scores[doc_id] = scores.get(doc_id, 0.0) + (tf / dl) * idf

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def bm25(
    query_terms: List[str],
    index: InvertedIndex,
    top_k: Optional[int] = 10,
    mode: str = "or",
    restrict_to: Set[int] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    """Rank documents for ``query_terms`` with Okapi BM25.

    ``k1`` controls term-frequency saturation; ``b`` controls how strongly
    document length is normalised (0 = none, 1 = full). ``top_k=None`` returns
    every scored document (used to compute an honest total hit count).
    """
    if not query_terms:
        return []

    candidates = _candidate_docs(query_terms, index, mode)
    if restrict_to is not None:
        candidates &= restrict_to
    if not candidates:
        return []

    n = max(index.num_docs, 1)
    # getattr keeps BM25 working even on an index pickled before avg_doc_len existed.
    avgdl = getattr(index, "avg_doc_len", 0.0) or (sum(index.doc_len.values()) / n)

    scores: Dict[int, float] = {}
    for term in query_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)  # BM25 IDF (always >= 0)
        for doc_id, tf in postings:
            if doc_id not in candidates:
                continue
            dl = index.doc_len.get(doc_id, 0)
            if not dl:
                continue
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1.0)) / denom

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]
