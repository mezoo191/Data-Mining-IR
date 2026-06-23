"""High-level search engine: ties together preprocessing, ranking and expansion."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .index import InvertedIndex
from .preprocess import Preprocessor
from .dense import DenseRetriever
from . import ranking, expansion

# Retrieval methods exposed to the API/UI.
#   bm25 / tfidf : pure lexical ranking
#   prf / wordnet: BM25 ranking with query expansion (PRF supports user feedback)
#   bert         : true dense semantic retrieval (sentence-transformer embeddings)
#   hybrid       : BM25 + dense fusion (reciprocal rank fusion)
METHODS = ("bm25", "tfidf", "prf", "wordnet", "bert", "hybrid")
_SEMANTIC = ("bert", "hybrid")


@dataclass
class SearchResult:
    query: str
    method: str
    elapsed_ms: float
    expansion_terms: List[str]
    total_hits: int
    results: List[dict]  # each: metadata + "score" + "rank"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "method": self.method,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "expansion_terms": self.expansion_terms,
            "total_hits": self.total_hits,
            "results": self.results,
        }


class SearchEngine:
    """Search over a prebuilt :class:`InvertedIndex`."""

    def __init__(self, index: InvertedIndex, dense: Optional[DenseRetriever] = None):
        self.index = index
        self.pre = Preprocessor()
        self.dense = dense

    # ------------------------------------------------------------------ #
    @classmethod
    def from_path(cls, index_path: str | Path, dense: Optional[DenseRetriever] = None):
        return cls(InvertedIndex.load(index_path), dense=dense)

    @property
    def categories(self) -> List[str]:
        cats = {m["category"] for m in self.index.meta.values()}
        return sorted(cats)

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        method: str = "bm25",
        top_k: int = 10,
        category: Optional[str] = None,
        relevant_ids: Optional[List[int]] = None,
    ) -> SearchResult:
        """Run a search.

        ``relevant_ids`` (only used by the ``prf`` method) turns pseudo-relevance
        feedback into *true* relevance feedback: expansion terms are drawn from
        the documents the user marked relevant rather than from the top hits.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")

        start = time.perf_counter()

        # --- optional category filter --------------------------------- #
        # Restrict the candidate set *before* ranking/truncation so a filtered
        # search still returns the true top_k within that category (rather than
        # only whichever category docs happened to land in the global top_k).
        restrict: Optional[set] = None
        if category:
            restrict = self.index.category_docs(category)
            if not restrict:
                return self._empty(query, method, start)

        # --- semantic retrieval (true dense BERT, and hybrid fusion) -- #
        # These embed the raw query, so they work even when every query token is
        # a stopword. total_hits reflects the returned page (the whole corpus is
        # ranked by similarity, so there is no lexical "match count").
        if method in _SEMANTIC and self.dense is not None:
            if method == "bert":
                ranked = self.dense.search(query, top_k=top_k, restrict_to=restrict)
            else:  # hybrid
                ranked = self._hybrid(query, self.pre.tokenize(query), top_k, restrict)
            return self._result(query, method, start, [], ranked, total=len(ranked))

        # --- lexical retrieval (bm25 / tfidf / prf / wordnet) --------- #
        # (also the graceful fallback for semantic methods when no embeddings)
        query_terms = self.pre.tokenize(query)
        if not query_terms:
            return self._empty(query, method, start)

        expansion_terms: List[str] = []
        if method == "prf":
            expansion_terms = expansion.prf_terms(query_terms, self.index, relevant_ids=relevant_ids)
        elif method == "wordnet":
            expansion_terms = expansion.wordnet_terms(query_terms, self.pre)

        # de-dup while keeping query terms first
        seen = set(query_terms)
        expansion_terms = [t for t in expansion_terms if not (t in seen or seen.add(t))]
        all_terms = query_terms + expansion_terms

        # top_k=None ranks every match so total_hits is an honest count.
        ranker = ranking.score if method == "tfidf" else ranking.bm25
        ranked = ranker(all_terms, self.index, top_k=None, mode="or", restrict_to=restrict)
        return self._result(query, method, start, expansion_terms, ranked[:top_k],
                            total=len(ranked))

    # ------------------------------------------------------------------ #
    def _hybrid(self, query, query_terms, top_k, restrict, depth: int = 200, rrf_k: int = 60):
        """Fuse BM25 and dense rankings with Reciprocal Rank Fusion."""
        bm = ranking.bm25(query_terms, self.index, top_k=depth, mode="or",
                          restrict_to=restrict) if query_terms else []
        de = self.dense.search(query, top_k=depth, restrict_to=restrict)
        fused: dict = {}
        for rank, (doc_id, _) in enumerate(bm):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(de):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def _empty(self, query: str, method: str, start: float) -> SearchResult:
        return SearchResult(query, method, (time.perf_counter() - start) * 1000, [], 0, [])

    def _result(self, query, method, start, expansion_terms, ranked, total) -> SearchResult:
        results = []
        for rank, (doc_id, sc) in enumerate(ranked, start=1):
            row = dict(self.index.meta[doc_id])
            row["score"] = round(sc, 5)
            row["rank"] = rank
            results.append(row)
        return SearchResult(
            query=query,
            method=method,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            expansion_terms=expansion_terms,
            total_hits=total,
            results=results,
        )
