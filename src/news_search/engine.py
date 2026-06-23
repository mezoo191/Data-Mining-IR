"""High-level search engine: ties together preprocessing, ranking and expansion."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .index import InvertedIndex
from .preprocess import Preprocessor
from . import ranking, expansion

# Retrieval methods exposed to the API/UI.
# "bm25" and "tfidf" are pure ranking; the rest add query expansion on top of
# BM25 ranking.
METHODS = ("bm25", "tfidf", "prf", "wordnet", "bert", "prf+bert")


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

    def __init__(self, index: InvertedIndex, bert: Optional[expansion.BertExpander] = None):
        self.index = index
        self.pre = Preprocessor()
        self.bert = bert

    # ------------------------------------------------------------------ #
    @classmethod
    def from_path(cls, index_path: str | Path, bert: Optional[expansion.BertExpander] = None):
        return cls(InvertedIndex.load(index_path), bert=bert)

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
    ) -> SearchResult:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")

        start = time.perf_counter()
        query_terms = self.pre.tokenize(query)
        expansion_terms: List[str] = []

        if not query_terms:
            return SearchResult(query, method, 0.0, [], 0, [])

        # --- gather expansion terms by method ------------------------- #
        if method == "prf":
            expansion_terms = expansion.prf_terms(query_terms, self.index)
        elif method == "wordnet":
            expansion_terms = expansion.wordnet_terms(query_terms, self.pre)
        elif method == "bert":
            expansion_terms = [t for t, _ in self._bert_expand(query, query_terms)]
        elif method == "prf+bert":
            expansion_terms = expansion.prf_terms(query_terms, self.index)
            expansion_terms += [t for t, _ in self._bert_expand(query, query_terms)]

        # de-dup while keeping query terms first
        seen = set(query_terms)
        expansion_terms = [t for t in expansion_terms if not (t in seen or seen.add(t))]
        all_terms = query_terms + expansion_terms

        # --- retrieve & rank ------------------------------------------ #
        # BM25 is the default ranker; "tfidf" keeps the classic scorer for
        # comparison. Expansion terms use OR retrieval so new docs can surface.
        if method == "tfidf":
            ranked = ranking.score(all_terms, self.index, top_k=top_k, mode="or")
        else:
            ranked = ranking.bm25(all_terms, self.index, top_k=top_k, mode="or")

        # --- optional category filter --------------------------------- #
        results: List[dict] = []
        rank = 0
        for doc_id, sc in ranked:
            meta = self.index.meta[doc_id]
            if category and meta.get("category") != category:
                continue
            rank += 1
            row = dict(meta)
            row["score"] = round(sc, 5)
            row["rank"] = rank
            results.append(row)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            query=query,
            method=method,
            elapsed_ms=elapsed_ms,
            expansion_terms=expansion_terms,
            total_hits=len(results),
            results=results,
        )

    # ------------------------------------------------------------------ #
    def _bert_expand(self, query: str, query_terms: List[str], top_n: int = 5):
        if self.bert is None:
            return []
        return self.bert.expand(query, query_terms, top_n=top_n)
