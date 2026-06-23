"""Offline evaluation with Precision@K, Recall, F1 and latency.

Relevance judgments are *pseudo-qrels* derived from the dataset's own category
labels: a document is treated as relevant to a query if its category matches the
category mapped to one of the query terms. This is approximate (Precision@K is
the most meaningful metric), but it lets us compare methods automatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Set

from .engine import SearchEngine
from .preprocess import Preprocessor

# Query keyword -> relevant category (extend as needed).
QUERY_CATEGORY_MAP = {
    "covid": "WELLNESS",
    "health": "WELLNESS",
    "vaccine": "WELLNESS",
    "election": "POLITICS",
    "president": "POLITICS",
    "vote": "POLITICS",
    "movie": "ENTERTAINMENT",
    "film": "ENTERTAINMENT",
    "stock": "BUSINESS",
    "market": "BUSINESS",
    "money": "BUSINESS",
    "game": "SPORTS",
    "sport": "SPORTS",
    "travel": "TRAVEL",
    "food": "FOOD & DRINK",
    "recipe": "FOOD & DRINK",
}


@dataclass
class MethodScore:
    method: str
    precision: float
    recall: float
    f1: float
    avg_ms: float


def build_pseudo_qrels(queries: List[str], engine: SearchEngine) -> Dict[str, Set[int]]:
    pre = engine.pre
    stem_to_cat = {pre.stem(k): v for k, v in QUERY_CATEGORY_MAP.items()}

    qrels: Dict[str, Set[int]] = {}
    for q in queries:
        cats = {stem_to_cat[t] for t in pre.tokenize(q) if t in stem_to_cat}
        if not cats:
            continue
        qrels[q] = {
            doc_id for doc_id, meta in engine.index.meta.items()
            if meta.get("category") in cats
        }
    return qrels


def evaluate(
    engine: SearchEngine,
    queries: List[str],
    methods: List[str] | None = None,
    top_k: int = 10,
) -> List[MethodScore]:
    methods = methods or ["bm25", "tfidf", "prf", "wordnet"]
    qrels = build_pseudo_qrels(queries, engine)
    scored_queries = [q for q in queries if q in qrels]

    out: List[MethodScore] = []
    for method in methods:
        tp_p = tp_r = tp_f = tot_ms = 0.0
        for q in scored_queries:
            res = engine.search(q, method=method, top_k=top_k)
            retrieved = {r["id"] for r in res.results}
            relevant = qrels[q]
            tp = len(retrieved & relevant)
            precision = tp / len(retrieved) if retrieved else 0.0
            recall = tp / len(relevant) if relevant else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            tp_p += precision
            tp_r += recall
            tp_f += f1
            tot_ms += res.elapsed_ms
        n = max(len(scored_queries), 1)
        out.append(MethodScore(method, tp_p / n, tp_r / n, tp_f / n, tot_ms / n))
    return out
