"""Query-expansion strategies.

Both techniques return *stemmed* expansion tokens to append to the original
query before re-ranking:

1. Relevance Feedback (Rocchio-style) — pull the most discriminative (TF-IDF)
   terms from a set of "relevant" documents. By default the set is the top
   retrieved docs (*pseudo*-relevance feedback); pass ``relevant_ids`` to use
   documents the user explicitly marked relevant (*true* relevance feedback).
2. WordNet synonyms — lexical expansion from the WordNet thesaurus.

(Semantic / BERT retrieval lives in :mod:`news_search.dense`.)
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .index import InvertedIndex
from .preprocess import Preprocessor
from . import ranking


# --------------------------------------------------------------------------- #
# 1. Relevance Feedback (pseudo by default, true when relevant_ids given)
# --------------------------------------------------------------------------- #
def prf_terms(
    query_terms: List[str],
    index: InvertedIndex,
    top_k_feedback: int = 10,
    top_terms: int = 5,
    relevant_ids: Optional[List[int]] = None,
) -> List[str]:
    """Extract expansion terms from a set of relevant documents (Rocchio).

    If ``relevant_ids`` is provided, those user-judged documents are the feedback
    set (true relevance feedback). Otherwise the top ``top_k_feedback`` BM25 hits
    are assumed relevant (pseudo-relevance feedback).
    """
    if relevant_ids:
        feedback_docs = [d for d in relevant_ids if d in index.forward]
    else:
        feedback_docs = [d for d, _ in ranking.bm25(query_terms, index,
                                                     top_k=top_k_feedback, mode="or")]
    if not feedback_docs:
        return []

    qset = set(query_terms)
    term_scores: Counter = Counter()
    for doc_id in feedback_docs:
        forward = index.forward.get(doc_id, {})
        dl = index.doc_len.get(doc_id, 0) or 1
        for term, freq in forward.items():
            if term in qset:
                continue
            term_scores[term] += (freq / dl) * index.idf.get(term, 0.0)

    return [t for t, _ in term_scores.most_common(top_terms)]


# --------------------------------------------------------------------------- #
# 2. WordNet synonyms
# --------------------------------------------------------------------------- #
def wordnet_terms(
    query_terms: List[str],
    preprocessor: Preprocessor,
    max_per_term: int = 3,
) -> List[str]:
    """Expand each (stemmed) query token with WordNet synonym stems."""
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")  # trigger LookupError early if missing
    except LookupError:
        return []

    qset = set(query_terms)
    added: List[str] = []
    seen: set = set()
    for token in query_terms:
        per_token = 0
        for syn in wordnet.synsets(token):
            if per_token >= max_per_term:
                break
            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ").lower()
                stemmed = preprocessor.stem(word)
                if stemmed in qset or stemmed in seen:
                    continue
                seen.add(stemmed)
                added.append(stemmed)
                per_token += 1
                if per_token >= max_per_term:  # cap synonyms *per query term*
                    break
    return added
