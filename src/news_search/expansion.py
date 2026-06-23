"""Query-expansion strategies.

Three independent techniques, all returning *stemmed* expansion tokens that can
be appended to the original query before re-ranking:

1. Pseudo-Relevance Feedback (PRF) — Rocchio-style: assume the top results are
   relevant and pull the most discriminative (TF-IDF) terms from them.
2. WordNet synonyms — lexical expansion from the WordNet thesaurus.
3. BERT embeddings — semantic expansion using sentence-transformer similarity
   over the index vocabulary. Loaded lazily (heavy dependency).
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .index import InvertedIndex
from .preprocess import Preprocessor
from . import ranking


# --------------------------------------------------------------------------- #
# 1. Pseudo-Relevance Feedback
# --------------------------------------------------------------------------- #
def prf_terms(
    query_terms: List[str],
    index: InvertedIndex,
    top_k_feedback: int = 10,
    top_terms: int = 5,
) -> List[str]:
    """Extract expansion terms from the top retrieved documents (PRF)."""
    initial = ranking.bm25(query_terms, index, top_k=top_k_feedback, mode="or")
    if not initial:
        return []

    qset = set(query_terms)
    term_scores: Counter = Counter()
    for doc_id, _ in initial:
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
    for token in query_terms:
        for syn in wordnet.synsets(token)[:3]:
            for lemma in syn.lemmas()[:2]:
                word = lemma.name().replace("_", " ").lower()
                stemmed = preprocessor.stem(word)
                if stemmed not in qset and stemmed not in added:
                    added.append(stemmed)
                if len([a for a in added]) >= max_per_term * len(query_terms):
                    break
    return added


# --------------------------------------------------------------------------- #
# 3. BERT embedding expansion (lazy, optional)
# --------------------------------------------------------------------------- #
class BertExpander:
    """Semantic query expansion via sentence-transformer embeddings.

    Embeds the most frequent vocabulary terms once, then finds terms whose
    embeddings are most similar to the query embedding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vocab_size: int = 15000):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self._model = None
        self._vocab: List[str] = []
        self._embeddings = None

    # Persist only the lightweight data (vocab + numpy embeddings), never the
    # live sentence-transformer model — pickling a torch model is fragile and
    # version-sensitive. The model is re-created lazily on first use.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, index: InvertedIndex, verbose: bool = True) -> "BertExpander":
        """Compute embeddings for the most frequent vocabulary terms."""
        model = self._ensure_model()
        terms = sorted(index.postings, key=lambda t: index.df(t), reverse=True)
        self._vocab = terms[: self.vocab_size]
        if verbose:
            print(f"Embedding {len(self._vocab):,} vocabulary terms with {self.model_name}...")
        self._embeddings = model.encode(
            self._vocab, batch_size=256, show_progress_bar=verbose, normalize_embeddings=True
        )
        return self

    def expand(self, query: str, query_terms: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        if self._embeddings is None:
            raise RuntimeError("BertExpander.fit() must be called before expand().")
        import numpy as np

        model = self._ensure_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ q_emb  # cosine sim (vectors are normalized)
        order = np.argsort(sims)[::-1]

        qset = set(query_terms)
        out: List[Tuple[str, float]] = []
        for idx in order:
            term = self._vocab[idx]
            if term not in qset and len(term) > 2:
                out.append((term, float(sims[idx])))
            if len(out) >= top_n:
                break
        return out
