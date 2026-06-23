"""Text preprocessing pipeline.

A single, reusable ``Preprocessor`` is used for *both* indexing and querying,
which guarantees that documents and queries are normalised the same way.

Pipeline: lowercase -> strip non-alphanumerics -> tokenize -> drop stopwords
-> Porter stemming.

The implementation **degrades gracefully**: it uses NLTK's tokenizer and
stopword list when their data files are available, and otherwise falls back to a
regex tokenizer and a bundled stopword list. Porter stemming is algorithmic and
needs no downloaded data. This keeps the engine runnable in restricted/offline
environments (CI, serverless) without a mandatory ``nltk.download`` step.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

from nltk.stem import PorterStemmer  # algorithmic, no data download required

_TOKEN_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

# Minimal English stopword list (used only if NLTK's corpus is unavailable).
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "couldn", "did", "didn", "do",
    "does", "doesn", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "isn", "it", "its", "itself", "just", "ll", "m", "ma", "me",
    "more", "most", "mustn", "my", "myself", "no", "nor", "not", "now", "o", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "same", "shan", "she", "should", "shouldn", "so",
    "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we", "were",
    "weren", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "wouldn", "y", "you", "your", "yours", "yourself",
    "yourselves",
}


def _load_stopwords() -> set:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


def _make_tokenizer():
    """Return a tokenizer callable, preferring NLTK punkt, falling back to regex."""
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("probe")  # force LookupError now if punkt is missing
        return word_tokenize
    except Exception:
        return str.split  # text is already punctuation-stripped, so split is fine


def ensure_nltk_data() -> None:
    """Best-effort download of optional NLTK corpora (punkt, stopwords, wordnet).

    Safe to call anywhere: failures are swallowed because the engine has
    fallbacks for everything except Porter stemming (which needs no data).
    """
    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class Preprocessor:
    """Deterministic text normaliser shared by the indexer and the query side."""

    def __init__(self, try_download: bool = False) -> None:
        if try_download:
            ensure_nltk_data()
        self._stemmer = PorterStemmer()
        self._stopwords = _load_stopwords()
        self._tokenize = _make_tokenizer()
        # Cache stemming — the same tokens recur constantly across 200k+ docs.
        self.stem = lru_cache(maxsize=200_000)(self._stemmer.stem)

    def tokenize(self, text: str) -> List[str]:
        """Run the full normalisation pipeline and return a list of tokens."""
        if not text:
            return []
        text = text.lower()
        text = _TOKEN_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            return []
        return [
            self.stem(tok)
            for tok in self._tokenize(text)
            if tok and tok not in self._stopwords
        ]
