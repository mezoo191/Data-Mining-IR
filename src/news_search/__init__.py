"""News Search Engine — a from-scratch Information Retrieval system.

Public API:
    from news_search import SearchEngine, build_index, load_corpus
"""
from .corpus import Document, load_corpus
from .index import InvertedIndex, build_index
from .engine import SearchEngine, SearchResult

__all__ = [
    "Document",
    "load_corpus",
    "InvertedIndex",
    "build_index",
    "SearchEngine",
    "SearchResult",
]

__version__ = "1.0.0"
