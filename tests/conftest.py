"""Shared pytest fixtures."""
from pathlib import Path

import pytest

from news_search import SearchEngine, build_index, load_corpus

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_news.jsonl"


@pytest.fixture(scope="session")
def engine() -> SearchEngine:
    """A SearchEngine over the committed sample corpus (built once per session)."""
    return SearchEngine(build_index(load_corpus(SAMPLE), verbose=False))
