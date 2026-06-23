"""Tests for the FastAPI layer (in-process, via TestClient).

We point INDEX_PATH at a non-existent file *before* importing the app so the
engine is built from the committed sample (fast, deterministic) and BERT is
disabled (no bert.pkl next to that path).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make the top-level `api` package importable
os.environ["INDEX_PATH"] = str(ROOT / "tests" / "_no_such_index.pkl")

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs startup/shutdown (lifespan)
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] > 100
    assert body["bert_available"] is False
    assert "bm25" in body["methods"]


def test_categories(client):
    r = client.get("/api/categories")
    assert r.status_code == 200
    assert isinstance(r.json()["categories"], list)


def test_search_basic(client):
    r = client.get("/api/search", params={"q": "health", "method": "bm25", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "bm25"
    assert len(body["results"]) <= 5
    assert body["total_hits"] >= len(body["results"])


def test_search_default_method_falls_back_to_bm25_without_bert(client):
    # No method given + BERT unavailable in this test deployment -> BM25, not an error.
    r = client.get("/api/search", params={"q": "health"})
    assert r.status_code == 200
    assert r.json()["method"] == "bm25"


def test_search_unknown_method_is_400(client):
    r = client.get("/api/search", params={"q": "health", "method": "nope"})
    assert r.status_code == 400


def test_search_bert_unavailable_is_400(client):
    r = client.get("/api/search", params={"q": "health", "method": "bert"})
    assert r.status_code == 400


def test_search_rejects_empty_query(client):
    r = client.get("/api/search", params={"q": ""})
    assert r.status_code == 422  # violates min_length=1


def test_search_rejects_overlong_query(client):
    r = client.get("/api/search", params={"q": "x" * 1000})
    assert r.status_code == 422  # violates max_length=512
