"""FastAPI backend for the News Search Engine.

Endpoints
---------
GET  /api/health              -> liveness + index stats
GET  /api/categories          -> list of categories (for the UI filter)
GET  /api/search              -> run a search

The index is loaded once at startup from ``artifacts/index.pkl`` (build it with
``python scripts/build_index.py``). If that file is missing, the API builds an
index from ``data/sample_news.jsonl`` on the fly so the app still works.

In production the built React app (``frontend/dist``) is served as static files,
so the whole thing runs as a single service.
"""
from __future__ import annotations

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from news_search import InvertedIndex, SearchEngine, build_index, load_corpus  # noqa: E402
from news_search.engine import METHODS  # noqa: E402

INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "artifacts" / "index.pkl"))
SAMPLE_PATH = ROOT / "data" / "sample_news.jsonl"
BERT_PATH = INDEX_PATH.with_name("bert.pkl")

state: dict = {}


def _load_engine() -> SearchEngine:
    if INDEX_PATH.exists():
        print(f"Loading index from {INDEX_PATH}")
        index = InvertedIndex.load(INDEX_PATH)
    else:
        print(f"Index not found at {INDEX_PATH}; building from sample {SAMPLE_PATH}")
        index = build_index(load_corpus(SAMPLE_PATH), verbose=False)

    bert = None
    if BERT_PATH.exists():
        try:
            with BERT_PATH.open("rb") as fh:
                bert = pickle.load(fh)
            print("Loaded BERT expander.")
        except Exception as exc:  # pragma: no cover
            print(f"Could not load BERT expander: {exc}")
    return SearchEngine(index, bert=bert)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = _load_engine()
    state["engine"] = engine
    print(
        f">>> Ready: {engine.index.num_docs:,} documents | "
        f"{engine.index.vocabulary_size:,} terms | "
        f"BERT {'ENABLED' if engine.bert is not None else 'disabled'}"
    )
    yield
    state.clear()


app = FastAPI(title="News Search Engine API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    engine: SearchEngine = state["engine"]
    return {
        "status": "ok",
        "documents": engine.index.num_docs,
        "vocabulary": engine.index.vocabulary_size,
        "bert_available": engine.bert is not None,
        "methods": list(METHODS),
    }


@app.get("/api/categories")
def categories():
    engine: SearchEngine = state["engine"]
    return {"categories": engine.categories}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    method: str = Query("tfidf", description=f"One of {METHODS}"),
    top_k: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
):
    engine: SearchEngine = state["engine"]
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'. Choose from {list(METHODS)}.")
    if method in ("bert", "prf+bert") and engine.bert is None:
        raise HTTPException(
            400,
            "BERT expansion is not available on this deployment. "
            "Rebuild the index with `--bert` to enable it.",
        )
    return engine.search(q, method=method, top_k=top_k, category=category).to_dict()


# --- Serve the built frontend (if present) -------------------------------- #
_DIST = ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def _index():
        return FileResponse(_DIST / "index.html")
