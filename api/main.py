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
    # NOTE: index.pkl / bert.pkl are loaded with ``pickle``, which executes
    # arbitrary code on load. These artifacts are produced locally by
    # ``scripts/build_index.py`` and are trusted; never point INDEX_PATH at a
    # file from an untrusted source.
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

# CORS is only needed for the Vite dev server (npm run dev on :5173); the built
# app is served same-origin from :8000. Scope to localhost dev origins by default;
# override with a comma-separated CORS_ORIGINS env var if deploying elsewhere.
_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:8000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _engine() -> SearchEngine:
    """Return the loaded engine or a clean 503 if startup hasn't finished."""
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(503, "Search engine is not ready yet.")
    return engine


@app.get("/api/health")
def health():
    engine = _engine()
    return {
        "status": "ok",
        "documents": engine.index.num_docs,
        "vocabulary": engine.index.vocabulary_size,
        "bert_available": engine.bert is not None,
        "methods": list(METHODS),
    }


@app.get("/api/categories")
def categories():
    engine = _engine()
    return {"categories": engine.categories}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=512, description="Search query"),
    method: Optional[str] = Query(None, description=f"One of {METHODS}; defaults to BERT"),
    top_k: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, max_length=100),
):
    engine = _engine()
    # Default to BERT (the best method) when it's available, else BM25 so a bare
    # request never fails on a lite (no-BERT) deployment.
    if method is None:
        method = "bert" if engine.bert is not None else "bm25"
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
