"""Dense (bi-encoder) semantic retrieval.

This is *true* BERT retrieval: every document is embedded once at build time
with a sentence-transformer, and queries are ranked by cosine similarity to the
query embedding — semantic matching rather than lexical term overlap. (Contrast
with the older approach, which only embedded vocabulary *terms* to expand the
query before BM25 ranking.)

Embeddings are L2-normalised, so cosine similarity is a single matrix-vector
product. For ~200k docs a brute-force NumPy scan is only tens of milliseconds;
FAISS/HNSW would be the next step for much larger corpora.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .corpus import Document


class DenseRetriever:
    """Embeds documents and ranks them by cosine similarity to the query."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.doc_ids: List[int] = []
        self.embeddings = None  # np.ndarray (num_docs, dim), float32, L2-normalised

    # Persist only vocab + numpy embeddings, never the live torch model
    # (pickling it is fragile/version-sensitive); recreate it lazily on use.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

    @staticmethod
    def _best_device() -> str:
        """Use a CUDA GPU when available, else CPU. (Requires a CUDA build of
        PyTorch; a CPU-only install always reports 'cpu'.)"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._best_device())
        return self._model

    def fit(self, docs: List[Document], verbose: bool = True, batch_size: int = 256) -> "DenseRetriever":
        """Embed every document's text once."""
        import numpy as np

        model = self._ensure_model()
        if verbose:
            print(f"Encoding on device: {self._best_device()}")
        self.doc_ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        if verbose:
            print(f"Embedding {len(texts):,} documents with {self.model_name}...")
        emb = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=verbose
        )
        self.embeddings = np.asarray(emb, dtype="float32")
        return self

    def search(
        self,
        query: str,
        top_k: Optional[int] = 10,
        restrict_to: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return ``(doc_id, cosine_score)`` ranked by similarity to ``query``."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []
        import numpy as np

        model = self._ensure_model()
        q = model.encode([query], normalize_embeddings=True)[0].astype("float32")
        sims = self.embeddings @ q  # cosine similarity (vectors are normalised)
        order = np.argsort(-sims)

        out: List[Tuple[int, float]] = []
        for idx in order:
            doc_id = self.doc_ids[idx]
            if restrict_to is not None and doc_id not in restrict_to:
                continue
            out.append((doc_id, float(sims[idx])))
            if top_k is not None and len(out) >= top_k:
                break
        return out
