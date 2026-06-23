"""Corpus loading for the HuffPost News Category Dataset.

The raw dataset is JSON-lines: one JSON object per line with fields
``headline``, ``short_description``, ``category``, ``date``, ``link``,
``authors``. We build a lightweight ``Document`` for each record. The text we
actually index is ``headline + short_description``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Document:
    id: int
    text: str  # headline + short_description, used for indexing
    headline: str
    short_description: str
    category: str
    date: str
    link: str
    authors: str = ""

    def to_meta(self) -> Dict[str, str]:
        """Serializable metadata returned to the API/UI (no heavy fields)."""
        return {
            "id": self.id,
            "headline": self.headline,
            "short_description": self.short_description,
            "category": self.category,
            "date": self.date,
            "link": self.link,
            "authors": self.authors,
        }


def load_corpus(path: str | Path, limit: Optional[int] = None) -> List[Document]:
    """Load JSON-lines news records into a list of :class:`Document`.

    Args:
        path: path to the ``.json``/``.jsonl`` dataset file.
        limit: optionally cap the number of documents (useful for demos/tests).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run `python scripts/download_data.py` or pass --data with a valid path."
        )

    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            headline = rec.get("headline", "")
            desc = rec.get("short_description", "")
            docs.append(
                Document(
                    id=len(docs),
                    text=f"{headline} {desc}".strip(),
                    headline=headline,
                    short_description=desc,
                    category=rec.get("category", "UNKNOWN"),
                    date=rec.get("date", ""),
                    link=rec.get("link", ""),
                    authors=rec.get("authors", ""),
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs
