"""Download the HuffPost News Category Dataset.

The full dataset (~210k articles, ~87 MB) is **not** committed to the repo.
This script fetches it into ``data/``.

Preferred: Kaggle API
---------------------
1. ``pip install kaggle`` and place your Kaggle API token at
   ``~/.kaggle/kaggle.json`` (Account -> Create New API Token).
2. ``python scripts/download_data.py``

Manual fallback
---------------
Download from
https://www.kaggle.com/datasets/rmisra/news-category-dataset
and place ``News_Category_Dataset_v3.json`` in the ``data/`` directory.

A small ``data/sample_news.jsonl`` (~700 docs) is committed so the project runs
out of the box without any download.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = DATA_DIR / "News_Category_Dataset_v3.json"
KAGGLE_SLUG = "rmisra/news-category-dataset"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Dataset already present: {TARGET}")
        return 0

    try:
        import kaggle  # noqa: F401
    except Exception:
        print(
            "Kaggle package/credentials not available.\n"
            "Install with `pip install kaggle`, add ~/.kaggle/kaggle.json, then re-run.\n"
            f"Or download manually from https://www.kaggle.com/datasets/{KAGGLE_SLUG}\n"
            f"and place the .json in {DATA_DIR}/."
        )
        return 1

    print(f"Downloading {KAGGLE_SLUG} via Kaggle API...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    print(f"Done. Files in {DATA_DIR}:")
    for p in DATA_DIR.glob("*.json"):
        print(" -", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
