"""Download the HuffPost News Category Dataset (full ~210k articles, ~87 MB).

The full dataset is **not** committed to the repo. This script fetches it into
``data/News_Category_Dataset_v3.json``.

How it works
------------
1. **Public mirror (default, no account needed).** The dataset is pulled over
   plain HTTPS from a Hugging Face mirror, so it "just works" without any login
   or API token. Override the URL with the ``DATASET_URL`` environment variable.
2. **Kaggle API (fallback).** If the mirror is unreachable and you have Kaggle
   API credentials configured, it falls back to the Kaggle CLI.

Note on Kaggle: the Kaggle API authenticates with an **API token**
(``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment
variables) — being logged into the kaggle.com **website** in your browser does
*not* count, which is why the API can report that you need to log in.

A small ``data/sample_news.jsonl`` (~700 docs) is committed so the project runs
out of the box without any download.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = DATA_DIR / "News_Category_Dataset_v3.json"
KAGGLE_SLUG = "rmisra/news-category-dataset"

# Public, no-auth mirror of the exact same JSON-lines file (overridable).
MIRROR_URL = os.getenv(
    "DATASET_URL",
    "https://huggingface.co/datasets/heegyu/news-category-dataset/resolve/main/data.json",
)


def _looks_valid(path: Path) -> bool:
    """Cheap sanity check: first non-empty line is a news record."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                import json

                rec = json.loads(line)
                return "headline" in rec and "category" in rec
    except Exception:
        return False
    return False


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done * 100 // total
        bar = "#" * (pct // 4)
        print(f"\r  [{bar:<25}] {pct:3d}%  ({done/1e6:5.1f} / {total/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  {done/1e6:5.1f} MB", end="", flush=True)


def _download_from_mirror() -> bool:
    """Stream the dataset from the public mirror to a temp file, then rename."""
    tmp = TARGET.with_suffix(".json.part")
    print(f"Downloading dataset from public mirror:\n  {MIRROR_URL}")
    try:
        req = urllib.request.Request(MIRROR_URL, headers={"User-Agent": "news-search/1.0"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, configurable URL)
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
        print()
    except Exception as exc:
        print(f"\n[warn] Mirror download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    if not _looks_valid(tmp):
        print("[warn] Downloaded file did not look like the expected dataset.")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(TARGET)
    return True


def _download_from_kaggle() -> bool:
    try:
        import kaggle  # noqa: F401
    except Exception:
        print(
            "[info] Kaggle fallback unavailable (no `kaggle` package or API token).\n"
            "       The Kaggle API needs an API *token* at ~/.kaggle/kaggle.json or the\n"
            "       KAGGLE_USERNAME / KAGGLE_KEY env vars — a website browser login is not enough."
        )
        return False

    import subprocess

    print(f"Downloading {KAGGLE_SLUG} via Kaggle API...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "-p", str(DATA_DIR), "--unzip"],
            check=True,
        )
    except Exception as exc:
        print(f"[warn] Kaggle download failed: {exc}")
        return False
    return TARGET.exists()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Dataset already present: {TARGET}")
        return 0

    if _download_from_mirror() or _download_from_kaggle():
        size_mb = TARGET.stat().st_size / 1e6
        print(f"Done. Saved {TARGET} ({size_mb:.1f} MB).")
        return 0

    print(
        "\n[error] Could not download the dataset automatically.\n"
        f"        Download it manually from https://www.kaggle.com/datasets/{KAGGLE_SLUG}\n"
        f"        and place News_Category_Dataset_v3.json in {DATA_DIR}/."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
