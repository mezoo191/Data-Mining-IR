"""Download a prebuilt dense BERT index so nobody has to retrain.

Embedding the full ~210k-document dataset is slow (especially on CPU). Once
someone has built ``artifacts/dense.pkl`` and hosted it (e.g. a Hugging Face
repo or a GitHub Release asset), set the ``MODEL_URL`` environment variable to
its direct-download URL and this script fetches it into ``artifacts/``.

This is for the FULL-dataset model; the sample model is fast to build locally.
If no model is downloaded, the run scripts fall back to building it.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS = ROOT / "artifacts"
TARGET = ARTIFACTS / "dense.pkl"
# Prebuilt full-dataset dense model (Hugging Face). Override with the MODEL_URL
# env var to point at your own copy (a direct https link or a Google Drive link).
DEFAULT_MODEL_URL = "https://huggingface.co/datasets/1xMezoo/IR-BERT-model/resolve/main/dense.pkl"
MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done * 100 // total
        bar = "#" * (pct // 4)
        print(f"\r  [{bar:<25}] {pct:3d}%  ({done/1e6:6.1f} / {total/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  {done/1e6:6.1f} MB", end="", flush=True)


def _is_gdrive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _download_gdrive(url: str, dest: Path) -> bool:
    """Google Drive needs gdown to get past the large-file virus-scan page."""
    try:
        import gdown
    except ImportError:
        print("[error] This looks like a Google Drive link, which requires gdown:\n"
              "          pip install gdown\n"
              "        (or it is already in requirements.txt — run the setup again).")
        return False
    try:
        out = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        return bool(out) and Path(out).exists()
    except Exception as exc:
        print(f"\n[warn] Google Drive download failed: {exc}")
        return False


def _download_http(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "news-search/1.0"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, configurable URL)
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with dest.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    _progress(done, total)
        print()
        return True
    except Exception as exc:
        print(f"\n[warn] model download failed: {exc}")
        return False


def _valid(path: Path) -> bool:
    """Confirm the download is a usable DenseRetriever."""
    try:
        import pickle
        from news_search.dense import DenseRetriever  # noqa: F401 (needed to unpickle)
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        return getattr(obj, "embeddings", None) is not None and bool(getattr(obj, "doc_ids", None))
    except Exception as exc:
        print(f"[warn] downloaded model failed validation: {exc}")
        return False


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Dense model already present: {TARGET}")
        return 0
    if not MODEL_URL:
        print(
            "[info] MODEL_URL not set — no prebuilt model to download; it will be built locally.\n"
            "       To share one: build artifacts/dense.pkl once, upload it (Hugging Face or a\n"
            "       GitHub Release), then set MODEL_URL to its direct download link."
        )
        return 1

    tmp = TARGET.with_suffix(".pkl.part")
    tmp.unlink(missing_ok=True)
    print(f"Downloading prebuilt dense model:\n  {MODEL_URL}")
    ok = _download_gdrive(MODEL_URL, tmp) if _is_gdrive(MODEL_URL) else _download_http(MODEL_URL, tmp)
    if not ok or not _valid(tmp):
        tmp.unlink(missing_ok=True)
        return 1

    tmp.replace(TARGET)
    print(f"Done. Saved {TARGET} ({TARGET.stat().st_size/1e6:.1f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
