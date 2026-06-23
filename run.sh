#!/usr/bin/env bash
# ============================================================
#  News Search Engine - one-shot setup + launch (macOS/Linux)
#
#  Usage:
#    ./run.sh            BEST MODE (default): sample dataset WITH BERT
#    ./run.sh lite       Sample dataset, no BERT (fastest)
#    ./run.sh full       Full ~210k dataset WITH BERT (best at scale, slow build)
#    ./run.sh full lite  Full dataset, no BERT
#
#  Best mode (BERT) is always the default. Pass "lite" (or "nobert") to opt out.
#  The browser opens automatically once the server is ready.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# Best mode = BERT enabled. It stays ON by default and is only turned off when
# the user explicitly opts out with "lite" / "nobert".
USE_BERT=1
USE_FULL=0
for arg in "$@"; do
  [ "$arg" = "lite" ] && USE_BERT=0
  [ "$arg" = "nobert" ] && USE_BERT=0
  [ "$arg" = "bert" ] && USE_BERT=1
  [ "$arg" = "full" ] && USE_FULL=1
done

# --- 0. Prerequisites ---------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "[error] Python 3 not found."; exit 1; }
if ! command -v npm >/dev/null 2>&1; then
  echo "[error] Node.js / npm not found - the web UI cannot be built."
  echo "        Install Node.js LTS from https://nodejs.org/ then run again."
  exit 1
fi

# --- 1. Stop any previous server holding port 8000 ----------
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti tcp:8000 2>/dev/null || true)
  if [ -n "$PIDS" ]; then echo "[setup] Stopping previous server on port 8000..."; kill -9 $PIDS 2>/dev/null || true; fi
elif command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp >/dev/null 2>&1 || true
fi

# --- 2. Python virtual environment --------------------------
if [ ! -d .venv ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 3. Python dependencies ---------------------------------
echo "[setup] Installing Python dependencies..."
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ "$USE_BERT" = "1" ]; then
  echo "[setup] Installing BERT dependencies (large download, one time)..."
  pip install -q -r requirements-bert.txt
fi

# --- 4. Choose dataset --------------------------------------
DATA_ARG=""
if [ "$USE_FULL" = "1" ]; then
  DATASET=""
  [ -f data/News_Category_Dataset_v3.json ] && DATASET="data/News_Category_Dataset_v3.json"
  [ -f News_Category_Dataset_v3.json ] && DATASET="News_Category_Dataset_v3.json"
  if [ -z "$DATASET" ]; then
    echo "[setup] Full dataset not found - downloading automatically..."
    python scripts/download_data.py
    [ -f data/News_Category_Dataset_v3.json ] && DATASET="data/News_Category_Dataset_v3.json"
  fi
  if [ -z "$DATASET" ]; then
    echo "[error] Full dataset still missing after download."
    exit 1
  fi
  echo "[setup] Using full dataset: $DATASET"
  DATA_ARG="--data $DATASET"
fi

# --- 5. Rebuild the index when the mode/dataset changes -----
SIG="sample"; [ "$USE_FULL" = "1" ] && SIG="full"
[ "$USE_BERT" = "1" ] && SIG="${SIG}+dense"
PREVSIG=""
[ -f artifacts/build.info ] && PREVSIG="$(cat artifacts/build.info)"
if [ "$SIG" != "$PREVSIG" ]; then
  [ -n "$PREVSIG" ] && echo "[setup] Build config changed ($PREVSIG -> $SIG); rebuilding..."
  rm -f artifacts/index.pkl artifacts/dense.pkl artifacts/bert.pkl
fi

# --- 6. Build the search index (only if missing) ------------
if [ "$USE_BERT" = "1" ]; then
  # Try a prebuilt dense model first (full dataset only); falls back to building.
  if [ ! -f artifacts/dense.pkl ] && [ "$USE_FULL" = "1" ]; then
    echo "[setup] Looking for a prebuilt dense model (MODEL_URL)..."
    python scripts/download_model.py || true
  fi
  if [ ! -f artifacts/dense.pkl ]; then
    echo "[setup] Building index + dense BERT embeddings..."
    python scripts/build_index.py $DATA_ARG --bert
  elif [ ! -f artifacts/index.pkl ]; then
    echo "[setup] Building search index to match the prebuilt model..."
    python scripts/build_index.py $DATA_ARG
  fi
  [ -f artifacts/dense.pkl ] || { echo "[error] Dense BERT embeddings were not created."; exit 1; }
else
  if [ ! -f artifacts/index.pkl ]; then
    echo "[setup] Building search index..."
    python scripts/build_index.py $DATA_ARG
  fi
fi
mkdir -p artifacts && echo "$SIG" > artifacts/build.info

# --- 7. Build the frontend (install once, always rebuild) ---
(
  cd frontend
  [ -d node_modules ] || { echo "[setup] Installing frontend packages..."; npm install; }
  echo "[setup] Building frontend..."
  npm run build
)
[ -f frontend/dist/index.html ] || { echo "[error] Frontend build failed."; exit 1; }

# --- 8. Open the browser once the server is up --------------
(
  for _ in $(seq 1 90); do
    if curl -fs http://localhost:8000/api/health >/dev/null 2>&1; then
      (command -v open >/dev/null && open http://localhost:8000) || \
      (command -v xdg-open >/dev/null && xdg-open http://localhost:8000) || true
      break
    fi
    sleep 0.7
  done
) &

# --- 9. Launch ----------------------------------------------
echo
echo "[run] Starting server at http://localhost:8000  (mode: $SIG)  (Ctrl+C to stop)"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir .
