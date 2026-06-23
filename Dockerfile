# Single-image deploy: FastAPI serves the API *and* the built React UI.
# Targets Hugging Face Spaces (Docker SDK) — listens on port 7860.
# The full dataset + prebuilt dense BERT model are fetched at build time, so the
# container starts instantly with no retraining.

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime ----
FROM python:3.11-slim
WORKDIR /app

# Python deps (core + BERT for the full semantic version)
COPY requirements.txt requirements-bert.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-bert.txt

# App code + committed sample + the built UI
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/sample_news.jsonl ./data/
COPY --from=frontend /app/frontend/dist ./frontend/dist

# NLTK corpora (better tokenisation + WordNet); safe to skip thanks to fallbacks
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"

# Fetch the full dataset + prebuilt dense model from Hugging Face, then build the
# matching lexical index. No GPU needed because the embeddings are prebuilt.
RUN python scripts/download_data.py \
 && python scripts/download_model.py \
 && python scripts/build_index.py --data data/News_Category_Dataset_v3.json \
 && rm -f data/News_Category_Dataset_v3.json

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir ."]
