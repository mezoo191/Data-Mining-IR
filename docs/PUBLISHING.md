# Publishing & deployment guide

Step-by-step to get this on GitHub and (optionally) live on the web. Run these
from the project root on your own machine (where `git`, `python` and `npm` work).

## 1. Add a screenshot (30 seconds)

The README shows `docs/screenshot.png`. To create it:

- Open `docs/ui-preview.html` in your browser, **or** run the app (see README),
- Take a screenshot and save it as `docs/screenshot.png`.

## 2. First commit

A fresh, clean history looks best on a portfolio repo. From the project root:

```bash
git add .
git commit -m "News search engine: TF-IDF retrieval, query expansion, FastAPI + React"
```

If `git status` shows leftover lock/config issues from the old repo, you can
start a brand-new history:

```bash
rm -rf .git
git init -b main
git add .
git commit -m "News search engine: TF-IDF retrieval, query expansion, FastAPI + React"
```

Confirm the large files are ignored — `git status` should **not** list
`data/News_Category_Dataset_v3.json` or any `.rar`/`.pkl`.

## 3. Create the GitHub repo and push

Using the GitHub CLI:

```bash
gh repo create news-search-engine --public --source=. --remote=origin --push
```

Or manually: create an empty repo on github.com, then:

```bash
git remote add origin https://github.com/<your-username>/news-search-engine.git
git push -u origin main
```

## 4. (Optional) Deploy a live demo

A live URL is the single most valuable thing to put on LinkedIn. The app is
designed to deploy as **one service** (FastAPI serves the built React app).

### Option A — Render (free tier, simplest)

1. `cd frontend && npm install && npm run build && cd ..` (creates `frontend/dist`).
   Commit `frontend/dist` for the simplest deploy, or run the build in Render's
   build command.
2. Push to GitHub.
3. On [render.com](https://render.com): New → Web Service → connect the repo.
   - **Build command:** `pip install -r requirements.txt && python scripts/build_index.py && cd frontend && npm install && npm run build`
   - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT --app-dir .`
4. Render gives you a public `https://news-search-engine.onrender.com` URL.

> The demo runs on the committed sample by default — light enough for free tiers.
> Skip `--bert` in deployment; the BERT model + PyTorch exceed free-tier memory.

### Option B — Railway / Fly.io

Same idea: one web service, build the frontend, run `uvicorn`. Add a
`Procfile` with:

```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT --app-dir .
```

## 5. Polish the repo page

- Add a short **About** description and topics on GitHub:
  `information-retrieval`, `search-engine`, `tf-idf`, `nlp`, `fastapi`, `react`.
- Pin the repo on your GitHub profile.
- Add the live demo URL to the repo's website field and to the README top.
