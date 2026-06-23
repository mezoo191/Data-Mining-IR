import { useEffect, useState } from "react";

const METHODS = [
  { id: "hybrid", label: "Hybrid", hint: "BM25 + BERT fusion (default)", semantic: true },
  { id: "bert", label: "BERT", hint: "Dense semantic retrieval", semantic: true },
  { id: "bm25", label: "BM25", hint: "Modern lexical ranking" },
  { id: "tfidf", label: "TF-IDF", hint: "Classic ranking, for comparison" },
  { id: "prf", label: "Relevance Feedback", hint: "Mark results relevant, then refine" },
  { id: "wordnet", label: "WordNet", hint: "BM25 + synonym expansion" },
];

// Only allow http(s) links through to href; guards against javascript:/data:
// URLs sneaking in from the dataset (defensive — the dataset is trusted).
function safeUrl(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? url : "#";
}

const EXAMPLES = [
  "covid vaccine health",
  "election president",
  "movie film review",
  "stock market crash",
  "climate change",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("hybrid");
  const [topK, setTopK] = useState(10);
  const [category, setCategory] = useState("");
  const [categories, setCategories] = useState([]);
  const [health, setHealth] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Doc ids the user marked relevant (for true relevance feedback on "prf").
  const [relevantIds, setRelevantIds] = useState(() => new Set());

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        setHealth(d);
        // Fall back to BM25 if BERT embeddings weren't built (e.g. `run.bat lite`).
        if (!d.bert_available) setMethod("bm25");
      })
      .catch(() => {});
    fetch("/api/categories")
      .then((r) => r.json())
      .then((d) => setCategories(d.categories || []))
      .catch(() => {});
  }, []);

  // withFeedback=true reuses the marked-relevant docs (the "Refine" action);
  // a fresh search clears them.
  async function runSearch(q = query, withFeedback = false) {
    if (!q.trim()) return;
    if (!withFeedback) setRelevantIds(new Set());
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({ q, method, top_k: topK });
      if (category) params.set("category", category);
      if (method === "prf" && withFeedback) {
        for (const id of relevantIds) params.append("relevant_ids", id);
      }
      const res = await fetch(`/api/search?${params}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setData(await res.json());
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  function toggleRelevant(id) {
    setRelevantIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const bertDisabled = health && !health.bert_available;
  const feedbackMode = method === "prf";

  return (
    <div className="app">
      <header className="hero">
        <h1>
          <span className="logo">◆</span> News Search Engine
        </h1>
        <p className="subtitle">
          Lexical (BM25/TF-IDF) &amp; semantic (BERT) retrieval with query expansion
          {health && (
            <span className="stats">
              {" "}· {health.documents.toLocaleString()} docs ·{" "}
              {health.vocabulary.toLocaleString()} terms
            </span>
          )}
        </p>
      </header>

      <section className="searchbar">
        <div className="input-row">
          <input
            type="text"
            value={query}
            placeholder="Search the news…"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()}
            autoFocus
          />
          <button onClick={() => runSearch()} disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => { setQuery(ex); runSearch(ex); }}>
              {ex}
            </button>
          ))}
        </div>

        <div className="controls">
          <div className="methods">
            {METHODS.map((m) => {
              const disabled = m.semantic && bertDisabled;
              return (
                <button
                  key={m.id}
                  className={`method ${method === m.id ? "active" : ""}`}
                  title={disabled ? "BERT not enabled on this deployment" : m.hint}
                  disabled={disabled}
                  onClick={() => setMethod(m.id)}
                >
                  {m.label}
                </button>
              );
            })}
          </div>

          <div className="filters">
            <label>
              Results
              <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[5, 10, 20, 30].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
            <label>
              Category
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="">All</option>
                {categories.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {feedbackMode && (
          <p className="fb-hint">
            Relevance feedback: tick the results that match what you want, then refine the search.
          </p>
        )}
      </section>

      {error && <div className="error">⚠ {error}</div>}

      {data && (
        <section className="results">
          <div className="results-meta">
            <span>
              <strong>{data.total_hits}</strong> result{data.total_hits !== 1 ? "s" : ""} for{" "}
              <em>“{data.query}”</em>
            </span>
            <span className="timing">{data.method} · {data.elapsed_ms} ms</span>
          </div>

          {data.expansion_terms?.length > 0 && (
            <div className="expansion">
              <span className="exp-label">Expanded with:</span>
              {data.expansion_terms.map((t) => (
                <span key={t} className="exp-term">{t}</span>
              ))}
            </div>
          )}

          {feedbackMode && data.results.length > 0 && (
            <div className="feedback-bar">
              <span>{relevantIds.size} marked relevant</span>
              <button
                onClick={() => runSearch(query, true)}
                disabled={loading || relevantIds.size === 0}
              >
                Refine with feedback
              </button>
            </div>
          )}

          {data.results.length === 0 && (
            <div className="empty">No documents matched this query.</div>
          )}

          <ol className="cards">
            {data.results.map((r) => (
              <li key={r.id} className={`card ${relevantIds.has(r.id) ? "marked" : ""}`}>
                <div className="card-head">
                  {feedbackMode && (
                    <input
                      type="checkbox"
                      className="relevant-check"
                      title="Mark relevant"
                      checked={relevantIds.has(r.id)}
                      onChange={() => toggleRelevant(r.id)}
                    />
                  )}
                  <span className="rank">#{r.rank}</span>
                  <a href={safeUrl(r.link)} target="_blank" rel="noreferrer" className="headline">
                    {r.headline}
                  </a>
                </div>
                <p className="desc">{r.short_description}</p>
                <div className="card-foot">
                  <span className="cat">{r.category}</span>
                  <span className="date">{r.date}</span>
                  <span className="score">score {r.score}</span>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      <footer className="foot">
        Built with FastAPI + React · inverted index, BM25 &amp; TF-IDF ranking, dense BERT
        retrieval, PRF / WordNet expansion &amp; relevance feedback
      </footer>
    </div>
  );
}
