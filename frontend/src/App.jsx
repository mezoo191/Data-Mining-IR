import { useEffect, useState } from "react";

const METHODS = [
  { id: "bm25", label: "BM25", hint: "Modern ranking" },
  { id: "tfidf", label: "TF-IDF", hint: "Classic ranking, for comparison" },
  { id: "prf", label: "Relevance Feedback", hint: "BM25 + pseudo-relevance feedback" },
  { id: "wordnet", label: "WordNet", hint: "BM25 + synonym expansion" },
  { id: "bert", label: "BERT", hint: "BM25 + semantic embeddings (default)" },
  { id: "prf+bert", label: "PRF + BERT", hint: "BM25 + hybrid expansion" },
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
  const [method, setMethod] = useState("bert");
  const [topK, setTopK] = useState(10);
  const [category, setCategory] = useState("");
  const [categories, setCategories] = useState([]);
  const [health, setHealth] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        setHealth(d);
        // Fall back to BM25 if BERT wasn't built (e.g. `run.bat lite`).
        if (!d.bert_available) setMethod("bm25");
      })
      .catch(() => {});
    fetch("/api/categories")
      .then((r) => r.json())
      .then((d) => setCategories(d.categories || []))
      .catch(() => {});
  }, []);

  async function runSearch(q = query) {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({ q, method, top_k: topK });
      if (category) params.set("category", category);
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

  const bertDisabled = health && !health.bert_available;

  return (
    <div className="app">
      <header className="hero">
        <h1>
          <span className="logo">◆</span> News Search Engine
        </h1>
        <p className="subtitle">
          TF-IDF information retrieval with query expansion
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
              const disabled = (m.id === "bert" || m.id === "prf+bert") && bertDisabled;
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

          {data.results.length === 0 && (
            <div className="empty">No documents matched this query.</div>
          )}

          <ol className="cards">
            {data.results.map((r) => (
              <li key={r.id} className="card">
                <div className="card-head">
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
        Built with FastAPI + React · inverted index, BM25 &amp; TF-IDF ranking, PRF / WordNet / BERT expansion
      </footer>
    </div>
  );
}
