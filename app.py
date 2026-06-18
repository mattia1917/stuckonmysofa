"""
StuckOnMySofa — Manga Recommender
==================================================
A content-based recommendation system for manga.
Uses TF-IDF vectorization and cosine similarity to find similar titles
or match a free-text description to the best entries in the dataset.

Authors  : Mattia, Jacopo, Alessio
Course   : Data Mining and Text Analytics 2025-2026
Professor: Dr. Alessandro Bruno — IULM University
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "top 1000 manga.csv")
TOP_N        = 5
SEPARATOR    = "─" * 60

# ── LOAD & PREPARE DATA ────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and build a combined feature string for each entry."""
    if not os.path.exists(path):
        print(f"\n[ERROR] Dataset not found at: {path}")
        print("Make sure 'top 1000 manga.csv' is in the same folder as this script.")
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.fillna("")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    # Combined feature string — titolo inglese, genres, themes, synopsis, demographics
    title_eng = df["title_english"].fillna("").str.lower() if "title_english" in df.columns else ""
    df["features"] = (
        title_eng + " " +
        df["genres"].str.lower().str.replace("|", " ") + " " +
        df["themes"].str.lower().str.replace("|", " ") + " " +
        df["synopsis"].str.lower() + " " +
        df["demographics"].str.lower()
    )
    return df


def build_tfidf_matrix(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer on the feature strings and return matrix + vectorizer."""
    vectorizer = TfidfVectorizer(
        stop_words="english",   # rimuove parole inutili (the, is, and...)
        ngram_range=(1, 2),     # considera parole singole e coppie di parole
        min_df=1,               # ignora parole che appaiono in meno di 2 fumetti
        max_df=0.85,            # ignora parole che appaiono in più dell'85% dei fumetti
        sublinear_tf=True,      # log-normalizza le frequenze per risultati più bilanciati
        max_features=8000       # vocabolario massimo di 8000 parole
    )
    tfidf_matrix = vectorizer.fit_transform(df["features"])
    return tfidf_matrix, vectorizer


# ── RECOMMENDATION LOGIC ───────────────────────────────────────────────────────

def recommend_by_title(title: str, df: pd.DataFrame, tfidf_matrix):
    """Find comics similar to a given title using cosine similarity."""

    # Cerca corrispondenza esatta prima, poi parziale
    matches = df[df["title"].str.lower() == title.strip().lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(
            title.strip().lower(), regex=False, na=False
        )]
    if matches.empty:
        return None

    # Calcola cosine similarity tra il fumetto scelto e tutti gli altri
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0  # escludi il titolo stesso dai risultati

    # Combina similarità (60%) e rating normalizzato (40%)
    rating_boost = pd.to_numeric(df["score"], errors="coerce").fillna(0).values / 10.0
    combined     = sim_scores * 0.60 + rating_boost * 0.40

    # Prendi i TOP_N fumetti con punteggio più alto
    top_indices = combined.argsort()[::-1][:TOP_N]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]
    return results, df.iloc[idx]


def recommend_by_description(query: str, df: pd.DataFrame,
                              tfidf_matrix, vectorizer):
    """Find comics that best match a free-text description."""

    # Trasforma la query con lo stesso vectorizer usato per il dataset
    query_vector = vectorizer.transform([query.lower()])
    sim_scores   = cosine_similarity(query_vector, tfidf_matrix).flatten()


# ── COMBINA SIMILARITÀ E RATING ───────────────────────────────────────────────

    rating_boost = pd.to_numeric(df["score"], errors="coerce").fillna(0).values / 10.0
    combined     = sim_scores * 0.60 + rating_boost * 0.40

    top_indices = combined.argsort()[::-1][:TOP_N]
    results     = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]

    results = results[results["similarity"] > 0.03]
    return results


# ── FLASK ─────────────────────────────────────────────────────────────────────

from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

print("\nLoading dataset...", end=" ", flush=True)
df            = load_dataset(DATASET_PATH)
tfidf_matrix, vectorizer = build_tfidf_matrix(df)
ALL_GENRES    = sorted(df["genres"].str.split("|").explode().str.strip().unique().tolist())
ALL_GENRES    = [g for g in ALL_GENRES if g]
print(f"Done. {len(df)} titles loaded.")


def row_to_dict(row, similarity=None):
    rating = float(row.get("score", 0) or 0)
    stars  = max(0, min(5, int(round(rating / 2))))
    genres = [g.strip() for g in str(row.get("genres", "")).split("|") if g.strip()]
    return {
        "title":       row.get("title", ""),
        "author":      row.get("authors", "") or None,
        "year":        str(row.get("published_from", ""))[:4] or "?",
        "genres":      genres,
        "audience":    row.get("demographics", ""),
        "rating":      round(rating, 2),
        "stars":       stars,
        "match":       round(float(similarity) * 100, 1) if similarity is not None else None,
        "themes":      [t.strip() for t in str(row.get("themes", "")).split("|") if t.strip()][:5],
    }


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, genres=ALL_GENRES, total=len(df))


@app.route("/search", methods=["POST"])
def search():
    data  = request.get_json() or {}
    mode  = data.get("mode", "description")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a search query."})

    if mode == "title":
        result = recommend_by_title(query, df, tfidf_matrix)
        if result is None:
            return jsonify({"error": f"Title not found. Try a partial name like 'berserk'."})
        results, found = result
        items = [row_to_dict(row, sim) for (_, row), sim in zip(results.iterrows(), results["similarity"])]
        return jsonify({"results": items, "found": found["title"], "mode": "title"})

    results = recommend_by_description(query, df, tfidf_matrix, vectorizer)
    if results is None or len(results) == 0:
        return jsonify({"error": "No results found. Try different keywords."})
    items = [row_to_dict(row, sim) for (_, row), sim in zip(results.iterrows(), results["similarity"])]
    return jsonify({"results": items, "mode": "description"})


@app.route("/genre/<genre_name>")
def genre(genre_name):
    mask = df["genres"].str.contains(genre_name, regex=False, na=False)
    filtered = df[mask].sort_values("score", ascending=False).head(10)
    items = [row_to_dict(row) for _, row in filtered.iterrows()]
    return jsonify({"results": items, "genre": genre_name})


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StuckOnMySofa</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ink: #1a1a2e; --paper: #f8f5f0; --cream: #f0ebe3;
    --accent: #e85d26; --muted: #7a7a8c; --border: #ddd8d0;
    --card: #ffffff; --shadow: 0 2px 12px rgba(0,0,0,0.08);
  }
  body { font-family: 'Inter', sans-serif; background: var(--paper); color: var(--ink); min-height: 100vh; }
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap');

  header { background: var(--ink); padding: 2rem 2rem 1.5rem; text-align: center; }
  header h1 { font-family: 'Playfair Display', serif; font-size: clamp(2rem,5vw,3.5rem); font-weight: 900; color: #fff; letter-spacing: -0.02em; }
  header h1 span { color: var(--accent); }
  header p { margin-top: 0.5rem; color: #9999aa; font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase; }
  .header-meta { margin-top: 0.8rem; color: #666677; font-size: 0.8rem; }

  .search-section { max-width: 760px; margin: 0 auto; padding: 2.5rem 1.5rem 1rem; }
  .mode-toggle { display: flex; background: var(--cream); border: 1px solid var(--border); border-radius: 10px; padding: 4px; margin-bottom: 1.2rem; gap: 4px; }
  .mode-btn { flex: 1; padding: 0.6rem 1rem; border: none; border-radius: 7px; font-size: 0.88rem; font-weight: 500; cursor: pointer; background: transparent; color: var(--muted); transition: all 0.2s; }
  .mode-btn.active { background: var(--ink); color: #fff; }
  .search-bar { display: flex; gap: 10px; }
  .search-input { flex: 1; padding: 0.85rem 1.1rem; font-size: 1rem; border: 1.5px solid var(--border); border-radius: 10px; background: var(--card); color: var(--ink); outline: none; transition: border-color 0.2s; }
  .search-input:focus { border-color: var(--accent); }
  .search-input::placeholder { color: #aaa; }
  .search-btn { padding: 0.85rem 1.6rem; background: var(--accent); color: #fff; border: none; border-radius: 10px; font-size: 0.95rem; font-weight: 600; cursor: pointer; }
  .search-btn:hover { background: #cf4d1e; }
  .search-hint { margin-top: 0.6rem; font-size: 0.82rem; color: var(--muted); }

  .chips-section, .genre-section { max-width: 760px; margin: 0 auto; padding: 0 1.5rem 1.5rem; }
  .chips-label, .section-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.6rem; }
  .chips, .genre-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip, .genre-pill { padding: 0.35rem 0.85rem; background: var(--cream); border: 1px solid var(--border); border-radius: 20px; font-size: 0.82rem; font-weight: 500; color: var(--ink); cursor: pointer; transition: all 0.15s; }
  .chip:hover { background: var(--ink); color: #fff; border-color: var(--ink); }
  .genre-pill:hover, .genre-pill.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  hr.divider { max-width: 760px; margin: 0 auto 1.5rem; border: none; border-top: 1px solid var(--border); }

  .results-section { max-width: 760px; margin: 0 auto; padding: 0 1.5rem 3rem; }
  .results-header { font-size: 0.8rem; color: var(--muted); margin-bottom: 1rem; font-weight: 500; }
  .results-header strong { color: var(--ink); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 10px; display: flex; gap: 1rem; animation: fadeIn 0.3s ease; }
  .card:hover { box-shadow: var(--shadow); transform: translateY(-1px); }
  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  .card-rank { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 900; color: var(--border); min-width: 2rem; line-height: 1; padding-top: 2px; }
  .card-body { flex: 1; }
  .card-title { font-size: 1rem; font-weight: 600; color: var(--ink); margin-bottom: 0.3rem; }
  .card-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
  .genre-tag { background: var(--cream); border: 1px solid var(--border); border-radius: 4px; padding: 1px 7px; font-size: 0.72rem; font-weight: 600; color: var(--ink); text-transform: uppercase; }
  .card-footer { display: flex; align-items: center; gap: 1rem; margin-top: 0.4rem; }
  .stars { color: #f5a623; font-size: 0.9rem; }
  .rating-num { font-size: 0.8rem; color: var(--muted); }
  .match-badge { margin-left: auto; background: #f0faf0; border: 1px solid #b8e0b8; color: #2d6a2d; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .match-bar-wrap { height: 3px; background: var(--cream); border-radius: 2px; margin-top: 6px; }
  .match-bar-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.6s ease; }
  .error-box { background: #fff5f5; border: 1px solid #fcc; border-radius: 10px; padding: 1rem 1.2rem; color: #c0392b; font-size: 0.9rem; }
  .loading { text-align: center; padding: 2rem; color: var(--muted); font-size: 0.9rem; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer { text-align: center; padding: 2rem; color: var(--muted); font-size: 0.78rem; border-top: 1px solid var(--border); background: var(--cream); }
</style>
</head>
<body>

<header>
  <h1>Stuck<span>On</span>My<span>Sofa</span></h1>
  <p>Manga Recommender</p>
  <div class="header-meta">{{ total }} titles · TF-IDF + Cosine Similarity</div>
</header>

<div class="search-section">
  <div class="mode-toggle">
    <button class="mode-btn active" id="btn-desc" onclick="setMode('description')">🔍 Describe what you want to read</button>
    <button class="mode-btn" id="btn-title" onclick="setMode('title')">📖 Search by title</button>
  </div>
  <div class="search-bar">
    <input type="text" class="search-input" id="search-input"
      placeholder="dark psychological, sports boxing, seinen..."
      onkeydown="if(event.key==='Enter') doSearch()" autocomplete="off"/>
    <button class="search-btn" onclick="doSearch()">Search</button>
  </div>
  <div class="search-hint" id="search-hint">
    Search by genres, themes, mood or audience. Example: dark psychological, sports boxing, Seinen
  </div>
</div>

<div class="genre-section" id="genre-section">
  <div class="section-title">Browse by genre</div>
  <div class="genre-grid">
    {% for g in genres %}
    <span class="genre-pill" onclick="browseGenre('{{ g }}')">{{ g }}</span>
    {% endfor %}
  </div>
</div>

<hr class="divider">

<div class="results-section">
  <div id="results-header" class="results-header" style="display:none"></div>
  <div id="results-container"></div>
</div>

<footer>
  StuckOnMySofa &mdash; IULM University, A.A. 2025-2026 &mdash; Mattia, Jacopo, Alessio<br>
  Data Mining and Text Analytics &mdash; Prof. Dr. Alessandro Bruno
</footer>

<script>
let currentMode = 'description';
let activeGenre = null;

function setMode(mode) {
  currentMode = mode;
  const input = document.getElementById('search-input');
  const hint  = document.getElementById('search-hint');
  const genres = document.getElementById('genre-section');
  document.getElementById('btn-desc').classList.toggle('active', mode === 'description');
  document.getElementById('btn-title').classList.toggle('active', mode === 'title');
  if (mode === 'title') {
    input.placeholder = 'Monster, Berserk, One Piece...';
    hint.textContent  = 'Enter a full or partial title to find similar manga';
    genres.style.display = 'none';
  } else {
    input.placeholder = 'dark psychological, sports boxing, seinen...';
    hint.textContent  = 'Search by genres, themes, mood or audience. Example: dark psychological, Seinen';
    genres.style.display = 'block';
  }
  input.focus();
  clearResults();
}


function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;
  document.querySelectorAll('.genre-pill').forEach(p => p.classList.remove('active'));
  activeGenre = null;
  showLoading();
  fetch('/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ mode: currentMode, query })
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showError(data.error); return; }
    const hdr = data.mode === 'title' && data.found
      ? `Because you searched for <strong>${esc(data.found)}</strong>`
      : `Top results for <strong>"${esc(query)}"</strong>`;
    showResults(data.results, hdr);
  })
  .catch(() => showError('Something went wrong. Please try again.'));
}

function browseGenre(genre) {
  if (activeGenre === genre) {
    document.querySelectorAll('.genre-pill').forEach(p => p.classList.remove('active'));
    activeGenre = null; clearResults(); return;
  }
  activeGenre = genre;
  document.querySelectorAll('.genre-pill').forEach(p => p.classList.toggle('active', p.textContent === genre));
  showLoading();
  fetch(`/genre/${encodeURIComponent(genre)}`)
  .then(r => r.json())
  .then(data => showResults(data.results, `Top manga in <strong>${esc(genre)}</strong>`))
  .catch(() => showError('Something went wrong.'));
}

function showLoading() {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML = '<div class="loading"><span class="spinner"></span>Finding recommendations...</div>';
}
function clearResults() {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML = '';
}
function showError(msg) {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML = `<div class="error-box">⚠️ ${esc(msg)}</div>`;
}
function esc(v) {
  return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
}

function showResults(items, headerText) {
  const hdr = document.getElementById('results-header');
  hdr.innerHTML = headerText;
  hdr.style.display = 'block';
  if (!items || !items.length) {
    document.getElementById('results-container').innerHTML = '<div class="error-box">No results found. Try different keywords.</div>';
    return;
  }
  document.getElementById('results-container').innerHTML = items.map((item, i) => {
    const stars = '★'.repeat(item.stars) + '☆'.repeat(5 - item.stars);
    const matchHTML = item.match !== null ? `<span class="match-badge">${item.match}% match</span>` : '';
    const matchBar  = item.match !== null ? `<div class="match-bar-wrap"><div class="match-bar-fill" style="width:${Math.min(100,item.match)}%"></div></div>` : '';
    const genres = (item.genres||[]).map(g => `<span class="genre-tag">${esc(g)}</span>`).join('');
    return `<div class="card" style="animation-delay:${i*0.05}s">
      <div class="card-rank">${i+1}</div>
      <div class="card-body">
        <div class="card-title">${esc(item.title)}</div>
        <div class="card-meta">
          ${item.author ? `<span>${esc(item.author)}</span>` : ''}
          ${item.year && item.year !== '?' ? `<span>${esc(item.year)}</span>` : ''}
          ${genres}
          ${item.audience ? `<span>${esc(item.audience)}</span>` : ''}
        </div>
        <div class="card-footer">
          <span class="stars">${stars}</span>
          <span class="rating-num">${item.rating}/10</span>
          ${matchHTML}
        </div>
        ${matchBar}
      </div>
    </div>`;
  }).join('');
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting StuckOnMySofa web interface...")
    print("Open your browser at: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
