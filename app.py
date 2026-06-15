"""
StuckOnMySofa — Web Interface
Flask app that serves the recommendation system via a web browser.
"""

import os
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── LOAD DATA AT STARTUP ──────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "comics_dataset_full.csv")

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    df["features"] = (
        df["tags"] + " " + df["tags"] + " " + df["tags"] + " " +
        df["genre"].str.lower() + " " + df["genre"].str.lower() + " " +
        df["audience"].str.lower()
    )
    return df

def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        max_features=8000
    )
    tfidf_matrix = vectorizer.fit_transform(df["features"])
    return tfidf_matrix, vectorizer

print("Loading dataset...", end=" ", flush=True)
df = load_dataset(DATASET_PATH)
tfidf_matrix, vectorizer = build_tfidf_matrix(df)
print(f"Done. {len(df)} titles loaded.")

# ── RECOMMENDATION LOGIC ──────────────────────────────────────────────────────

def recommend_by_title(title):
    matches = df[df["title"].str.lower() == title.strip().lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(
            title.strip().lower(), regex=False, na=False
        )]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0
    top_indices = sim_scores.argsort()[::-1][:5]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]
    return results, df.iloc[idx]

def recommend_by_description(query):
    query_vector = vectorizer.transform([query.lower()])
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:5]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]
    results = results[results["similarity"] > 0.01]
    return results

def row_to_dict(row, similarity=None):
    year = str(row["year"]).replace(".0", "") if str(row["year"]) not in ["nan", ""] else "?"
    return {
        "title": row["title"],
        "author": row["author"] if row["author"] not in ["", "nan"] else None,
        "year": year,
        "genre": row["genre"],
        "audience": row["audience"],
        "rating": round(float(row["goodreads_rating"] or 0), 2),
        "stars": int(round(float(row["goodreads_rating"] or 0))),
        "match": round(float(similarity) * 100, 1) if similarity is not None else None,
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    genres = sorted(df["genre"].unique().tolist())
    total = len(df)
    return render_template_string(HTML_TEMPLATE, genres=genres, total=total)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    mode = data.get("mode", "description")
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a search query."})

    if mode == "title":
        results, found = recommend_by_title(query)
        if results is None:
            return jsonify({"error": f"Title '{query}' not found. Try a partial name like 'batman'."})
        items = [row_to_dict(row, sim) for (_, row), sim in
                 zip(results.iterrows(), results["similarity"])]
        return jsonify({
            "results": items,
            "found": found["title"],
            "mode": "title"
        })
    else:
        results = recommend_by_description(query)
        if results is None or len(results) == 0:
            return jsonify({"error": "No results found. Try different keywords."})
        items = [row_to_dict(row, sim) for (_, row), sim in
                 zip(results.iterrows(), results["similarity"])]
        return jsonify({"results": items, "mode": "description"})

@app.route("/genre/<genre_name>")
def genre(genre_name):
    filtered = df[df["genre"] == genre_name].sort_values(
        "goodreads_rating", ascending=False
    ).head(10)
    items = [row_to_dict(row) for _, row in filtered.iterrows()]
    return jsonify({"results": items, "genre": genre_name})

# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StuckOnMySofa</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --ink:     #1a1a2e;
    --paper:   #f8f5f0;
    --cream:   #f0ebe3;
    --accent:  #e85d26;
    --muted:   #7a7a8c;
    --border:  #ddd8d0;
    --card:    #ffffff;
    --shadow:  0 2px 12px rgba(0,0,0,0.08);
  }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--paper);
    color: var(--ink);
    min-height: 100vh;
  }

  /* ── HEADER ── */
  header {
    background: var(--ink);
    padding: 2rem 2rem 1.5rem;
    text-align: center;
  }
  header h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
  }
  header h1 span { color: var(--accent); }
  header p {
    margin-top: 0.5rem;
    color: #9999aa;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .header-meta {
    margin-top: 0.8rem;
    color: #666677;
    font-size: 0.8rem;
  }

  /* ── SEARCH SECTION ── */
  .search-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 1rem;
  }

  /* Mode toggle */
  .mode-toggle {
    display: flex;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 1.2rem;
    gap: 4px;
  }
  .mode-btn {
    flex: 1;
    padding: 0.6rem 1rem;
    border: none;
    border-radius: 7px;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    background: transparent;
    color: var(--muted);
    transition: all 0.2s;
  }
  .mode-btn.active {
    background: var(--ink);
    color: #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
  }

  /* Search bar */
  .search-bar {
    display: flex;
    gap: 10px;
    align-items: stretch;
  }
  .search-input {
    flex: 1;
    padding: 0.85rem 1.1rem;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    border: 1.5px solid var(--border);
    border-radius: 10px;
    background: var(--card);
    color: var(--ink);
    outline: none;
    transition: border-color 0.2s;
  }
  .search-input:focus { border-color: var(--accent); }
  .search-input::placeholder { color: #aaa; }
  .search-btn {
    padding: 0.85rem 1.6rem;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    white-space: nowrap;
  }
  .search-btn:hover { background: #cf4d1e; }
  .search-btn:active { transform: scale(0.98); }

  /* Hint text */
  .search-hint {
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--muted);
    min-height: 1.2em;
  }

  /* ── KEYWORD CHIPS ── */
  .chips-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1.5rem 1.5rem;
  }
  .chips-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.6rem;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chip {
    padding: 0.35rem 0.85rem;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--ink);
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .chip:hover {
    background: var(--ink);
    color: #fff;
    border-color: var(--ink);
  }

  /* ── GENRE GRID ── */
  .genre-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1.5rem 2rem;
  }
  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.8rem;
  }
  .genre-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .genre-pill {
    padding: 0.4rem 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 0.82rem;
    color: var(--ink);
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
  }
  .genre-pill:hover {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .genre-pill.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }

  /* ── DIVIDER ── */
  .divider {
    max-width: 720px;
    margin: 0 auto 1.5rem;
    border: none;
    border-top: 1px solid var(--border);
  }

  /* ── RESULTS ── */
  .results-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1.5rem 3rem;
  }
  .results-header {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1rem;
    font-weight: 500;
  }
  .results-header strong { color: var(--ink); }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 10px;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    transition: box-shadow 0.15s, transform 0.15s;
    animation: fadeIn 0.3s ease;
  }
  .card:hover {
    box-shadow: var(--shadow);
    transform: translateY(-1px);
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--border);
    min-width: 2rem;
    line-height: 1;
    padding-top: 2px;
  }
  .card-body { flex: 1; }
  .card-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.3;
    margin-bottom: 0.3rem;
  }
  .card-meta {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    align-items: center;
  }
  .card-meta .genre-tag {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--ink);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .card-footer {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.4rem;
  }
  .stars { color: #f5a623; font-size: 0.9rem; letter-spacing: 1px; }
  .rating-num { font-size: 0.8rem; color: var(--muted); }
  .match-badge {
    margin-left: auto;
    background: #f0faf0;
    border: 1px solid #b8e0b8;
    color: #2d6a2d;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  /* Match bar */
  .match-bar-wrap {
    height: 3px;
    background: var(--cream);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .match-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.6s ease;
  }

  /* Error */
  .error-box {
    background: #fff5f5;
    border: 1px solid #fcc;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c0392b;
    font-size: 0.9rem;
    animation: fadeIn 0.3s ease;
  }

  /* Loading */
  .loading {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.9rem;
    animation: fadeIn 0.2s ease;
  }
  .spinner {
    display: inline-block;
    width: 20px; height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Footer */
  footer {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border);
    background: var(--cream);
  }
</style>
</head>
<body>

<header>
  <h1>Stuck<span>On</span>My<span>Sofa</span></h1>
  <p>Comic & Graphic Novel Recommender</p>
  <div class="header-meta">{{ total | int }} titles · TF-IDF + Cosine Similarity</div>
</header>

<div class="search-section">
  <!-- Mode toggle -->
  <div class="mode-toggle">
    <button class="mode-btn active" id="btn-desc" onclick="setMode('description')">
      🔍 Describe what you want
    </button>
    <button class="mode-btn" id="btn-title" onclick="setMode('title')">
      📖 Search by title
    </button>
  </div>

  <!-- Search bar -->
  <div class="search-bar">
    <input
      type="text"
      class="search-input"
      id="search-input"
      placeholder="dark political superhero adult..."
      onkeydown="if(event.key==='Enter') doSearch()"
      autocomplete="off"
    />
    <button class="search-btn" onclick="doSearch()">Search</button>
  </div>
  <div class="search-hint" id="search-hint">
    Describe themes, mood, genre, audience — the more specific, the better
  </div>
</div>

<!-- Keyword chips -->
<div class="chips-section" id="chips-section">
  <div class="chips-label">Quick keywords</div>
  <div class="chips">
    <span class="chip" onclick="addChip('dark')">dark</span>
    <span class="chip" onclick="addChip('political')">political</span>
    <span class="chip" onclick="addChip('superhero')">superhero</span>
    <span class="chip" onclick="addChip('horror')">horror</span>
    <span class="chip" onclick="addChip('fantasy')">fantasy</span>
    <span class="chip" onclick="addChip('manga')">manga</span>
    <span class="chip" onclick="addChip('romance')">romance</span>
    <span class="chip" onclick="addChip('sci-fi')">sci-fi</span>
    <span class="chip" onclick="addChip('mystery')">mystery</span>
    <span class="chip" onclick="addChip('coming-of-age')">coming of age</span>
    <span class="chip" onclick="addChip('pirates')">pirates</span>
    <span class="chip" onclick="addChip('psychological')">psychological</span>
    <span class="chip" onclick="addChip('historical')">historical</span>
    <span class="chip" onclick="addChip('memoir')">memoir</span>
    <span class="chip" onclick="addChip('adult')">adult</span>
    <span class="chip" onclick="addChip('young-adult')">young adult</span>
    <span class="chip" onclick="addChip('children')">children</span>
    <span class="chip" onclick="addChip('comedy')">comedy</span>
    <span class="chip" onclick="addChip('war')">war</span>
    <span class="chip" onclick="addChip('ninja')">ninja</span>
  </div>
</div>

<!-- Genre grid -->
<div class="genre-section" id="genre-section">
  <div class="section-title">Browse by genre</div>
  <div class="genre-grid">
    {% for g in genres %}
    <span class="genre-pill" onclick="browseGenre('{{ g }}')">{{ g }}</span>
    {% endfor %}
  </div>
</div>

<hr class="divider">

<!-- Results -->
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
  const chips = document.getElementById('chips-section');
  const genres = document.getElementById('genre-section');
  document.getElementById('btn-desc').classList.toggle('active', mode === 'description');
  document.getElementById('btn-title').classList.toggle('active', mode === 'title');

  if (mode === 'title') {
    input.placeholder = 'Watchmen, Berserk, One Piece...';
    hint.textContent  = 'Enter a full or partial title to find similar comics';
    chips.style.display  = 'none';
    genres.style.display = 'none';
  } else {
    input.placeholder = 'dark political superhero adult...';
    hint.textContent  = 'Describe themes, mood, genre, audience — the more specific, the better';
    chips.style.display  = 'block';
    genres.style.display = 'block';
  }
  input.focus();
  clearResults();
}

function addChip(word) {
  const input = document.getElementById('search-input');
  const val = input.value.trim();
  if (!val.includes(word)) {
    input.value = val ? val + ' ' + word : word;
  }
  input.focus();
}

function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;

  // reset active genre
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
    if (data.error) {
      showError(data.error);
    } else {
      let headerText = '';
      if (data.mode === 'title' && data.found) {
        headerText = `Because you searched for <strong>${data.found}</strong>`;
      } else {
        headerText = `Top results for <strong>"${query}"</strong>`;
      }
      showResults(data.results, headerText);
    }
  })
  .catch(() => showError('Something went wrong. Please try again.'));
}

function browseGenre(genre) {
  // toggle
  if (activeGenre === genre) {
    activeGenre = null;
    document.querySelectorAll('.genre-pill').forEach(p => p.classList.remove('active'));
    clearResults();
    return;
  }
  activeGenre = genre;
  document.querySelectorAll('.genre-pill').forEach(p => {
    p.classList.toggle('active', p.textContent === genre);
  });

  showLoading();
  fetch(`/genre/${encodeURIComponent(genre)}`)
  .then(r => r.json())
  .then(data => {
    showResults(data.results, `Top titles in <strong>${genre}</strong>`);
  })
  .catch(() => showError('Something went wrong. Please try again.'));
}

function showLoading() {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML =
    '<div class="loading"><span class="spinner"></span>Finding recommendations...</div>';
}

function clearResults() {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML = '';
}

function showError(msg) {
  document.getElementById('results-header').style.display = 'none';
  document.getElementById('results-container').innerHTML =
    `<div class="error-box">⚠️ ${msg}</div>`;
}

function showResults(items, headerText) {
  const hdr = document.getElementById('results-header');
  hdr.innerHTML = headerText;
  hdr.style.display = 'block';

  const container = document.getElementById('results-container');
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="error-box">No results found. Try different keywords.</div>';
    return;
  }

  container.innerHTML = items.map((item, i) => {
    const stars = '★'.repeat(item.stars) + '☆'.repeat(5 - item.stars);
    const matchHTML = item.match !== null ? `
      <span class="match-badge">${item.match}% match</span>
    ` : '';
    const matchBar = item.match !== null ? `
      <div class="match-bar-wrap">
        <div class="match-bar-fill" style="width:${item.match}%"></div>
      </div>
    ` : '';
    const author = item.author ? `<span>${item.author}</span>` : '';
    const year   = item.year && item.year !== '?' ? `<span>${item.year}</span>` : '';

    return `
      <div class="card" style="animation-delay:${i * 0.05}s">
        <div class="card-rank">${i + 1}</div>
        <div class="card-body">
          <div class="card-title">${item.title}</div>
          <div class="card-meta">
            ${author}
            ${year}
            <span class="genre-tag">${item.genre}</span>
            <span>${item.audience}</span>
          </div>
          <div class="card-footer">
            <span class="stars">${stars}</span>
            <span class="rating-num">${item.rating}/5</span>
            ${matchHTML}
          </div>
          ${matchBar}
        </div>
      </div>
    `;
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
