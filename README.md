# 📚 StuckOnMySofa — Manga Recommender

> A content-based recommendation system for manga.  
> I hope it may help you when you feel stuck on your sofa, deciding what to read.

Built for the Data Mining and Text Analytics course — IULM University, A.A. 2025-2026.

---

## What it does

StuckOnMySofa recommends manga based on:

- **A title** — find manga similar to one you already like
- **A description** — describe what you want to read in plain English
- **A genre** — browse the best titles in a category

The system uses **TF-IDF vectorization** and **cosine similarity** to match your query against a curated dataset of 462 top-rated manga from MyAnimeList.

---

## Dataset

- **Source**: MyAnimeList (MAL) — top 600 manga by score
- **After filtering**: 462 titles (explicit content removed)
- **Features used**: genres, themes, synopsis, demographics
- **Rating scale**: MAL score (0–10)

---

## Getting Started

### Requirements

- [Anaconda](https://www.anaconda.com/) or Python 3.10+

### Setup

```bash
# 1. Create and activate the conda environment
conda create -n stuckonmysofa python=3.10
conda activate stuckonmysofa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the terminal interface
python stuckonmysofa.py

# 4. Or run the web interface
python app.py
# Then open http://127.0.0.1:5000 in your browser
```

### Files in this folder

| File | Description |
|------|-------------|
| `stuckonmysofa.py` | Terminal-based recommender (main deliverable) |
| `app.py` | Web interface (Flask) |
| `manga_dataset_600_taggato.csv` | Dataset — 462 curated manga from MyAnimeList |
| `requirements.txt` | Python dependencies |

---

## How it works

1. Each manga is represented as a **TF-IDF vector** built from its genres, themes, synopsis and demographic
2. When you search, your query is transformed into the same vector space
3. **Cosine similarity** measures how close your query is to each manga
4. The final score combines similarity (60%) and MAL rating (40%)
5. The top 5 results are returned

---

## Authors

Mattia, Jacopo, Alessio  
Data Mining and Text Analytics — Prof. Dr. Alessandro Bruno  
IULM University, A.A. 2025-2026
