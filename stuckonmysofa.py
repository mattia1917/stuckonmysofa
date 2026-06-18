"""
StuckOnMySofa — Manga Recommender
==================================================
A content-based recommendation system for manga.

The system uses TF-IDF vectorization and cosine similarity to:
1. find manga similar to a title already known by the user;
2. find manga that match a free-text description written by the user;
3. browse the best manga in a selected genre.

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

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "manga_dataset_top_1000_rating_english_titles.csv"
)

TOP_N = 5
SEPARATOR = "─" * 60


# ── LOAD & PREPARE DATA ───────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and build a combined feature string for each entry."""
    if not os.path.exists(path):
        print(f"\n[ERROR] Dataset not found at: {path}")
        print(
            "top 1000 manga.csv"
            "is in the same folder as this script."
        )
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.fillna("")

    # Combined feature string — genres, themes, synopsis, demographics
    df["features"] = (
        df["genres"].str.lower().str.replace("|", " ", regex=False) + " " +
        df["themes"].str.lower().str.replace("|", " ", regex=False) + " " +
        df["synopsis"].str.lower() + " " +
        df["demographics"].str.lower()
    )

    return df


def build_tfidf_matrix(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer on the feature strings and return matrix + vectorizer."""
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


# ── RECOMMENDATION LOGIC ──────────────────────────────────────────────────────

def recommend_by_title(title: str, df: pd.DataFrame, tfidf_matrix):
    """Find manga similar to a given title using cosine similarity."""

    # Search for an exact title first, then for a partial title
    matches = df[df["title"].str.lower() == title.strip().lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(
            title.strip().lower(), regex=False, na=False
        )]

    if matches.empty:
        return None

    # Retrieve the vector of the selected manga from the TF-IDF matrix
    idx = matches.index[0]

    # Compare that vector with every other manga vector in the matrix
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0

    # Combined score: 60% cosine similarity + 40% MAL score normalized from 0-10 to 0-1
    rating_boost = pd.to_numeric(df["score"], errors="coerce").fillna(0).values / 10.0
    combined = sim_scores * 0.60 + rating_boost * 0.40

    # Select the TOP_N manga with the highest combined score
    top_indices = combined.argsort()[::-1][:TOP_N]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]

    return results, df.iloc[idx]


def recommend_by_description(
    query: str,
    df: pd.DataFrame,
    tfidf_matrix,
    vectorizer
):
    """Find manga that best match a free-text description."""

    # Transform the user's query using the vocabulary already learned from the dataset
    query_vector = vectorizer.transform([query.lower()])

    # Compare the query vector with every manga vector in the matrix
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Combined score: 60% cosine similarity + 40% MAL score normalized from 0-10 to 0-1
    rating_boost = pd.to_numeric(df["score"], errors="coerce").fillna(0).values / 10.0
    combined = sim_scores * 0.60 + rating_boost * 0.40

    # Select the TOP_N manga with the highest combined score
    top_indices = combined.argsort()[::-1][:TOP_N]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sim_scores[top_indices]

    # Remove results with almost zero textual similarity
    results = results[results["similarity"] > 0.01]

    return results


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────

def stars(score) -> str:
    """Convert a MAL score from 0-10 into a 5-star string."""
    try:
        filled = int(round(float(score) / 2))
    except (TypeError, ValueError):
        filled = 0

    filled = max(0, min(5, filled))

    return "★" * filled + "☆" * (5 - filled)


def format_year(value) -> str:
    """Extract the publication year from the published_from column."""
    text = str(value).strip()

    if text in ["", "nan", "NaT"]:
        return "?"

    return text[:4]


def print_header():
    """Print the program header."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║{:^58}║".format("📚  StuckOnMySofa"))
    print("║{:^58}║".format("Manga Recommender"))
    print("╚" + "═" * 58 + "╝")
    print()


def print_entry(row, rank: int = None, similarity: float = None):
    """Print a single manga entry in a formatted way."""
    prefix = f"  {rank}. " if rank else "  ► "
    year = format_year(row["published_from"])

    print(f"{prefix}{row['title']}  ({year})")

    if row["authors"] not in ["", "nan"]:
        print(f"     Author : {row['authors']}")

    print(f"     Genre  : {row['genres']}")
    print(f"     Target : {row['demographics']}")
    print(f"     Rating : {stars(row['score'])} {float(row['score'] or 0):.2f}/10")

    if similarity is not None:
        print(f"     Match  : {similarity * 100:.1f}%")

    print()


def print_recommendations(results: pd.DataFrame, found=None):
    """Print the list of recommendations."""
    print(SEPARATOR)

    if found is not None:
        print(f"  Because you searched for: {found['title']}")
        print(f"  ({found['genres']} · {found['demographics']})")

    print(SEPARATOR)
    print()

    if results is None or len(results) == 0:
        print("  No recommendations found. Try a different search.\n")
        return

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print_entry(row, rank=rank, similarity=row.get("similarity"))


def get_available_genres(df: pd.DataFrame):
    """Return the list of individual genres contained in the dataset."""
    genre_set = set()

    for value in df["genres"]:
        for genre in str(value).split("|"):
            genre = genre.strip()
            if genre:
                genre_set.add(genre)

    return sorted(genre_set)


# ── MAIN INTERACTIVE LOOP ─────────────────────────────────────────────────────

def main():
    # Load the dataset and build the TF-IDF matrix once at startup
    print("\nLoading dataset...", end=" ", flush=True)
    df = load_dataset(DATASET_PATH)
    tfidf_matrix, vectorizer = build_tfidf_matrix(df)
    print(f"Done. {len(df)} titles loaded.")

    print_header()

    while True:
        print(SEPARATOR)
        print("  What would you like to do?")
        print()
        print("  [1]  Find manga similar to a title")
        print("  [2]  Describe what you want to read")
        print("  [3]  Browse by genre")
        print("  [0]  Exit")
        print()

        choice = input("  Your choice: ").strip()

        # ── MODE 1: search by title ────────────────────────────────────────
        if choice == "1":
            print()
            title = input("  Enter a title: ").strip()

            if not title:
                continue

            result = recommend_by_title(title, df, tfidf_matrix)

            if result is None:
                print(f"\n  Title '{title}' not found. Try a partial title like 'monster'.\n")
                continue

            results, found = result

            print(f"\n  Found: {found['title']}\n")
            print_recommendations(results, found=found)

        # ── MODE 2: free-text description ──────────────────────────────────
        elif choice == "2":
            print()
            print("  Describe what you want to read.")
            print("  Example: 'dark psychological revenge'")
            print("           'pirates adventure friendship'")
            print("           'school romance comedy'")
            print()

            query = input("  Your description: ").strip()

            if not query:
                continue

            results = recommend_by_description(query, df, tfidf_matrix, vectorizer)

            print("\n  Top recommendations for your description:")
            print()
            print_recommendations(results)

        # ── MODE 3: browse by genre ────────────────────────────────────────
        elif choice == "3":
            genres = get_available_genres(df)

            print()
            print("  Available genres:")
            print()

            for i, genre in enumerate(genres, start=1):
                count = df["genres"].str.contains(genre, regex=False, na=False).sum()
                print(f"  [{i:>2}]  {genre}  ({count} titles)")

            print()

            pick = input("  Choose a number: ").strip()

            try:
                genre = genres[int(pick) - 1]
                filtered = df[df["genres"].str.contains(
                    genre, regex=False, na=False
                )].sort_values("score", ascending=False)

                print(f"\n  Top {min(TOP_N, len(filtered))} titles in '{genre}':\n")

                for rank, (_, row) in enumerate(filtered.head(TOP_N).iterrows(), start=1):
                    print_entry(row, rank=rank)

            except (ValueError, IndexError):
                print("\n  Invalid choice.\n")

        # ── EXIT ───────────────────────────────────────────────────────────
        elif choice == "0":
            print()
            print("  Thanks for using StuckOnMySofa. Happy reading! 📚")
            print()
            break

        else:
            print("\n  Invalid choice. Please enter 1, 2, 3 or 0.\n")


if __name__ == "__main__":
    main()
