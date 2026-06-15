# StuckOnMySofa — Comic & Graphic Novel Recommender

> A content-based recommendation system for comics and graphic novels.  
> Built for the Data Mining and Text Analytics course — IULM University, A.A. 2025-2026.

---

## What is StuckOnMySofa?

StuckOnMySofa helps you decide what to read next when you are stuck on your sofa choosing for good comics watchinng reviews instead of reading.

You can either search by title — "find me something like Watchmen" — or describe what you are in the mood for — "dark political" — and the system will recommend the best matching comics and graphic novels from its dataset of over 15,000 titles.

The system uses **TF-IDF vectorization** and **cosine similarity** — core data mining techniques covered in this course — to compare titles based on their genre, themes, audience, and content tags.

---

## Dataset

The dataset contains **15,797 comics and graphic novels**, assembled from three sources:

- **MyAnimeList** — manga titles with genre, themes, demographics and scores
- **Kaggle: Marvel Comics Dataset** — Marvel series with titles, authors and ratings
- **Kaggle: Complete DC Comic Books** — DC series with titles, authors and release dates

Each entry includes: title, author, year, genre, audience, pages, rating, and content tags.

Genres covered: Superhero, Manga, Fantasy, Romance, Slice of Life, Science Fiction, Mystery, Horror, Action, Comedy, War, Drama, Historical, Sports, Adventure, Western, Political, Memoir, Crime.

---

## Features

- **Mode 1 — Similar titles**: enter a title and get 5 similar recommendations with match score
- **Mode 2 — Free-text search**: describe what you want to read in plain language
- **Mode 3 — Browse by genre**: explore titles filtered by genre, sorted by rating

---

## How It Works

1. The dataset is loaded from `comics_dataset_full.csv`
2. Each entry's tags, genre and audience are combined into a single feature string
3. A **TF-IDF vectorizer** transforms all feature strings into numerical vectors, applying stop words removal and log-normalization (`sublinear_tf=True`)
4. When you search, your query is transformed with the same vectorizer into the same vector space
5. **Cosine similarity** ranks all entries by how close they are to your query
6. The top 5 results are displayed with their similarity score and a visual match bar

---

## Getting Started

### Prerequisites

- Python 3.10
- Anaconda

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stuckonmysofa.git
   cd stuckonmysofa
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n stuckonmysofa python=3.10
   conda activate stuckonmysofa
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

```bash
python stuckonmysofa.py
```

The program starts an interactive menu in your terminal. No additional configuration required.

---

## Project Structure

```
stuckonmysofa/
├── stuckonmysofa.py           # Main script
├── comics_dataset_full.csv    # Dataset (15,797 titles)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── ACKNOWLEDGMENTS.md         # Credits and contributions
```

---

## Authors

Mattia, Jacopo, Alessio — IULM University, A.A. 2025-2026  
Course: Data Mining and Text Analytics  
Professor: Dr. Alessandro Bruno
