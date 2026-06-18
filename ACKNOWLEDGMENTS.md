# Acknowledgments

## StuckOnMySofa — Manga Recommender  
**IULM University — Data Mining and Text Analytics — A.A. 2025-2026**  
**Prof. Dr. Alessandro Bruno**

---

## Team contributions

### Mattia
Project lead and dataset curation. Responsible for the overall architecture of the recommendation system, selection and cleaning of the MyAnimeList dataset (filtering explicit content, validating ratings and metadata), and integration of the web interface. Also handled GitHub repository management and final documentation.

### Jacopo
Core algorithm implementation. Responsible for the TF-IDF vectorization pipeline, cosine similarity logic, and the combined scoring system (similarity + rating boost). Wrote and tested both `recommend_by_title()` and `recommend_by_description()`, and contributed to the explanation of the mathematical concepts in the project documentation.

### Alessio
Interface development and demo preparation. Responsible for the terminal interface (`stuckonmysofa.py`) display functions, the Flask web application (`app.py`) including genre/theme browsing, and the live demo script for the exam presentation.

---

## Data source

**MyAnimeList (MAL)** — https://myanimelist.net  
Dataset of the top 600 manga by MAL score, retrieved via the Kaggle public dataset.  
Ratings and metadata are property of MyAnimeList and its user community.

---

## Libraries

| Library | Purpose |
|---------|---------|
| [pandas](https://pandas.pydata.org/) | Dataset loading and manipulation |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorization and cosine similarity |
| [Flask](https://flask.palletsprojects.com/) | Web interface |

---

## License

This project is released under the MIT License. See `LICENSE` for details.
