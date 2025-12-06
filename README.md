# MoviCue (v1.7)
A MovieLens-driven movie recommender that learns genre-aware embeddings and serves similarity-based recommendations via FastAPI.

# How to Demo
### Prerequisites
```bash
- Python 3.12
- pip
```
### Setting Up
```bash
git clone https://github.com/rjd14me/MoviCue.git  
cd MoviCue
```
```bash
pip install -r requirements.txt
```
### Getting Started
```bash
python src/api.py
```
Then go to [this web address](http://127.0.0.1:8000)
### Try it Out
- Type any movie title in the search box,and pick one from the drop-down suggestions.
- See the list of recommended movies based on similarity scores.  

## Features
- Uses MovieLens (movies.csv, links.csv, ratings.csv) to bootstrap metadata and ratings.
- Title normalization/tokenization, franchise keying, and search-key construction for better fuzzy lookup.
- User–genre deep model (PyTorch) trained on filtered ratings with validation MSE reported.
- Genre embedding extraction for cosine-similar recommendations with configurable blending of signals (genre/title/franchise/year/rating).
- FastAPI endpoints:
  -/api/search fuzzy search with character-ratio scoring
  - /api/random random movie picker
  - /api/recommend similarity recommendations with filters and explanation payload
  - / Jinja2-rendered UI with dark/light theme toggle
- Frontend: live search, randomizer, filters modal, result badges, and “why” rationale per recommendation.

## Key skills demonstrated:
- Data engineering: MovieLens ingestion, genre encoding, user/rating filtering
- ML: PyTorch embedding model, cosine similarity, metric reporting
- API: FastAPI with typed query params, health endpoint, static assets
- Frontend: Vanilla JS UX (debounce, modals, badges), responsive CSS, theming

```bash
Modules: pandas, numpy, fastapi, uvicorn[standard], jinja2, torch, scikit-learn
```

## Project layout

```text
movie-recommender/
  data/
    movies.csv
    ratings.csv
    links.csv            
  src/
    api.py
    data_preparation.py
    models/
      deep_model.py
  templates/
    index.html
  static/
    style.css
  requirements.txt
  README.md
  
```