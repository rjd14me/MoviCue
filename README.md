# MovieCue (v1.6)

## Features
- Uses data from the MovieLens set  (`movies.csv`, `links.csv` and `ratings.csv`)
- Learns movie embeddings from movie metadata.
- Displays a simple web UI where you type a movie title and get similar movies.
- Includes a live search dropdown that updates as you type, allowing users to select specific movies quickly.
- Fuzzy Search Capabilities
- Random Movie searching
- Filters to filter by year and ratings

## Key skills demonstrated:
- Data Handling
- Python and project structure
- FastAPI
- Simple deep learning (PyTorch)
- Frontend (HTML/CSS)

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