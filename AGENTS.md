# movie-recommender

AI Lab coursework: movie recommendations with user- and item-based KNN and user grouping with K-Means on MovieLens ratings, evaluated by precision and recall.

## Layout

- `movie_recommender/` this project's code, `data.py` (download and split), `recommend.py` (the recommenders and relevance scoring), `groups.py` (K-Means taste groups), `report.py` (which outputs make up the results)
- `pxlab/` vendored copy of the shared engine (see `pxlab/VERSION`). Don't edit it here; it is replaced on the next sync.
- `main.py` runs the full study and rewrites `docs/results.md`; `app.py` is the Streamlit demo; `app_overview.md` the demo's Overview text
- `outputs/` figures and CSV tables, `models/` the chosen neighbour counts and each user's taste group (committed for the hosted app), `data/` the download (ignored)
- `docs/` methodology, results, deploy

## Commands

- Full study: `uv run python main.py`
- Demo: `uv run streamlit run app.py`
- Lint: `uv run ruff check . && uv run ruff format --check .`

## Conventions

- Beginner-student style on purpose: plain functions, docstrings on every function, comments explaining the AI concepts.
- Seed 42 everywhere. Settings (neighbours, k) are chosen on a validation split of the training ratings, never on the hidden test ratings.
- After changing the study, re-run `main.py` and refresh the README results and `app_overview.md`.
