# Continuity

## State

Study, demo and docs done; published.

## Decisions

- D001 The shared engine is vendored in `pxlab/` so the repo stands alone.
- D002 The recommenders live in this project, not in pxlab: no other project uses them.
- D003 Scores are sums of neighbours' centred ratings, not averages: averages pushed obscure movies to the top and lost to popularity.
- D004 Neighbour counts are chosen on a validation split carved from the training ratings; the hidden test ratings are only for the final scores.
- D005 MovieLens is downloaded at run time (main.py and the demo), not committed.

## Next

Nothing open.
