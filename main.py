"""
main.py - run the whole movie recommendation study with one command:

    python main.py

It:
    1. loads 100,836 MovieLens ratings and hides 20% of every user's
       ratings as the test,
    2. picks the number of neighbours for user- and item-based KNN on a
       validation split carved from the training ratings,
    3. groups the users by genre taste with K-Means,
    4. scores five recommenders on the hidden ratings: did they recommend
       movies the user went on to rate 4 stars or more?
    5. writes docs/results.md.
Figures and tables go to outputs/movies/.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from movie_recommender.data import load_movielens, split_ratings
from movie_recommender.groups import taste_groups
from movie_recommender.recommend import (
    build_matrix,
    evaluate,
    group_scores,
    item_knn_scores,
    popularity_scores,
    top_n,
    user_knn_scores,
)
from movie_recommender.report import FOLDER, results_sections
from pxlab.common import MODELS_FOLDER, RANDOM_STATE, save_figure, save_table, set_plot_style, write_results_doc

NEIGHBOURS = [10, 20, 40, 80, 160]
LIST_LENGTH = 10


def choose_neighbours(train):
    """
    Try each neighbour count on a validation split of the training ratings
    (the test ratings stay untouched) and keep the best by precision.

    Input:
        train: training ratings.
    Returns:
        (best k for user-based, best k for item-based, sweep table).
    """
    fit, validation = split_ratings(train, seed=RANDOM_STATE + 1)
    matrix = build_matrix(fit)
    rows = []
    for method, scorer in [("User-based KNN", user_knn_scores), ("Item-based KNN", item_knn_scores)]:
        for k in NEIGHBOURS:
            scores = evaluate(top_n(matrix, scorer(matrix, k), LIST_LENGTH), validation, matrix["movies"])
            rows.append({"Method": method, "Neighbours": k, **scores})
    sweep = pd.DataFrame(rows).drop(columns="Users scored").round(4)
    precision = f"Precision@{LIST_LENGTH}"
    best = sweep.loc[sweep.groupby("Method")[precision].idxmax()].set_index("Method")["Neighbours"]
    return int(best["User-based KNN"]), int(best["Item-based KNN"]), sweep


def plot_neighbours(sweep, user_k, item_k):
    """
    Precision and coverage against the number of neighbours.

    Inputs:
        sweep: the table from choose_neighbours().
        user_k, item_k: the chosen counts.
    Returns:
        The figure.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for method, chosen in [("User-based KNN", user_k), ("Item-based KNN", item_k)]:
        rows = sweep[sweep["Method"] == method]
        for axis, column in [(left, f"Precision@{LIST_LENGTH}"), (right, "Coverage")]:
            line = axis.plot(rows["Neighbours"], rows[column], marker="o", label=method)[0]
            value = rows.loc[rows["Neighbours"] == chosen, column].iloc[0]
            axis.scatter([chosen], [value], s=140, facecolors="none", edgecolors=line.get_color())
    for axis, title in [(left, "Precision@10 (higher is more relevant)"), (right, "Coverage (share of catalogue)")]:
        axis.set_xscale("log", base=2)
        axis.set_xticks(NEIGHBOURS, [str(k) for k in NEIGHBOURS])
        axis.set_xlabel("Neighbours (k)")
        axis.set_title(title)
        axis.legend()
    figure.suptitle("Choosing k on a validation split (circled: chosen)")
    figure.tight_layout()
    return figure


def plot_methods(table):
    """
    Precision against coverage for each recommender.

    Input:
        table: the method comparison.
    Returns:
        The figure.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    for _index, row in table.iterrows():
        axis.scatter(row["Coverage"], row[f"Precision@{LIST_LENGTH}"], s=90)
        axis.annotate(
            row["Method"], (row["Coverage"], row[f"Precision@{LIST_LENGTH}"]), xytext=(6, 4), textcoords="offset points"
        )
    axis.set_xlabel("Coverage: share of the catalogue recommended to anyone")
    axis.set_ylabel(f"Precision@{LIST_LENGTH}")
    axis.set_title("Relevance against variety")
    return figure


def example_table(user, matrix, recommendations, train, hidden, movies):
    """
    One user's favourite movies and what each method recommends them.

    Inputs:
        user: the user id.
        matrix: from build_matrix().
        recommendations: dictionary method -> (user -> movie list).
        train, hidden: the ratings.
        movies: the movies table.
    Returns:
        A table with one column per list.
    """
    titles = movies.set_index("movie")["title"]
    liked = set(hidden[(hidden["user"] == user) & (hidden["rating"] >= 4)]["movie"])
    favourites = train[train["user"] == user].sort_values("rating", ascending=False)["movie"].head(LIST_LENGTH)
    columns = {"Rated highest (training)": [titles[m] for m in favourites]}
    for method, picks in recommendations.items():
        columns[method] = [titles[m] + (" [liked]" if m in liked else "") for m in picks[user]]
    return pd.DataFrame(columns)


def main():
    """
    The full study.

    Returns:
        Nothing.
    """
    set_plot_style()
    ratings, movies = load_movielens()
    train, hidden = split_ratings(ratings)
    print(f"Ratings: {len(ratings):,} ({len(train):,} training, {len(hidden):,} hidden)")

    user_k, item_k, sweep = choose_neighbours(train)
    save_table(sweep, f"{FOLDER}/neighbours.csv")
    save_figure(plot_neighbours(sweep, user_k, item_k), f"{FOLDER}/neighbours.png")
    print(f"  neighbours: user-based {user_k}, item-based {item_k}")

    matrix = build_matrix(train)
    labels, names = taste_groups(train, movies, matrix)

    generator = np.random.default_rng(RANDOM_STATE)
    scorers = {
        f"User-based KNN (k = {user_k})": lambda: user_knn_scores(matrix, user_k),
        f"Item-based KNN (k = {item_k})": lambda: item_knn_scores(matrix, item_k),
        f"Taste group ({len(names)} K-Means groups)": lambda: group_scores(matrix, labels),
        "Most popular (baseline)": lambda: popularity_scores(matrix),
        "Random (baseline)": lambda: generator.random(matrix["centred"].shape),
    }
    recommendations, rows = {}, []
    for method, scorer in scorers.items():
        recommendations[method] = top_n(matrix, scorer(), LIST_LENGTH)
        rows.append({"Method": method, **evaluate(recommendations[method], hidden, matrix["movies"])})
        print(f"  {method}: precision@10 {rows[-1][f'Precision@{LIST_LENGTH}']:.3f}")
    table = pd.DataFrame(rows).round(4)
    save_table(table, f"{FOLDER}/methods.csv")
    save_figure(plot_methods(table), f"{FOLDER}/methods.png")

    # What the demo needs: the chosen settings and each user's taste group.
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    with open(os.path.join(MODELS_FOLDER, "settings.json"), "w", encoding="utf-8") as file:
        json.dump({"user_neighbours": user_k, "item_neighbours": item_k}, file, indent=2)
    groups = pd.DataFrame({"user": matrix["users"], "group": [names[label] for label in labels]})
    groups.to_csv(os.path.join(MODELS_FOLDER, "user_groups.csv"), index=False)

    # An example user: the one with the median number of ratings.
    counts = train["user"].value_counts().sort_values()
    user = int(counts.index[len(counts) // 2])
    shown = {method.split(" (")[0]: picks for method, picks in recommendations.items() if "Random" not in method}
    save_table(example_table(user, matrix, shown, train, hidden, movies), f"{FOLDER}/example.csv")

    write_results_doc(
        "Results",
        f"Produced by `python main.py`. 20% of each user's ratings are hidden (seed 42); a recommendation is "
        f"relevant if the user rated it 4 stars or more. Only movies with at least 5 training ratings "
        f"({len(matrix['movies']):,}) can be recommended. The example user is user {user}.",
        results_sections(),
    )
    print("  Done. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
