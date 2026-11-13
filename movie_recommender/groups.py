"""
groups.py - group users by genre taste with K-Means.

Each user is described by the share of their ratings in each genre. K-Means
(from pxlab) groups users with similar shares, and each group is named
after the two genres it watches most above the average user.
"""

import pandas as pd

from movie_recommender.data import genre_list
from movie_recommender.recommend import genre_profiles, name_taste_groups
from movie_recommender.report import FOLDER
from pxlab.cluster.analysis import (
    choose_k_by_stability,
    kmeans_pipeline,
    plot_profile_heatmap,
    profiles,
    relabel_by,
    scale,
)
from pxlab.common import save_figure, save_table

K_CHOICES = range(3, 7)
FAVOURITES = 5


def group_favourites(train, movies, matrix, labels, names):
    """
    Each group's favourite movies: the highest summed centred rating from
    its members (liked by many, and liked a lot).

    Inputs:
        train: training ratings.
        movies: the movies table.
        matrix: from recommend.build_matrix().
        labels: the group of each user, in matrix row order.
        names: dictionary group -> name.
    Returns:
        A table with one row per group.
    """
    titles = movies.set_index("movie")["title"]
    rows = []
    for group, name in names.items():
        totals = pd.Series(matrix["centred"][labels == group].sum(axis=0), index=matrix["movies"])
        best = totals.sort_values(ascending=False).index[:FAVOURITES]
        rows.append({"Group": name, "Favourite movies": "; ".join(titles[m] for m in best)})
    return pd.DataFrame(rows)


def taste_groups(train, movies, matrix, save=True):
    """
    Choose k, fit K-Means on the genre profiles, name and describe the
    groups.

    Inputs:
        train: training ratings.
        movies: the movies table.
        matrix: from recommend.build_matrix() (fixes the user order).
        save: True to write the tables and figure.
    Returns:
        (group of each user in matrix row order, dictionary group -> name).
    """
    genres = genre_list(movies)
    taste = genre_profiles(train, movies, matrix["users"], genres)
    k, stable = choose_k_by_stability(scale(taste, genres), K_CHOICES)
    labels = kmeans_pipeline(k).fit_predict(taste[genres])
    labels = relabel_by(labels, -pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy())
    names = name_taste_groups(taste, labels)
    if save:
        save_table(stable, f"{FOLDER}/group_k.csv")
        table = profiles(taste, labels, genres, names).rename(columns={"Customers": "Users"})
        save_table(table.round(3), f"{FOLDER}/taste_groups.csv")
        top_genres = taste.mean().sort_values(ascending=False).index[:12].tolist()
        save_figure(
            plot_profile_heatmap(taste, labels, top_genres, names, "Genre taste of each group (12 most-rated genres)"),
            f"{FOLDER}/taste_groups.png",
        )
        save_table(group_favourites(train, movies, matrix, labels, names), f"{FOLDER}/group_favourites.csv")
        print(f"  taste groups: k = {k} ({', '.join(names.values())})")
    return labels, names
