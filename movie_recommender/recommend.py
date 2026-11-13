"""
recommend.py - four ways to recommend movies, and how to score them.

    Popularity (baseline): the most-rated movies, the same for everyone.
    User-based KNN: find the k users whose ratings are most like yours,
        and recommend what they liked.
    Item-based KNN: for each movie, find the k movies rated most alike;
        recommend movies similar to the ones you liked.
    Group-based: recommend what your K-Means taste group liked.

"Alike" is cosine similarity between mean-centred ratings: each user's
average rating is subtracted first, so a generous rater's 4 and a harsh
rater's 3 can both mean "liked it". Missing ratings count as 0 (neutral).

Scores are worked out for every user and movie at once with matrix
products, which is the same sum as a loop over neighbours, only faster.
"""

import numpy as np
import pandas as pd

from movie_recommender.data import GOOD_RATING, MIN_RATINGS


def build_matrix(train):
    """
    The users x movies rating table, keeping movies with at least
    MIN_RATINGS training ratings (too few ratings give unreliable
    similarities).

    Input:
        train: training ratings (user, movie, rating).
    Returns:
        Dictionary with "users" and "movies" (the ids in row / column
        order), "rated" (1 where rated, else 0), "centred" (rating minus
        the user's average, 0 where unrated) and "means" (each user's
        average rating).
    """
    counts = train["movie"].value_counts()
    kept = train[train["movie"].isin(counts[counts >= MIN_RATINGS].index)]
    table = kept.pivot_table(index="user", columns="movie", values="rating")
    ratings = table.to_numpy()
    rated = ~np.isnan(ratings)
    means = np.nanmean(ratings, axis=1)
    centred = np.where(rated, ratings - means[:, None], 0.0)
    return {
        "users": table.index.to_numpy(),
        "movies": table.columns.to_numpy(),
        "rated": rated.astype(float),
        "centred": centred,
        "means": means,
    }


def cosine_similarity(rows):
    """
    Cosine similarity between every pair of rows: the cosine of the angle
    between them, 1 for the same direction, 0 for unrelated, -1 opposite.

    Input:
        rows: a 2-D array.
    Returns:
        A square array of similarities, with 0 on the diagonal (an item is
        never its own neighbour).
    """
    lengths = np.linalg.norm(rows, axis=1)
    lengths[lengths == 0] = 1
    unit = rows / lengths[:, None]
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, 0)
    return similarity


def keep_top_k(similarity, k):
    """
    Keep only each row's k largest positive similarities (its nearest
    neighbours) and set the rest to 0.

    Inputs:
        similarity: a square similarity array.
        k: neighbours to keep.
    Returns:
        A new array with at most k non-zero values per row.
    """
    kept = np.zeros_like(similarity)
    top = np.argpartition(-similarity, k, axis=1)[:, :k]
    rows = np.arange(len(similarity))[:, None]
    kept[rows, top] = similarity[rows, top]
    return np.clip(kept, 0, None)


def user_knn_scores(matrix, k):
    """
    User-based KNN. A user's score for a movie is the similarity-weighted
    sum of their k nearest neighbours' centred ratings of it. A sum, not an
    average: a movie one neighbour loved shouldn't beat one that most
    neighbours liked, and an average would rank them the other way.

    Inputs:
        matrix: from build_matrix().
        k: neighbours per user.
    Returns:
        users x movies array of scores.
    """
    neighbours = keep_top_k(cosine_similarity(matrix["centred"]), k)
    return neighbours @ matrix["centred"]


def item_knn_scores(matrix, k):
    """
    Item-based KNN. Each movie keeps its k most similar movies. A user's
    score for a movie is the similarity-weighted sum of their own centred
    ratings of the movies it is similar to (a sum, for the same reason as
    in user_knn_scores).

    Inputs:
        matrix: from build_matrix().
        k: similar movies kept per movie.
    Returns:
        users x movies array of scores.
    """
    return matrix["centred"] @ item_neighbours(matrix, k).T


def item_neighbours(matrix, k):
    """
    Each movie's k most similar movies (the item-based KNN "model").

    Inputs:
        matrix: from build_matrix().
        k: similar movies kept per movie.
    Returns:
        movies x movies array: row i holds movie i's neighbours'
        similarities, 0 elsewhere.
    """
    return keep_top_k(cosine_similarity(matrix["centred"].T), k)


def recommend_for_liked(matrix, neighbours, liked, n=10):
    """
    Item-based recommendations for someone who isn't in the data, from a
    few movies they say they like. Each liked movie counts as rated above
    their average (+1), so a movie's score is the sum of its similarities
    to the liked movies.

    Inputs:
        matrix: from build_matrix().
        neighbours: from item_neighbours().
        liked: movie ids.
        n: how many to recommend.
    Returns:
        List of (movie id, score), best first.
    """
    columns = np.flatnonzero(np.isin(matrix["movies"], liked))
    scores = neighbours[:, columns].sum(axis=1)
    scores[columns] = -np.inf
    best = np.argsort(-scores)[:n]
    return [(matrix["movies"][column], scores[column]) for column in best if scores[column] > 0]


def popularity_scores(matrix):
    """
    The popularity baseline: every user gets the movies with the most
    ratings.

    Input:
        matrix: from build_matrix().
    Returns:
        users x movies array of scores.
    """
    counts = matrix["rated"].sum(axis=0)
    return np.tile(counts, (len(matrix["users"]), 1))


def group_scores(matrix, labels):
    """
    Group-based: a user's score for a movie is the sum of the centred
    ratings it got from users in the same taste group: what the group
    liked most, and most often.

    Inputs:
        matrix: from build_matrix().
        labels: the group of each user, in matrix row order.
    Returns:
        users x movies array of scores.
    """
    scores = np.full(matrix["centred"].shape, -np.inf)
    for group in np.unique(labels):
        members = labels == group
        scores[members] = matrix["centred"][members].sum(axis=0)
    return scores


def top_n(matrix, scores, n=10):
    """
    The n best-scored movies each user hasn't rated yet.

    Inputs:
        matrix: from build_matrix().
        scores: users x movies scores.
        n: how many to recommend.
    Returns:
        Dictionary user id -> list of movie ids, best first.
    """
    masked = np.where(matrix["rated"] > 0, -np.inf, scores)
    best = np.argsort(-masked, axis=1)[:, :n]
    result = {}
    for row, user in enumerate(matrix["users"]):
        picks = [column for column in best[row] if np.isfinite(masked[row, column])]
        result[user] = list(matrix["movies"][picks])
    return result


def evaluate(recommendations, hidden, catalogue, n=10):
    """
    Score recommendations against the hidden ratings.

    Relevant movies are hidden ones the user rated GOOD_RATING or more
    (and that are in the catalogue, so they could be recommended at all).
    Only users with at least one relevant hidden movie are scored.

    Inputs:
        recommendations: dictionary user -> list of movie ids.
        hidden: the hidden ratings.
        catalogue: the movie ids that can be recommended.
        n: the list length used.
    Returns:
        Dictionary: precision@n (share of the n recommended that were
        relevant), recall@n (share of the relevant that were recommended),
        hit rate (share of users with at least one relevant
        recommendation), coverage (share of the catalogue recommended to
        anyone), users scored.
    """
    relevant = hidden[(hidden["rating"] >= GOOD_RATING) & hidden["movie"].isin(catalogue)]
    relevant = relevant.groupby("user")["movie"].apply(set)
    precision, recall, hits = [], [], []
    for user, movies in relevant.items():
        picked = recommendations.get(user, [])[:n]
        found = len(movies.intersection(picked))
        precision.append(found / n)
        recall.append(found / len(movies))
        hits.append(found > 0)
    shown = {movie for picks in recommendations.values() for movie in picks[:n]}
    return {
        f"Precision@{n}": float(np.mean(precision)),
        f"Recall@{n}": float(np.mean(recall)),
        "Hit rate": float(np.mean(hits)),
        "Coverage": len(shown) / len(catalogue),
        "Users scored": len(relevant),
    }


def genre_profiles(train, movies, users, genres):
    """
    Each user's taste: the share of their ratings that fall in each genre
    (a movie with three genres counts towards all three).

    Inputs:
        train: training ratings.
        movies: the movies table.
        users: user ids, in the order wanted.
        genres: the genre names.
    Returns:
        DataFrame, one row per user, one column per genre.
    """
    exploded = train.merge(movies[["movie", "genres"]], on="movie").explode("genres")
    exploded = exploded[exploded["genres"].isin(genres)]
    counts = pd.crosstab(exploded["user"], exploded["genres"]).reindex(index=users, columns=genres, fill_value=0)
    per_user = train.groupby("user").size().reindex(users)
    return counts.div(per_user, axis=0)


def name_taste_groups(profiles, labels):
    """
    Name each taste group after the two genres it watches most above the
    average user.

    Inputs:
        profiles: the genre profiles.
        labels: the group of each user.
    Returns:
        Dictionary group number -> name such as "Drama & Romance fans".
    """
    lift = profiles.groupby(labels).mean() - profiles.mean()
    names = {}
    for group, row in lift.iterrows():
        first, second = row.sort_values(ascending=False).index[:2]
        names[int(group)] = f"{first} & {second} fans"
    return names
