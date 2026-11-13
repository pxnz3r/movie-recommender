"""
deeper.py - checks that go beyond the average precision of each recommender.

    Intervals: precision and recall are averages over users, so resampling
        the users (bootstrap) shows how much they could move by chance.
    Popularity bias: does a method only ever suggest the blockbusters?
        Measured by how many training ratings its picks have, and how much
        of the catalogue's long tail it ever shows.
    Activity: users with few ratings are the hard case (cold start). The
        scores split by how many ratings each user gave.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from movie_recommender.data import GOOD_RATING
from pxlab.common import RANDOM_STATE

BOOTSTRAP_ROUNDS = 1000
ACTIVITY_GROUPS = 4


def per_user_scores(recommendations, hidden, catalogue, n=10):
    """
    Precision, recall and hit for every scored user (the same rules as
    evaluate() in recommend.py, kept per user instead of averaged).

    Inputs:
        recommendations: dictionary user -> list of movie ids.
        hidden: the hidden ratings.
        catalogue: the movie ids that can be recommended.
        n: the list length used.
    Returns:
        A DataFrame: user, Precision, Recall, Hit.
    """
    relevant = hidden[(hidden["rating"] >= GOOD_RATING) & hidden["movie"].isin(catalogue)]
    relevant = relevant.groupby("user")["movie"].apply(set)
    rows = []
    for user, movies in relevant.items():
        found = len(movies.intersection(recommendations.get(user, [])[:n]))
        rows.append({"user": user, "Precision": found / n, "Recall": found / len(movies), "Hit": float(found > 0)})
    return pd.DataFrame(rows)


def bootstrap_intervals(scores, rounds=BOOTSTRAP_ROUNDS):
    """
    95% bootstrap intervals for the average precision, recall and hit rate:
    draw the users again with replacement many times and keep the middle
    95% of the averages.

    Inputs:
        scores: dictionary method -> DataFrame from per_user_scores().
        rounds: how many resamples.
    Returns:
        A long DataFrame: Method, Score, Value, Low, High.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    rows = []
    for method, table in scores.items():
        draws = generator.integers(0, len(table), (rounds, len(table)))
        for column in ["Precision", "Recall", "Hit"]:
            values = table[column].to_numpy()
            means = values[draws].mean(axis=1)
            rows.append(
                {
                    "Method": method,
                    "Score": "Hit rate" if column == "Hit" else f"{column}@10",
                    "Value": round(float(values.mean()), 4),
                    "Low": round(float(np.percentile(means, 2.5)), 4),
                    "High": round(float(np.percentile(means, 97.5)), 4),
                }
            )
    return pd.DataFrame(rows)


def plot_intervals(intervals):
    """
    Precision@10 with its 95% interval for every method.

    Input:
        intervals: from bootstrap_intervals().
    Returns:
        The figure.
    """
    part = intervals[intervals["Score"] == "Precision@10"].sort_values("Value")
    figure, axis = plt.subplots(figsize=(8, 0.6 * len(part) + 1.5))
    positions = np.arange(len(part))
    errors = [part["Value"] - part["Low"], part["High"] - part["Value"]]
    axis.errorbar(part["Value"], positions, xerr=errors, fmt="o", capsize=4)
    axis.set_yticks(positions, part["Method"])
    axis.set_xlabel("Precision@10 (95% bootstrap interval over users)")
    axis.set_title("How sure are the precision numbers?")
    figure.tight_layout()
    return figure


def popularity_table(recommendations, train, catalogue, n=10):
    """
    How popular each method's picks are.

    Inputs:
        recommendations: dictionary method -> (user -> list of movie ids).
        train: the training ratings.
        catalogue: the movie ids that can be recommended.
        n: the list length used.
    Returns:
        A DataFrame: Method, Median ratings per pick, Picks from the top
        10% most rated, Long-tail movies shown (share of the bottom 50% of
        the catalogue ever recommended).
    """
    counts = train["movie"].value_counts().reindex(catalogue, fill_value=0)
    top = set(counts[counts >= counts.quantile(0.9)].index)
    tail = set(counts[counts <= counts.median()].index)
    rows = []
    for method, picks in recommendations.items():
        shown = [movie for movies in picks.values() for movie in movies[:n]]
        rows.append(
            {
                "Method": method,
                "Median ratings per pick": int(np.median(counts.loc[shown])),
                "Picks from the top 10% most rated": round(np.mean([movie in top for movie in shown]), 4),
                "Long-tail movies shown": round(len(tail.intersection(shown)) / len(tail), 4),
            }
        )
    return pd.DataFrame(rows)


def activity_table(scores, train):
    """
    Precision@10 split into user groups by how many ratings they gave.

    Inputs:
        scores: dictionary method -> DataFrame from per_user_scores().
        train: the training ratings.
    Returns:
        A wide DataFrame: one row per activity group (with its user count),
        one column per method.
    """
    activity = train["user"].value_counts()
    groups = pd.qcut(activity, ACTIVITY_GROUPS)
    names = {band: f"{int(band.left) + 1} to {int(band.right)} ratings" for band in groups.cat.categories}
    rows = []
    for band in groups.cat.categories:
        users = set(groups[groups == band].index)
        row = {"Users": names[band]}
        for method, table in scores.items():
            inside = table[table["user"].isin(users)]
            row["Scored users"] = len(inside)
            row[method] = round(float(inside["Precision"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_activity(table):
    """
    Precision@10 per activity group, one line per method.

    Input:
        table: from activity_table().
    Returns:
        The figure.
    """
    figure, axis = plt.subplots(figsize=(8, 4.5))
    methods = [column for column in table.columns if column not in ("Users", "Scored users")]
    for method in methods:
        axis.plot(table["Users"], table[method], marker="o", label=method)
    axis.set_xlabel("Training ratings per user")
    axis.set_ylabel("Precision@10")
    axis.set_title("Light users are harder to recommend for")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure
