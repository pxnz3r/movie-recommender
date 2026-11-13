"""
data.py - the MovieLens ratings and the train/test split.

MovieLens "latest small": 100,836 ratings (0.5 to 5 stars) of 9,742
movies by 610 users of movielens.org. main.py downloads it the first time.
"""

import os
import zipfile

import numpy as np
import pandas as pd

from pxlab.common import DATA_FOLDER, RANDOM_STATE, download_once

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = os.path.join(DATA_FOLDER, "ml-latest-small.zip")
GOOD_RATING = 4.0  # a held-out movie rated this or higher counts as relevant
MIN_RATINGS = 5  # movies with fewer training ratings are never recommended
TEST_SHARE = 0.2


def load_movielens():
    """
    Download (once) and read the ratings and movies.

    Returns:
        (ratings DataFrame with user, movie, rating; movies DataFrame with
        movie, title, genres as a list).
    """
    download_once(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        ratings = pd.read_csv(archive.open("ml-latest-small/ratings.csv"))
        movies = pd.read_csv(archive.open("ml-latest-small/movies.csv"))
    ratings = ratings.rename(columns={"userId": "user", "movieId": "movie"})[["user", "movie", "rating"]]
    movies = movies.rename(columns={"movieId": "movie"})
    movies["genres"] = movies["genres"].str.split("|")
    return ratings, movies


def split_ratings(ratings, share=TEST_SHARE, seed=RANDOM_STATE):
    """
    Hide a random share of every user's ratings. The recommenders learn
    from the rest and are scored on whether they recommend the hidden
    movies the user rated highly.

    Inputs:
        ratings: the ratings table.
        share: the share of each user's ratings to hide.
        seed: random seed.
    Returns:
        (training ratings, hidden ratings).
    """
    generator = np.random.default_rng(seed)
    hidden = ratings.groupby("user", group_keys=False)[["user", "movie", "rating"]].apply(
        lambda rows: rows.sample(frac=share, random_state=int(generator.integers(1_000_000)))
    )
    return ratings.drop(hidden.index), hidden


def genre_list(movies):
    """
    Every genre name, without the "(no genres listed)" placeholder.

    Input:
        movies: the movies table.
    Returns:
        A sorted list of genres.
    """
    genres = {genre for row in movies["genres"] for genre in row}
    genres.discard("(no genres listed)")
    return sorted(genres)
