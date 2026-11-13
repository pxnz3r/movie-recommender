"""
app.py - the movie recommender demo.

    streamlit run app.py

Tabs:
    Overview        what this is and what was found (app_overview.md).
    Recommend       recommendations for a MovieLens user, or for movies
                    you pick.
    Taste groups    the K-Means groups of users.
    Results         the saved tables and figures from main.py.

The demo learns from all 100,836 ratings (main.py hides 20% to score the
methods), using the neighbour counts main.py chose.
"""

import json
import os

import pandas as pd
import streamlit as st

from movie_recommender.data import load_movielens
from movie_recommender.recommend import (
    build_matrix,
    item_neighbours,
    recommend_for_liked,
    top_n,
    user_knn_scores,
)
from movie_recommender.report import FOLDER, results_sections
from pxlab.app import setup_page, show_figure, show_markdown_file, show_results_tabs, show_table
from pxlab.common import MODELS_FOLDER

LIST_LENGTH = 10
STARTER_MOVIES = ["Toy Story (1995)", "Finding Nemo (2003)", "Shrek (2001)"]


@st.cache_resource
def load_everything():
    """
    The ratings, the movies, the rating matrix, both KNN models and each
    user's taste group. Built once per server.

    Returns:
        Dictionary of everything the tabs need.
    """
    ratings, movies = load_movielens()
    with open(os.path.join(MODELS_FOLDER, "settings.json"), encoding="utf-8") as file:
        settings = json.load(file)
    matrix = build_matrix(ratings)
    groups = pd.read_csv(os.path.join(MODELS_FOLDER, "user_groups.csv")).set_index("user")["group"]
    return {
        "ratings": ratings,
        "titles": movies.set_index("movie")["title"],
        "matrix": matrix,
        "settings": settings,
        "user_picks": top_n(matrix, user_knn_scores(matrix, settings["user_neighbours"]), LIST_LENGTH),
        "item_model": item_neighbours(matrix, settings["item_neighbours"]),
        "groups": groups,
    }


def numbered(titles):
    """
    A list of titles as a table numbered from 1.

    Input:
        titles: list of strings.
    Returns:
        A DataFrame.
    """
    return pd.DataFrame({"Movie": titles}, index=range(1, len(titles) + 1))


def show_user_mode(data):
    """
    Recommendations for one of the 610 MovieLens users.

    Input:
        data: from load_everything().
    Returns:
        Nothing.
    """
    matrix, titles = data["matrix"], data["titles"]
    users = list(matrix["users"])
    user = st.selectbox("User", users, index=users.index(414) if 414 in users else 0)
    own = data["ratings"][data["ratings"]["user"] == user].sort_values("rating", ascending=False)
    row = users.index(user)
    liked = [movie for movie in own[own["rating"] >= 4]["movie"] if movie in set(matrix["movies"])]
    item_picks = [movie for movie, _score in recommend_for_liked(matrix, data["item_model"], liked, LIST_LENGTH)]
    st.markdown(
        f"User {user} rated **{len(own):,} movies**, on average {matrix['means'][row]:.1f} stars. "
        f"Taste group: **{data['groups'].get(user, 'not grouped')}**."
    )
    first, second, third = st.columns(3)
    with first:
        st.subheader("Rated highest", anchor=False)
        top = own.head(LIST_LENGTH)
        table = numbered([titles[m] for m in top["movie"]])
        table["Stars"] = top["rating"].to_numpy()
        st.dataframe(table, width="stretch")
    with second:
        st.subheader("Users like them liked", anchor=False)
        st.dataframe(numbered([titles[m] for m in data["user_picks"][user]]), width="stretch")
    with third:
        st.subheader("Similar to what they liked", anchor=False)
        st.dataframe(numbered([titles[m] for m in item_picks]), width="stretch")
    st.caption(
        f"Middle: user-based KNN, from the {data['settings']['user_neighbours']} users whose ratings are most like "
        f"theirs. Right: item-based KNN, from the movies most like the ones they rated 4 stars or more."
    )


def show_pick_mode(data):
    """
    Item-based recommendations for movies the visitor picks.

    Input:
        data: from load_everything().
    Returns:
        Nothing.
    """
    matrix, titles = data["matrix"], data["titles"]
    # Offer the catalogue most-rated first, so well-known movies come up first.
    counts = pd.Series(matrix["rated"].sum(axis=0), index=matrix["movies"]).sort_values(ascending=False)
    options = [titles[m] for m in counts.index]
    ids = {titles[m]: m for m in counts.index}
    chosen = st.multiselect("Movies you like", options, default=[t for t in STARTER_MOVIES if t in ids])
    if not chosen:
        st.markdown("Pick a movie or two to get recommendations.")
        return
    picks = recommend_for_liked(matrix, data["item_model"], [ids[title] for title in chosen], LIST_LENGTH)
    st.subheader("You might like", anchor=False)
    st.dataframe(
        pd.DataFrame(
            {"Movie": [titles[m] for m, _score in picks], "Score": [round(score, 2) for _m, score in picks]},
            index=range(1, len(picks) + 1),
        ),
        width="stretch",
    )
    st.caption(
        "Item-based KNN: each movie's score is the sum of its similarity to the movies you picked, counting only "
        f"each movie's {data['settings']['item_neighbours']} nearest neighbours."
    )


def show_groups_page():
    """
    The Taste groups tab, from main.py's outputs.

    Returns:
        Nothing.
    """
    st.markdown(
        "Each user is described by the share of their ratings in each genre, and K-Means groups users with "
        "similar shares. Each group is named after the two genres it watches most above the average user."
    )
    show_figure(f"{FOLDER}/taste_groups.png")
    show_table("Each group's favourite movies", f"{FOLDER}/group_favourites.csv")


def main():
    """
    Build the page.

    Returns:
        Nothing.
    """
    setup_page("Movie recommender")
    overview, recommend, groups, results = st.tabs(["Overview", "Recommend", "Taste groups", "Results"])
    with overview:
        show_markdown_file("app_overview.md")
    with recommend:
        data = load_everything()
        mode = st.radio("Recommend for", ["A MovieLens user", "Movies I pick"], horizontal=True)
        if mode == "A MovieLens user":
            show_user_mode(data)
        else:
            show_pick_mode(data)
    with groups:
        show_groups_page()
    with results:
        show_results_tabs(results_sections())


main()
