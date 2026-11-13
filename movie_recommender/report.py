"""
report.py - which saved tables and figures make up the results. The app's
Results tab and docs/results.md both read this list, so they always match.
"""

FOLDER = "movies"


def results_sections():
    """
    The results, grouped into sections.

    Returns:
        A list of (section name, items) for pxlab's show_results_tabs() and
        write_results_doc().
    """
    return [
        (
            "Relevance",
            [
                ("table", "Recommenders on the hidden ratings (10 recommendations per user)", f"{FOLDER}/methods.csv"),
                ("figure", f"{FOLDER}/methods.png"),
            ],
        ),
        (
            "Deeper checks",
            [
                ("figure", f"{FOLDER}/intervals.png"),
                ("table", "Scores with 95% bootstrap intervals over users", f"{FOLDER}/intervals.csv"),
                ("table", "Popularity bias: how mainstream each method's picks are", f"{FOLDER}/popularity.csv"),
                ("figure", f"{FOLDER}/activity.png"),
                ("table", "Precision@10 by how many ratings a user gave", f"{FOLDER}/activity.csv"),
            ],
        ),
        (
            "Neighbours",
            [
                ("figure", f"{FOLDER}/neighbours.png"),
                ("table", "Choosing the number of neighbours (validation split)", f"{FOLDER}/neighbours.csv"),
            ],
        ),
        (
            "Taste groups",
            [
                ("figure", f"{FOLDER}/taste_groups.png"),
                ("table", "Taste groups", f"{FOLDER}/taste_groups.csv"),
                ("table", "Each group's favourite movies", f"{FOLDER}/group_favourites.csv"),
                ("table", "Choosing the number of groups", f"{FOLDER}/group_k.csv"),
            ],
        ),
        ("Example", [("table", "One user's recommendations", f"{FOLDER}/example.csv")]),
    ]
