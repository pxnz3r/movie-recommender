# Methodology

## Questions

1. Can K-nearest neighbours recommend movies a user will like, and does it beat recommending the most popular movies to everyone?
2. Can users be grouped by taste with clustering, and do the groups make useful recommendations?
3. How relevant are the recommendations, measured on ratings the recommenders never saw?

## Data

MovieLens "latest small": 100,836 ratings from 0.5 to 5 stars, of 9,742 movies by 610 users, each with at least 20 ratings. Each movie has one or more of 19 genres.

**Split.** 20% of every user's ratings are hidden at random (seed 42): 80,672 training ratings, 20,164 hidden. Hiding a share of every user, rather than whole users, means every user has a history to recommend from and some ratings to check against.

**Catalogue.** Only movies with at least 5 training ratings can be recommended (3,222 movies). With fewer ratings, similarities between movies rest on one or two people and are noise.

## Similarity

Ratings form a users x movies table. Each user's average rating is subtracted from their ratings (mean-centring), so a generous rater's 4 and a harsh rater's 3 can both mean "liked it", and a missing rating counts as 0, neutral.

Similarity is the cosine of the angle between two rows (users) or two columns (movies) of the centred table: 1 for the same direction, 0 for unrelated, -1 for opposite tastes. Only each user's or movie's k largest positive similarities are kept: its k nearest neighbours.

## The recommenders

| Method | A user's score for a movie |
| --- | --- |
| User-based KNN | sum over the user's k nearest users of similarity x that user's centred rating of the movie |
| Item-based KNN | sum over the movies the user rated of the user's centred rating x the movie's similarity to it (only among each movie's k nearest movies) |
| Taste group | sum of the centred ratings the movie got from users in the same K-Means group |
| Most popular (baseline) | the number of ratings the movie has, the same for every user |
| Random (baseline) | a random number |

Each user gets the 10 best-scored movies they haven't rated. Scores are sums, not averages: an average ranks a movie one neighbour loved above one that most neighbours liked, and in a first trial that pushed obscure movies to the top and fell below the popularity baseline.

## Taste groups

Each user is described by the share of their training ratings in each genre (a movie with three genres counts towards all three). The shares are standardised and grouped with K-Means.

k is chosen from 3 to 6 by **stability**: K-Means is run from 10 random starts and the groupings are compared with the adjusted Rand index. k = 3 is the most stable (mean 0.67; every other k is below 0.45). Groups are named after the two genres they watch most above the average user.

## Choosing the number of neighbours

The hidden ratings are only for the final scores. To choose k, 20% of each user's training ratings are split off again as a validation set (seed 43), and k = 10, 20, 40, 80 and 160 are scored on it. The best by precision is user-based k = 160 and item-based k = 80.

## Measuring relevance

A hidden movie is **relevant** if the user rated it 4 stars or more and it is in the catalogue. Users with no relevant hidden movie (13 of 610) are left out of the averages.

| Measure | Meaning |
| --- | --- |
| Precision@10 | share of the 10 recommendations that are relevant |
| Recall@10 | share of the user's relevant movies that were recommended |
| Hit rate | share of users with at least one relevant recommendation |
| Coverage | share of the catalogue recommended to anyone, a measure of variety |

These numbers understate real relevance: a recommended movie the user never rated counts as a miss, though they might have liked it.

## Limits

- One random split; differences of a point or less between methods are within noise.
- Popularity is a strong baseline on MovieLens because many users rate the same well-known movies.
- The demo learns from all ratings, so its recommendations are slightly different from the ones scored here.
