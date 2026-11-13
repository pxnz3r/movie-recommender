### What this is

Movie recommendations from 100,836 ratings (0.5 to 5 stars) that 610 people gave on MovieLens. Two ways of using **K-nearest neighbours**, plus **K-Means** to group users by taste:

- **User-based KNN**: find the users whose ratings look most like yours and recommend what they liked.
- **Item-based KNN**: for each movie, find the movies rated most alike; recommend movies similar to the ones you liked.
- **Taste groups**: K-Means groups users by the share of their ratings in each genre; recommend what your group liked.

"Alike" is cosine similarity between ratings with each user's average subtracted, so a generous rater's 4 and a harsh rater's 3 can both mean "liked it".

### What to try

- **Recommend**: pick a MovieLens user to see what they rated highest next to both KNN lists, or pick movies you like and get item-based recommendations.
- **Taste groups**: the three groups and their favourite movies.
- **Results**: how each method scored and how the number of neighbours was chosen.

### What was found

20% of each user's ratings were hidden. A recommendation counts as relevant if the user rated it 4 stars or more.

| Method | Relevant out of 10 | Users with at least one hit | Catalogue covered |
| --- | --- | --- | --- |
| User-based KNN (160 neighbours) | 1.61 | 68% | 4% |
| Item-based KNN (80 neighbours) | 1.28 | 59% | 42% |
| Taste group (K-Means, 3 groups) | 1.24 | 58% | 3% |
| Most popular (baseline) | 1.24 | 57% | 2% |
| Random (baseline) | 0.04 | 4% | 85% |

- **User-based KNN is the most relevant**, 30% better than recommending the most popular movies to everyone.
- **Item-based KNN trades a little relevance for variety.** It recommends 42% of the catalogue at least once; user-based KNN and popularity keep recommending the same few hundred well-known movies.
- **Taste groups alone do no better than popularity.** Three groups are too coarse: each group's favourites are mostly the same classics. The groups are still a useful description of the users: Comedy & Romance fans (41%), Thriller & Crime fans (38%), Adventure & Action fans (21%).

Code and every figure: [github.com/pxnz3r/movie-recommender](https://github.com/pxnz3r/movie-recommender).
