

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
MOVIES = pd.DataFrame(
    [
        (1, "The Matrix", "action scifi", "hacker simulation dystopia ai"),
        (2, "Inception", "scifi thriller", "dream heist mind"),
        (3, "Interstellar", "scifi drama", "space time blackhole family"),
        (4, "The Dark Knight", "action crime thriller", "superhero joker gotham"),
        (5, "Avengers: Endgame", "action scifi adventure", "superhero team time"),
        (6, "Toy Story", "animation comedy family", "toys friendship adventure"),
        (7, "Finding Nemo", "animation family adventure", "ocean fish father"),
        (8, "The Lion King", "animation drama family", "africa lion kingdom"),
        (9, "The Notebook", "romance drama", "love letters memory"),
        (10, "Titanic", "romance drama", "ship love disaster"),
        (11, "La La Land", "romance musical drama", "music dreams love"),
        (12, "The Godfather", "crime drama", "mafia family power"),
        (13, "Pulp Fiction", "crime thriller", "hitmen violence nonlinear"),
        (14, "Superbad", "comedy", "teenagers friendship party"),
        (15, "The Hangover", "comedy", "friends vegas party"),
        (16, "Get Out", "horror thriller", "racism suspense twist"),
        (17, "A Quiet Place", "horror thriller scifi", "silence family monsters"),
        (18, "3 Idiots", "comedy drama", "engineering college friendship"),
        (19, "Dangal", "drama sport biography", "wrestling father daughters"),
        (20, "Ex Machina", "scifi thriller drama", "ai robot turing test"),
    ],
    columns=["movie_id", "title", "genres", "keywords"],
)

# Synthetic ratings: each user has genre tastes (-1 dislike ... +1 love).
# A rating = 3 + taste of the movie's genres + small random noise, clipped to 1-5.
# Each user rates ~60% of the movies (the rest are unseen = candidates to recommend).
TASTES = {
    "Aarav":  {"scifi": 1.0, "action": 0.6, "thriller": 0.4, "romance": -0.8, "comedy": -0.3},
    "Diya":   {"romance": 1.0, "drama": 0.6, "musical": 0.8, "horror": -0.9, "action": -0.4},
    "Rohan":  {"crime": 1.0, "thriller": 0.7, "action": 0.4, "animation": -0.6, "romance": -0.3},
    "Meera":  {"animation": 1.0, "family": 0.8, "drama": 0.3, "horror": -0.8, "crime": -0.4},
    "Kabir":  {"comedy": 1.0, "animation": 0.4, "family": 0.3, "drama": -0.3, "crime": -0.5},
    "Sneha":  {"horror": 1.0, "thriller": 0.8, "scifi": 0.5, "romance": -0.7, "family": -0.5},
    "Vikram": {"scifi": 0.9, "action": 0.9, "adventure": 0.5, "romance": -0.5, "musical": -0.6},
    "Ananya": {"romance": 0.9, "drama": 0.8, "biography": 0.8, "sport": 0.5, "horror": -0.6},
    "Ishaan": {"comedy": 0.8, "action": 0.5, "adventure": 0.6, "drama": -0.2, "romance": -0.4},
    "Tara":   {"drama": 1.0, "crime": 0.5, "biography": 0.6, "comedy": 0.2, "horror": -0.5},
}


def make_ratings(seed=7, coverage=0.6):
    rng = np.random.default_rng(seed)
    rows = {}
    for user, taste in TASTES.items():
        row = {}
        for _, m in MOVIES.iterrows():
            if rng.random() > coverage:
                continue                      # user has not seen this movie
            score = 3 + sum(taste.get(g, 0) for g in m["genres"].split())
            score += rng.normal(0, 0.4)
            row[m["movie_id"]] = int(np.clip(round(score), 1, 5))
        rows[user] = row
    return pd.DataFrame(rows).T.reindex(columns=MOVIES["movie_id"])


ratings = make_ratings()      # rows = users, columns = movie ids, NaN = unrated


def title_of(movie_id):
    return MOVIES.loc[MOVIES["movie_id"] == movie_id, "title"].iat[0]


# ---------------------------------------------------------------------------
# 2. CONTENT-BASED FILTERING
# ---------------------------------------------------------------------------
def build_content_similarity(movies):
    """TF-IDF over genres + keywords, then movie-to-movie cosine similarity."""
    text = (movies["genres"] + " " + movies["genres"] + " " + movies["keywords"])
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(text)          # genres counted twice = more weight
    return cosine_similarity(matrix)


CONTENT_SIM = build_content_similarity(MOVIES)


def similar_movies(title, n=5):
    """Item-to-item: 'Because you watched X'."""
    idx = MOVIES.index[MOVIES["title"].str.lower() == title.lower()]
    if len(idx) == 0:
        raise ValueError(f"Movie '{title}' not found.")
    scores = pd.Series(CONTENT_SIM[idx[0]], index=MOVIES["title"])
    return scores.drop(MOVIES.loc[idx[0], "title"]).nlargest(n)


def content_scores(user, ratings_df=ratings):
    """Score every movie by similarity to the user's liked movies."""
    row = ratings_df.loc[user]
    rated = row.dropna()
    if rated.empty:
        return pd.Series(0.0, index=ratings_df.columns)
    weights = rated - rated.mean()              # above-average ratings push up,
    idx = [MOVIES.index[MOVIES["movie_id"] == m][0] for m in rated.index]
    profile = (CONTENT_SIM[idx] * weights.values[:, None]).sum(axis=0)
    return pd.Series(profile, index=ratings_df.columns)


# ---------------------------------------------------------------------------
# 3. COLLABORATIVE FILTERING (user-based)
# ---------------------------------------------------------------------------
def user_similarity(ratings_df):
    """
    Pearson-style similarity computed only on movies both users rated,
    shrunk towards 0 when the overlap is small (few shared movies = weak evidence).
    """
    users = ratings_df.index
    means = ratings_df.mean(axis=1)
    centred = ratings_df.sub(means, axis=0)
    sim = pd.DataFrame(0.0, index=users, columns=users)
    for a in users:
        for b in users:
            if a == b:
                continue
            common = ratings_df.loc[a].notna() & ratings_df.loc[b].notna()
            n = int(common.sum())
            if n < 2:
                continue
            x, y = centred.loc[a, common], centred.loc[b, common]
            denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
            if denom:
                sim.loc[a, b] = (x * y).sum() / denom * (n / (n + 2))
    return sim


def _predict(user, movie, ratings_df, sim, k=4):
    """Predicted rating of `movie` for `user` from the k most similar users."""
    user_means = ratings_df.mean(axis=1)
    neighbours = sim[user].nlargest(k)
    neighbours = neighbours[neighbours > 0]
    num = den = 0.0
    for other, s_ in neighbours.items():
        r = ratings_df.loc[other, movie]
        if not np.isnan(r):
            num += s_ * (r - user_means[other])
            den += abs(s_)
    return user_means[user] + num / den if den else np.nan


def collaborative_scores(user, ratings_df=ratings, k=4):
    """Predicted rating for every movie, for `user`."""
    sim = user_similarity(ratings_df)
    return pd.Series({m: _predict(user, m, ratings_df, sim, k)
                      for m in ratings_df.columns})


# ---------------------------------------------------------------------------
# 4. HYBRID + RECOMMEND
# ---------------------------------------------------------------------------
def _minmax(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def recommend(user, n=5, method="hybrid", alpha=0.5, ratings_df=ratings):
    """
    method: 'content', 'collaborative' or 'hybrid'
    alpha : weight of collaborative score in the hybrid blend
    """
    seen = ratings_df.loc[user].dropna().index
    cb = _minmax(content_scores(user, ratings_df))
    cf = collaborative_scores(user, ratings_df)
    cf = _minmax(cf.fillna(cf.mean()))

    if method == "content":
        final = cb
    elif method == "collaborative":
        final = cf
    else:
        final = alpha * cf + (1 - alpha) * cb

    final = final.drop(seen).nlargest(n)
    return pd.DataFrame({
        "movie": [title_of(m) for m in final.index],
        "score": final.round(3).values,
    })


# ---------------------------------------------------------------------------
# 5. EVALUATION (hold out one rating per user, predict it, compute RMSE)
# ---------------------------------------------------------------------------
def evaluate(runs=20, seed=42):
    """
    Repeat `runs` times: hide one random rating per user, predict it, and
    compare against a naive baseline (average rating of that movie).
    Returns (CF RMSE, baseline RMSE) averaged over all runs.
    """
    rng = np.random.default_rng(seed)
    cf_err, base_err = [], []
    for _ in range(runs):
        train = ratings.copy()
        held = []
        for user in train.index:
            movie = rng.choice(train.loc[user].dropna().index)
            held.append((user, movie, train.loc[user, movie]))
            train.loc[user, movie] = np.nan

        item_means = train.mean()
        sim = user_similarity(train)          # compute once per run
        for user, movie, actual in held:
            pred = _predict(user, movie, train, sim)
            base = item_means[movie]
            if np.isnan(base):
                base = 3.0
            if np.isnan(pred):
                pred = base
            cf_err.append((np.clip(pred, 1, 5) - actual) ** 2)
            base_err.append((base - actual) ** 2)

    return np.sqrt(np.mean(cf_err)), np.sqrt(np.mean(base_err))


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    user = "Aarav"
    liked = ratings.loc[user].dropna().sort_values(ascending=False).head(3)
    print(f"\nTop-rated by {user}: " +
          ", ".join(f"{title_of(m)} ({int(r)})" for m, r in liked.items()))

    for method in ("content", "collaborative", "hybrid"):
        print(f"\n--- {method.upper()} recommendations for {user} ---")
        print(recommend(user, n=3, method=method).to_string(index=False))

    print("\n--- Because you watched 'Inception' ---")
    print(similar_movies("Inception", n=3).round(3).to_string())

    cf_rmse, base_rmse = evaluate()
    print("\n--- Evaluation (hold-out RMSE, lower is better) ---")
    print(f"Collaborative filtering : {cf_rmse:.3f}")
    print(f"Item-average baseline   : {base_rmse:.3f}")

    # Interactive mode
    print("\n" + "-" * 60)
    print("Try it yourself. Users:", ", ".join(ratings.index))
    while True:
        name = input("\nEnter a user name (or 'q' to quit): ").strip().title()
        if name.lower() == "q":
            break
        if name not in ratings.index:
            print("Unknown user, try again.")
            continue
        print(recommend(name, n=5).to_string(index=False))


if __name__ == "__main__":
    main()
