"""
recommender.py
Three recommendation strategies (no scikit-surprise required):
  1. Collaborative Filtering  – SVD via scipy / KNN via sklearn
  2. Content-Based Filtering  – TF-IDF genre vectors + cosine similarity
  3. Hybrid                   – CF score weighted by content similarity
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data_loader import load_ratings, load_movies, ALL_GENRES


# ---------------------------------------------------------------------------
# Helper: build user-item rating matrix
# ---------------------------------------------------------------------------

def build_rating_matrix(ratings: pd.DataFrame):
    """
    Returns:
        matrix     : (n_users x n_movies) dense numpy array
        user_ids   : list of UserIDs in row order
        movie_ids  : list of MovieIDs in column order
        uid_to_idx : dict UserID -> row index
        mid_to_idx : dict MovieID -> col index
    """
    user_ids  = sorted(ratings['UserID'].unique())
    movie_ids = sorted(ratings['MovieID'].unique())
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    mid_to_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = ratings['UserID'].map(uid_to_idx).values
    cols = ratings['MovieID'].map(mid_to_idx).values
    data = ratings['Rating'].values.astype(float)

    matrix = csr_matrix((data, (rows, cols)),
                         shape=(len(user_ids), len(movie_ids)))
    return matrix, user_ids, movie_ids, uid_to_idx, mid_to_idx


# ---------------------------------------------------------------------------
# 1a. Collaborative Filtering — SVD (scipy)
# ---------------------------------------------------------------------------

class SVDRecommender:
    """
    Matrix-factorisation CF using truncated SVD.
    Fills missing ratings with each user's mean before decomposition.
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.predicted  = None
        self.user_ids   = None
        self.movie_ids  = None
        self.uid_to_idx = None
        self.mid_to_idx = None
        self.user_means = None

    def fit(self, ratings: pd.DataFrame):
        matrix, self.user_ids, self.movie_ids, self.uid_to_idx, self.mid_to_idx = \
            build_rating_matrix(ratings)

        dense = matrix.toarray().astype(float)

        # Fill missing values with user mean
        self.user_means = np.true_divide(
            dense.sum(axis=1),
            (dense != 0).sum(axis=1),
            where=(dense != 0).sum(axis=1) != 0
        )
        norm = dense.copy()
        for i in range(dense.shape[0]):
            mask = norm[i] == 0
            norm[i, mask] = self.user_means[i]

        k = min(self.n_factors, min(norm.shape) - 1)
        if k < 1:
            self.predicted = norm
            print(f"SVD fitted  (fallback, users={len(self.user_ids)}, movies={len(self.movie_ids)})")
            return self

        U, sigma, Vt = svds(norm, k=k)
        self.predicted = U @ np.diag(sigma) @ Vt
        print(f"SVD fitted  (factors={k}, users={len(self.user_ids)}, movies={len(self.movie_ids)})")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.uid_to_idx or movie_id not in self.mid_to_idx:
            return 3.0  # global fallback
        i = self.uid_to_idx[user_id]
        j = self.mid_to_idx[movie_id]
        return float(np.clip(self.predicted[i, j], 1.0, 5.0))

    def top_n(self, user_id: int, movies: pd.DataFrame,
              rated_ids: set, n: int = 10) -> pd.DataFrame:
        candidates = movies[~movies['MovieID'].isin(rated_ids)].copy()
        candidates['PredRating'] = candidates['MovieID'].apply(
            lambda mid: self.predict(user_id, mid)
        )
        return (candidates.sort_values('PredRating', ascending=False)
                          .head(n)[['MovieID', 'Title', 'Genres', 'PredRating']]
                          .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 1b. Collaborative Filtering — KNN (sklearn)
# ---------------------------------------------------------------------------

class KNNRecommender:
    """
    User-based or item-based KNN collaborative filtering.
    """

    def __init__(self, k: int = 40, user_based: bool = True):
        self.k          = k
        self.user_based = user_based
        self.knn        = None
        self.matrix     = None   # dense (users x movies) or (movies x users)
        self.user_ids   = None
        self.movie_ids  = None
        self.uid_to_idx = None
        self.mid_to_idx = None
        self.user_means = None

    def fit(self, ratings: pd.DataFrame):
        sparse, self.user_ids, self.movie_ids, self.uid_to_idx, self.mid_to_idx = \
            build_rating_matrix(ratings)

        dense = sparse.toarray().astype(float)

        # Per-user mean-centring
        self.user_means = np.zeros(dense.shape[0])
        for i in range(dense.shape[0]):
            rated = dense[i][dense[i] != 0]
            if len(rated):
                self.user_means[i] = rated.mean()
                dense[i][dense[i] != 0] -= self.user_means[i]

        if self.user_based:
            self.matrix = dense                   # rows = users
        else:
            self.matrix = dense.T                 # rows = movies

        self.knn = NearestNeighbors(
            n_neighbors=min(self.k + 1, self.matrix.shape[0]),
            metric='cosine', algorithm='brute', n_jobs=1
        )
        self.knn.fit(self.matrix)
        mode = 'user' if self.user_based else 'item'
        print(f"KNN fitted  ({mode}-based, k={self.k})")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.uid_to_idx or movie_id not in self.mid_to_idx:
            return 3.0
        u_idx = self.uid_to_idx[user_id]
        m_idx = self.mid_to_idx[movie_id]

        if self.user_based:
            vec = self.matrix[u_idx].reshape(1, -1)
            distances, indices = self.knn.kneighbors(vec)
            neighbor_idxs = indices[0][1:]   # exclude self
            neighbor_dist = distances[0][1:]
            weights = 1 - neighbor_dist      # cosine similarity = 1 - cosine_distance
            ratings_col = self.matrix[neighbor_idxs, m_idx]
            # Recover mean-centred ratings
            bias = self.user_means[u_idx]
            neighbour_means = self.user_means[neighbor_idxs]
            rated_mask = ratings_col != 0
            if not rated_mask.any():
                return float(np.clip(bias, 1, 5))
            w = weights[rated_mask]
            r = ratings_col[rated_mask] + neighbour_means[rated_mask]
            pred = bias + np.dot(w, r) / (w.sum() + 1e-9)
        else:
            vec = self.matrix[m_idx].reshape(1, -1)
            distances, indices = self.knn.kneighbors(vec)
            neighbor_movie_idxs = indices[0][1:]
            neighbor_dist = distances[0][1:]
            weights = 1 - neighbor_dist
            user_row = self.matrix.T[u_idx]   # original dense row for this user
            ratings_row = user_row[neighbor_movie_idxs]
            rated_mask = ratings_row != 0
            if not rated_mask.any():
                return float(np.clip(self.user_means[u_idx], 1, 5))
            w = weights[rated_mask]
            r = ratings_row[rated_mask] + self.user_means[u_idx]
            pred = self.user_means[u_idx] + np.dot(w, r - self.user_means[u_idx]) / (w.sum() + 1e-9)

        return float(np.clip(pred, 1.0, 5.0))

    def top_n(self, user_id: int, movies: pd.DataFrame,
              rated_ids: set, n: int = 10) -> pd.DataFrame:
        candidates = movies[~movies['MovieID'].isin(rated_ids)].copy()
        candidates['PredRating'] = candidates['MovieID'].apply(
            lambda mid: self.predict(user_id, mid)
        )
        return (candidates.sort_values('PredRating', ascending=False)
                          .head(n)[['MovieID', 'Title', 'Genres', 'PredRating']]
                          .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. Content-Based Filtering (TF-IDF on genres)
# ---------------------------------------------------------------------------

def build_content_model(movies: pd.DataFrame):
    """
    Build TF-IDF matrix on genre strings.
    Returns (tfidf_matrix, vectorizer, movies_indexed).
    """
    movies = movies.copy()
    movies['GenreTokens'] = movies['Genres'].str.replace('|', ' ', regex=False)

    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies['GenreTokens'])

    movies = movies.set_index('MovieID')
    print(f"Content model built  (matrix shape: {tfidf_matrix.shape})")
    return tfidf_matrix, vectorizer, movies


def build_content_artifact(movies: pd.DataFrame) -> dict[str, object]:
    """Return the content model pieces in a dictionary for artifact storage."""
    tfidf_matrix, vectorizer, movies_indexed = build_content_model(movies)
    return {
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'movies_indexed': movies_indexed,
    }


def content_similar_movies(movie_id: int, tfidf_matrix,
                            movies_indexed: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return N most genre-similar movies to a given movie."""
    if movie_id not in movies_indexed.index:
        return pd.DataFrame()

    idx = movies_indexed.index.get_loc(movie_id)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0

    top_indices = np.argsort(sim_scores)[::-1][:n]
    top_ids = movies_indexed.index[top_indices]

    result = movies_indexed.loc[top_ids, ['Title', 'Genres']].copy()
    result['Similarity'] = sim_scores[top_indices]
    return result.reset_index()[['MovieID', 'Title', 'Genres', 'Similarity']]


def top_n_content(user_id: int, ratings: pd.DataFrame,
                  tfidf_matrix, movies_indexed: pd.DataFrame,
                  n: int = 10) -> pd.DataFrame:
    """Content-based top-N: user profile = weighted mean of rated movie vectors."""
    user_ratings = ratings[ratings['UserID'] == user_id]
    rated_ids    = set(user_ratings['MovieID'].tolist())

    valid_ids = [mid for mid in user_ratings['MovieID'] if mid in movies_indexed.index]
    if not valid_ids:
        return pd.DataFrame()

    indices = [movies_indexed.index.get_loc(mid) for mid in valid_ids]
    weights = np.array([
        user_ratings[user_ratings['MovieID'] == mid]['Rating'].values[0]
        for mid in valid_ids
    ], dtype=float)
    weights /= weights.sum()

    user_profile = np.average(tfidf_matrix[indices].toarray(), axis=0, weights=weights)

    unseen_movies   = movies_indexed[~movies_indexed.index.isin(rated_ids)]
    unseen_indices  = [movies_indexed.index.get_loc(mid) for mid in unseen_movies.index]

    sim_scores = cosine_similarity(
        user_profile.reshape(1, -1),
        tfidf_matrix[unseen_indices].toarray()
    ).flatten()

    result = unseen_movies[['Title', 'Genres']].copy()
    result['Similarity'] = sim_scores
    return (result.reset_index()
                  .sort_values('Similarity', ascending=False)
                  .head(n)[['MovieID', 'Title', 'Genres', 'Similarity']]
                  .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 3. Hybrid Recommender
# ---------------------------------------------------------------------------

def top_n_hybrid(svd_model: SVDRecommender, user_id: int,
                 ratings: pd.DataFrame, movies: pd.DataFrame,
                 tfidf_matrix, movies_indexed: pd.DataFrame,
                 n: int = 10, alpha: float = 0.7) -> pd.DataFrame:
    """
    Hybrid score = alpha * CF_normalised + (1-alpha) * content_similarity
    """
    user_ratings = ratings[ratings['UserID'] == user_id]
    rated_ids    = set(user_ratings['MovieID'].tolist())
    candidates   = movies[~movies['MovieID'].isin(rated_ids)].copy()

    # CF scores
    candidates['CF_score'] = candidates['MovieID'].apply(
        lambda mid: svd_model.predict(user_id, mid)
    )
    cf_min, cf_max = candidates['CF_score'].min(), candidates['CF_score'].max()
    if cf_max > cf_min:
        candidates['CF_norm'] = (candidates['CF_score'] - cf_min) / (cf_max - cf_min)
    else:
        candidates['CF_norm'] = 0.5

    # Content scores
    valid_rated = [mid for mid in user_ratings['MovieID'] if mid in movies_indexed.index]
    if valid_rated:
        indices = [movies_indexed.index.get_loc(mid) for mid in valid_rated]
        weights = np.array([
            user_ratings[user_ratings['MovieID'] == mid]['Rating'].values[0]
            for mid in valid_rated
        ], dtype=float)
        weights /= weights.sum()
        user_profile = np.average(tfidf_matrix[indices].toarray(), axis=0, weights=weights)

        valid_cand_mask = candidates['MovieID'].isin(movies_indexed.index)
        cand_indices    = [movies_indexed.index.get_loc(mid)
                           for mid in candidates.loc[valid_cand_mask, 'MovieID']]
        sim_scores = np.zeros(len(candidates))
        if cand_indices:
            sims = cosine_similarity(
                user_profile.reshape(1, -1),
                tfidf_matrix[cand_indices].toarray()
            ).flatten()
            sim_scores[valid_cand_mask.values] = sims
    else:
        sim_scores = np.zeros(len(candidates))

    candidates['Content_score'] = sim_scores
    candidates['Hybrid_score']  = (alpha * candidates['CF_norm'] +
                                   (1 - alpha) * candidates['Content_score'])

    return (candidates.sort_values('Hybrid_score', ascending=False)
                      .head(n)[['MovieID', 'Title', 'Genres',
                                 'CF_score', 'Content_score', 'Hybrid_score']]
                      .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 4. Cold-start: cluster-based fallback
# ---------------------------------------------------------------------------

def top_n_for_cluster(cluster_id: int, features_labeled: pd.DataFrame,
                      ratings: pd.DataFrame, movies: pd.DataFrame,
                      n: int = 10) -> pd.DataFrame:
    """Recommend top-N highest-rated movies among users in the given cluster."""
    cluster_users   = features_labeled[features_labeled['Cluster'] == cluster_id].index
    cluster_ratings = ratings[ratings['UserID'].isin(cluster_users)]

    ranked = (cluster_ratings.groupby('MovieID')
                             .agg(AvgRating=('Rating', 'mean'),
                                  NumRatings=('Rating', 'count'))
                             .sort_values(['AvgRating', 'NumRatings'], ascending=False))
    filtered = ranked[ranked['NumRatings'] >= 10]
    if filtered.empty:
        filtered = ranked

    top = (filtered.head(n)
                  .reset_index()
                  .merge(movies[['MovieID', 'Title', 'Genres']], on='MovieID'))
    return top[['MovieID', 'Title', 'Genres', 'AvgRating', 'NumRatings']]


def build_cluster_summary(features_labeled: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Summarize ratings by cluster and movie."""
    merged = ratings.merge(
        features_labeled[['Cluster']],
        left_on='UserID',
        right_index=True,
        how='left',
    )
    return (merged.groupby(['Cluster', 'MovieID'], as_index=False)
                  .agg(AvgRating=('Rating', 'mean'),
                       NumRatings=('Rating', 'count')))


# ---------------------------------------------------------------------------
# Cross-validation helper (RMSE / MAE) — used by evaluate.py
# ---------------------------------------------------------------------------

def cross_validate_svd(ratings: pd.DataFrame, n_factors: int = 50,
                        n_splits: int = 5):
    """K-fold cross-validation for SVDRecommender. Returns (rmse_list, mae_list)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(ratings))
    rmse_list, mae_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), 1):
        train_df = ratings.iloc[train_idx].reset_index(drop=True)
        test_df  = ratings.iloc[test_idx].reset_index(drop=True)

        model = SVDRecommender(n_factors=n_factors)
        model.fit(train_df)

        preds, trues = [], []
        for _, row in test_df.iterrows():
            preds.append(model.predict(int(row['UserID']), int(row['MovieID'])))
            trues.append(row['Rating'])

        rmse = np.sqrt(mean_squared_error(trues, preds))
        mae  = np.mean(np.abs(np.array(trues) - np.array(preds)))
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(f"  Fold {fold}/{n_splits}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    return rmse_list, mae_list


if __name__ == '__main__':
    ratings = load_ratings()
    movies  = load_movies()

    print("Training SVD ...")
    svd = SVDRecommender(n_factors=50).fit(ratings)

    demo_user = 1
    rated     = set(ratings[ratings['UserID'] == demo_user]['MovieID'].tolist())

    print(f"\nTop-10 CF (SVD) for user {demo_user}:")
    print(svd.top_n(demo_user, movies, rated, n=10).to_string(index=False))

    print("\nBuilding content model ...")
    tfidf_matrix, _, movies_indexed = build_content_model(movies)

    print(f"\nTop-10 Content-Based for user {demo_user}:")
    print(top_n_content(demo_user, ratings, tfidf_matrix, movies_indexed).to_string(index=False))

    print(f"\nTop-10 Hybrid for user {demo_user}:")
    print(top_n_hybrid(svd, demo_user, ratings, movies,
                        tfidf_matrix, movies_indexed).to_string(index=False))
