"""
recommender.py
Reusable recommender primitives for the final conversational project.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_rating_matrix(ratings: pd.DataFrame):
    """
    Returns:
        matrix: sparse user-item matrix
        user_ids: sorted user ids in row order
        movie_ids: sorted movie ids in column order
        uid_to_idx: user id to row index
        mid_to_idx: movie id to column index
    """
    user_ids = sorted(ratings["UserID"].unique())
    movie_ids = sorted(ratings["MovieID"].unique())
    uid_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    mid_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    rows = ratings["UserID"].map(uid_to_idx).values
    cols = ratings["MovieID"].map(mid_to_idx).values
    data = ratings["Rating"].values.astype(float)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    return matrix, user_ids, movie_ids, uid_to_idx, mid_to_idx


class SVDRecommender:
    """Matrix-factorization collaborative filtering using truncated SVD."""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.predicted = None
        self.user_ids = None
        self.movie_ids = None
        self.uid_to_idx = None
        self.mid_to_idx = None
        self.user_means = None

    def fit(self, ratings: pd.DataFrame):
        (
            matrix,
            self.user_ids,
            self.movie_ids,
            self.uid_to_idx,
            self.mid_to_idx,
        ) = build_rating_matrix(ratings)

        dense = matrix.toarray().astype(float)
        self.user_means = np.true_divide(
            dense.sum(axis=1),
            (dense != 0).sum(axis=1),
            where=(dense != 0).sum(axis=1) != 0,
        )

        normalized = dense.copy()
        for idx in range(dense.shape[0]):
            mask = normalized[idx] == 0
            normalized[idx, mask] = self.user_means[idx]

        k = min(self.n_factors, min(normalized.shape) - 1)
        if k < 1:
            self.predicted = normalized
            print(
                f"SVD fitted  (fallback, users={len(self.user_ids)}, movies={len(self.movie_ids)})"
            )
            return self

        u_mat, sigma, vt_mat = svds(normalized, k=k)
        self.predicted = u_mat @ np.diag(sigma) @ vt_mat
        print(f"SVD fitted  (factors={k}, users={len(self.user_ids)}, movies={len(self.movie_ids)})")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.uid_to_idx or movie_id not in self.mid_to_idx:
            return 3.0
        row_idx = self.uid_to_idx[user_id]
        col_idx = self.mid_to_idx[movie_id]
        return float(np.clip(self.predicted[row_idx, col_idx], 1.0, 5.0))

    def top_n(self, user_id: int, movies: pd.DataFrame, rated_ids: set, n: int = 10) -> pd.DataFrame:
        candidates = movies[~movies["MovieID"].isin(rated_ids)].copy()
        candidates["PredRating"] = candidates["MovieID"].apply(lambda movie_id: self.predict(user_id, movie_id))
        return (
            candidates.sort_values("PredRating", ascending=False)
            .head(n)[["MovieID", "Title", "Genres", "PredRating"]]
            .reset_index(drop=True)
        )


class KNNRecommender:
    """User-based or item-based KNN collaborative filtering."""

    def __init__(self, k: int = 40, user_based: bool = True):
        self.k = k
        self.user_based = user_based
        self.knn = None
        self.matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.uid_to_idx = None
        self.mid_to_idx = None
        self.user_means = None

    def fit(self, ratings: pd.DataFrame):
        (
            sparse,
            self.user_ids,
            self.movie_ids,
            self.uid_to_idx,
            self.mid_to_idx,
        ) = build_rating_matrix(ratings)

        dense = sparse.toarray().astype(float)
        self.user_means = np.zeros(dense.shape[0])
        for idx in range(dense.shape[0]):
            rated = dense[idx][dense[idx] != 0]
            if len(rated):
                self.user_means[idx] = rated.mean()
                dense[idx][dense[idx] != 0] -= self.user_means[idx]

        self.matrix = dense if self.user_based else dense.T
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k + 1, self.matrix.shape[0]),
            metric="cosine",
            algorithm="brute",
            n_jobs=1,
        )
        self.knn.fit(self.matrix)
        mode = "user" if self.user_based else "item"
        print(f"KNN fitted  ({mode}-based, k={self.k})")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.uid_to_idx or movie_id not in self.mid_to_idx:
            return 3.0

        user_idx = self.uid_to_idx[user_id]
        movie_idx = self.mid_to_idx[movie_id]

        if self.user_based:
            vec = self.matrix[user_idx].reshape(1, -1)
            distances, indices = self.knn.kneighbors(vec)
            neighbor_idxs = indices[0][1:]
            neighbor_dist = distances[0][1:]
            weights = 1 - neighbor_dist
            ratings_col = self.matrix[neighbor_idxs, movie_idx]
            bias = self.user_means[user_idx]
            neighbor_means = self.user_means[neighbor_idxs]
            rated_mask = ratings_col != 0
            if not rated_mask.any():
                return float(np.clip(bias, 1, 5))
            kept_weights = weights[rated_mask]
            kept_ratings = ratings_col[rated_mask] + neighbor_means[rated_mask]
            prediction = bias + np.dot(kept_weights, kept_ratings) / (kept_weights.sum() + 1e-9)
        else:
            vec = self.matrix[movie_idx].reshape(1, -1)
            distances, indices = self.knn.kneighbors(vec)
            neighbor_movie_idxs = indices[0][1:]
            neighbor_dist = distances[0][1:]
            weights = 1 - neighbor_dist
            user_row = self.matrix.T[user_idx]
            ratings_row = user_row[neighbor_movie_idxs]
            rated_mask = ratings_row != 0
            if not rated_mask.any():
                return float(np.clip(self.user_means[user_idx], 1, 5))
            kept_weights = weights[rated_mask]
            kept_ratings = ratings_row[rated_mask] + self.user_means[user_idx]
            prediction = self.user_means[user_idx] + np.dot(
                kept_weights,
                kept_ratings - self.user_means[user_idx],
            ) / (kept_weights.sum() + 1e-9)

        return float(np.clip(prediction, 1.0, 5.0))

    def top_n(self, user_id: int, movies: pd.DataFrame, rated_ids: set, n: int = 10) -> pd.DataFrame:
        candidates = movies[~movies["MovieID"].isin(rated_ids)].copy()
        candidates["PredRating"] = candidates["MovieID"].apply(lambda movie_id: self.predict(user_id, movie_id))
        return (
            candidates.sort_values("PredRating", ascending=False)
            .head(n)[["MovieID", "Title", "Genres", "PredRating"]]
            .reset_index(drop=True)
        )


def build_content_model(movies: pd.DataFrame):
    """Build a TF-IDF representation from movie genres."""
    movies = movies.copy()
    movies["GenreTokens"] = movies["Genres"].str.replace("|", " ", regex=False)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies["GenreTokens"])
    movies = movies.set_index("MovieID")

    print(f"Content model built  (matrix shape: {tfidf_matrix.shape})")
    return tfidf_matrix, vectorizer, movies


def build_content_artifact(movies: pd.DataFrame) -> dict[str, object]:
    """Return the content model pieces needed by the evaluator."""
    tfidf_matrix, vectorizer, movies_indexed = build_content_model(movies)
    return {
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer,
        "movies_indexed": movies_indexed,
    }


def build_cluster_summary(features_labeled: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean rating and rating count by cluster and movie."""
    merged = ratings.merge(
        features_labeled[["Cluster"]],
        left_on="UserID",
        right_index=True,
        how="left",
    )
    return merged.groupby(["Cluster", "MovieID"], as_index=False).agg(
        AvgRating=("Rating", "mean"),
        NumRatings=("Rating", "count"),
    )
