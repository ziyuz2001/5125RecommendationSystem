import pandas as pd

from recommender import (
    SVDRecommender,
    build_cluster_summary,
    build_content_artifact,
)


def test_svd_recommender_can_fit_tiny_dataset():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 2, 2],
            "MovieID": [11, 12, 11, 12],
            "Rating": [4.0, 5.0, 2.0, 3.0],
        }
    )

    model = SVDRecommender(n_factors=2)
    model.fit(ratings)

    assert model.predict(1, 12) >= 1.0


def test_svd_recommender_handles_single_user_matrix():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 1],
            "MovieID": [11, 12, 13],
            "Rating": [4.0, 5.0, 3.0],
        }
    )

    model = SVDRecommender(n_factors=2)
    model.fit(ratings)

    assert model.predict(1, 12) >= 1.0


def test_svd_recommender_handles_single_movie_matrix():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 2, 3],
            "MovieID": [11, 11, 11],
            "Rating": [4.0, 5.0, 3.0],
        }
    )

    model = SVDRecommender(n_factors=2)
    model.fit(ratings)

    assert model.predict(2, 11) >= 1.0


def test_build_content_artifact_returns_expected_mapping():
    movies = pd.DataFrame(
        {
            "MovieID": [11, 12],
            "Title": ["A", "B"],
            "Genres": ["Action|Comedy", "Drama"],
        }
    )

    artifact = build_content_artifact(movies)

    assert set(artifact) == {"tfidf_matrix", "vectorizer", "movies_indexed"}
    assert list(artifact["movies_indexed"].index) == [11, 12]
    assert artifact["tfidf_matrix"].shape[0] == 2


def test_build_cluster_summary_aggregates_by_cluster_and_movie():
    features_labeled = pd.DataFrame(
        {"Cluster": [0, 0, 1]},
        index=pd.Index([1, 2, 3], name="UserID"),
    )
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 2, 3, 3],
            "MovieID": [10, 11, 10, 10, 12],
            "Rating": [4.0, 2.0, 5.0, 3.0, 1.0],
        }
    )

    summary = build_cluster_summary(features_labeled, ratings)

    expected = pd.DataFrame(
        {
            "Cluster": [0, 0, 1, 1],
            "MovieID": [10, 11, 10, 12],
            "AvgRating": [4.5, 2.0, 3.0, 1.0],
            "NumRatings": [2, 1, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected)
