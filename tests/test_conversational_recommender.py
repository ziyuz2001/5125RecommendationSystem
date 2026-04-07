from pathlib import Path

import pandas as pd

from evaluate import build_holdout_split
from evaluate import compute_recommendation_metrics
from evaluate import choose_recommender_winners
from evaluate import save_split_artifact
from artifact_store import load_joblib
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


def test_build_holdout_split_returns_train_and_test_frames():
    train_df, test_df = build_holdout_split(test_size=0.2, random_state=42)

    assert {"UserID", "MovieID", "Rating"}.issubset(train_df.columns)
    assert {"UserID", "MovieID", "Rating"}.issubset(test_df.columns)
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_save_split_artifact_round_trip(monkeypatch, tmp_path):
    import artifact_store

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", artifact_dir)

    train_df = pd.DataFrame({"UserID": [1], "MovieID": [10], "Rating": [4.0]})
    test_df = pd.DataFrame({"UserID": [1], "MovieID": [11], "Rating": [5.0]})

    saved_path = save_split_artifact(train_df, test_df)
    payload = load_joblib("eval_split.joblib")

    assert saved_path.exists()
    pd.testing.assert_frame_equal(payload["train"], train_df)
    pd.testing.assert_frame_equal(payload["test"], test_df)


def test_compute_recommendation_metrics_aggregates_users_deterministically():
    scored = pd.DataFrame(
        {
            "UserID": [1, 1, 1, 2, 2],
            "actual_rating": [5.0, 2.0, 4.0, 3.0, 5.0],
            "predicted_score": [0.9, 0.1, 0.8, 0.2, 0.7],
        }
    )

    metrics = compute_recommendation_metrics(scored, k=2, threshold=3.5)

    assert metrics["users_evaluated"] == 2.0
    assert metrics["precision_at_k"] == 0.75
    assert metrics["recall_at_k"] == 1.0
    assert round(metrics["f1_at_k"], 4) == 0.8333


def test_choose_recommender_winners_tracks_overall_and_app_models(monkeypatch, tmp_path):
    import artifact_store

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", artifact_dir)

    winners = choose_recommender_winners(
        [
            {"model_name": "svd", "f1_at_k": 0.40, "text_compatible": False},
            {"model_name": "hybrid", "f1_at_k": 0.35, "text_compatible": True},
            {"model_name": "content", "f1_at_k": 0.30, "text_compatible": True},
        ]
    )

    assert winners["overall_winner"] == "svd"
    assert winners["app_winner"] == "hybrid"
    assert (artifact_dir / "recommender_selection.json").exists()
