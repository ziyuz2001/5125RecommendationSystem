"""
evaluate.py
Shared frozen holdout evaluation for the recommender benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from artifact_store import save_csv, save_joblib, save_json
from data_loader import load_movies, load_ratings
from recommender import KNNRecommender, SVDRecommender, build_content_artifact


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def build_holdout_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a deterministic per-user train/test split.

    Each user with at least two ratings contributes at least one test row and
    at least one train row. Single-rating users stay entirely in train.
    """

    ratings = load_ratings().copy()
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    sort_cols = [col for col in ["Timestamp", "MovieID"] if col in ratings.columns]

    for user_id, group in ratings.groupby("UserID", sort=True):
        group = group.sort_values(sort_cols).reset_index(drop=True) if sort_cols else group.reset_index(drop=True)
        if len(group) < 2:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(len(group) * test_size)))
        n_test = min(n_test, len(group) - 1)
        test_rows = group.sample(n=n_test, random_state=random_state + int(user_id)).index
        test_group = group.loc[test_rows]
        train_group = group.drop(test_rows)

        train_parts.append(train_group)
        test_parts.append(test_group)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()

    sort_train = [col for col in ["UserID", "MovieID", "Timestamp"] if col in train_df.columns]
    sort_test = [col for col in ["UserID", "MovieID", "Timestamp"] if col in test_df.columns]
    train_df = train_df.sort_values(sort_train).reset_index(drop=True) if sort_train else train_df.reset_index(drop=True)
    test_df = test_df.sort_values(sort_test).reset_index(drop=True) if sort_test else test_df.reset_index(drop=True)
    return train_df, test_df


def save_split_artifact(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Path:
    """Persist the frozen split so later tasks can reuse the exact benchmark data."""

    payload = {
        "train": train_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    return save_joblib("eval_split.joblib", payload)


def choose_recommender_winners(rows: list[dict]) -> dict[str, str]:
    """Select overall and app-facing recommender winners from benchmark rows."""

    if not rows:
        payload = {"overall_winner": "", "app_winner": ""}
        save_json("recommender_selection.json", payload)
        return payload

    ranked = sorted(rows, key=lambda row: float(row["f1_at_k"]), reverse=True)
    overall_winner = str(ranked[0]["model_name"])
    text_compatible_rows = [
        row for row in ranked if bool(row.get("text_compatible")) and row["model_name"] in {"content", "hybrid"}
    ]
    app_winner = str(text_compatible_rows[0]["model_name"]) if text_compatible_rows else ""

    payload = {
        "overall_winner": overall_winner,
        "app_winner": app_winner,
    }
    save_json("recommender_selection.json", payload)
    return payload


def compute_recommendation_metrics(
    scored_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 3.5,
) -> dict[str, float]:
    """
    Compute macro-averaged Precision@K, Recall@K, and F1@K over users.

    The input frame must contain:
    - UserID
    - actual_rating
    - predicted_score
    """

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []

    for _, user_df in scored_df.groupby("UserID", sort=True):
        relevant_mask = user_df["actual_rating"] >= threshold
        relevant_count = int(relevant_mask.sum())
        if relevant_count == 0:
            continue

        ranked = user_df.sort_values("predicted_score", ascending=False).head(k)
        hits = int((ranked["actual_rating"] >= threshold).sum())
        denom = min(k, len(user_df))
        precision = hits / denom if denom else 0.0
        recall = hits / relevant_count if relevant_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "precision_at_k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "f1_at_k": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "users_evaluated": float(len(precision_scores)),
    }


def _build_user_content_profile(
    user_train: pd.DataFrame,
    content_artifact: dict[str, object],
) -> np.ndarray | None:
    tfidf_matrix = content_artifact["tfidf_matrix"]
    movies_indexed: pd.DataFrame = content_artifact["movies_indexed"]

    valid = user_train[user_train["MovieID"].isin(movies_indexed.index)]
    if valid.empty:
        return None

    movie_ids = valid["MovieID"].tolist()
    weights = valid["Rating"].astype(float).to_numpy()
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return None

    weights = weights / weight_sum
    indices = [movies_indexed.index.get_loc(movie_id) for movie_id in movie_ids]
    profile = np.average(tfidf_matrix[indices].toarray(), axis=0, weights=weights)
    return profile


def _score_content_candidates(
    user_train: pd.DataFrame,
    candidate_movie_ids: Iterable[int],
    content_artifact: dict[str, object],
) -> dict[int, float]:
    tfidf_matrix = content_artifact["tfidf_matrix"]
    movies_indexed: pd.DataFrame = content_artifact["movies_indexed"]

    profile = _build_user_content_profile(user_train, content_artifact)
    if profile is None:
        return {int(movie_id): 1.0 for movie_id in candidate_movie_ids}

    candidate_movie_ids = [int(movie_id) for movie_id in candidate_movie_ids if int(movie_id) in movies_indexed.index]
    if not candidate_movie_ids:
        return {}

    candidate_indices = [movies_indexed.index.get_loc(movie_id) for movie_id in candidate_movie_ids]
    similarities = cosine_similarity(profile.reshape(1, -1), tfidf_matrix[candidate_indices].toarray()).flatten()
    return {
        movie_id: float(np.clip(1.0 + 4.0 * similarity, 1.0, 5.0))
        for movie_id, similarity in zip(candidate_movie_ids, similarities)
    }


def _score_holdout_items(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_bundle: dict[str, object],
) -> pd.DataFrame:
    movies = load_movies()
    movie_lookup = movies.set_index("MovieID")
    rows: list[dict[str, object]] = []

    for user_id, user_test in test_df.groupby("UserID", sort=True):
        user_train = train_df[train_df["UserID"] == user_id]
        candidate_movie_ids = user_test["MovieID"].tolist()

        content_scores: dict[int, float] = {}
        if model_name in {"content", "hybrid"}:
            content_scores = _score_content_candidates(
                user_train=user_train,
                candidate_movie_ids=candidate_movie_ids,
                content_artifact=model_bundle["content_artifact"],
            )

        for row in user_test.itertuples(index=False):
            movie_id = int(row.MovieID)
            actual_rating = float(row.Rating)

            if model_name == "svd":
                predicted_score = float(model_bundle["svd_model"].predict(int(user_id), movie_id))
            elif model_name == "knn":
                predicted_score = float(model_bundle["knn_model"].predict(int(user_id), movie_id))
            elif model_name == "content":
                predicted_score = float(content_scores.get(movie_id, 1.0))
            elif model_name == "hybrid":
                svd_score = float(model_bundle["svd_model"].predict(int(user_id), movie_id))
                content_score = float(content_scores.get(movie_id, 1.0))
                predicted_score = float(np.clip(0.7 * svd_score + 0.3 * content_score, 1.0, 5.0))
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            movie_meta = movie_lookup.loc[movie_id] if movie_id in movie_lookup.index else None
            rows.append(
                {
                    "model_name": model_name,
                    "UserID": int(user_id),
                    "MovieID": movie_id,
                    "Title": movie_meta["Title"] if movie_meta is not None else "",
                    "Genres": movie_meta["Genres"] if movie_meta is not None else "",
                    "actual_rating": actual_rating,
                    "predicted_score": predicted_score,
                    "abs_error": abs(actual_rating - predicted_score),
                    "relevant": actual_rating >= 3.5,
                }
            )

    return pd.DataFrame(rows)


def _plot_recommender_comparison(metrics_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = metrics_df.sort_values("f1_at_k", ascending=False).reset_index(drop=True)
    sns.barplot(data=plot_df, x="model_name", y="f1_at_k", ax=ax, color="steelblue")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1@K")
    ax.set_title("Recommender Comparison on Shared Holdout Split")
    ax.set_ylim(0, max(0.05, float(plot_df["f1_at_k"].max()) * 1.15))
    for idx, value in enumerate(plot_df["f1_at_k"]):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = PLOTS_DIR / "recommender_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_metrics_by_k(
    scored_frames: list[pd.DataFrame],
    ks: tuple[int, ...] = (5, 10, 15, 20),
    threshold: float = 3.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scored_df in scored_frames:
        if scored_df.empty:
            continue

        model_name = str(scored_df["model_name"].iloc[0])
        text_compatible = model_name in {"content", "hybrid"}
        for k in ks:
            metrics = compute_recommendation_metrics(scored_df, k=k, threshold=threshold)
            rows.append(
                {
                    "model_name": model_name,
                    "text_compatible": text_compatible,
                    "k": k,
                    "precision_at_k": metrics["precision_at_k"],
                    "recall_at_k": metrics["recall_at_k"],
                    "f1_at_k": metrics["f1_at_k"],
                    "users_evaluated": metrics["users_evaluated"],
                }
            )

    return pd.DataFrame(rows).sort_values(["k", "f1_at_k"], ascending=[True, False]).reset_index(drop=True)


def _plot_recommender_metrics_by_k(metrics_by_k_df: pd.DataFrame) -> Path:
    plot_df = metrics_by_k_df.melt(
        id_vars=["model_name", "k"],
        value_vars=["precision_at_k", "recall_at_k", "f1_at_k"],
        var_name="metric",
        value_name="score",
    )

    metric_labels = {
        "precision_at_k": "Precision@K",
        "recall_at_k": "Recall@K",
        "f1_at_k": "F1@K",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, metric_name in zip(axes, ["Precision@K", "Recall@K", "F1@K"]):
        subset = plot_df[plot_df["metric"] == metric_name]
        sns.lineplot(
            data=subset,
            x="k",
            y="score",
            hue="model_name",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(metric_name)
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(0.05, float(plot_df["score"].max()) * 1.1))
        ax.legend(title="Model")

    fig.suptitle("Recommender Metrics Across K Values", fontsize=13)
    fig.tight_layout()
    path = PLOTS_DIR / "recommender_metrics_by_k.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_error_report(scored_frames: list[pd.DataFrame], top_n: int = 25) -> pd.DataFrame:
    if not scored_frames:
        return pd.DataFrame(
            columns=[
                "model_name",
                "UserID",
                "MovieID",
                "Title",
                "Genres",
                "actual_rating",
                "predicted_score",
                "abs_error",
                "relevant",
            ]
        )

    scored = pd.concat(scored_frames, ignore_index=True)
    error_rows: list[pd.DataFrame] = []
    for model_name, group in scored.groupby("model_name", sort=True):
        error_rows.append(group.sort_values("abs_error", ascending=False).head(top_n))
    return pd.concat(error_rows, ignore_index=True)


def _sample_benchmark_users(
    test_df: pd.DataFrame,
    max_users: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_users is None:
        return test_df

    user_ids = sorted(test_df["UserID"].unique())
    if len(user_ids) <= max_users:
        return test_df

    rng = np.random.default_rng(random_state)
    selected_users = sorted(rng.choice(user_ids, size=max_users, replace=False).tolist())
    return test_df[test_df["UserID"].isin(selected_users)].reset_index(drop=True)


def run_evaluation(
    k: int = 10,
    threshold: float = 3.5,
    test_size: float = 0.2,
    random_state: int = 42,
    max_users: int | None = 50,
) -> dict[str, object]:
    """
    Run the frozen-split recommender benchmark and persist artifacts.
    """

    train_df, test_df = build_holdout_split(test_size=test_size, random_state=random_state)
    save_split_artifact(train_df, test_df)
    benchmark_test_df = _sample_benchmark_users(test_df, max_users=max_users, random_state=random_state)

    movies = load_movies()
    content_artifact = build_content_artifact(movies)
    svd_model = SVDRecommender(n_factors=50).fit(train_df)
    knn_model = KNNRecommender(k=40, user_based=False).fit(train_df)

    model_bundle = {
        "svd_model": svd_model,
        "knn_model": knn_model,
        "content_artifact": content_artifact,
    }

    scored_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    for model_name in ["svd", "knn", "content", "hybrid"]:
        scored_df = _score_holdout_items(train_df, benchmark_test_df, model_name, model_bundle)
        scored_frames.append(scored_df)

        metrics = compute_recommendation_metrics(scored_df, k=k, threshold=threshold)
        metrics_rows.append(
            {
                "model_name": model_name,
                "text_compatible": model_name in {"content", "hybrid"},
                "k": k,
                "precision_at_k": metrics["precision_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "f1_at_k": metrics["f1_at_k"],
                "users_evaluated": metrics["users_evaluated"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_at_k", ascending=False).reset_index(drop=True)
    save_csv("recommender_metrics.csv", metrics_df)
    _plot_recommender_comparison(metrics_df)
    metrics_by_k_df = _build_metrics_by_k(scored_frames, threshold=threshold)
    save_csv("recommender_metrics_by_k.csv", metrics_by_k_df)
    _plot_recommender_metrics_by_k(metrics_by_k_df)
    winners = choose_recommender_winners(metrics_rows)

    errors_df = _build_error_report(scored_frames)
    save_csv("recommender_errors.csv", errors_df)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "benchmark_test_df": benchmark_test_df,
        "metrics": metrics_df,
        "metrics_by_k": metrics_by_k_df,
        "errors": errors_df,
        "winners": winners,
    }


if __name__ == "__main__":
    run_evaluation()
