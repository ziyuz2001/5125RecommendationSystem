from __future__ import annotations

import pandas as pd

from artifact_store import load_joblib, load_json
from data_loader import load_movies


def choose_recommendation_mode(parsed: dict[str, object]) -> str:
    if parsed["positive_genres"] or parsed["positive_keywords"]:
        return "primary"
    return "cluster_fallback"


def score_movies_from_genres(
    movies: pd.DataFrame,
    positive_genres: list[str],
    negative_genres: list[str],
) -> pd.DataFrame:
    scored = movies.copy()
    scored["Genres"] = scored["Genres"].fillna("")
    scored["genre_score"] = 0.0

    for genre in positive_genres:
        scored["genre_score"] += scored["Genres"].str.contains(genre, regex=False).astype(float)
    for genre in negative_genres:
        scored["genre_score"] -= scored["Genres"].str.contains(genre, regex=False).astype(float)

    return scored.sort_values(["genre_score", "Title"], ascending=[False, True]).reset_index(drop=True)


def apply_cluster_prior(scored: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    cluster_prior = (
        cluster_summary.groupby("MovieID", as_index=False)
        .agg(cluster_prior=("AvgRating", "mean"))
    )
    merged = scored.merge(cluster_prior, on="MovieID", how="left").fillna({"cluster_prior": 0.0})
    merged["final_score"] = merged["genre_score"] + 0.2 * merged["cluster_prior"]
    return merged.sort_values(["final_score", "Title"], ascending=[False, True]).reset_index(drop=True)


def _top_movies_for_best_cluster(cluster_summary: pd.DataFrame, movies: pd.DataFrame, n: int) -> pd.DataFrame:
    best_cluster = (
        cluster_summary.groupby("Cluster")["AvgRating"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    ranked = (
        cluster_summary[cluster_summary["Cluster"] == best_cluster]
        .sort_values(["AvgRating", "NumRatings", "MovieID"], ascending=[False, False, True])
        .drop_duplicates(subset=["MovieID"])
        .head(n)
        .loc[:, ["MovieID"]]
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index
    return (
        ranked.merge(movies[["MovieID", "Title", "Genres"]], on="MovieID", how="left")
        .sort_values("rank")
        .drop(columns=["rank"])
        .reset_index(drop=True)
    )


def recommend_from_preferences(parsed: dict[str, object], n: int = 10) -> pd.DataFrame:
    selection = load_json("recommender_selection.json")
    app_winner = selection["app_winner"]
    movies = load_movies()

    if choose_recommendation_mode(parsed) == "cluster_fallback":
        cluster_summary = load_joblib("cluster_summary.joblib")
        return _top_movies_for_best_cluster(cluster_summary, movies, n=n)

    ranked = score_movies_from_genres(
        movies,
        parsed["positive_genres"],
        parsed["negative_genres"],
    )
    if app_winner == "hybrid":
        cluster_summary = load_joblib("cluster_summary.joblib")
        ranked = apply_cluster_prior(ranked, cluster_summary)
    else:
        ranked["final_score"] = ranked["genre_score"]

    ranked["source_model"] = app_winner
    return ranked.head(n)[["MovieID", "Title", "Genres", "final_score", "source_model"]].reset_index(drop=True)
