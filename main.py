"""
main.py
Full pipeline for the MovieLens 1M Recommendation System.

Usage:
    python main.py                    # run entire pipeline
    python main.py --step eda         # only EDA
    python main.py --step cluster     # only clustering
    python main.py --step recommend   # only recommendation demo
    python main.py --step evaluate    # only evaluation
    python main.py --user 42 --n 10   # top-10 recommendations for user 42
"""

import argparse
import os

# ── project modules ──────────────────────────────────────────────────────────
from data_loader  import build_database, load_ratings, load_movies, load_users
from eda          import run_eda
from clustering   import build_user_features, run_clustering_and_save_artifacts
from recommender  import (
    SVDRecommender, KNNRecommender,
    build_content_model,
    top_n_content, top_n_hybrid, top_n_for_cluster
)
from evaluate     import run_evaluation
from text_classifier import run_classification_pipeline


# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── pipeline steps ───────────────────────────────────────────────────────────

def step_database():
    section("STEP 0 – Build SQLite Database")
    build_database()


def step_eda():
    section("STEP 1 – Exploratory Data Analysis")
    run_eda()


def step_clustering():
    section("STEP 2 – User Clustering (K-Means)")
    features_labeled, scaler, km = run_clustering_and_save_artifacts()
    return features_labeled, scaler, km


def step_recommend(user_id: int = 1, n: int = 10):
    section(f"STEP 3 – Recommendations for User {user_id}")

    ratings = load_ratings()
    movies  = load_movies()
    users   = load_users()

    rated_ids = set(ratings[ratings['UserID'] == user_id]['MovieID'].tolist())

    # — Collaborative Filtering (SVD) —
    print(f"\nTraining SVD ...")
    svd_model = SVDRecommender(n_factors=50).fit(ratings)

    print(f"\n[CF – SVD] Top-{n} for user {user_id}:")
    recs_cf = svd_model.top_n(user_id, movies, rated_ids, n=n)
    print(recs_cf.to_string(index=False))

    # — Collaborative Filtering (KNN, item-based) —
    print(f"\nTraining KNN (item-based) ...")
    knn_model = KNNRecommender(k=40, user_based=False).fit(ratings)
    print(f"\n[CF – KNN (item-based)] Top-{n} for user {user_id}:")
    recs_knn = knn_model.top_n(user_id, movies, rated_ids, n=n)
    print(recs_knn.to_string(index=False))

    # — Content-Based —
    print("\nBuilding content model ...")
    tfidf_matrix, vectorizer, movies_indexed = build_content_model(movies)
    print(f"\n[Content-Based] Top-{n} for user {user_id}:")
    recs_cb = top_n_content(user_id, ratings, tfidf_matrix, movies_indexed, n=n)
    print(recs_cb.to_string(index=False))

    # — Hybrid —
    print(f"\n[Hybrid] Top-{n} for user {user_id}:")
    recs_h = top_n_hybrid(svd_model, user_id, ratings, movies,
                           tfidf_matrix, movies_indexed, n=n, alpha=0.7)
    print(recs_h.to_string(index=False))

    # — Cold-Start (cluster-based) —
    print("\nRunning clustering for cold-start demo ...")
    features = build_user_features(ratings, users)
    from clustering import run_kmeans
    features_labeled, _, _ = run_kmeans(features, n_clusters=5)

    cluster_id = features_labeled.loc[user_id, 'Cluster'] if user_id in features_labeled.index else 0
    print(f"\n[Cold-Start] User {user_id} → Cluster {cluster_id}  |  Top-{n}:")
    recs_cold = top_n_for_cluster(cluster_id, features_labeled, ratings, movies, n=n)
    print(recs_cold.to_string(index=False))


def step_evaluate():
    section("STEP 4 – Model Evaluation")
    run_evaluation()


def step_classify():
    section("STEP 5 – Train Clause Polarity Classifier")
    run_classification_pipeline()


def step_benchmark():
    section("STEP 6 – Benchmark Recommenders and Save Artifacts")
    run_evaluation()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MovieLens 1M Recommendation System')
    parser.add_argument('--step', choices=['all', 'db', 'eda', 'cluster', 'recommend', 'evaluate', 'classify', 'benchmark'],
                        default='all', help='Which pipeline step to run')
    parser.add_argument('--user', type=int, default=1,
                        help='User ID for recommendation demo')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of recommendations to show')
    args = parser.parse_args()

    if args.step in ('all', 'db'):
        step_database()

    if args.step in ('all', 'eda'):
        step_eda()

    if args.step in ('all', 'cluster'):
        step_clustering()

    if args.step in ('all', 'recommend'):
        step_recommend(user_id=args.user, n=args.n)

    if args.step in ('all', 'evaluate'):
        step_evaluate()

    if args.step in ('all', 'classify'):
        step_classify()

    if args.step in ('all', 'benchmark'):
        step_benchmark()

    print("\nDone. Plots are in ./plots/  |  Database is movielens.db")


if __name__ == '__main__':
    main()
