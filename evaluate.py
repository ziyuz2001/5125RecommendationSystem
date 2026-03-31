"""
evaluate.py
Model evaluation (no scikit-surprise):
  - 5-fold cross-validation (RMSE / MAE) for SVDRecommender
  - Precision@K and Recall@K for top-N recommendations
  - Silhouette score summary for clustering
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data_loader import load_ratings, load_movies
from recommender import SVDRecommender, build_content_model, top_n_content

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')


# ---------------------------------------------------------------------------
# 1. Cross-Validation: RMSE and MAE
# ---------------------------------------------------------------------------

def evaluate_cf_models(ratings: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    5-fold CV for SVD with different numbers of latent factors.
    Returns a DataFrame with mean RMSE and MAE.
    """
    factor_configs = [20, 50, 100]
    results = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(ratings))

    for n_factors in factor_configs:
        name = f'SVD (k={n_factors})'
        print(f"  Cross-validating {name} ...")
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
            print(f"    Fold {fold}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        results.append({
            'Model':     name,
            'RMSE_mean': np.mean(rmse_list),
            'RMSE_std':  np.std(rmse_list),
            'MAE_mean':  np.mean(mae_list),
            'MAE_std':   np.std(mae_list),
        })

    df = pd.DataFrame(results).sort_values('RMSE_mean')
    print("\nCV Results:")
    print(df[['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std']].to_string(index=False))
    return df


def plot_cv_results(cv_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, std_col, color in [
        (axes[0], 'RMSE_mean', 'RMSE_std', 'steelblue'),
        (axes[1], 'MAE_mean',  'MAE_std',  'salmon')
    ]:
        bars = ax.barh(cv_df['Model'], cv_df[metric],
                       xerr=cv_df[std_col], color=color, edgecolor='white', capsize=4)
        ax.set_xlabel(metric.replace('_mean', ''), fontsize=12)
        ax.set_title(f'Model Comparison – {metric.replace("_mean", "")}', fontsize=13)
        for bar in bars:
            ax.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.4f}', va='center', fontsize=9)

    fig.suptitle('5-Fold Cross-Validation Results', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '10_cv_rmse_mae.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 2. Precision@K and Recall@K
# ---------------------------------------------------------------------------

def evaluate_precision_recall(ratings: pd.DataFrame,
                               k_values=(5, 10, 20),
                               threshold: float = 3.5,
                               sample_users: int = 500) -> pd.DataFrame:
    """
    Train SVD on 80% of each sampled user's ratings, test on 20%.
    Compute Precision@K, Recall@K, F1@K.
    """
    # Use a sample of users for speed
    rng = np.random.default_rng(42)
    user_ids = rng.choice(ratings['UserID'].unique(),
                          size=min(sample_users, ratings['UserID'].nunique()),
                          replace=False)

    train_rows, test_rows = [], []
    for uid in user_ids:
        u_df = ratings[ratings['UserID'] == uid]
        split = max(1, int(len(u_df) * 0.8))
        u_shuffled = u_df.sample(frac=1, random_state=42)
        train_rows.append(u_shuffled.iloc[:split])
        test_rows.append(u_shuffled.iloc[split:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    model = SVDRecommender(n_factors=50)
    model.fit(train_df)

    # Build predictions dict: uid -> list of (pred, true)
    user_preds = {}
    for _, row in test_df.iterrows():
        uid = int(row['UserID'])
        mid = int(row['MovieID'])
        pred = model.predict(uid, mid)
        user_preds.setdefault(uid, []).append((pred, row['Rating']))

    rows = []
    for k in k_values:
        precisions, recalls = [], []
        for uid, preds in user_preds.items():
            top_k      = sorted(preds, key=lambda x: x[0], reverse=True)[:k]
            n_hit      = sum(1 for _, true_r in top_k if true_r >= threshold)
            n_relevant = sum(1 for _, true_r in preds if true_r >= threshold)
            precisions.append(n_hit / k)
            recalls.append(n_hit / n_relevant if n_relevant > 0 else 0)

        p, r = np.mean(precisions), np.mean(recalls)
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0
        rows.append({'K': k, 'Precision@K': p, 'Recall@K': r, 'F1@K': f1})

    df = pd.DataFrame(rows)
    print("\nPrecision / Recall @ K  (SVD k=50, threshold=3.5):")
    print(df.to_string(index=False))
    return df


def plot_precision_recall(pr_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, color in [
        (axes[0], 'Precision@K', 'steelblue'),
        (axes[1], 'Recall@K',    'salmon'),
        (axes[2], 'F1@K',        'seagreen')
    ]:
        ax.bar(pr_df['K'].astype(str), pr_df[col], color=color, edgecolor='white')
        ax.set_xlabel('K')
        ax.set_ylabel(col)
        ax.set_title(col)
        for i, v in enumerate(pr_df[col]):
            ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

    fig.suptitle('Top-N Evaluation (SVD k=50, threshold ≥ 3.5)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '11_precision_recall.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 3. Clustering evaluation
# ---------------------------------------------------------------------------

def evaluate_clustering(features: pd.DataFrame) -> pd.DataFrame:
    """Silhouette scores for K = 2..8."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    feat_cols = [c for c in features.columns if c != 'Cluster']
    X = StandardScaler().fit_transform(features[feat_cols].values)

    rows = []
    for k in range(2, 9):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels, sample_size=2000, random_state=42)
        rows.append({'K': k, 'Inertia': km.inertia_, 'Silhouette': sil})

    df = pd.DataFrame(rows)
    print("\nClustering Evaluation:")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation():
    print("Loading data...")
    ratings = load_ratings()

    print("\n[1/3] Cross-Validating SVD models (5-fold) ...")
    cv_df = evaluate_cf_models(ratings, n_splits=5)
    plot_cv_results(cv_df)

    print("\n[2/3] Precision / Recall @ K ...")
    pr_df = evaluate_precision_recall(ratings)
    plot_precision_recall(pr_df)

    print("\n[3/3] Clustering evaluation ...")
    from clustering import build_user_features
    from data_loader import load_users
    users    = load_users()
    features = build_user_features(ratings, users)
    clust_df = evaluate_clustering(features)

    print("\nEvaluation complete. Plots saved to ./plots/")
    return cv_df, pr_df, clust_df


if __name__ == '__main__':
    run_evaluation()
