"""
clustering.py
K-Means and DBSCAN user clustering based on demographics + rating behaviour.
Satisfies the Classification/Clustering requirement of the assignment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data_loader import load_ratings, load_users, AGE_MAP, OCCUPATION_MAP, ALL_GENRES

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_user_features(ratings: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per user combining:
      - Demographics  : Gender (binary), Age (numeric), Occupation (one-hot)
      - Rating stats  : mean rating, number of ratings, rating std
      - Genre profile : fraction of ratings in each of the 18 genres
    """
    from data_loader import load_movies
    movies = load_movies()

    # --- rating statistics per user ---
    rating_stats = (ratings.groupby('UserID')['Rating']
                            .agg(mean_rating='mean',
                                 num_ratings='count',
                                 std_rating='std')
                            .fillna(0)
                            .reset_index())

    # --- genre profile: fraction of ratings that are genre X ---
    merged = ratings.merge(movies[['MovieID', 'Genres']], on='MovieID')
    genre_rows = []
    for g in ALL_GENRES:
        mask = merged['Genres'].str.contains(g, regex=False)
        g_ratings = merged[mask].groupby('UserID').size().rename(g)
        genre_rows.append(g_ratings)

    genre_df = pd.concat(genre_rows, axis=1).fillna(0)
    # normalise to fraction
    genre_df = genre_df.div(genre_df.sum(axis=1), axis=0).fillna(0)
    genre_df = genre_df.reset_index()  # UserID back as column

    # --- demographics ---
    demo = users[['UserID', 'Gender', 'Age', 'Occupation']].copy()
    demo['Gender'] = (demo['Gender'] == 'M').astype(int)   # M=1, F=0

    # one-hot encode Occupation (21 categories)
    occ_dummies = pd.get_dummies(demo['Occupation'], prefix='occ')
    demo = pd.concat([demo.drop('Occupation', axis=1), occ_dummies], axis=1)

    # --- merge everything ---
    features = (demo
                .merge(rating_stats, on='UserID')
                .merge(genre_df,     on='UserID'))

    features = features.set_index('UserID')
    return features


# ---------------------------------------------------------------------------
# Elbow + silhouette to find optimal K
# ---------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=42))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')

    ax2.plot(list(k_range), silhouettes, 'rs-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')

    fig.suptitle('Optimal K Selection', fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '07_optimal_k.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\nBest K by silhouette score: {best_k}  (score={max(silhouettes):.4f})")
    return best_k, inertias, silhouettes


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def run_kmeans(features: pd.DataFrame, n_clusters: int = 5):
    """Fit K-Means, attach cluster labels to features, return (features_with_labels, scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    features = features.copy()
    features['Cluster'] = labels

    score = silhouette_score(X, labels, sample_size=3000, random_state=42)
    print(f"K-Means  k={n_clusters}  |  Silhouette = {score:.4f}")
    return features, scaler, km


# ---------------------------------------------------------------------------
# Cluster profiling
# ---------------------------------------------------------------------------

def profile_clusters(features_labeled: pd.DataFrame, users: pd.DataFrame):
    """Print a readable summary of each cluster."""
    users_copy = users.copy()
    users_copy['AgeLabel'] = users_copy['Age'].map(AGE_MAP)
    users_copy['OccupationLabel'] = users_copy['Occupation'].map(OCCUPATION_MAP)
    users_copy['Gender'] = users_copy['Gender'].map({'M': 'Male', 'F': 'Female'})

    merged = features_labeled[['mean_rating', 'num_ratings', 'Cluster']].merge(
        users_copy[['UserID', 'Gender', 'AgeLabel', 'OccupationLabel']],
        left_index=True, right_on='UserID'
    )

    genre_cols = [g for g in ALL_GENRES if g in features_labeled.columns]

    print("\n" + "=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)
    for cid in sorted(features_labeled['Cluster'].unique()):
        subset = merged[merged['Cluster'] == cid]
        fl_sub = features_labeled[features_labeled['Cluster'] == cid]

        top_genres = fl_sub[genre_cols].mean().sort_values(ascending=False).head(3)
        top_occ = subset['OccupationLabel'].value_counts().index[0]
        top_gender = subset['Gender'].value_counts().index[0]
        top_age = subset['AgeLabel'].value_counts().index[0]

        print(f"\nCluster {cid}  ({len(subset):,} users)")
        print(f"  Avg rating   : {subset['mean_rating'].mean():.2f}")
        print(f"  Avg #ratings : {subset['num_ratings'].mean():.0f}")
        print(f"  Dominant gender : {top_gender}")
        print(f"  Dominant age    : {top_age}")
        print(f"  Dominant occ    : {top_occ}")
        print(f"  Top genres      : {', '.join(top_genres.index.tolist())}")


# ---------------------------------------------------------------------------
# Visualise clusters with PCA (2-D)
# ---------------------------------------------------------------------------

def plot_clusters_pca(features_labeled: pd.DataFrame):
    feat_cols = [c for c in features_labeled.columns if c != 'Cluster']
    X = StandardScaler().fit_transform(features_labeled[feat_cols].values)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    df_plot = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Cluster': features_labeled['Cluster'].astype(str)
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette('tab10', n_colors=df_plot['Cluster'].nunique())
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster',
                    palette=palette, alpha=0.5, s=15, ax=ax)
    ax.set_title('User Clusters (PCA 2-D Projection)', fontsize=13)
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '08_cluster_pca.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Genre heatmap per cluster
# ---------------------------------------------------------------------------

def plot_genre_heatmap(features_labeled: pd.DataFrame):
    genre_cols = [g for g in ALL_GENRES if g in features_labeled.columns]
    cluster_genre = (features_labeled.groupby('Cluster')[genre_cols]
                                     .mean()
                                     .T)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cluster_genre, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, ax=ax)
    ax.set_title('Average Genre Profile per Cluster', fontsize=13)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Genre')
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '09_genre_heatmap.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Assign a new user to the nearest cluster (cold-start helper)
# ---------------------------------------------------------------------------

def assign_cluster(new_user_features: np.ndarray, scaler: StandardScaler, km: KMeans) -> int:
    """
    Given a 1-D feature vector for a new user (same order as training features),
    return the predicted cluster ID.
    """
    x = scaler.transform(new_user_features.reshape(1, -1))
    return int(km.predict(x)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_clustering():
    print("Loading data...")
    ratings = load_ratings()
    users   = load_users()

    print("Building user feature matrix...")
    features = build_user_features(ratings, users)
    print(f"  Feature matrix shape: {features.shape}")

    # Scale for K-selection
    scaler_tmp = StandardScaler()
    X_scaled = scaler_tmp.fit_transform(features.values)

    print("\nSearching for optimal K...")
    best_k, _, _ = find_optimal_k(X_scaled)

    print(f"\nRunning K-Means with k={best_k}...")
    features_labeled, scaler, km = run_kmeans(features, n_clusters=best_k)

    profile_clusters(features_labeled, users)

    print("\nGenerating cluster visualisations...")
    plot_clusters_pca(features_labeled)
    plot_genre_heatmap(features_labeled)

    print("\nClustering complete.")
    return features_labeled, scaler, km


if __name__ == '__main__':
    run_clustering()
