# Movie Recommendation System — MovieLens 1M
> **Final Project | Data Science**
> Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) by GroupLens Research

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Requirements & Installation](#3-requirements--installation)
4. [Dataset Description](#4-dataset-description)
5. [How to Run](#5-how-to-run)
6. [Stage 0 — Data Loading & Database](#6-stage-0--data-loading--database-data_loaderpy)
7. [Stage 1 — Exploratory Data Analysis](#7-stage-1--exploratory-data-analysis-edapy)
8. [Stage 2 — User Clustering](#8-stage-2--user-clustering-clusteringpy)
9. [Stage 3 — Recommendation Models](#9-stage-3--recommendation-models-recommenderpy)
10. [Stage 4 — Model Evaluation](#10-stage-4--model-evaluation-evaluatepy)
11. [Results Summary](#11-results-summary)
12. [Citation](#12-citation)

---

## 1. Project Overview

This project builds a **Content-Based and Collaborative Filtering Recommendation System (CRS)** for movies using the MovieLens 1M dataset. The system predicts personalised movie ratings for users and generates a **Top-N recommendation list** using three strategies:

| Strategy | Description |
|----------|-------------|
| **Collaborative Filtering (CF)** | Learns from user rating patterns using SVD and KNN |
| **Content-Based Filtering** | Recommends movies with similar genres using TF-IDF + cosine similarity |
| **Hybrid** | Combines CF and content scores with a weighted formula |
| **Cold-Start Fallback** | Uses K-Means user clusters to recommend movies for brand-new users |

The project also satisfies the **mandatory classification/clustering requirement** by applying **K-Means clustering** on user demographic and behavioural features to segment users into meaningful groups.

---

## 2. Project Structure

```
movie-recommender/
│
├── data_loader.py          # Stage 0: Load .dat files, build SQLite database
├── eda.py                  # Stage 1: Exploratory Data Analysis + visualisations
├── clustering.py           # Stage 2: K-Means user clustering
├── recommender.py          # Stage 3: SVD / KNN / Content-Based / Hybrid models
├── evaluate.py             # Stage 4: RMSE, MAE, Precision@K, Recall@K evaluation
├── main.py                 # Pipeline runner (runs all stages or individual steps)
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── movielens.db            # SQLite database (auto-generated on first run)
└── plots/                  # All output charts (auto-generated)
    ├── 01_rating_distribution.png
    ├── 02_ratings_per_user.png
    ├── 03_top_movies.png
    ├── 04_genre_popularity.png
    ├── 05_user_demographics.png
    ├── 06_avg_rating_by_genre.png
    ├── 07_optimal_k.png
    ├── 08_cluster_pca.png
    ├── 09_genre_heatmap.png
    ├── 10_cv_rmse_mae.png
    └── 11_precision_recall.png
```

> **Note:** The raw data folder `ml-1m/` should be placed one level **above** the project folder:
> ```
> Downloads/
> ├── ml-1m/
> │   └── ml-1m/
> │       ├── ratings.dat
> │       ├── movies.dat
> │       └── users.dat
> └── movie-recommender/   ← project folder
> ```

---

## 3. Requirements & Installation

### Python Version
- Python **3.8 or above** (tested on 3.12)

### Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

### Install

```bash
pip install -r requirements.txt
```

> `scikit-surprise` is **not required** — collaborative filtering is implemented directly using `scipy.sparse.linalg.svds` and `sklearn.neighbors.NearestNeighbors`, making the project fully cross-platform without needing a C++ compiler.

---

## 4. Dataset Description

**Source:** GroupLens Research — [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

The dataset contains anonymous movie ratings collected from the MovieLens website in 2000.

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | 1,000,209 | Star ratings from 1–5 (whole numbers only) |
| `movies.dat` | `MovieID::Title::Genres` | ~3,900 | Movie titles and pipe-separated genre tags |
| `users.dat` | `UserID::Gender::Age::Occupation::Zip` | 6,040 | Demographic information per user |

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Ratings | 1,000,209 |
| Unique Users | 6,040 |
| Unique Movies | 3,706 |
| Rating Scale | 1 – 5 stars |
| Mean Rating | 3.582 |
| Matrix Sparsity | ~95.5% |
| Min Ratings per User | 20 |

### Genre Categories (18 total)
`Action · Adventure · Animation · Children's · Comedy · Crime · Documentary · Drama · Fantasy · Film-Noir · Horror · Musical · Mystery · Romance · Sci-Fi · Thriller · War · Western`

### Age Groups
`Under 18 · 18-24 · 25-34 · 35-44 · 45-49 · 50-55 · 56+`

### Occupation Categories (21 total)
`other · academic/educator · artist · clerical/admin · college/grad student · customer service · doctor/health care · executive/managerial · farmer · homemaker · K-12 student · lawyer · programmer · retired · sales/marketing · scientist · self-employed · technician/engineer · tradesman/craftsman · unemployed · writer`

---

## 5. How to Run

### Run the entire pipeline at once

```bash
cd movie-recommender
python main.py
```

### Run individual stages

```bash
python main.py --step db           # Stage 0: Build SQLite database only
python main.py --step eda          # Stage 1: Exploratory Data Analysis only
python main.py --step cluster      # Stage 2: User Clustering only
python main.py --step recommend    # Stage 3: Recommendation demo (user 1, top-10)
python main.py --step evaluate     # Stage 4: Model Evaluation only
```

### Get recommendations for a specific user

```bash
python main.py --step recommend --user 42 --n 10
# --user  : UserID (1–6040)
# --n     : number of recommendations to return
```

---

## 6. Stage 0 — Data Loading & Database (`data_loader.py`)

### Purpose

This is the **foundation stage** of the entire pipeline. Before any modelling can happen, raw data must be loaded and organised into a structured, queryable format.

### What it does

1. **Reads the three `.dat` files** — each line is `::` separated (not standard CSV), so a custom parser is used with `pandas.read_csv(sep='::')`.
2. **Stores all three tables into a SQLite database** (`movielens.db`) using `sqlite3`.
3. **Creates indexes** on frequently queried columns (`UserID`, `MovieID`) for fast lookups.
4. **Provides helper functions** (`query()`, `get_all_ratings()`, etc.) that any other module can call to access data without re-parsing the raw files.

### Why SQLite?

- Avoids re-parsing 1M+ rows every time the code runs.
- Allows SQL queries for analysis (e.g. joins across ratings, movies, users).
- Lightweight — no database server required, just a single `.db` file.
- Reusable across sessions: once built, the database persists on disk.

### Key functions

| Function | Description |
|----------|-------------|
| `load_ratings()` | Read `ratings.dat` → DataFrame |
| `load_movies()` | Read `movies.dat` → DataFrame |
| `load_users()` | Read `users.dat` → DataFrame |
| `build_database()` | Load all three files and write to `movielens.db` |
| `get_connection()` | Return DB connection (builds DB if not yet created) |
| `query(sql)` | Run any SQL query and return a DataFrame |

### Constants defined here (used by all other modules)

- `AGE_MAP` — maps numeric age codes to readable labels
- `OCCUPATION_MAP` — maps numeric occupation codes to readable labels
- `ALL_GENRES` — list of 18 genre strings

---

## 7. Stage 1 — Exploratory Data Analysis (`eda.py`)

### Purpose

Before building any model, it is essential to **understand the structure, distribution, and quality of the data**. EDA reveals patterns and potential issues that inform modelling decisions.

### What it does

Generates **6 visualisation plots** saved to `./plots/`:

| Plot | File | What it shows |
|------|------|---------------|
| Rating Distribution | `01_rating_distribution.png` | Bar chart of how many 1★–5★ ratings exist |
| Ratings per User | `02_ratings_per_user.png` | Histogram of rating activity per user; median line |
| Top 20 Most Rated Movies | `03_top_movies.png` | Movies that received the most ratings |
| Genre Popularity | `04_genre_popularity.png` | Number of movies per genre |
| User Demographics | `05_user_demographics.png` | Gender (pie), Age (bar), Top-10 Occupations (bar) |
| Average Rating by Genre | `06_avg_rating_by_genre.png` | Which genres are rated highest on average |

Also prints a **dataset summary** to the console:
```
Total ratings : 1,000,209
Unique users  : 6,040
Unique movies : 3,706
Rating range  : 1.0 – 5.0
Mean rating   : 3.582
Sparsity      : 95.53%
```

### Why sparsity matters

With 6,040 users and 3,706 movies, the rating matrix has ~22.4 million possible entries. Only ~1 million are filled — a **95.5% sparsity** rate. This confirms that most users have only seen a tiny fraction of all movies, which is exactly the problem collaborative filtering is designed to solve.

### Key findings from EDA

- Most ratings are 3★ or 4★ — users tend to rate movies they enjoyed.
- Drama and Comedy are the most abundant genres.
- Film-Noir and War movies receive the highest average ratings despite being less common.
- The majority of users are male, aged 25–34, working in tech or education.

---

## 8. Stage 2 — User Clustering (`clustering.py`)

### Purpose

This stage fulfils the **mandatory classification/clustering requirement**. Users are grouped into segments based on both their **demographics** and **rating behaviour**, enabling:

1. **Audience insight** — discovering meaningful user segments (e.g. "young female romance fans", "retired drama enthusiasts")
2. **Cold-start recommendations** — when a new user has no rating history, assign them to a cluster and recommend that cluster's top-rated movies

### Feature Engineering

Each user is represented as a feature vector combining:

| Feature Group | Features | Details |
|---------------|----------|---------|
| Demographics | Gender | Binary encoded (M=1, F=0) |
| Demographics | Age | Numeric (7 age groups) |
| Demographics | Occupation | One-hot encoded (21 categories) |
| Rating behaviour | Mean rating | Average star rating given |
| Rating behaviour | Number of ratings | How active the user is |
| Rating behaviour | Std of ratings | How varied their ratings are |
| Genre profile | 18 genre fractions | Fraction of ratings that fall into each genre |

**Total features per user: 44**

All features are **standardised** using `StandardScaler` before clustering to prevent high-variance features from dominating.

### Finding the Optimal Number of Clusters (K)

Two methods are used together:

- **Elbow Method** — plot inertia (within-cluster sum of squares) vs K. Look for the "elbow" where improvement slows.
- **Silhouette Score** — measures how well-separated clusters are. Ranges from -1 (bad) to +1 (perfect). The K with the highest silhouette score is selected automatically.

Plot saved as `07_optimal_k.png`.

### K-Means Clustering

Once the best K is selected, `KMeans` from `scikit-learn` is fitted on the standardised feature matrix. Each user is assigned a cluster label.

### Cluster Profiling

For each cluster, the system prints:
- Number of users in the cluster
- Average rating and number of ratings
- Dominant gender, age group, and occupation
- Top 3 preferred genres

**Example output:**
```
Cluster 1  (1,083 users)
  Avg rating       : 3.67
  Avg #ratings     : 155
  Dominant gender  : Female
  Dominant age     : 25-34
  Dominant occ     : college/grad student
  Top genres       : Comedy, Drama, Romance
```

### Visualisations

| Plot | File | Description |
|------|------|-------------|
| PCA scatter | `08_cluster_pca.png` | 2-D projection of all 6040 users, coloured by cluster |
| Genre heatmap | `09_genre_heatmap.png` | Average genre preference of each cluster |

### Key functions

| Function | Description |
|----------|-------------|
| `build_user_features()` | Build the 44-feature matrix (one row per user) |
| `find_optimal_k()` | Elbow + Silhouette plots, returns best K |
| `run_kmeans()` | Fit K-Means, attach cluster labels, return Silhouette score |
| `profile_clusters()` | Print human-readable cluster summary |
| `assign_cluster()` | Predict which cluster a new user belongs to |

---

## 9. Stage 3 — Recommendation Models (`recommender.py`)

### Purpose

This is the **core stage** — building, training, and using the actual recommendation models. Four recommendation strategies are implemented.

---

### Strategy 1 — Collaborative Filtering: SVD (`SVDRecommender`)

**How it works:**

Collaborative Filtering is based on the idea that *users who agreed in the past tend to agree in the future*. SVD (Singular Value Decomposition) decomposes the sparse user-item rating matrix into three matrices:

```
R ≈ U × Σ × Vᵀ
```

- **U** — user latent factor matrix (each user as a vector of hidden preferences)
- **Σ** — diagonal matrix of singular values (importance of each factor)
- **Vᵀ** — movie latent factor matrix (each movie as a vector of hidden attributes)

Missing ratings are filled with the user's mean rating before decomposition. Predictions are clipped to the [1.0, 5.0] range.

**Implementation:** `scipy.sparse.linalg.svds` with configurable number of latent factors (`n_factors=50` by default).

**Key methods:**

| Method | Description |
|--------|-------------|
| `fit(ratings)` | Build rating matrix, run SVD decomposition |
| `predict(user_id, movie_id)` | Predict a single rating |
| `top_n(user_id, movies, rated_ids, n)` | Return top-N unseen movie recommendations |

---

### Strategy 2 — Collaborative Filtering: KNN (`KNNRecommender`)

**How it works:**

KNN-based CF finds the K most similar users (user-based) or movies (item-based) using cosine similarity, then predicts ratings as a weighted average of their ratings. Mean-centring is applied to remove individual rating bias.

**Implementation:** `sklearn.neighbors.NearestNeighbors` with cosine distance metric.

**Parameters:**
- `k=40` — number of neighbours
- `user_based=True/False` — switch between user-based and item-based

---

### Strategy 3 — Content-Based Filtering

**How it works:**

Each movie is represented as a **TF-IDF vector** built from its genre string (e.g. `"Action Adventure Sci-Fi"`). A user's profile is the **weighted average** of the TF-IDF vectors of movies they have rated, weighted by their actual ratings. Unseen movies are then ranked by **cosine similarity** to this user profile.

This approach does not rely on other users' data — it purely uses the content of movies the target user has already seen.

**Key functions:**

| Function | Description |
|----------|-------------|
| `build_content_model(movies)` | Build TF-IDF matrix for all movies |
| `content_similar_movies(movie_id, ...)` | Find N most genre-similar movies to a given movie |
| `top_n_content(user_id, ...)` | Build user profile and rank unseen movies |

---

### Strategy 4 — Hybrid Recommender

**How it works:**

Combines the SVD CF score and the content-based similarity score using a weighted formula:

```
Hybrid Score = α × CF_score_normalised + (1 − α) × Content_similarity
```

- **α = 0.7** by default — CF contributes 70%, content similarity 30%
- CF scores are min-max normalised to [0, 1] before combining
- This balances the accuracy of CF with the genre-awareness of content filtering

**Why hybrid?**
- Pure CF struggles when a user has rated very few movies
- Pure content-based is too narrow (always recommends similar genres)
- Hybrid reduces both weaknesses

---

### Strategy 5 — Cold-Start Fallback (Cluster-based)

**How it works:**

For **brand-new users** with no rating history, neither CF nor content-based filtering can work. Instead:

1. The new user is assigned to the nearest K-Means cluster based on their demographics
2. The system recommends the top-rated movies among all users in that cluster (minimum 10 ratings required to filter noise)

This connects the **clustering stage** directly to the **recommendation pipeline**.

---

## 10. Stage 4 — Model Evaluation (`evaluate.py`)

### Purpose

A recommendation system is only useful if it can be measured objectively. This stage evaluates all models using standard metrics.

---

### Evaluation 1 — Cross-Validation: RMSE & MAE

**Method:** 5-fold cross-validation on the full ratings dataset.

For each fold, the model is trained on 80% of the data and tested on the remaining 20%. Rating predictions are compared to ground-truth ratings.

**Metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **RMSE** | √(Σ(pred − true)²/n) | Penalises large errors more; lower = better |
| **MAE** | Σ\|pred − true\|/n | Average absolute error; lower = better |

Models tested: `SVD (k=20)`, `SVD (k=50)`, `SVD (k=100)`

Plot saved as `10_cv_rmse_mae.png`.

---

### Evaluation 2 — Precision@K and Recall@K

**Method:** Train SVD on 80% of each user's ratings, recommend Top-K movies, check against the held-out 20%.

A movie is considered **relevant** if the user's true rating ≥ 3.5 stars.

**Metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision@K** | (Relevant in Top-K) / K | What fraction of recommendations are actually good? |
| **Recall@K** | (Relevant in Top-K) / (All relevant) | What fraction of good movies were found? |
| **F1@K** | 2 × P × R / (P + R) | Harmonic mean of Precision and Recall |

Evaluated at K = 5, 10, 20.

Plot saved as `11_precision_recall.png`.

---

### Evaluation 3 — Clustering: Silhouette Score

The quality of the K-Means clustering is measured using the **Silhouette Score** for K = 2 to 8:

- Score close to **+1** → clusters are dense and well-separated
- Score close to **0** → clusters overlap
- Score close to **-1** → samples may be in the wrong cluster

This helps validate that the user segments are meaningful, not arbitrary.

---

## 11. Results Summary

### Clustering (K=10)

| Cluster | Users | Dominant Profile | Top Genres |
|---------|-------|-----------------|------------|
| 0 | 236 | Male, 25-34, Doctor | Drama, Comedy, Action |
| 1 | 1,083 | Female, 25-34, Student | Comedy, Drama, Romance |
| 2 | 195 | Male, Under 18, K-12 | Comedy, Drama, Action |
| 8 | 1,589 | Male, 25-34, Executive | Drama, Comedy, Thriller |
| 9 | 142 | Male, 56+, Retired | Drama, Comedy, Action |

### Recommendation Sample (User 1)

**SVD Top-5:**
| Movie | Predicted Rating |
|-------|----------------|
| Forrest Gump (1994) | 4.44 |
| Braveheart (1995) | 4.42 |
| Jurassic Park (1993) | 4.34 |
| Lion King, The (1994) | 4.32 |
| Dances with Wolves (1990) | 4.31 |

**Hybrid Top-5:**
| Movie | CF Score | Content Score | Hybrid Score |
|-------|----------|---------------|--------------|
| Lion King, The (1994) | 4.32 | 0.70 | 0.850 |
| Little Mermaid, The (1989) | 4.29 | 0.74 | 0.847 |
| Jungle Book, The (1967) | 4.23 | 0.75 | 0.821 |
| Lady and the Tramp (1955) | 4.23 | 0.74 | 0.817 |
| Boys Don't Cry (1999) | 4.28 | 0.61 | 0.807 |

---

## 12. Citation

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
> DOI: http://dx.doi.org/10.1145/2827872

---

*Built with Python · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn*
