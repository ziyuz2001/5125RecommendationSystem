"""
eda.py
Exploratory Data Analysis for MovieLens 1M.
Generates and saves plots to ./plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from data_loader import (
    load_ratings, load_movies, load_users,
    AGE_MAP, OCCUPATION_MAP, ALL_GENRES
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')


def save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Rating distribution
# ---------------------------------------------------------------------------

def plot_rating_distribution(ratings: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = ratings['Rating'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
    ax.set_xlabel('Rating (stars)', fontsize=12)
    ax.set_ylabel('Number of Ratings', fontsize=12)
    ax.set_title('Rating Distribution', fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, v in zip(counts.index, counts.values):
        ax.text(i, v + 5000, f'{v:,}', ha='center', fontsize=9)
    save(fig, '01_rating_distribution.png')


# ---------------------------------------------------------------------------
# 2. Ratings per user (histogram)
# ---------------------------------------------------------------------------

def plot_ratings_per_user(ratings: pd.DataFrame):
    per_user = ratings.groupby('UserID')['Rating'].count()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_user, bins=50, color='teal', edgecolor='white')
    ax.set_xlabel('Number of Ratings per User', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('How Many Movies Has Each User Rated?', fontsize=14)
    ax.axvline(per_user.median(), color='red', linestyle='--',
               label=f'Median = {per_user.median():.0f}')
    ax.legend()
    save(fig, '02_ratings_per_user.png')


# ---------------------------------------------------------------------------
# 3. Top 20 most rated movies
# ---------------------------------------------------------------------------

def plot_top_movies(ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 20):
    counts = (ratings.groupby('MovieID')['Rating']
                     .count()
                     .reset_index(name='NumRatings')
                     .merge(movies[['MovieID', 'Title']], on='MovieID')
                     .sort_values('NumRatings', ascending=False)
                     .head(n))
    # Shorten long titles
    counts['ShortTitle'] = counts['Title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str[:40]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts['ShortTitle'][::-1], counts['NumRatings'][::-1], color='salmon')
    ax.set_xlabel('Number of Ratings', fontsize=12)
    ax.set_title(f'Top {n} Most Rated Movies', fontsize=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, '03_top_movies.png')


# ---------------------------------------------------------------------------
# 4. Genre popularity
# ---------------------------------------------------------------------------

def plot_genre_popularity(movies: pd.DataFrame):
    genre_counts = {g: movies['Genres'].str.contains(g).sum() for g in ALL_GENRES}
    genre_df = (pd.Series(genre_counts)
                  .sort_values(ascending=False)
                  .reset_index())
    genre_df.columns = ['Genre', 'Count']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(genre_df['Genre'], genre_df['Count'], color='mediumseagreen', edgecolor='white')
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    ax.set_title('Movie Count by Genre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    save(fig, '04_genre_popularity.png')


# ---------------------------------------------------------------------------
# 5. User demographics
# ---------------------------------------------------------------------------

def plot_user_demographics(users: pd.DataFrame):
    users = users.copy()
    users['AgeLabel'] = users['Age'].map(AGE_MAP)
    users['OccupationLabel'] = users['Occupation'].map(OCCUPATION_MAP)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Gender
    gender_counts = users['Gender'].value_counts()
    axes[0].pie(gender_counts, labels=gender_counts.index,
                autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'])
    axes[0].set_title('Gender Distribution')

    # Age
    age_order = [AGE_MAP[k] for k in sorted(AGE_MAP)]
    age_counts = users['AgeLabel'].value_counts().reindex(age_order).fillna(0)
    axes[1].bar(age_counts.index, age_counts.values, color='cornflowerblue', edgecolor='white')
    axes[1].set_title('Age Distribution')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Users')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Top 10 occupations
    top_occ = users['OccupationLabel'].value_counts().head(10)
    axes[2].barh(top_occ.index[::-1], top_occ.values[::-1], color='mediumorchid')
    axes[2].set_title('Top 10 Occupations')
    axes[2].set_xlabel('Users')

    fig.suptitle('User Demographics', fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, '05_user_demographics.png')


# ---------------------------------------------------------------------------
# 6. Average rating by genre
# ---------------------------------------------------------------------------

def plot_avg_rating_by_genre(ratings: pd.DataFrame, movies: pd.DataFrame):
    merged = ratings.merge(movies[['MovieID', 'Genres']], on='MovieID')
    avg_ratings = {}
    for g in ALL_GENRES:
        mask = merged['Genres'].str.contains(g)
        avg_ratings[g] = merged.loc[mask, 'Rating'].mean()

    genre_df = (pd.Series(avg_ratings)
                  .sort_values(ascending=False)
                  .reset_index())
    genre_df.columns = ['Genre', 'AvgRating']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(genre_df['Genre'], genre_df['AvgRating'],
                  color='goldenrod', edgecolor='white')
    ax.set_ylim(3.0, 4.2)
    ax.set_ylabel('Average Rating', fontsize=12)
    ax.set_title('Average Rating by Genre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)
    save(fig, '06_avg_rating_by_genre.png')


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(ratings, movies, users):
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total ratings : {len(ratings):,}")
    print(f"Unique users  : {ratings['UserID'].nunique():,}")
    print(f"Unique movies : {ratings['MovieID'].nunique():,}")
    print(f"Rating range  : {ratings['Rating'].min()} – {ratings['Rating'].max()}")
    print(f"Mean rating   : {ratings['Rating'].mean():.3f}")
    print(f"Sparsity      : {1 - len(ratings) / (ratings['UserID'].nunique() * ratings['MovieID'].nunique()):.4%}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda():
    print("Loading data...")
    ratings = load_ratings()
    movies  = load_movies()
    users   = load_users()

    print_summary(ratings, movies, users)

    print("\nGenerating plots...")
    plot_rating_distribution(ratings)
    plot_ratings_per_user(ratings)
    plot_top_movies(ratings, movies)
    plot_genre_popularity(movies)
    plot_user_demographics(users)
    plot_avg_rating_by_genre(ratings, movies)
    print("\nEDA complete. All plots saved to ./plots/")


if __name__ == '__main__':
    run_eda()
