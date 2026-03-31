"""
data_loader.py
Load MovieLens 1M .dat files and store them in a SQLite database.
"""

import os
import sqlite3
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml-1m', 'ml-1m')
DB_PATH  = os.path.join(os.path.dirname(__file__), 'movielens.db')


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """UserID::MovieID::Rating::Timestamp"""
    path = os.path.join(data_dir, 'ratings.dat')
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1'
    )
    df['Rating'] = df['Rating'].astype(float)
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """MovieID::Title::Genres  (genres are pipe-separated)"""
    path = os.path.join(data_dir, 'movies.dat')
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    return df


def load_users(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """UserID::Gender::Age::Occupation::Zip-code"""
    path = os.path.join(data_dir, 'users.dat')
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip'],
        encoding='latin-1'
    )
    return df


# ---------------------------------------------------------------------------
# SQLite database
# ---------------------------------------------------------------------------

def build_database(data_dir: str = DATA_DIR, db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Load all three files into a SQLite database (tables: ratings, movies, users).
    Returns an open connection.
    """
    print("Loading raw data...")
    ratings = load_ratings(data_dir)
    movies  = load_movies(data_dir)
    users   = load_users(data_dir)

    print(f"  ratings : {len(ratings):,} rows")
    print(f"  movies  : {len(movies):,} rows")
    print(f"  users   : {len(users):,} rows")

    print(f"\nWriting database → {db_path}")
    conn = sqlite3.connect(db_path)
    ratings.to_sql('ratings', conn, if_exists='replace', index=False)
    movies.to_sql('movies',   conn, if_exists='replace', index=False)
    users.to_sql('users',     conn, if_exists='replace', index=False)

    # Useful indexes for faster queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user   ON ratings(UserID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_movie  ON ratings(MovieID)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_id      ON movies(MovieID)")
    conn.commit()
    print("Database ready.\n")
    return conn


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the existing database (creates it if missing)."""
    if not os.path.exists(db_path):
        return build_database(db_path=db_path)
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------------------
# Convenience query helpers
# ---------------------------------------------------------------------------

def query(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    return pd.read_sql_query(sql, conn)


def get_all_ratings(db_path: str = DB_PATH) -> pd.DataFrame:
    return query("SELECT * FROM ratings", db_path)


def get_all_movies(db_path: str = DB_PATH) -> pd.DataFrame:
    return query("SELECT * FROM movies", db_path)


def get_all_users(db_path: str = DB_PATH) -> pd.DataFrame:
    return query("SELECT * FROM users", db_path)


# ---------------------------------------------------------------------------
# Age / Occupation label maps (from README)
# ---------------------------------------------------------------------------

AGE_MAP = {
    1:  'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}

OCCUPATION_MAP = {
    0:  'other',
    1:  'academic/educator',
    2:  'artist',
    3:  'clerical/admin',
    4:  'college/grad student',
    5:  'customer service',
    6:  'doctor/health care',
    7:  'executive/managerial',
    8:  'farmer',
    9:  'homemaker',
    10: 'K-12 student',
    11: 'lawyer',
    12: 'programmer',
    13: 'retired',
    14: 'sales/marketing',
    15: 'scientist',
    16: 'self-employed',
    17: 'technician/engineer',
    18: 'tradesman/craftsman',
    19: 'unemployed',
    20: 'writer'
}

ALL_GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


if __name__ == '__main__':
    conn = build_database()
    print(query("SELECT COUNT(*) AS total_ratings FROM ratings"))
    print(query("SELECT * FROM movies LIMIT 5"))
    print(query("SELECT * FROM users  LIMIT 5"))
