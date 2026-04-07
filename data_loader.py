"""
data_loader.py
Load MovieLens 1M raw files and expose shared label constants.
"""

import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ml-1m")


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load ratings.dat."""
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    df["Rating"] = df["Rating"].astype(float)
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load movies.dat."""
    path = os.path.join(data_dir, "movies.dat")
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )


def load_users(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load users.dat."""
    path = os.path.join(data_dir, "users.dat")
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip"],
        encoding="latin-1",
    )


AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}


OCCUPATION_MAP = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
