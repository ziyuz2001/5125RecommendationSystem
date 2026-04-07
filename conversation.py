from __future__ import annotations

import re

from data_loader import ALL_GENRES


GENRE_ALIASES = {
    "sci fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "romcom": "Romance",
    "rom com": "Romance",
    "pixar": "Animation",
    "kids": "Children's",
    "family": "Children's",
}


def split_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks = re.split(r"\bbut\b|,|;|\.", normalized, flags=re.IGNORECASE)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    if alias == "romcom":
        return re.compile(r"(?<!\w)rom\s?coms?(?!\w)")
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")


def extract_genres(text: str) -> list[str]:
    lower_text = text.lower()
    genres: set[str] = set()

    for genre in ALL_GENRES:
        if re.search(rf"\b{re.escape(genre.lower())}\b", lower_text):
            genres.add(genre)

    for alias, genre in GENRE_ALIASES.items():
        if _alias_pattern(alias).search(lower_text):
            genres.add(genre)

    return sorted(genres)
