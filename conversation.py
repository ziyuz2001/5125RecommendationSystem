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


def parse_preferences(text: str, classifier) -> dict[str, object]:
    clauses = split_clauses(text)
    parsed: dict[str, object] = {
        "clauses": [],
        "positive_genres": [],
        "negative_genres": [],
        "positive_keywords": [],
        "negative_keywords": [],
    }

    if not clauses:
        return parsed

    labels = classifier.predict(clauses)
    if len(labels) != len(clauses):
        raise ValueError(
            f"classifier.predict() returned {len(labels)} labels for {len(clauses)} clauses"
        )

    positive_genres: set[str] = set()
    negative_genres: set[str] = set()

    for clause, label in zip(clauses, labels):
        genres = extract_genres(clause)
        parsed["clauses"].append({"text": clause, "label": label, "genres": genres})

        if label == "positive":
            positive_genres.update(genres)
        elif label == "negative":
            negative_genres.update(genres)

    parsed["positive_genres"] = sorted(positive_genres)
    parsed["negative_genres"] = sorted(negative_genres)
    return parsed
