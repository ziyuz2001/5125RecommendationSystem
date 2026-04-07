from conversation import split_clauses, extract_genres


def test_split_clauses_breaks_mixed_input():
    text = "I like sci-fi and Pixar, but I hate horror."
    clauses = split_clauses(text)

    assert clauses == ["I like sci-fi and Pixar", "I hate horror"]


def test_extract_genres_maps_known_aliases():
    genres = extract_genres("I like sci-fi and I hate rom coms")

    assert "Sci-Fi" in genres
    assert "Romance" in genres


def test_extract_genres_does_not_match_unrelated_text():
    genres = extract_genres("This is a multifamily listing")

    assert "Children's" not in genres
