from conversation import split_clauses, extract_genres, parse_preferences


class FakeClassifier:
    def predict(self, texts):
        return ["positive", "negative"]


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


def test_parse_preferences_groups_positive_and_negative_signals():
    result = parse_preferences(
        "I like sci-fi and Pixar, but I hate horror",
        classifier=FakeClassifier(),
    )

    assert "Sci-Fi" in result["positive_genres"]
    assert "Animation" in result["positive_genres"]
    assert "Horror" in result["negative_genres"]


class EmptyTrackingClassifier:
    def __init__(self):
        self.called_with = None

    def predict(self, texts):
        self.called_with = list(texts)
        return []


def test_parse_preferences_returns_empty_structure_for_blank_input():
    classifier = EmptyTrackingClassifier()

    result = parse_preferences("   ... ,, ; ", classifier=classifier)

    assert classifier.called_with is None
    assert result == {
        "clauses": [],
        "positive_genres": [],
        "negative_genres": [],
        "positive_keywords": [],
        "negative_keywords": [],
    }


class WrongLengthClassifier:
    def predict(self, texts):
        return ["positive"]


def test_parse_preferences_raises_on_label_count_mismatch():
    try:
        parse_preferences("I like sci-fi and Pixar, but I hate horror", classifier=WrongLengthClassifier())
    except ValueError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched label count")
