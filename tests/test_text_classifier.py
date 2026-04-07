from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from text_classifier import (
    build_candidates,
    fit_and_save_best_classifier,
    load_clause_dataset,
    train_classifier_candidates,
)


def test_conversational_dataset_shape_and_labels():
    path = Path("data/conversational_polarity.csv")
    assert path.exists()

    df = pd.read_csv(path)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"]) == {"positive", "negative"}
    assert len(df) >= 200
    assert df["text"].str.len().min() > 0


def test_clause_dataset_loads_with_expected_columns():
    df = load_clause_dataset()

    assert list(df.columns) == ["text", "label"]


def test_train_classifier_candidates_returns_three_models():
    df = load_clause_dataset()

    result = train_classifier_candidates(df, test_size=0.2, random_state=42)

    model_names = set(result["metrics"]["model_name"])
    assert model_names == {"logistic_regression", "linear_svm", "multinomial_nb"}


def test_train_classifier_candidates_contract_is_sorted_and_consistent():
    df = load_clause_dataset()

    result = train_classifier_candidates(df, test_size=0.2, random_state=42)

    expected_model_names = {"logistic_regression", "linear_svm", "multinomial_nb"}
    assert set(result["models"]) == expected_model_names

    metrics = result["metrics"]
    assert metrics["macro_f1"].tolist() == sorted(metrics["macro_f1"], reverse=True)
    assert result["best_model_name"] == metrics.iloc[0]["model_name"]
    for model in result["models"].values():
        predictions = model.predict(result["x_test"])
        assert len(predictions) == len(result["x_test"])


class _RecordingEstimator(BaseEstimator):
    def __init__(self):
        self.fit_size = None

    def fit(self, x, y):
        self.fit_size = len(x)
        return self

    def predict(self, x):
        return ["positive"] * len(x)


def test_fit_and_save_best_classifier_refits_best_model_on_full_dataset(monkeypatch):
    df = pd.DataFrame(
        {
            "text": ["one", "two", "three", "four", "five", "six"],
            "label": ["positive", "positive", "negative", "negative", "positive", "negative"],
        }
    )
    split_model = _RecordingEstimator().fit(df["text"].iloc[:2], df["label"].iloc[:2])
    saved = {}

    monkeypatch.setattr("text_classifier.load_clause_dataset", lambda: df)
    monkeypatch.setattr(
        "text_classifier.train_classifier_candidates",
        lambda frame, test_size=0.2, random_state=42: {
            "models": {"linear_svm": split_model},
            "metrics": pd.DataFrame(
                [
                    {
                        "model_name": "linear_svm",
                        "accuracy": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "macro_f1": 1.0,
                    }
                ]
            ),
            "best_model_name": "linear_svm",
            "x_test": frame["text"].iloc[:2],
            "y_test": frame["label"].iloc[:2],
        },
    )
    monkeypatch.setattr(
        "text_classifier.build_candidates",
        lambda: {"linear_svm": _RecordingEstimator()},
    )
    monkeypatch.setattr(
        "text_classifier.save_joblib",
        lambda name, obj: saved.update({"name": name, "obj": obj}),
    )
    monkeypatch.setattr("text_classifier.save_csv", lambda name, df: None)

    result = fit_and_save_best_classifier()

    assert saved["name"] == "classifier_model.joblib"
    assert saved["obj"].fit_size == len(df)
    assert result["models"]["linear_svm"] is saved["obj"]
    assert result["models"]["linear_svm"].fit_size == len(df)
