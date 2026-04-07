from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from artifact_store import save_csv, save_joblib


DATA_PATH = Path(__file__).resolve().parent / "data" / "conversational_polarity.csv"
BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"


def load_clause_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ["text", "label"]].dropna().reset_index(drop=True)
    return df


def build_candidates() -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("model", LinearSVC()),
            ]
        ),
        "multinomial_nb": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("model", MultinomialNB()),
            ]
        ),
    }


def train_classifier_candidates(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    candidates = build_candidates()
    trained_models: dict[str, Pipeline] = {}
    metrics_rows: list[dict[str, object]] = []

    for model_name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        trained_models[model_name] = pipeline
        metrics_rows.append(
            {
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        by="macro_f1", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model_name"])

    return {
        "models": trained_models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "x_test": x_test,
        "y_test": y_test,
    }


def fit_and_save_best_classifier() -> dict[str, object]:
    df = load_clause_dataset()
    result = train_classifier_candidates(df)
    best_model_name = result["best_model_name"]
    result["heldout_models"] = dict(result["models"])
    best_model = build_candidates()[best_model_name]
    best_model.fit(df["text"], df["label"])
    result["models"][best_model_name] = best_model
    save_joblib("classifier_model.joblib", best_model)
    save_csv("classifier_metrics.csv", result["metrics"])
    return result


def save_confusion_plot(y_true, y_pred) -> Path:
    labels = ["positive", "negative"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    PLOTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Classifier Confusion Matrix")
    fig.tight_layout()

    plot_path = PLOTS_DIR / "classifier_confusion_matrix.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def run_classification_pipeline() -> dict[str, object]:
    result = fit_and_save_best_classifier()
    best_name = result["best_model_name"]
    heldout_model = result["heldout_models"][best_name]
    preds = heldout_model.predict(result["x_test"])
    x_test = result["x_test"].reset_index(drop=True)
    y_test = result["y_test"].reset_index(drop=True)
    errors = pd.DataFrame(
        {
            "text": x_test,
            "actual": y_test,
            "predicted": preds,
        }
    )
    errors = errors.loc[errors["actual"] != errors["predicted"]].reset_index(drop=True)
    save_csv("classifier_errors.csv", errors)
    save_confusion_plot(y_test, preds)
    return result
