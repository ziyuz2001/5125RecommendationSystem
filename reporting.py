"""
reporting.py
Generate lightweight markdown summaries from saved artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from artifact_store import artifact_path, load_csv, load_json, save_text


def _try_load_csv(name: str) -> pd.DataFrame | None:
    path = artifact_path(name)
    if not path.exists():
        return None
    return load_csv(name)


def _try_load_json(name: str) -> dict | None:
    path = artifact_path(name)
    if not path.exists():
        return None
    return load_json(name)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _classifier_summary_lines() -> list[str]:
    metrics_df = _try_load_csv("classifier_metrics.csv")
    if metrics_df is None or metrics_df.empty:
        return ["## Classification", "", "Classifier results have not been generated yet."]

    best_row = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
    lines = [
        "## Classification",
        "",
        f"- Best classifier: `{best_row['model_name']}`",
        f"- Accuracy: `{_format_float(best_row['accuracy'])}`",
        f"- Macro F1: `{_format_float(best_row['macro_f1'])}`",
        f"- Macro Precision / Recall: `{_format_float(best_row['precision'])}` / `{_format_float(best_row['recall'])}`",
    ]
    return lines


def _recommender_summary_lines() -> list[str]:
    metrics_df = _try_load_csv("recommender_metrics.csv")
    selection = _try_load_json("recommender_selection.json")
    if metrics_df is None or metrics_df.empty:
        return ["## Recommender Evaluation", "", "Recommender benchmark results have not been generated yet."]

    best_row = metrics_df.sort_values("f1_at_k", ascending=False).iloc[0]
    lines = [
        "## Recommender Evaluation",
        "",
        f"- Best overall recommender: `{best_row['model_name']}`",
        f"- Precision@{int(best_row['k'])}: `{_format_float(best_row['precision_at_k'])}`",
        f"- Recall@{int(best_row['k'])}: `{_format_float(best_row['recall_at_k'])}`",
        f"- F1@{int(best_row['k'])}: `{_format_float(best_row['f1_at_k'])}`",
        f"- Users evaluated: `{int(best_row['users_evaluated'])}`",
    ]
    if selection:
        app_winner = selection.get("app_winner")
        if app_winner:
            lines.append(f"- Best text-compatible app model: `{app_winner}`")
    return lines


def _clustering_summary_lines() -> list[str]:
    metrics_df = _try_load_csv("clustering_metrics.csv")
    size_df = _try_load_csv("cluster_sizes.csv")
    if metrics_df is None or metrics_df.empty:
        return ["## Clustering", "", "Clustering results have not been generated yet."]

    selected = metrics_df.loc[metrics_df["is_selected"] == True]  # noqa: E712
    selected_row = selected.iloc[0] if not selected.empty else metrics_df.sort_values("silhouette", ascending=False).iloc[0]

    lines = [
        "## Clustering",
        "",
        f"- Selected K: `{int(selected_row['k'])}`",
        f"- Selected silhouette score: `{_format_float(selected_row['silhouette'])}`",
        f"- Selected inertia: `{_format_float(selected_row['inertia'])}`",
    ]

    if size_df is not None and not size_df.empty:
        largest = size_df.sort_values("user_count", ascending=False).iloc[0]
        lines.append(
            f"- Largest cluster: `Cluster {int(largest['cluster_id'])}` with `{int(largest['user_count'])}` users"
        )
    return lines


def _classify_text_error(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in [" but ", " although ", " however "]):
        return "mixed-sentiment phrasing"
    if any(token in lowered for token in ["not", "no ", "don't", "hate", "avoid"]):
        return "negation cue"
    if any(token in lowered for token in ["light", "fun", "something", "kind of"]):
        return "vague preference wording"
    return "short ambiguous clause"


def _classifier_error_lines() -> list[str]:
    errors_df = _try_load_csv("classifier_errors.csv")
    if errors_df is None:
        return ["### Classification", "", "- No classifier error file has been generated yet."]
    if errors_df.empty:
        return ["### Classification", "", "- No classifier misclassifications were recorded on the held-out split."]

    categorized = errors_df.copy()
    categorized["error_pattern"] = categorized["text"].astype(str).map(_classify_text_error)
    top_pattern = categorized["error_pattern"].value_counts().idxmax()
    example_row = categorized.iloc[0]

    return [
        "### Classification",
        "",
        f"- Held-out misclassifications: `{len(categorized)}`",
        f"- Most common error pattern: `{top_pattern}`",
        f"- Example error: predicted `{example_row['predicted']}` for `{example_row['text']}` even though the true label was `{example_row['actual']}`",
    ]


def _recommender_error_lines() -> list[str]:
    errors_df = _try_load_csv("recommender_errors.csv")
    if errors_df is None or errors_df.empty:
        return ["### Recommender", "", "- No recommender error file has been generated yet."]

    mae_by_model = (
        errors_df.groupby("model_name")["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    worst_model = mae_by_model.iloc[0]
    top_genres = (
        errors_df.assign(PrimaryGenre=errors_df["Genres"].fillna("").str.split("|").str[0].replace("", "Unknown"))
        ["PrimaryGenre"]
        .value_counts()
        .head(3)
        .index.tolist()
    )

    return [
        "### Recommender",
        "",
        f"- Highest mean absolute error among saved top errors: `{worst_model['model_name']}` at `{_format_float(worst_model['abs_error'])}`",
        f"- Genres appearing most often in large-error cases: `{', '.join(top_genres)}`",
        "- Current error file is biased toward worst-case examples because it stores the largest absolute-error rows per model.",
    ]


def _clustering_error_lines() -> list[str]:
    metrics_df = _try_load_csv("clustering_metrics.csv")
    size_df = _try_load_csv("cluster_sizes.csv")
    if metrics_df is None or size_df is None or metrics_df.empty or size_df.empty:
        return ["### Clustering", "", "- Clustering diagnostic files are not available yet."]

    selected = metrics_df.loc[metrics_df["is_selected"] == True]  # noqa: E712
    selected_row = selected.iloc[0] if not selected.empty else metrics_df.sort_values("silhouette", ascending=False).iloc[0]
    imbalance = int(size_df["user_count"].max()) - int(size_df["user_count"].min())
    return [
        "### Clustering",
        "",
        f"- Selected silhouette score is `{_format_float(selected_row['silhouette'])}`, which indicates usable but not perfectly separated user segments.",
        f"- Cluster size spread is `{imbalance}` users between the largest and smallest cluster, so the fallback path is somewhat imbalanced.",
        "- Clustering is used as a fallback mechanism, so overlap between neighboring user groups is acceptable but should be acknowledged in the report.",
    ]


def write_results_summary() -> Path:
    lines = [
        "# Results Summary",
        "",
        "This file summarizes the latest saved offline results for the final conversational movie recommender pipeline.",
        "",
        *_classifier_summary_lines(),
        "",
        *_recommender_summary_lines(),
        "",
        *_clustering_summary_lines(),
        "",
        "## Visual Outputs",
        "",
        "- `plots/classifier_confusion_matrix.png`",
        "- `plots/classifier_model_comparison.png`",
        "- `plots/recommender_comparison.png`",
        "- `plots/recommender_metrics_by_k.png`",
        "- `plots/optimal_k.png`",
        "- `plots/cluster_pca.png`",
        "- `plots/genre_heatmap.png`",
        "- `plots/cluster_sizes.png`",
    ]
    return save_text("results_summary.md", "\n".join(lines) + "\n")


def write_error_analysis() -> Path:
    lines = [
        "# Error Analysis",
        "",
        "This file records the main observed failure modes from the saved classifier, recommender, and clustering outputs.",
        "",
        *_classifier_error_lines(),
        "",
        *_recommender_error_lines(),
        "",
        *_clustering_error_lines(),
    ]
    return save_text("error_analysis.md", "\n".join(lines) + "\n")


def refresh_reports() -> tuple[Path, Path]:
    return write_results_summary(), write_error_analysis()

