# Conversational Movie Recommender

This project turns the original MovieLens 1M recommender baseline into a conversational movie recommender. A user can describe likes and dislikes in natural language, the system classifies each clause as positive or negative, extracts preference signals, and returns movie recommendations with a clustering-based fallback when the text signal is weak.

## What The Repo Does

- trains a binary clause-level text classification model on `data/conversational_polarity.csv`
- benchmarks `svd`, `knn`, `content`, and `hybrid` recommenders on one frozen MovieLens split
- saves offline artifacts for the classifier, benchmark results, and clustering fallback data
- serves a Streamlit demo for conversational recommendation

## Main Files

- `text_classifier.py`: loads the polarity dataset, compares candidate classifiers, saves the best classifier, and writes evaluation outputs
- `conversation.py`: splits user text into clauses and extracts genre-level preference signals
- `evaluate.py`: builds the frozen holdout split, benchmarks recommenders, and saves comparison artifacts
- `clustering.py`: runs K-Means clustering and saves cluster artifacts used for fallback recommendations
- `conversational_recommender.py`: scores movies from parsed preferences and applies cluster priors when needed
- `app.py`: Streamlit demo entrypoint
- `main.py`: CLI entrypoint for offline training, benchmarking, clustering, and legacy baseline steps

## Setup

Use Python 3.10+ and install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

The project expects the bundled MovieLens 1M data in `ml-1m/`.

## Offline Workflow

Train the clause classifier:

```bash
python main.py --step classify
```

Benchmark recommenders on the frozen shared split:

```bash
python main.py --step benchmark
```

The benchmark uses one frozen per-user split and evaluates a deterministic user subset from that split so all four recommenders, including `knn`, are compared on the same practical offline workload.

Generate clustering artifacts for fallback recommendations:

```bash
python main.py --step cluster
```

Launch the conversational demo:

```bash
streamlit run app.py
```

## Conversational Flow

1. The user enters a mixed preference sentence such as `I like sci-fi and Pixar, but I hate horror.`
2. `conversation.py` splits the sentence into clauses.
3. `text_classifier.py` labels each clause as `positive` or `negative`.
4. Parsed genres and keywords are sent to `conversational_recommender.py`.
5. The app uses the best text-compatible recommender winner from `artifacts/recommender_selection.json`.
6. If the parsed preference signal is too weak, the system falls back to cluster-based recommendations.

## Artifacts

Generated files are written under `artifacts/` and `plots/`. Important outputs include:

- `artifacts/classifier_model.joblib`
- `artifacts/classifier_metrics.csv`
- `artifacts/classifier_errors.csv`
- `artifacts/recommender_metrics.csv`
- `artifacts/recommender_errors.csv`
- `artifacts/recommender_selection.json`
- `artifacts/eval_split.joblib`
- `artifacts/cluster_summary.joblib`
- `plots/classifier_confusion_matrix.png`
- `plots/recommender_comparison.png`

## Tests

Run the test suite from the repo root:

```bash
pytest -v
```

## Rubric Mapping

- problem formulation: conversational recommendation from free-text preferences
- data preparation: MovieLens 1M loaders plus an in-repo conversational polarity dataset
- clustering: K-Means user segmentation with saved fallback artifacts
- text feature engineering: TF-IDF features for clause classification
- classification: multiple binary classifiers compared on one held-out split
- recommender comparison: shared-split benchmark across `svd`, `knn`, `content`, and `hybrid`
- evaluation: `Precision@K`, `Recall@K`, and `F1@K` plus saved plots and tables
- error analysis: classifier and recommender error CSV outputs
- visualization and demo: confusion matrix, recommender comparison plot, and Streamlit app

## Notes

- The overall recommender winner is stored separately from the app winner because the app can only use text-compatible recommenders.
- The clustering path remains part of the graded project as both an evaluated analysis component and a fallback strategy.
