# Conversational Movie Recommender Final Project

This repository contains the completed final version of a conversational movie recommender built on the MovieLens 1M dataset. The final system accepts a natural-language preference statement, classifies each clause as positive or negative, extracts preference signals, and returns movie recommendations through a text-compatible recommendation pipeline with a clustering-based fallback.

## Final Project Summary

The completed project delivered four main outcomes:

- a binary clause-level text classification pipeline trained on an in-repo conversational preference dataset
- a frozen-split recommender benchmark comparing `svd`, `knn`, `content`, and `hybrid`
- persisted offline artifacts for classifier outputs, recommender benchmarking, and clustering fallback
- a Streamlit demo that exposes the conversational recommendation flow for grading and manual testing

## Completed System Components

- `text_classifier.py`
  Loads `data/conversational_polarity.csv`, trains multiple TF-IDF classifiers, selects the best model, and saves metrics, misclassification outputs, and the confusion matrix.
- `conversation.py`
  Splits free-text input into clauses and extracts structured preference signals such as positive and negative genres.
- `evaluate.py`
  Builds one frozen per-user holdout split, benchmarks the candidate recommenders on the same evaluation set, and saves model-comparison artifacts and winner metadata.
- `clustering.py`
  Runs K-Means user clustering, produces clustering visualizations, and saves artifacts used for fallback recommendation behavior.
- `conversational_recommender.py`
  Converts parsed preference signals into scored movie rankings and applies cluster priors when the selected app model is hybrid or when fallback behavior is needed.
- `app.py`
  Serves the final Streamlit interface for conversational recommendation.
- `main.py`
  Provides the offline pipeline entry points for the completed final workflow: classifier training, recommender benchmarking, and clustering artifact generation.

## Final Conversational Flow

1. The user enters a mixed preference statement such as `I like sci-fi and Pixar, but I hate horror.`
2. The input is split into short clauses.
3. Each clause is classified as `positive` or `negative`.
4. Genres and related preference signals are extracted from the classified clauses.
5. The app loads the saved recommender winner metadata and uses the best text-compatible model for recommendation.
6. If the parsed signal is too weak for primary scoring, the system falls back to cluster-based recommendations.

## Reproducing The Final Pipeline

Use Python 3.10+ and install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The repository expects the bundled MovieLens 1M files in `ml-1m/`.

### Run the entire pipeline at once

Run commands from the repository root:

```bash
python main.py
```

### Run individual stages

```bash
python main.py --step classify     # Step 1: Final clause polarity classifier training
python main.py --step benchmark    # Step 2: Final shared-split recommender benchmark
python main.py --step cluster      # Step 3: Clustering artifacts for fallback recommendations
```

The benchmark step uses one frozen per-user split and evaluates a deterministic user subset from that split so all four recommenders, including `knn`, are compared on the same practical offline workload.

### Launch the final demo application

```bash
streamlit run app.py
```

## Final Outputs

The completed offline workflow produces artifacts under `artifacts/` and visual outputs under `plots/`. Key final outputs include:

- `artifacts/classifier_model.joblib`
- `artifacts/classifier_metrics.csv`
- `artifacts/classifier_errors.csv`
- `artifacts/recommender_metrics.csv`
- `artifacts/recommender_errors.csv`
- `artifacts/recommender_selection.json`
- `artifacts/eval_split.joblib`
- `artifacts/cluster_summary.joblib`
- `artifacts/cluster_model.joblib`
- `plots/classifier_confusion_matrix.png`
- `plots/recommender_comparison.png`
- `plots/07_optimal_k.png`
- `plots/08_cluster_pca.png`
- `plots/09_genre_heatmap.png`

The saved recommender selection metadata separates the best overall benchmark performer from the best text-compatible app model. In the current completed pipeline, the benchmark artifacts identify `knn` as the overall winner and `hybrid` as the app winner.

## Verification

Run the automated test suite from the repository root:

```bash
pytest -v
```

## Rubric Coverage

- problem formulation: conversational recommendation from free-text preferences
- data preparation: MovieLens 1M data loaders plus an in-repo conversational polarity dataset
- clustering: K-Means user segmentation and saved fallback artifacts
- text feature engineering: TF-IDF clause representation for classification
- classification: multi-model binary classifier comparison with saved outputs
- recommender comparison: shared-split benchmark across `svd`, `knn`, `content`, and `hybrid`
- evaluation: `Precision@K`, `Recall@K`, and `F1@K` with saved comparison tables and plots
- error analysis: classifier and recommender error CSV artifacts
- visualization and demo: confusion matrix, recommender comparison plot, clustering plots, and the Streamlit app

## Final Notes

- The conversational app uses the best text-compatible recommender rather than the best overall recommender because the app operates for text-only users without MovieLens rating histories.
- Clustering remained part of the final deliverable as both an evaluated modeling component and a fallback recommendation path.
