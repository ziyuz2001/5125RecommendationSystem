"""
main.py
CLI entrypoint for the completed conversational movie recommender workflow.

Usage:
    python main.py                    # run the final offline pipeline
    python main.py --step classify    # train the clause classifier
    python main.py --step benchmark   # benchmark recommenders and save artifacts
    python main.py --step cluster     # generate clustering fallback artifacts
"""

import argparse

from clustering import run_clustering_and_save_artifacts
from evaluate import run_evaluation
from text_classifier import run_classification_pipeline


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def step_classify():
    section("STEP 1 - Train Clause Polarity Classifier")
    run_classification_pipeline()


def step_benchmark():
    section("STEP 2 - Benchmark Recommenders and Save Artifacts")
    run_evaluation()


def step_clustering():
    section("STEP 3 - Generate Clustering Fallback Artifacts")
    return run_clustering_and_save_artifacts()


def step_all():
    step_classify()
    step_benchmark()
    step_clustering()


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Movie Recommender final workflow"
    )
    parser.add_argument(
        "--step",
        choices=["all", "classify", "benchmark", "cluster"],
        default="all",
        help="Which final pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "all":
        step_all()
    elif args.step == "classify":
        step_classify()
    elif args.step == "benchmark":
        step_benchmark()
    elif args.step == "cluster":
        step_clustering()

    print("\nDone. Final outputs are in ./artifacts/ and ./plots/")


if __name__ == "__main__":
    main()
