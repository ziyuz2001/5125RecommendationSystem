import pandas as pd
import streamlit as st

from artifact_store import load_joblib
from conversational_recommender import recommend_from_preferences
from conversation import parse_preferences


def load_classifier():
    try:
        return load_joblib("classifier_model.joblib")
    except FileNotFoundError:
        return None


def main():
    st.set_page_config(page_title="Conversational Movie Recommender", layout="wide")
    st.title("Conversational Movie Recommender")
    st.write("Tell me what kinds of movies you like and dislike in one sentence.")

    classifier = load_classifier()
    if classifier is None:
        st.error("Missing classifier artifact. Run `python main.py --step classify` first.")
        return

    initial_text = st.text_area(
        "Preference input",
        value="I like sci-fi and Pixar, but I hate horror.",
    )
    refinement_text = st.text_input("Optional refinement", value="")
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    if st.button("Generate Recommendations"):
        combined = initial_text.strip()
        if refinement_text.strip():
            combined = f"{combined}. {refinement_text.strip()}"

        parsed = parse_preferences(combined, classifier=classifier)
        st.subheader("Parsed preferences")
        st.write(
            {
                "positive_genres": parsed["positive_genres"],
                "negative_genres": parsed["negative_genres"],
            }
        )
        st.subheader("Clause analysis")
        st.dataframe(pd.DataFrame(parsed["clauses"]))
        st.subheader("Recommendations")
        st.dataframe(recommend_from_preferences(parsed, n=top_n))


if __name__ == "__main__":
    main()
