from pathlib import Path

import pandas as pd


def test_conversational_dataset_shape_and_labels():
    path = Path("data/conversational_polarity.csv")
    assert path.exists()

    df = pd.read_csv(path)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"]) == {"positive", "negative"}
    assert len(df) >= 200
    assert df["text"].str.len().min() > 0
