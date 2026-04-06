from pathlib import Path

import artifact_store
from artifact_store import artifact_path, load_json, save_json


def test_json_artifact_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", tmp_path)

    payload = {"winner": "hybrid", "metric": 0.42}

    save_json("selection.json", payload)
    saved = artifact_path("selection.json")

    assert Path(saved).exists()
    assert load_json("selection.json") == payload


def test_artifact_path_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "ARTIFACT_DIR", tmp_path)

    try:
        artifact_path("../escape.json")
    except ValueError:
        pass
    else:
        raise AssertionError("artifact_path should reject paths outside ARTIFACT_DIR")
