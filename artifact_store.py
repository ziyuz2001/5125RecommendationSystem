import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def artifact_path(name: str) -> Path:
    base_dir = ARTIFACT_DIR.resolve()
    path = (base_dir / name).resolve()
    try:
        path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("artifact path must stay within ARTIFACT_DIR") from exc
    return path


def save_joblib(name: str, obj: Any) -> Path:
    path = artifact_path(name)
    joblib.dump(obj, path)
    return path


def load_joblib(name: str) -> Any:
    return joblib.load(artifact_path(name))


def save_csv(name: str, df: pd.DataFrame) -> Path:
    path = artifact_path(name)
    df.to_csv(path, index=False)
    return path


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(artifact_path(name))


def save_json(name: str, payload: dict[str, Any]) -> Path:
    path = artifact_path(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def load_json(name: str) -> dict[str, Any]:
    with artifact_path(name).open("r", encoding="utf-8") as f:
        return json.load(f)
