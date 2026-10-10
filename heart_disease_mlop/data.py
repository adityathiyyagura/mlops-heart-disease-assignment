from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "heart.csv"


def ensure_data_file(file_path: str | Path | None = None) -> Path:
    target_path = Path(file_path) if file_path is not None else DEFAULT_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)

    return target_path


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    path = ensure_data_file(file_path)
    df = pd.read_csv(path, header=None, names=FEATURE_NAMES + ["target"], na_values="?")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df


def prepare_features(df: pd.DataFrame):
    working_df = df.copy()
    for column in FEATURE_NAMES:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    working_df = working_df.dropna().copy()
    X = working_df[FEATURE_NAMES]
    y = working_df["target"].astype(int)
    return X, y
