import os
from statistics import mean, stdev
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_disease_mlop.data import load_data, prepare_features

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "heart_disease_model.joblib"
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def _cross_validation_metrics(model, X, y, folds: int = 5):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=splitter, scoring=SCORING)
    return {
        metric: {
            "mean": mean(scores[f"test_{metric}"].tolist()),
            "std": stdev(scores[f"test_{metric}"].tolist()),
        }
        for metric in SCORING
    }


def build_model(model_name: str):
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
            ]
        )

    raise ValueError(f"Unsupported model type: {model_name}")


def evaluate_models(data_path: str | Path = "data/heart.csv"):
    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = {}

    for model_name in ["logistic_regression", "random_forest"]:
        model = build_model(model_name)
        cross_validation = _cross_validation_metrics(model, X_train, y_train)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics[model_name] = {
            "cross_validation": cross_validation,
            "held_out_test": {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            },
        }

    return metrics


def load_best_model(model_path: str | Path = MODEL_PATH):
    model_file = Path(model_path)
    if not model_file.exists():
        train_and_track()
    return joblib.load(model_file)


def train_and_track(data_path: str | Path = "data/heart.csv", experiment_name: str = "heart_disease"):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{BASE_DIR / 'mlruns'}")
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name = None
    best_model = None
    best_score = -1.0
    model_results = {}

    for model_name in ["logistic_regression", "random_forest"]:
        with mlflow.start_run(run_name=model_name):
            model = build_model(model_name)
            cross_validation = _cross_validation_metrics(model, X_train, y_train)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            model_results[model_name] = {
                "cross_validation": cross_validation,
                "held_out_test": metrics,
            }

            mlflow.log_params(model.named_steps["model"].get_params())
            tracked_metrics = {
                f"cv_{metric}_{stat}": value
                for metric, summary in cross_validation.items()
                for stat, value in summary.items()
            }
            tracked_metrics.update({f"held_out_{metric}": value for metric, value in metrics.items()})
            mlflow.log_metrics(tracked_metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            if cross_validation["roc_auc"]["mean"] > best_score:
                best_score = cross_validation["roc_auc"]["mean"]
                best_name = model_name
                best_model = model

    if best_model is None:
        raise RuntimeError("No valid model was trained.")

    joblib.dump(best_model, MODEL_PATH)
    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "metrics": model_results[best_name]["held_out_test"],
        "model_results": model_results,
        "selection_metric": "mean five-fold CV ROC-AUC",
    }
