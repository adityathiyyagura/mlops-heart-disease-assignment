import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease_mlop.data import load_data, prepare_features
from heart_disease_mlop.visualization import generate_eda_plots, generate_evaluation_plots

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "heart_disease_model.joblib"
CATEGORICAL_FEATURES = ["cp", "restecg", "thal"]
NUMERIC_FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "slope", "ca"
]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",
}
PARAMETER_GRIDS = {
    "logistic_regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 6],
        "model__min_samples_leaf": [1, 2],
    },
}


def _tune_model(model_name: str, X, y, folds: int = 5):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=build_model(model_name),
        param_grid=PARAMETER_GRIDS[model_name],
        scoring=SCORING,
        refit="roc_auc",
        cv=splitter,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)

    results = search.cv_results_
    best_index = search.best_index_
    cross_validation = {
        metric: {
            "mean": float(results[f"mean_test_{metric}"][best_index]),
            "std": float(results[f"std_test_{metric}"][best_index]),
        }
        for metric in SCORING
    }
    return search.best_estimator_, cross_validation, search.best_params_


def _held_out_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, predictions, probabilities


def build_model(model_name: str):
    if model_name not in PARAMETER_GRIDS:
        raise ValueError(f"Unsupported model type: {model_name}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "numeric",
                StandardScaler() if model_name == "logistic_regression" else "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
            ]
        )


def evaluate_models(data_path: str | Path = "data/heart.csv"):
    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = {}

    for model_name in ["logistic_regression", "random_forest"]:
        model, cross_validation, best_params = _tune_model(model_name, X_train, y_train)
        held_out, _, _ = _held_out_metrics(model, X_test, y_test)

        metrics[model_name] = {
            "cross_validation": cross_validation,
            "held_out_test": held_out,
            "best_params": best_params,
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
    eda_plot_dir = generate_eda_plots(df, ARTIFACTS_DIR / "eda")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name = None
    best_model = None
    best_score = -1.0
    model_results = {}

    for model_name in ["logistic_regression", "random_forest"]:
        with mlflow.start_run(run_name=model_name):
            model, cross_validation, best_params = _tune_model(model_name, X_train, y_train)
            metrics, predictions, probabilities = _held_out_metrics(model, X_test, y_test)
            model_results[model_name] = {
                "cross_validation": cross_validation,
                "held_out_test": metrics,
                "best_params": best_params,
            }

            mlflow.log_params(model.named_steps["model"].get_params())
            mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
            tracked_metrics = {
                f"cv_{metric}_{stat}": value
                for metric, summary in cross_validation.items()
                for stat, value in summary.items()
            }
            tracked_metrics.update({f"held_out_{metric}": value for metric, value in metrics.items()})
            mlflow.log_metrics(tracked_metrics)
            evaluation_plot_dir = generate_evaluation_plots(
                y_test,
                predictions,
                probabilities,
                ARTIFACTS_DIR / "model_evaluations" / model_name,
            )
            mlflow.log_artifacts(str(eda_plot_dir), artifact_path="eda")
            mlflow.log_artifacts(str(evaluation_plot_dir), artifact_path="evaluation_plots")
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
