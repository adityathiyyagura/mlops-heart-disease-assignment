from heart_disease_mlop.data import FEATURE_NAMES, load_data, prepare_features
from heart_disease_mlop.modeling import build_model, evaluate_models
from heart_disease_mlop.visualization import generate_eda_plots, generate_evaluation_plots


def test_build_model_returns_classifier():
    model = build_model("logistic_regression")
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_model_preprocessor_one_hot_encodes_nominal_features():
    df = load_data("data/heart.csv")
    X, _ = prepare_features(df)
    model = build_model("logistic_regression")

    transformed = model.named_steps["preprocessor"].fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(FEATURE_NAMES)


def test_evaluate_models_returns_metrics():
    metrics = evaluate_models("data/heart.csv")
    assert "logistic_regression" in metrics
    assert "random_forest" in metrics
    for model_metrics in metrics.values():
        assert set(model_metrics["cross_validation"]) == {
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
        }
        assert set(model_metrics["held_out_test"]) == {
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
        }
        assert model_metrics["best_params"]
        for summary in model_metrics["cross_validation"].values():
            assert 0.0 <= summary["mean"] <= 1.0
            assert summary["std"] >= 0.0


def test_eda_plot_generator_creates_all_artifacts(tmp_path):
    plots_dir = generate_eda_plots(load_data("data/heart.csv"), tmp_path)

    assert {path.name for path in plots_dir.glob("*.png")} == {
        "class_balance.png",
        "age_distribution.png",
        "correlation_heatmap.png",
    }


def test_evaluation_plot_generator_creates_confusion_and_roc(tmp_path):
    plots_dir = generate_evaluation_plots(
        y_true=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        probabilities=[0.1, 0.6, 0.7, 0.9],
        output_dir=tmp_path,
    )

    assert {path.name for path in plots_dir.glob("*.png")} == {
        "confusion_matrix.png",
        "roc_curve.png",
    }
