from heart_disease_mlop.modeling import build_model, evaluate_models


def test_build_model_returns_classifier():
    model = build_model("logistic_regression")
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


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
        for summary in model_metrics["cross_validation"].values():
            assert 0.0 <= summary["mean"] <= 1.0
            assert summary["std"] >= 0.0
