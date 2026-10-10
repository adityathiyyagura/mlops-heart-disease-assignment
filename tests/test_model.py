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
    assert metrics["logistic_regression"]["accuracy"] >= 0.0
    assert metrics["random_forest"]["accuracy"] >= 0.0
