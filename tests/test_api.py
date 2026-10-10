from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_endpoint_returns_prediction():
    response = client.post(
        "/predict",
        json={
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "risk" in payload
    assert "confidence" in payload


def test_metrics_endpoint_exposes_prometheus_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "http_requests_total" in content
