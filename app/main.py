import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from heart_disease_mlop.data import FEATURE_NAMES
from heart_disease_mlop.modeling import load_best_model, train_and_track

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_model.joblib"

logger = logging.getLogger("heart_disease_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency in seconds", ["method", "path"])

app = FastAPI(title="Heart Disease Risk API")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    logger.info("method=%s path=%s status=%s duration=%.4fs", request.method, request.url.path, response.status_code, duration)
    return response


class PatientInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.on_event("startup")
def load_model():
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
    else:
        training_result = train_and_track()
        app.state.model = training_result["best_model"]


@app.get("/")
def root():
    return {"message": "Heart Disease Risk API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PatientInput):
    if not hasattr(app.state, "model"):
        if MODEL_PATH.exists():
            app.state.model = joblib.load(MODEL_PATH)
        else:
            training_result = train_and_track()
            app.state.model = training_result["best_model"]

    features = pd.DataFrame([
        {
            "age": payload.age,
            "sex": payload.sex,
            "cp": payload.cp,
            "trestbps": payload.trestbps,
            "chol": payload.chol,
            "fbs": payload.fbs,
            "restecg": payload.restecg,
            "thalach": payload.thalach,
            "exang": payload.exang,
            "oldpeak": payload.oldpeak,
            "slope": payload.slope,
            "ca": payload.ca,
            "thal": payload.thal,
        }
    ], columns=FEATURE_NAMES)

    model = app.state.model
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "risk": "High risk" if prediction == 1 else "Low risk",
        "confidence": round(confidence, 4),
    }
