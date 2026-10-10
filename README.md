# Heart Disease MLOps Assignment

This project implements a small but complete MLOps pipeline for the Heart Disease UCI dataset.

## Project goals
- Download and clean the dataset
- Explore and summarize the data
- Train and compare multiple classifiers
- Track experiments with MLflow
- Save the best model for reuse
- Expose a prediction API with FastAPI
- Containerize the service with Docker
- Add tests and CI workflow

## Folder structure
- `heart_disease_mlop/`: project source code
- `app/`: FastAPI API application
- `tests/`: unit tests
- `data/`: downloaded dataset
- `artifacts/`: saved model files
- `.github/workflows/`: CI automation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m heart_disease_mlop.train
```

## Run API locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Example prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 3
  }'
```

## Docker

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```
