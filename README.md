# Heart Disease MLOps Assignment

This project implements a small but complete MLOps pipeline for the Heart Disease UCI dataset.

## Repository

GitHub: [mlops-heart-disease-assignment](https://github.com/adityathiyyagura/mlops-heart-disease-assignment)

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
docker build -t heart-disease-api:cv-tuned .
docker run --rm -p 8000:8000 heart-disease-api:cv-tuned
```

## Local Kubernetes (Docker Desktop)

Enable Kubernetes in Docker Desktop settings and wait until the cluster is running. These commands target the local Docker Desktop context; confirm the context before applying manifests so you do not deploy to a cloud cluster by accident.

```bash
kubectl config use-context docker-desktop
kubectl config current-context
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl get pods -l app=heart-disease-api
kubectl get svc heart-disease-api
```

The Docker Desktop LoadBalancer exposes the API locally at `http://127.0.0.1` on port 80. Test the service with:

```bash
curl http://127.0.0.1/health
curl http://127.0.0.1/metrics
```

Open `http://127.0.0.1/docs` to submit a sample request to `/predict`. Keep Docker Desktop Kubernetes running while using this URL. This is a local endpoint, not a public cloud URL.

To remove the local deployment:

```bash
kubectl delete -f k8s/deployment.yaml
```
