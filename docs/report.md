# Heart Disease Prediction MLOps Project Report

## 1. Objective
The project aims to build an end-to-end machine learning pipeline for predicting heart disease risk using the UCI Heart Disease dataset. The solution includes data preprocessing, model training, experiment tracking, model packaging, CI/CD testing, API serving, Dockerization, and deployment configuration.

Repository: [mlops-heart-disease-assignment](https://github.com/adityathiyyagura/mlops-heart-disease-assignment)

## 2. Dataset and problem definition
Dataset: Heart Disease UCI dataset
Goal: Binary classification of whether a patient is likely to have heart disease based on medical attributes.

## 3. Data acquisition and EDA
The dataset is downloaded from the UCI repository and cleaned by converting missing values, removing invalid rows, and converting the target to a binary label:
- 0 = no heart disease
- 1 = heart disease present

EDA includes:
- target class balance
- age distribution
- correlation heatmap
- summary statistics

Artifacts generated in the `artifacts/eda` directory.

## 4. Modeling choices
Two models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier

The data was split once into an 80% development partition and a 20% held-out test partition using stratification and a fixed random seed. Five-fold `StratifiedKFold` cross-validation was run only on the development partition. The held-out partition was excluded from cross-validation and used for the final test metrics. Cross-validation values below are mean ± sample standard deviation across the five folds. The Random Forest was selected using mean CV ROC-AUC; the held-out results are reported separately and were not used for model selection.

### Five-fold stratified cross-validation (development data)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.827 ± 0.066 | 0.849 ± 0.096 | 0.769 ± 0.108 | 0.869 ± 0.077 |
| Random Forest | 0.810 ± 0.084 | 0.833 ± 0.124 | 0.760 ± 0.142 | 0.888 ± 0.062 |

### Final held-out test results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.833 | 0.846 | 0.786 | 0.950 |
| Random Forest (selected by CV) | 0.867 | 0.885 | 0.821 | 0.940 |

## 5. Experiment tracking
MLflow was configured to log the following for each run:
- hyperparameters
- mean and standard deviation of five-fold CV accuracy, precision, recall, and ROC-AUC
- separate held-out test accuracy, precision, recall, and ROC-AUC
- model artifact
- experiment metadata

The experiment is stored under `mlruns/`.

## 6. Reproducibility
The project stores the trained model and uses a consistent feature schema. Model inputs are passed in the same column order as training to maintain reproducibility.

## 7. API and packaging
A FastAPI service exposes a `/predict` endpoint. It accepts JSON patient data and returns:
- prediction
- risk label
- confidence score

The app is containerized using Docker and can be run locally with:

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## 8. CI/CD
GitHub Actions runs Ruff linting, pytest, and model training on pushes and pull requests. Each run uploads the JUnit test report, training log, serialized model, and MLflow data. The latest hosted run completed successfully. Evidence screenshots: [run summary](../screenshots/ci_run_summary.png) and [job steps](../screenshots/ci_job_steps.png).

![GitHub Actions run summary](../screenshots/ci_run_summary.png)

![GitHub Actions job steps](../screenshots/ci_job_steps.png)

## 9. Deployment and monitoring
The API was deployed to the local Docker Desktop Kubernetes cluster (Kubernetes v1.34.1, context `docker-desktop`). Both replicas reached Running/Ready. The LoadBalancer is available at `http://127.0.0.1` on port 80; this is local-only, not a public cloud endpoint.

The Kubernetes service returned successful responses for `/health`, `/predict`, and `/metrics`. A sample prediction returned class `0` with confidence `0.685`. The API exposes Prometheus metrics and logs requests. Deployment evidence: [pod status](../screenshots/kubernetes_pods.png), [service status](../screenshots/kubernetes_service.png), [health](../screenshots/kubernetes_health_response.png), [prediction](../screenshots/kubernetes_predict_response.png), and [metrics](../screenshots/kubernetes_metrics_response.png).

To deploy locally, select the `docker-desktop` context and run `kubectl apply -f k8s/deployment.yaml`, then verify with `kubectl rollout status deployment/heart-disease-api` and `kubectl get pods,svc`.

## 10. Architecture diagram

```mermaid
flowchart LR
    A[Data Source] --> B[Data Pipeline]
    B --> C[EDA & Feature Prep]
    C --> D[Train Models]
    D --> E[MLflow Tracking]
    E --> F[Best Model]
    F --> G[FastAPI Service]
    G --> H[Docker Container]
    H --> I[Kubernetes Deployment]
    G --> J[Prometheus Metrics]
```

## 11. Setup instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m heart_disease_mlop.train
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 12. Conclusion
This project demonstrates an end-to-end MLOps workflow from data ingestion through model training, CI, Docker packaging, local Kubernetes deployment, and API monitoring. The service is not publicly deployed; local access is available at `http://127.0.0.1` while Docker Desktop Kubernetes is running.
