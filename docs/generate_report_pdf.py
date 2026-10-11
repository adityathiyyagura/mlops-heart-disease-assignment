from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Image, Paragraph, Preformatted, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parent.parent
PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT = 0.68 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * LEFT
CONTENT_BOTTOM = 0.65 * inch
CONTENT_HEIGHT = 9.52 * inch


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=28, leading=34, textColor=colors.HexColor("#16324F"), alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=14, leading=20,
        textColor=colors.HexColor("#426579"), alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="PageTitle", parent=styles["Heading1"], fontSize=19, leading=23,
        textColor=colors.HexColor("#16324F"), spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading2"], fontSize=12, leading=15,
        textColor=colors.HexColor("#167C80"), spaceBefore=5, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontSize=9.2, leading=13,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSmall", parent=styles["BodyText"], fontSize=8.1, leading=10.8,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportCode", parent=styles["Code"], fontName="Courier", fontSize=7.1,
        leading=9.2, leftIndent=5, rightIndent=5, backColor=colors.HexColor("#F2F5F7"),
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta", parent=styles["Normal"], fontSize=10, leading=15,
        alignment=TA_CENTER, textColor=colors.HexColor("#34495E"),
    ))
    return styles


def _p(text, style):
    return Paragraph(text, style)


def _bullet_list(items, style):
    return [_p("&bull;&nbsp;&nbsp;" + item, style) for item in items]


def _table(rows, widths=None, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C4CC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _image(path, width, height):
    image_path = BASE / path
    if not image_path.exists():
        raise FileNotFoundError(f"Required report image not found: {image_path}")
    return Image(str(image_path), width=width, height=height, kind="proportional")


def _draw_page(canvas, page_number, title, blocks, styles, cover=False):
    if not cover:
        canvas.setFillColor(colors.HexColor("#426579"))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(LEFT, PAGE_HEIGHT - 0.38 * inch, "HEART DISEASE MLOPS | PROJECT REPORT")
        canvas.setStrokeColor(colors.HexColor("#D5E0E5"))
        canvas.line(LEFT, PAGE_HEIGHT - 0.47 * inch, PAGE_WIDTH - LEFT, PAGE_HEIGHT - 0.47 * inch)

    frame = Frame(
        LEFT, CONTENT_BOTTOM, CONTENT_WIDTH, CONTENT_HEIGHT,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0,
    )
    content = list(blocks)
    if title:
        content.insert(0, _p(title, styles["PageTitle"]))
        content.insert(1, Spacer(1, 5))
    frame.addFromList(content, canvas)
    if content:
        raise RuntimeError(f"Report page {page_number} overflowed by {len(content)} flowable(s).")

    canvas.setStrokeColor(colors.HexColor("#D5E0E5"))
    canvas.line(LEFT, 0.49 * inch, PAGE_WIDTH - LEFT, 0.49 * inch)
    canvas.setFillColor(colors.HexColor("#526777"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(LEFT, 0.31 * inch, "UCI Heart Disease | Reproducible MLOps case study")
    canvas.drawRightString(PAGE_WIDTH - LEFT, 0.31 * inch, f"{page_number} / 10")
    canvas.showPage()


def _report_pages(styles):
    body = styles["ReportBody"]
    small = styles["ReportSmall"]
    section = styles["SectionTitle"]
    pages = []

    pages.append((None, [
        Spacer(1, 0.65 * inch),
        _p("Heart Disease<br/>Prediction MLOps", styles["ReportTitle"]),
        Spacer(1, 0.2 * inch),
        _p("End-to-end development, experiment tracking, serving, and operational readiness", styles["ReportSubtitle"]),
        Spacer(1, 0.7 * inch),
        _p("Machine Learning Operations (MLOps) | Assignment 01", styles["CoverMeta"]),
        Spacer(1, 0.12 * inch),
        _p("Dataset: UCI Heart Disease, Cleveland subset", styles["CoverMeta"]),
        Spacer(1, 0.12 * inch),
        _p("Repository: github.com/adityathiyyagura/mlops-heart-disease-assignment", styles["CoverMeta"]),
        Spacer(1, 0.65 * inch),
        _p("Executive summary", section),
        _p("This project develops a reproducible binary classifier and wraps it in an API-oriented MLOps workflow. It covers source data acquisition, cleaning, exploratory analysis, model comparison, MLflow logging, automated tests, Docker packaging, Kubernetes configuration, request logging, and Prometheus metrics.", body),
        _p("The evaluated models are Logistic Regression and Random Forest. Model selection uses mean ROC-AUC from five-fold stratified cross-validation on development data. A separate stratified 20% holdout is kept out of cross-validation and reported as a final test set.", body),
        _p("Selected by tuned CV: Random Forest (mean CV ROC-AUC 0.897). Held-out test ROC-AUC: 0.932. These results are educational model metrics, not a clinical risk assessment or validation.", body),
        Spacer(1, 0.12 * inch),
        _p("System at a glance", section),
        _table([
            ["Data source -> cleaning/EDA -> stratified CV and held-out test"],
            ["MLflow tracking -> selected fitted pipeline -> FastAPI"],
            ["Docker image -> Kubernetes manifest | /metrics -> Prometheus scrape"],
        ], [7.05 * inch], font_size=8),
        Spacer(1, 0.12 * inch),
        _p("Prepared from the project implementation and locally verified outputs. Deployment evidence is identified explicitly where it has not yet been captured.", styles["CoverMeta"]),
    ]))

    pages.append(("1. Dataset acquisition and preparation", [
        _p("Data source and scope", section),
        _p("The project uses the processed Cleveland Heart Disease dataset from the UCI Machine Learning Repository. It has 13 predictor columns and one disease target. The original target has values 0 through 4; this project maps 0 to no disease and any positive value to disease present.", body),
        _p("The source is obtained by the download utility at scripts/download_data.py. The raw file is retained at data/heart.csv, while data/heart_cleaned.csv contains the complete-case data used for the deliverable. The cleaning path converts '?' to missing values and numeric-coerces feature columns.", body),
        _p("Dataset counts", section),
        _table([
            ["Measure", "Observed value", "Interpretation"],
            ["Rows loaded for EDA", "303", "Target-valid source records before feature-row removal"],
            ["Rows in cleaned CSV", "297", "Six records removed because feature values were missing"],
            ["Columns", "14", "13 predictors plus binary target"],
            ["Clean target 0", "160", "No disease in cleaned records"],
            ["Clean target 1", "137", "Disease present in cleaned records"],
            ["Missing values after cleaning", "0", "Verified in data/heart_cleaned.csv"],
        ], [2.0 * inch, 1.35 * inch, 3.7 * inch]),
        Spacer(1, 8),
        _p("Cleaning and leakage controls", section),
        *_bullet_list([
            "The target is excluded from the input matrix and converted to a binary integer label.",
            "Rows with missing predictor values are removed before training; no test labels are used to fit the estimator.",
            "The train/test split is stratified to retain approximate class proportions in both partitions.",
            "Scaling is inside the Logistic Regression pipeline, so it is fitted independently within each CV fold and on development data for the final model.",
        ], body),
        _p("This is complete-case handling rather than imputation. It is simple and reproducible for this small dataset, but may introduce bias if missingness is systematic.", small),
    ]))

    pages.append(("2. Exploratory data analysis", [
        _p("EDA is generated from the target-valid source data before feature rows with missing values are excluded. This is why EDA class counts (164 / 139) differ from cleaned modeling counts (160 / 137). The notebook and scripts save plots as artifacts for review.", body),
        _p("Target distribution and age", section),
        _table([[
            _image("artifacts/eda/class_balance.png", 3.3 * inch, 2.1 * inch),
            _image("artifacts/eda/age_distribution.png", 3.3 * inch, 2.1 * inch),
        ]], [3.45 * inch, 3.45 * inch]),
        Spacer(1, 4),
        _p("Correlation overview", section),
        _image("artifacts/eda/correlation_heatmap.png", 6.7 * inch, 3.7 * inch),
        Spacer(1, 5),
        _p("Interpretation", section),
        _p("The target classes are moderately imbalanced rather than severely skewed. Age spans adult and older patients, with most records concentrated in middle-to-late adulthood. The correlation heatmap provides a first-pass view of linear associations; it is descriptive and does not establish causality or feature importance. Missing values in source predictors are handled after this descriptive stage for model preparation.", small),
    ]))

    pages.append(("3. Features and model pipeline", [
        _p("The model uses all 13 input attributes. Nominal fields cp, restecg, and thal are one-hot encoded with unknown values ignored. The other ten fields remain numeric: Logistic Regression standardizes them, while Random Forest uses them unscaled.", body),
        _p("Input schema", section),
        _table([
            ["Feature group", "Fields", "Preparation"],
            ["Demographic", "age, sex", "Numeric; source coding retained"],
            ["Clinical measurements", "trestbps, chol, thalach, oldpeak", "Numeric coercion; complete-case rows"],
            ["Nominal categories", "cp, restecg, thal", "OneHotEncoder(handle_unknown=ignore)"],
            ["Binary/ordinal/numeric", "sex, fbs, exang, slope, ca plus measurements", "Retained numeric; scaled for Logistic Regression"],
            ["Target", "target", "0 = absent; 1-4 mapped to present"],
        ], [1.5 * inch, 2.55 * inch, 3.0 * inch]),
        Spacer(1, 8),
        _p("Estimator construction", section),
        _table([
            ["Candidate", "Pipeline", "Reproducibility controls"],
            ["Logistic Regression", "One-hot + StandardScaler -> LogisticRegression", "Grid search C/class_weight; random_state=42"],
            ["Random Forest", "One-hot + passthrough -> RandomForestClassifier", "Grid search tree size/depth/leaf; random_state=42"],
        ], [1.45 * inch, 3.35 * inch, 2.25 * inch]),
        Spacer(1, 8),
        _p("Inference uses the same named feature order as training. The API validates that all expected fields are present through its request schema. Model output is a binary class and the maximum class probability is returned as confidence; this confidence is not calibrated clinical probability.", body),
        _p("The preprocessing is part of the persisted sklearn Pipeline, so training and inference share the same category encoding and scaling transformations.", small),
    ]))

    pages.append(("4. Validation protocol and model results", [
        _p("A stratified 80/20 split is made once using random_state=42. Five-fold StratifiedKFold GridSearchCV (shuffle enabled, random_state=42) searches each model using only the 80% development partition and refits by ROC-AUC. The 20% holdout is reserved for independent final metrics and is not used for tuning.", body),
        _p("Search and selected settings", section),
        _table([
            ["Model", "Grid", "Selected settings"],
            ["Logistic Regression", "C: 0.1, 1, 10; class_weight: None/balanced", "C=0.1; class_weight=None"],
            ["Random Forest", "n_estimators: 100/200; max_depth: None/6; min_samples_leaf: 1/2", "200 trees; no depth cap; leaf=2"],
        ], [1.35 * inch, 3.1 * inch, 2.6 * inch], font_size=7.1),
        Spacer(1, 6),
        _p("Cross-validation on development partition (mean +/- sample standard deviation)", section),
        _table([
            ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            ["Logistic Regression", "0.818 +/- 0.091", "0.833 +/- 0.112", "0.769 +/- 0.119", "0.888 +/- 0.070"],
            ["Random Forest", "0.810 +/- 0.089", "0.832 +/- 0.115", "0.760 +/- 0.150", "0.897 +/- 0.059"],
        ], [1.4 * inch, 1.12 * inch, 1.12 * inch, 1.12 * inch, 1.12 * inch], font_size=7.3),
        Spacer(1, 9),
        _p("Final held-out test metrics", section),
        _table([
            ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            ["Logistic Regression", "0.850", "0.880", "0.786", "0.951"],
            ["Random Forest (CV-selected)", "0.800", "0.833", "0.714", "0.932"],
        ], [1.4 * inch, 1.12 * inch, 1.12 * inch, 1.12 * inch, 1.12 * inch], font_size=7.3),
        Spacer(1, 9),
        _p("Selection and interpretation", section),
        _p("Random Forest is selected by mean tuned CV ROC-AUC (0.897 versus 0.888). Logistic Regression scores higher on held-out ROC-AUC (0.951 versus 0.932), demonstrating why the holdout remains separate and results are reported independently rather than used for model selection.", body),
        _p("The standard deviations indicate fold-to-fold variability on a small dataset. These point estimates have substantial uncertainty and are not claims of generalization to a different hospital, population, or contemporary cohort.", small),
    ]))

    pages.append(("5. Experiment tracking and reproducibility", [
        _p("MLflow creates one run per model. Each run records tuned and estimator parameters, a serialized sklearn pipeline, held-out metrics, CV summary metrics, all three EDA plots, and held-out confusion matrix and ROC curve. CV metric names use cv_<metric>_mean/std; holdout metrics use the held_out_ prefix.", body),
        _p("Tracked CV summaries", section),
        _table([
            ["Metric", "Logistic Regression mean / std", "Random Forest mean / std"],
            ["Accuracy", "0.818 / 0.091", "0.810 / 0.089"],
            ["Precision", "0.833 / 0.112", "0.832 / 0.115"],
            ["Recall", "0.769 / 0.119", "0.760 / 0.150"],
            ["ROC-AUC", "0.888 / 0.070", "0.897 / 0.059"],
        ], [1.35 * inch, 2.85 * inch, 2.85 * inch]),
        Spacer(1, 8),
        _p("Project artifacts and entry points", section),
        _table([
            ["Path", "Purpose"],
            ["heart_disease_mlop/data.py", "Download, load, and prepare dataset"],
            ["heart_disease_mlop/modeling.py", "Pipelines, CV, holdout evaluation, MLflow, model persistence"],
            ["heart_disease_mlop/train.py", "Command-line model training entry point"],
            ["notebooks/", "EDA and model comparison notebooks"],
            ["artifacts/heart_disease_model.joblib", "Persisted selected fitted pipeline"],
            ["mlruns/", "Local MLflow tracking data (runtime output)"],
        ], [3.05 * inch, 4.0 * inch]),
        Spacer(1, 7),
        _p("A fixed random seed is used for splitting, CV shuffling, and estimators. For stronger production reproducibility, pin all dependency versions and record a dataset checksum; dependency ranges are currently specified in requirements.txt.", small),
    ]))

    pages.append(("6. Prediction API and container", [
        _p("The FastAPI service exposes health and prediction routes. A request contains the 13 clinical predictor values; the response contains a binary prediction, a human-readable risk label, and a confidence score. The request schema rejects missing or malformed fields before model inference.", body),
        _p("API surface", section),
        _table([
            ["Route", "Method", "Purpose"],
            ["/health", "GET", "Basic process health check"],
            ["/predict", "POST", "Validate features and return class, label, confidence"],
            ["/metrics", "GET", "Expose Prometheus text-format request counters and latency"],
        ], [1.25 * inch, 1.0 * inch, 4.8 * inch]),
        Spacer(1, 8),
        _p("Example request", section),
        Preformatted(
            '{\n'
            '  "age": 52, "sex": 1, "cp": 0, "trestbps": 125,\n'
            '  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,\n'
            '  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3\n'
            '}', styles["ReportCode"]),
        Spacer(1, 5),
        _p("Local run commands", section),
        Preformatted(
            'python -m heart_disease_mlop.train\n'
            'docker build -t heart-disease-api:cv-tuned .\n'
            'docker run --rm -p 8000:8000 heart-disease-api:cv-tuned\n'
            'curl http://127.0.0.1:8000/health', styles["ReportCode"]),
        Spacer(1, 7),
        _p("The model is saved as a joblib pipeline and loaded by the service. The Dockerfile installs requirements, copies the project, exposes port 8000, and starts Uvicorn. A local TestClient prediction request returned HTTP 200 during verification; the confidence value is a model score, not a calibrated medical probability.", body),
        _p("The API is an educational demonstration and must not be used to diagnose or make treatment decisions.", small),
    ]))

    pages.append(("7. Testing and continuous integration", [
        _p("Automated tests cover the data interface, one-hot model preprocessing, model evaluation, and API prediction/metrics routes. The project workflow is triggered on pushes and pull requests to main/master.", body),
        _p("Current workflow stages", section),
        _table([
            ["Stage", "Action", "Status in workflow"],
            ["Checkout", "Fetch repository source", "Configured"],
            ["Runtime", "Set up Python 3.12", "Configured"],
            ["Dependencies", "Install requirements.txt", "Configured"],
            ["Tests", "Run pytest", "Configured"],
            ["Training", "Run python -m heart_disease_mlop.train", "Configured"],
        ], [1.25 * inch, 3.1 * inch, 2.7 * inch]),
        Spacer(1, 9),
        _p("Quality gates and evidence", section),
        *_bullet_list([
            "Local verification after the encoding/tuning changes: 9 tests passed; Ruff passed. FastAPI lifecycle and TestClient deprecations remain non-failing warnings.",
            "Training was executed locally and produced the CV-selected Random Forest artifact and MLflow runs.",
            "GitHub Actions runs Ruff lint, pytest, and model training; it writes JUnit and training logs and uploads the model plus MLflow run data.",
            "Fresh hosted verification: GitHub Actions run 38026080022, attempt 2, for commit 18775b8 completed successfully on 2026-10-11. Lint, all 9 tests, model training, and artifact upload passed. See <link href=\"https://github.com/adityathiyyagura/mlops-heart-disease-assignment/actions/runs/38026080022\">the run details</link>.",
            "The screenshots below are from an earlier successful run and are retained as visual examples; the linked run is the current verification. Images: screenshots/ci_run_summary.png and screenshots/ci_job_steps.png.",
        ], body),
        _p("Hosted workflow evidence", section),
        _table([[
            _image("screenshots/ci_run_summary.png", 3.2 * inch, 1.95 * inch),
            _image("screenshots/ci_job_steps.png", 3.2 * inch, 1.95 * inch),
        ], ["Earlier run summary", "Earlier job steps"]], [3.45 * inch, 3.45 * inch]),
        Spacer(1, 5),
        _p("The uploaded workflow artifact preserves test results, the training log, the trained model, and MLflow tracking data for inspection.", small),
    ]))

    pages.append(("8. Deployment and observability", [
        _p("Deployment configuration", section),
        _p("The API was deployed to local Docker Desktop Kubernetes (v1.34.1, context docker-desktop). Both replicas reached Running/Ready. The LoadBalancer service is available at http://127.0.0.1:80 on this machine; there is no public cloud URL.", body),
        _p("The updated model is packaged as heart-disease-api:cv-tuned and referenced by the Kubernetes manifest. Deployment is re-verified with kubectl rollout status and kubectl get pods/service. The Service exposes port 80 and targets container port 8000.", body),
        _p("Request monitoring", section),
        _table([
            ["Signal", "Implementation", "Use"],
            ["Request count", "http_requests_total labels method/path/status", "Traffic and response status trends"],
            ["Latency", "http_request_duration_seconds histogram", "Request duration distribution"],
            ["Application log", "Method, path, status, elapsed seconds", "Local/container request diagnostics"],
            ["Scrape config", "monitoring/prometheus.yml", "Prometheus target configuration"],
        ], [1.35 * inch, 3.0 * inch, 2.7 * inch]),
        Spacer(1, 8),
        _p("Observed locally", section),
        _p("The updated Kubernetes LoadBalancer returned successful responses for /health, /predict, and /metrics. The sample prediction was class 0 with confidence 0.6524. Request logs were emitted by the API containers; metric counters are process-local to each replica.", body),
        _p("Kubernetes pod and service screenshots", section),
        _table([[
            _image("screenshots/kubernetes_pods.png", 3.2 * inch, 1.95 * inch),
            _image("screenshots/kubernetes_service.png", 3.2 * inch, 1.95 * inch),
        ], ["Two replicas Running/Ready", "LoadBalancer on localhost:80"]], [3.45 * inch, 3.45 * inch]),
        Spacer(1, 5),
        _p("Endpoint screenshots are also included in screenshots/kubernetes_health_response.png, screenshots/kubernetes_predict_response.png, and screenshots/kubernetes_metrics_response.png.", small),
    ]))

    pages.append(("9. Limitations, conclusion, and references", [
        _p("Limitations and responsible use", section),
        *_bullet_list([
            "Small, historical Cleveland subset: performance estimates may not transfer to another clinical setting or current population.",
            "Complete-case row removal can bias estimates if missingness is not random.",
            "Nominal cp, restecg, and thal fields are one-hot encoded; other binary and ordinal fields remain numeric and should be reviewed with domain expertise.",
            "The reported confidence is not probability calibration, and the model is not a diagnostic device.",
            "The held-out test set is one split; external validation and calibration analysis are not performed.",
        ], body),
        _p("Conclusion", section),
        _p("The implementation demonstrates a reproducible progression from public data to a tracked classifier, tested API, Docker image, verified local Kubernetes deployment, and request metrics. Five-fold stratified CV selects Random Forest by mean ROC-AUC while the final holdout remains separate. The deployment is local-only; there is no public cloud endpoint.", body),
        _p("References", section),
        *_bullet_list([
            "UCI Machine Learning Repository, Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease",
            "Project repository: https://github.com/adityathiyyagura/mlops-heart-disease-assignment",
            "scikit-learn model evaluation and StratifiedKFold documentation: https://scikit-learn.org/stable/modules/cross_validation.html",
            "MLflow Tracking documentation: https://mlflow.org/docs/latest/ml/tracking/",
            "FastAPI documentation: https://fastapi.tiangolo.com/",
        ], small),
        Spacer(1, 7),
        _p("Submission evidence checklist", section),
        _p("Code, data, notebooks/scripts, tests, Dockerfile, requirements, CI workflow, report, Kubernetes manifest, and genuine CI/Docker/local-Kubernetes screenshots are included. Replace the generated slideshow with an actual pipeline recording before final submission.", small),
    ]))

    return pages


def build_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    pages = _report_pages(styles)
    if len(pages) != 10:
        raise RuntimeError(f"Expected 10 report pages, built {len(pages)}")

    canvas = Canvas(str(output_path), pagesize=letter)
    for index, (title, blocks) in enumerate(pages, start=1):
        _draw_page(canvas, index, title, blocks, styles, cover=index == 1)
    canvas.save()


if __name__ == "__main__":
    output = BASE / "docs" / "report.pdf"
    build_pdf(output)
    print(f"10-page PDF generated at: {output}")
