from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parent.parent


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Heart Disease Prediction MLOps Project Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    sections = [
        ("1. Objective", "This project implements an end-to-end MLOps pipeline for the UCI Heart Disease dataset. The goal is to build a reproducible classifier that predicts risk of heart disease and serve it through a containerized API."),
        ("2. Dataset", "The dataset contains medical attributes such as age, sex, chest pain type, cholesterol, blood pressure, and exercise-induced angina. The target is the binary heart-disease indicator."),
        ("3. Data Cleaning and EDA", "The dataset was downloaded from the UCI repository, missing values were handled, and the target variable was converted to binary labels. The cleaned data is stored in data/heart_cleaned.csv. Summary statistics and plots were generated for class balance, age distribution, and correlation analysis."),
        ("4. Feature Engineering", "The feature schema includes the 13 numerical input attributes used by the classifier, with consistent column ordering preserved for training and inference. This ensures the same preprocessing pipeline is used for deployment."),
        ("5. Model Development", "The data was split with stratification into an 80% development partition and a 20% held-out test partition. Five-fold StratifiedKFold cross-validation was performed on development data only. Models were selected by mean CV ROC-AUC; the held-out test set was not used for model selection."),
        ("6. Experiment Tracking", "MLflow was configured for training runs. Parameters, metrics, and model artifacts were logged for each run to enable reproducibility and comparison of different experiments."),
        ("7. Packaging and Reproducibility", "The final model is serialized with joblib and saved under artifacts/heart_disease_model.joblib. The project includes a clean dependency file and reproducible data pipeline scripts."),
        ("8. API and Containerization", "A FastAPI application exposes a POST /predict endpoint that accepts patient attributes and returns the prediction, risk label, and confidence. The service is packaged in Docker and runs locally with uvicorn."),
        ("9. CI/CD and Deployment", "The repository includes a GitHub Actions workflow that installs dependencies, runs tests, and performs a training step. A Kubernetes deployment manifest is also included for local deployment and scaling."),
        ("10. Monitoring and Reporting", "The API exposes Prometheus-compatible metrics at /metrics and logs each request. This makes the model service observable and suitable for production-style monitoring workflows. The screenshots and generated artifacts are saved in the screenshots directory and used for reporting."),
    ]

    for heading, content in sections:
        story.append(Paragraph(heading, styles['Heading2']))
        story.append(Paragraph(content, styles['BodyText']))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Five-fold stratified CV on development data (mean +/- sample standard deviation)", styles['Heading2']))
    cv_table = Table([
        ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
        ["Logistic Regression", "0.827 +/- 0.066", "0.849 +/- 0.096", "0.769 +/- 0.108", "0.869 +/- 0.077"],
        ["Random Forest", "0.810 +/- 0.084", "0.833 +/- 0.124", "0.760 +/- 0.142", "0.888 +/- 0.062"],
    ], colWidths=[1.35 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    cv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cv_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Final held-out test results", styles['Heading2']))
    test_table = Table([
        ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
        ["Logistic Regression", "0.833", "0.846", "0.786", "0.950"],
        ["Random Forest (selected by CV)", "0.867", "0.885", "0.821", "0.940"],
    ], colWidths=[1.35 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    test_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Selected model: Random Forest, based on the highest mean five-fold CV ROC-AUC (0.888). Its held-out ROC-AUC is 0.940. These results describe this split and dataset; they are not a clinical validation.", styles['BodyText']))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Conclusion", styles['Heading2']))
    story.append(Paragraph("This assignment demonstrates a full MLOps cycle: data ingestion, cleaning, EDA, feature engineering, model selection, training, experiment tracking, packaging, API deployment, monitoring, and documentation. The resulting system is reproducible and ready for further cloud deployment and production hardening.", styles['BodyText']))

    doc.build(story)


if __name__ == "__main__":
    output = BASE / 'docs' / 'report.pdf'
    build_pdf(output)
    print(f'PDF generated at: {output}')
