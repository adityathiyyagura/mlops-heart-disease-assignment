from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def generate_eda_plots(df, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = No disease, 1 = Disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    return output_dir


def generate_evaluation_plots(y_true, predictions, probabilities, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No disease", "Disease"],
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Held-out Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title("Held-out ROC Curve")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(figure)
    return output_dir
