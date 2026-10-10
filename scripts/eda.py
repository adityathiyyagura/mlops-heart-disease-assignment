import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heart_disease_mlop.data import load_data

OUTPUT_DIR = ROOT / "artifacts" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:\n", df["target"].value_counts().to_string())
    print("\nSummary statistics:\n", df.describe().round(2).to_string())

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = No disease, 1 = Disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
    plt.close()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
