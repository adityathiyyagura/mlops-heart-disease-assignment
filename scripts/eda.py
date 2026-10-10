import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heart_disease_mlop.data import load_data
from heart_disease_mlop.visualization import generate_eda_plots

OUTPUT_DIR = ROOT / "artifacts" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:\n", df["target"].value_counts().to_string())
    print("\nSummary statistics:\n", df.describe().round(2).to_string())
    generate_eda_plots(df, OUTPUT_DIR)
    print(f"\nEDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
