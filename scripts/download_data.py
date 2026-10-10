import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heart_disease_mlop.data import ensure_data_file


if __name__ == "__main__":
    path = ensure_data_file()
    print(f"Dataset downloaded to: {path}")
