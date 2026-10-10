import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heart_disease_mlop.modeling import load_best_model


if __name__ == "__main__":
    model = load_best_model()
    sample = {
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
        "thal": 3,
    }

    prediction = model.predict([list(sample.values())])
    probabilities = model.predict_proba([list(sample.values())])[0]
    print({
        "sample": sample,
        "prediction": int(prediction[0]),
        "confidence": round(float(max(probabilities)), 4),
    })
