from joblib import load
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "sklearn_category_model.joblib"

class SklearnCategoryClassifier:
    def __init__(self):
        self.model = load(MODEL_PATH)

    def predict(self, text: str):
        return self.model.predict([text])[0]
