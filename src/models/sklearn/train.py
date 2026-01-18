import pandas as pd
from joblib import dump
from pathlib import Path

from pipeline import build_pipeline

BASE_DIR = Path(__file__).resolve().parents[3]   # project root
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sklearn_category_model.joblib"

def train():
    df = pd.read_csv("data/books_cleaned.csv")

    # Clean minimal rows
    df = df.dropna(subset=["description", "categories"])

    X = df["description"]
    y = df["categories"]

    model = build_pipeline()
    model.fit(X, y)

    dump(model, MODEL_PATH)
    print("sklearn category classifier trained and saved")

if __name__ == "__main__":
    train()
