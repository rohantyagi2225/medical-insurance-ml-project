"""Train a reproducible medical insurance charges model from a public CSV dataset.

Usage:
    python train_model.py

The script downloads the commonly used Medical Cost Personal Dataset automatically,
trains a preprocessing + Random Forest pipeline, evaluates it on a held-out test set,
and writes insurance_model.pkl beside this script.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/insurance.csv"
)
MODEL_PATH = Path(__file__).resolve().parent / "insurance_model.pkl"
REQUIRED_COLUMNS = [
    "age", "sex", "bmi", "children", "smoker", "region", "charges"
]
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]


def main():
    print("Downloading the dataset...")
    df = pd.read_csv(DATA_URL)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[REQUIRED_COLUMNS].dropna()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    with MODEL_PATH.open("wb") as f:
        pickle.dump(pipeline, f)

    print(f"Dataset rows used: {len(df):,}")
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Random Forest R²: {r2:.4f}")
    print(f"Random Forest RMSE: {rmse:,.2f}")
    print(f"Saved model pipeline: {MODEL_PATH}")
    print("Educational model only; not suitable for real insurance underwriting.")


if __name__ == "__main__":
    main()
