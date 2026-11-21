
"""
Multiple Linear Regression baseline for Kaggle Diamonds.

Uses the same dataset, filtering, and train/val/test split
as train.py for Residual Feature Attention.

Reports MSE, RMSE, MAE, and R² on all three splits.
"""

import argparse
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent

TARGET = "price"

NUMERIC = [
    "carat", "depth", "table", "x", "y", "z"
]

CATEGORICAL = [
    "cut", "color", "clarity"
]

FEATURES = [
    "carat", "cut", "color", "clarity",
    "depth", "table", "x", "y", "z"
]


def load_dataset(csv_path=None):
    if csv_path is not None:
        path = Path(csv_path).expanduser().resolve()
    else:
        path = ROOT / "diamonds.csv"

        if not path.exists():
            print("Downloading Diamonds from Kaggle...")
            path = Path(
                kagglehub.dataset_download(
                    "shivam2503/diamonds",
                    path="diamonds.csv"
                )
            )

    print(f"Loading dataset: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].copy()

    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_rows = len(df)

    # Identical filtering to attention train.py
    valid = (
        df[TARGET].gt(0)
        & df[["x", "y", "z"]].gt(0).all(axis=1)
        & df[["x", "y", "z"]].lt(15).all(axis=1)
    )

    df = df.loc[valid].reset_index(drop=True)

    print(f"Original rows: {original_rows:,}")
    print(f"Usable rows:   {len(df):,}")

    return df


def create_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])


def evaluate(name, model, X, y):
    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"MSE  : ${mse:,.2f} squared")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"MAE  : ${mae:,.2f}")
    print(f"R²   : {r2:.5f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)

    args = parser.parse_args()

    print("=" * 55)
    print("MULTIPLE LINEAR REGRESSION - DIAMONDS BASELINE")
    print("=" * 55)

    df = load_dataset(args.csv)

    if args.max_rows > 0 and args.max_rows < len(df):
        df = df.sample(
            n=args.max_rows,
            random_state=args.seed
        ).reset_index(drop=True)

    # Exactly the same split procedure as train.py
    train, remaining = train_test_split(
        df,
        test_size=0.20,
        random_state=args.seed
    )

    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=args.seed
    )

    print(f"\nTrain:      {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test:       {len(test):,}")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = create_model()

    print("\nFitting multiple linear regression...")
    model.fit(X_train, y_train)

    coefficients = model.named_steps["regressor"].coef_
    print(f"Regression coefficients: {len(coefficients)}")
    print("Training complete.")

    evaluate("TRAIN", model, X_train, y_train)
    evaluate("VALIDATION", model, X_val, y_val)
    evaluate("TEST", model, X_test, y_test)


if __name__ == "__main__":
    main()

