#!/usr/bin/env python3
import os
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle


def load_and_engineer_features(path: Path) -> pd.DataFrame:
    """
    Load the parquet feature table, convert date fields to numeric, and one-hot encode language.
    """
    df = pd.read_parquet(path)

    # Convert date strings to datetime
    for col in ["created_at", "updated_at", "pushed_at"]:
        df[col] = pd.to_datetime(df[col])

    # Numeric features from dates
    now = pd.Timestamp.now(tz="UTC")
    df["age_days"] = (now - df["created_at"]).dt.days
    df["days_since_push"] = (now - df["pushed_at"]).dt.days

    # One-hot encode language
    df = pd.get_dummies(df, columns=["language"], prefix="lang")

    return df


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_model(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main():
    # Paths
    features_path = Path("data/features/features.parquet")
    metrics_dir = Path("models") / "metrics"
    artifacts_dir = Path("models") / "artifacts"
    ensure_dir(metrics_dir)
    ensure_dir(artifacts_dir)

    # Load and engineer features
    df = load_and_engineer_features(features_path)

    # Prepare target and feature matrix
    y = df["stargazers_count"]
    X = df.drop(columns=[
        "full_name", "stargazers_count", "created_at",
        "updated_at", "pushed_at", "watchers_count"
    ])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    metrics = {}

    # Train, evaluate, and save
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        metrics[name] = {"r2": score}
        print(f"{name} R2: {score:.4f}")
        save_model(model, artifacts_dir / f"{name}.pkl")

    # Save metrics
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
