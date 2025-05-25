#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def load_features(path: Path) -> pd.DataFrame:
    """
    Load the parquet feature table with engineered features.
    """
    return pd.read_parquet(path)


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

    # Load features
    df = load_features(features_path)

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Features", fontsize=16)
    plt.tight_layout()
    output_path = "src/features/feature_correlation_heatmap.png"
    plt.savefig(output_path)
    plt.close()

    output_path

    # Target: log1p_stars
    y = df["log1p_stars"]
    X = df.drop(columns=["full_name", "log1p_stars", "log1p_watchers", "log1p_forks"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    base_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgb": XGBRegressor(random_state=42, verbosity=0),
        "lgbm": LGBMRegressor(random_state=42),
    }

    # Simplified grids for tuning
    param_grids = {
        "xgb": {"n_estimators": [100], "max_depth": [3]},
        "lgbm": {
            "n_estimators": [100, 300],
            "max_depth": [3, 6, -1],
            "learning_rate": [0.01, 0.1],
        },
    }

    metrics = {}

    # Train, tune, evaluate, save
    for name, model in base_models.items():
        print(f"\n===== Processing {name} =====")
        if name in param_grids:
            grid = GridSearchCV(
                model, param_grids[name], cv=3, scoring="r2", n_jobs=1, verbose=0
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            metrics[name] = {
                "cv_r2_log": grid.best_score_,
                "best_params": grid.best_params_,
            }
            print(f"{name} best CV R² (log): {grid.best_score_:.4f}")
            save_model(best, artifacts_dir / f"{name}.pkl")
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            metrics[name] = {"test_r2_log": score}
            print(f"{name} test R² (log): {score:.4f}")
            save_model(model, artifacts_dir / f"{name}.pkl")

    # Write metrics
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
