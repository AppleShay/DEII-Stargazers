import time
import pickle
import inspect
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


# 1. Load data from CSV and rename
print("\nStarting Model training...\n")
start_time = time.time()

df = pd.read_csv("features.csv")
df = df.rename(columns={"recently_upload": "recently_updated"})
df["log1p_stars"] = np.log1p(df["stars"])  # log-transform target

# 2. Select and transform features
original_features = [
    "issues", "size_kb", "topics", "commits", "commits_per_day",
    "forks_per_day", "days_since_update", "age_days",
    "has_homepage", "recently_updated"
]
#log-transform
X = df[original_features].copy()
X["log1p_issues"] = np.log1p(X["issues"])
X["log1p_size_kb"] = np.log1p(X["size_kb"])
X["log1p_commits"] = np.log1p(X["commits"])
X["log1p_commits_per_day"] = np.log1p(X["commits_per_day"])
X["log1p_forks_per_day"] = np.log1p(X["forks_per_day"])
X["log1p_days_since_update"] = np.log1p(X["days_since_update"])
# Replace with transformed features
X = X[[
    "log1p_issues", "log1p_size_kb", "topics",
    "log1p_commits", "log1p_commits_per_day", "log1p_forks_per_day",
    "log1p_days_since_update", "age_days", "has_homepage", "recently_updated"
]]
y = df["log1p_stars"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing pipeline
preprocessor = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# 5.Define baseline models (no tuning)
baseline_models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),
    "LightGBM": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(random_state=42))
    ]),
}

# 6. Ray Tune wrapper
def train_model(config, model_cls, model_name):
    allowed_keys = inspect.signature(model_cls).parameters.keys()
    filtered_config = {k: v for k, v in config.items() if k in allowed_keys}
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_cls(**filtered_config))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    tune.report({"r2": r2})


# 7. search space
search_spaces = {
    "Random Forest": {
        "model_cls": RandomForestRegressor,
        "config": {
            "n_estimators": tune.choice([50, 100]),
            "max_depth": tune.choice([5, 7]),
            "random_state": 42,
        },
    },
    "XGBoost": {
        "model_cls": XGBRegressor,
        "config": {
            "n_estimators": tune.choice([50, 100]),
            "max_depth": tune.choice([5, 7]),
            "learning_rate": tune.choice([0.1, 0.2]),
            "random_state": 42,
            "verbosity": 0,
        },
    }
}


# 8. Run tuning

tuned_models = {}
for model_name, setup in search_spaces.items():
    result = tune.run(
        tune.with_parameters(train_model, model_cls=setup["model_cls"], model_name=model_name),
        config=setup["config"],
        metric="r2",
        mode="max",
        num_samples=20,
        scheduler=ASHAScheduler(),
        progress_reporter=CLIReporter(metric_columns=["r2"]),
        name=f"tune_{model_name.replace(' ', '_')}"
    )
    best_config = result.get_best_config(metric="r2", mode="max")
    allowed_keys = inspect.signature(setup["model_cls"]).parameters.keys()
    filtered_config = {k: v for k, v in best_config.items() if k in allowed_keys}
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", setup["model_cls"](**filtered_config))
    ])
    best_pipeline.fit(X_train, y_train)
    tuned_models[model_name] = best_pipeline


# 9. Combine all models
all_models = {**baseline_models, **tuned_models}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)


# 10. Evaluation
print("Model Evaluation")
print("-" * 40)
results = {}
for name, model in all_models.items():
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)

    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

    results[name] = {"r2": r2, "rmse": rmse}
    print(f"{name}:")
    print(f"  R² Score (log space): {r2:.4f}")
    print(f"  RMSE (original scale): {rmse:.4f}")

# 11. Save best model
final_model_name = max(results, key=lambda x: results[x]["r2"])
final_model = all_models[final_model_name]

with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print(f"\nBest model: {final_model_name} (R² = {results[final_model_name]['r2']:.4f})")
print("Saved to final_model.pkl")

end_time = time.time()
print(f"\n Model training finished in {end_time - start_time:.2f} seconds.\n")