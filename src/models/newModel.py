import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

from ray import tune
from ray.tune.sklearn import TuneSearchCV

# === 1. Load data ===
df = pd.read_parquet("data/features/features2.parquet")
print(f"✅ Loaded data: {df.shape}")
print(df.columns.tolist())

# === 2. Select features ===
numerical_features = [
    "log1p_forks",
    "log1p_issues",
    "log1p_size_kb",
    "age_days",
    "activity_ratio",
    "issues_per_size",
    # "avg_growth_rate",
    "log1p_commits",
    "log1p_commits_per_day",
    # "log1p_forks_per_day",
    # "log1p_watchers",
    "log1p_watchers_per_fork",
    "log1p_days_since_update",
    "creation_year",
    "creation_month",
]

X = df[numerical_features]
y = df["log1p_stars"]  # already log-transformed in the feature script

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Preprocessing pipeline ===
preprocessor = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Define parameter grids for Ray Tune
xgb_search_space = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.loguniform(1e-3, 0.3),
    "subsample": tune.uniform(0.5, 1.0),
}

rf_search_space = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 10),
    "min_samples_split": tune.randint(2, 10),
    "min_samples_leaf": tune.randint(1, 4),
}
# Initialize models with Ray Tune wrappers
xgb_tune = TuneSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_distributions=xgb_search_space,
    n_trials=10,
    scoring="r2",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_tune = TuneSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_search_space,
    n_trials=10,
    scoring="r2",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit both models
xgb_tune.fit(X_train_prep, y_train)
rf_tune.fit(X_train_prep, y_train)

# Evaluate
xgb_preds = xgb_tune.predict(X_test_prep)
rf_preds = rf_tune.predict(X_test_prep)

xgb_r2 = r2_score(y_test, xgb_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("XGBoost R2:", xgb_r2)
print("RandomForest R2:", rf_r2)

# Select and save the best model
best_model = xgb_tune.best_estimator_ if xgb_r2 > rf_r2 else rf_tune.best_estimator_

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", best_model)
])

final_pipeline.fit(X_train, y_train)

with open("best_model.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)

print("Best model saved to best_model.pkl")