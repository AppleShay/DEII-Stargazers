import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle

# === 1. Load data ===
df = pd.read_parquet("data/features/features2.parquet")
print(f"✅ Loaded data: {df.shape}")
print(df.columns.tolist())

# === 2. Select features ===
numerical_features = [
    # "log1p_forks", 
    "log1p_issues", "log1p_size_kb", "age_days", "activity_ratio",
    # "fork_star_ratio",
    "issues_per_size", "avg_growth_rate", "log1p_commits",
    "log1p_commits_per_day", 
    # "log1p_forks_per_day", 
    # "log1p_watchers",
    # "log1p_watchers_per_fork",
     "log1p_days_since_update", "creation_year", "creation_month"
]

X = df[numerical_features]
y = df["log1p_stars"]  # already log-transformed in your feature script

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Preprocessing pipeline ===
preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# === 5. Define models ===
models = {
    "Linear Regression": Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())]),
    "Ridge Regression": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))]),
    "Lasso Regression": Pipeline([("preprocessor", preprocessor), ("model", Lasso(alpha=0.1))]),
    "Random Forest": Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
    "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))])
}

# === 6. Train and evaluate ===
print("\n📊 Model Evaluation")
print("-" * 50)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # return to original scale
    y_test_orig = np.expm1(y_test)

    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

    results[name] = {"r2": r2, "rmse": rmse}

    print(f"{name}:")
    print(f"  R² Score (log space): {r2:.4f}")
    print(f"  RMSE (original scale): {rmse:.4f}")

    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  5-Fold CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    print("-" * 30)

# === 7. Save best model ===
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = models[best_model_name]

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n🏆 Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
print("✅ Saved to best_model.pkl")
