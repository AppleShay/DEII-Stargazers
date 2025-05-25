from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random


def fetch_random_repos(n=5):
    # Using a common search query to get trending/popular repos
    url = "https://api.github.com/search/repositories"
    params = {
        "q": "stars:>1000",  # only popular repos
        "sort": "stars",
        "order": "desc",
        "per_page": 100,  # get 100 and sample from it
        "page": random.randint(1, 10),  # pick a random page for variety
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    items = resp.json().get("items", [])
    return [item["full_name"] for item in random.sample(items, k=min(n, len(items)))]


app = FastAPI()

model = joblib.load("models/artifacts/best_model.pkl")
TOKEN_PATH = Path("~/.config/star-predictor/token_shay.txt").expanduser()
GITHUB_TOKEN = TOKEN_PATH.read_text().strip()
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Feature extraction function


def fetch_commit_count(full_name: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    url = f"https://api.github.com/repos/{full_name}/commits"
    params = {"since": since, "per_page": 100}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return 0
    return len(resp.json())


def extract_features(item):
    try:
        full_name = item.get("full_name", "")
        forks = item.get("forks_count", 0)
        watchers = item.get("watchers_count", 0)
        created_at = pd.to_datetime(item.get("created_at"))
        updated_at = pd.to_datetime(item.get("updated_at"))
        now = pd.Timestamp.now(tz="UTC")

        age_days = (now - created_at).days
        days_since_update = (now - updated_at).days

        commits = fetch_commit_count(full_name)

        data = {
            "log1p_forks": np.log1p(forks),
            "log1p_issues": np.log1p(item.get("open_issues_count", 0)),
            "log1p_size_kb": np.log1p(item.get("size", 0)),
            "age_days": age_days,
            "activity_ratio": days_since_update / (age_days + 1),
            "issues_per_size": item.get("open_issues_count", 0)
            / (item.get("size", 0) + 1),
            "log1p_commits": np.log1p(commits),
            "log1p_commits_per_day": (
                np.log1p(commits / age_days) if age_days > 0 else 0
            ),
            "log1p_watchers_per_fork": np.log1p(watchers / forks) if forks > 0 else 0,
            "log1p_days_since_update": np.log1p(days_since_update),
            "creation_year": created_at.year,
            "creation_month": created_at.month,
        }
        return list(data.values()), data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the StarGazers Predictor API"}


@app.get("/predict_random_repos")
def predict_random_repos():
    sample_repos = fetch_random_repos(n=5)

    predictions = []

    for repo in sample_repos:
        url = f"https://api.github.com/repos/{repo}"
        try:
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
            item = resp.json()

            features, detailed = extract_features(item)
            pred_log = model.predict([features])[0]
            predicted_stars = int(round(np.expm1(pred_log)))

            actual_stars = item.get("stargazers_count", -1)
            predictions.append(
                {
                    "repo": repo,
                    "predicted_stars": predicted_stars,
                    "actual_stars": actual_stars,
                }
            )
        except Exception as e:
            predictions.append({"repo": repo, "error": str(e)})

    return JSONResponse(content=predictions)
