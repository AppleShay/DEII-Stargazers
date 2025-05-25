#!/usr/bin/env python3
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd


# Load GitHub token for API calls
def get_token() -> str:
    path = Path(
        os.getenv("GITHUB_TOKEN_PATH", "~/.config/star-predictor/token_feruz.txt")
    ).expanduser()
    return path.read_text().strip()


HEADERS = {"Authorization": f"token {get_token()}"}


# Fetch commit count in last 30 days for a repo
def fetch_commit_count(full_name: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    url = f"https://api.github.com/repos/{full_name}/commits"
    params = {"since": since, "per_page": 100}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return len(resp.json())


# Yield each repo item from the raw JSON pages
def load_raw_pages(raw_dir: Path):
    for path in sorted(raw_dir.glob("repos_page_*.json")):
        with open(path) as f:
            page = json.load(f)
        yield from page.get("items", [])


# Build a feature row including commit velocity and topics count
def build_feature_row(item: dict) -> dict:
    full_name = item.get("full_name", "")
    row = {
        "full_name": full_name,
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "issues": item.get("open_issues_count", 0),
        "size_kb": item.get("size", 0),
        "topics": len(item.get("topics", [])),
        "created_at": item.get("created_at", None),
        "watchers": item.get("watchers_count", 0),
        "has_homepage": int(bool(item.get("homepage"))),
        "updated_at": item.get("updated_at", None),
    }

    # One more feaature
    forks = row["forks"]
    watchers = row["watchers"]
    row["watchers_per_fork"] = watchers / forks if forks > 0 else 0

    # commit velocity
    try:
        commits = fetch_commit_count(full_name)
        row["commits"] = commits
    except:
        row["commits"] = 0
    return row


def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Looking for files in:", raw_dir.resolve())
    print("Found files:", list(raw_dir.glob("repos_page_*.json")))

    data = [build_feature_row(item) for item in load_raw_pages(raw_dir)]
    df = pd.DataFrame(data)
    print("Columns available:", df.columns.tolist())

    # parse dates & compute age
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    now = pd.Timestamp.now(tz="UTC")

    # project age and time from last update
    df["age_days"] = (now - df["created_at"]).dt.days
    df["days_since_update"] = (now - df["updated_at"]).dt.days

    # exract more features
    df["activity_ratio"] = df["days_since_update"] / (df["age_days"] + 1)
    df["fork_star_ratio"] = df["forks"] / (df["stars"] + 1)
    df["issues_per_size"] = df["issues"] / (df["size_kb"] + 1)
    df["avg_growth_rate"] = df["stars"] / (df["age_days"] + 1)

    df["creation_year"] = df["created_at"].dt.year
    df["creation_month"] = df["created_at"].dt.month

    # dropping stuff
    df = df.drop(columns=["created_at", "updated_at"])

    # derive rates
    df["commits_per_day"] = df["commits"] / df["age_days"].replace(0, np.nan)
    df["forks_per_day"] = df["forks"] / df["age_days"].replace(0, np.nan)

    # log-transform skewed counts
    for col in [
        "stars",
        "forks",
        "issues",
        "size_kb",
        "topics",
        "commits",
        "commits_per_day",
        "forks_per_day",
        "watchers",
        "watchers_per_fork",
        "days_since_update",
    ]:
        df["log1p_" + col] = np.log1p(df[col].fillna(0))

    # one-hot language (if exists)
    # assume original build_features added language if needed
    if "language" in df.columns:
        df = pd.get_dummies(df, columns=["language"], prefix="lang")

    # final feature set: drop raw columns
    raw_cols = [
        "stars",
        "forks",
        "issues",
        "size_kb",
        "topics",
        "commits",
        "commits_per_day",
        "forks_per_day",
        "watchers",
        "watchers_per_fork",
        "days_since_update",
    ]
    df_final = df.drop(columns=raw_cols)

    # write out
    features_path = out_dir / "features2.parquet"
    df_final.to_parquet(features_path, index=False)
    print(
        f"✓ Wrote {len(df_final)} rows × {df_final.shape[1]} features to {features_path}"
    )


if __name__ == "__main__":
    main()
