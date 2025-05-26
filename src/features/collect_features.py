#!/usr/bin/env python3
import os
import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone


def get_token() -> str:
    path = Path(
        os.getenv("GITHUB_TOKEN_PATH", "~/.config/star-predictor/token_linjia.txt")
    ).expanduser()
    return path.read_text().strip()


HEADERS = {"Authorization": f"token {get_token()}"}


def fetch_page(query: str, page: int = 1, per_page: int = 100) -> dict:
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page,
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()

def fetch_commit_count(full_name: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    url = f"https://api.github.com/repos/{full_name}/commits"
    params = {"since": since, "per_page": 100}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return len(resp.json())

def build_feature_row(repo: dict) -> dict:
    row = {
        "full_name": repo.get("full_name", ""),
        "issues": repo.get("open_issues_count", 0),
        "size_kb": repo.get("size", 0),
        "topics": len(repo.get("topics", [])),
        "forks": repo.get("forks_count", 0),   
        "stars": repo.get("stargazers_count", 0),        
        "created_at": repo.get("created_at", None),
        "updated_at": repo.get("updated_at", None),
        "has_homepage": int(bool(repo.get("homepage"))),
        "watchers": repo.get("watchers_count", 0),
    }
    forks = row["forks"]
    watchers = row["watchers"]
    row["watchers_per_fork"] = watchers / forks if forks > 0 else 0

    # commit count
    try:
        commits = fetch_commit_count(row["full_name"])
        row["commits"] = commits
    except Exception as e:
        row["commits"] = 0
        print(f"Error: {e}")
    return row

def main():
    query = "language:python stars:>50"
    all_items = []

    for page in range(1, 11):  # 10 × 100 = 1000
        data = fetch_page(query, page=page)
        items = data.get("items", [])
        all_items.extend([build_feature_row(repo) for repo in items])
        print(f" Page {page} collected")

    df = pd.DataFrame(all_items)
    #timestamps
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    now = pd.Timestamp.now(tz="UTC")

    #create new features
    df["age_days"] = (now - df["created_at"]).dt.days
    df["days_since_update"] = (now - df["updated_at"]).dt.days

    # recently upload
    df["recently_upload"] = (df["days_since_update"] <= 7).astype(int)
    df["activity_ratio"] = df["days_since_update"] / (df["age_days"] + 1)
    df["fork_star_ratio"] = df["forks"] / (df["stars"] + 1)
    df["issues_per_size"] = df["issues"] / (df["size_kb"] + 1)
    df["avg_growth_rate"] = df["stars"] / (df["age_days"] + 1)

    # time features
    df["creation_year"] = df["created_at"].dt.year
    df["creation_month"] = df["created_at"].dt.month

    # drop unused columns
    df = df.drop(columns=["created_at", "updated_at"])

    # derive rates
    df["commits_per_day"] = df["commits"] / df["age_days"].replace(0, 1)
    df["forks_per_day"] = df["forks"] / df["age_days"].replace(0, 1)

    # select features needed
    features_to_keep = [
    "issues",
    "size_kb",
    "topics",
    "commits",
    "commits_per_day",
    "forks_per_day",
    "age_days",
    "days_since_update",
    "recently_upload",
    "has_homepage",
    "stars",
    "watchers",
    "watchers_per_fork",
    "activity_ratio",
    "fork_star_ratio",
    "issues_per_size",
    "avg_growth_rate",
    "creation_year",
    "creation_month",
    ]

    df_final = df[features_to_keep]

    out_path = Path("data/feature/all_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(f" Saved {len(df_final)} rows to {out_path}")

if __name__ == "__main__":
    main()
