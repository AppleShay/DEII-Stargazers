import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

# GitHub token path (same as your main pipeline)
TOKEN_PATH = Path("~/.config/star-predictor/token_shay.txt").expanduser()
GITHUB_TOKEN = TOKEN_PATH.read_text().strip()

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {GITHUB_TOKEN}",
}

REPOS = [
    "tiangolo/fastapi",
    "scikit-learn/scikit-learn",
    "pallets/flask",
    "psf/requests",
    "huggingface/transformers",
    "explosion/spaCy",
    "facebook/react",
    "vuejs/vue",
    "microsoft/vscode",
    "keras-team/keras",
]

GITHUB_API = "https://api.github.com/repos/"
FASTAPI_URL = "http://127.0.0.1:8000/predict"


def get_commit_count(full_name):
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    url = f"{GITHUB_API}{full_name}/commits"
    response = requests.get(
        url, headers=HEADERS, params={"since": since, "per_page": 100}
    )
    return len(response.json()) if response.ok else 0


def extract_features(repo_json):
    created = datetime.strptime(repo_json["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    updated = datetime.strptime(repo_json["updated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    now = datetime.now(timezone.utc)

    return {
        "watchers_count": repo_json["watchers_count"],
        "forks_count": repo_json["forks_count"],
        "issues_count": repo_json["open_issues_count"],
        "size_kb": repo_json["size"],
        "age_days": (now - created).days,
        "commits": get_commit_count(repo_json["full_name"]),
        "days_since_update": (now - updated).days,
        "creation_year": created.year,
        "creation_month": created.month,
    }


def main():
    for repo in REPOS:
        print(f"\n🔍 Fetching: {repo}")
        r = requests.get(GITHUB_API + repo, headers=HEADERS)
        if not r.ok:
            print(f"❌ Failed to get repo data: {repo}")
            continue

        features = extract_features(r.json())
        pred = requests.post(FASTAPI_URL, json=features)

        if pred.ok:
            print(f"✅ Predicted stars for {repo}: {pred.json()['predicted_stars']}")
        else:
            print(f"❌ Failed to predict for {repo}: {pred.text}")


if __name__ == "__main__":
    main()
