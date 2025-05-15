#!/usr/bin/env python3
import os
import requests
import json
from pathlib import Path

def get_token() -> str:
    """
    Reads your GitHub PAT from the path configured in GITHUB_TOKEN_PATH
    (defaults to ~/.config/star-predictor/token_shay.txt).
    """
    token_path = Path(os.getenv(
        "GITHUB_TOKEN_PATH",
        "~/.config/star-predictor/token_shay.txt"
    )).expanduser()
    return token_path.read_text().strip()

HEADERS = {"Authorization": f"token {get_token()}"}

def fetch_page(query: str, page: int = 1, per_page: int = 100) -> dict:
    """
    Fetch one page of search results from GitHub.
    """
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()

def main():
    # TODO: parameterize query via argparse later
    query = "language:python stars:>50"
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 10 pages × 100 results = 1000 repos
    for page in range(1, 11):
        data = fetch_page(query, page=page)
        filepath = out_dir / f"repos_page_{page:02}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved page {page} → {filepath}")

if __name__ == "__main__":
    main()
