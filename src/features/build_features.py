#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

def load_raw_pages(raw_dir: Path):
    for path in sorted(raw_dir.glob("repos_page_*.json")):
        with open(path) as f:
            page = json.load(f)
        yield from page["items"]

def build_feature_row(item: dict) -> dict:
    return {
        "full_name":       item["full_name"],
        "stargazers_count":item["stargazers_count"],  # target
        "forks_count":     item["forks_count"],
        "watchers_count":  item["watchers_count"],
        "open_issues":     item["open_issues_count"],
        "size_kb":         item["size"],
        "created_at":      item["created_at"],
        "updated_at":      item["updated_at"],
        "pushed_at":       item["pushed_at"],
        "subscribers":     item.get("subscribers_count",0),
        "network_count":   item.get("network_count",0),
        "language":        item["language"] or "None",
    }

def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_feature_row(item) for item in load_raw_pages(raw_dir)]
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "features.parquet", index=False)
    print(f"✓ Wrote {len(df)} rows to {out_dir / 'features.parquet'}")

if __name__ == "__main__":
    main()
