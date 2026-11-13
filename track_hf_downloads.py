import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ID = "anglixue/TenK10K_multiome"
START_MARKER = "<!-- download-stats:start -->"
END_MARKER = "<!-- download-stats:end -->"


def update_readme(readme, downloads, total_downloads, timestamp):
    content = readme.read_text(encoding="utf-8")
    if content.count(START_MARKER) != 1 or content.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one pair of download stats markers")
    start = content.index(START_MARKER) + len(START_MARKER)
    end = content.index(END_MARKER)
    if start > end:
        raise ValueError("README download stats markers are out of order")

    stats = (
        f"\n\n- Downloads in the last 30 days: **{downloads:,}**\n"
        f"- Total downloads: **{total_downloads:,}**\n\n"
        f"Last updated: {timestamp} UTC.\n\n"
    )
    return content[:start] + stats + content[end:]


def main():
    response = requests.get(
        f"https://huggingface.co/api/datasets/{REPO_ID}",
        params=[("expand[]", "downloads"), ("expand[]", "downloadsAllTime")],
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    downloads = data.get("downloads")
    total_downloads = data.get("downloadsAllTime")
    if any(type(value) is not int or value < 0 for value in (downloads, total_downloads)):
        raise ValueError("Hugging Face API response is missing valid download counts")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    readme = Path("README.md")
    updated_readme = update_readme(readme, downloads, total_downloads, timestamp)

    new_row = pd.DataFrame([
        {
            "time": timestamp,
            "downloads_last_month": downloads,
        }
    ])
    csv_file = "hf_download_tracking.csv"
    if os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file)
        if "time" in old_df.columns:
            old_df["time"] = pd.to_datetime(old_df["time"], format="mixed", errors="raise")
            old_df["time"] = old_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        updated_df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        updated_df = new_row

    updated_df.to_csv(csv_file, index=False)
    readme.write_text(updated_readme, encoding="utf-8")
    print(updated_df.tail())


if __name__ == "__main__":
    main()
