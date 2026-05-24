import os
from datetime import datetime

import pandas as pd
import requests

repo_id = "anglixue/TenK10K_multiome"

url = f"https://huggingface.co/api/datasets/{repo_id}"

response = requests.get(url)
data = response.json()

downloads = data["downloads"]

new_row = pd.DataFrame([
    {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

print(updated_df.tail())
