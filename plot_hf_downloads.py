import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hf_download_tracking.csv")

df["time"] = pd.to_datetime(df["time"], format="mixed", errors="raise")

plt.figure(figsize=(8,5))

plt.plot(
    df["time"],
    df["downloads_last_month"],
    marker="o"
)

plt.xlabel("Date")
plt.ylabel("Downloads Last Month")
plt.title("TenK10K Multiome HF Downloads")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("hf_download_trend.png")
