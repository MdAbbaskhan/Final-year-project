import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print(">>> START QUALITY ANALYSIS")

BASE = Path(__file__).resolve().parent
csv_path = BASE / "multi_model_results_scored.csv"

print("Looking for:", csv_path)

if not csv_path.exists():
    print("❌ ERROR: CSV NOT FOUND")
    quit()

df = pd.read_csv(csv_path)
print("CSV Loaded. Rows:", len(df))

print("\n=== FIRST 3 ROWS ===")
print(df.head(3))

# Calculate average quality
df["quality_avg"] = df[["relevance_score","correctness_score","completeness_score"]].mean(axis=1)

print("\n=== Quality Avg calculated ===")
print(df[["model","mode","quality_avg","latency_sec"]])

# Group by model + mode
grouped = df.groupby(["model","mode"])["quality_avg"].mean().reset_index()

print("\n=== GROUPED RESULTS ===")
print(grouped)

# SAVE summary
out = BASE / "quality_summary.csv"
grouped.to_csv(out, index=False)
print("\n💾 Saved ->", out)

# Scatter: latency vs quality
plt.scatter(df["latency_sec"], df["quality_avg"])
plt.xlabel("Latency (sec)")
plt.ylabel("Average Quality Score")
plt.title("Latency vs Quality Trade-off")
plt.show()
