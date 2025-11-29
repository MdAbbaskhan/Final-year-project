import pandas as pd
from pathlib import Path

print(">>> add_manual_scores.py STARTED")

# Always work relative to this file
BASE = Path(__file__).resolve().parent
csv_path = BASE / "multi_model_results.csv"

print(f"Looking for CSV at: {csv_path}")

if not csv_path.exists():
    print("❌ ERROR: multi_model_results.csv NOT FOUND")
    raise SystemExit(1)

# Load original file
df = pd.read_csv(csv_path)
print(f"✅ Loaded rows: {len(df)}")

# Add empty scoring columns (you will fill them later in Excel)
df["relevance_score"] = 0
df["correctness_score"] = 0
df["completeness_score"] = 0

output_path = BASE / "multi_model_results_scored.csv"
df.to_csv(output_path, index=False)

print("✅ New file saved:")
print(f"   {output_path}")
print("👉 Open this file in Excel and fill in the scores.")
