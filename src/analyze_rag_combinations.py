import pandas as pd
from pathlib import Path

# 🔹 Go to project root (one level above src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# 🔹 Use the SCORED file you already created
csv_path = BASE_DIR / "llm_rag_3x3_results_scored.csv"
print(f"📁 Loading: {csv_path}")

df = pd.read_csv(csv_path)

print("\n=== RAW ROWS LOADED ===")
print(df.head())

# ---- Compute average quality per row ----
score_cols = ["relevance_score", "correctness_score", "completeness_score"]

# (They are already integers; still safe to compute mean)
df["quality_avg"] = df[score_cols].mean(axis=1)

# ---- Group by LLM × RAG Technique ----
summary = (
    df.groupby(["llm_id", "rag_id"])
      .agg(
          quality_avg=("quality_avg", "mean"),
          latency_sec=("latency_sec", "mean"),
      )
      .reset_index()
      .sort_values(by=["llm_id", "rag_id"])
)

print("\n=== SUMMARY (LLM × RAG) ===")
print(summary)

# Save to CSV (inside src/)
out_path = Path(__file__).resolve().parent / "rag_combo_summary.csv"
summary.to_csv(out_path, index=False)

print(f"\n📁 Saved summary: {out_path}")
