import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print(">>> analyze_rag_results.py started")

# 1. Load CSVs (handle messy lines safely)
try:
    rag = pd.read_csv("rag_results.csv", engine="python", on_bad_lines="skip")
    multi = pd.read_csv("multi_model_results.csv", engine="python", on_bad_lines="skip")
except FileNotFoundError as e:
    print("❌ CSV file not found:", e)
    input("Press Enter to exit...")
    raise

print("✅ rag_results rows:", len(rag))
print("✅ multi_model_results rows:", len(multi))

# 2. Tag runs
rag["run"] = "rag_run_1"
multi["run"] = "multi_model_run_1"

# 3. Combine
df = pd.concat([rag, multi], ignore_index=True)
print("✅ Combined total rows:", len(df))
print("\n🔹 First 5 rows:")
print(df.head())

# 4. Latency summary per run+model+mode
latency_summary = (
    df.groupby(["run", "model", "mode"])["latency_sec"]
      .agg(["count", "mean", "min", "max"])
      .reset_index()
)

print("\n📊 Latency Summary (per run, model, mode):")
print(latency_summary)

# Save to CSV so you can open in Excel
latency_summary.to_csv("latency_summary.csv", index=False)
print("💾 Saved: latency_summary.csv")

# 5. Average latency: RAG vs NO-RAG
latency_mode = (
    df.groupby("mode")["latency_sec"]
      .mean()
      .reset_index()
)

print("\n⚡ Avg Latency by mode (RAG vs NO-RAG):")
print(latency_mode)

latency_mode.to_csv("latency_by_mode.csv", index=False)
print("💾 Saved: latency_by_mode.csv")

# 6. Plot: RAG vs NO-RAG latency
plt.figure()
latency_mode.plot(
    x="mode",
    y="latency_sec",
    kind="bar",
    legend=False
)
plt.xlabel("Mode")
plt.ylabel("Average latency (sec)")
plt.title("Average Latency: RAG vs No-RAG")
plt.tight_layout()
plt.savefig("latency_by_mode.png")
print("🖼️ Saved: latency_by_mode.png")
plt.close()

# 7. Per-model + mode latency
latency_model_mode = (
    df.groupby(["model", "mode"])["latency_sec"]
      .mean()
      .unstack()
)

print("\n📊 Avg Latency per model (RAG vs NO-RAG):")
print(latency_model_mode)

latency_model_mode.to_csv("latency_by_model_mode.csv")
print("💾 Saved: latency_by_model_mode.csv")

plt.figure()
latency_model_mode.plot(kind="bar")
plt.xlabel("Model")
plt.ylabel("Average latency (sec)")
plt.title("Average Latency per Model (RAG vs No-RAG)")
plt.tight_layout()
plt.savefig("latency_by_model_mode.png")
print("🖼️ Saved: latency_by_model_mode.png")
plt.close()

print("\n✅ DONE. Generated files in this folder:")
print("   - latency_summary.csv")
print("   - latency_by_mode.csv")
print("   - latency_by_mode.png")
print("   - latency_by_model_mode.csv")
print("   - latency_by_model_mode.png")

input("\nPress Enter to exit...")
