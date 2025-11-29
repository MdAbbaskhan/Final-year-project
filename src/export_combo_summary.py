import pandas as pd
from pymongo import MongoClient
from pathlib import Path

# Load the RAG combo summary CSV
csv_path = Path(__file__).resolve().parent / "rag_combo_summary.csv"

if not csv_path.exists():
    print("❌ ERROR: rag_combo_summary.csv not found.")
    print("➡️ Run analyze_rag_combinations.py first.")
    exit()

df = pd.read_csv(csv_path)
print(f"\n🔥 Loaded {len(df)} summary rows")

# --- MongoDB Connection ---
client = MongoClient("mongodb://localhost:27017")
db = client["rag_eval_db"]
collection = db["rag_combo_summary"]

# Remove previous stored results
collection.delete_many({})
print("🧹 Cleared old MongoDB combo summary")

# Insert new results
collection.insert_many(df.to_dict(orient="records"))
print("🚀 Stored combo summary in MongoDB (rag_combo_summary)")
