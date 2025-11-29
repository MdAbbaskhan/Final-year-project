# src/export_mongo_results.py

from pymongo import MongoClient
import pandas as pd
from pathlib import Path

# --- Connect to MongoDB (same DB as multi_model_eval.py) ---
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_eval_db"
COLLECTION_NAME = "eval_3x3_results"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# --- Fetch all documents ---
docs = list(col.find({}))  # grab everything

if not docs:
    print("⚠️ No documents found in MongoDB. Did you run multi_model_eval.py?")
    raise SystemExit

# Drop the MongoDB _id field (not needed in CSV)
for d in docs:
    d.pop("_id", None)

df = pd.DataFrame(docs)

# --- Save to CSV next to this script (in src folder) ---
out_path = Path(__file__).resolve().parent / "mongo_eval_dump.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

print(f"✅ Exported {len(df)} rows from MongoDB.")
print(f"📁 CSV saved at: {out_path}")
print("\nFirst few rows:")
print(df.head())
