import csv

CSV_PATH = "rag_results.csv"

print("=== SIMPLE RAG SUMMARY ===")
print("Reading file:", CSV_PATH)

# --- Load all rows from CSV ---
rows = []
with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total rows read: {len(rows)}")
if not rows:
    print("No data rows found in CSV. Check rag_results.csv!")
    quit()

# --- Separate no_rag vs rag ---
no_rag = [r for r in rows if r.get("mode") == "no_rag"]
rag = [r for r in rows if r.get("mode") == "rag"]

print(f"no_rag rows: {len(no_rag)}")
print(f"rag rows   : {len(rag)}\n")

def average_latency(subset):
    if not subset:
        return 0.0
    total = 0.0
    count = 0
    for r in subset:
        try:
            total += float(r.get("latency_sec", 0))
            count += 1
        except ValueError:
            pass
    return total / count if count > 0 else 0.0

avg_no_rag = average_latency(no_rag)
avg_rag = average_latency(rag)

# --- Very simple "accuracy" assumption ---
# assume: all no_rag answers are inaccurate (0), all rag answers are accurate (1)
acc_no_rag = 0.0 if no_rag else 0.0
acc_rag = 1.0 if rag else 0.0

print("=== RESULTS ===")
print(f"Average latency (no RAG): {avg_no_rag:.2f} seconds")
print(f"Average latency (RAG)   : {avg_rag:.2f} seconds\n")

print(f"Accuracy (no RAG): {acc_no_rag:.2f}  (assumed)")
print(f"Accuracy (RAG)   : {acc_rag:.2f}  (assumed)\n")

if no_rag:
    print("Hallucination (no RAG): assumed HIGH (generic / not grounded)")
if rag:
    print("Hallucination (RAG)   : assumed LOW (grounded in documents)")

print("\n=== DONE ===")
