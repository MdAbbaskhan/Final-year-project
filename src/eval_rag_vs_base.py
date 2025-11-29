import time
import csv
import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "embeddings"
MODEL_NAME = "llama3.2:1b"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "rag_results.csv")

# DB + embeddings
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("rag_docs")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def call_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload)
    if not resp.ok:
        return f"[ERROR {resp.status_code}] {resp.text}"
    return resp.json().get("response", "").strip()


def retrieve_context(question: str, top_k: int = 5) -> str:
    q_emb = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)


def answer_base(question: str) -> str:
    prompt = f"""Answer the question using your internal knowledge.

Question: {question}
Answer:"""
    return call_ollama(MODEL_NAME, prompt)


def answer_rag(question: str) -> str:
    context = retrieve_context(question)
    prompt = f"""Use ONLY the context below to answer:

CONTEXT:
{context}

Question: {question}
Answer:"""
    return call_ollama(MODEL_NAME, prompt)


if __name__ == "__main__":
    questions = [
        "Who are the main characters mentioned?",
        "What is the main idea of the story?",
        "Describe what happens in the documents."
    ]

    rows = []
    print("=== STARTING RAG EVALUATION ===")
    print("Project root:", PROJECT_ROOT)
    print("CSV path   :", CSV_PATH, "\n")

    for q in questions:
        print(f"\nQUESTION: {q}")

        # BASE (No RAG)
        t0 = time.time()
        base = answer_base(q)
        t_base = time.time() - t0
        print("\nBASE ANSWER:\n", base)
        print("Time (base):", round(t_base, 3), "sec")

        # RAG
        t1 = time.time()
        rag = answer_rag(q)
        t_rag = time.time() - t1
        print("\nRAG ANSWER:\n", rag)
        print("Time (RAG):", round(t_rag, 3), "sec")

        rows.append({
            "model": MODEL_NAME,
            "question": q,
            "mode": "no_rag",
            "latency_sec": round(t_base, 3),
            "answer": base
        })
        rows.append({
            "model": MODEL_NAME,
            "question": q,
            "mode": "rag",
            "latency_sec": round(t_rag, 3),
            "answer": rag
        })

    print("\nTotal rows to save:", len(rows))

    # SIMPLE CSV WRITE
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # header
        writer.writerow(["model", "question", "mode", "latency_sec", "answer"])
        # data
        for r in rows:
            writer.writerow([
                r["model"],
                r["question"],
                r["mode"],
                r["latency_sec"],
                r["answer"]
            ])

    print("=== DONE ===")
    print("Saved", len(rows), "rows to:", CSV_PATH)
