import time
import csv
from datetime import datetime

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# =========================
# CONFIG
# =========================

DB_PATH = "embeddings"  # ChromaDB path on disk

# 3 LLMs (Ollama models)
LLMS = {
    "LLM1": "llama3.2:3b",
    "LLM2": "mistral:7b",
    "LLM3": "gemma:2b",
}

# 3 RAG techniques
RAG_TYPES = ["RAG1", "RAG2", "RAG3"]  # RAG1: simple, RAG2: multi-query, RAG3: rerank

# MongoDB config
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "rag_eval_db"
MONGO_COLLECTION_NAME = "eval_3x3_results"


# =========================
# SETUP: Chroma + Embedder
# =========================

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("rag_docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Insert some example docs if DB is empty
docs = [
    "Alice followed the White Rabbit down a rabbit-hole and arrived in a strange land.",
    "The Queen of Hearts loved to shout 'Off with his head!' during trials.",
    "The story involves a trial with the King and Queen of Hearts, the Knave, and a mysterious letter.",
    "An old man and a boy live in a village by the sea, dreaming about lions on the beach.",
    "The main themes in the documents are curiosity, justice, and the bond between an old man and a boy.",
]

if collection.count() == 0:
    print("🔹 Adding seed documents to ChromaDB...")
    embs = embedder.encode(docs).astype(float).tolist()
    collection.add(
        documents=docs,
        embeddings=embs,
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
else:
    print("🔹 Documents already exist — skipping add().")


# =========================
# MongoDB helper
# =========================

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION_NAME]


def save_eval_row_mongo(
    pipeline: str,
    llm_id: str,
    model: str,
    rag_id: str,
    question: str,
    latency_sec: float,
    answer: str,
    relevance_score=None,
    correctness_score=None,
    completeness_score=None,
):
    col = get_mongo_collection()
    doc = {
        "pipeline": pipeline,          # e.g. "LLM1_RAG2"
        "llm_id": llm_id,              # e.g. "LLM1"
        "model": model,                # e.g. "llama3.2:3b"
        "rag_id": rag_id,              # e.g. "RAG2"
        "question": question,
        "mode": "rag",                 # all runs here are RAG-mode
        "latency_sec": float(latency_sec),
        "answer": answer,
        "relevance_score": relevance_score,
        "correctness_score": correctness_score,
        "completeness_score": completeness_score,
        "created_at": datetime.utcnow(),
    }
    col.insert_one(doc)


# =========================
# Ollama API helper
# =========================

def call_ollama(model: str, prompt: str) -> str:
    """Call a local Ollama model and return the response text."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        # long timeout for big models
        resp = requests.post(url, json=payload, timeout=180)
    except requests.exceptions.Timeout:
        return "[REQUEST TIMEOUT from Ollama]"
    except requests.exceptions.RequestException as e:
        return f"[REQUEST ERROR] {e}"

    if not resp.ok:
        return f"[HTTP {resp.status_code}] {resp.text}"

    data = resp.json()
    return data.get("response", "").strip()


# =========================
# RAG Techniques
# =========================

def retrieve_simple_topk(question: str, top_k: int = 3) -> str:
    """RAG1: Simple dense retrieval: top-k embeddings."""
    q_emb = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs_found = results.get("documents", [[]])[0]
    return "\n\n".join(docs_found)


def expand_queries_with_llm(question: str, model: str, num_variants: int = 2):
    """Use LLM to generate a few rephrasings of the question."""
    prompt = f"""
You will help improve retrieval for a search system.

Original question:
{question}

Generate {num_variants} different rephrasings of this question, one per line, with no numbering or extra text.
"""
    resp = call_ollama(model, prompt)
    lines = [line.strip() for line in resp.splitlines() if line.strip()]
    # Keep original + first N variants
    return [question] + lines[:num_variants]


def retrieve_multi_query(question: str, model: str, top_k_each: int = 2) -> str:
    """RAG2: Multi-query / query expansion RAG."""
    queries = expand_queries_with_llm(question, model)
    all_docs = []
    seen = set()

    for q in queries:
        q_emb = embedder.encode([q])[0]
        results = collection.query(query_embeddings=[q_emb], n_results=top_k_each)
        docs_found = results.get("documents", [[]])[0]
        for d in docs_found:
            if d not in seen:
                seen.add(d)
                all_docs.append(d)

    return "\n\n".join(all_docs)


def retrieve_with_rerank(question: str, model: str, initial_k: int = 8, final_k: int = 3) -> str:
    """RAG3: retrieve many, rerank with LLM relevance scoring."""
    q_emb = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=initial_k)
    docs_found = results.get("documents", [[]])[0]

    scored = []
    for d in docs_found:
        score_prompt = f"""
We are doing passage retrieval.

Question:
{question}

Document:
{d}

Rate how relevant this document is to the question on a scale of 1 to 5.
Answer with ONLY a single number.
"""
        score_txt = call_ollama(model, score_prompt)
        try:
            score = float(score_txt.strip().split()[0])
        except Exception:
            score = 1.0
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_docs = [d for score, d in scored[:final_k]]
    return "\n\n".join(top_docs)


def get_context_for_rag(question: str, llm_model: str, rag_id: str) -> str:
    """Dispatch to the selected RAG technique."""
    if rag_id == "RAG1":
        return retrieve_simple_topk(question, top_k=3)
    elif rag_id == "RAG2":
        return retrieve_multi_query(question, llm_model, top_k_each=2)
    elif rag_id == "RAG3":
        return retrieve_with_rerank(question, llm_model, initial_k=8, final_k=3)
    else:
        raise ValueError(f"Unknown RAG type: {rag_id}")


# =========================
# MAIN EVALUATION LOOP
# =========================

if __name__ == "__main__":
    # You can change questions as needed
    questions = [
        "Who are the main characters mentioned?",
        "What is the main idea of the story?",
        "Describe what happens in the documents.",
    ]

    rows = []
    output_file = "llm_rag_3x3_results.csv"

    print("\n=== START 3×3 LLM + RAG EVALUATION ===")

    for llm_id, model_name in LLMS.items():
        for rag_id in RAG_TYPES:
            pipeline_id = f"{llm_id}_{rag_id}"  # e.g. "LLM1_RAG2"

            print("\n=====================================")
            print(f"PIPELINE: {pipeline_id}  (Model={model_name}, RAG={rag_id})")
            print("=====================================")

            for q in questions:
                print("\n-------------------------------------")
                print("QUESTION:", q)

                # Get context according to RAG technique
                context = get_context_for_rag(q, model_name, rag_id)

                rag_prompt = f"""You are a factual assistant.
Use ONLY the information in the CONTEXT to answer the QUESTION.
If the answer is not clearly in the context, reply exactly: "I don't know."

CONTEXT:
{context}

QUESTION:
{q}

ANSWER:"""

                t0 = time.time()
                rag_answer = call_ollama(model_name, rag_prompt)
                t_rag = time.time() - t0

                print("\n📚 RAG ANSWER:")
                print(rag_answer)
                print("⏱️ Time (RAG):", round(t_rag, 3), "sec")

                # Save in memory for CSV
                rows.append(
                    {
                        "pipeline": pipeline_id,
                        "llm_id": llm_id,
                        "model": model_name,
                        "rag_id": rag_id,
                        "question": q,
                        "mode": "rag",
                        "latency_sec": round(t_rag, 3),
                        "answer": rag_answer,
                    }
                )

                # Save to MongoDB
                save_eval_row_mongo(
                    pipeline=pipeline_id,
                    llm_id=llm_id,
                    model=model_name,
                    rag_id=rag_id,
                    question=q,
                    latency_sec=t_rag,
                    answer=rag_answer,
                )

    # Save all results to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pipeline",
                "llm_id",
                "model",
                "rag_id",
                "question",
                "mode",
                "latency_sec",
                "answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== DONE ===")
    print(f"📁 Results saved to {output_file}")
    print(f"🗄️ MongoDB collection: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
