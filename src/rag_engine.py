import chromadb
from sentence_transformers import SentenceTransformer
import requests

DB_PATH = "embeddings"

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("rag_docs")

# Load the same embedding model as preprocess
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve_context(question: str, top_k: int = 5) -> str:
    """Embed the question and retrieve top_k similar chunks from ChromaDB."""
    print("Embedding question and retrieving context from ChromaDB...")
    q_emb = embedder.encode([question])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    docs_list = results.get("documents", [[]])
    if not docs_list or not docs_list[0]:
        print("No documents returned from ChromaDB.")
        return ""

    docs = docs_list[0]
    context = "\n\n".join(docs)
    print(f"Retrieved {len(docs)} context chunks.")
    return context


def call_ollama(model: str, prompt: str) -> str:
    """Call local Ollama model via HTTP API."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    print(f"Sending request to Ollama model = {model} ...")
    resp = requests.post(url, json=payload)

    if not resp.ok:
        print("Ollama returned an error:")
        print("Status:", resp.status_code)
        print("Body:", resp.text)
        return f"[OLLAMA ERROR {resp.status_code}] {resp.text}"

    data = resp.json()
    return data.get("response", "").strip()


def rag_answer(question: str, model: str = "phi3") -> str:
    """Full RAG pipeline: retrieve context + generate answer."""
    context = retrieve_context(question)

    prompt = f"""You are a helpful domain expert.
Use ONLY the information in the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

If the answer is not clearly present in the context, say you are not sure.
Answer:"""

    answer = call_ollama(model, prompt)
    return answer


if __name__ == "__main__":
    q = "Explain the main idea of the documents."
    print("Question:", q)
    ans = rag_answer(q, model="phi3")
    print("\nRAG Answer:\n")
    print(ans)
