import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

DATA_PATH = "data"
DB_PATH = "embeddings"


def load_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        full_path = os.path.join(DATA_PATH, file)

        # TXT files
        if file.lower().endswith(".txt"):
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())

        # PDF files
        elif file.lower().endswith(".pdf"):
            docs.append(load_text_from_pdf(full_path))

    return docs


def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


if __name__ == "__main__":
    print("=== PREPROCESS STARTED ===")

    docs = load_documents()
    print("Documents found:", len(docs))

    if not docs:
        print("⚠️ No PDF or TXT files found in /data folder.")
        exit()

    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print("Chunks created:", len(chunks))

    # Embedding model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("rag_docs")

    embeddings = embedder.encode(chunks)

    for i, emb in enumerate(embeddings):
        collection.add(
            ids=[str(i)],
            documents=[chunks[i]],
            embeddings=[emb],
        )

    print("=== STORED", len(chunks), "CHUNKS IN CHROMADB ===")
