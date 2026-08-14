"""
Ingestion script: loads all 8 policy documents, embeds them with
all-MiniLM-L6-v2, and stores the embeddings in a ChromaDB collection.
Run once before starting the API: python ingest.py
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
COLLECTION_NAME = "zepto_policies"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def load_documents() -> list[dict]:
    docs = []
    for i in range(1, 9):
        doc_id = f"doc_{i:02d}"
        path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"id": doc_id, "text": text})
    return docs


def ingest_documents():
    print("Loading documents...")
    docs = load_documents()

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate collection for a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    print("Embedding documents...")
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(
        f"Ingested {len(docs)} documents into ChromaDB collection '{COLLECTION_NAME}' "
        f"at {CHROMA_PATH}"
    )


if __name__ == "__main__":
    ingest_documents()
