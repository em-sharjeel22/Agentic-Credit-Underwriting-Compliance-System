# ============================================
# QUERY THE RAG SYSTEM
# ============================================

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
KB_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")

INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(KB_DIR, "chunk_metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def build_local_query_vector(query):
    tokens = [token.lower() for token in query.replace("\n", " ").split() if token]
    vector = np.zeros(64, dtype="float32")
    for token in tokens:
        index = abs(hash(token)) % 64
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def load_vectorstore():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Vector store files are missing. Run the build step first.")

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    model = SentenceTransformer(MODEL_NAME)
    return index, chunks, model


def search(query, index, chunks, model, top_k=2):
    try:
        query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    except Exception:
        query_vector = build_local_query_vector(query).reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)
    return [
        {
            "rank": rank + 1,
            "section": chunks[item]["section"],
            "text": chunks[item]["text"],
            "distance": float(distances[0][rank]),
        }
        for rank, item in enumerate(indices[0])
    ]


if __name__ == "__main__":
    try:
        print("Loading vector store...")
        index, chunks, model = load_vectorstore()
        print("Vector store is ready.\n")
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    test_queries = [
        "What is the maximum tenure for auto loans?",
        "What is the debt burden ratio limit?",
        "How are credit card advances classified as loss?",
    ]

    for query in test_queries:
        print(f"Query: \"{query}\"")
        print("=" * 60)
        for result in search(query, index, chunks, model):
            print(f"#{result['rank']} — {result['section']} (distance: {result['distance']:.3f})")
            print(result["text"][:250] + "...")
        print()
