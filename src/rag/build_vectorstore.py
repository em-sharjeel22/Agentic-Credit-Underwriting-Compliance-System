# ============================================
# BUILD VECTOR STORE
# Convert chunks into embeddings and save them in a FAISS index.
# ============================================

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
KB_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")

CHUNKS_PATH = os.path.join(KB_DIR, "chunks.json")
INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(KB_DIR, "chunk_metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def build_local_embeddings(texts):
    vectors = []
    for text in texts:
        tokens = [token.lower() for token in text.replace("\n", " ").split() if token]
        vector = np.zeros(64, dtype="float32")
        for token in tokens:
            index = abs(hash(token)) % 64
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vectors.append(vector)
    return np.stack(vectors)


def build_local_embeddings(texts, dim=384):
    vectors = []
    for text in texts:
        tokens = [token.lower() for token in text.replace("\n", " ").split() if token]
        vector = np.zeros(dim, dtype="float32")
        for token in tokens:
            idx = abs(hash(token)) % dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vectors.append(vector)
    return np.stack(vectors)

def load_chunks():
    print("Loading chunks...")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def build_embeddings(chunks):
    print(f"Loading embedding model: {MODEL_NAME}")
    texts = [chunk["text"] for chunk in chunks]
    print(f"Converting {len(texts)} chunks to embeddings...")

    try:
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings.astype("float32")
    except Exception as exc:
        print(f"Using local fallback embeddings because the model download failed: {exc}")
        embeddings = build_local_embeddings(texts)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings


def build_faiss_index(embeddings):
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    print(f"Index built with {index.ntotal} vectors and {dimension} dimensions")
    return index


def save_everything(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)
    print(f"Saved index: {INDEX_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    try:
        chunks = load_chunks()
        embeddings = build_embeddings(chunks)
        index = build_faiss_index(embeddings)
        save_everything(index, chunks)
        print("Vector store is ready.")
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc