"""
Vector Store Construction & Embedding Generation
------------------------------------------------
Converts document chunks to 384-dimensional vectors and indices via FAISS.
Includes a 384-dim fallback encoder to avoid dimension mismatch crashes.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"

CHUNKS_PATH = KB_DIR / "chunks.json"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def build_local_embeddings(texts, dim=EMBED_DIM):
    """Fallback hash vectorizer matching SentenceTransformer embedding dimension (384)."""
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
    return np.stack(vectors).astype("float32")


def create_embeddings(texts, embed_model=None):
    if isinstance(embed_model, str):
        embed_model = SentenceTransformer(embed_model)

    if embed_model is not None:
        try:
            return embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        except Exception:
            pass

    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
    except Exception:
        return build_local_embeddings(texts, dim=EMBED_DIM)


def build_faiss_index(chunks_or_embeddings, embed_model=None):
    """
    Overloaded interface accepting either a NumPy embedding matrix or raw chunk dictionaries.
    """
    if isinstance(chunks_or_embeddings, np.ndarray):
        embeddings = chunks_or_embeddings
    else:
        texts = [chunk["text"] for chunk in chunks_or_embeddings]
        embeddings = create_embeddings(texts, embed_model)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index


def save_vectorstore(index, chunks):
    os.makedirs(KB_DIR, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        idx = build_faiss_index(chunks)
        save_vectorstore(idx, chunks)
        print("✅ Vector store created successfully.")