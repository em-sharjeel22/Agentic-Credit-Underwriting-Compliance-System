"""
RAG Context Retrieval
---------------------
Queries FAISS index and extracts formatted context for LLM synthesis.
"""

import json
import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.rag.build_vectorstore import create_embeddings, EMBED_MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"


def load_vectorstore():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Vector store files are missing. Run build_pipeline() first.")

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        model = None

    return index, chunks, model


def search(query, index, chunks, embed_model=None, top_k=3):
    query_vector = create_embeddings([query], embed_model)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "rank": rank + 1,
                "section": chunks[idx]["section"],
                "text": chunks[idx]["text"],
                "distance": float(distances[0][rank]),
            })

    # Exact match fallback
    for chunk in chunks:
        if query.strip().lower() in chunk["section"].lower():
            results.insert(0, {
                "rank": 0,
                "section": chunk["section"],
                "text": chunk["text"],
                "distance": 0.0,
            })
            break

    return results


def retrieve_context(question, index, chunks, embed_model=None, top_k=3):
    results = search(question, index, chunks, embed_model=embed_model, top_k=top_k)
    context = "\n\n".join([f"[{r['section']}]\n{r['text']}" for r in results])
    sources = list(dict.fromkeys([r['section'] for r in results]))
    return context, sources

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query RAG index")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    args = parser.parse_args()

    # Load vectorstore
    index, chunks, embed_model = load_vectorstore()

    # Retrieve context
    context, sources = retrieve_context(args.query, index, chunks, embed_model=embed_model)

    print("\n=== Retrieved Context ===\n")
    print(context)
    print("\n=== Sources ===\n")
    print(sources)
