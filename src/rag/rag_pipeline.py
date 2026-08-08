"""
RAG Pipeline Orchestrator
-------------------------
Coordinates ingestion, indexing, retrieval, and LLM query answering.
"""

import os
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

from src.rag.ingest_documents import load_and_chunk_documents
from src.rag.build_vectorstore import build_faiss_index, EMBED_MODEL_NAME, save_vectorstore
from src.rag.query_rag import retrieve_context, load_vectorstore
from src.rag.generate_answer import generate_answer, get_llm_client

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"


def build_pipeline(source_dir: str = None, overwrite: bool = False):
    os.makedirs(KB_DIR, exist_ok=True)
    if INDEX_PATH.exists() and METADATA_PATH.exists() and not overwrite:
        print("✅ Existing FAISS index found — skipping rebuild.")
        return

    print("📚 Loading and chunking documents...")
    chunks = load_and_chunk_documents(source_dir)

    print("🔢 Building FAISS index...")
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        embed_model = None

    index = build_faiss_index(chunks, embed_model)
    save_vectorstore(index, chunks)
    print("✅ Pipeline construction complete.")


def query_pipeline(question: str, top_k: int = 3):
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("FAISS index or metadata missing. Run build_pipeline() first.")

    index, chunks, embed_model = load_vectorstore()
    llm_client = get_llm_client()

    context, sources = retrieve_context(question, index, chunks, embed_model, top_k=top_k)
    answer = generate_answer(question, context, llm_client)

    return answer, sources


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG pipeline.")
    parser.add_argument("--source-dir", default=None, help="Directory/file of documents to ingest.")
    parser.add_argument("--question", help="Query to ask the RAG system.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild index even if it exists.")
    args = parser.parse_args()

    if args.question:
        ans, srcs = query_pipeline(args.question)
        print(f"\n💬 Answer:\n{ans}\n\n📚 Sources: {', '.join(srcs)}")
    else:
        build_pipeline(args.source_dir, overwrite=args.overwrite)