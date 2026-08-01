"""
RAG Pipeline Orchestrator
-------------------------
Coordinates document ingestion, vector indexing, and query answering.
Integrates with Groq or Hugging Face LLMs for compliance and insight generation.
"""

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

from src.rag.ingest_documents import load_and_chunk_documents
from src.rag.build_vectorstore import build_faiss_index
from src.rag.query_rag import retrieve_context, generate_answer

# ── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── LLM setup ───────────────────────────────────────────
def get_llm_client():
    """Initialize Groq or Hugging Face client."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("HF_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY or HF_API_KEY.")
    return InferenceClient(provider="groq", api_key=key)

# ── Pipeline orchestration ──────────────────────────────
def build_pipeline(source_dir: str, overwrite: bool = False):
    """
    Ingest documents, embed them, and build FAISS index.
    """
    os.makedirs(KB_DIR, exist_ok=True)
    if INDEX_PATH.exists() and not overwrite:
        print("✅ Existing FAISS index found — skipping rebuild.")
        return

    print("📚 Loading and chunking documents...")
    chunks = load_and_chunk_documents(source_dir)
    print(f"✅ Loaded {len(chunks)} chunks.")

    print("🔢 Building FAISS index...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = build_faiss_index(chunks, embed_model)
    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Index saved → {INDEX_PATH}")
    print(f"✅ Metadata saved → {METADATA_PATH}")

def query_pipeline(question: str, top_k: int = 3):
    """
    Retrieve context and generate an answer using the LLM.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("FAISS index or metadata missing. Run build_pipeline() first.")

    print("🔍 Loading index and metadata...")
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    llm_client = get_llm_client()

    print("🧠 Retrieving context...")
    context, sources = retrieve_context(question, index, chunks, embed_model, top_k=top_k)

    print("💬 Generating answer...")
    answer = generate_answer(question, context, llm_client)

    print("\n🧾 Final Answer:")
    print(answer)
    print("\n📚 Sources:", ", ".join(sources))
    return answer, sources

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG pipeline.")
    parser.add_argument("--source-dir", default="data/knowledge_base", help="Directory of documents to ingest.")
    parser.add_argument("--question", help="Query to ask the RAG system.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild index even if it exists.")
    args = parser.parse_args()

    if args.question:
        query_pipeline(args.question)
    else:
        build_pipeline(args.source_dir, overwrite=args.overwrite)
