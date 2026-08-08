"""
Document Ingestion & Sliding Window Chunking
-------------------------------------------
Ingests raw text and produces chunks with 200-char overlap to prevent text truncation.
"""

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_TXT_PATH = KB_DIR / "sbp_consumer_financing_source.txt"
CHUNKS_PATH = KB_DIR / "chunks.json"

SOURCE_CITATION = (
    "State Bank of Pakistan. \"Prudential Regulations for Consumer Financing\" "
    "(Updated August 03, 2016). Banking Policy & Regulations Department."
)


def load_source_text(source_path=None):
    path = Path(source_path) if source_path else SOURCE_TXT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if "PRUDENTIAL REGULATIONS \nFOR \nCONSUMER FINANCING" in text:
        text = text.split("PRUDENTIAL REGULATIONS \nFOR \nCONSUMER FINANCING", 1)[1]
    return text


def chunk_by_regulation(text, chunk_size=2000, overlap=200):
    parts = re.split(r'(REGULATION\s+[RO]-\d+)', text)
    chunks = []
    step = chunk_size - overlap

    # 1. Process Intro / Definitions
    intro_text = parts[0].strip()
    for i in range(0, len(intro_text), step):
        segment = intro_text[i : i + chunk_size].strip()
        if len(segment) > 100:
            chunks.append({
                "id": f"intro_{len(chunks)}",
                "section": "Definitions & Minimum Requirements",
                "text": segment,
                "source": SOURCE_CITATION,
            })

    # 2. Process Regulation Sections using Sliding Window
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        base_id = header.lower().replace(" ", "_").replace("-", "_")

        if len(body) <= chunk_size:
            chunks.append({
                "id": base_id,
                "section": header,
                "text": f"{header}\n{body}",
                "source": SOURCE_CITATION,
            })
        else:
            part_num = 1
            for j in range(0, len(body), step):
                segment = body[j : j + chunk_size].strip()
                if len(segment) > 50:
                    chunks.append({
                        "id": f"{base_id}_part_{part_num}",
                        "section": f"{header} (Part {part_num})",
                        "text": f"{header}\n{segment}",
                        "source": SOURCE_CITATION,
                    })
                    part_num += 1

    return chunks


def save_chunks(chunks, destination_path=None):
    dest = Path(destination_path) if destination_path else CHUNKS_PATH
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def load_and_chunk_documents(source_dir=None):
    if source_dir:
        src = Path(source_dir)
        source_path = src if src.is_file() else src / "sbp_consumer_financing_source.txt"
    else:
        source_path = SOURCE_TXT_PATH

    text = load_source_text(source_path)
    chunks = chunk_by_regulation(text)
    save_chunks(chunks)
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"✅ Ingested {len(chunks)} chunks.")