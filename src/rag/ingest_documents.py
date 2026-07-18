# ============================================
# RAG DOCUMENT INGESTION
# SBP Prudential Regulations for Consumer Financing
#
# SBP ka server automated PDF downloads block karta hai
# (bot detection). Yeh script ek locally-cached, verified
# text transcription se kaam karta hai — fully reproducible,
# koi PDF-parsing dependency nahi chahiye.
# ============================================

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_TXT_PATH = KB_DIR / "sbp_consumer_financing_source.txt"
CHUNKS_PATH = KB_DIR / "chunks.json"

SOURCE_URL = "https://www.sbp.org.pk/publications/prudential/PRs-Consumer.pdf"
SOURCE_CITATION = (
    "State Bank of Pakistan. \"Prudential Regulations for Consumer Financing\" "
    "(Updated August 03, 2016). Banking Policy & Regulations Department. "
    f"{SOURCE_URL}"
)


def load_source_text():
    print(f"\n📂 Loading source text...")
    print(f"📚 Source: {SOURCE_CITATION}")
    text = SOURCE_TXT_PATH.read_text(encoding="utf-8")
    text = text.split("PRUDENTIAL REGULATIONS \nFOR \nCONSUMER FINANCING", 1)[1]
    print(f"✅ Loaded {len(text):,} characters")
    return text


def chunk_by_regulation(text):
    print("\n✂️  Chunking by regulation...")
    parts = re.split(r'(REGULATION\s+[RO]-\d+)', text)
    chunks = []

    intro_text = parts[0].strip()
    for i in range(0, len(intro_text), 1800):
        segment = intro_text[i:i + 2000].strip()
        if len(segment) > 100:
            chunks.append({
                "id": f"intro_{len(chunks)}",
                "section": "Definitions & Minimum Requirements",
                "text": segment,
                "source": SOURCE_CITATION,
            })

    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip()
        chunks.append({
            "id": header.lower().replace(" ", "_").replace("-", "_"),
            "section": header,
            "text": f"{header}\n{body[:2500]}",
            "source": SOURCE_CITATION,
        })

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def save_chunks(chunks):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved → {CHUNKS_PATH}")


if __name__ == "__main__":
    if not SOURCE_TXT_PATH.exists():
        print(f"❌ Source file nahi mili: {SOURCE_TXT_PATH}")
        raise SystemExit(1)

    text = load_source_text()
    chunks = chunk_by_regulation(text)
    save_chunks(chunks)

    print(f"\n🎉 INGESTION COMPLETE! {len(chunks)} chunks ready.")