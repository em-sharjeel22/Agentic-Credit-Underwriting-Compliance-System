import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

base = Path(r'C:\Users\M.LAPTOP\Downloads\Saylani-FYP')
index = faiss.read_index(str(base / 'data' / 'knowledge_base' / 'faiss_index.bin'))
chunks = json.loads((base / 'data' / 'knowledge_base' / 'chunk_metadata.json').read_text(encoding='utf-8'))
model = SentenceTransformer('all-MiniLM-L6-v2')
query = 'What are the limits for consumer financing?'
q = model.encode([query], convert_to_numpy=True).astype('float32')
distances, indices = index.search(q, 3)
for item in indices[0]:
    chunk = chunks[item]
    print('--- ' + chunk['section'] + ' ---')
    print(chunk['text'][:900])
    print()
