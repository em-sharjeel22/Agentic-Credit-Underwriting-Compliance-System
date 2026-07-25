import json
import os
import faiss
import joblib
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from constants import (
    EMBED_MODEL_NAME,
    INDEX_PATH,
    METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
_model = None
_threshold = None
_feature_names = None
_index = None
_chunks = None
_embed_model = None
_llm_client = None
def load_risk_resources():
    global _model, _threshold, _feature_names
    if _model is None:
        print("📂 Loading Risk Agent (XGBoost model)...")
        bundle = joblib.load(MODEL_PATH)
        _model = bundle["model"]
        _threshold = bundle["threshold"]
        _feature_names = _model.get_booster().feature_names
        print(f"✅ Risk model ready | threshold: {_threshold}")
    return _model, _threshold, _feature_names
def load_compliance_resources():
    global _index, _chunks, _embed_model, _llm_client
    if _index is None:
        print("\n📂 Loading Compliance Agent (FAISS + embeddings)...")
        _index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        api_key = os.environ.get("GROQ_API_KEY")
        _llm_client = (
            InferenceClient(provider="groq", api_key=api_key)
            if api_key
            else None
        )
        print("✅ Compliance agent ready\n")
    return _index, _chunks, _embed_model, _llm_client
def preload_all():
    load_risk_resources()
    load_compliance_resources()