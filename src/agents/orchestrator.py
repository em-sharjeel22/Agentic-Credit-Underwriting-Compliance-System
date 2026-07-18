# ============================================
# LANGGRAPH ORCHESTRATOR
# Risk agent and compliance agent combined into one workflow.
# ============================================

import json
import os
from typing import List, TypedDict

import faiss
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "credit_model.pkl")
KB_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(KB_DIR, "chunk_metadata.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class UnderwritingState(TypedDict):
    applicant: dict
    risk_probability: float
    risk_decision: str
    top_risk_factors: List[dict]
    compliance_question: str
    compliance_answer: str
    compliance_sources: List[str]
    final_report: str


print("Loading risk agent...")
_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_threshold = _bundle["threshold"]
_feature_names = _model.get_booster().feature_names
print(f"Risk model ready with threshold {_threshold}")

print("Loading compliance agent...")
_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as handle:
    _chunks = json.load(handle)


def _build_local_query_vector(query: str) -> np.ndarray:
    tokens = [token.lower() for token in query.replace("\n", " ").split() if token]
    vector = np.zeros(64, dtype="float32")
    for token in tokens:
        index = abs(hash(token)) % 64
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

try:
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as exc:
    _embed_model = None
    print(f"Using local embedding fallback because the model could not be loaded: {exc}")

_llm_client = InferenceClient(provider="groq", api_key=os.environ["GROQ_API_KEY"])
print("Compliance agent ready")


def risk_agent(state: UnderwritingState) -> dict:
    X = pd.DataFrame([state["applicant"]])[_feature_names]

    proba = _model.predict_proba(X)[0, 1]
    decision = "REJECT" if proba >= _threshold else "APPROVE"

    booster = _model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=_feature_names)
    raw = booster.predict(dmatrix, pred_contribs=True)
    shap_vals = raw[0, :-1]

    contributions = sorted(
        zip(_feature_names, shap_vals, X.iloc[0].values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:5]

    return {
        "risk_probability": float(proba),
        "risk_decision": decision,
        "top_risk_factors": [
            {"feature": feature, "impact": float(value), "value": float(raw_value)}
            for feature, value, raw_value in contributions
        ],
    }


def compliance_agent(state: UnderwritingState) -> dict:
    query = state.get("compliance_question") or (
        f"What SBP regulations apply when a bank {state['risk_decision'].lower()}s "
        f"a consumer financing application with credit limit {state['applicant'].get('LIMIT_BAL', 'N/A')}?"
    )

    if _embed_model is not None:
        try:
            query_vector = _embed_model.encode([query], convert_to_numpy=True).astype("float32")
        except Exception:
            query_vector = _build_local_query_vector(query).reshape(1, -1)
    else:
        query_vector = _build_local_query_vector(query).reshape(1, -1)

    _, indices = _index.search(query_vector, 3)
    retrieved = [_chunks[index] for index in indices[0]]
    context = "\n\n".join([f"[{chunk['section']}]\n{chunk['text']}" for chunk in retrieved])

    completion = _llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an SBP compliance officer. Use only the provided regulation text "
                    "and quote the regulation number where relevant."
                ),
            },
            {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )

    return {
        "compliance_question": query,
        "compliance_answer": completion.choices[0].message.content,
        "compliance_sources": [chunk["section"] for chunk in retrieved],
    }


def final_report(state: UnderwritingState) -> dict:
    factors_text = "\n".join([
        f"   {'Positive' if factor['impact'] > 0 else 'Negative'} {factor['feature']}: {factor['impact']:+.3f}"
        for factor in state["top_risk_factors"]
    ])

    report = f"""
{'=' * 55}
CREDIT UNDERWRITING REPORT
{'=' * 55}

RISK ASSESSMENT
   Decision            : {state['risk_decision']}
   Default Probability : {state['risk_probability'] * 100:.1f}%

   Top Factors:
{factors_text}

COMPLIANCE CHECK
   {state['compliance_answer']}

   Sources: {', '.join(state['compliance_sources'])}
{'=' * 55}
"""
    return {"final_report": report}


def build_graph():
    workflow = StateGraph(UnderwritingState)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("compliance_agent", compliance_agent)
    workflow.add_node("final_report", final_report)

    workflow.add_edge(START, "risk_agent")
    workflow.add_edge("risk_agent", "compliance_agent")
    workflow.add_edge("compliance_agent", "final_report")
    workflow.add_edge("final_report", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    sample_applicant = test_df.drop("target", axis=1).iloc[0].to_dict()

    print("Running underwriting workflow...")
    result = graph.invoke({"applicant": sample_applicant})

    print(result["final_report"])