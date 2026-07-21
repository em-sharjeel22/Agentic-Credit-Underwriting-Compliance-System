import os
import json
from typing import TypedDict, List, Dict

import joblib
import pandas as pd
import xgboost as xgb
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "credit_model.pkl")
KB_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(KB_DIR, "chunk_metadata.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# SBP Regulation R-8 thresholds — source document se hardcoded.
# Numeric compliance checks LLM pe depend nahi karne chahiye,
# isliye yeh CODE mein deterministic check hai.
SBP_R8_PERSONAL_CLEAN_CAP = 2_000_000
SBP_R8_AGGREGATE_CAP = 5_000_000


class UnderwritingState(TypedDict):
    applicant: dict
    data_warnings: List[str]
    sbp_flags: Dict[str, bool]
    risk_probability: float
    risk_decision: str
    top_risk_factors: List[dict]
    compliance_question: str
    compliance_answer: str
    compliance_sources: List[str]
    final_report: str


print("📂 Loading Risk Agent (XGBoost model)...")
_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_threshold = _bundle["threshold"]
_feature_names = _model.get_booster().feature_names
print(f"✅ Risk model ready | threshold: {_threshold}")

print("\n📂 Loading Compliance Agent (FAISS + embeddings)...")
_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _chunks = json.load(f)
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
_llm_client = InferenceClient(provider="groq", api_key=os.environ["GROQ_API_KEY"])
print("✅ Compliance agent ready\n")


def engineer_features(raw: dict) -> dict:
    """Real-time feature engineering — Spark ke bina (API ke liye)"""
    bill_cols = [raw[f"BILL_AMT{i}"] for i in range(1, 7)]
    pay_amt_cols = [raw[f"PAY_AMT{i}"] for i in range(1, 7)]
    pay_status_cols = [raw["PAY_0"]] + [raw[f"PAY_{i}"] for i in range(2, 7)]

    avg_bill = sum(bill_cols) / 6
    utilization_ratio = round(avg_bill / raw["LIMIT_BAL"], 4) if raw["LIMIT_BAL"] else 0.0
    avg_pay_delay = round(sum(pay_status_cols) / 6, 3)
    max_pay_delay = max(pay_status_cols)
    months_late = sum(1 for p in pay_status_cols if p > 0)
    total_paid = sum(pay_amt_cols)
    total_billed = sum(bill_cols)
    payment_ratio = round(total_paid / total_billed, 4) if total_billed > 0 else 1.0
    delay_trend = raw["PAY_0"] - raw["PAY_6"]

    return {
        **raw,
        "utilization_ratio": utilization_ratio,
        "avg_pay_delay": avg_pay_delay,
        "max_pay_delay": max_pay_delay,
        "months_late": months_late,
        "payment_ratio": payment_ratio,
        "delay_trend": delay_trend,
    }


def data_agent(state: UnderwritingState) -> dict:
    """
    Pipeline mein sabse pehle chalta hai:
    1. Business-logic validation (FastAPI sirf type-check karta hai)
    2. SBP R-8 thresholds ka deterministic (code-based) check —
       taake Compliance Agent ko exact facts milein, guess nahi
    """
    print("🗂️  [Data Agent] Validating + flagging applicant data...")
    applicant = state["applicant"]
    warnings = []

    if applicant.get("LIMIT_BAL", 0) <= 0:
        warnings.append("LIMIT_BAL is zero or negative")
    if not (18 <= applicant.get("AGE", 0) <= 100):
        warnings.append("AGE is outside plausible range (18-100)")
    if applicant.get("EDUCATION") not in [1, 2, 3, 4]:
        warnings.append("EDUCATION code not in expected set (1-4)")
    if applicant.get("MARRIAGE") not in [1, 2, 3]:
        warnings.append("MARRIAGE code not in expected set (1-3)")

    limit_bal = applicant.get("LIMIT_BAL", 0)
    sbp_flags = {
        "exceeds_r8_personal_clean_cap": limit_bal > SBP_R8_PERSONAL_CLEAN_CAP,
        "exceeds_r8_aggregate_cap": limit_bal > SBP_R8_AGGREGATE_CAP,
    }

    print(f"   → Warnings: {len(warnings)} | R-8 clean cap exceeded: {sbp_flags['exceeds_r8_personal_clean_cap']}")
    return {"data_warnings": warnings, "sbp_flags": sbp_flags}


def risk_agent(state: UnderwritingState) -> dict:
    print("🔍 [Risk Agent] Analyzing applicant...")
    X = pd.DataFrame([state["applicant"]])[_feature_names]

    proba = _model.predict_proba(X)[0, 1]
    decision = "REJECT" if proba >= _threshold else "APPROVE"

    booster = _model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=_feature_names)
    raw = booster.predict(dmatrix, pred_contribs=True)
    shap_vals = raw[0, :-1]

    contributions = sorted(
        zip(_feature_names, shap_vals, X.iloc[0].values),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]

    print(f"   → Probability: {proba*100:.1f}% | Decision: {decision}")
    return {
        "risk_probability": float(proba),
        "risk_decision": decision,
        "top_risk_factors": [
            {"feature": f, "impact": float(v), "value": float(val)}
            for f, v, val in contributions
        ],
    }


def compliance_agent(state: UnderwritingState) -> dict:
    print("📖 [Compliance Agent] Checking SBP regulations...")
    flags = state.get("sbp_flags", {})

    if flags.get("exceeds_r8_aggregate_cap"):
        query = (
            "What SBP regulation applies when a customer's aggregate clean "
            "credit card and personal loan exposure exceeds Rs 5,000,000?"
        )
    elif flags.get("exceeds_r8_personal_clean_cap"):
        query = (
            "What SBP regulation applies when a customer's clean credit "
            "card limit exceeds Rs 2,000,000?"
        )
    else:
        query = state.get("compliance_question") or (
            f"What SBP regulations apply when a bank {state['risk_decision'].lower()}s "
            f"a consumer financing application with credit limit {state['applicant'].get('LIMIT_BAL', 'N/A')}?"
        )

    query_vector = _embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(query_vector, 3)
    retrieved = [_chunks[i] for i in indices[0]]
    context = "\n\n".join([f"[{c['section']}]\n{c['text']}" for c in retrieved])

    completion = _llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": (
                "Tu ek SBP compliance officer hai. SIRF diye gaye regulation text "
                "ke basis pe jawab do, regulation number quote karo."
            )},
            {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.1,
    )

    print(f"   → Checked against {len(retrieved)} regulations")
    return {
        "compliance_question": query,
        "compliance_answer": completion.choices[0].message.content,
        "compliance_sources": [c["section"] for c in retrieved],
    }


def final_report(state: UnderwritingState) -> dict:
    print("📄 [Report Agent] Compiling final report...")

    factors_text = "\n".join([
        f"   {'🔴' if f['impact'] > 0 else '🟢'} {f['feature']}: {f['impact']:+.3f}"
        for f in state["top_risk_factors"]
    ])
    warnings_text = "\n".join(f"   ⚠️  {w}" for w in state.get("data_warnings", [])) or "   None"

    report = f"""
{'='*55}
CREDIT UNDERWRITING REPORT
{'='*55}

DATA VALIDATION
{warnings_text}

RISK ASSESSMENT
   Decision            : {state['risk_decision']}
   Default Probability : {state['risk_probability']*100:.1f}%

   Top Factors:
{factors_text}

COMPLIANCE CHECK
   {state['compliance_answer']}

   Sources: {', '.join(state['compliance_sources'])}
{'='*55}
"""
    return {"final_report": report}


def build_graph():
    workflow = StateGraph(UnderwritingState)
    workflow.add_node("data_agent", data_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("compliance_agent", compliance_agent)
    workflow.add_node("final_report", final_report)

    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "risk_agent")
    workflow.add_edge("risk_agent", "compliance_agent")
    workflow.add_edge("compliance_agent", "final_report")
    workflow.add_edge("final_report", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    sample_applicant = test_df.drop("target", axis=1).iloc[0].to_dict()

    print("🚀 Running full underwriting pipeline...\n")
    result = graph.invoke({"applicant": sample_applicant})
    print(result["final_report"])