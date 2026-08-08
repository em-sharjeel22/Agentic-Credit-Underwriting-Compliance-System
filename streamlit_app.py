"""
Sentinel - Agentic Credit Underwriting & Compliance System (corrected)
This version includes robust handling for:
 - missing model file (models/credit_model.pkl)
 - unpickling errors when a custom XGBWrapper class is required
 - optional auto-creation of a small dummy model for local development
 - graceful handling when torchvision or sentence-transformers are unavailable
"""

import os
import json
import re
from pathlib import Path
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import faiss
import matplotlib.pyplot as plt
import streamlit as st

# Try to import torchvision to reduce noisy import errors from transformers.
# If unavailable, we continue but log a warning.
try:
    import torchvision  # noqa: F401
    TORCHVISION_AVAILABLE = True
except Exception:
    TORCHVISION_AVAILABLE = False

# SentenceTransformer may be heavy or unavailable in some environments; import lazily.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from huggingface_hub import InferenceClient

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"
KB_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"
REPORTS_DIR = BASE_DIR / "reports"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# SBP Regulation R-8 thresholds
SBP_R8_PERSONAL_CLEAN_CAP = 2_000_000
SBP_R8_AGGREGATE_CAP = 5_000_000

SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

FEATURE_CATEGORIES = {
    "Demographics": {"SEX", "EDUCATION", "MARRIAGE", "AGE"},
    "Repayment Behavior": {
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "max_pay_delay", "avg_pay_delay", "months_late", "delay_trend",
    },
    "Financial Capacity": {
        "LIMIT_BAL",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "utilization_ratio", "payment_ratio",
    },
}

st.set_page_config(page_title="Sentinel - Credit Underwriting", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------
# Provide a local XGBWrapper class so that unpickling bundles that reference
# a custom XGBWrapper (from other environments) will succeed.
# When a pickle references "main.XGBWrapper" or similar, Python looks up
# that name in the importing module. Defining this class here avoids
# "Can't get attribute 'XGBWrapper' on <module 'main'...>" errors.
# ---------------------------------------------------------------------
class XGBWrapper:
    """
    Minimal wrapper that exposes predict_proba and get_booster, and a feature_names property.
    This mirrors the wrapper used when creating dummy bundles; if your real model
    is a different object, prefer saving a plain xgboost Booster or a sklearn-compatible model.
    """
    def __init__(self, booster, feature_count: int = 29, feature_names: list | None = None):
        self._booster = booster
        if feature_names:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"f{i}" for i in range(feature_count)]

    def predict_proba(self, X_df):
        # Accept DataFrame-like or numpy array
        if hasattr(X_df, "values"):
            arr = X_df.values
        else:
            arr = np.asarray(X_df)
        d = xgb.DMatrix(arr)
        p = self._booster.predict(d)
        return np.vstack([1 - p, p]).T

    def get_booster(self):
        return self._booster

    @property
    def feature_names(self):
        return self._feature_names


# ── Resource loaders ──────────────────────────────────────────

def get_groq_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading risk model...")
def load_risk_model(auto_create_dummy: bool = False):
    """
    Loads the model bundle from models/credit_model.pkl.

    Behavior:
    - If file missing: show a clear Streamlit error and stop, unless auto_create_dummy=True.
    - If unpickling fails due to missing custom class, the local XGBWrapper above
      will allow many common pickles to load successfully.
    - If auto_create_dummy=True, a small synthetic xgboost model is created and saved.
    """
    if not MODEL_PATH.exists():
        st.error(
            f"❌ **Model file not found** at `{MODEL_PATH}`.\n\n"
            "Place `credit_model.pkl` in the `models/` folder, or enable auto_create_dummy=True for local testing."
        )
        if auto_create_dummy:
            try:
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=300, n_features=29, n_informative=10, random_state=42)
                dtrain = xgb.DMatrix(X, label=y)
                params = {"objective": "binary:logistic", "verbosity": 0}
                bst = xgb.train(params, dtrain, num_boost_round=20)
                model = XGBWrapper(bst, feature_count=29)
                bundle = {"model": model, "threshold": 0.5}
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle, MODEL_PATH)
                st.warning("A dummy model was created at models/credit_model.pkl for development.")
            except Exception as e:
                st.error(f"Failed to auto-create dummy model: {e}")
                st.stop()
        else:
            st.stop()

    # Attempt to load the bundle. If the pickle references XGBWrapper, the class above
    # will be available in this module and unpickling should succeed.
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        # Provide a helpful error message and guidance
        st.error(
            "Failed to load model file "
            f"`{MODEL_PATH}`: {e}\n\n"
            "Common causes:\n"
            "- The file is missing or path is incorrect.\n"
            "- The pickle references a custom class that isn't defined in this module.\n\n"
            "Fixes:\n"
            "- Ensure `models/credit_model.pkl` exists and is a joblib dump of a dict with keys 'model' and 'threshold'.\n"
            "- If you created the bundle with a custom wrapper, either recreate the bundle using a plain xgboost Booster or\n"
            "  add the same wrapper class to this file (XGBWrapper is provided as a compatibility helper).\n"
            "You can enable auto_create_dummy=True in load_risk_model() for a local dummy model."
        )
        st.stop()

    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        st.error("Model bundle is invalid. Expected a dict with keys 'model' and 'threshold'.")
        st.stop()

    return bundle["model"], bundle["threshold"]


@st.cache_resource(show_spinner="Loading compliance knowledge base...")
def load_compliance_resources():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        st.warning(
            f"⚠️ **Compliance knowledge base files not found** in `{KB_DIR}`.\n"
            "Regulation search and automated compliance checks will be limited until the FAISS index and metadata are provided."
        )
        return None, [], None

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning(
            "⚠️ `sentence-transformers` is not installed. Semantic search will be disabled. "
            "Install `sentence-transformers` to enable compliance retrieval."
        )
        return None, [], None

    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        st.error(f"Failed to read FAISS index at `{INDEX_PATH}`: {e}")
        return None, [], None

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        st.error(f"Failed to read metadata file `{METADATA_PATH}`: {e}")
        return None, [], None

    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        st.warning(f"Failed to load embedding model `{EMBED_MODEL_NAME}`: {e}")
        return index, chunks, None

    return index, chunks, embed_model


@st.cache_resource(show_spinner=False)
def load_llm_client():
    key = get_groq_key()
    if not key:
        return None
    try:
        return InferenceClient(provider="groq", api_key=key)
    except Exception as e:
        st.warning(f"Failed to initialize LLM client: {e}")
        return None


# ── Agent logic (unchanged, kept robust) ─────────────────────

def engineer_features(raw: dict) -> dict:
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


def run_data_agent(raw: dict) -> dict:
    warnings = []
    if raw.get("LIMIT_BAL", 0) <= 0:
        warnings.append("LIMIT_BAL is zero or negative.")
    if not (18 <= raw.get("AGE", 0) <= 100):
        warnings.append("AGE is outside the plausible range (18-100).")
    if raw.get("EDUCATION") not in [1, 2, 3, 4]:
        warnings.append("EDUCATION code is not in the expected set (1-4).")
    if raw.get("MARRIAGE") not in [1, 2, 3]:
        warnings.append("MARRIAGE code is not in the expected set (1-3).")

    limit_bal = raw.get("LIMIT_BAL", 0)
    sbp_flags = {
        "exceeds_r8_personal_clean_cap": limit_bal > SBP_R8_PERSONAL_CLEAN_CAP,
        "exceeds_r8_aggregate_cap": limit_bal > SBP_R8_AGGREGATE_CAP,
    }
    return {"warnings": warnings, "sbp_flags": sbp_flags}


def run_risk_agent(model, threshold, applicant: dict):
    feature_names = None
    try:
        feature_names = model.get_booster().feature_names
    except Exception:
        try:
            feature_names = getattr(model, "feature_names", None)
        except Exception:
            feature_names = None

    if not feature_names:
        feature_names = list(pd.DataFrame([applicant]).columns)

    X = pd.DataFrame([applicant]).reindex(columns=feature_names)

    proba = float(model.predict_proba(X)[0, 1])
    decision = "REJECT" if proba >= threshold else "APPROVE"

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    raw_contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_values = raw_contribs[0, :-1]

    contributions = sorted(
        zip(feature_names, shap_values, X.iloc[0].values),
        key=lambda x: abs(x[1]), reverse=True,
    )
    return proba, decision, contributions


def normalize_regulation_id(regulation_type: str, regulation_number: str) -> str:
    return f"REGULATION {regulation_type.upper()}-{int(regulation_number)}"


def find_exact_regulation(query: str, chunks: list):
    match = re.search(
        r"(?<![A-Z0-9])(?:REGULATION\s+)?([RO])\s*-?\s*(\d+)(?![A-Z0-9])",
        query.upper(),
    )
    if not match:
        return None
    requested_id = normalize_regulation_id(match.group(1), match.group(2))
    for chunk in chunks:
        if chunk.get("id", "").lower() == requested_id.lower().replace("regulation ", "regulation_").replace("-", "_"):
            return chunk
    for chunk in chunks:
        section = chunk.get("section", "").upper().strip()
        if section == requested_id or section.startswith(requested_id + ":"):
            return chunk
    return None


def retrieve_and_answer(query, index, chunks, embed_model, llm_client, top_k=3):
    if index is None or not chunks or embed_model is None:
        return "Compliance knowledge base is not available.", []

    exact_chunk = find_exact_regulation(query, chunks)

    if exact_chunk is not None:
        retrieved = [exact_chunk]
    else:
        query_vector = embed_model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = index.search(query_vector, top_k)
        retrieved = [chunks[int(i)] for i in indices[0] if 0 <= int(i) < len(chunks)]

    if not retrieved:
        return ("I could not find a matching SBP regulation in the knowledge base.", [])

    context = "\n\n".join(f"[{c.get('section', 'Unknown section')}]\n{c.get('text', '')}" for c in retrieved)
    sources = list(dict.fromkeys(c.get("section", "Unknown section") for c in retrieved))

    if llm_client is None:
        return ("LLM is not configured (missing GROQ_API_KEY). Showing the raw retrieved regulation text instead:\n\n" + context, sources)

    try:
        completion = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": (
                    "You are an SBP compliance officer. Answer ONLY using the provided regulation text. "
                    "If the user explicitly requested a regulation number, explain that regulation clearly in simple language. "
                    "Always cite only regulation numbers present in the provided context."
                )},
                {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content, sources
    except Exception as e:
        return f"LLM completion failed: {e}\n\nRaw retrieved text:\n\n{context}", sources


def run_compliance_agent(index, chunks, embed_model, llm_client, applicant, decision, sbp_flags):
    if sbp_flags.get("exceeds_r8_aggregate_cap"):
        query = "What SBP regulation applies when a customer's aggregate clean credit card and personal loan exposure exceeds Rs 5,000,000?"
    elif sbp_flags.get("exceeds_r8_personal_clean_cap"):
        query = "What SBP regulation applies when a customer's clean credit card limit exceeds Rs 2,000,000?"
    else:
        query = f"What SBP regulations apply when a bank {decision.lower()}s a consumer financing application with credit limit {applicant.get('LIMIT_BAL', 'N/A')}?"
    return retrieve_and_answer(query, index, chunks, embed_model, llm_client)


def run_reporting_agent(llm_client, decision, proba, contributions, compliance_answer):
    if llm_client is None:
        return None
    factors_text = ", ".join(f"{f[0]} ({'raises' if f[1] > 0 else 'lowers'} risk)" for f in contributions[:5])
    try:
        completion = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": (
                    "You are a senior credit underwriting officer writing a short decision memo for a bank manager. "
                    "Combine the risk assessment and compliance findings into 3-4 clear sentences. Use only the facts given below - never invent numbers or regulations."
                )},
                {"role": "user", "content": (
                    f"Decision: {decision}\nDefault probability: {proba * 100:.1f}%\nKey factors: {factors_text}\nCompliance findings: {compliance_answer}"
                )},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM reporting failed: {e}"


REQUIRED_APPLICANT_FIELDS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def parse_applicant_from_text(description: str, llm_client):
    if llm_client is None:
        return None, "LLM is not configured (missing GROQ_API_KEY)."
    if not description.strip():
        return None, "Please enter a description first."

    system_prompt = (
        "You are a data-entry assistant. Extract these 23 fields from the applicant description as a JSON object with EXACTLY these keys: "
        "LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, "
        "BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, "
        "PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6.\n"
        "SEX: 1=male, 2=female. EDUCATION: 1=graduate school, 2=university, 3=high school, 4=other. "
        "MARRIAGE: 1=married, 2=single, 3=other. PAY_0/PAY_2-6: -1=paid on time, 1 or higher=that many months late.\n"
        "If a field is not mentioned, use a reasonable default (0 for amounts, -1 for payment status, 35 for age, 1 for sex, 2 for education, 2 for marriage).\n"
        "Output ONLY the JSON object - no markdown fences, no extra text."
    )

    try:
        completion = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": description}],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, "The AI's response wasn't valid JSON - please try rephrasing."
    except Exception as e:
        return None, f"AI extraction failed: {e}"

    missing = [k for k in REQUIRED_APPLICANT_FIELDS if k not in parsed]
    if missing:
        return None, f"The AI's response was missing fields: {', '.join(missing)}"

    return parsed, None


# ── Visualization helpers (unchanged) ───────────────────────

def render_shap_chart(contributions):
    top = contributions[:8]
    labels = [f[0] for f in top][::-1]
    values = [f[1] for f in top][::-1]
    colors = ["#d64545" if v > 0 else "#2f9e58" for v in values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Impact on default risk (SHAP value)")
    ax.set_title("Why the model reached this score")
    fig.tight_layout()
    return fig


def verify_summary_citations(polished_summary: str, compliance_sources: list) -> bool:
    cited = {c.upper() for c in re.findall(r"REGULATION\s+[RO]-\d+", polished_summary, re.IGNORECASE)}
    sources = {s.upper() for s in compliance_sources}
    return cited.issubset(sources)


def payment_history_dataframe(applicant):
    months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1 (latest)"]
    bills = [applicant[f"BILL_AMT{i}"] for i in range(6, 0, -1)]
    payments = [applicant[f"PAY_AMT{i}"] for i in range(6, 0, -1)]
    return pd.DataFrame({"Bill amount": bills, "Amount paid": payments}, index=months)


def repayment_status_dataframe(applicant):
    months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1 (latest)"]
    status = [applicant["PAY_6"], applicant["PAY_5"], applicant["PAY_4"], applicant["PAY_3"], applicant["PAY_2"], applicant["PAY_0"]]
    return pd.DataFrame({"Repayment status (-1=on time, 1+=months late)": status}, index=months)


def limit_vs_sbp_caps_dataframe(applicant):
    return pd.DataFrame(
        {"Amount (PKR)": [applicant["LIMIT_BAL"], SBP_R8_PERSONAL_CLEAN_CAP, SBP_R8_AGGREGATE_CAP]},
        index=["Applicant's limit", "SBP clean cap (R-8)", "SBP aggregate cap (R-8)"],
    )


def applicant_summary_line(applicant):
    sex = SEX_LABELS.get(applicant.get("SEX"), "Unknown")
    education = EDUCATION_LABELS.get(applicant.get("EDUCATION"), "Unknown")
    marriage = MARRIAGE_LABELS.get(applicant.get("MARRIAGE"), "Unknown")
    age = applicant.get("AGE", "Unknown")
    return f"{sex}, {age} years old, {education}, {marriage}"


def categorize_contributions(contributions):
    totals = {name: 0.0 for name in FEATURE_CATEGORIES}
    for feature, shap_value, _ in contributions:
        for category, members in FEATURE_CATEGORIES.items():
            if feature in members:
                totals[category] += shap_value
                break
    return totals


def render_category_chart(category_totals):
    labels = list(category_totals.keys())
    values = list(category_totals.values())
    colors = ["#d64545" if v > 0 else "#2f9e58" for v in values]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Net impact on default risk (summed SHAP)")
    ax.set_title("Risk by feature category")
    fig.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────

st.title("Sentinel - Credit Underwriting")
st.caption("Live risk scoring, per-applicant SHAP explanations, and SBP compliance checking.")

with st.expander("How this pipeline works (4 agents)", expanded=True):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1. Data Agent**")
        st.caption("Validates the applicant's numbers, checks them against SBP's exposure caps, and computes the engineered features.")
    with p2:
        st.markdown("**2. Risk Agent**")
        st.caption("The trained XGBoost model scores default probability and explains the score with SHAP.")
    with p3:
        st.markdown("**3. Compliance Agent**")
        st.caption("Retrieves the most relevant SBP regulation and explains it in plain language via an LLM.")
    with p4:
        st.markdown("**4. Reporting Agent**")
        st.caption("Combines the three outputs above into one short, readable memo.")

# Load resources. For local development you can enable auto_create_dummy=True to create a small model.
model, threshold = load_risk_model(auto_create_dummy=False)
index, chunks, embed_model = load_compliance_resources()
llm_client = load_llm_client()

live_tab, chat_tab, reports_tab = st.tabs(["Live Underwriting", "Ask about Regulations", "Model Training Reports"])

with live_tab:
    st.subheader("Applicant profile")
    st.caption("Enter an applicant's details and click Analyze. Every chart below is computed live from these exact numbers.")

    st.markdown("**🤖 AI quick-fill (optional)**")
    st.caption("Describe the applicant in plain English and let AI fill in the fields below.")
    quickfill_text = st.text_area(
        "Applicant description",
        placeholder=("e.g. 35-year-old married university graduate, credit limit 50000, always pays on time, spends around 15000 a month on the card."),
        label_visibility="collapsed",
        key="quickfill_text",
    )
    if st.button("Parse with AI"):
        parsed, error = parse_applicant_from_text(quickfill_text, llm_client)
        if error:
            st.warning(error)
        else:
            for field_key, value in parsed.items():
                st.session_state[field_key] = value
            st.success("Fields filled in below — review and adjust before clicking Analyze.")

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            limit_bal = st.number_input("Credit limit (LIMIT_BAL)", min_value=0, value=50000, step=1000, key="LIMIT_BAL")
            age = st.number_input("Age", min_value=18, max_value=100, value=35, key="AGE")
        with c2:
            sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: SEX_LABELS[x], key="SEX")
            education = st.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: EDUCATION_LABELS[x], key="EDUCATION")
        with c3:
            marriage = st.selectbox("Marital status", options=[1, 2, 3], format_func=lambda x: MARRIAGE_LABELS[x], key="MARRIAGE")

        st.markdown("**Repayment status, last 6 months** (-1 = paid on time, 1+ = months late)")
        pay_cols = st.columns(6)
        pay_labels = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        pay_values = []
        for col, label in zip(pay_cols, pay_labels):
            with col:
                pay_values.append(st.number_input(label, min_value=-2, max_value=9, value=0, key=label))

        st.markdown("**Bill amount, last 6 months**")
        bill_cols_ui = st.columns(6)
        bill_values = []
        for i, col in enumerate(bill_cols_ui, start=1):
            with col:
                bill_values.append(st.number_input(f"BILL_AMT{i}", min_value=0, value=10000, key=f"BILL_AMT{i}"))

        st.markdown("**Amount paid, last 6 months**")
        payamt_cols_ui = st.columns(6)
        payamt_values = []
        for i, col in enumerate(payamt_cols_ui, start=1):
            with col:
                payamt_values.append(st.number_input(f"PAY_AMT{i}", min_value=0, value=2000, key=f"PAY_AMT{i}"))

        submitted = st.form_submit_button("Analyze", use_container_width=True)

    if submitted:
        raw_applicant = {
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
            "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_values[0], "PAY_2": pay_values[1], "PAY_3": pay_values[2],
            "PAY_4": pay_values[3], "PAY_5": pay_values[4], "PAY_6": pay_values[5],
        }
        for i in range(1, 7):
            raw_applicant[f"BILL_AMT{i}"] = bill_values[i - 1]
            raw_applicant[f"PAY_AMT{i}"] = payamt_values[i - 1]

        data_result = run_data_agent(raw_applicant)
        applicant = engineer_features(raw_applicant)
        proba, decision, contributions = run_risk_agent(model, threshold, applicant)
        compliance_answer, compliance_sources = run_compliance_agent(index, chunks, embed_model, llm_client, applicant, decision, data_result["sbp_flags"])
        polished_summary = run_reporting_agent(llm_client, decision, proba, contributions, compliance_answer)

        for w in data_result["warnings"]:
            st.warning(w)

        st.caption(applicant_summary_line(raw_applicant))

        if polished_summary:
            st.markdown("---")
            st.markdown("**Underwriting memo**")
            if not verify_summary_citations(polished_summary, compliance_sources):
                st.warning("This summary may reference a regulation that was not actually retrieved — verify against the Compliance check section below.")
            st.info(polished_summary)

        st.markdown("---")
        st.markdown("**Engineered features** (computed live from the 6-month history)")
        engineered_names = ["utilization_ratio", "avg_pay_delay", "max_pay_delay", "months_late", "payment_ratio", "delay_trend"]
        engineered_df = pd.DataFrame({"Feature": engineered_names, "Value": [applicant[name] for name in engineered_names]})
        st.dataframe(engineered_df, hide_index=True, use_container_width=True)

        res1, res2 = st.columns([1, 2])
        with res1:
            st.metric("Default probability", f"{proba * 100:.1f}%")
            if decision == "APPROVE":
                st.success(f"Decision: {decision}")
            else:
                st.error(f"Decision: {decision}")
            st.progress(min(proba, 1.0))
            st.markdown(f"**Credit utilization: {applicant['utilization_ratio'] * 100:.1f}%**")
            st.progress(min(applicant["utilization_ratio"], 1.0))
            st.markdown("**Payment amounts (live)**")
            st.line_chart(payment_history_dataframe(applicant))

        with res2:
            st.markdown("**Why this score (live SHAP explanation)**")
            fig = render_shap_chart(contributions)
            st.pyplot(fig)
            st.markdown("**Repayment status trend (live)**")
            st.bar_chart(repayment_status_dataframe(applicant))

        st.markdown("---")
        cat1, cat2 = st.columns(2)
        with cat1:
            st.markdown("**Risk by feature category (live)**")
            category_totals = categorize_contributions(contributions)
            st.pyplot(render_category_chart(category_totals))
        with cat2:
            st.markdown("**Requested limit vs. SBP caps (live)**")
            st.bar_chart(limit_vs_sbp_caps_dataframe(applicant))

        st.markdown("---")
        st.markdown("**Compliance check**")
        st.write(compliance_answer)
        st.caption(f"Sources: {', '.join(compliance_sources)}")

with chat_tab:
    st.subheader("Ask about SBP regulations")
    st.caption("Ask any question about the SBP Prudential Regulations for Consumer Financing directly - independent of the applicant form.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")

    question = st.chat_input("e.g. What is the maximum tenure for auto financing?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching regulations..."):
                answer, sources = retrieve_and_answer(question, index, chunks, embed_model, llm_client)
            st.write(answer)
            st.caption(f"Sources: {', '.join(sources)}")
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

with reports_tab:
    st.subheader("Model training reports")
    st.caption("These figures describe the overall model, generated once during training - they do not change with the applicant form above.")

    descriptions = {
        "confusion_matrix.png": "Confusion matrix on the held-out test set.",
        "correlation_heatmap.png": "Correlation heatmap of the key predictive variables.",
        "shap_feature_importance.png": "Overall SHAP feature importance across all test applicants.",
        "shap_summary.png": "SHAP summary plot across the whole test set.",
        "target_distribution.png": "Distribution of the target outcome in the training data.",
        "missing_values.png": "Missing-value profile of the training data.",
    }

    report_files = sorted(REPORTS_DIR.glob("*.png")) if REPORTS_DIR.exists() else []
    if not report_files:
        st.info("No report images found in the reports/ folder.")
    else:
        cols = st.columns(2)
        for i, path in enumerate(report_files):
            with cols[i % 2]:
                st.image(str(path), caption=descriptions.get(path.name, path.name), use_container_width=True)
