"""
Sentinel - Agentic Credit Underwriting & Compliance System

Live Streamlit dashboard: enter an applicant's profile and get a
real-time risk score, a SHAP explanation for that specific applicant,
and an SBP regulatory compliance check - all computed live from the
form inputs, not pre-generated.
"""

import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import pandas as pd
import xgboost as xgb
import faiss
import matplotlib.pyplot as plt
import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"
KB_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"
REPORTS_DIR = BASE_DIR / "reports"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# SBP Regulation R-8 thresholds, hardcoded from the source document so
# the numeric compliance check is deterministic rather than LLM-guessed.
SBP_R8_PERSONAL_CLEAN_CAP = 2_000_000
SBP_R8_AGGREGATE_CAP = 5_000_000

st.set_page_config(page_title="Sentinel - Credit Underwriting", page_icon="🏦", layout="wide")


def get_groq_key():
    """Streamlit Cloud uses st.secrets; local dev falls back to an environment variable."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading risk model...")
def load_risk_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["threshold"]


@st.cache_resource(show_spinner="Loading compliance knowledge base...")
def load_compliance_resources():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, chunks, embed_model


@st.cache_resource(show_spinner=False)
def load_llm_client():
    key = get_groq_key()
    if not key:
        return None
    return InferenceClient(provider="groq", api_key=key)


# ── Agent logic ────────────────────────────────────────────
# Kept self-contained in this file on purpose: importing from src/agents/
# would add a cross-folder dependency that is fragile on a fresh cloud
# checkout. Each function below plays the same role as the matching
# agent in src/agents/orchestrator.py.

def engineer_features(raw: dict) -> dict:
    """Turns the 23 raw fields into the same engineered features used at training time."""
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
    """Validates the applicant's numbers and flags any SBP R-8 exposure-cap breaches."""
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
    """Scores the applicant and returns SHAP contributions computed natively by XGBoost
    (via pred_contribs), which avoids the shap library's fragile model-format parser."""
    feature_names = model.get_booster().feature_names
    X = pd.DataFrame([applicant])[feature_names]

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


def retrieve_and_answer(query, index, chunks, embed_model, llm_client, top_k=3):
    """Shared retrieval + LLM-answer logic. Used by BOTH the compliance agent
    (triggered automatically from an applicant's data) and the standalone
    regulation chatbot (free-form questions typed by the user) - so both entry
    points are backed by the exact same compliance engine."""
    query_vector = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    context = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c in retrieved)
    sources = [c["section"] for c in retrieved]

    if llm_client is None:
        return (
            "LLM is not configured (missing GROQ_API_KEY). Showing the raw retrieved "
            "regulation text instead:\n\n" + context,
            sources,
        )

    completion = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": (
                "You are an SBP compliance officer. Answer ONLY using the regulation "
                "text provided below. Always cite the regulation number."
            )},
            {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content, sources


def run_compliance_agent(index, chunks, embed_model, llm_client, applicant, decision, sbp_flags):
    """Builds an applicant-specific compliance question, then delegates to
    retrieve_and_answer for the actual retrieval + generation."""
    if sbp_flags.get("exceeds_r8_aggregate_cap"):
        query = (
            "What SBP regulation applies when a customer's aggregate clean "
            "credit card and personal loan exposure exceeds Rs 5,000,000?"
        )
    elif sbp_flags.get("exceeds_r8_personal_clean_cap"):
        query = (
            "What SBP regulation applies when a customer's clean credit "
            "card limit exceeds Rs 2,000,000?"
        )
    else:
        query = (
            f"What SBP regulations apply when a bank {decision.lower()}s a consumer "
            f"financing application with credit limit {applicant.get('LIMIT_BAL', 'N/A')}?"
        )
    return retrieve_and_answer(query, index, chunks, embed_model, llm_client)


def run_reporting_agent(llm_client, decision, proba, contributions, compliance_answer):
    """Synthesizes the Risk Agent and Compliance Agent outputs into one polished,
    human-readable underwriting memo. Uses only the facts already produced by
    the other agents - it summarizes, it does not add new information."""
    if llm_client is None:
        return None

    factors_text = ", ".join(
        f"{f[0]} ({'raises' if f[1] > 0 else 'lowers'} risk)" for f in contributions[:5]
    )

    completion = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": (
                "You are a senior credit underwriting officer writing a short "
                "decision memo for a bank manager. Combine the risk assessment "
                "and compliance findings into 3-4 clear sentences. Use only the "
                "facts given below - never invent numbers or regulations."
            )},
            {"role": "user", "content": (
                f"Decision: {decision}\n"
                f"Default probability: {proba * 100:.1f}%\n"
                f"Key factors: {factors_text}\n"
                f"Compliance findings: {compliance_answer}"
            )},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# ── Live chart builders ────────────────────────────────────

def render_shap_chart(contributions):
    """Bar chart of THIS applicant's top SHAP factors - recomputed on every submission."""
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
    """Lightweight hallucination guard: checks that every 'REGULATION X-N' the
    polished memo cites was actually among the retrieved sources. This catches
    the highest-stakes failure mode (inventing a compliance citation) even
    though the summary step is only asked to rephrase, not add new facts."""
    import re
    cited = {c.upper() for c in re.findall(r"REGULATION\s+[RO]-\d+", polished_summary, re.IGNORECASE)}
    sources = {s.upper() for s in compliance_sources}
    return cited.issubset(sources)


def payment_history_dataframe(applicant):
    """Six-month bill/payment AMOUNT series built directly from the form inputs."""
    months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1 (latest)"]
    bills = [applicant[f"BILL_AMT{i}"] for i in range(6, 0, -1)]
    payments = [applicant[f"PAY_AMT{i}"] for i in range(6, 0, -1)]
    return pd.DataFrame({"Bill amount": bills, "Amount paid": payments}, index=months)


def repayment_status_dataframe(applicant):
    """Six-month repayment STATUS series - a different lens from the amounts
    chart above: this shows behavior (on-time vs. months late), which is what
    max_pay_delay and delay_trend are actually derived from."""
    months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1 (latest)"]
    status = [
        applicant["PAY_6"], applicant["PAY_5"], applicant["PAY_4"],
        applicant["PAY_3"], applicant["PAY_2"], applicant["PAY_0"],
    ]
    return pd.DataFrame({"Repayment status (-1=on time, 1+=months late)": status}, index=months)


def payment_delay_dataframe(applicant):
    """Delay-status series across the same six months, oldest to newest —
    a behavior-pattern view, distinct from the dollar-amount view above."""
    months = ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1 (latest)"]
    delay_keys = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
    delays = [applicant[k] for k in delay_keys]
    return pd.DataFrame({"Payment delay (months)": delays}, index=months)


def limit_vs_sbp_caps_dataframe(applicant):
    """Where this applicant's requested limit sits relative to the two
    SBP R-8 exposure caps — a compliance-context view."""
    return pd.DataFrame(
        {"Amount (PKR)": [applicant["LIMIT_BAL"], SBP_R8_PERSONAL_CLEAN_CAP, SBP_R8_AGGREGATE_CAP]},
        index=["Applicant's limit", "SBP clean cap (R-8)", "SBP aggregate cap (R-8)"],
    )


# ── UI ─────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">Sentinel - Credit Underwriting</h1>
        <p style="margin:0; font-size:1.02rem;">Live risk scoring, per-applicant SHAP
        explanations, and SBP compliance checking.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

model, threshold = load_risk_model()
index, chunks, embed_model = load_compliance_resources()
llm_client = load_llm_client()

live_tab, chat_tab, reports_tab = st.tabs(
    ["Live Underwriting", "Ask about Regulations", "Model Training Reports"]
)

with live_tab:
    st.subheader("Applicant profile")
    st.caption(
        "Enter an applicant's details and click Analyze. Every chart below is "
        "computed live from these exact numbers."
    )

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            limit_bal = st.number_input("Credit limit (LIMIT_BAL)", min_value=0, value=50000, step=1000)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
        with c2:
            sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = st.selectbox(
                "Education", options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x],
            )
        with c3:
            marriage = st.selectbox(
                "Marital status", options=[1, 2, 3],
                format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
            )

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
                bill_values.append(st.number_input(f"BILL_AMT{i}", min_value=0, value=10000, key=f"bill_{i}"))

        st.markdown("**Amount paid, last 6 months**")
        payamt_cols_ui = st.columns(6)
        payamt_values = []
        for i, col in enumerate(payamt_cols_ui, start=1):
            with col:
                payamt_values.append(st.number_input(f"PAY_AMT{i}", min_value=0, value=2000, key=f"payamt_{i}"))

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
        compliance_answer, compliance_sources = run_compliance_agent(
            index, chunks, embed_model, llm_client, applicant, decision, data_result["sbp_flags"]
        )
        polished_summary = run_reporting_agent(
            llm_client, decision, proba, contributions, compliance_answer
        )

        for w in data_result["warnings"]:
            st.warning(w)

        if polished_summary:
            st.markdown("---")
            st.markdown("**Underwriting memo**")
            if not verify_summary_citations(polished_summary, compliance_sources):
                st.warning(
                    "This summary may reference a regulation that was not actually "
                    "retrieved — verify against the Compliance check section below."
                )
            st.info(polished_summary)

        st.markdown("---")
        st.markdown("**Engineered features** (computed live from the 6-month history)")
        engineered_names = [
            "utilization_ratio", "avg_pay_delay", "max_pay_delay",
            "months_late", "payment_ratio", "delay_trend",
        ]
        engineered_df = pd.DataFrame({
            "Feature": engineered_names,
            "Value": [applicant[name] for name in engineered_names],
        })
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
        st.markdown("**Compliance check**")
        st.write(compliance_answer)
        st.caption(f"Sources: {', '.join(compliance_sources)}")

with chat_tab:
    st.subheader("Ask about SBP regulations")
    st.caption(
        "Ask any question about the SBP Prudential Regulations for Consumer "
        "Financing directly - independent of the applicant form."
    )

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

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

with reports_tab:
    st.subheader("Model training reports")
    st.caption(
        "These figures describe the overall model, generated once during training - "
        "they do not change with the applicant form above."
    )

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