import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
REPORTS_DIR = BASE_DIR / "reports"
INDEX_PATH = KB_DIR / "faiss_index.bin"
METADATA_PATH = KB_DIR / "chunk_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="SBP Consumer Financing Assistant", page_icon="🏦", layout="wide")


def load_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        st.error("Knowledge base files are missing. Run the pipeline first.")
        st.stop()

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception:
        model = None

    return index, chunks, model


def build_local_query_vector(query: str) -> np.ndarray:
    tokens = [token.lower() for token in query.replace("\n", " ").split() if token]
    vector = np.zeros(64, dtype="float32")
    for token in tokens:
        index = abs(hash(token)) % 64
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def search(query: str, index, chunks, model, top_k: int = 5):
    try:
        if model is not None:
            query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
        else:
            query_vector = build_local_query_vector(query).reshape(1, -1)
    except Exception:
        query_vector = build_local_query_vector(query).reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)
    return [
        {
            "rank": rank + 1,
            "section": chunks[item]["section"],
            "text": chunks[item]["text"],
            "distance": float(distances[0][rank]),
        }
        for rank, item in enumerate(indices[0])
    ]


def discover_reports():
    supported = [".png", ".jpg", ".jpeg"]
    report_files = []
    for ext in supported:
        report_files.extend(REPORTS_DIR.glob(f"*{ext}"))
    return sorted(report_files, key=lambda path: path.name)


def render_report_gallery(report_files):
    if not report_files:
        st.info("No report images were found in the reports folder yet.")
        return

    descriptions = {
        "confusion_matrix.png": "Confusion matrix showing the model's classification outcomes.",
        "correlation_heatmap.png": "Correlation heatmap for the most important predictive variables.",
        "income_distribution.png": "Distribution of applicant income across the dataset.",
        "missing_values.png": "Missing-value profile to highlight the quality of the training data.",
        "shap_feature_importance.png": "SHAP-based feature importance for the risk model.",
        "shap_summary.png": "SHAP summary plot showing how features influence the model output.",
        "target_distribution.png": "Distribution of the target outcome for the dataset.",
    }

    columns = st.columns(2)
    for index, path in enumerate(report_files):
        with columns[index % 2]:
            st.image(str(path), caption=descriptions.get(path.name, "Model evaluation report."), use_container_width=True)
            st.caption(path.name)


st.title("SBP Consumer Financing Assistant")
st.caption("A professional Streamlit dashboard for searching regulation references and reviewing the credit-risk model reports.")

st.markdown("")

index, chunks, model = load_vectorstore()
report_files = discover_reports()

with st.sidebar:
    st.header("Search the regulations", divider="blue")
    st.markdown("<div style='font-size:0.95rem; color:#245a96;'>Ask a regulation question and retrieve the most relevant passages from the local knowledge base.</div>", unsafe_allow_html=True)
    query = st.text_area(
        "Ask a question about the regulations",
        value="What is the maximum tenure for auto financing?",
        height=140,
    )
    top_k = st.slider("Number of results", min_value=1, max_value=8, value=3)
    run = st.button("Search", use_container_width=True)

    st.markdown("")
    st.caption("Reference sources: local knowledge base and report images in the repository.")

overview, search_tab, reports_tab = st.tabs(["Overview", "Regulation search", "Model reports"])

with overview:
    st.subheader("How this interface works")
    st.write("The app combines a local knowledge base of SBP regulation chunks with a lightweight FAISS search layer. The search results are shown together with the relevant section text so the response is grounded in the project data.")

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Knowledge-base chunks", len(chunks))
    col2.metric("Report images", len(report_files))
    col3.metric("Search index", "FAISS")

    st.markdown("")
    st.subheader("Data reference and explanation")
    st.write(
        "The dataset was sourced from the State Bank of Pakistan's Prudential Regulations for Consumer Financing. Since the SBP website restricts automated web scraping, the document was retrieved manually and cached locally. The complete retrieval process and data provenance are documented in sbp_consumer_financing_source.txt."
    )
    st.write("The search experience is intentionally grounded in repository data rather than external web sources.")

with search_tab:
    if run:
        with st.spinner("Searching the knowledge base..."):
            results = search(query, index, chunks, model, top_k=top_k)

        st.subheader("Search results")
        if not results:
            st.info("No results found for that query.")
        else:
            for result in results:
                with st.expander(f"{result['rank']}. {result['section']}", expanded=(result['rank'] == 1)):
                    st.write(result["text"][:1800])
                    st.caption(f"Similarity distance: {result['distance']:.3f}")
    else:
        st.info("Enter a question in the sidebar and press Search to retrieve relevant regulation text.")

with reports_tab:
    st.subheader("Model and analysis reports")
    st.write("The following figures provide a professional summary of the underlying credit-risk workflow and the data used for training and evaluation.")
    render_report_gallery(report_files)
