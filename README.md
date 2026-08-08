# 🏦 Sentinel - Agentic Credit Underwriting & Compliance System

Sentinel is an AI-powered credit underwriting and regulatory compliance
system that combines **machine learning, explainable AI,
Retrieval-Augmented Generation (RAG), and LLM-based agents** to evaluate
consumer financing applications[cite: 12].

* **GitHub Repository:** [em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System](https://github.com/em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System)
* **Live Streamlit App:** [Access Streamlit UI](https://agentic-credit-underwriting-compliance-system-bympbc8ykiuewtr9.streamlit.app/)

The system performs live applicant risk assessment, explains model
decisions using XGBoost feature contributions, checks relevant SBP
Prudential Regulations for Consumer Financing, and generates an
AI-powered underwriting summary[cite: 12].

------------------------------------------------------------------------

## 🚀 Features

### 📊 1. Live Credit Risk Prediction

The system uses a trained **XGBoost classification model** to predict
the probability that an applicant may default[cite: 12].

The model analyzes:

- Credit limit[cite: 12]
- Demographic information[cite: 12]
- Repayment history[cite: 12]
- Previous payment delays[cite: 12]
- Billing amounts[cite: 12]
- Payment amounts[cite: 12]
- Financial behavior[cite: 12]
- Engineered financial features[cite: 12]

The output includes:

- Default probability[cite: 12]
- Approve or Reject decision[cite: 12]
- Key factors influencing the prediction[cite: 12]

------------------------------------------------------------------------

### 🔍 2. Explainable AI

Instead of providing only a prediction, Sentinel explains why the model
produced that result[cite: 12].

The system uses XGBoost's native:

``` python
pred_contribs=True
```[cite: 12]

to calculate feature contributions for the specific applicant[cite: 12].

The application displays:

- Top factors increasing risk[cite: 12]
- Top factors decreasing risk[cite: 12]
- SHAP-style contribution chart[cite: 12]
- Risk grouped into[cite: 12]:
    - Demographics[cite: 12]
    - Repayment Behavior[cite: 12]
    - Financial Capacity[cite: 12]

------------------------------------------------------------------------

### 📜 3. SBP Regulatory Compliance Checking

The system contains a knowledge base built from the **State Bank of
Pakistan Prudential Regulations for Consumer Financing**[cite: 12].

A RAG pipeline retrieves relevant regulations and uses an LLM to explain
them in plain language[cite: 12].

The system includes deterministic checks for SBP Regulation R-8 exposure
thresholds[cite: 12]:

- Personal clean financing exposure cap[cite: 12]
- Aggregate consumer financing exposure cap[cite: 12]

Important numeric compliance checks are performed directly in code
rather than relying on an LLM[cite: 12].

------------------------------------------------------------------------

### 🤖 4. Regulation Question Answering

Users can ask free-form questions about SBP regulations[cite: 12].

Examples:

``` text
How can I be eligible for consumer financing?
```[cite: 12]

``` text
What is the maximum tenure for auto financing?
```[cite: 12]

``` text
What does Regulation R-8 say?
```[cite: 12]

``` text
Explain Regulation R-3
```[cite: 12]

------------------------------------------------------------------------

## 🔎 Hybrid Regulation Retrieval

Sentinel uses two retrieval strategies[cite: 12].

### Exact Regulation Lookup

If the user explicitly requests a regulation such as[cite: 12]:

``` text
R-3
R3
Regulation R-3
Explain R3
What is O-2?
```[cite: 12]

the application detects the regulation ID using regular expressions and
retrieves the exact regulation directly from the knowledge base[cite: 12].

This avoids relying on semantic similarity for exact regulation
identifiers[cite: 12].

### Semantic Search

For natural-language questions, the application uses[cite: 12]:

``` text
SentenceTransformer
        ↓
all-MiniLM-L6-v2 embeddings
        ↓
FAISS vector search
        ↓
Relevant SBP regulation chunks
        ↓
LLM explanation
```[cite: 12]

### Why Hybrid Retrieval?

Semantic search is useful for questions such as[cite: 12]:

> How can I qualify for a consumer loan?[cite: 12]

However, a query such as[cite: 12]:

> Regulation R-3?[cite: 12]

should not depend only on vector similarity[cite: 12].

The system therefore follows[cite: 12]:

``` text
User Query
    │
    ▼
Detect Regulation ID
    │
    ├── Yes ──► Exact Regulation Lookup
    │
    └── No ───► Semantic FAISS Search
```[cite: 12]

This prevents incorrect results from being returned when the user
explicitly requests a specific regulation[cite: 12].

------------------------------------------------------------------------

# 🧠 Agent Architecture

Sentinel follows a multi-agent workflow[cite: 12].

``` text
Applicant Input
      │
      ▼
┌─────────────────┐
│   Data Agent    │
└─────────────────┘
      │
      │ Validation
      │ SBP Exposure Checks
      │ Feature Engineering
      ▼
┌─────────────────┐
│   Risk Agent    │
└─────────────────┘
      │
      │ XGBoost Prediction
      │ Feature Contributions
      ▼
┌─────────────────┐
│ Compliance Agent│
└─────────────────┘
      │
      │ Exact Regulation Lookup
      │ OR
      │ FAISS Semantic Retrieval
      ▼
┌─────────────────┐
│ Reporting Agent │
└─────────────────┘
      │
      ▼
Final Underwriting Memo
```[cite: 12]

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   Applicant Form      Regulation Chat      Training Reports
          │                  │
          ▼                  ▼
     Data Agent       Regulation Retriever
          │                  │
          ▼                  │
 Feature Engineering          │
          │                  │
          ▼                  ▼
      Risk Agent ───► Exact Lookup / FAISS
          │                  │
          ▼                  ▼
 XGBoost Prediction       SBP Context
          │                  │
          └────────┬─────────┘
                   ▼
            Compliance Agent
                   │
                   ▼
             Reporting Agent
                   │
                   ▼
         Underwriting Decision
```[cite: 12]

------------------------------------------------------------------------

# 🧩 Core Components

## 1. Data Agent

The Data Agent[cite: 12]:

- Validates applicant data[cite: 12]
- Checks invalid credit limits[cite: 12]
- Validates applicant age[cite: 12]
- Validates education and marital status codes[cite: 12]
- Performs deterministic SBP R-8 exposure checks[cite: 12]
- Computes engineered features[cite: 12]

Example thresholds[cite: 12]:

``` python
SBP_R8_PERSONAL_CLEAN_CAP = 2_000_000
SBP_R8_AGGREGATE_CAP = 5_000_000
```[cite: 12]

------------------------------------------------------------------------

## 2. Risk Agent

The Risk Agent uses a trained XGBoost model[cite: 12].

``` text
Applicant Data
      ↓
Feature Engineering
      ↓
XGBoost Model
      ↓
Default Probability
      ↓
Approval / Rejection Decision
```[cite: 12]

The decision is calculated using a saved model threshold[cite: 12]:

``` python
decision = "REJECT" if proba >= threshold else "APPROVE"
```[cite: 12]

------------------------------------------------------------------------

## 3. Explainability Layer

The project uses XGBoost's native contribution calculation[cite: 12]:

``` python
booster.predict(
    dmatrix,
    pred_contribs=True
)
```[cite: 12]

This provides per-applicant feature contributions without requiring the
external `shap` package during prediction[cite: 12].

The system visualizes[cite: 12]:

- Individual feature impact[cite: 12]
- Positive and negative risk contributions[cite: 12]
- Feature-category risk impact[cite: 12]

------------------------------------------------------------------------

## 4. Compliance Agent

The Compliance Agent determines which SBP regulation is relevant to an
applicant[cite: 12].

For example[cite: 12]:

``` text
Applicant exceeds aggregate exposure cap
        ↓
Retrieve relevant SBP Regulation
        ↓
Explain compliance implications
```[cite: 12]

The Compliance Agent uses[cite: 12]:

``` text
Exact Regulation Detection
        OR
Semantic Retrieval
        ↓
FAISS
        ↓
Relevant Regulation Text
        ↓
Groq LLM
        ↓
Plain Language Explanation
```[cite: 12]

------------------------------------------------------------------------

# 📚 RAG Pipeline

The regulatory knowledge base is processed using the following pipeline[cite: 12]:

``` text
SBP Prudential Regulations PDF
            ↓
Document Extraction
            ↓
Regulation Detection
            ↓
Chunk Generation
            ↓
Metadata Storage
            ↓
SentenceTransformer Embeddings
            ↓
FAISS Vector Index
            ↓
User Query
            ↓
Exact Lookup OR Semantic Search
            ↓
Relevant Regulation
            ↓
LLM Answer
```[cite: 12]

------------------------------------------------------------------------

## 🗂️ Knowledge Base Structure

``` text
data/
└── knowledge_base/
    ├── faiss_index.bin
    └── chunk_metadata.json
```[cite: 12]

Each regulation chunk contains metadata similar to[cite: 12]:

``` json
{
    "id": "regulation_r_3",
    "section": "REGULATION R-3",
    "text": "Regulation content..."
}
```[cite: 12]

This metadata enables deterministic regulation lookup[cite: 12].

------------------------------------------------------------------------

# 🖥️ Application Interface

The Streamlit application contains three main sections[cite: 12].

## 1. Live Underwriting

Users enter[cite: 12]:

- Credit limit[cite: 12]
- Age[cite: 12]
- Gender[cite: 12]
- Education[cite: 12]
- Marital status[cite: 12]
- Six-month repayment history[cite: 12]
- Six-month billing history[cite: 12]
- Six-month payment history[cite: 12]

The system generates[cite: 12]:

- Default probability[cite: 12]
- Approval or rejection decision[cite: 12]
- SHAP-style explanation[cite: 12]
- Engineered features[cite: 12]
- Payment history charts[cite: 12]
- Repayment behavior charts[cite: 12]
- Compliance findings[cite: 12]
- AI-generated underwriting memo[cite: 12]

------------------------------------------------------------------------

## 2. Ask About Regulations

Users can ask questions directly about SBP Prudential Regulations[cite: 12].

Examples[cite: 12]:

``` text
How do I qualify for consumer financing?
```[cite: 12]

``` text
What is the maximum financing period?
```[cite: 12]

``` text
Explain Regulation R-3
```[cite: 12]

``` text
What does R-8 say?
```[cite: 12]

------------------------------------------------------------------------

## 3. Model Training Reports

The application can display pre-generated model reports including[cite: 12]:

- Confusion matrix[cite: 12]
- Correlation heatmap[cite: 12]
- SHAP feature importance[cite: 12]
- SHAP summary[cite: 12]
- Target distribution[cite: 12]
- Missing value analysis[cite: 12]

------------------------------------------------------------------------

# 📁 Project Structure

``` text
Agentic-Credit-Underwriting-Compliance-System/
│
├── data/
│   ├── raw/
│   │   └── credit_card_default.csv
│   │
│   └── knowledge_base/
│       ├── faiss_index.bin
│       └── chunk_metadata.json
│
├── models/
│   └── credit_model.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── shap_feature_importance.png
│   ├── shap_summary.png
│   ├── target_distribution.png
│   └── missing_values.png
│
├── src/
│   ├── agents/
│   │   └── orchestrator.py
│   │
│   ├── models/
│   │   └── train.py
│   │
│   ├── rag/
│   │   ├── ingest_documents.py
│   │   ├── build_vectorstore.py
│   │   ├── query_rag.py
│   │   └── generate_answer.py
│   │
│   └── spark/
│       ├── data_ingestion.py
│       ├── preprocessing.py
│       └── feature_engineering.py
│
├── streamlit_app.py
├── requirements.txt
├── dvc.yaml
└── README.md
```[cite: 12]

------------------------------------------------------------------------

# ⚙️ Installation

Clone the repository[cite: 12]:

``` bash
git clone [https://github.com/em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System.git](https://github.com/em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System.git)
```[cite: 12]

Move into the project directory[cite: 12]:

``` bash
cd Agentic-Credit-Underwriting-Compliance-System
```[cite: 12]

Create a virtual environment[cite: 12]:

``` bash
python -m venv venv
```[cite: 12]

Activate it[cite: 12].

### Windows

``` bash
venv\Scripts\activate
```[cite: 12]

### Linux / macOS

``` bash
source venv/bin/activate
```[cite: 12]

Install dependencies[cite: 12]:

``` bash
pip install -r requirements.txt
```[cite: 12]

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a `.env` file[cite: 12]:

``` env
GROQ_API_KEY=your_groq_api_key
```[cite: 12]

For Streamlit Cloud, add[cite: 12]:

``` toml
GROQ_API_KEY = "your_groq_api_key"
```[cite: 12]

The LLM is used for[cite: 12]:

- Regulation explanations[cite: 12]
- Applicant quick-fill from natural language[cite: 12]
- Compliance summaries[cite: 12]
- Final underwriting memo generation[cite: 12]

------------------------------------------------------------------------

# ▶️ Running the Application

Run[cite: 12]:

``` bash
streamlit run streamlit_app.py
```[cite: 12]

Then open the local URL displayed by Streamlit[cite: 12]. You can also test the live cloud deployment directly at the [Streamlit UI App](https://agentic-credit-underwriting-compliance-system-bympbc8ykiuewtr9.streamlit.app/).

------------------------------------------------------------------------

# 🧪 Example Regulation Queries

Try[cite: 12]:

``` text
What are the requirements for consumer financing?
```[cite: 12]

``` text
How can I be eligible for a loan?
```[cite: 12]

``` text
Regulation R-3?
```[cite: 12]

``` text
Explain R3
```[cite: 12]

``` text
What does Regulation R-8 say?
```[cite: 12]

``` text
What is the maximum tenure for auto financing?
```[cite: 12]

------------------------------------------------------------------------

# 🛠️ Technologies Used

## Machine Learning

- Python[cite: 12]
- XGBoost[cite: 12]
- Scikit-learn[cite: 12]
- Pandas[cite: 12]

## Explainable AI

- XGBoost `pred_contribs`[cite: 12]

## RAG

- Sentence Transformers[cite: 12]
- `all-MiniLM-L6-v2`[cite: 12]
- FAISS[cite: 12]

## Generative AI

- Hugging Face InferenceClient[cite: 12]
- Groq[cite: 12]
- Llama 3.3[cite: 12]

## Application

- Streamlit[cite: 12]

## MLOps

- DVC[cite: 12]
- MLflow[cite: 12]

## Big Data Processing

- PySpark[cite: 12]

------------------------------------------------------------------------

# 🔐 Safety and Compliance Design

The system separates deterministic compliance checks from LLM-generated
explanations[cite: 12].

For important numeric limits, the application performs direct code-based
checks[cite: 12].

The LLM is used to[cite: 12]:

- Explain retrieved regulations[cite: 12]
- Generate summaries[cite: 12]
- Extract structured applicant data from natural language[cite: 12]

Deterministic code is responsible for[cite: 12]:

- Numeric threshold checks[cite: 12]
- Risk prediction[cite: 12]
- Applicant feature calculation[cite: 12]
- Exact regulation identifier lookup[cite: 12]

------------------------------------------------------------------------

# ⚠️ Disclaimer

This project is an educational and technical demonstration of[cite: 12]:

- Credit risk modeling[cite: 12]
- Explainable AI[cite: 12]
- RAG systems[cite: 12]
- Agentic AI workflows[cite: 12]
- Regulatory document retrieval[cite: 12]

It is **not intended to replace professional credit underwriting, legal
advice, regulatory interpretation, or official decisions made by banks
or the State Bank of Pakistan**[cite: 12].

Before using such a system in a real financial environment, additional
requirements would be necessary, including[cite: 12]:

- Regulatory approval[cite: 12]
- Data governance[cite: 12]
- Security controls[cite: 12]
- Model monitoring[cite: 12]
- Bias testing[cite: 12]
- Audit logging[cite: 12]
- Human review[cite: 12]
- Production-grade validation[cite: 12]

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible future improvements include[cite: 12]:

- Hybrid retrieval using BM25 + FAISS[cite: 12]
- Regulation-specific metadata filtering[cite: 12]
- Cross-encoder reranking[cite: 12]
- Citation-level source highlighting[cite: 12]
- Conversation memory for regulation chat[cite: 12]
- Model drift monitoring[cite: 12]
- Automated regulatory document updates[cite: 12]
- Human-in-the-loop approval[cite: 12]
- Docker deployment[cite: 12]
- Role-based access control[cite: 12]
- Applicant audit trail[cite: 12]
- Multi-document regulatory knowledge bases[cite: 12]

------------------------------------------------------------------------

# 👨‍💻 Author

**Muhammad Sharjeel**[cite: 12]

AI / Machine Learning \| MLOps \| RAG \| Agentic AI[cite: 12]

------------------------------------------------------------------------

## ⭐ If You Found This Project Useful

Consider giving the repository a star[cite: 12].