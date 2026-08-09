# 🏦 Sentinel - Agentic Credit Underwriting & Compliance System

Sentinel is an AI-powered credit underwriting and regulatory compliance
system that combines **machine learning, explainable AI,
Retrieval-Augmented Generation (RAG), and LLM-based agents** to evaluate
consumer financing applications.

* **GitHub Repository:** [em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System](https://github.com/em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System)
* **Live Streamlit App:** [Access Streamlit UI](https://agentic-credit-underwriting-compliance-system-bympbc8ykiuewtr9.streamlit.app/)

The system performs live applicant risk assessment, explains model
decisions using XGBoost feature contributions, checks relevant SBP
Prudential Regulations for Consumer Financing, and generates an
AI-powered underwriting summary.

------------------------------------------------------------------------

## 🚀 Features

### 📊 1. Live Credit Risk Prediction

The system uses a trained **XGBoost classification model** to predict
the probability that an applicant may default.

The model analyzes:

-   Credit limit
-   Demographic information
-   Repayment history
-   Previous payment delays
-   Billing amounts
-   Payment amounts
-   Financial behavior
-   Engineered financial features

The output includes:

-   Default probability
-   Approve or Reject decision
-   Key factors influencing the prediction

------------------------------------------------------------------------

### 🔍 2. Explainable AI

Instead of providing only a prediction, Sentinel explains why the model
produced that result.

The system uses XGBoost's native:

``` python
pred_contribs=True
```

to calculate feature contributions for the specific applicant.

The application displays:

-   Top factors increasing risk
-   Top factors decreasing risk
-   SHAP-style contribution chart
-   Risk grouped into:
    -   Demographics
    -   Repayment Behavior
    -   Financial Capacity

------------------------------------------------------------------------

### 📜 3. SBP Regulatory Compliance Checking

The system contains a knowledge base built from the **State Bank of
Pakistan Prudential Regulations for Consumer Financing**.

A RAG pipeline retrieves relevant regulations and uses an LLM to explain
them in plain language.

The system includes deterministic checks for SBP Regulation R-8 exposure
thresholds:

-   Personal clean financing exposure cap
-   Aggregate consumer financing exposure cap

Important numeric compliance checks are performed directly in code
rather than relying on an LLM.

------------------------------------------------------------------------

### 🤖 4. Regulation Question Answering

Users can ask free-form questions about SBP regulations.

Examples:

``` text
How can I be eligible for consumer financing?
```

``` text
What is the maximum tenure for auto financing?
```

``` text
What does Regulation R-8 say?
```

``` text
Explain Regulation R-3
```

------------------------------------------------------------------------

## 🔎 Hybrid Regulation Retrieval

Sentinel uses two retrieval strategies.

### Exact Regulation Lookup

If the user explicitly requests a regulation such as:

``` text
R-3
R3
Regulation R-3
Explain R3
What is O-2?
```

the application detects the regulation ID using regular expressions and
retrieves the exact regulation directly from the knowledge base.

This avoids relying on semantic similarity for exact regulation
identifiers.

### Semantic Search

For natural-language questions, the application uses:

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
```

### Why Hybrid Retrieval?

Semantic search is useful for questions such as:

> How can I qualify for a consumer loan?

However, a query such as:

> Regulation R-3?

should not depend only on vector similarity.

The system therefore follows:

``` text
User Query
    │
    ▼
Detect Regulation ID
    │
    ├── Yes ──► Exact Regulation Lookup
    │
    └── No ───► Semantic FAISS Search
```

This prevents incorrect results from being returned when the user
explicitly requests a specific regulation.

------------------------------------------------------------------------

# 🧠 Agent Architecture

Sentinel follows a multi-agent workflow.

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
```

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
```

------------------------------------------------------------------------

# 🧩 Core Components

## 1. Data Agent

The Data Agent:

-   Validates applicant data
-   Checks invalid credit limits
-   Validates applicant age
-   Validates education and marital status codes
-   Performs deterministic SBP R-8 exposure checks
-   Computes engineered features

Example thresholds:

``` python
SBP_R8_PERSONAL_CLEAN_CAP = 2_000_000
SBP_R8_AGGREGATE_CAP = 5_000_000
```

------------------------------------------------------------------------

## 2. Risk Agent

The Risk Agent uses a trained XGBoost model.

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
```

The decision is calculated using a saved model threshold:

``` python
decision = "REJECT" if proba >= threshold else "APPROVE"
```

------------------------------------------------------------------------

## 3. Explainability Layer

The project uses XGBoost's native contribution calculation:

``` python
booster.predict(
    dmatrix,
    pred_contribs=True
)
```

This provides per-applicant feature contributions without requiring the
external `shap` package during prediction.

The system visualizes:

-   Individual feature impact
-   Positive and negative risk contributions
-   Feature-category risk impact

------------------------------------------------------------------------

## 4. Compliance Agent

The Compliance Agent determines which SBP regulation is relevant to an
applicant.

For example:

``` text
Applicant exceeds aggregate exposure cap
        ↓
Retrieve relevant SBP Regulation
        ↓
Explain compliance implications
```

The Compliance Agent uses:

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
```

------------------------------------------------------------------------

# 📚 RAG Pipeline

The regulatory knowledge base is processed using the following pipeline:

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
```

------------------------------------------------------------------------

## 🗂️ Knowledge Base Structure

``` text
data/
└── knowledge_base/
    ├── faiss_index.bin
    └── chunk_metadata.json
```

Each regulation chunk contains metadata similar to:

``` json
{
    "id": "regulation_r_3",
    "section": "REGULATION R-3",
    "text": "Regulation content..."
}
```

This metadata enables deterministic regulation lookup.

------------------------------------------------------------------------

# 🖥️ Application Interface

The Streamlit application contains three main sections.

## 1. Live Underwriting

Users enter:

-   Credit limit
-   Age
-   Gender
-   Education
-   Marital status
-   Six-month repayment history
-   Six-month billing history
-   Six-month payment history

The system generates:

-   Default probability
-   Approval or rejection decision
-   SHAP-style explanation
-   Engineered features
-   Payment history charts
-   Repayment behavior charts
-   Compliance findings
-   AI-generated underwriting memo

------------------------------------------------------------------------

## 2. Ask About Regulations

Users can ask questions directly about SBP Prudential Regulations.

Examples:

``` text
How do I qualify for consumer financing?
```

``` text
What is the maximum financing period?
```

``` text
Explain Regulation R-3
```

``` text
What does R-8 say?
```

------------------------------------------------------------------------

## 3. Model Training Reports

The application can display pre-generated model reports including:

-   Confusion matrix
-   Correlation heatmap
-   SHAP feature importance
-   SHAP summary
-   Target distribution
-   Missing value analysis

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
```

------------------------------------------------------------------------

# ⚙️ Installation

Clone the repository:

``` bash
git clone https://github.com/em-sharjeel22/Agentic-Credit-Underwriting-Compliance-System.git
```

Move into the project directory:

``` bash
cd Agentic-Credit-Underwriting-Compliance-System
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate it.

### Windows

``` bash
venv\Scripts\activate
```

### Linux / macOS

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔑 Environment Variables

Create a `.env` file:

``` env
GROQ_API_KEY=your_groq_api_key
```

For Streamlit Cloud, add:

``` toml
GROQ_API_KEY = "your_groq_api_key"
```

The LLM is used for:

-   Regulation explanations
-   Applicant quick-fill from natural language
-   Compliance summaries
-   Final underwriting memo generation

------------------------------------------------------------------------

# ▶️ Running the Application

Run:

``` bash
streamlit run streamlit_app.py
```

Then open the local URL displayed by Streamlit.

------------------------------------------------------------------------

# 🧪 Example Regulation Queries

Try:

``` text
What are the requirements for consumer financing?
```

``` text
How can I be eligible for a loan?
```

``` text
Regulation R-3?
```

``` text
Explain R3
```

``` text
What does Regulation R-8 say?
```

``` text
What is the maximum tenure for auto financing?
```

------------------------------------------------------------------------

# 🛠️ Technologies Used

## Machine Learning

-   Python
-   XGBoost
-   Scikit-learn
-   Pandas

## Explainable AI

-   XGBoost `pred_contribs`

## RAG

-   Sentence Transformers
-   `all-MiniLM-L6-v2`
-   FAISS

## Generative AI

-   Hugging Face InferenceClient
-   Groq
-   Llama 3.3

## Application

-   Streamlit

## MLOps

-   DVC
-   MLflow

## Big Data Processing

-   PySpark

------------------------------------------------------------------------

# 🔐 Safety and Compliance Design

The system separates deterministic compliance checks from LLM-generated
explanations.

For important numeric limits, the application performs direct code-based
checks.

The LLM is used to:

-   Explain retrieved regulations
-   Generate summaries
-   Extract structured applicant data from natural language

Deterministic code is responsible for:

-   Numeric threshold checks
-   Risk prediction
-   Applicant feature calculation
-   Exact regulation identifier lookup

------------------------------------------------------------------------

# ⚠️ Disclaimer

This project is an educational and technical demonstration of:

-   Credit risk modeling
-   Explainable AI
-   RAG systems
-   Agentic AI workflows
-   Regulatory document retrieval

It is **not intended to replace professional credit underwriting, legal
advice, regulatory interpretation, or official decisions made by banks
or the State Bank of Pakistan**.

Before using such a system in a real financial environment, additional
requirements would be necessary, including:

-   Regulatory approval
-   Data governance
-   Security controls
-   Model monitoring
-   Bias testing
-   Audit logging
-   Human review
-   Production-grade validation

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible future improvements include:

-   Hybrid retrieval using BM25 + FAISS
-   Regulation-specific metadata filtering
-   Cross-encoder reranking
-   Citation-level source highlighting
-   Conversation memory for regulation chat
-   Model drift monitoring
-   Automated regulatory document updates
-   Human-in-the-loop approval
-   Docker deployment
-   Role-based access control
-   Applicant audit trail
-   Multi-document regulatory knowledge bases

------------------------------------------------------------------------

# 👨‍💻 Author

**Muhammad Sharjeel**

AI / Machine Learning \| MLOps \| RAG \| Agentic AI

------------------------------------------------------------------------

## ⭐ If You Found This Project Useful

Consider giving the repository a star.
