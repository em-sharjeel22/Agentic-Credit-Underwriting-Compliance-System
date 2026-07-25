# ============================================
# FASTAPI BACKEND
# Graph aur models SIRF EK BAAR, server startup pe
# load hote hain — har request unhi ko reuse karti hai
# ============================================

import os
import sys
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "src" / "agents"))

DB_PATH = BASE_DIR / "data" / "decisions.db"

_graph = None   # startup pe set hota hai, phir sab requests isi ko use karti hain


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            applicant_data TEXT NOT NULL,
            risk_probability REAL,
            risk_decision TEXT,
            top_risk_factors TEXT,
            compliance_question TEXT,
            compliance_answer TEXT,
            compliance_sources TEXT
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    from orchestrator import build_graph   # yahi pe models/FAISS/LLM client load hote hain
    init_db()
    print("🚀 Building agent graph (one-time, startup pe)...")
    _graph = build_graph()
    print("✅ API ready — models loaded, koi per-request reload nahi hoga")
    yield


app = FastAPI(
    title="Sentinel — Credit Underwriting API",
    description="Agentic credit risk scoring + SBP compliance checking",
    version="1.0.0",
    lifespan=lifespan,
)


class ApplicantRequest(BaseModel):
    LIMIT_BAL: float = Field(..., description="Credit limit")
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class RiskFactor(BaseModel):
    feature: str
    impact: float
    value: float


class UnderwritingResponse(BaseModel):
    data_warnings: list[str]
    risk_probability: float
    risk_decision: str
    top_risk_factors: list[RiskFactor]
    compliance_answer: str
    compliance_sources: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Sentinel Underwriting API"}


@app.post("/underwrite", response_model=UnderwritingResponse)
def underwrite(applicant: ApplicantRequest):
    from features import engineer_features
    applicant_dict = engineer_features(applicant.model_dump())

    try:
        result = _graph.invoke({"applicant": applicant_dict})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    _log_decision(applicant_dict, result)

    return UnderwritingResponse(
        data_warnings=result["data_warnings"],
        risk_probability=result["risk_probability"],
        risk_decision=result["risk_decision"],
        top_risk_factors=result["top_risk_factors"],
        compliance_answer=result["compliance_answer"],
        compliance_sources=result["compliance_sources"],
    )


@app.get("/decisions")
def get_decision_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, timestamp, risk_decision, risk_probability FROM decisions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _log_decision(applicant, result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO decisions
           (timestamp, applicant_data, risk_probability, risk_decision,
            top_risk_factors, compliance_question, compliance_answer, compliance_sources)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(applicant),
            result["risk_probability"],
            result["risk_decision"],
            json.dumps(result["top_risk_factors"]),
            result["compliance_question"],
            result["compliance_answer"],
            json.dumps(result["compliance_sources"]),
        ),
    )
    conn.commit()
    conn.close()