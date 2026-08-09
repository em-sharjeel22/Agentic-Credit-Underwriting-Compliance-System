import pandas as pd
import xgboost as xgb
from utils.resources import load_risk_resources
from state import UnderwritingState
def risk_agent(state: UnderwritingState) -> dict:
    print("🔍 [Risk Agent] Analyzing applicant...")
    model, threshold, feature_names = load_risk_resources()
    X = pd.DataFrame([state["applicant"]])[feature_names]
    proba = model.predict_proba(X)[0, 1]
    decision = "REJECT" if proba >= threshold else "APPROVE"
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    raw = booster.predict(dmatrix, pred_contribs=True)
    shap_vals = raw[0, :-1]
    contributions = sorted(
        zip(feature_names, shap_vals, X.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]
    print(f"   → Probability: {proba * 100:.1f}% | Decision: {decision}")
    return {
        "risk_probability": float(proba),
        "risk_decision": decision,
        "top_risk_factors": [
            {"feature": f, "impact": float(v), "value": float(val)}
            for f, v, val in contributions
        ],
    }