# ============================================
# SHAP EXPLAINABILITY (v2 — native XGBoost method)
#
# shap.TreeExplainer ka binary parser XGBoost ke naye
# versions ke saath baar-baar toot raha hai (known,
# long-standing issue). Isliye hum XGBoost ke apne
# built-in SHAP computation (pred_contribs) use kar
# rahe hain — same result, zero fragile parsing.
# ============================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_model_and_data():
    print("\n📂 Loading model + test data...")
    bundle = joblib.load(os.path.join(MODELS_DIR, "credit_model.pkl"))
    model, threshold = bundle["model"], bundle["threshold"]

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    print(f"✅ Model loaded | Threshold: {threshold:.2f} | Test rows: {len(X_test)}")
    return model, threshold, X_test, y_test


def compute_shap_contributions(model, X):
    """
    XGBoost khud SHAP values nikal sakta hai — bina shap
    library ke fragile binary parser ka use kiye.

    booster.predict(..., pred_contribs=True) har feature ka
    exact contribution deta hai, plus ek extra 'bias' column
    (yeh average/base risk level hai) — yehi cheez shap.TreeExplainer
    bhi deta, bas yeh rasta reliable hai
    """
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    raw = booster.predict(dmatrix, pred_contribs=True)

    shap_values = raw[:, :-1]        # sab features ka contribution
    base_value = float(raw[0, -1])   # average applicant ka base risk
    return shap_values, base_value


def generate_global_plots(shap_values, X_test):
    """
    GLOBAL explanation = "overall kaunse features sabse
    zyada matter karte hain saare applicants ke liye"
    """
    print("\n" + "="*50)
    print("🌍 GLOBAL FEATURE IMPORTANCE")
    print("="*50)

    try:
        plt.close('all')
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close('all')
        print("📊 Chart saved → reports/shap_summary.png")
    except Exception as e:
        print(f"⚠️  Summary plot skipped: {e}")

    try:
        plt.close('all')
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(REPORTS_DIR, "shap_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close('all')
        print("📊 Chart saved → reports/shap_feature_importance.png")
    except Exception as e:
        print(f"⚠️  Bar plot skipped: {e}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(X_test.columns, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print("\n📋 Top 10 Most Important Features (overall):")
    for i, (feat, val) in enumerate(ranking[:10], 1):
        nice_name = feat.replace("_encoded", "").replace("_", " ").title()
        print(f"   {i:2d}. {nice_name:25s} avg impact: {val:.4f}")


def explain_single_applicant(model, X_row, threshold):
    """
    INDIVIDUAL explanation = "IS applicant ka score is
    tarah bana — yeh raha breakdown"
    """
    proba = model.predict_proba(X_row)[0, 1]
    decision = "❌ REJECT (High Risk)" if proba >= threshold else "✅ APPROVE (Low Risk)"

    shap_vals, _ = compute_shap_contributions(model, X_row)
    shap_vals = shap_vals[0]

    contributions = list(zip(X_row.columns, shap_vals, X_row.iloc[0].values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "probability": proba,
        "decision": decision,
        "top_factors": contributions[:6],
    }


def print_explanation(result, label="Applicant"):
    print(f"\n{'─'*50}")
    print(f"📄 {label}")
    print(f"{'─'*50}")
    print(f"   Default Probability : {result['probability']*100:.1f}%")
    print(f"   Decision            : {result['decision']}")
    print(f"\n   Top Factors (sabse zyada asar dalne wale):")
    for feat, shap_val, raw_val in result["top_factors"]:
        arrow = "🔴 Risk ↑" if shap_val > 0 else "🟢 Risk ↓"
        nice_name = feat.replace("_encoded", "").replace("_", " ").title()
        print(f"      {nice_name:22s} {arrow:10s} (impact: {shap_val:+.3f}, value: {raw_val})")


def run_explainability():
    model, threshold, X_test, y_test = load_model_and_data()

    print("\n🧠 Computing SHAP contributions (native XGBoost method)...")
    shap_values, base_value = compute_shap_contributions(model, X_test)
    print(f"✅ Done! Base risk level (average applicant): {base_value:+.3f}")

    generate_global_plots(shap_values, X_test)

    print("\n" + "="*50)
    print("👤 INDIVIDUAL APPLICANT EXAMPLES")
    print("="*50)

    all_proba = model.predict_proba(X_test)[:, 1]
    riskiest_idx = int(np.argmax(all_proba))
    safest_idx   = int(np.argmin(all_proba))

    result_risky = explain_single_applicant(model, X_test.iloc[[riskiest_idx]], threshold)
    result_safe  = explain_single_applicant(model, X_test.iloc[[safest_idx]], threshold)

    print_explanation(result_risky, label="Sabse Risky Applicant (Test Set)")
    print_explanation(result_safe,  label="Sabse Safe Applicant (Test Set)")

    print("\n" + "="*50)
    print("🎉 SHAP EXPLAINABILITY COMPLETE!")
    print("="*50)
    print("👉 reports/shap_summary.png aur shap_feature_importance.png dekho")


if __name__ == "__main__":
    run_explainability()