# ============================================
# THRESHOLD CHECK + FINALIZE
# Model retrain nahi karna — sirf decision boundary
# (threshold) choose karna hai. Table dekho, apna
# choice CHOSEN_THRESHOLD mein daalo, dobara run karo
# ============================================

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 👇 Table dekhne ke baad yahan final choice daalo, phir dobara run karo
CHOSEN_THRESHOLD = 0.55   # None = sirf table dikhao, kuch save mat karo

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "credit_model.pkl")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Current saved threshold: {bundle['threshold']:.2f}\n")
print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
for t in [0.40, 0.45, 0.50, 0.55, 0.59, 0.65]:
    y_pred = (y_proba >= t).astype(int)
    marker = " ← current" if abs(t - bundle["threshold"]) < 0.01 else ""
    print(f"{t:<10.2f} {accuracy_score(y_test,y_pred):<10.4f} "
          f"{precision_score(y_test,y_pred):<10.4f} {recall_score(y_test,y_pred):<10.4f} "
          f"{f1_score(y_test,y_pred):<10.4f}{marker}")

if CHOSEN_THRESHOLD is not None:
    bundle["threshold"] = CHOSEN_THRESHOLD
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n✅ Threshold updated to {CHOSEN_THRESHOLD} — saved to credit_model.pkl")
else:
    print("\n👉 Table dekho, CHOSEN_THRESHOLD set karo (upar), dobara run karo save karne ke liye.")