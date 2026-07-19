# ============================================
# XGBOOST TRAINING + MLFLOW TRACKING (v3)
# Dataset: Credit Card Default (30,000 rows)
#
# v2 se farak: wider hyperparameter search + zyada trials —
# 24,000 training rows ke sath deeper/complex models bhi
# overfit nahi karte (jaisा 838 rows ke sath hota tha)
# ============================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}")
mlflow.set_experiment("credit_underwriting")


def load_processed_data():
    print("\n📂 Loading processed data...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(f"✅ Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


def split_features_target(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def compute_scale_pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    weight = neg / pos
    print(f"⚖️  Class balance → Repaid: {neg} | Defaulted: {pos} | scale_pos_weight: {weight:.2f}")
    return weight


def tune_hyperparameters(X_train, y_train, scale_pos_weight, n_trials=75):
    """
    30x zyada data hai ab — isliye search space wider kiya:
    deeper trees (max_depth 10 tak) aur zyada estimators (800 tak)
    ab overfit karne ka risk kam hai
    """
    print("\n" + "="*50)
    print(f"🔍 OPTUNA: {n_trials} COMBINATIONS TEST HO RAHI HAIN (5-15 min lagenge — 24,000 rows hain ab)")
    print("="*50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric":      "logloss",
            "random_state":     42,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42,
    })

    print(f"\n✅ Best CV ROC-AUC: {study.best_value:.4f}")
    return best_params


def find_best_threshold(model_params, X_train, y_train, cv_splits=5):
    print("\n🎯 Best decision threshold dhoond rahe hain...")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(**model_params)
    oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (oof_proba >= t).astype(int)
        f1 = f1_score(y_train, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"✅ Best threshold: {best_thresh:.2f} (training OOF F1: {best_f1:.4f})")
    return best_thresh


def evaluate_model(model, X_test, y_test, threshold):
    print("\n" + "="*50)
    print("📊 FINAL TEST SET EVALUATION")
    print("="*50)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    naive_baseline = max(y_test.value_counts(normalize=True))
    print(f"\n📏 Naive Baseline (hamesha majority class bolna): {naive_baseline:.4f}")
    print("📈 Hamara Model:")
    for name, value in metrics.items():
        print(f"   {name.upper():12s}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Repaid", "Defaulted"],
                yticklabels=["Repaid", "Defaulted"])
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("\n📊 Chart saved → reports/confusion_matrix.png")

    return metrics, naive_baseline


def run_training():
    train_df, test_df = load_processed_data()
    X_train, y_train = split_features_target(train_df)
    X_test, y_test   = split_features_target(test_df)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    with mlflow.start_run(run_name="xgboost_tuned_v3_wider_search"):
        best_params = tune_hyperparameters(X_train, y_train, scale_pos_weight, n_trials=75)
        mlflow.log_params(best_params)

        best_threshold = find_best_threshold(best_params, X_train, y_train)
        mlflow.log_param("decision_threshold", best_threshold)

        print("\n" + "="*50)
        print("🚀 FINAL MODEL TRAINING")
        print("="*50)
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        print("✅ Model trained!")

        metrics, naive_baseline = evaluate_model(model, X_test, y_test, best_threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("naive_baseline_accuracy", naive_baseline)

        try:
            if not hasattr(model, "_estimator_type"):
                model._estimator_type = "classifier"
            mlflow.xgboost.log_model(model, artifact_path="model")
            print("✅ Model artifact logged to MLflow")
        except Exception as e:
            print(f"⚠️  MLflow artifact logging skipped ({type(e).__name__}) — not critical")

        model_path = os.path.join(MODELS_DIR, "credit_model.pkl")
        joblib.dump({"model": model, "threshold": best_threshold}, model_path)
        print(f"\n💾 Model saved → {model_path}")

        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETE!")
        print("="*50)


if __name__ == "__main__":
    run_training()


