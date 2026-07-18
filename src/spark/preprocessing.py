# ============================================
# PREPROCESSING + FEATURE ENGINEERING
# Dataset: Default of Credit Card Clients (30,000 rows)
# ============================================

import os
from functools import reduce
from operator import add
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split as sk_train_test_split
from spark_session import get_spark_session

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credit_card_default.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data(spark):
    print("\n📂 Loading raw data...")
    df = spark.read.csv(RAW_DATA_PATH, header=True, inferSchema=True)
    print(f"✅ Loaded! Rows: {df.count()} | Columns: {len(df.columns)}")
    return df


def clean_categorical_columns(df):
    """
    EDUCATION aur MARRIAGE mein kuch 'undocumented' categories hain
    (jaise EDUCATION=0,5,6 jo official codebook mein defined nahi) —
    inko ek consistent 'Other' bucket mein daal rahe hain
    """
    print("\n" + "="*50)
    print("🧹 CLEANING CATEGORICAL COLUMNS")
    print("="*50)

    df = df.withColumn(
        "EDUCATION",
        F.when(F.col("EDUCATION").isin([0, 5, 6]), 4).otherwise(F.col("EDUCATION"))
    )
    print("   ✅ EDUCATION: unknown categories (0,5,6) → 4 (Other)")

    df = df.withColumn(
        "MARRIAGE",
        F.when(F.col("MARRIAGE") == 0, 3).otherwise(F.col("MARRIAGE"))
    )
    print("   ✅ MARRIAGE: unknown category (0) → 3 (Other)")

    return df


def sum_cols(cols):
    """Multiple columns ko safely add karne ka helper"""
    return reduce(add, [F.col(c) for c in cols])


def create_new_features(df):
    """
    Yeh dataset German Credit se zyada 'rich' hai — 6 mahine ki
    payment history hai, isliye genuinely useful risk signals
    banane ka mauka milta hai
    """
    print("\n" + "="*50)
    print("🛠️  CREATING NEW FEATURES")
    print("="*50)

    bill_cols        = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols     = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols  = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

    # ── Utilization Ratio ─────────────────────
    # Kitna credit limit use kar raha hai — top risk signals mein se ek
    df = df.withColumn("avg_bill_amt", sum_cols(bill_cols) / 6)
    df = df.withColumn(
        "utilization_ratio",
        F.round(F.col("avg_bill_amt") / F.col("LIMIT_BAL"), 4)
    )
    print("   ✅ utilization_ratio = avg(BILL_AMT 6 months) / LIMIT_BAL")

    # ── Payment Delay Behavior ────────────────
    df = df.withColumn("avg_pay_delay", F.round(sum_cols(pay_status_cols) / 6, 3))
    print("   ✅ avg_pay_delay = average payment status (6 months)")

    df = df.withColumn(
        "max_pay_delay",
        F.greatest(*[F.col(c) for c in pay_status_cols])
    )
    print("   ✅ max_pay_delay = worst delay in 6 months")

    late_flags = [F.when(F.col(c) > 0, 1).otherwise(0) for c in pay_status_cols]
    df = df.withColumn("months_late", reduce(add, late_flags))
    print("   ✅ months_late = kitne mahine late payment hua (0-6)")

    # ── Payment vs Bill Ratio ─────────────────
    df = df.withColumn("total_paid_6m", sum_cols(pay_amt_cols))
    df = df.withColumn("total_billed_6m", sum_cols(bill_cols))
    df = df.withColumn(
        "payment_ratio",
        F.when(F.col("total_billed_6m") > 0,
               F.round(F.col("total_paid_6m") / F.col("total_billed_6m"), 4)
        ).otherwise(1.0)
    )
    print("   ✅ payment_ratio = total paid / total billed (6 months)")

    # ── Trend: Recent vs Purana Behavior ──────
    df = df.withColumn("delay_trend", F.col("PAY_0") - F.col("PAY_6"))
    print("   ✅ delay_trend = recent delay - purana delay (+ve = bigadta ja raha hai)")

    print("\n✅ 6 new features created!")
    return df


def select_final_features(df):
    print("\n" + "="*50)
    print("📋 SELECTING FINAL FEATURES")
    print("="*50)

    base_cols = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    engineered_cols = [
        "utilization_ratio", "avg_pay_delay", "max_pay_delay",
        "months_late", "payment_ratio", "delay_trend",
    ]
    final_cols = base_cols + engineered_cols + ["target"]

    df_final = df.select(final_cols)
    print(f"✅ Final feature count: {len(final_cols) - 1} (+ target)")
    return df_final


def split_and_save_stratified(df_final, train_ratio=0.8):
    print("\n" + "="*50)
    print("✂️  STRATIFIED TRAIN/TEST SPLIT")
    print("="*50)

    full_pdf = df_final.toPandas()
    X = full_pdf.drop("target", axis=1)
    y = full_pdf["target"]

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=1 - train_ratio, stratify=y, random_state=42
    )

    train_pdf = X_train.copy(); train_pdf["target"] = y_train
    test_pdf  = X_test.copy();  test_pdf["target"]  = y_test

    print(f"📊 Train set: {len(train_pdf)} rows | Defaulted: {(y_train==1).mean()*100:.1f}%")
    print(f"📊 Test set : {len(test_pdf)} rows | Defaulted: {(y_test==1).mean()*100:.1f}%")

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")
    train_pdf.to_csv(train_path, index=False)
    test_pdf.to_csv(test_path, index=False)

    print(f"✅ Saved → {train_path} ({train_pdf.shape})")
    print(f"✅ Saved → {test_path} ({test_pdf.shape})")


def run_full_preprocessing():
    spark = get_spark_session()
    try:
        df = load_raw_data(spark)
        df = clean_categorical_columns(df)
        df = create_new_features(df)
        df_final = select_final_features(df)
        split_and_save_stratified(df_final)

        print("\n" + "="*50)
        print("🎉 PREPROCESSING COMPLETE!")
        print("="*50)
        print("👉 Ab train.py dobara chalao (naya, bada dataset ke saath)")
    finally:
        spark.stop()
        print("🔴 Spark stopped.")


if __name__ == "__main__":
    run_full_preprocessing()