# ============================================
# EDA = Exploratory Data Analysis
# Understanding the data before modeling it.
#
# FIXED: this previously pointed at german_credit.csv (the
# small prototype dataset) and used its column names. The
# project moved to credit_card_default.csv (30,000 rows)
# long ago; this version matches that dataset.
# ============================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import functions as F
from spark_session import get_spark_session

# ── Paths (relative to this script's own location, so it
#    works no matter which directory you run it from) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credit_card_default.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data(spark, path):
    print("\n📂 Loading data...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"✅ Data loaded! Rows: {df.count():,} | Columns: {len(df.columns)}")
    return df


def basic_info(df):
    print("\n" + "=" * 50)
    print("📊 BASIC INFORMATION")
    print("=" * 50)
    print(f"Total Applications : {df.count():,}")
    print(f"Total Features     : {len(df.columns)}")
    print("\n🔍 First 5 Rows:")
    df.show(5, truncate=True)
    print("\n📋 Column Data Types:")
    df.printSchema()


def check_target(df):
    """
    TARGET = 1 → defaulted next month (bad)
    TARGET = 0 → paid on time         (good)
    """
    print("\n" + "=" * 50)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("=" * 50)

    target_dist = df.groupBy("target") \
        .count() \
        .withColumn("Percentage", F.round(F.col("count") / df.count() * 100, 2))
    target_dist.show()

    pdf = target_dist.toPandas().sort_values("target")
    plt.figure(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]
    bars = plt.bar(["Paid on Time", "Defaulted"], pdf["count"], color=colors, edgecolor="black")
    for bar, pct in zip(bars, pdf["Percentage"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                  f"{pct}%", ha="center", fontsize=12, fontweight="bold")
    plt.title("Payment Outcome Distribution", fontsize=15, fontweight="bold")
    plt.ylabel("Number of Clients")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "target_distribution.png"), dpi=150)
    plt.close()
    print("📊 Chart saved → reports/target_distribution.png")


def check_missing_values(df):
    print("\n" + "=" * 50)
    print("❓ MISSING VALUES CHECK")
    print("=" * 50)

    total_rows = df.count()
    missing = []
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            pct = round(null_count / total_rows * 100, 2)
            missing.append((col_name, null_count, pct))

    missing_df = pd.DataFrame(missing, columns=["Column", "Missing Count", "Missing %"])

    if missing_df.empty:
        print("✅ No missing values found in any column.")
    else:
        missing_df = missing_df.sort_values("Missing %", ascending=False)
        print(f"\n⚠️  Columns with missing values: {len(missing_df)}")
        print(missing_df.to_string(index=False))

        plt.figure(figsize=(10, 6))
        plt.barh(missing_df["Column"], missing_df["Missing %"], color="#e67e22")
        plt.xlabel("Missing %")
        plt.title("Columns with Missing Data", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "missing_values.png"), dpi=150)
        plt.close()
        print("📊 Chart saved → reports/missing_values.png")


def analyze_key_features(df):
    print("\n" + "=" * 50)
    print("🔑 KEY FEATURES ANALYSIS")
    print("=" * 50)

    # ── Credit Limit Distribution ─────────────
    print("\n💰 Credit Limit (LIMIT_BAL) Statistics:")
    df.select("LIMIT_BAL").describe().show()

    limit_pdf = df.select("LIMIT_BAL", "target").toPandas()
    plt.figure(figsize=(10, 5))
    for target, color, label in [(0, "#2ecc71", "Paid on Time"), (1, "#e74c3c", "Defaulted")]:
        subset = limit_pdf[limit_pdf["target"] == target]["LIMIT_BAL"]
        plt.hist(subset, bins=50, alpha=0.6, color=color, label=label, edgecolor="black")
    plt.title("Credit Limit Distribution: Paid vs Defaulted", fontsize=13, fontweight="bold")
    plt.xlabel("Credit Limit (NT dollars)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "limit_distribution.png"), dpi=150)
    plt.close()
    print("📊 Chart saved → reports/limit_distribution.png")

    # ── Age vs Default ────────────────────────
    print("\n👤 Age Statistics:")
    df.select("AGE").describe().show()

    # ── Sex vs Default ────────────────────────
    print("\n👤 Sex vs Default Rate (1=Male, 2=Female):")
    df.groupBy("SEX") \
        .agg(F.count("*").alias("Total"),
             F.round(F.mean("target") * 100, 2).alias("Default Rate %")) \
        .show()

    # ── Education vs Default ──────────────────
    print("\n🎓 Education vs Default Rate (1=Grad school, 2=University, 3=High school, 4=Other):")
    df.groupBy("EDUCATION") \
        .agg(F.count("*").alias("Total"),
             F.round(F.mean("target") * 100, 2).alias("Default Rate %")) \
        .orderBy("Default Rate %", ascending=False) \
        .show()

    # ── Most Recent Repayment Status vs Default ──
    print("\n📅 Most Recent Payment Status (PAY_0) vs Default Rate:")
    df.groupBy("PAY_0") \
        .agg(F.count("*").alias("Total"),
             F.round(F.mean("target") * 100, 2).alias("Default Rate %")) \
        .orderBy("PAY_0") \
        .show()

    print("✅ Key features analyzed!")


def correlation_analysis(df):
    print("\n" + "=" * 50)
    print("🔗 CORRELATION ANALYSIS")
    print("=" * 50)

    numeric_cols = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2",
        "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2",
        "target",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    corr_pdf = df.select(existing).toPandas()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pdf.corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5)
    plt.title("Feature Correlation Map\n(Green=Positive, Red=Negative)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("📊 Chart saved → reports/correlation_heatmap.png")


def run_full_eda(data_path=RAW_DATA_PATH):
    spark = get_spark_session()
    try:
        df = load_data(spark, data_path)
        basic_info(df)
        check_target(df)
        check_missing_values(df)
        analyze_key_features(df)
        correlation_analysis(df)
        print("\n" + "=" * 50)
        print("🎉 EDA COMPLETE! Check reports/ folder")
        print("=" * 50)
    finally:
        spark.stop()
        print("🔴 Spark stopped.")


if __name__ == "__main__":
    run_full_eda()