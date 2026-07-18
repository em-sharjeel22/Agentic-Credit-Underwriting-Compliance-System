

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import functions as F
from spark_session import get_spark_session

# ── Paths ────────────────────────────────────
RAW_DATA_PATH  = "../../data/raw/german_credit.csv"
REPORTS_PATH   = "../../reports/"
os.makedirs(REPORTS_PATH, exist_ok=True)


def load_data(spark, path):
    """
    Data load karo
    CSV file ko PySpark mein laao
    """
    print("\n📂 Loading data...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"✅ Data loaded! Rows: {df.count():,} | Columns: {len(df.columns)}")
    return df


def basic_info(df):
    """
    Basic info — jaise kisi insaan se
    pehli baar milne par poochte hain:
    'Aap kaun hain? Kitne saal ke hain?'
    """
    print("\n" + "="*50)
    print("📊 BASIC INFORMATION")
    print("="*50)
    print(f"Total Applications : {df.count():,}")
    print(f"Total Features     : {len(df.columns)}")
    print("\n🔍 First 5 Rows:")
    df.show(5, truncate=True)
    print("\n📋 Column Data Types:")
    df.printSchema()


def check_target(df):
    """
    Target Variable check karo
    TARGET = 1 → Loan default kiya (bura)
    TARGET = 0 → Loan wapas kiya  (acha)
    """
    print("\n" + "="*50)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*50)

    target_dist = df.groupBy("TARGET") \
                    .count() \
                    .withColumn("Percentage",
                        F.round(F.col("count") / df.count() * 100, 2))
    target_dist.show()

    # Plot
    pdf = target_dist.toPandas()
    plt.figure(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]
    bars = plt.bar(
        ["Loan Repaid ✅", "Loan Defaulted ❌"],
        pdf["count"],
        color=colors,
        edgecolor="black"
    )
    for bar, pct in zip(bars, pdf["Percentage"]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 500,
                 f'{pct}%', ha='center', fontsize=12, fontweight='bold')

    plt.title("Loan Repaid vs Defaulted", fontsize=15, fontweight='bold')
    plt.ylabel("Number of Applicants")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}target_distribution.png", dpi=150)
    plt.close()
    print("📊 Chart saved → reports/target_distribution.png")


def check_missing_values(df):
    """
    Missing values = Data mein khaali jagahein
    Jaise form mein kuch fields khali chhod di hon
    """
    print("\n" + "="*50)
    print("❓ MISSING VALUES CHECK")
    print("="*50)

    total_rows = df.count()
    missing = []

    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            pct = round(null_count / total_rows * 100, 2)
            missing.append((col_name, null_count, pct))

    missing_df = pd.DataFrame(missing,
                              columns=["Column", "Missing Count", "Missing %"])
    missing_df = missing_df.sort_values("Missing %", ascending=False)

    print(f"\n⚠️  Columns with missing values: {len(missing_df)}")
    print(missing_df.head(20).to_string(index=False))

    # Plot top 15 missing
    top15 = missing_df.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top15["Column"], top15["Missing %"], color="#e67e22")
    plt.xlabel("Missing %")
    plt.title("Top 15 Columns with Missing Data", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}missing_values.png", dpi=150)
    plt.close()
    print("📊 Chart saved → reports/missing_values.png")

    return missing_df


def analyze_key_features(df):
    """
    Important features analyze karo
    Jo cheezein loan default se related hain
    """
    print("\n" + "="*50)
    print("🔑 KEY FEATURES ANALYSIS")
    print("="*50)

    # ── Credit Amount Distribution ────────────
    print("\n💰 Credit Amount Statistics:")
    df.select("credit_amount").describe().show()

    income_pdf = df.select("credit_amount", "target").toPandas()
    plt.figure(figsize=(10, 5))
    for target, color, label in [(0, "#2ecc71", "Repaid"),
                                  (1, "#e74c3c", "Defaulted")]:
        subset = income_pdf[income_pdf["target"] == target]["credit_amount"]
        plt.hist(subset, bins=30, alpha=0.6,
                 color=color, label=label, edgecolor="black")
    plt.title("Credit Amount Distribution: Repaid vs Defaulted",
              fontsize=13, fontweight='bold')
    plt.xlabel("Credit Amount")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}income_distribution.png", dpi=150)
    plt.close()
    print("📊 Chart saved → reports/income_distribution.png")

    # ── Age Distribution ───────────────────────
    print("\n👤 Age Statistics:")
    df.select("age").describe().show()

    # ── Job vs Default ─────────────────────────
    print("\n💼 Job Type vs Default Rate:")
    df.groupBy("job") \
      .agg(F.count("*").alias("Total"),
           F.sum("target").alias("Defaults"),
           F.round(F.mean("target") * 100, 2).alias("Default Rate %")) \
      .orderBy("Default Rate %", ascending=False) \
      .show()

    # ── Housing vs Default ─────────────────────
    print("\n🏠 Housing Type vs Default Rate:")
    df.groupBy("housing") \
      .agg(F.count("*").alias("Total"),
           F.round(F.mean("target") * 100, 2).alias("Default Rate %")) \
      .orderBy("Default Rate %", ascending=False) \
      .show()

    print("✅ Key features analyzed!")
    
def correlation_analysis(df):
    """
    Correlation = Konsi cheez loan default se
    kitni related hai?
    """
    print("\n" + "="*50)
    print("🔗 CORRELATION ANALYSIS")
    print("="*50)

    numeric_cols = [
        "duration", "credit_amount", "installment_rate",
        "residence_since", "age", "existing_credits",
        "dependents", "target"
    ]

    existing = [c for c in numeric_cols if c in df.columns]
    corr_pdf = df.select(existing).toPandas()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pdf.corr(), annot=True, fmt=".2f",
                cmap="RdYlGn", center=0,
                square=True, linewidths=0.5)
    plt.title("Feature Correlation Map\n(Green=Positive, Red=Negative)",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}correlation_heatmap.png", dpi=150)
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
        print("\n" + "="*50)
        print("🎉 EDA COMPLETE! Check reports/ folder")
        print("="*50)
    finally:
        spark.stop()
        print("🔴 Spark stopped.")


if __name__ == "__main__":
    run_full_eda()