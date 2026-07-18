import os

# ============================================
# FORCE JAVA 17 — is line ko sabse pehle rakhna hai,
# PySpark import se bhi pehle!
# Yeh ensure karta hai ke chahe PowerShell mein
# koi bhi Java "active" ho, humara script hamesha
# Java 17 hi use karega.
# ============================================
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17.0.19"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession


def get_spark_session(app_name="CreditUnderwriting"):
    """
    Spark Session = Our big data processing engine
    Isko ek baar start karo, poora project mein use karo
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("✅ Spark Engine Started Successfully!")
    print(f"🔧 Spark Version: {spark.version}")
    print(f"☕ Java Used: {os.environ['JAVA_HOME']}")

    return spark


if __name__ == "__main__":
    spark = get_spark_session()
    print("🚀 Ready to process data!")
    spark.stop()