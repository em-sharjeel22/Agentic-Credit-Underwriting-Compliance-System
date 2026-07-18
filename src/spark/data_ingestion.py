# ============================================
# DATA INGESTION — Default of Credit Card Clients (UCI)
# ============================================

import os
from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# UCI ne generic X1, X2... diye hain — readable naam de rahe hain,
# taake aage EDA/SHAP charts mein "X6" ki jagah "PAY_0" dikhe
COLUMN_RENAME_MAP = {
    "X1": "LIMIT_BAL",  "X2": "SEX",        "X3": "EDUCATION",
    "X4": "MARRIAGE",   "X5": "AGE",
    "X6": "PAY_0",       "X7": "PAY_2",      "X8": "PAY_3",
    "X9": "PAY_4",       "X10": "PAY_5",     "X11": "PAY_6",
    "X12": "BILL_AMT1",  "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",  "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",   "X19": "PAY_AMT2",  "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",   "X22": "PAY_AMT5",  "X23": "PAY_AMT6",
}


def download_credit_card_dataset():
    print("\n📥 Downloading 'Default of Credit Card Clients' (UCI, id=350)...")
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.rename(columns=COLUMN_RENAME_MAP)
    y = dataset.data.targets

    df = X.copy()
    df["target"] = y.iloc[:, 0]

    save_path = RAW_DATA_DIR / "credit_card_default.csv"
    df.to_csv(save_path, index=False)

    print(f"✅ Downloaded! Shape: {df.shape}")
    print(f"💾 Saved → {save_path}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n🎯 Target Distribution:")
    print(df["target"].value_counts(normalize=True).round(3))

    return df


if __name__ == "__main__":
    download_credit_card_dataset()