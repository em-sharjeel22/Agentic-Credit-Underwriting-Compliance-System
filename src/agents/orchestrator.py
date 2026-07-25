"""
=========================================================
File: orchestrator.py

Purpose:
    Entry point for the Credit Underwriting
    Multi-Agent System.
=========================================================
"""

import os
import pandas as pd

from agents.graph import build_graph

# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

TEST_DATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv",
)


def load_sample_applicant():
    """
    Load one applicant from the processed dataset.
    """

    test_df = pd.read_csv(TEST_DATA)

    applicant = (
        test_df
        .drop("target", axis=1)
        .iloc[0]
        .to_dict()
    )

    return applicant


def main():

    print("\n===================================")
    print(" CREDIT UNDERWRITING SYSTEM ")
    print("===================================\n")

    applicant = load_sample_applicant()

    graph = build_graph()

    result = graph.invoke(
        {
            "applicant": applicant
        }
    )

    print(result["final_report"])


if __name__ == "__main__":
    main()