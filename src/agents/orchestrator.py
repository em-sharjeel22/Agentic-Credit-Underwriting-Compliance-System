# import os
import pandas as pd

from constants import PROJECT_ROOT
from utils.resources import preload_all
from graph import build_graph # Import it from your dedicated graph file!

preload_all()  # preload models and resources for all agents

if __name__ == "__main__":
    graph = build_graph()
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    sample_applicant = test_df.drop("target", axis=1).iloc[0].to_dict()
    
    print("🚀 Running full underwriting pipeline...\n")
    result = graph.invoke({"applicant": sample_applicant})
    print(result["final_report"])