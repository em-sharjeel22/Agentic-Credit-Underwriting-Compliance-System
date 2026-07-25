import pandas as pd

from compliance_agent import compliance_agent
from constants import PROJECT_ROOT
from data_agent import data_agent
from features import engineer_features
from report_agent import report_agent
from resources import preload_all
from risk_agent import risk_agent
from state import UnderwritingState
preload_all()  # preload models and resources for all agents

def build_graph():
    workflow = StateGraph(UnderwritingState)
    workflow.add_node("data_agent", data_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("compliance_agent", compliance_agent)
    workflow.add_node("report_agent", report_agent)
    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "risk_agent")
    workflow.add_edge("risk_agent", "compliance_agent")
    workflow.add_edge("compliance_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    return workflow.compile()

if __name__ == "__main__":
    graph = build_graph()
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    sample_applicant = test_df.drop("target", axis=1).iloc[0].to_dict()
    print("🚀 Running full underwriting pipeline...\n")
    result = graph.invoke({"applicant": sample_applicant})
    print(result["final_report"])
    
