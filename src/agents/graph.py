"""
=========================================================
File: graph.py

Purpose:
    Build the LangGraph workflow by connecting all agents.

Workflow:

    START
      │
      ▼
 Data Agent
      │
      ▼
 Risk Agent
      │
      ▼
 Compliance Agent
      │
      ▼
 Report Agent
      │
      ▼
      END
=========================================================
"""

from langgraph.graph import StateGraph, START, END

from agents.state import UnderwritingState
from agents.data_agent import data_agent
from agents.risk_agent import risk_agent
from agents.compliance_agent import compliance_agent
from agents.report_agent import report_agent


def build_graph():
    """
    Build and compile the underwriting workflow.
    """

    workflow = StateGraph(UnderwritingState)

    # ----------------------------------------
    # Register Agents
    # ----------------------------------------

    workflow.add_node(
        "data_agent",
        data_agent,
    )

    workflow.add_node(
        "risk_agent",
        risk_agent,
    )

    workflow.add_node(
        "compliance_agent",
        compliance_agent,
    )

    workflow.add_node(
        "report_agent",
        report_agent,
    )

    # ----------------------------------------
    # Workflow
    # ----------------------------------------

    workflow.add_edge(
        START,
        "data_agent",
    )

    workflow.add_edge(
        "data_agent",
        "risk_agent",
    )

    workflow.add_edge(
        "risk_agent",
        "compliance_agent",
    )

    workflow.add_edge(
        "compliance_agent",
        "report_agent",
    )

    workflow.add_edge(
        "report_agent",
        END,
    )

    return workflow.compile()