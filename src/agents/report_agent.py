from state import UnderwritingState
def report_agent(state: UnderwritingState) -> dict:
    print("📄 [Report Agent] Compiling final report...")
    factors_text = "\n".join([
        f"   {'🔴' if f['impact'] > 0 else '🟢'} {f['feature']}: {f['impact']:+.3f}"
        for f in state["top_risk_factors"]
    ])
    warnings_text = (
        "\n".join(f"   ⚠️  {w}" for w in state.get("data_warnings", [])) or "   None"
    )
    report = f"""
{'='*55}
CREDIT UNDERWRITING REPORT
{'='*55}
DATA VALIDATION
{warnings_text}
RISK ASSESSMENT
   Decision            : {state['risk_decision']}
   Default Probability : {state['risk_probability']*100:.1f}%
   Top Factors:
{factors_text}
COMPLIANCE CHECK
   {state['compliance_answer']}
   Sources: {', '.join(state['compliance_sources'])}
{'='*55}
"""
    return {"final_report": report}
