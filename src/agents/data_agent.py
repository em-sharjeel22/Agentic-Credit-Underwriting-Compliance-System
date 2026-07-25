from constants import SBP_R8_AGGREGATE_CAP, SBP_R8_PERSONAL_CLEAN_CAP
from state import UnderwritingState
def data_agent(state: UnderwritingState) -> dict:
    """
    Validates applicant data and flags SBP R-8 exposure-cap breaches.
    Runs first in the pipeline.
    """
    print("🗂️  [Data Agent] Validating + flagging applicant data...")
    applicant = state["applicant"]
    warnings = []
    if applicant.get("LIMIT_BAL", 0) <= 0:
        warnings.append("LIMIT_BAL is zero or negative")
    if not (18 <= applicant.get("AGE", 0) <= 100):
        warnings.append("AGE is outside plausible range (18-100)")
    if applicant.get("EDUCATION") not in [1, 2, 3, 4]:
        warnings.append("EDUCATION code not in expected set (1-4)")
    if applicant.get("MARRIAGE") not in [1, 2, 3]:
        warnings.append("MARRIAGE code not in expected set (1-3)")
    limit_bal = applicant.get("LIMIT_BAL", 0)
    sbp_flags = {
        "exceeds_r8_personal_clean_cap": limit_bal > SBP_R8_PERSONAL_CLEAN_CAP,
        "exceeds_r8_aggregate_cap": limit_bal > SBP_R8_AGGREGATE_CAP,
    }
    print(
        f"   → Warnings: {len(warnings)} | "
        f"R-8 clean cap exceeded: {sbp_flags['exceeds_r8_personal_clean_cap']}"
    )
    return {"data_warnings": warnings, "sbp_flags": sbp_flags}
