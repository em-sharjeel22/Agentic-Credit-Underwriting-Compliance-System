from constants import LLM_MODEL
from utils.resources import load_compliance_resources
from state import UnderwritingState

# OPTIMIZATION: Load heavy ML models and vector index once globally, 
# rather than on every single agent execution.
INDEX, CHUNKS, EMBED_MODEL, LLM_CLIENT = load_compliance_resources()

def build_compliance_query(state: UnderwritingState) -> str:
    flags = state.get("sbp_flags", {})
    
    if flags.get("exceeds_r8_aggregate_cap"):
        return (
            "What SBP regulation applies when a customer's aggregate clean "
            "credit card and personal loan exposure exceeds Rs 5,000,000?"
        )
    if flags.get("exceeds_r8_personal_clean_cap"):
        return (
            "What SBP regulation applies when a customer's clean credit "
            "card limit exceeds Rs 2,000,000?"
        )
        
    # SAFETY: Use safe .get() with fallback defaults to prevent KeyErrors
    decision = state.get("risk_decision", "review").lower()
    applicant = state.get("applicant", {})
    credit_limit = applicant.get("LIMIT_BAL", "N/A")
    
    return state.get("compliance_question") or (
        f"What SBP regulations apply when a bank {decision}s "
        f"a consumer financing application with credit limit {credit_limit}?"
    )

def compliance_agent(state: UnderwritingState) -> dict:
    print("📖 [Compliance Agent] Checking SBP regulations...")
    
    query = build_compliance_query(state)
    
    # Vector search using the globally loaded model and index
    query_vector = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = INDEX.search(query_vector, 3)
    
    retrieved = [CHUNKS[i] for i in indices[0]]
    context = "\n\n".join([f"[{c['section']}]\n{c['text']}" for c in retrieved])
    
    if LLM_CLIENT is None:
        answer = (
            "LLM is not configured (missing GROQ_API_KEY). "
            "Showing retrieved regulation text instead:\n\n" + context
        )
    else:
        completion = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SBP compliance officer. Answer ONLY using the "
                        "regulation text provided below. Always cite the regulation number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Regulations:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content
        
    print(f"   → Checked against {len(retrieved)} regulations")
    
    return {
        "compliance_question": query,
        "compliance_answer": answer,
        "compliance_sources": [c["section"] for c in retrieved],
    }