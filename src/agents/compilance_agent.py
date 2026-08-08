import logging
from typing import Dict, Any, List
from constants import LLM_MODEL
from utils.resources import load_compliance_resources
from state import UnderwritingState
import re
import logging
from typing import Dict, Any, List
from constants import LLM_MODEL
# ... (rest of your imports)

# Configure standard logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OPTIMIZATION: Load heavy ML models and vector index once globally.
try:
    INDEX, CHUNKS, EMBED_MODEL, LLM_CLIENT = load_compliance_resources()
except Exception as e:
    logger.error(f"Failed to load compliance resources: {e}")
    INDEX, CHUNKS, EMBED_MODEL, LLM_CLIENT = None, [], None, None

def build_compliance_query(state: UnderwritingState) -> str:
    """
    Constructs the appropriate compliance query based on the underwriting state flags.
    """
    flags: Dict[str, Any] = state.get("sbp_flags", {})
    
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
    decision: str = state.get("risk_decision", "review").lower()
    applicant: Dict[str, Any] = state.get("applicant", {})
    credit_limit: Any = applicant.get("LIMIT_BAL", "N/A")
    
    return state.get("compliance_question") or (
        f"What SBP regulations apply when a bank {decision}s "
        f"a consumer financing application with credit limit {credit_limit}?"
    )



def compliance_agent(state: UnderwritingState) -> Dict[str, Any]:
    """
    Agent responsible for checking SBP regulations using Hybrid RAG (Regex + Vector).
    """
    logger.info("📖 [Compliance Agent] Checking SBP regulations...")
    
    query = build_compliance_query(state)
    
    # Fallback if resources failed to load during global initialization
    if INDEX is None or EMBED_MODEL is None or not CHUNKS:
        logger.error("Compliance resources are not available.")
        return {
            "compliance_question": query,
            "compliance_answer": "System error: Compliance resources unavailable.",
            "compliance_sources": [],
        }
    
    try:
        retrieved = []
        
        # 1. Check if the query is asking for a specific regulation (e.g., "R-1" or "O-5")
        specific_reg_match = re.search(r'(Regulation|Reg)?\s*([R|O]-\d+)', query, re.IGNORECASE)

        if specific_reg_match:
            # 2. Extract the exact ID (e.g., "R-1")
            reg_id = specific_reg_match.group(2).upper()
            logger.info(f"Direct regulation lookup detected for: {reg_id}")
            
            # 3. Do an exact keyword match against the CHUNKS first
            for chunk in CHUNKS:
                if reg_id in chunk.get("section", "").upper():
                    retrieved.append(chunk)

        # 4. If exact match fails (or wasn't requested), fall back to Vector Search
        if not retrieved:
            query_vector = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
            _, indices = INDEX.search(query_vector, 5) # Increased to 5 for better semantic reach
            retrieved = [CHUNKS[i] for i in indices[0] if 0 <= i < len(CHUNKS)]
        
        # 5. Format the retrieved chunks for the LLM
        context = "\n\n".join([f"[{c.get('section', 'Unknown')}]\n{c.get('text', '')}" for c in retrieved])
        sources = [c.get("section", "Unknown") for c in retrieved]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "compliance_question": query,
            "compliance_answer": "Error retrieving compliance documents.",
            "compliance_sources": [],
        }
    
    if LLM_CLIENT is None:
        answer = (
            "LLM is not configured (missing GROQ_API_KEY). "
            "Showing retrieved regulation text instead:\n\n" + context
        )
    else:
        try:
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
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            answer = "Error generating answer from the LLM."
            
    logger.info(f"   → Checked against {len(retrieved)} regulations")
    
    return {
        "compliance_question": query,
        "compliance_answer": answer,
        "compliance_sources": sources,
    }