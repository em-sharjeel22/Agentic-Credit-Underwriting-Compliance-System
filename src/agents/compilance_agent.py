import re
import logging
from typing import Dict, Any, List
from constants import LLM_MODEL
from utils.resources import load_compliance_resources
from state import UnderwritingState

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
    Agent responsible for checking SBP regulations using Intent Routing (Summary -> Regex -> Vector).
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
        # 0. Catch Global/Aggregation Queries first (Bypasses search for count/listing questions)
        summary_match = re.search(r'(how many|list all|total number of|what are the)\s*regulations', query, re.IGNORECASE)
        
        if summary_match:
            logger.info("📊 [DEBUG] Global summary query detected. Bypassing search mechanisms.")
            return {
                "compliance_question": query,
                "compliance_answer": (
                    "Based on the State Bank of Pakistan's Prudential Regulations for Consumer Financing, "
                    "there are a total of 29 specific regulations provided in this dataset: \n"
                    "- 21 'R' (Risk) Regulations (R-1 through R-21)\n"
                    "- 8 'O' (Operations) Regulations (O-1 through O-8)\n"
                    "There is also a 'Definitions & Minimum Requirements' section."
                ),
                "compliance_sources": ["System Summary"]
            }

        retrieved = []
        
        # 1. Aggressive regex to catch variations like "R-1", "R1", "R 1", or "O-5"
        specific_reg_match = re.search(r'\b([R|O])\s*[-_]?\s*(\d+)\b', query, re.IGNORECASE)

        if specific_reg_match:
            reg_letter = specific_reg_match.group(1).upper() # Gets 'R'
            reg_num = specific_reg_match.group(2)            # Gets '1'
            reg_id = f"{reg_letter}-{reg_num}"               # Standardizes to 'R-1'
            
            logger.info(f"🔍 [DEBUG] Agent caught exact request for: {reg_id}")
            
            # 2. Search both section titles and body text to be safe
            for chunk in CHUNKS:
                section = chunk.get("section", "").upper()
                text = chunk.get("text", "").upper()
                
                if reg_id in section or f"REGULATION {reg_id}" in text:
                    retrieved.append(chunk)
                    logger.info(f"✅ [DEBUG] Successfully grabbed {reg_id} directly from JSON!")
                    break # Found it, stop searching

        # 3. Fallback to FAISS if not found (or if not a specific regulation query)
        if not retrieved:
            if specific_reg_match:
                logger.warning(f"⚠️ [DEBUG] {reg_id} was requested but NOT found in JSON. Falling back to FAISS.")
            
            query_vector = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
            _, indices = INDEX.search(query_vector, 5) 
            retrieved = [CHUNKS[i] for i in indices[0] if 0 <= i < len(CHUNKS)]
        
        # 4. Format the retrieved chunks for the LLM
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