"""
LLM Answer Generation
---------------------
Generates compliance answers using Groq / Hugging Face Inference API.
"""

import os
from huggingface_hub import InferenceClient

SYSTEM_PROMPT = """You are an SBP compliance officer explaining consumer financing regulations.
Use only the provided regulation text. If the answer is not covered by the supplied text, say so clearly and do not speculate.
Always quote the regulation number when relevant."""


def get_llm_client():
    key = os.getenv("GROQ_API_KEY") or os.getenv("HF_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY or HF_API_KEY environment variable.")
    return InferenceClient(provider="groq", api_key=key)


def generate_answer(query, context, llm_client=None):
    if llm_client is None:
        llm_client = get_llm_client()

    completion = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content