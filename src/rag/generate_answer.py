import os

from huggingface_hub import InferenceClient

from query_rag import load_vectorstore, search

client = InferenceClient(
    provider="groq",
    api_key=os.environ["GROQ_API_KEY"],
)

SYSTEM_PROMPT = """You are an SBP compliance officer explaining consumer financing regulations.
Use only the provided regulation text. If the answer is not covered by the supplied text, say so clearly and do not speculate.
Always quote the regulation number when relevant."""


def generate_answer(query, index, chunks, model, top_k=3):
    results = search(query, index, chunks, model, top_k=top_k)
    context = "\n\n".join([f"[{result['section']}]\n{result['text']}" for result in results])

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Regulations:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content, results


if __name__ == "__main__":
    print("Loading vector store...")
    index, chunks, model = load_vectorstore()
    print("Vector store is ready.\n")

    query = "Can a bank finance a used car that is 10 years old?"
    print(f"Query: {query}\n")

    answer, sources = generate_answer(query, index, chunks, model)
    print("Answer:")
    print(answer)
    print(f"\nSources: {[source['section'] for source in sources]}")