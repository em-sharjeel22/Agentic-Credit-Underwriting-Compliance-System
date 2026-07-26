# src/genai/generate_insights.py
import os
import argparse
import pandas as pd

def call_genai_api(prompt: str) -> str:
    """
    Replace this stub with your provider's client call.
    Keep API keys in env vars and never hardcode them.
    """
    # Example pseudo-call:
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=512)
    # return resp.output_text
    return "GENAI_RESPONSE_PLACEHOLDER: Replace call_genai_api with real API call."

def summarize_dataframe(df: pd.DataFrame, n_rows=5) -> str:
    stats = df.describe().transpose().round(3)
    top = df.head(n_rows).to_dict(orient="records")
    return f"Stats:\n{stats}\n\nSample rows:\n{top}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # load parquet or csv
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    prompt = (
        "You are a data analyst assistant. Given the following dataset summary, "
        "write 6 concise, actionable insights about credit default risk and suggest 3 next steps for modeling.\n\n"
    )
    prompt += summarize_dataframe(df)

    genai_text = call_genai_api(prompt)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(genai_text)

if __name__ == "__main__":
    main()
