import argparse
import json
import os
import sys

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import (
    EXTRACTION_PROMPT_TEMPLATE,
    MODEL_NAME,
    OLLAMA_URL,
    RAG_RESULTS_DIR,
    RETRIEVAL_RESULTS_DIR,
)

PROMPT_STYLES = {
    "default": None,  # uses build_prompt() which has its own template
    "extraction": EXTRACTION_PROMPT_TEMPLATE,
}

MODEL = MODEL_NAME


def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )

    data = response.json()

    if "message" in data:
        return data["message"]["content"]
    else:
        print("ERROR FROM OLLAMA:", data)
        return "ERROR"


def build_prompt(question, retrieved_chunks):
    MAX_CHUNKS = 3
    retrieved_chunks = retrieved_chunks[:MAX_CHUNKS]

    context = "\n\n".join(
        [f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
    )

    prompt = f"""You must answer the question ONLY using the provided context.
Do NOT use prior knowledge.
If the answer is not in the context, say "Not found".

Context:
{context}

Question:
{question}

Answer:"""

    return prompt


def build_extraction_prompt(
    question: str, retrieved_chunks: list, template: str
) -> str:
    MAX_CHUNKS = 3
    retrieved_chunks = retrieved_chunks[:MAX_CHUNKS]
    context = "\n\n".join(
        [f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
    )
    return template.format(context=context, question=question)


def load_retrieval_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_on_file(retrieval_path, prompt_template=None):
    data = load_retrieval_file(retrieval_path)

    results = []

    for item in data["results"]:
        question = item["question"]
        expected = item["expected_answer"]
        chunks = item["retrieved_chunks"]

        if prompt_template is not None:
            prompt = build_extraction_prompt(question, chunks, prompt_template)
        else:
            prompt = build_prompt(question, chunks)

        answer = ask_llm(prompt)

        results.append(
            {
                "question": question,
                "expected_answer": expected,
                "model_answer": answer,
                "num_chunks_used": min(len(chunks), 3),
            }
        )

        print(f"Answered: {question}")

    return results


def save_results(results, filename):
    os.makedirs(RAG_RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(RAG_RESULTS_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved RAG results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG generation on retrieval results."
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only process retrieval files whose name contains this substring "
        "(e.g. 'bge_' or 'size_200_overlap_40_k_5').",
    )
    parser.add_argument(
        "--prompt-style",
        default="default",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style to use: 'default' (grounded) or 'extraction' (short-answer). "
        "Extraction outputs are saved with an 'extraction_rag_' prefix.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip retrieval files whose RAG output already exists (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-run RAG even if the output file already exists.",
    )
    args = parser.parse_args()

    retrieval_files = sorted(
        f for f in os.listdir(RETRIEVAL_RESULTS_DIR) if f.endswith(".json")
    )

    if not retrieval_files:
        raise ValueError("No retrieval files found")

    if args.filter:
        retrieval_files = [f for f in retrieval_files if args.filter in f]
        if not retrieval_files:
            raise ValueError(f"No retrieval files match filter '{args.filter}'")
        print(f"Filter '{args.filter}' matched {len(retrieval_files)} file(s).")

    prompt_template = PROMPT_STYLES[args.prompt_style]
    output_prefix = "extraction_rag_" if args.prompt_style == "extraction" else "rag_"
    print(f"Prompt style: {args.prompt_style}  |  Output prefix: {output_prefix}\n")

    skipped = 0
    for file in retrieval_files:
        output_filename = f"{output_prefix}{file}"
        output_path = os.path.join(RAG_RESULTS_DIR, output_filename)

        if args.skip_existing and os.path.exists(output_path):
            print(f"[skip] {file}  (output already exists)")
            skipped += 1
            continue

        path = os.path.join(RETRIEVAL_RESULTS_DIR, file)
        print(f"\nRunning RAG for: {file}")
        results = run_rag_on_file(path, prompt_template=prompt_template)
        save_results(results, output_filename)

    if skipped:
        print(
            f"\n{skipped} file(s) skipped (already processed). "
            "Use --no-skip-existing to rerun them."
        )


if __name__ == "__main__":
    main()
