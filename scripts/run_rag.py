import json
import os
import requests
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import (
    MODEL_NAME,
    OLLAMA_URL,
    RETRIEVAL_RESULTS_DIR,
    RESULTS_DIR,
)

MODEL = MODEL_NAME

# 👉 новая папка (важно!)
RAG_RESULTS_DIR = os.path.join(RESULTS_DIR, "rag")


def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
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
        [f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
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


def load_retrieval_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_on_file(retrieval_path):
    data = load_retrieval_file(retrieval_path)

    results = []

    for item in data["results"]:
        question = item["question"]
        expected = item["expected_answer"]
        chunks = item["retrieved_chunks"]

        prompt = build_prompt(question, chunks)
        answer = ask_llm(prompt)

        results.append({
            "question": question,
            "expected_answer": expected,
            "model_answer": answer,
            "num_chunks_used": min(len(chunks), 3)
        })

        print(f"Answered: {question}")

    return results


def save_results(results, filename):
    os.makedirs(RAG_RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(RAG_RESULTS_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved RAG results to {output_path}")


def main():
    retrieval_files = [
        f for f in os.listdir(RETRIEVAL_RESULTS_DIR)
        if f.endswith(".json")
    ]

    if not retrieval_files:
        raise ValueError("No retrieval files found")

    for file in retrieval_files:
        path = os.path.join(RETRIEVAL_RESULTS_DIR, file)

        print(f"\nRunning RAG for: {file}")

        results = run_rag_on_file(path)

        output_filename = f"rag_{file}"
        save_results(results, output_filename)


if __name__ == "__main__":
    main()