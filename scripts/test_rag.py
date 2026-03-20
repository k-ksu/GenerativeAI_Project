import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma:2b"

def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    return response.json()["message"]["content"]


# === load 1 retrieval file ===
with open("results/retrieval/retrieval_size_200_overlap_40_k_1.json") as f:
    data = json.load(f)

# === take first question ===
item = data["results"][0]

question = item["question"]
chunk_text = item["retrieved_chunks"][0]["text"]

prompt = f"""
Answer ONLY using the context.

Context:
{chunk_text}

Question:
{question}

Answer:
"""

print("PROMPT:\n", prompt[:500], "\n")

answer = ask_llm(prompt)

print("\nMODEL ANSWER:\n", answer)