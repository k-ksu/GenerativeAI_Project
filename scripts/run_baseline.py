import json
import requests
import os

MODEL = "gemma:2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

QUESTIONS_PATH = "../data/questions.json"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )

    data = response.json()

    if "message" in data:
        return data["message"]["content"]
    else:
        print("ERROR FROM OLLAMA:", data)
        return "ERROR"

def main():
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    results = []

    for item in questions:
        q = item["question"]

        prompt = f"""Answer the following history question concisely.

Question: {q}
Answer:"""

        answer = ask_llm(prompt)

        results.append({
            "question": q,
            "expected_answer": item["answer"],
            "model_answer": answer
        })

        print(f"Answered: {q}")

    with open("../results/baseline_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()