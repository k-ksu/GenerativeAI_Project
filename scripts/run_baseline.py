import json
import requests
import os
from config import MODEL_NAME, OLLAMA_URL, QUESTIONS_FILE, RESULTS_DIR, BASELINE_OUTPUTS_FILE, DEFAULT_PROMPT_TEMPLATE

MODEL = MODEL_NAME
QUESTIONS_PATH = QUESTIONS_FILE
RESULTS_PATH = BASELINE_OUTPUTS_FILE
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

        prompt = DEFAULT_PROMPT_TEMPLATE.format(question=q)

        answer = ask_llm(prompt)

        results.append({
            "question": q,
            "expected_answer": item["answer"],
            "model_answer": answer
        })

        print(f"Answered: {q}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BASELINE_OUTPUTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()