import json
import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAG_DIR = os.path.join(RESULTS_DIR, "rag")
REPORT_PATH = os.path.join(RESULTS_DIR, "stage3_report.json")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(expected, predicted):
    expected_norm = normalize_text(expected)
    predicted_norm = normalize_text(predicted)
    expected_tokens = set(expected_norm.split())
    predicted_tokens = set(predicted_norm.split())
    return expected_tokens.issubset(predicted_tokens)


def evaluate_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_questions = len(data)
    num_answered = 0
    num_em = 0
    total_chunks_used = 0

    for item in data:
        expected = item["expected_answer"]
        predicted = item["model_answer"]
        chunks_used = item.get("num_chunks_used", 1)

        total_chunks_used += chunks_used

        if predicted.strip().lower() not in ["not found", "", "error"]:
            num_answered += 1

        if exact_match(expected, predicted):
            num_em += 1

    coverage = num_answered / num_questions
    avg_chunks_used = total_chunks_used / num_questions
    em_rate = num_em / num_questions

    return {
        "num_questions": num_questions,
        "exact_match": em_rate,
        "coverage": coverage,
        "avg_chunks_used": avg_chunks_used,
    }


def main():
    rag_files = [f for f in os.listdir(RAG_DIR) if f.endswith(".json")]
    all_metrics = {}

    for file in rag_files:
        path = os.path.join(RAG_DIR, file)
        metrics = evaluate_file(path)
        all_metrics[file] = metrics
        print(f"{file}: EM={metrics['exact_match']:.2f}, coverage={metrics['coverage']:.2f}, avg_chunks={metrics['avg_chunks_used']:.2f}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved all metrics to {REPORT_PATH}")


if __name__ == "__main__":
    main()