import json
import os
import re
import string
from collections import Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REFUSAL_PATTERNS
from config import BASELINE_OUTPUTS_FILE, EVAL_REPORT_FILE

RESULTS_PATH = BASELINE_OUTPUTS_FILE
REPORT_PATH = EVAL_REPORT_FILE


# -----------------------------------------------------------
# Text Normalization Utilities
# -----------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    This follows the standard normalization approach used in QA benchmarks (e.g., SQuAD).
    """
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def tokenize(text: str):
    """
    Tokenize normalized text by whitespace.
    """
    return normalize_text(text).split()


# -----------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------

def exact_match(prediction: str, gold: str) -> int:
    """
    Returns 1 if normalized prediction exactly matches normalized gold answer.
    Otherwise returns 0.
    """
    return int(normalize_text(prediction) == normalize_text(gold))


def f1_score(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 score.
    This measures overlap between prediction and gold tokens.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def simple_accuracy(prediction: str, gold: str) -> int:
    """
    Baseline compatibility metric.
    Returns 1 if gold answer appears as substring in prediction.
    """
    return int(gold.lower() in prediction.lower())

def is_refusal(prediction: str) -> bool:
    """
    Detect whether the model refused to answer.
    We flag typical refusal phrases.
    """
    prediction_lower = prediction.lower()
    return any(pattern in prediction_lower for pattern in REFUSAL_PATTERNS)


# -----------------------------------------------------------
# Main Evaluation Logic
# -----------------------------------------------------------

def main():
    # Check if results file exists
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results file not found at {RESULTS_PATH}")

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    if total == 0:
        raise ValueError("Results file is empty.")

    exact_matches = 0
    f1_total = 0.0
    accuracy_total = 0
    refusals = 0

    error_cases = []

    for item in data:
        prediction = item["model_answer"]
        gold = item["expected_answer"]

        # Compute metrics
        em = exact_match(prediction, gold)
        f1 = f1_score(prediction, gold)
        acc = simple_accuracy(prediction, gold)
        refusal_flag = is_refusal(prediction)

        exact_matches += em
        f1_total += f1
        accuracy_total += acc

        if refusal_flag:
            refusals += 1

        # Store incorrect cases based on substring accuracy (main metric)
        if acc == 0:
            error_cases.append({
                "question": item["question"],
                "expected_answer": gold,
                "model_answer": prediction,
                "f1_score": round(f1, 3),
                "is_refusal": refusal_flag
            })

    # Aggregate metrics
    exact_match_score = exact_matches / total
    average_f1 = f1_total / total
    accuracy_score = accuracy_total / total
    refusal_rate = refusals / total

    # Print summary to console
    print("========== Evaluation Summary ==========")
    print(f"Total Questions: {total}")
    print(f"Accuracy (substring): {accuracy_score:.3f}")
    print(f"Exact Match: {exact_match_score:.3f}")
    print(f"Average F1: {average_f1:.3f}")
    print(f"Refusal Rate: {refusal_rate:.3f}")
    print(f"Incorrect Answers: {len(error_cases)}")
    print("========================================")

    # Save detailed report
    report = {
        "total_questions": total,
        "accuracy_substring": round(accuracy_score, 4),
        "exact_match": round(exact_match_score, 4),
        "average_f1": round(average_f1, 4),
        "refusal_rate": round(refusal_rate, 4),
        "incorrect_cases": error_cases
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()