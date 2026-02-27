import json

RESULTS_PATH = "../results/baseline_outputs.json"

def simple_match(pred, gold):
    return gold.lower() in pred.lower()

def main():
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    correct = 0

    for item in data:
        if simple_match(item["model_answer"], item["expected_answer"]):
            correct += 1

    total = len(data)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Correct: {correct}/{total}")

if __name__ == "__main__":
    main()