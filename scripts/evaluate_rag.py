import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_MODEL_NAME,
    RAG_RESULTS_DIR,
    REFUSAL_PATTERNS,
    REPORTS_DIR,
)
from pipeline_utils import embed_texts_semantic

PAREN_PATTERN = re.compile(r"\([^)]*\)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str):
    return normalize_text(text).split()


def build_aliases(answer: str):
    """Return a set of normalised answer variants (handle 'X or Y' / 'X (Y)')."""
    aliases = {answer}
    no_parens = PAREN_PATTERN.sub(" ", answer).strip()
    aliases.add(no_parens)
    for part in re.split(r"\bor\b|/", no_parens, flags=re.IGNORECASE):
        part = part.strip(" ,")
        if part:
            aliases.add(part)
    return {normalize_text(a) for a in aliases if a.strip()}


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


def exact_match(prediction: str, gold: str) -> int:
    pred_norm = normalize_text(prediction)
    aliases = build_aliases(gold)
    return int(any(alias == pred_norm for alias in aliases))


def contains_match(prediction: str, gold: str) -> int:
    """Relaxed: 1 if any alias appears as a substring in the prediction."""
    pred_norm = normalize_text(prediction)
    aliases = build_aliases(gold)
    return int(any(alias and alias in pred_norm for alias in aliases))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_semantic_similarities(predictions: list, golds: list, model_name: str):
    """
    Embed all predictions and gold answers together, then compute pairwise
    cosine similarity for each (prediction_i, gold_i) pair.
    Returns a list of float similarity scores.
    """
    n = len(predictions)
    all_texts = predictions + golds
    all_embeddings = embed_texts_semantic(all_texts, model_name)

    pred_embeddings = all_embeddings[:n]
    gold_embeddings = all_embeddings[n:]

    similarities = []
    for p_emb, g_emb in zip(pred_embeddings, gold_embeddings):
        p_vec = p_emb.reshape(1, -1)
        g_vec = g_emb.reshape(1, -1)
        sim = float(cosine_similarity(p_vec, g_vec)[0][0])
        similarities.append(round(sim, 4))

    return similarities


def evaluate_file(path: str, semantic_model: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return None

    predictions = [item["model_answer"] for item in data]
    golds = [item["expected_answer"] for item in data]

    print(f"  Computing semantic similarity for {os.path.basename(path)} ...")
    sem_sims = compute_semantic_similarities(predictions, golds, semantic_model)

    per_question = []
    total_em = 0
    total_contains = 0
    total_f1 = 0.0
    total_sem = 0.0
    total_refusals = 0

    for item, sem_sim in zip(data, sem_sims):
        pred = item["model_answer"]
        gold = item["expected_answer"]

        em = exact_match(pred, gold)
        cm = contains_match(pred, gold)
        f1 = f1_score(pred, gold)
        refusal = is_refusal(pred)

        total_em += em
        total_contains += cm
        total_f1 += f1
        total_sem += sem_sim
        if refusal:
            total_refusals += 1

        per_question.append(
            {
                "question": item["question"],
                "expected_answer": gold,
                "model_answer": pred,
                "exact_match": em,
                "contains_match": cm,
                "f1_score": round(f1, 4),
                "semantic_similarity": sem_sim,
                "is_refusal": refusal,
            }
        )

    n = len(data)
    return {
        "num_questions": n,
        "exact_match": round(total_em / n, 4),
        "contains_match": round(total_contains / n, 4),
        "average_f1": round(total_f1 / n, 4),
        "average_semantic_similarity": round(total_sem / n, 4),
        "refusal_rate": round(total_refusals / n, 4),
        "per_question": per_question,
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    rag_files = sorted(
        name for name in os.listdir(RAG_RESULTS_DIR) if name.endswith(".json")
    )
    if not rag_files:
        raise ValueError(f"No RAG result files found in {RAG_RESULTS_DIR}")

    print(f"Found {len(rag_files)} RAG result file(s).")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}\n")

    report = {}
    for filename in rag_files:
        path = os.path.join(RAG_RESULTS_DIR, filename)
        print(f"Evaluating: {filename}")
        metrics = evaluate_file(path, EMBEDDING_MODEL_NAME)
        if metrics is None:
            print(f"  [WARN] Empty file, skipping.")
            continue

        summary = {k: v for k, v in metrics.items() if k != "per_question"}
        print(
            f"  EM={summary['exact_match']:.3f}  "
            f"contains={summary['contains_match']:.3f}  "
            f"F1={summary['average_f1']:.3f}  "
            f"sem_sim={summary['average_semantic_similarity']:.3f}  "
            f"refusals={summary['refusal_rate']:.3f}"
        )
        report[filename] = metrics

    output_path = os.path.join(REPORTS_DIR, "rag_evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFull report saved to {output_path}")

    print("\n--- Summary (sorted by exact_match) ---")
    rows = [
        (
            name,
            m["exact_match"],
            m["contains_match"],
            m["average_f1"],
            m["average_semantic_similarity"],
        )
        for name, m in report.items()
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    header = f"{'Config':<55} {'EM':>6} {'Sub':>6} {'F1':>6} {'Sem':>6}"
    print(header)
    print("-" * len(header))
    for name, em, sub, f1, sem in rows:
        short = name.replace("rag_retrieval_", "").replace(".json", "")
        print(f"{short:<55} {em:>6.3f} {sub:>6.3f} {f1:>6.3f} {sem:>6.3f}")


if __name__ == "__main__":
    main()
