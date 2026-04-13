import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAG_RESULTS_DIR, REPORTS_DIR, RETRIEVAL_RESULTS_DIR

PAREN_PATTERN = re.compile(r"\([^)]*\)")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "by",
    "with",
    "was",
    "were",
    "is",
    "are",
    "be",
    "as",
    "at",
    "from",
    "that",
    "this",
    "it",
    "its",
    "their",
    "his",
    "her",
    "he",
    "she",
    "they",
    "them",
    "which",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "into",
    "during",
    "after",
    "before",
    "under",
    "over",
    "than",
    "then",
    "also",
    "called",
    "known",
    "name",
    "title",
    "question",
    "answer",
    "context",
    "provided",
    "passage",
}
REFUSAL_PATTERNS = [
    "not found",
    "cannot answer",
    "cannot determine",
    "does not provide",
    "does not mention",
    "cannot be answered",
    "not in the context",
    "from the provided context",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = PAREN_PATTERN.sub(" ", text)
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def build_aliases(answer: str):
    aliases = {answer}
    no_parens = PAREN_PATTERN.sub(" ", answer)
    aliases.add(no_parens)
    for part in re.split(r"\bor\b|/", no_parens, flags=re.IGNORECASE):
        part = part.strip(" ,")
        if part:
            aliases.add(part)
    return [alias for alias in sorted({normalize_text(x) for x in aliases}) if alias]


def is_refusal(answer_text: str) -> bool:
    normalized = normalize_text(answer_text)
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def informative_tokens(text: str):
    tokens = []
    for token in normalize_text(text).split():
        if len(token) <= 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def classify_faithfulness(model_answer: str, expected_answer: str, context_text: str):
    if not model_answer.strip() or is_refusal(model_answer):
        return "unsupported", 0.0

    answer_norm = normalize_text(model_answer)
    context_norm = normalize_text(context_text)
    aliases = build_aliases(expected_answer)

    alias_in_answer = any(alias and alias in answer_norm for alias in aliases)
    alias_in_context = any(alias and alias in context_norm for alias in aliases)

    answer_tokens = informative_tokens(model_answer)
    if answer_tokens:
        overlap_count = sum(
            1 for token in answer_tokens if token in context_norm.split()
        )
        overlap_ratio = overlap_count / len(answer_tokens)
    else:
        overlap_ratio = 0.0

    if alias_in_answer and alias_in_context:
        return "supported", 1.0

    if overlap_ratio >= 0.75 and alias_in_context:
        return "supported", overlap_ratio

    if alias_in_context or alias_in_answer or overlap_ratio >= 0.4:
        return "partially_supported", overlap_ratio

    return "unsupported", overlap_ratio


def matching_retrieval_filename(rag_filename: str) -> str:
    if rag_filename.startswith("extraction_rag_"):
        return rag_filename[len("extraction_rag_") :]
    if rag_filename.startswith("rag_"):
        return rag_filename[len("rag_") :]
    raise ValueError(f"Unexpected RAG filename: {rag_filename}")


def score_file(rag_path: str, retrieval_path: str):
    with open(rag_path, "r", encoding="utf-8") as file:
        rag_data = json.load(file)
    with open(retrieval_path, "r", encoding="utf-8") as file:
        retrieval_data = json.load(file)

    supported = 0
    partial = 0
    unsupported = 0
    per_question = []

    for rag_item, retrieval_item in zip(rag_data, retrieval_data["results"]):
        context_text = "\n\n".join(
            chunk["text"] for chunk in retrieval_item["retrieved_chunks"]
        )
        label, overlap_ratio = classify_faithfulness(
            rag_item["model_answer"],
            rag_item["expected_answer"],
            context_text,
        )

        if label == "supported":
            supported += 1
        elif label == "partially_supported":
            partial += 1
        else:
            unsupported += 1

        per_question.append(
            {
                "question_id": retrieval_item["question_id"],
                "question": rag_item["question"],
                "label": label,
                "token_overlap_ratio": round(overlap_ratio, 4),
                "expected_answer": rag_item["expected_answer"],
                "model_answer": rag_item["model_answer"],
            }
        )

    total = len(rag_data)
    faithfulness_score = (supported + 0.5 * partial) / total

    return {
        "num_questions": total,
        "supported": supported,
        "partially_supported": partial,
        "unsupported": unsupported,
        "faithfulness_score": round(faithfulness_score, 4),
        "per_question": per_question,
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    rag_files = sorted(
        name for name in os.listdir(RAG_RESULTS_DIR) if name.endswith(".json")
    )
    if not rag_files:
        raise ValueError(f"No RAG files found in {RAG_RESULTS_DIR}")

    report = {}
    for rag_filename in rag_files:
        retrieval_filename = matching_retrieval_filename(rag_filename)
        rag_path = os.path.join(RAG_RESULTS_DIR, rag_filename)
        retrieval_path = os.path.join(RETRIEVAL_RESULTS_DIR, retrieval_filename)
        metrics = score_file(rag_path, retrieval_path)
        report[rag_filename] = metrics
        print(
            f"{rag_filename}: "
            f"faithfulness={metrics['faithfulness_score']:.4f}, "
            f"supported={metrics['supported']}, "
            f"partial={metrics['partially_supported']}, "
            f"unsupported={metrics['unsupported']}"
        )

    output_path = os.path.join(REPORTS_DIR, "faithfulness_report.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nSaved faithfulness report to {output_path}")


if __name__ == "__main__":
    main()
