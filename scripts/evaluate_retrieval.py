import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import QUESTIONS_FILE, REPORTS_DIR, RETRIEVAL_LABELS_FILE, RETRIEVAL_RESULTS_DIR


WHITESPACE_PATTERN = re.compile(r"\s+")
PAREN_PATTERN = re.compile(r"\([^)]*\)")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = PAREN_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def build_default_aliases(answer: str):
    aliases = {answer}
    no_parens = PAREN_PATTERN.sub(" ", answer)
    aliases.add(no_parens)
    for part in re.split(r"\bor\b|/", no_parens, flags=re.IGNORECASE):
        part = part.strip(" ,")
        if part:
            aliases.add(part)
    return [alias for alias in sorted({normalize_text(x) for x in aliases}) if alias]


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    by_id = {}
    for index, item in enumerate(questions, start=1):
        question_id = f"q_{index:03d}"
        by_id[question_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "answer_aliases": build_default_aliases(item["answer"]),
        }
    return by_id


def load_labels():
    with open(RETRIEVAL_LABELS_FILE, "r", encoding="utf-8") as file:
        labels = json.load(file)
    return {item["question_id"]: item for item in labels}


def evaluate_file(path: str, questions_by_id, labels_by_id):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    total = len(payload["results"])
    doc_hits = 0
    answer_hits = 0
    missed_questions = []

    for item in payload["results"]:
        question_id = item["question_id"]
        label = labels_by_id[question_id]
        question_meta = questions_by_id[question_id]

        expected_docs = set(label["source_doc_ids"])
        chunk_docs = {chunk["doc_id"] for chunk in item["retrieved_chunks"]}
        doc_hit = not expected_docs.isdisjoint(chunk_docs)

        normalized_chunk_text = " ".join(
            normalize_text(chunk["text"]) for chunk in item["retrieved_chunks"]
        )
        answer_hit = any(
            alias and alias in normalized_chunk_text
            for alias in question_meta["answer_aliases"]
        )

        if doc_hit:
            doc_hits += 1
        if answer_hit:
            answer_hits += 1
        if not doc_hit or not answer_hit:
            missed_questions.append(
                {
                    "question_id": question_id,
                    "question": item["question"],
                    "source_doc_ids": sorted(expected_docs),
                    "retrieved_doc_ids": sorted(chunk_docs),
                    "document_hit": doc_hit,
                    "answer_hit": answer_hit,
                }
            )

    return {
        "embedding_backend": payload["embedding_backend"],
        "embedding_model": payload["embedding_model"],
        "chunk_size": payload["chunk_size"],
        "overlap": payload["overlap"],
        "top_k": payload["top_k"],
        "num_questions": total,
        "document_recall": round(doc_hits / total, 4),
        "answer_recall": round(answer_hits / total, 4),
        "missed_questions": missed_questions,
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    questions_by_id = load_questions()
    labels_by_id = load_labels()

    retrieval_files = sorted(
        name for name in os.listdir(RETRIEVAL_RESULTS_DIR) if name.endswith(".json")
    )
    if not retrieval_files:
        raise ValueError(f"No retrieval files found in {RETRIEVAL_RESULTS_DIR}")

    report = {}
    for filename in retrieval_files:
        path = os.path.join(RETRIEVAL_RESULTS_DIR, filename)
        metrics = evaluate_file(path, questions_by_id, labels_by_id)
        report[filename] = metrics
        print(
            f"{filename}: "
            f"doc_recall={metrics['document_recall']:.4f}, "
            f"answer_recall={metrics['answer_recall']:.4f}"
        )

    output_path = os.path.join(REPORTS_DIR, "retrieval_report.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nSaved retrieval report to {output_path}")


if __name__ == "__main__":
    main()
