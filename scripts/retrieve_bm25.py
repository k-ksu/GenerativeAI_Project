import json
import os
import re
import sys

from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHUNKS_DIR,
    QUESTIONS_FILE,
    RETRIEVAL_RESULTS_DIR,
    TOP_K_VALUES,
)

TOKENIZE_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str):
    return TOKENIZE_PATTERN.findall(text.lower())


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    normalized_questions = []
    for index, item in enumerate(questions, start=1):
        normalized_questions.append(
            {
                "question_id": f"q_{index:03d}",
                "question": item["question"],
                "answer": item.get("answer", ""),
            }
        )
    return normalized_questions


def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_bm25_index(chunks):
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def build_output_filename(chunk_size: int, overlap: int, top_k: int) -> str:
    return f"bm25_retrieval_size_{chunk_size}_overlap_{overlap}_k_{top_k}.json"


def retrieve_top_k(questions, chunks, bm25_index, top_k: int):
    results = []

    for question in questions:
        query_tokens = tokenize(question["question"])
        scores = bm25_index.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        retrieved_chunks = []
        for rank, chunk_index in enumerate(top_indices, start=1):
            chunk = chunks[chunk_index]
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source_file": chunk["source_file"],
                    "score": float(scores[chunk_index]),
                    "text": chunk["text"],
                }
            )

        results.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "expected_answer": question["answer"],
                "top_k": top_k,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    return results


def save_results(payload_meta, results, output_path: str, top_k: int):
    output = {
        "embedding_backend": "bm25",
        "embedding_model": "bm25",
        "embedding_dimension": None,
        "chunk_size": payload_meta["chunk_size"],
        "overlap": payload_meta["overlap"],
        "top_k": top_k,
        "num_questions": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Saved BM25 retrieval results to {output_path}")


def main():
    os.makedirs(RETRIEVAL_RESULTS_DIR, exist_ok=True)

    chunk_filenames = sorted(
        name for name in os.listdir(CHUNKS_DIR) if name.endswith(".json")
    )
    if not chunk_filenames:
        raise ValueError(f"No chunk files found in {CHUNKS_DIR}")

    questions = load_questions()

    for filename in chunk_filenames:
        path = os.path.join(CHUNKS_DIR, filename)
        payload = load_chunks(path)
        chunks = payload["chunks"]

        print(f"\nBuilding BM25 index for {filename} ({len(chunks)} chunks) ...")
        bm25_index = build_bm25_index(chunks)

        for top_k in TOP_K_VALUES:
            results = retrieve_top_k(questions, chunks, bm25_index, top_k)
            output_filename = build_output_filename(
                payload["chunk_size"],
                payload["overlap"],
                top_k,
            )
            output_path = os.path.join(RETRIEVAL_RESULTS_DIR, output_filename)
            save_results(payload, results, output_path, top_k)


if __name__ == "__main__":
    main()
