import json
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BGE_EMBEDDINGS_DIR,
    BGE_MODEL_NAME,
    QUESTIONS_FILE,
    RETRIEVAL_RESULTS_DIR,
    TOP_K_VALUES,
)
from pipeline_utils import embed_texts_semantic


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


def load_embeddings(path: str):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    chunk_embeddings = np.array(
        [chunk["embedding"] for chunk in payload["chunks"]],
        dtype=np.float32,
    )
    return payload, chunk_embeddings


def embed_queries(questions):
    texts = [item["question"] for item in questions]
    return embed_texts_semantic(texts, BGE_MODEL_NAME)


def build_output_filename(chunk_size: int, overlap: int, top_k: int) -> str:
    return f"bge_retrieval_size_{chunk_size}_overlap_{overlap}_k_{top_k}.json"


def retrieve_top_k(questions, payload, chunk_embeddings, top_k: int):
    query_embeddings = embed_queries(questions)
    similarity_matrix = cosine_similarity(query_embeddings, chunk_embeddings)
    chunks = payload["chunks"]

    results = []
    for question_index, question in enumerate(questions):
        scores = similarity_matrix[question_index]
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_chunks = []
        for rank, chunk_index in enumerate(top_indices, start=1):
            chunk = chunks[int(chunk_index)]
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source_file": chunk["source_file"],
                    "score": float(scores[int(chunk_index)]),
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


def save_results(payload, results, output_path: str, top_k: int):
    output = {
        "embedding_backend": payload["embedding_backend"],
        "embedding_model": payload["embedding_model"],
        "embedding_dimension": payload["embedding_dimension"],
        "chunk_size": payload["chunk_size"],
        "overlap": payload["overlap"],
        "top_k": top_k,
        "num_questions": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"Saved BGE retrieval results to {output_path}")


def main():
    os.makedirs(RETRIEVAL_RESULTS_DIR, exist_ok=True)

    if not os.path.isdir(BGE_EMBEDDINGS_DIR):
        raise FileNotFoundError(
            f"BGE embeddings directory not found: {BGE_EMBEDDINGS_DIR}\n"
            "Run scripts/embed_chunks_bge.py first."
        )

    embedding_files = sorted(
        name for name in os.listdir(BGE_EMBEDDINGS_DIR) if name.endswith(".json")
    )
    if not embedding_files:
        raise ValueError(f"No BGE embedding files found in {BGE_EMBEDDINGS_DIR}")

    questions = load_questions()

    print(f"Embedding {len(questions)} queries with {BGE_MODEL_NAME} ...")

    for filename in embedding_files:
        path = os.path.join(BGE_EMBEDDINGS_DIR, filename)
        payload, chunk_embeddings = load_embeddings(path)

        for top_k in TOP_K_VALUES:
            results = retrieve_top_k(questions, payload, chunk_embeddings, top_k)
            output_filename = build_output_filename(
                payload["chunk_size"],
                payload["overlap"],
                top_k,
            )
            output_path = os.path.join(RETRIEVAL_RESULTS_DIR, output_filename)
            save_results(payload, results, output_path, top_k)


if __name__ == "__main__":
    main()
