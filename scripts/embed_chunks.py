import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHUNKS_DIR,
    EMBEDDING_BACKEND,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_DIR,
)
from pipeline_utils import embed_texts_hashing


def build_output_path(input_filename: str) -> str:
    stem = os.path.splitext(input_filename)[0]
    return os.path.join(EMBEDDINGS_DIR, f"{stem}_embeddings.json")


def embed_chunk_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    chunks = payload["chunks"]
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts_hashing(texts, EMBEDDING_DIMENSION)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "embedding": embedding.tolist(),
            }
        )

    result = {
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": payload["chunk_size"],
        "overlap": payload["overlap"],
        "num_chunks": payload["num_chunks"],
        "chunks": embedded_chunks,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved embeddings to {output_path}")


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    chunk_filenames = sorted(
        name for name in os.listdir(CHUNKS_DIR) if name.endswith(".json")
    )
    if not chunk_filenames:
        raise ValueError(f"No chunk files found in {CHUNKS_DIR}")

    for filename in chunk_filenames:
        input_path = os.path.join(CHUNKS_DIR, filename)
        output_path = build_output_path(filename)
        embed_chunk_file(input_path, output_path)


if __name__ == "__main__":
    main()
