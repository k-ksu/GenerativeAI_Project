import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_CONFIGS, CHUNKS_DIR, CLEANED_DATA_DIR


def normalize_doc_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def read_processed_documents():
    documents = []
    for filename in sorted(os.listdir(CLEANED_DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(CLEANED_DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as file:
            text = " ".join(file.read().split())
        if not text:
            continue
        documents.append(
            {
                "doc_id": normalize_doc_id(filename),
                "source_file": filename,
                "text": text,
            }
        )
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(
            {
                "start_word": start,
                "end_word": end,
                "text": " ".join(chunk_words),
            }
        )
        if end == len(words):
            break

    return chunks


def build_chunk_records(documents, chunk_size: int, overlap: int):
    records = []

    for document in documents:
        doc_chunks = chunk_text(document["text"], chunk_size, overlap)
        for index, chunk in enumerate(doc_chunks):
            records.append(
                {
                    "doc_id": document["doc_id"],
                    "source_file": document["source_file"],
                    "chunk_id": f"{document['doc_id']}_chunk_{index:04d}",
                    "chunk_index": index,
                    "text": chunk["text"],
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                }
            )

    return records


def save_chunks(chunk_records, chunk_size: int, overlap: int):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    output_path = os.path.join(
        CHUNKS_DIR,
        f"chunks_size_{chunk_size}_overlap_{overlap}.json",
    )
    payload = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(chunk_records),
        "chunks": chunk_records,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunk_records)} chunks to {output_path}")


def main():
    documents = read_processed_documents()
    if not documents:
        raise ValueError(f"No cleaned documents found in {CLEANED_DATA_DIR}")

    for config in CHUNK_CONFIGS:
        chunk_records = build_chunk_records(
            documents,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
        )
        save_chunks(chunk_records, config["chunk_size"], config["overlap"])


if __name__ == "__main__":
    main()
