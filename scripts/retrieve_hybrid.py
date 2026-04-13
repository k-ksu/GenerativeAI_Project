import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RETRIEVAL_RESULTS_DIR,
    TOP_K_VALUES,
)

RRF_K = 60

CANDIDATE_POOL_K = max(TOP_K_VALUES)


def load_retrieval_file(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_matching_pairs(retrieval_dir: str):
    suffix = f"_k_{CANDIDATE_POOL_K}.json"
    all_files = os.listdir(retrieval_dir)

    bge_files = {
        f: os.path.join(retrieval_dir, f)
        for f in all_files
        if f.startswith("bge_") and f.endswith(suffix)
    }
    bm25_files = {
        f: os.path.join(retrieval_dir, f)
        for f in all_files
        if f.startswith("bm25_") and f.endswith(suffix)
    }

    pairs = []
    for bge_name, bge_path in sorted(bge_files.items()):
        bm25_name = bge_name.replace("bge_", "bm25_", 1)
        if bm25_name in bm25_files:
            pairs.append((bge_path, bm25_files[bm25_name]))
        else:
            print(f"[WARN] No BM25 counterpart found for {bge_name}, skipping.")

    return pairs


def rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def fuse_results_for_question(bge_chunks, bm25_chunks):
    bge_ranks = {c["chunk_id"]: c["rank"] for c in bge_chunks}
    bm25_ranks = {c["chunk_id"]: c["rank"] for c in bm25_chunks}

    all_chunk_ids = set(bge_ranks) | set(bm25_ranks)

    chunk_meta = {}
    for c in bge_chunks:
        chunk_meta[c["chunk_id"]] = c
    for c in bm25_chunks:
        if c["chunk_id"] not in chunk_meta:
            chunk_meta[c["chunk_id"]] = c

    scored = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in bge_ranks:
            score += rrf_score(bge_ranks[chunk_id])
        if chunk_id in bm25_ranks:
            score += rrf_score(bm25_ranks[chunk_id])

        meta = chunk_meta[chunk_id]
        scored.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "source_file": meta["source_file"],
                "text": meta["text"],
                "rrf_score": round(score, 6),
                "bge_rank": bge_ranks.get(chunk_id),
                "bm25_rank": bm25_ranks.get(chunk_id),
            }
        )

    scored.sort(key=lambda x: x["rrf_score"], reverse=True)
    return scored


def build_output_filename(chunk_size: int, overlap: int, top_k: int) -> str:
    return f"hybrid_retrieval_size_{chunk_size}_overlap_{overlap}_k_{top_k}.json"


def save_results(meta: dict, results: list, output_path: str, top_k: int):
    output = {
        "embedding_backend": "hybrid_rrf",
        "embedding_model": f"bge+bm25 (RRF k={RRF_K})",
        "embedding_dimension": None,
        "chunk_size": meta["chunk_size"],
        "overlap": meta["overlap"],
        "top_k": top_k,
        "num_questions": len(results),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    print(f"Saved hybrid retrieval results to {output_path}")


def process_pair(bge_path: str, bm25_path: str):
    bge_payload = load_retrieval_file(bge_path)
    bm25_payload = load_retrieval_file(bm25_path)

    chunk_size = bge_payload["chunk_size"]
    overlap = bge_payload["overlap"]
    meta = {"chunk_size": chunk_size, "overlap": overlap}

    bge_results = {item["question_id"]: item for item in bge_payload["results"]}
    bm25_results = {item["question_id"]: item for item in bm25_payload["results"]}

    question_ids = sorted(bge_results.keys())

    fused_per_question = {}
    for qid in question_ids:
        bge_item = bge_results[qid]
        bm25_item = bm25_results.get(qid, {"retrieved_chunks": []})
        fused = fuse_results_for_question(
            bge_item["retrieved_chunks"],
            bm25_item["retrieved_chunks"],
        )
        fused_per_question[qid] = {
            "question_id": qid,
            "question": bge_item["question"],
            "expected_answer": bge_item["expected_answer"],
            "fused_chunks": fused,
        }

    for top_k in TOP_K_VALUES:
        results = []
        for qid in question_ids:
            item = fused_per_question[qid]
            top_chunks = item["fused_chunks"][:top_k]
            ranked_chunks = [
                {
                    "rank": rank + 1,
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "source_file": c["source_file"],
                    "score": c["rrf_score"],
                    "text": c["text"],
                }
                for rank, c in enumerate(top_chunks)
            ]
            results.append(
                {
                    "question_id": qid,
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "top_k": top_k,
                    "retrieved_chunks": ranked_chunks,
                }
            )

        output_filename = build_output_filename(chunk_size, overlap, top_k)
        output_path = os.path.join(RETRIEVAL_RESULTS_DIR, output_filename)
        save_results(meta, results, output_path, top_k)


def main():
    os.makedirs(RETRIEVAL_RESULTS_DIR, exist_ok=True)

    pairs = find_matching_pairs(RETRIEVAL_RESULTS_DIR)
    if not pairs:
        raise ValueError(
            f"No matching BGE+BM25 pairs found in {RETRIEVAL_RESULTS_DIR}.\n"
            "Run scripts/retrieve_bge.py and scripts/retrieve_bm25.py first."
        )

    print(f"Found {len(pairs)} BGE+BM25 pairs. Fusing with RRF (k={RRF_K}) ...")
    for bge_path, bm25_path in pairs:
        print(f"\n  BGE:  {os.path.basename(bge_path)}")
        print(f"  BM25: {os.path.basename(bm25_path)}")
        process_pair(bge_path, bm25_path)


if __name__ == "__main__":
    main()
