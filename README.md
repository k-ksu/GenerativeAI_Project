# Mini-RAG Project — World History QA

**Research question:** How do retrieval design choices (chunk size, top-k, embedding model, retrieval method, prompt style) affect the faithfulness and accuracy of a local LLM's answers?

**Team:** Janna Ivanova & Ksenia Korchagina — Innopolis University  
**Compute:** Apple Silicon Mac M3/M4, fully local (no cloud APIs)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Corpus & Data](#corpus--data)
4. [Pipeline — How It Works](#pipeline--how-it-works)
5. [Running the Pipeline](#running-the-pipeline)
6. [Evaluation Scripts](#evaluation-scripts)
7. [Results by Stage](#results-by-stage)
8. [File Structure](#file-structure)

---

## Project Overview

This is a **mini Retrieval-Augmented Generation (RAG)** system for factual question answering over a World History document corpus.

The system is built in four stages:

| Stage | What was built |
|---|---|
| **1 — Proposal** | Research plan, corpus design, golden dataset spec |
| **2 — Baseline** | No-RAG baseline: `gemma:2b` answers questions from memory only |
| **3 — RAG Pipeline** | Full RAG: chunking → MiniLM embeddings → cosine retrieval → grounded generation |
| **4 — Improvements** | BGE-large embeddings, BM25, Hybrid RRF retrieval, extraction-style prompt, richer evaluation |

---

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
brew install ollama
ollama serve          # keep this running in a separate terminal
ollama pull gemma:2b
```

> **Note:** `ollama serve` must be running whenever you execute any script that calls the LLM (`run_baseline.py`, `run_rag.py`). All embedding and retrieval scripts work without Ollama.

---

## Corpus & Data

The corpus lives in `data/raw/` — 15 Wikipedia PDF articles covering European history and art:

```
Art of Europe, History of Europe, History of France, History of Germany,
History of Italy, History of Spain, Renaissance, French art,
20th-century Western painting, Ancient Greek architecture, Gothic architecture,
Famous Europeans, 12 Famous French Artists, 20 Influential Leaders,
List of conflicts in Europe
```

The golden evaluation dataset is `data/questions.json` — **44 manually verified QA pairs**.

If you add new PDFs to `data/raw/`, re-run the full preparation pipeline:

```bash
python scripts/prepare_data.py         # PDF → plain text
python scripts/clean_processed_texts.py # clean OCR artefacts
python scripts/chunk_documents.py       # split into chunks
python scripts/embed_chunks.py          # MiniLM embeddings
python scripts/embed_chunks_bge.py      # BGE-large embeddings
```

---

## Pipeline — How It Works

```
data/raw/ (PDFs)
    │
    ▼  prepare_data.py
data/processed/ (raw text)
    │
    ▼  clean_processed_texts.py
data/cleaned/ (cleaned text)
    │
    ▼  chunk_documents.py
data/chunks/ (chunk_size × overlap configs)
    │
    ├──▶ embed_chunks.py      → data/embeddings/     (MiniLM, 384-dim)
    └──▶ embed_chunks_bge.py  → data/embeddings_bge/ (BGE-large, 1024-dim)
                │
                ▼  retrieve.py / retrieve_bge.py / retrieve_bm25.py / retrieve_hybrid.py
        results/retrieval/ (ranked chunk lists per question)
                │
                ▼  run_rag.py
        results/rag/ (model answers)
                │
                ▼  evaluate_rag.py / evaluate_retrieval.py / score_faithfulness.py
        results/reports/ (metrics)
```

### Retrieval methods

| Method | Script | How it works |
|---|---|---|
| **MiniLM dense** | `retrieve.py` | Cosine similarity, `all-MiniLM-L6-v2` (384-dim) |
| **BGE dense** | `retrieve_bge.py` | Cosine similarity, `BAAI/bge-large-en-v1.5` (1024-dim) |
| **BM25** | `retrieve_bm25.py` | Lexical TF-IDF-like ranking via `rank_bm25` |
| **Hybrid RRF** | `retrieve_hybrid.py` | Reciprocal Rank Fusion of BGE + BM25 rankings |

### Prompt styles (in `run_rag.py`)

| Style | Behaviour |
|---|---|
| `default` | Grounded prompt: "Answer ONLY from context, say Not found if absent" |
| `extraction` | Short-answer prompt: "Extract the exact answer — name, date, or brief phrase" |

---

## Running the Pipeline

### Full pipeline from scratch (MiniLM)

```bash
python scripts/prepare_data.py
python scripts/clean_processed_texts.py
python scripts/chunk_documents.py
python scripts/embed_chunks.py
python scripts/retrieve.py
python scripts/run_rag.py
```

### Add BGE-large retrieval

```bash
python scripts/embed_chunks_bge.py   # ~5 min, downloads 1.3 GB model on first run
python scripts/retrieve_bge.py
python scripts/run_rag.py --filter "bge_"
```

### Add BM25 retrieval

```bash
python scripts/retrieve_bm25.py
python scripts/run_rag.py --filter "bm25_"
```

### Add Hybrid retrieval (requires BGE + BM25 to be done first)

```bash
python scripts/retrieve_hybrid.py
python scripts/run_rag.py --filter "hybrid_"
```

### Run with extraction-style prompt

```bash
# on a specific config (fast):
python scripts/run_rag.py --filter "size_200_overlap_40_k_5" --prompt-style extraction

# on all files:
python scripts/run_rag.py --prompt-style extraction
```

### Useful flags for `run_rag.py`

```bash
--filter "bge_"              # only process files whose name contains this substring
--prompt-style extraction    # use extraction prompt instead of default
--no-skip-existing           # rerun even if output already exists
```

### Baseline (no retrieval)

```bash
python scripts/run_baseline.py       # saves results/baseline_outputs.json
python scripts/evaluate.py           # substring accuracy, F1, exact match
```

---

## Evaluation Scripts

| Script | Input | Output | Metrics |
|---|---|---|---|
| `evaluate_retrieval.py` | `results/retrieval/` | `results/reports/retrieval_report.json` | document_recall, answer_recall |
| `evaluate_rag.py` | `results/rag/` | `results/reports/rag_evaluation_report.json` | contains_match, F1, semantic_similarity, refusal_rate |
| `score_faithfulness.py` | `results/rag/` + `results/retrieval/` | `results/reports/faithfulness_report.json` | faithfulness_score (supported / partial / unsupported) |
| `scripts/visualization/metrics_counter.py` | `results/rag/` | `results/stage3_report.json` | exact_match (relaxed), coverage |
| `scripts/visualization/result_graphics.py` | `results/stage3_report.json` | charts (matplotlib) | bar + scatter plots |

---

## Results by Stage

### Stage 2 — Baseline (no RAG)

Model: `gemma:2b` answering from memory, no documents.

- **Substring accuracy: 0.57** (25 / 44 correct)
- Main failure modes: factual hallucinations, refusals, imprecise wording

### Stage 3 — RAG with MiniLM (6 configurations)

| Configuration | Exact Match (relaxed) |
|---|---|
| size=200, overlap=40, k=5 | **0.159** ✓ best |
| size=200, overlap=40, k=1 | 0.113 |
| size=500, overlap=100, k=1 | 0.113 |
| size=1000, overlap=200, k=5 | 0.113 |
| size=500, overlap=100, k=5 | 0.091 |
| size=1000, overlap=200, k=1 | **0.023** ✗ worst |

Key finding: 7× performance gap from retrieval design alone, generator unchanged.

### Stage 4 — Retrieval & Prompt Improvements

All results at `size=200, overlap=40, k=5`.  
Metrics from `evaluate_rag.py` (contains_match = relaxed substring) and `score_faithfulness.py`.

| Method | Prompt | Contains↑ | F1↑ | Sem.Sim↑ | Refusals↓ | Faithfulness↑ |
|---|---|---|---|---|---|---|
| MiniLM | default | 0.227 | 0.070 | 0.460 | 0.705 | 0.239 |
| BGE-large | default | 0.204 | 0.065 | 0.465 | 0.705 | 0.261 |
| BM25 | default | 0.204 | 0.043 | 0.440 | 0.773 | 0.205 |
| Hybrid RRF | default | 0.159 | 0.062 | 0.453 | 0.750 | 0.193 |
| MiniLM | extraction | 0.341 | 0.092 | **0.494** | 0.568 | 0.318 |
| BGE-large | extraction | 0.318 | 0.084 | 0.487 | 0.523 | 0.375 |
| BM25 | extraction | 0.273 | 0.064 | 0.465 | 0.568 | 0.318 |
| **Hybrid RRF** | **extraction** | **0.364** | **0.092** | 0.470 | **0.455** | **0.386** |

**Retrieval quality** (answer_recall at k=5):

| Method | size=200 | size=500 | size=1000 |
|---|---|---|---|
| MiniLM | 0.795 | 0.841 | 0.841 |
| BM25 | 0.705 | 0.864 | 0.841 |
| BGE-large | **0.886** | 0.864 | 0.864 |
| Hybrid RRF | 0.864 | 0.864 | **0.886** |

**Key Stage 4 findings:**
- Extraction prompt is the single biggest win: refusals drop from ~70% to ~45–57%
- BGE-large retrieves correct answers in 88.6% of cases vs 79.5% for MiniLM
- Hybrid RRF + extraction prompt = best overall across contains, F1, and faithfulness
- BM25 alone is weaker than dense retrieval for semantic history questions, but helps in the hybrid

---

## File Structure

```
GenerativeAI_Project/
├── config.py                        # all paths, model names, hyperparameters
├── pipeline_utils.py                # shared embedding functions
├── requirements.txt
│
├── data/
│   ├── raw/                         # 15 source PDFs
│   ├── processed/                   # extracted plain text
│   ├── cleaned/                     # cleaned text
│   ├── chunks/                      # chunked JSON (3 size configs)
│   ├── embeddings/                  # MiniLM embeddings
│   ├── embeddings_bge/              # BGE-large embeddings
│   ├── questions.json               # 44 golden QA pairs
│   └── retrieval_labels.json        # ground-truth source doc IDs per question
│
├── scripts/
│   ├── prepare_data.py              # PDF → text
│   ├── clean_processed_texts.py     # text cleaning
│   ├── chunk_documents.py           # text → chunks
│   ├── embed_chunks.py              # MiniLM embeddings
│   ├── embed_chunks_bge.py          # BGE-large embeddings  [Stage 4]
│   ├── retrieve.py                  # dense cosine retrieval (MiniLM)
│   ├── retrieve_bge.py              # dense cosine retrieval (BGE) [Stage 4]
│   ├── retrieve_bm25.py             # BM25 lexical retrieval        [Stage 4]
│   ├── retrieve_hybrid.py           # Hybrid RRF (BGE + BM25)       [Stage 4]
│   ├── run_baseline.py              # no-RAG baseline
│   ├── run_rag.py                   # RAG generation (--filter, --prompt-style)
│   ├── evaluate.py                  # baseline evaluation
│   ├── evaluate_retrieval.py        # retrieval recall metrics       [Stage 4]
│   ├── evaluate_rag.py              # RAG metrics + semantic sim     [Stage 4]
│   ├── score_faithfulness.py        # faithfulness scoring
│   └── visualization/
│       ├── metrics_counter.py       # aggregate exact match report
│       └── result_graphics.py       # matplotlib charts
│
└── results/
    ├── baseline_outputs.json
    ├── stage3_report.json
    ├── retrieval/                   # all retrieval JSON files
    ├── rag/                         # all RAG answer JSON files
    └── reports/
        ├── retrieval_report.json    # document_recall + answer_recall
        ├── faithfulness_report.json # faithfulness scores
        └── rag_evaluation_report.json # EM, F1, semantic similarity
```
