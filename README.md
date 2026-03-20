# Mini-RAG Project – Baseline

## Project Overview

This project investigates question answering over a domain-specific corpus (World History documents).  

In Stage 2, we implement a **baseline system** where a local Large Language Model (LLM) answers factual questions **without retrieval**.  

This serves as a controlled reference point before introducing retrieval mechanisms in later stages.


## Python Environment Setup

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Baseline Definition

Baseline = Local LLM answering questions without access to external documents.

Model used: **gemma:2b** (via Ollama)

No retrieval.
No embeddings.
No document grounding.


## Setup Instructions (Mac M1/M2)

### 1. Install Ollama

```bash
brew install ollama
```

### 2. Start Ollama server

```bash
ollama serve
```

### 3. Download model

```bash
ollama pull gemma:2b
```


## Prepare data

If you add new pdf documnets you should run:

```bash
python scripts/prepare_data.py
```

## Running the Baseline

```bash
python scripts/run_baseline.py
```

Outputs will be saved to <u>results/baseline_outputs.json</u>


## Evaluation

```bash
python scripts/evaluate.py
```

Metric: Accuracy = Correct answers / Total questions. This provides a quantitative baseline before introducing retrieval.



## Results on Stage 2

The model's accuracy is currently approximately **0.57**, as the number of correct answers is **25/44**.

Major issues:

1. Completely incorrect answers (factual errors)

Q4: Expected Augustus, but the model answered Julius Caesar (who was assassinated 17 years before the creation of the empire).

2. The model refuses to answer

Q8, Q15, Q21, Q22, Q39: "The context does not mention..." - although the answers are in the documents

3. Incomplete/imprecise wording

Q36: Expected "The Arnolfini Portrait", the model answered "The Arnolfina" (misnamed)

## Results on Stage 3

By Stage 3 we have a fully working RAG pipeline connected end-to-end. The pipeline takes a question, retrieves relevant text chunks from our corpus, builds a grounded prompt, and generates an answer using a local LLM — all without any internet access or cloud APIs.

**What the pipeline does, step by step:**
- splits documents into chunks (200 / 500 / 1000 tokens) with configurable overlap
- embeds every chunk using `all-MiniLM-L6-v2`
- at query time, embeds the question the same way and ranks chunks by cosine similarity
- puts the top-k most similar chunks into a prompt and tells `gemma:2b` to answer only from that context

We ran six configurations varying chunk size, overlap, and k, and evaluated them on our 44-question golden dataset using exact match:

| Configuration | Exact Match | Coverage |
|---|---|---|
| size=200, overlap=40, **k=5** | **0.159** ✓ best | 1.0 |
| size=200, overlap=40, k=1 | 0.113 | 1.0 |
| size=500, overlap=100, k=1 | 0.113 | 1.0 |
| size=1000, overlap=200, k=5 | 0.113 | 1.0 |
| size=500, overlap=100, k=5 | 0.091 | 1.0 |
| size=1000, overlap=200, **k=1** | **0.023** ✗ worst | 1.0 |

A quick note on coverage: it's 1.0 for everything, which sounds great but actually just means the model always returned *something* — not that the answer was correct. A lot of those "answers" are refusals phrased differently. So we mostly look at exact match and failure cases, not coverage.

**Main takeaways:**
- Small chunks + broad retrieval (k=5) works best
- Very large chunks with narrow retrieval (k=1) is the worst combination — nearly useless at 0.023
- There's a 7× performance gap between best and worst config, and the only thing we changed was retrieval design

