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


