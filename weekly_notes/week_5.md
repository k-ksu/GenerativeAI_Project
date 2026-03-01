## Weekly Report — Week 5  

### 1. Environment Setup

We configured a local inference environment:

- Installed Ollama
- Downloaded `gemma:2b` model
- Verified local API-based inference

The system runs fully locally (no external API calls).

### 2. Baseline Definition

Baseline = Direct LLM prompting (No Retrieval).

System characteristics:
- Model: gemma:2b
- Prompt-based QA

### 3. Implementation

We implemented:

- Clean inference script (`run_baseline.py`)
- Minimal dataset pipeline
- Evaluation script (`evaluate.py`)
- Reproducible project structure
- Requirements file for dependencies

Outputs:
- JSON file with model answers
- Accuracy metric computed automatically

### 4. Evaluation Results

The baseline was evaluated on the 50-question Dataset.

Measured:
- Exact factual match accuracy
- Hallucination rate (incorrect or fabricated answers)

Observations:
- Model performs reasonably on common historical facts.
- Hallucinations occur for more specific or less common events.
- Some answers are overly generic.


### 5. Sanity Check

We verified:
- System stability across multiple runs.
- Deterministic behavior for identical prompts.
- Proper handling of all dataset entries.


### 6. Current Status

✔ Working baseline system  
✔ Quantitative metric implemented  
✔ Ready for Stage 3 (retrieval integration)  

Project status: On schedule.