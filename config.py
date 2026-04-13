RAW_DATA_DIR = "data/raw"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_EMBEDDING_DIMENSION = 1024
BGE_EMBEDDINGS_DIR = "data/embeddings_bge"
PROCESSED_DATA_DIR = "data/processed"
CLEANED_DATA_DIR = "data/cleaned"
CHUNKS_DIR = "data/chunks"
EMBEDDINGS_DIR = "data/embeddings"
QUESTIONS_FILE = "data/questions.json"
RETRIEVAL_LABELS_FILE = "data/retrieval_labels.json"
RESULTS_DIR = "results"
RETRIEVAL_RESULTS_DIR = f"{RESULTS_DIR}/retrieval"
RAG_RESULTS_DIR = f"{RESULTS_DIR}/rag"
REPORTS_DIR = f"{RESULTS_DIR}/reports"
BASELINE_OUTPUTS_FILE = f"{RESULTS_DIR}/baseline_outputs.json"
EVAL_REPORT_FILE = f"{RESULTS_DIR}/evaluation_report.json"

MODEL_NAME = "gemma:2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

REFUSAL_PATTERNS = [
    "cannot answer",
    "do not have enough information",
    "context does not mention",
    "not enough information",
    "cannot determine",
]

CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 1000, "overlap": 200},
]
EMBEDDING_BACKEND = "semantic"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
TOP_K_VALUES = [1, 3, 5, 10]

DEFAULT_PROMPT_TEMPLATE = """Answer the following history question concisely.

Question: {question}
Answer:"""

EXTRACTION_PROMPT_TEMPLATE = """Extract the exact answer from the context below.
Give a short factual answer only — a name, date, or brief phrase.
Do not explain. Do not repeat the question.
If the answer is not in the context, say "Not found".

Context:
{context}

Question: {question}

Answer:"""
