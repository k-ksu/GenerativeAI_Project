# -----------------------
# Paths
# -----------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CLEANED_DATA_DIR = "data/cleaned"
CHUNKS_DIR = "data/chunks"
EMBEDDINGS_DIR = "data/embeddings"
QUESTIONS_FILE = "data/questions.json"
RESULTS_DIR = "results"
RETRIEVAL_RESULTS_DIR = f"{RESULTS_DIR}/retrieval"
REPORTS_DIR = f"{RESULTS_DIR}/reports"
BASELINE_OUTPUTS_FILE = f"{RESULTS_DIR}/baseline_outputs.json"
EVAL_REPORT_FILE = f"{RESULTS_DIR}/evaluation_report.json"

# -----------------------
# Model / API
# -----------------------
MODEL_NAME = "gemma:2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# -----------------------
# Evaluation Settings
# -----------------------
REFUSAL_PATTERNS = [
    "cannot answer",
    "do not have enough information",
    "context does not mention",
    "not enough information",
    "cannot determine"
]

# -----------------------
# Retrieval Pipeline
# -----------------------
CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 1000, "overlap": 200},
]
EMBEDDING_BACKEND = "hashing"
EMBEDDING_MODEL_NAME = "sklearn-hashing-v1"
EMBEDDING_DIMENSION = 2048
TOP_K_VALUES = [1, 5]

# -----------------------
# Other constants
# -----------------------
DEFAULT_PROMPT_TEMPLATE = """Answer the following history question concisely.

Question: {question}
Answer:"""
