# -----------------------
# Paths
# -----------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
QUESTIONS_FILE = "data/questions.json"
RESULTS_DIR = "results"
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
# Other constants
# -----------------------
DEFAULT_PROMPT_TEMPLATE = """Answer the following history question concisely.

Question: {question}
Answer:"""