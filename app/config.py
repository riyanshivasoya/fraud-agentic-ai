"""Central settings. Change model names and paths here or with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
AUDIT_LOG = LOG_DIR / "audit_log.jsonl"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
RULES_JSON = DATA_DIR / "rules.json"
TEST_CASES_JSON = DATA_DIR / "test_cases.json"

# Local models through Ollama (https://ollama.com). Pull them first, e.g. `ollama pull llama3.2:3b`.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# The three open-weight models compared in Stage 3. Same prompts, cases and output format for all.
DEFAULT_MODELS = [
    m.strip()
    for m in os.getenv("COMPARE_MODELS", "llama3.2:3b,qwen2.5:3b,mistral:7b").split(",")
    if m.strip()
]

# Optional hosted open models through any OpenAI-compatible API (Groq, OpenRouter, LM Studio...).
OPENAI_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")  # never hard-code keys

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "240"))
LLM_TEMPERATURE = 0.0  # deterministic as possible, so models are compared fairly

# Guardrail setting: "redact" removes suspected injection text before it reaches the model.
# "flag_only" keeps the text (still flagged) so you can test how each model resists it.
INJECTION_MODE = os.getenv("INJECTION_MODE", "redact")

# Demo access control. In a real bank this would be single sign-on with roles from IAM.
ROLES = {
    "Fraud Analyst": {"can_decide": ["human_review", "exception_review"]},
    "Senior Investigator": {"can_decide": ["human_review", "exception_review", "escalation"]},
    "Viewer (read only)": {"can_decide": []},
}

LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
