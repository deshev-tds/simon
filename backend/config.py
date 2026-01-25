import os
from pathlib import Path

SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
DEBUG_MODE = True
QUIET_LOGS = os.environ.get("SIMON_QUIET_LOGS") == "1"
def _get_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


def _normalize_lm_studio_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


LM_STUDIO_URL = _normalize_lm_studio_url(
    _get_str_env("SIMON_LM_STUDIO_URL", _get_str_env("LM_STUDIO_URL", "http://localhost:1234/v1"))
)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_PUBLIC_DIR = FRONTEND_DIR / "public"
CERTS_DIR = ROOT_DIR / "certs"
DATA_DIR = Path(os.environ.get("SIMON_DATA_DIR", str(BASE_DIR / "data")))
MODELS_DIR = Path(os.environ.get("SIMON_MODELS_DIR", str(BASE_DIR / "models")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ESP_AUDIO_DIR = DATA_DIR / "esp_audio"
ESP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SAVE_ESP_AUDIO = True
ESP_AUDIO_MAX_FILES = 20

INDEX_HTML_PATH = FRONTEND_DIR / "index.html"
try:
    INDEX_HTML = INDEX_HTML_PATH.read_text(encoding="utf-8")
except Exception:
    INDEX_HTML = "<!doctype html><html><body><h1>Frontend not found</h1></body></html>"
DB_PATH = DATA_DIR / "history.db"

WHISPER_MODEL_NAME = "distil-medium.en"
TTS_VOICE = "am_fenrir"
DEFAULT_LLM_MODEL = "local-model"

MAX_RECENT_MESSAGES = 40
ANCHOR_MESSAGES = 7
RAG_THRESHOLD = 0.35

FTS_MAX_HITS = 5
FTS_MIN_TOKEN_LEN = 3
FTS_PER_SESSION = True
FTS_DEDUP_MIN_LEN = 15
FTS_RECURSIVE_DEPTH = 2
FTS_RECURSIVE_MAX_QUERIES = 10
FTS_RECURSIVE_MAX_BRANCHES = 4
FTS_RECURSIVE_OVERSAMPLE = 5
FTS_RECURSIVE_MIN_MATCH = 2


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


MEM_SEED_LIMIT = _get_int_env("SIMON_MEM_SEED_LIMIT", 50000)
MEM_MAX_ROWS = _get_int_env("SIMON_MEM_MAX_ROWS", MEM_SEED_LIMIT)
MEM_PRUNE_INTERVAL_S = _get_int_env("SIMON_MEM_PRUNE_INTERVAL_S", 60)
RAG_DEBUG_VERBOSE = _get_int_env("SIMON_RAG_DEBUG_VERBOSE", 0) > 0
LLM_TIMEOUT_S = _get_int_env("SIMON_LLM_TIMEOUT_S", 0)

AGENT_MAX_TURNS = 4
AGENT_TRIGGER_KEYWORDS = {
    "research",
    "analyze",
    "deep dive",
    "\u043f\u0440\u043e\u0443\u0447\u0438",
    "\u0430\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0430\u0439",
    "deep mode",
}
MAX_TOOL_OUTPUT_CHARS = 12000

TEST_MODE = os.environ.get("SIMON_TEST_MODE") == "1"
SKIP_AUDIO_MODELS = os.environ.get("SIMON_SKIP_AUDIO_MODELS") == "1"
SKIP_VECTOR_MEMORY = os.environ.get("SIMON_SKIP_VECTOR_MEMORY") == "1"

__all__ = [
    "SAMPLE_RATE",
    "TTS_SAMPLE_RATE",
    "DEBUG_MODE",
    "QUIET_LOGS",
    "LM_STUDIO_URL",
    "BASE_DIR",
    "ROOT_DIR",
    "FRONTEND_DIR",
    "FRONTEND_PUBLIC_DIR",
    "CERTS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "ESP_AUDIO_DIR",
    "SAVE_ESP_AUDIO",
    "ESP_AUDIO_MAX_FILES",
    "INDEX_HTML_PATH",
    "INDEX_HTML",
    "DB_PATH",
    "WHISPER_MODEL_NAME",
    "TTS_VOICE",
    "DEFAULT_LLM_MODEL",
    "MAX_RECENT_MESSAGES",
    "ANCHOR_MESSAGES",
    "RAG_THRESHOLD",
    "FTS_MAX_HITS",
    "FTS_MIN_TOKEN_LEN",
    "FTS_PER_SESSION",
    "FTS_DEDUP_MIN_LEN",
    "FTS_RECURSIVE_DEPTH",
    "FTS_RECURSIVE_MAX_QUERIES",
    "FTS_RECURSIVE_MAX_BRANCHES",
    "FTS_RECURSIVE_OVERSAMPLE",
    "FTS_RECURSIVE_MIN_MATCH",
    "_get_int_env",
    "MEM_SEED_LIMIT",
    "MEM_MAX_ROWS",
    "MEM_PRUNE_INTERVAL_S",
    "RAG_DEBUG_VERBOSE",
    "LLM_TIMEOUT_S",
    "AGENT_MAX_TURNS",
    "AGENT_TRIGGER_KEYWORDS",
    "MAX_TOOL_OUTPUT_CHARS",
    "TEST_MODE",
    "SKIP_AUDIO_MODELS",
    "SKIP_VECTOR_MEMORY",
]
