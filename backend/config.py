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
CORPUS_DB_PATH = DATA_DIR / "corpus.db"

WHISPER_MODEL_NAME = "distil-medium.en"
TTS_VOICE = "bm_fable"
DEFAULT_LLM_MODEL = _get_str_env("SIMON_DEFAULT_LLM_MODEL", "")

MAX_RECENT_MESSAGES = 2
ANCHOR_MESSAGES = 2
RAG_THRESHOLD = 0.35

FTS_MAX_HITS = 5
FTS_MIN_TOKEN_LEN = 3
FTS_PER_SESSION = True
FTS_DEDUP_MIN_LEN = 15
FTS_RECURSIVE_DEPTH = 2
FTS_RECURSIVE_MAX_QUERIES = 10
FTS_RECURSIVE_MAX_BRANCHES = 4
FTS_RECURSIVE_OVERSAMPLE = 5
FTS_RECURSIVE_MIN_MATCH = 1


def _get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


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
MEM_HOT_SESSION_LIMIT = _get_int_env("SIMON_MEM_HOT_SESSION_LIMIT", 100)
RAG_DEBUG_VERBOSE = _get_int_env("SIMON_RAG_DEBUG_VERBOSE", 0) > 0
LLM_TIMEOUT_S = _get_int_env("SIMON_LLM_TIMEOUT_S", 0)
LOCAL_WEAK_THRESHOLD = _get_float_env("SIMON_LOCAL_WEAK_THRESHOLD", 0.45)
ARCHIVE_STRONG_THRESHOLD = _get_float_env("SIMON_ARCHIVE_STRONG_THRESHOLD", 0.25)
ARCHIVE_EXPLICIT_THRESHOLD = _get_float_env("SIMON_ARCHIVE_EXPLICIT_THRESHOLD", 0.45)
ARCHIVE_WEAK_MAX_AGE_DAYS = _get_int_env("SIMON_ARCHIVE_WEAK_MAX_AGE_DAYS", 730)
ARCHIVE_FTS_MAX_SESSIONS = _get_int_env("SIMON_ARCHIVE_FTS_MAX_SESSIONS", 3)
ARCHIVE_FTS_MAX_HITS = _get_int_env("SIMON_ARCHIVE_FTS_MAX_HITS", 5)
ARCHIVE_AUTO_IMPORT = os.environ.get("SIMON_ARCHIVE_AUTO_IMPORT") == "1"
ARCHIVE_AUTO_IMPORT_JSON = _get_str_env("SIMON_ARCHIVE_JSON_PATH", "")
ARCHIVE_AUTO_IMPORT_SINCE_LAST = os.environ.get("SIMON_ARCHIVE_SINCE_LAST", "1") == "1"
ARCHIVE_AUTO_IMPORT_MAX_CODE_RATIO = _get_float_env("SIMON_ARCHIVE_MAX_CODE_RATIO", 0.8)

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

RLM_ENABLED = os.environ.get("SIMON_RLM_ENABLED", "1") == "1"
RLM_FALLBACK_ENABLED = os.environ.get("SIMON_RLM_FALLBACK_ENABLED", "1") == "1"
RLM_MAX_DEBT_RATIO = _get_float_env("SIMON_RLM_MAX_DEBT_RATIO", 2.5)
RLM_MIN_DEBT_FOR_CHECK = _get_float_env("SIMON_RLM_MIN_DEBT_FOR_CHECK", 1.2)
RLM_RECENT_WINDOW_TURNS = _get_int_env("SIMON_RLM_RECENT_WINDOW_TURNS", 6)
RLM_MIN_QUERY_LEN = _get_int_env("SIMON_RLM_MIN_QUERY_LEN", 20)
RLM_VECTOR_WEAK_DIST = _get_float_env("SIMON_RLM_VECTOR_WEAK_DIST", 0.55)
RLM_MIN_FTS_HITS = _get_int_env("SIMON_RLM_MIN_FTS_HITS", 1)
RLM_MAX_HOPS = _get_int_env("SIMON_RLM_MAX_HOPS", 3)
RLM_FTS_WEAK_SCORE = float(os.getenv("SIMON_RLM_FTS_WEAK_SCORE", "-1.0"))
RLM_DEEP_HISTORY_MAX_MSGS = _get_int_env("SIMON_RLM_DEEP_HISTORY_MAX_MSGS", 2)
RLM_DEEP_HISTORY_MAX_CHARS = _get_int_env("SIMON_RLM_DEEP_HISTORY_MAX_CHARS", 600)
RLM_TRACE = os.environ.get("SIMON_RLM_TRACE", "0") == "1"
RLM_STREAM = os.environ.get("SIMON_RLM_STREAM", "1") == "1"

LIVE_STT_ENABLED = os.environ.get("SIMON_LIVE_STT_ENABLED", "1") == "1"
LIVE_STT_WINDOW_S = _get_float_env("SIMON_LIVE_STT_WINDOW_S", 30.0)
LIVE_STT_STEP_S = _get_float_env("SIMON_LIVE_STT_STEP_S", 2.0)
LIVE_STT_KEEP_S = _get_float_env("SIMON_LIVE_STT_KEEP_S", 4.0)
LIVE_STT_MIN_WINDOW_S = _get_float_env("SIMON_LIVE_STT_MIN_WINDOW_S", 1.0)
LIVE_STT_OVERLAP_WORDS = _get_int_env("SIMON_LIVE_STT_OVERLAP_WORDS", 12)

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
    "CORPUS_DB_PATH",
    "WHISPER_MODEL_NAME",
    "TTS_VOICE",
    "DEFAULT_LLM_MODEL",
    "MAX_RECENT_MESSAGES",
    "ANCHOR_MESSAGES",
    "RAG_THRESHOLD",
    "LOCAL_WEAK_THRESHOLD",
    "ARCHIVE_STRONG_THRESHOLD",
    "ARCHIVE_EXPLICIT_THRESHOLD",
    "ARCHIVE_WEAK_MAX_AGE_DAYS",
    "ARCHIVE_FTS_MAX_SESSIONS",
    "ARCHIVE_FTS_MAX_HITS",
    "ARCHIVE_AUTO_IMPORT",
    "ARCHIVE_AUTO_IMPORT_JSON",
    "ARCHIVE_AUTO_IMPORT_SINCE_LAST",
    "ARCHIVE_AUTO_IMPORT_MAX_CODE_RATIO",
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
    "MEM_HOT_SESSION_LIMIT",
    "RAG_DEBUG_VERBOSE",
    "LLM_TIMEOUT_S",
    "AGENT_MAX_TURNS",
    "AGENT_TRIGGER_KEYWORDS",
    "MAX_TOOL_OUTPUT_CHARS",
    "RLM_ENABLED",
    "RLM_FALLBACK_ENABLED",
    "RLM_MAX_DEBT_RATIO",
    "RLM_MIN_DEBT_FOR_CHECK",
    "RLM_RECENT_WINDOW_TURNS",
    "RLM_MIN_QUERY_LEN",
    "RLM_VECTOR_WEAK_DIST",
    "RLM_MIN_FTS_HITS",
    "RLM_MAX_HOPS",
    "RLM_FTS_WEAK_SCORE",
    "RLM_DEEP_HISTORY_MAX_MSGS",
    "RLM_DEEP_HISTORY_MAX_CHARS",
    "RLM_TRACE",
    "RLM_STREAM",
    "LIVE_STT_ENABLED",
    "LIVE_STT_WINDOW_S",
    "LIVE_STT_STEP_S",
    "LIVE_STT_KEEP_S",
    "LIVE_STT_MIN_WINDOW_S",
    "LIVE_STT_OVERLAP_WORDS",
    "TEST_MODE",
    "SKIP_AUDIO_MODELS",
    "SKIP_VECTOR_MEMORY",
]
