import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import numpy as np
import asyncio
import time
import soundfile as sf
import io
import re
import threading
from collections import deque
import math
from pathlib import Path
import os
import sqlite3
import av
import traceback
import uuid
import json
import shutil
import tempfile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# --- CHROMA DB & EMBEDDINGS ---
import chromadb
from chromadb.utils import embedding_functions

# --- AI IMPORTS ---
from faster_whisper import WhisperModel
from openai import OpenAI
from kokoro_onnx import Kokoro

# --- CONFIG & DEBUG ---
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
DEBUG_MODE = True
QUIET_LOGS = os.environ.get("SIMON_QUIET_LOGS") == "1"
LM_STUDIO_URL = "http://localhost:1234/v1"
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

# --- MODEL CONFIG ---
WHISPER_MODEL_NAME = "distil-medium.en"  # High-ish Accuracy
TTS_VOICE = "am_fenrir"
DEFAULT_LLM_MODEL = "local-model"

# --- CONTEXT CONFIG ---
MAX_RECENT_MESSAGES = 40
ANCHOR_MESSAGES = 7
RAG_THRESHOLD = 0.35

# --- FTS5 CONFIG ---
FTS_MAX_HITS = 5
FTS_MIN_TOKEN_LEN = 3
FTS_PER_SESSION = True
FTS_DEDUP_MIN_LEN = 15

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

# --- AGENT CONFIG (Hardened) ---
AGENT_MAX_TURNS = 4          # Max loop iterations
AGENT_TRIGGER_KEYWORDS = {"research", "analyze", "deep dive", "проучи", "анализирай", "deep mode"}
MAX_TOOL_OUTPUT_CHARS = 12000 # Budget for tool returns

# --- GLOBAL MODELS ---
TEST_MODE = os.environ.get("SIMON_TEST_MODE") == "1"
SKIP_AUDIO_MODELS = os.environ.get("SIMON_SKIP_AUDIO_MODELS") == "1"
print("Loading AI Models... (TEST MODE)" if TEST_MODE else "Loading AI Models... (Stable Agentic Mode)")


class _DummySTT:
    def transcribe(self, *args, **kwargs):
        return [], None


class _DummyTTS:
    def create(self, text, voice=None, speed=1.0, lang=None):
        return np.zeros(160, dtype=np.float32), SAMPLE_RATE


class _DummyCompletions:
    def create(self, *args, **kwargs):
        msg = type("Msg", (), {"content": "", "role": "assistant", "tool_calls": None})()
        choice = type("Choice", (), {"message": msg, "delta": type("Delta", (), {"content": ""})()})()
        if kwargs.get("stream"):
            def _gen():
                yield type("Chunk", (), {"choices": [choice]})()
            return _gen()
        return type("Resp", (), {"choices": [choice], "usage": None})()


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class _DummyModels:
    def list(self):
        return type("ModelList", (), {"data": []})()


class _DummyClient:
    def __init__(self):
        self.chat = _DummyChat()
        self.models = _DummyModels()


if TEST_MODE:
    stt_model = _DummySTT()
    kokoro = _DummyTTS()
    client = _DummyClient()
else:
    if SKIP_AUDIO_MODELS:
        stt_model = _DummySTT()
        kokoro = _DummyTTS()
    else:
        # 1. STT
        stt_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

        # 2. TTS
        kokoro = Kokoro(str(MODELS_DIR / "kokoro-v0_19.onnx"), str(MODELS_DIR / "voices.bin"))

    # 3. LLM Client
    llm_timeout = LLM_TIMEOUT_S if LLM_TIMEOUT_S > 0 else None
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=llm_timeout)

current_model_lock = threading.Lock()
current_model = DEFAULT_LLM_MODEL

# --- METRICS ---
METRICS_HISTORY = deque(maxlen=200)
METRICS_LOCK = threading.Lock()

try:
    import tiktoken
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        if not text:
            return 0
        return len(_TOKEN_ENCODER.encode(text))
except Exception:
    def count_tokens(text):
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 3))


def init_metrics(mode, session_id):
    return {
        "id": str(uuid.uuid4()),
        "mode": mode,
        "session_id": session_id,
        "status": "in_progress",
        "start_time": time.time(),
        "end_time": 0,
        "audio_bytes": 0,
        "audio_decode": None,
        "stt": None,
        "rag": 0,
        "ctx": 0,
        "ttft": None,
        "llm_total": None,
        "tts_first": None,
        "tts_total": None,
        "total_latency": 0,
        "input_chars": 0,
        "output_chars": 0,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "overhead": None,
    }


def _metric_value(metrics, key):
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def estimate_tokens_from_text(text):
    if not text:
        return 0
    return count_tokens(text)


def estimate_tokens_from_messages(messages):
    total_tokens = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_tokens += count_tokens(content)
    return total_tokens


def finalize_metrics(metrics, status):
    if metrics.get("_recorded"):
        return
    if not metrics.get("end_time"):
        metrics["end_time"] = time.time()
    metrics["status"] = status
    metrics["total_latency"] = metrics["end_time"] - metrics["start_time"]
    steps_total = (
        _metric_value(metrics, "audio_decode")
        + _metric_value(metrics, "stt")
        + _metric_value(metrics, "rag")
        + _metric_value(metrics, "ctx")
        + _metric_value(metrics, "llm_total")
    )
    metrics["overhead"] = metrics["total_latency"] - steps_total
    generation_time = _metric_value(metrics, "llm_total") - _metric_value(metrics, "ttft")
    if generation_time > 0 and _metric_value(metrics, "output_tokens") > 0:
        metrics["tokens_per_second"] = _metric_value(metrics, "output_tokens") / generation_time
    payload = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with METRICS_LOCK:
        METRICS_HISTORY.append(payload)
    metrics["_recorded"] = True


# --- SQLITE HISTORY & FTS SETUP ---
db_lock = threading.Lock()

def init_db():
    needs_reset = False
    existing_conn = None

    if DB_PATH.exists():
        try:
            existing_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            existing_conn.execute("PRAGMA foreign_keys = ON;")
            existing_conn.execute("PRAGMA journal_mode=WAL;")
            existing_conn.execute("PRAGMA synchronous=NORMAL;")
            existing_conn.execute("PRAGMA busy_timeout=5000;")

            sess_cols = {r[1] for r in existing_conn.execute("PRAGMA table_info('sessions')").fetchall()}
            msg_cols = {r[1] for r in existing_conn.execute("PRAGMA table_info('messages')").fetchall()}
            required_sessions = {"id", "title", "summary", "tags", "model", "metadata", "created_at", "updated_at"}
            required_messages = {"id", "session_id", "role", "content", "tokens", "audio_path", "created_at"}
            if not required_sessions.issubset(sess_cols) or not required_messages.issubset(msg_cols):
                needs_reset = True
        except Exception:
            needs_reset = True

        if needs_reset:
            try:
                existing_conn.close()
            except Exception:
                pass
            DB_PATH.unlink(missing_ok=True)
        else:
            return existing_conn

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            model TEXT,
            metadata TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            updated_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            audio_path TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")

    conn.commit()
    return conn

db_conn = init_db()

def _ensure_fts5(conn: sqlite3.Connection):
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __fts5_test")
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] SQLite FTS5 unavailable: {e}")
        return False

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            session_id UNINDEXED,
            role UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    conn.execute("DROP TRIGGER IF EXISTS messages_ai")
    conn.execute("DROP TRIGGER IF EXISTS messages_ad")
    conn.execute("DROP TRIGGER IF EXISTS messages_au")
    conn.execute("""
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, session_id, role)
            VALUES (new.id, new.content, new.session_id, new.role);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER messages_au AFTER UPDATE OF content, session_id, role ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content, session_id, role)
            VALUES (new.id, new.content, new.session_id, new.role);
        END;
    """)

    try:
        cur = conn.execute("SELECT COUNT(1) FROM messages_fts")
        existing = cur.fetchone()[0]
        if existing == 0:
            conn.execute("""
                INSERT INTO messages_fts(rowid, content, session_id, role)
                SELECT id, content, session_id, role FROM messages
            """)
    except Exception:
        pass

    conn.commit()
    return True

try:
    _ensure_fts5(db_conn)
except Exception as _e:
    if DEBUG_MODE:
        print(f"[WARN] FTS5 setup failed: {_e}")


def init_mem_db():
    conn = sqlite3.connect(
        "file:memdb?mode=memory&cache=shared",
        uri=True,
        check_same_thread=False
    )
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA journal_mode=MEMORY;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            model TEXT,
            metadata TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            updated_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            audio_path TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")

    conn.commit()
    try:
        _ensure_fts5(conn)
    except Exception as _e:
        if DEBUG_MODE:
            print(f"[WARN] In-memory FTS5 setup failed: {_e}")
    return conn


def seed_mem_db_from_disk(disk_conn, mem_conn, limit=MEM_SEED_LIMIT, batch_size=1000, cutoff_ts=None):
    if limit <= 0:
        return
    ro_conn = None
    try:
        ro_conn = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro",
            uri=True,
            check_same_thread=False
        )
        ro_conn.execute("PRAGMA busy_timeout=5000;")
        params = []
        sql = """
            SELECT session_id, role, content, tokens, audio_path, created_at
            FROM messages
        """
        if cutoff_ts is not None:
            sql += " WHERE created_at <= ?"
            params.append(float(cutoff_ts))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        rows = ro_conn.execute(sql, tuple(params)).fetchall()
        if not rows:
            return
        rows = list(reversed(rows))
        if DEBUG_MODE:
            print(f"[MEM] Seeding in-memory FTS with {len(rows)} rows...")

        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            with db_lock:
                mem_conn.executemany(
                    "INSERT INTO messages(session_id, role, content, tokens, audio_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    chunk
                )
                mem_conn.commit()
        if DEBUG_MODE:
            print("[MEM] In-memory FTS seed complete.")
    except Exception as _e:
        if DEBUG_MODE:
            print(f"[WARN] In-memory FTS seed failed: {_e}")
    finally:
        try:
            if ro_conn:
                ro_conn.close()
        except Exception:
            pass


def prune_mem_db(mem_conn, max_rows=MEM_MAX_ROWS, interval_s=MEM_PRUNE_INTERVAL_S, stop_event=None):
    if max_rows <= 0:
        return
    if TEST_MODE:
        interval_s = max(1, interval_s)
    else:
        interval_s = max(5, interval_s)
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        time.sleep(interval_s)
        if stop_event is not None and stop_event.is_set():
            break
        try:
            with db_lock:
                cur = mem_conn.execute("SELECT COUNT(1) FROM messages")
                count = cur.fetchone()[0] or 0
                if count <= max_rows:
                    continue
                cutoff_row = mem_conn.execute(
                    "SELECT id FROM messages ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (max_rows - 1,)
                ).fetchone()
                if not cutoff_row:
                    continue
                cutoff_id = cutoff_row[0]
                mem_conn.execute("DELETE FROM messages WHERE id < ?", (cutoff_id,))
                mem_conn.commit()
            if DEBUG_MODE:
                print(f"[MEM] Pruned in-memory FTS to {max_rows} rows.")
        except Exception as _e:
            if DEBUG_MODE:
                print(f"[WARN] In-memory FTS prune failed: {_e}")


mem_conn = init_mem_db()
_mem_threads_started = False


def start_mem_threads(stop_event=None):
    global _mem_threads_started
    if _mem_threads_started:
        return None
    _mem_threads_started = True
    _mem_seed_cutoff = time.time()
    seed_thread = threading.Thread(
        target=seed_mem_db_from_disk,
        args=(db_conn, mem_conn, MEM_SEED_LIMIT, 1000, _mem_seed_cutoff),
        daemon=True
    )
    prune_thread = threading.Thread(
        target=prune_mem_db,
        args=(mem_conn, MEM_MAX_ROWS, MEM_PRUNE_INTERVAL_S, stop_event),
        daemon=True
    )
    seed_thread.start()
    prune_thread.start()
    return seed_thread, prune_thread


if not TEST_MODE:
    start_mem_threads()


def create_session(title=None, model=None, summary=None, tags=None):
    ts = time.time()
    with db_lock:
        cur = db_conn.execute(
            "INSERT INTO sessions(title, summary, tags, model, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, summary or "", tags or "", model, None, ts, ts)
        )
        db_conn.commit()
        return cur.lastrowid


def list_sessions(limit=50):
    with db_lock:
        cur = db_conn.execute(
            """
            SELECT id, COALESCE(title, '') as title, COALESCE(summary, '') as summary,
                   COALESCE(tags, '') as tags, model, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cur.fetchall()
        return [{
            "id": r[0],
            "title": r[1],
            "summary": r[2],
            "tags": r[3],
            "model": r[4],
            "created_at": r[5],
            "updated_at": r[6],
        } for r in rows]


def get_session_meta(session_id):
    with db_lock:
        cur = db_conn.execute(
            """
            SELECT id, COALESCE(title,''), COALESCE(summary,''), COALESCE(tags,''), model, created_at, updated_at
            FROM sessions WHERE id=?
            """,
            (session_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "summary": row[2],
        "tags": row[3],
        "model": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def session_exists(session_id):
    with db_lock:
        cur = db_conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,))
        return cur.fetchone() is not None


def touch_session(session_id):
    with db_lock:
        db_conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), session_id))
        db_conn.commit()


def load_session_messages(session_id):
    with db_lock:
        cur = db_conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        )
        return [{"role": r[0], "content": r[1]} for r in cur.fetchall()]


def get_session_transcript(session_id, max_chars=6000):
    """Retrieves plain text transcript for deep analysis tools."""
    try:
        msgs = load_session_messages(session_id)
        if not msgs:
            return ""
        lines = []
        total_len = 0
        for m in msgs:
            line = f"{m['role'].upper()}: {m['content']}"
            if total_len + len(line) > max_chars:
                break
            lines.append(line)
            total_len += len(line)
        return "\n".join(lines)
    except Exception:
        return ""


def save_interaction(session_id, user_text, ai_text):
    ts = time.time()
    with db_lock:
        db_conn.execute(
            "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", user_text, 0, ts)
        )
        db_conn.execute(
            "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", ai_text, 0, ts)
        )
        db_conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, session_id))
        db_conn.commit()
        try:
            mem_conn.execute(
                "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, "user", user_text, 0, ts)
            )
            mem_conn.execute(
                "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, "assistant", ai_text, 0, ts)
            )
            mem_conn.commit()
        except Exception as _e:
            if DEBUG_MODE:
                print(f"[WARN] In-memory DB insert failed: {_e}")


def set_session_title(session_id, title):
    with db_lock:
        db_conn.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, time.time(), session_id))
        db_conn.commit()


def get_session_window(session_id, anchor=ANCHOR_MESSAGES, recent=MAX_RECENT_MESSAGES):
    meta = get_session_meta(session_id)
    if not meta:
        return None

    anchor = max(0, anchor)
    recent = max(0, recent)

    with db_lock:
        anchor_rows = db_conn.execute(
            "SELECT id, role, content, created_at, tokens FROM messages WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
            (session_id, anchor)
        ).fetchall()
        recent_rows = db_conn.execute(
            "SELECT id, role, content, created_at, tokens FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, recent)
        ).fetchall()

    anchor_ids = {r[0] for r in anchor_rows}
    recent_rows = [r for r in reversed(recent_rows) if r[0] not in anchor_ids]

    def map_row(row):
        return {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": row[3],
            "tokens": row[4] or 0,
        }

    return {
        "session": meta,
        "anchors": [map_row(r) for r in anchor_rows],
        "recents": [map_row(r) for r in recent_rows],
    }


# --- FTS5 SEARCH (With AND->OR Fallback) ---
_word_re = re.compile(r"\w+", re.UNICODE)


def _fts_sanitize_query(text: str, join_op=" ") -> str:
    if not text:
        return ""
    toks = [t.lower() for t in _word_re.findall(text)]
    toks = [t for t in toks if len(t) >= FTS_MIN_TOKEN_LEN]
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    # Default is AND (space). Optional OR.
    return join_op.join(out[:12])


def fts_search_messages(
    query_text: str,
    session_id: int | None = None,
    limit: int = FTS_MAX_HITS,
    cutoff_ts: float | None = None,
    conn: sqlite3.Connection | None = None,
    lock=None,
):
    # Strategy 1: AND query (strict)
    q = _fts_sanitize_query(query_text, " ")
    if not q:
        return []

    def run_query(match_query):
        sql = """
            SELECT m.role, m.content, m.created_at, m.session_id, bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
        """
        params = [match_query]
        if FTS_PER_SESSION and session_id is not None:
            sql += " AND m.session_id = ?"
            params.append(int(session_id))
        if cutoff_ts is not None:
            sql += " AND m.created_at >= ?"
            params.append(float(cutoff_ts))
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(int(max(1, min(limit, 50))))
        
        try:
            target_conn = conn or db_conn
            target_lock = lock or db_lock
            with target_lock:
                return target_conn.execute(sql, tuple(params)).fetchall()
        except Exception:
            return []

    rows = run_query(q)
    
    # Strategy 2: OR query (fallback)
    if not rows:
        q_or = _fts_sanitize_query(query_text, " OR ")
        if q_or and q_or != q:
            rows = run_query(q_or)

    return [{
        "role": r[0],
        "content": r[1],
        "created_at": r[2],
        "session_id": r[3],
        "score": float(r[4]) if r[4] is not None else None
    } for r in rows]


# --- MEMORY SYSTEM (Hardened) ---
class MemoryManager:
    def __init__(self):
        print("Initializing Vector Memory (ChromaDB)...")
        self.lock = threading.Lock() # Fix 1: Thread safety
        self.chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "simon_db"))
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="simon_memories",
            embedding_function=self.emb_fn
        )
        print(" Memory Loaded.")

    def search(self, query_text, n_results=3, days_filter=None, session_filter=None):
        # Build Where Clause
        where_clause = {}
        
        if session_filter is not None:
            where_clause["session_id"] = int(session_filter)
        elif days_filter is not None:
            cutoff_ts = time.time() - (days_filter * 24 * 3600)
            where_clause["timestamp"] = {"$gte": cutoff_ts}
            
        # If empty, Chroma wants None, not {}
        if not where_clause:
            where_clause = None

        with self.lock:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'distances', 'metadatas']
            )
        
        docs = results['documents'][0] if results['documents'] else []
        dists = results['distances'][0] if results['distances'] else []
        metas = results['metadatas'][0] if results['metadatas'] else []

        # Automatic Deep Recall logic (only if we didn't force a session filter)
        if session_filter is None and days_filter is not None:
            need_deep = False
            if len(docs) < n_results: need_deep = True
            elif dists and dists[0] > 0.45: need_deep = True
            
            if need_deep:
                if DEBUG_MODE: print("   [MEMORY] Triggering Deep Recall (Global)...")
                with self.lock:
                    g_res = self.collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        include=['documents', 'distances', 'metadatas']
                    )
                g_docs = g_res['documents'][0] if g_res['documents'] else []
                if len(g_docs) > len(docs):
                    docs, dists, metas = g_docs, g_res['distances'][0], g_res['metadatas'][0]

        return docs, dists, metas

    def save(self, user_text, ai_text, session_id):
        # Fix C: Global Dedup (no filters, atomic check + add)
        with self.lock:
            try:
                res = self.collection.query(
                    query_texts=[user_text],
                    n_results=1,
                    include=['distances']
                )
                dists = res['distances'][0] if res.get('distances') else []
            except Exception:
                dists = []

            if dists and dists[0] < 0.2:
                if DEBUG_MODE:
                    print(f" Memory duplication detected (Dist: {dists[0]:.4f}). Skipping save.")
                return

            memory_text = f"User: {user_text} | AI: {ai_text}"
            metadata = {
                "role": "conversation",
                "timestamp": time.time(),
                "session_id": int(session_id) if session_id else 0
            }

            self.collection.add(
                documents=[memory_text],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )


SKIP_VECTOR_MEMORY = os.environ.get("SIMON_SKIP_VECTOR_MEMORY") == "1"

if TEST_MODE or SKIP_VECTOR_MEMORY:
    class _DummyMemory:
        def search(self, *args, **kwargs):
            return [], [], []

        def save(self, *args, **kwargs):
            return None

    memory = _DummyMemory()
else:
    memory = MemoryManager()
print("All Models Loaded.")


# --- TOOLS DEFINITION (OPENAI SCHEMA) ---
SIMON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search long-term memory (vectors) and chat history (FTS). Use to find facts, past conversations, or project details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The search query"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "recent", "session"],
                        "description": "Scope: 'global' (all time), 'recent' (60 days), 'session' (current chat)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_deep_context",
            "description": "Reads a past session transcript. Use ONLY for complex analysis of a specific known session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The ID of the session to analyze."
                    },
                    "instruction": {
                        "type": "string",
                        "description": "What to extract/summarize."
                    }
                },
                "required": ["session_id", "instruction"]
            }
        }
    }
]

# --- TOOL IMPLEMENTATION (With Limits) ---
def tool_search_memory(query: str, scope: str = "recent", session_id: int = None):
    # Fix B: Respect session scope in Vector Search
    days = 60 if scope == "recent" else None
    sess_filter = session_id if scope == "session" else None
    cutoff_ts = time.time() - (60 * 24 * 3600) if scope == "recent" else None
    
    # 1. Vector Search
    docs, dists, metas = memory.search(query, n_results=3, days_filter=days, session_filter=sess_filter)
    
    # 2. FTS Search
    fts_res = []
    if scope in ["recent", "session"]:
        target_sess = session_id if scope == "session" else None
        fts_res = fts_search_messages(
            query,
            session_id=target_sess,
            limit=3,
            cutoff_ts=cutoff_ts,
            conn=mem_conn,
            lock=db_lock,
        )

    results = []
    # Format output (capped)
    for doc, meta in zip(docs, metas):
        sess = meta.get("session_id", "?")
        ts = meta.get("timestamp", 0)
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        results.append(f"[VECTOR] (Sess:{sess}, {date_str}): {doc[:150]}...")

    for item in fts_res:
        role = item['role']
        content = item['content'].replace("\n", " ")
        sess = item.get("session_id", "?")
        results.append(f"[EXACT] (Sess:{sess}, {role}): {content[:150]}...")

    if not results:
        return "No relevant memories found."
    
    # Fix D: Total size budget enforcement (simple slice)
    full_resp = "\n".join(results)
    if len(full_resp) > MAX_TOOL_OUTPUT_CHARS:
        return full_resp[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"
    return full_resp

def tool_analyze_deep(session_id: int, instruction: str):
    # Fix D: Strict Cap
    transcript = get_session_transcript(session_id, max_chars=6000)
    if not transcript:
        return "Session not found or empty."

    try:
        # Recursive call (separate context)
        sub_response = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": "You are a data extractor. Analyze the transcript based on the instruction. Be brief."},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nINSTRUCTION: {instruction}"}
            ],
            temperature=0.0,
            max_tokens=400
        )
        return f"ANALYSIS RESULT: {sub_response.choices[0].message.content}"
    except Exception as e:
        return f"Analysis failed: {str(e)}"


def log_console(msg, type="INFO"):
    if DEBUG_MODE and not QUIET_LOGS:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{type}] {msg}")


def get_current_model():
    with current_model_lock:
        return current_model


def set_current_model(name: str):
    global current_model
    clean_name = name.strip()
    if not clean_name:
        return current_model
    with current_model_lock:
        current_model = clean_name
        return current_model


def warm_model(name: str):
    try:
        client.chat.completions.create(
            model=name,
            messages=[{"role": "system", "content": "ping"}],
            max_tokens=1,
            temperature=0
        )
        return True
    except Exception as e:
        log_console(f"Model warmup failed for '{name}': {e}", "ERR")
        return False


def print_perf_report(metrics):
    if QUIET_LOGS:
        return
    total_pipeline = metrics['end_time'] - metrics['start_time']
    C_GREEN = '\033[92m'
    C_END = '\033[0m'
    print(f"\n{C_GREEN}--- RAG PERF REPORT ---{C_END}")
    print(
        f"   Decode: {_metric_value(metrics, 'audio_decode'):.3f}s | "
        f"STT: {_metric_value(metrics, 'stt'):.3f}s | "
        f"MEMORY: {_metric_value(metrics, 'rag'):.3f}s | "
        f"CTX: {_metric_value(metrics, 'ctx'):.3f}s"
    )
    print(
        f"   LLM TTFT: {_metric_value(metrics, 'ttft'):.3f}s | "
        f"LLM Total: {_metric_value(metrics, 'llm_total'):.3f}s"
    )
    print(
        f"   TTS First: {_metric_value(metrics, 'tts_first'):.3f}s | "
        f"TTS Total: {_metric_value(metrics, 'tts_total'):.3f}s"
    )
    print(
        f"   Tokens In: {_metric_value(metrics, 'input_tokens'):.0f} | "
        f"Tokens Out: {_metric_value(metrics, 'output_tokens'):.0f} | "
        f"TPS: {_metric_value(metrics, 'tokens_per_second'):.2f} | "
        f"Overhead: {_metric_value(metrics, 'overhead'):.3f}s"
    )
    print(f"   TOTAL: {total_pipeline:.3f}s")
    print(f"------------------------------------------\n")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelPayload(BaseModel):
    name: str


def _ensure_frames_list(resampled):
    if resampled is None:
        return []
    if isinstance(resampled, list):
        return resampled
    return [resampled]


def convert_webm_to_numpy(webm_bytes):
    try:
        bio = io.BytesIO(webm_bytes)
        container = av.open(bio, format="webm")
        audio_stream = None
        for s in container.streams:
            if s.type == "audio":
                audio_stream = s
                break
        if audio_stream is None:
            return None

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio_stream):
            out_frames = _ensure_frames_list(resampler.resample(frame))
            for of in out_frames:
                arr = of.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr)
        for of in _ensure_frames_list(resampler.resample(None)):
            arr = of.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            chunks.append(arr)
        if not chunks:
            return None
        pcm = np.concatenate(chunks).astype(np.int16)
        return pcm.astype(np.float32) / 32768.0
    except Exception:
        return None


def numpy_to_wav_bytes(audio_np, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


def _safe_usage_dict(usage):
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = getattr(usage, "prompt_tokens", None) if not isinstance(usage, dict) else usage.get("prompt_tokens")
    completion_tokens = getattr(usage, "completion_tokens", None) if not isinstance(usage, dict) else usage.get("completion_tokens")
    total_tokens = getattr(usage, "total_tokens", None) if not isinstance(usage, dict) else usage.get("total_tokens")
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _msg_to_dict(msg):
    if msg is None:
        return {}
    if isinstance(msg, dict):
        out = {"role": msg.get("role"), "content": msg.get("content")}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            out["tool_calls"] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    out["tool_calls"].append({
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                        "function": {
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        },
                    })
                else:
                    fn_obj = getattr(tc, "function", None)
                    out["tool_calls"].append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(fn_obj, "name", None) if fn_obj else None,
                            "arguments": getattr(fn_obj, "arguments", None) if fn_obj else None,
                        },
                    })
        return out

    out = {"role": getattr(msg, "role", None), "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = []
        for tc in tool_calls:
            fn_obj = getattr(tc, "function", None)
            out["tool_calls"].append({
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None),
                "function": {
                    "name": getattr(fn_obj, "name", None) if fn_obj else None,
                    "arguments": getattr(fn_obj, "arguments", None) if fn_obj else None,
                },
            })
    return out


def _safe_args(raw_args):
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except Exception:
        return {"_raw": str(raw_args)}


def _serialize_chat_completion(model_name: str, content: str, usage=None):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": _safe_usage_dict(usage),
    }


def _convert_to_mp3_bytes(samples: np.ndarray, sample_rate: int):
    """Convert float32 mono samples to MP3 bytes using pydub/ffmpeg."""
    audio_int16 = np.clip(samples, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    mp3_io = io.BytesIO()
    segment.export(mp3_io, format="mp3", bitrate="128k")
    mp3_io.seek(0)
    return mp3_io


def _trim_old_esp_audio():
    try:
        files = sorted(ESP_AUDIO_DIR.glob("esp_audio_*.wav"), key=lambda p: p.stat().st_mtime)
        while len(files) > ESP_AUDIO_MAX_FILES:
            files[0].unlink(missing_ok=True)
            files = files[1:]
    except Exception:
        pass


def _analyze_audio(audio_np: np.ndarray, sample_rate: int):
    if audio_np is None or sample_rate <= 0:
        return None
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    audio_np = audio_np.astype(np.float32)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
    clip_threshold = 0.98
    clipping = float(np.mean(np.abs(audio_np) >= clip_threshold)) if audio_np.size else 0.0
    dc_offset = float(np.mean(audio_np)) if audio_np.size else 0.0
    duration = float(len(audio_np)) / float(sample_rate)
    return {
        "sample_rate": int(sample_rate),
        "channels": 1,
        "duration_s": round(duration, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "clipping_pct": round(clipping * 100.0, 2),
        "dc_offset": round(dc_offset, 5),
    }


def _sanitize_client_messages(messages):
    cleaned = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            content_text = str(content or "").strip().lower()
            if content_text in {"", "user", "system", "assistant"}:
                continue
        cleaned.append({"role": role, "content": content})
    return cleaned


# --- OPENAI-COMPATIBLE REST ENDPOINTS (ESP32) ---
@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str | None = Form(None),
    prompt: str | None = Form(None),
):
    tmp_path = None
    try:
        audio_bytes = await file.read()
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        stats = None
        try:
            audio_np, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
            stats = _analyze_audio(audio_np, sr)
        except Exception:
            stats = None

        def run_transcribe():
            segs, _ = stt_model.transcribe(
                str(tmp_path),
                beam_size=5,
                vad_filter=False,
                language=language,
            )
            return " ".join([s.text for s in segs]).strip()

        text = await asyncio.to_thread(run_transcribe)
        if stats:
            log_console(
                f"ESP audio stats: sr={stats['sample_rate']}Hz dur={stats['duration_s']}s "
                f"peak={stats['peak']} rms={stats['rms']} clip={stats['clipping_pct']}% dc={stats['dc_offset']}",
                "AUDIO",
            )
        if SAVE_ESP_AUDIO:
            try:
                out_path = ESP_AUDIO_DIR / f"esp_audio_{uuid.uuid4().hex}.wav"
                shutil.copyfile(tmp_path, out_path)
                _trim_old_esp_audio()
            except Exception:
                pass
        return {"text": text}
    except Exception as e:
        log_console(f"Transcription failed: {e}", "ERR")
        return JSONResponse(content={"error": "transcription_failed", "message": str(e)}, status_code=500)
    finally:
        try:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/chat/completions")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        messages = _sanitize_client_messages(data.get("messages", []))
        model_name = get_current_model()
        temperature = data.get("temperature", 0.7)

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        ai_text = resp.choices[0].message.content
        return _serialize_chat_completion(model_name, ai_text, getattr(resp, "usage", None))
    except Exception as e:
        log_console(f"Chat completion failed: {e}", "ERR")
        return JSONResponse(content={"error": "chat_failed", "message": str(e)}, status_code=500)


@app.post("/v1/audio/speech")
async def speech_endpoint(request: Request):
    try:
        data = await request.json()
        text = data.get("input", "")
        if not text:
            return JSONResponse(content={"error": "empty_input", "message": "No input text provided."}, status_code=400)

        samples, sr = await asyncio.to_thread(kokoro.create, text, voice=TTS_VOICE, speed=1.0, lang="en-us")

        try:
            mp3_io = await asyncio.to_thread(_convert_to_mp3_bytes, samples, sr)
            return StreamingResponse(mp3_io, media_type="audio/mpeg")
        except Exception as mp3_err:
            log_console(f"MP3 conversion failed, falling back to WAV: {mp3_err}", "ERR")
            wav_bytes = await asyncio.to_thread(numpy_to_wav_bytes, samples, sr)
            return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
    except Exception as e:
        log_console(f"TTS failed: {e}", "ERR")
        return JSONResponse(content={"error": "tts_failed", "message": str(e)}, status_code=500)

# --- HELPER FOR NORMALIZATION ---
def _norm(text):
    return " ".join((text or "").lower().split())

def build_rag_context(user_text, history, memory_manager, metrics, session_id: int | None = None):
    t_start = time.time()

    def window_messages(msgs, first=10, last=10):
        if len(msgs) <= first + last:
            return msgs
        return msgs[:first] + msgs[-last:]

    history_window = window_messages(history)

    # 1) Vector memories (UPDATED: Unpack metadatas + Two-Stage retrieval)
    # Search last 60 days first
    retrieved_docs, distances, metadatas = memory_manager.search(user_text, n_results=3, days_filter=60)

    # 2) SQLite FTS5 keyword recall
    fts_hits = []
    try:
        fts_hits = fts_search_messages(
            user_text,
            session_id=session_id,
            limit=FTS_MAX_HITS,
            conn=mem_conn,
            lock=db_lock,
        )
    except Exception:
        fts_hits = []

    metrics['rag'] = time.time() - t_start

    rag_payload = []
    valid_memories = []
    vector_contents_norm = [] 

    # Process Vector Results
    if retrieved_docs:
        if not QUIET_LOGS:
            print(f"\n \033[96m[VECTOR SEARCH] Query: '{user_text}'\033[0m")
            print(f"   \033[90m--------------------------------------------------\033[0m")
        for i, (doc, dist, meta) in enumerate(zip(retrieved_docs, distances, metadatas)):
            score_color = '\033[92m' if dist < 0.8 else '\033[93m'
            
            ts = meta.get("timestamp", 0) if meta else 0
            age_days = (time.time() - ts) / (24 * 3600) if ts > 0 else 0
            src_session = meta.get("session_id", "?") if meta else "?"
            
            debug_str = f"Sess:{src_session} | {age_days:.1f}d ago"
            if not QUIET_LOGS:
                print(f"    #{i+1} [{score_color}Dist: {dist:.4f}\033[0m] [{debug_str}] \033[3m\"{doc[:60]}...\"\033[0m")
            
            if dist < RAG_THRESHOLD:
                valid_memories.append(doc)
                vector_contents_norm.append(_norm(doc)) 
                rag_payload.append({
                    "doc": doc[:40] + "...", 
                    "dist": round(float(dist), 3),
                    "age_days": round(age_days, 1),
                    "session": src_session
                })
        if not QUIET_LOGS:
            print(f"   \033[90m--------------------------------------------------\033[0m\n")
    else:
        if not QUIET_LOGS:
            print(f"\n \033[90m[VECTOR SEARCH] No memories found.\033[0m\n")

    anchor = history_window[:ANCHOR_MESSAGES]
    remaining = history_window[ANCHOR_MESSAGES:]
    recent = remaining[-MAX_RECENT_MESSAGES:]

    rag_injection = []

    if valid_memories:
        memory_block = "\n".join([f"- {m}" for m in valid_memories])
        rag_injection.append({
            "role": "system",
            "content": f"Relevant past memories:\n{memory_block}\n(Use these to personalize, but prioritize current context.)"
        })

    # FTS Injection
    final_fts_lines = []
    dedup_count = 0
    
    if fts_hits:
        for h in fts_hits:
            role = h.get("role", "?")
            raw_content = h.get("content") or ""
            content_norm = _norm(raw_content)
            
            is_duplicate = False
            if len(content_norm) >= FTS_DEDUP_MIN_LEN:
                for v_norm in vector_contents_norm:
                    if content_norm in v_norm:
                        is_duplicate = True
                        break
            
            if is_duplicate:
                dedup_count += 1
                continue

            clean_content = raw_content.strip().replace("\n", " ")
            if len(clean_content) > 240:
                clean_content = clean_content[:240] + "..."
            
            score = h.get("score")
            if score is None:
                final_fts_lines.append(f"- [{role}] {clean_content}")
            else:
                final_fts_lines.append(f"- [{role}] {clean_content} (bm25={score:.2f})")

        if final_fts_lines:
            scope_desc = "this session" if FTS_PER_SESSION and session_id is not None else "full chat history"
            fts_block = "\n".join(final_fts_lines)
            rag_injection.append({
                "role": "system",
                "content": (
                    f"Keyword recall (SQLite FTS5 over {scope_desc}). These are verbatim excerpts that matched your query:\n"
                    f"{fts_block}\n"
                    "(Treat as raw evidence; reconcile with the current turn.)"
                )
            })

        if RAG_DEBUG_VERBOSE and final_fts_lines:
            rag_payload.append({"fts_preview": final_fts_lines[:FTS_MAX_HITS]})

        try:
            rag_payload.append({"fts_hits": len(fts_hits), "deduplicated": dedup_count})
        except Exception:
            pass

    return anchor + rag_injection + recent, rag_payload


# --- STREAMING LOGIC (HYBRID AGENT - PHASED EXECUTION) ---
async def process_and_stream_response(user_text, websocket, history, metrics, stop_event, session_id, generate_audio=True):
    t_ctx_start = time.time()
    
    # --- DECISION: FAST PATH vs DEEP PATH ---
    is_deep_mode = any(k in user_text.lower() for k in AGENT_TRIGGER_KEYWORDS)
    
    current_messages = []
    rag_payload = []

    if is_deep_mode:
        log_console("ACTIVATING AGENTIC LOOP", "AGENT")
        await websocket.send_text("SYS:THINKING: Entering Deep Mode...")
        # Prepare Agent Prompt
        current_messages = [
            {"role": "system", "content": "You are Simon (Deep Mode). Use your tools to verify facts from memory/history before answering. Do not hallucinate."},
            *history[-10:], 
            {"role": "user", "content": user_text}
        ]
        tools_list = SIMON_TOOLS
    else:
        # Fast Path (Old RAG)
        context_msgs, rag_payload = build_rag_context(user_text, history, memory, metrics, session_id)
        current_messages = context_msgs + [{"role": "user", "content": user_text}]
        tools_list = None

    metrics['ctx'] = time.time() - t_ctx_start
    metrics['input_chars'] = len(user_text)
    if metrics.get("input_tokens") is None:
        metrics["input_tokens"] = estimate_tokens_from_messages(current_messages)

    if rag_payload:
        await websocket.send_text(f"RAG:{json.dumps(rag_payload)}")

    q = asyncio.Queue(maxsize=64)
    response_holder = {"text": ""}
    sentence_endings = re.compile(r'[.!?]+')

    # --- TTS CONSUMER (UNCHANGED) ---
    async def tts_consumer():
        first_audio_generated = False
        while True:
            if stop_event.is_set(): break
            try:
                item = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError: continue
            if item is None: break
            clean_text = item
            if not clean_text: continue

            wav_bytes = None
            if generate_audio:
                if metrics.get("tts_total") is None: metrics["tts_total"] = 0
                tts_start = time.time()
                samples, sr = await asyncio.to_thread(kokoro.create, clean_text, voice=TTS_VOICE, speed=1.0, lang="en-us")
                if stop_event.is_set(): break
                wav_bytes = await asyncio.to_thread(numpy_to_wav_bytes, samples, sr)
                metrics["tts_total"] += time.time() - tts_start

                if not first_audio_generated:
                    metrics['tts_first'] = time.time() - metrics['start_time']
                    first_audio_generated = True

            if not stop_event.is_set():
                await websocket.send_text(f"LOG:AI: {clean_text}")
                if generate_audio and wav_bytes:
                    await websocket.send_bytes(wav_bytes)

    # --- LLM PRODUCER (Fix A: SEPARATED PHASES) ---
    def llm_producer_threadsafe(loop, stop_evt):
        try:
            model_name = get_current_model()
            log_console(f"Using model: {model_name} | Deep: {is_deep_mode}", "AI")
            llm_start = time.time()
            metrics["_llm_start"] = llm_start
            
            final_text_buffer = ""
            tool_chars_used = 0

            def enqueue_text(text):
                def _do_put():
                    try:
                        q.put_nowait(text)
                    except asyncio.QueueFull:
                        if DEBUG_MODE:
                            log_console("TTS queue full; dropping chunk", "WARN")
                loop.call_soon_threadsafe(_do_put)

            # --- PHASE 1: THINKING LOOP (No streaming, Tool usage) ---
            if is_deep_mode:
                turn_count = 0
                while turn_count < AGENT_MAX_TURNS and not stop_evt.is_set():
                    turn_count += 1

                    try:
                        # Non-streaming call for logic
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=current_messages,
                            temperature=0.7,
                            stream=False,
                            tools=tools_list,
                            tool_choice="auto"
                        )
                    except Exception as e:
                        print(f"Agent Loop Error: {e}")
                        break

                    msg_dict = _msg_to_dict(response.choices[0].message)
                    tool_calls = msg_dict.get("tool_calls") or []
                    if tool_calls:
                        current_messages.append(msg_dict)
                        first_tool = tool_calls[0] if tool_calls else {}
                        first_fn = (first_tool.get("function") or {}).get("name") or first_tool.get("name")
                        if first_fn:
                            log_console(f"Tool Call: {first_fn}", "AGENT")
                        asyncio.run_coroutine_threadsafe(websocket.send_text("SYS:THINKING: Consulting memory..."), loop)

                        for tool_call in tool_calls:
                            fn = tool_call.get("function") or {}
                            fn_name = fn.get("name") or tool_call.get("name")
                            args = _safe_args(fn.get("arguments"))
                            result = "Error: Unknown tool"

                            if fn_name == "search_memory":
                                query = args.get("query", "")
                                scope = args.get("scope", "recent")
                                result = tool_search_memory(query, scope, session_id)
                            elif fn_name == "analyze_deep_context":
                                asyncio.run_coroutine_threadsafe(websocket.send_text("SYS:THINKING: Deep reading transcript..."), loop)
                                target_session = args.get("session_id")
                                instruction = args.get("instruction", "")
                                if target_session is None or not instruction:
                                    result = "Error: analyze_deep_context requires session_id and instruction."
                                else:
                                    result = tool_analyze_deep(target_session, instruction)

                            result_str = str(result)
                            remain = MAX_TOOL_OUTPUT_CHARS - tool_chars_used
                            if remain <= 0:
                                result_str = "[TOOL_BUDGET_EXCEEDED]"
                            else:
                                result_str = result_str[:remain]
                            tool_chars_used += len(result_str)

                            tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                            tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
                            current_messages.append(tool_msg)
                    else:
                        # No tools; stop thinking and let Phase 2 produce the final answer.
                        break

            # --- PHASE 2: SPEAKING (Streaming Response) ---
            # Now we stream the final answer based on accumulated history
            if not stop_evt.is_set():
                try:
                    final_messages = current_messages
                    if is_deep_mode:
                        final_messages = current_messages + [{
                            "role": "system",
                            "content": "Now produce the final answer for the user. Do not call tools."
                        }]

                    stream_start = time.time()
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=final_messages,
                        temperature=0.7,
                        stream=True
                    )

                    current_sentence = ""
                    for chunk in stream:
                        if stop_evt.is_set():
                            break

                        choices = getattr(chunk, "choices", None)
                        if choices is None and isinstance(chunk, dict):
                            choices = chunk.get("choices")
                        if not choices:
                            continue
                        choice0 = choices[0]
                        delta = getattr(choice0, "delta", None)
                        if delta is None and isinstance(choice0, dict):
                            delta = choice0.get("delta")
                        if not delta:
                            continue
                        token = getattr(delta, "content", None)
                        if token is None and isinstance(delta, dict):
                            token = delta.get("content")
                        if not token:
                            continue
                        if metrics.get("ttft") is None:
                            metrics["ttft"] = time.time() - stream_start

                        final_text_buffer += token
                        current_sentence += token

                        if sentence_endings.search(current_sentence[-2:]) and len(current_sentence.strip()) > 5:
                            raw_t = current_sentence.strip()
                            clean_t = re.sub(r'[*#_`~]+', '', raw_t).strip()
                            if clean_t:
                                enqueue_text(clean_t)
                            current_sentence = ""

                    if current_sentence.strip() and not stop_evt.is_set():
                        raw_t = current_sentence.strip()
                        clean_t = re.sub(r'[*#_`~]+', '', raw_t).strip()
                        if clean_t:
                            enqueue_text(clean_t)

                except Exception as e:
                    print(f"Streaming Error: {e}")

            response_holder["text"] = final_text_buffer

        except Exception as e:
            print(f"LLM Error: {e}")
        finally:
            if metrics.get("_llm_start"):
                metrics["llm_total"] = time.time() - metrics["_llm_start"]
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    consumer_task = asyncio.create_task(tts_consumer())
    loop = asyncio.get_running_loop()
    producer_task = asyncio.create_task(asyncio.to_thread(llm_producer_threadsafe, loop, stop_event))

    try:
        await asyncio.gather(producer_task, consumer_task)
    except asyncio.CancelledError:
        stop_event.set()

    if not stop_event.is_set():
        full_reply = response_holder["text"]
        if full_reply:
            metrics["output_chars"] = len(full_reply)
            if metrics.get("output_tokens") is None:
                metrics["output_tokens"] = estimate_tokens_from_text(full_reply)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": full_reply})

            # SAVE WITH METADATA + GLOBAL DEDUP
            threading.Thread(target=memory.save, args=(user_text, full_reply, session_id)).start()
            threading.Thread(target=save_interaction, args=(session_id, user_text, full_reply)).start()
            if len(history) <= 2 and user_text:
                snippet = (user_text[:40] + "...") if len(user_text) > 40 else user_text
                threading.Thread(target=set_session_title, args=(session_id, snippet)).start()

            metrics['end_time'] = time.time()
            print_perf_report(metrics)
            finalize_metrics(metrics, "ok")
            await websocket.send_text("DONE")
        else:
            metrics["end_time"] = time.time()
            finalize_metrics(metrics, "empty_reply")
            await websocket.send_text("DONE") # Graceful empty exit
    else:
        metrics['end_time'] = time.time()
        finalize_metrics(metrics, "aborted")
        await websocket.send_text("LOG: --- ABORTED ---")


@app.get("/")
async def get():
    return HTMLResponse(INDEX_HTML)


@app.get("/manifest.json")
async def get_manifest():
    manifest_path = FRONTEND_PUBLIC_DIR / "manifest.json"
    if manifest_path.exists():
        return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))
    return JSONResponse(content={}, status_code=404)


@app.get("/admin")
async def get_admin():
    admin_path = FRONTEND_PUBLIC_DIR / "admin.html"
    if admin_path.exists():
        return HTMLResponse(admin_path.read_text(encoding="utf-8"))
    return HTMLResponse("Admin page not found.", status_code=404)


@app.get("/metrics")
async def get_metrics(limit: int = 50):
    with METRICS_LOCK:
        items = list(METRICS_HISTORY)[-max(1, min(limit, 500)):]
    return {"items": items, "count": len(items)}


@app.get("/models")
async def list_models():
    try:
        data = await asyncio.to_thread(client.models.list)
        raw_models = getattr(data, "data", data)
        models = []
        for m in raw_models:
            if m is None:
                continue
            if hasattr(m, "id"):
                models.append(m.id)
            elif isinstance(m, dict) and m.get("id"):
                models.append(m["id"])
            elif isinstance(m, str):
                models.append(m)
    except Exception as e:
        log_console(f"Model list failed: {e}", "ERR")
        models = [get_current_model()]
    return {"models": models, "current": get_current_model()}


@app.post("/model")
async def set_model(payload: ModelPayload):
    new_model = payload.name.strip()
    if not new_model:
        return JSONResponse(content={"error": "Model name required."}, status_code=400)
    active = set_current_model(new_model)
    log_console(f"Model switched to: {active}", "AI")
    warmed = await asyncio.to_thread(warm_model, active)
    if not warmed:
        return JSONResponse(content={"error": f"Failed to load model '{active}'."}, status_code=500)
    return {"status": "ok", "current": active}


@app.get("/sessions")
async def get_sessions():
    return {"sessions": list_sessions()}


class SessionPayload(BaseModel):
    title: str | None = None
    model: str | None = None
    summary: str | None = None
    tags: str | None = None


@app.get("/sessions/{session_id}/window")
async def session_window(session_id: int, anchor: int = ANCHOR_MESSAGES, recent: int = MAX_RECENT_MESSAGES):
    data = get_session_window(session_id, anchor, recent)
    if not data:
        return JSONResponse(content={"error": "Session not found."}, status_code=404)
    return data


@app.post("/sessions")
async def create_new_session(payload: SessionPayload):
    session_id = create_session(payload.title, payload.model, payload.summary, payload.tags)
    return {
        "id": session_id,
        "title": payload.title or "",
        "summary": payload.summary or "",
        "model": payload.model
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_console("Client Connected", "NET")
    try:
        await websocket.send_text(f"SYS:MODEL:{get_current_model()}")
    except Exception:
        pass
    current_session_id = create_session(None)
    session_history = load_session_messages(current_session_id)
    try:
        await websocket.send_text(f"SYS:SESSION:{current_session_id}")
    except Exception:
        pass

    current_task = None
    stop_event = threading.Event()
    audio_buffer = io.BytesIO()

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "text" in message:
                    text_msg = message["text"]
                    if text_msg == "STOP":
                        stop_event.set()
                        if current_task:
                            current_task.cancel()
                        continue
                    if text_msg.startswith("SESSION:"):
                        try:
                            new_id = int(text_msg.split("SESSION:", 1)[1].strip())
                            if not session_exists(new_id):
                                new_id = create_session(None)
                            session_history = load_session_messages(new_id)
                            current_session_id = new_id
                            await websocket.send_text(f"SYS:SESSION:{current_session_id}")
                        except Exception as e:
                            log_console(f"Session switch failed: {e}", "ERR")
                        continue
                    if text_msg.startswith("MODEL:"):
                        new_model = text_msg.split("MODEL:", 1)[1].strip()
                        if new_model:
                            set_current_model(new_model)
                            warmed = await asyncio.to_thread(warm_model, new_model)
                            if warmed:
                                await websocket.send_text(f"SYS:MODEL:{new_model}")
                            else:
                                await websocket.send_text(f"SYS: Model load failed for {new_model}")
                        continue
                    if text_msg == "CMD:COMMIT_AUDIO":
                        stop_event.clear()
                        if current_task:
                            current_task.cancel()
                        audio_bytes = audio_buffer.getvalue()
                        audio_buffer = io.BytesIO()
                        metrics = init_metrics("voice", current_session_id)
                        metrics["audio_bytes"] = len(audio_bytes)
                        if not audio_bytes:
                            finalize_metrics(metrics, "empty_audio")
                            await websocket.send_text("DONE")
                            continue

                        t_decode = time.time()
                        audio_np = convert_webm_to_numpy(audio_bytes)
                        metrics["audio_decode"] = time.time() - t_decode
                        if audio_np is None:
                            finalize_metrics(metrics, "decode_failed")
                            await websocket.send_text("DONE")
                            continue

                        t_stt = time.time()

                        # --- FIX: Run blocking transcription in a thread ---
                        def run_transcribe():
                            segs, _ = stt_model.transcribe(
                                audio_np,
                                beam_size=5,
                                vad_filter=False,
                                language="en",
                                initial_prompt="Use context."
                            )
                            return list(segs)

                        segments = await asyncio.to_thread(run_transcribe)
                        user_text = " ".join([s.text for s in segments]).strip()
                        metrics['stt'] = time.time() - t_stt
                        metrics["input_chars"] = len(user_text)

                        if not user_text or user_text in ["You", "you", "Thank you", "MBC", "You."]:
                            finalize_metrics(metrics, "empty_transcript")
                            await websocket.send_text("DONE")
                            continue

                        await websocket.send_text(f"LOG:User: {user_text}")
                        current_task = asyncio.create_task(
                            process_and_stream_response(
                                user_text,
                                websocket,
                                session_history,
                                metrics,
                                stop_event,
                                current_session_id,
                                generate_audio=True
                            )
                        )
                        continue

                    # --- TEXT CHAT HANDLER (SILENT MODE) ---
                    if not text_msg.startswith("SYS:") and not text_msg.startswith("LOG:"):
                        user_text = text_msg.strip()
                        if user_text:
                            await websocket.send_text(f"LOG:User: {user_text}")
                            metrics = init_metrics("text", current_session_id)
                            current_task = asyncio.create_task(
                                process_and_stream_response(
                                    user_text,
                                    websocket,
                                    session_history,
                                    metrics,
                                    stop_event,
                                    current_session_id,
                                    generate_audio=False
                                )
                            )
                        continue
                    # ---------------------------------------

                if "bytes" in message:
                    data = message["bytes"]
                    if data:
                        if audio_buffer.tell() == 0:
                            stop_event.clear()
                            if current_task:
                                current_task.cancel()
                        audio_buffer.write(data)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if not QUIET_LOGS:
            print(f"Error: {e}")


if __name__ == "__main__":
    ssl_keyfile = CERTS_DIR / "key.pem"
    ssl_certfile = CERTS_DIR / "cert.pem"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=str(ssl_keyfile),
        ssl_certfile=str(ssl_certfile),
    )
