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
LM_STUDIO_URL = "http://localhost:1234/v1"
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_PUBLIC_DIR = FRONTEND_DIR / "public"
CERTS_DIR = ROOT_DIR / "certs"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
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

# --- GLOBAL MODELS ---
print("Loading AI Models... (Surgical RAG Mode)")

# 1. STT
stt_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

# 2. TTS
kokoro = Kokoro(str(MODELS_DIR / "kokoro-v0_19.onnx"), str(MODELS_DIR / "voices.bin"))

# 3. LLM Client
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
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

# --- SQLITE HISTORY ---
db_lock = threading.Lock()
def init_db():
    needs_reset = False
    existing_conn = None

    if DB_PATH.exists():
        try:
            existing_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
            existing_conn.execute("PRAGMA foreign_keys = ON;")
            return existing_conn

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
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

# --- MEMORY SYSTEM ---
class MemoryManager:
    def __init__(self):
        print("Initializing Vector Memory (ChromaDB)...")
        self.chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "simon_db"))
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="simon_memories",
            embedding_function=self.emb_fn
        )
        print(" Memory Loaded.")

    def search(self, query_text, n_results=3):
        if self.collection.count() == 0:
            return [], []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'distances']
        )
        docs = results['documents'][0] if results['documents'] else []
        dists = results['distances'][0] if results['distances'] else []
        return docs, dists

    def save(self, user_text, ai_text):
        existing_docs, existing_dists = self.search(user_text, n_results=1)
        if existing_dists and existing_dists[0] < 0.2:
            if DEBUG_MODE:
                print(f" Memory duplication detected (Dist: {existing_dists[0]:.4f}). Skipping save.")
            return

        memory_text = f"User: {user_text} | AI: {ai_text}"
        self.collection.add(
            documents=[memory_text],
            metadatas=[{"role": "conversation", "timestamp": time.time()}],
            ids=[str(uuid.uuid4())]
        )

memory = MemoryManager()
print("All Models Loaded.")

def log_console(msg, type="INFO"):
    if DEBUG_MODE:
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
    if resampled is None: return []
    if isinstance(resampled, list): return resampled
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
        if audio_stream is None: return None

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio_stream):
            out_frames = _ensure_frames_list(resampler.resample(frame))
            for of in out_frames:
                arr = of.to_ndarray()
                if arr.ndim == 2: arr = arr[0]
                chunks.append(arr)
        for of in _ensure_frames_list(resampler.resample(None)):
            arr = of.to_ndarray()
            if arr.ndim == 2: arr = arr[0]
            chunks.append(arr)
        if not chunks: return None
        pcm = np.concatenate(chunks).astype(np.int16)
        return pcm.astype(np.float32) / 32768.0
    except Exception: return None

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
    """
    Mimics OpenAI Whisper API for the ESP32 client.
    Accepts multipart upload and returns {"text": "..."}.
    """
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
    """
    Mimics OpenAI Chat Completion API for the ESP32 client.
    """
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
    """
    Mimics OpenAI TTS API for the ESP32 client.
    Generates MP3; falls back to WAV if MP3 fails.
    """
    try:
        data = await request.json()
        text = data.get("input", "")
        if not text:
            return JSONResponse(content={"error": "empty_input", "message": "No input text provided."}, status_code=400)

        # Generate audio with Kokoro (mono float32)
        samples, sr = await asyncio.to_thread(kokoro.create, text, voice=TTS_VOICE, speed=1.0, lang="en-us")

        # Try MP3 first (preferred by ESP32 firmware)
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

def build_rag_context(user_text, history, memory_manager, metrics):
    t_start = time.time()
    def window_messages(msgs, first=10, last=10):
        if len(msgs) <= first + last:
            return msgs
        return msgs[:first] + msgs[-last:]

    history_window = window_messages(history)
    retrieved_docs, distances = memory_manager.search(user_text, n_results=3)
    metrics['rag'] = time.time() - t_start
    
    rag_payload = []
    valid_memories = []

    if retrieved_docs:
        print(f"\n \033[96m[VECTOR SEARCH] Query: '{user_text}'\033[0m")
        print(f"   \033[90m--------------------------------------------------\033[0m")
        for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
            score_color = '\033[92m' if dist < 0.8 else '\033[93m'
            print(f"    #{i+1} [{score_color}Dist: {dist:.4f}\033[0m] \033[3m\"{doc[:60]}...\"\033[0m")
            if dist < RAG_THRESHOLD:
                valid_memories.append(doc)
                rag_payload.append({"doc": doc[:40] + "...", "dist": round(float(dist), 3)})
        print(f"   \033[90m--------------------------------------------------\033[0m\n")
    else:
        print(f"\n \033[90m[VECTOR SEARCH] No memories found.\033[0m\n")

    anchor = history_window[:ANCHOR_MESSAGES]
    remaining = history_window[ANCHOR_MESSAGES:]
    recent = remaining[-MAX_RECENT_MESSAGES:]
    
    rag_injection = []
    if valid_memories:
        memory_block = "\n".join([f"- {m}" for m in valid_memories])
        rag_injection = [{
            "role": "system", 
            "content": f"Relevant past memories:\n{memory_block}\n(Use these to personalize, but prioritize current context.)"
        }]

    return anchor + rag_injection + recent, rag_payload

# --- STREAMING LOGIC (UPDATED: Supports Silent Text Mode) ---
async def process_and_stream_response(user_text, websocket, history, metrics, stop_event, session_id, generate_audio=True):
    t_ctx_start = time.time()
    
    messages_to_send, rag_payload = build_rag_context(user_text, history, memory, metrics)
    messages_to_send.append({"role": "user", "content": user_text})
    
    metrics['ctx'] = time.time() - t_ctx_start
    metrics['input_chars'] = len(user_text)
    if metrics.get("input_tokens") is None:
        metrics["input_tokens"] = estimate_tokens_from_messages(messages_to_send)

    if rag_payload:
        await websocket.send_text(f"RAG:{json.dumps(rag_payload)}")

    q = asyncio.Queue(maxsize=64)
    response_holder = {"text": ""}
    sentence_endings = re.compile(r'[.!?]+')

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

            # --- AUDIO GENERATION LOGIC ---
            wav_bytes = None
            if generate_audio:
                if metrics.get("tts_total") is None:
                    metrics["tts_total"] = 0
                tts_start = time.time()
                samples, sr = await asyncio.to_thread(kokoro.create, clean_text, voice=TTS_VOICE, speed=1.0, lang="en-us")
                if stop_event.is_set(): break
                wav_bytes = await asyncio.to_thread(numpy_to_wav_bytes, samples, sr)
                metrics["tts_total"] += time.time() - tts_start

                if not first_audio_generated:
                    metrics['tts_first'] = time.time() - metrics['start_time']
                    first_audio_generated = True

            # --- SEND TO CLIENT ---
            if not stop_event.is_set():
                # Always send text log (for chat UI)
                await websocket.send_text(f"LOG:AI: {clean_text}")
                # Only send audio if requested (Voice Mode)
                if generate_audio and wav_bytes:
                    await websocket.send_bytes(wav_bytes)

    def llm_producer_threadsafe(loop, stop_evt):
        full_response = ""
        current_sentence = ""
        try:
            model_name = get_current_model()
            log_console(f"Using model: {model_name}", "AI")
            llm_start = time.time()
            metrics["_llm_start"] = llm_start
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages_to_send,
                    temperature=0.7,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages_to_send,
                    temperature=0.7,
                    stream=True,
                )

            for chunk in stream:
                if stop_evt.is_set():
                    stream.close()
                    break
                usage = getattr(chunk, "usage", None)
                if usage:
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                    else:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                    if prompt_tokens is not None:
                        metrics["input_tokens"] = prompt_tokens
                    if completion_tokens is not None:
                        metrics["output_tokens"] = completion_tokens
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                if not getattr(delta, "content", None): continue
                if metrics.get("ttft") is None:
                    metrics["ttft"] = time.time() - llm_start
                token = delta.content
                full_response += token
                current_sentence += token

                if sentence_endings.search(current_sentence[-2:]) and len(current_sentence.strip()) > 5:
                    raw_text = current_sentence.strip()
                    clean_text = re.sub(r'[*#_`~]+', '', raw_text).strip()
                    if clean_text:
                        asyncio.run_coroutine_threadsafe(q.put(clean_text), loop)
                    current_sentence = ""
            
            if current_sentence.strip() and not stop_evt.is_set():
                raw_text = current_sentence.strip()
                clean_text = re.sub(r'[*#_`~]+', '', raw_text).strip()
                if clean_text: asyncio.run_coroutine_threadsafe(q.put(clean_text), loop)

        except Exception as e:
            print(f"LLM Error: {e}")
        finally:
            if metrics.get("_llm_start"):
                metrics["llm_total"] = time.time() - metrics["_llm_start"]
            response_holder["text"] = full_response
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
        metrics["output_chars"] = len(full_reply)
        if metrics.get("output_tokens") is None:
            metrics["output_tokens"] = estimate_tokens_from_text(full_reply)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": full_reply})
        
        threading.Thread(target=memory.save, args=(user_text, full_reply)).start()
        threading.Thread(target=save_interaction, args=(session_id, user_text, full_reply)).start()
        if len(history) <= 2 and user_text:
            snippet = (user_text[:40] + "...") if len(user_text) > 40 else user_text
            threading.Thread(target=set_session_title, args=(session_id, snippet)).start()
        
        metrics['end_time'] = time.time()
        print_perf_report(metrics)
        finalize_metrics(metrics, "ok")
        await websocket.send_text("DONE")
    else:
        metrics['end_time'] = time.time()
        finalize_metrics(metrics, "aborted")
        await websocket.send_text("LOG: --- ABORTED ---")

@app.get("/")
async def get(): return HTMLResponse(INDEX_HTML)

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
            if m is None: continue
            if hasattr(m, "id"): models.append(m.id)
            elif isinstance(m, dict) and m.get("id"): models.append(m["id"])
            elif isinstance(m, str): models.append(m)
    except Exception as e:
        log_console(f"Model list failed: {e}", "ERR")
        models = [get_current_model()]
    return {"models": models, "current": get_current_model()}

@app.post("/model")
async def set_model(payload: ModelPayload):
    new_model = payload.name.strip()
    if not new_model: return JSONResponse(content={"error": "Model name required."}, status_code=400)
    active = set_current_model(new_model)
    log_console(f"Model switched to: {active}", "AI")
    warmed = await asyncio.to_thread(warm_model, active)
    if not warmed: return JSONResponse(content={"error": f"Failed to load model '{active}'."}, status_code=500)
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
    try: await websocket.send_text(f"SYS:MODEL:{get_current_model()}")
    except: pass
    current_session_id = create_session(None)
    session_history = load_session_messages(current_session_id)
    try: await websocket.send_text(f"SYS:SESSION:{current_session_id}")
    except: pass
    
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
                        if current_task: current_task.cancel()
                        continue
                    if text_msg.startswith("SESSION:"):
                        try:
                            new_id = int(text_msg.split("SESSION:", 1)[1].strip())
                            if not session_exists(new_id):
                                new_id = create_session(None)
                            session_history = load_session_messages(new_id)
                            current_session_id = new_id
                            await websocket.send_text(f"SYS:SESSION:{current_session_id}")
                        except Exception as e: log_console(f"Session switch failed: {e}", "ERR")
                        continue
                    if text_msg.startswith("MODEL:"):
                        new_model = text_msg.split("MODEL:", 1)[1].strip()
                        if new_model:
                            set_current_model(new_model)
                            warmed = await asyncio.to_thread(warm_model, new_model)
                            if warmed: await websocket.send_text(f"SYS:MODEL:{new_model}")
                            else: await websocket.send_text(f"SYS: Model load failed for {new_model}")
                        continue
                    if text_msg == "CMD:COMMIT_AUDIO":
                        stop_event.clear()
                        if current_task: current_task.cancel()
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
                            # Transcribe returns a generator, convert to list inside the thread
                            segs, _ = stt_model.transcribe(
                                audio_np, 
                                beam_size=5, 
                                vad_filter=False, 
                                language="en", 
                                initial_prompt="Use context."
                            )
                            return list(segs) # Consume generator here!

                        segments = await asyncio.to_thread(run_transcribe)
                        user_text = " ".join([s.text for s in segments]).strip()
                        metrics['stt'] = time.time() - t_stt
                        metrics["input_chars"] = len(user_text)

                        if not user_text or user_text in ["You", "you", "Thank you", "MBC", "You."]: 
                            finalize_metrics(metrics, "empty_transcript")
                            await websocket.send_text("DONE")
                            continue

                        await websocket.send_text(f"LOG:User: {user_text}")
                        current_task = asyncio.create_task(process_and_stream_response(user_text, websocket, session_history, metrics, stop_event, current_session_id, generate_audio=True))
                        continue
                    
                    # --- TEXT CHAT HANDLER (SILENT MODE) ---
                    if not text_msg.startswith("SYS:") and not text_msg.startswith("LOG:"):
                        user_text = text_msg.strip()
                        if user_text:
                            await websocket.send_text(f"LOG:User: {user_text}")
                            metrics = init_metrics("text", current_session_id)
                            # generate_audio=False -> Silent
                            current_task = asyncio.create_task(
                                process_and_stream_response(user_text, websocket, session_history, metrics, stop_event, current_session_id, generate_audio=False)
                            )
                        continue
                    # ---------------------------------------

                if "bytes" in message:
                    data = message["bytes"]
                    if data:
                        if audio_buffer.tell() == 0:
                            stop_event.clear()
                            if current_task: current_task.cancel()
                        audio_buffer.write(data)

    except WebSocketDisconnect: pass
    except Exception as e: print(f"Error: {e}")

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
