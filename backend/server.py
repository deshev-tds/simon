import os
import socket
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
import subprocess
import sys
from pathlib import Path
import uuid
import json
import shutil
import tempfile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.audio import (
    _analyze_audio,
    _convert_to_mp3_bytes,
    _trim_old_esp_audio,
    convert_webm_to_numpy,
    numpy_to_wav_bytes,
)
from backend.config import *
import backend.agent as agent
import backend.db as db
import backend.tools as tools
from backend.metrics import *
from backend.live_stt import LiveTranscriber

# --- AI IMPORTS ---
from faster_whisper import WhisperModel
from openai import OpenAI
from kokoro_onnx import Kokoro

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


def _allow_remote_model_downloads() -> bool:
    if os.environ.get("SIMON_AUDIO_LOCAL_ONLY") == "1":
        return False
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        return False
    try:
        sock = socket.create_connection(("huggingface.co", 443), timeout=1.0)
        sock.close()
        return True
    except OSError:
        return False


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
        whisper_local_only = not _allow_remote_model_downloads()
        whisper_cache_dir = MODELS_DIR / "whisper"
        try:
            stt_model = WhisperModel(
                WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                download_root=str(whisper_cache_dir),
                local_files_only=whisper_local_only,
            )
        except Exception as exc:
            print(f"STT model unavailable ({exc}). Falling back to dummy STT.")
            stt_model = _DummySTT()

        # 2. TTS
        try:
            kokoro = Kokoro(str(MODELS_DIR / "kokoro-v0_19.onnx"), str(MODELS_DIR / "voices.bin"))
        except Exception as exc:
            print(f"TTS model unavailable ({exc}). Falling back to dummy TTS.")
            kokoro = _DummyTTS()

    # 3. LLM Client
    llm_timeout = LLM_TIMEOUT_S if LLM_TIMEOUT_S > 0 else None
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=llm_timeout)

current_model_lock = threading.Lock()
current_model = DEFAULT_LLM_MODEL


# --- SQLITE HISTORY & FTS SETUP ---
db_lock = db.db_lock
db_conn = None
mem_conn = None
corpus_conn = None
_ensure_fts5 = db._ensure_fts5


def init_db():
    global db_conn
    db_conn = db.init_db()
    return db_conn


def init_mem_db():
    global mem_conn
    mem_conn = db.init_mem_db()
    return mem_conn


def init_corpus_db():
    global corpus_conn
    corpus_conn = db.init_corpus_db()
    return corpus_conn


db_conn = init_db()
try:
    _ensure_fts5(db_conn)
except Exception as _e:
    if DEBUG_MODE:
        print(f"[WARN] FTS5 setup failed: {_e}")

mem_conn = init_mem_db()
corpus_conn = init_corpus_db()
_mem_threads_started = False


def start_mem_threads(stop_event=None):
    global _mem_threads_started
    if _mem_threads_started:
        return None
    _mem_threads_started = True
    return db.start_mem_threads(
        db_conn=db_conn,
        mem_conn=mem_conn,
        mem_seed_limit=MEM_SEED_LIMIT,
        mem_max_rows=MEM_MAX_ROWS,
        mem_prune_interval_s=MEM_PRUNE_INTERVAL_S,
        stop_event=stop_event,
        db_lock=db_lock,
    )


# Hot-session memdb is synced per-session; no global seed threads by default.

create_session = db.create_session
list_sessions = db.list_sessions
get_session_meta = db.get_session_meta
session_exists = db.session_exists
touch_session = db.touch_session
load_session_messages = db.load_session_messages
get_session_transcript = db.get_session_transcript
save_interaction = db.save_interaction
set_session_title = db.set_session_title
get_session_window = db.get_session_window
fts_search_messages = db.fts_search_messages
fts_recursive_search = db.fts_recursive_search


# --- MEMORY SYSTEM (Hardened) ---
import backend.memory as memory_mod
import backend.memory_intents as memory_intents

MemoryManager = memory_mod.MemoryManager
memory = memory_mod.memory
print("All Models Loaded.")

# --- ARCHIVE MEMORY INTENT ---

SIMON_TOOLS = tools.SIMON_TOOLS
tool_search_memory = tools.tool_search_memory
tool_analyze_deep = tools.tool_analyze_deep
_safe_args = tools._safe_args
process_and_stream_response = agent.process_and_stream_response




def log_console(msg, type="INFO"):
    if DEBUG_MODE and not QUIET_LOGS:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{type}] {msg}")


def _start_archive_auto_import():
    if TEST_MODE or not ARCHIVE_AUTO_IMPORT:
        return
    if not ARCHIVE_AUTO_IMPORT_JSON:
        log_console("Archive auto-import enabled but SIMON_ARCHIVE_JSON_PATH is empty.", "WARN")
        return
    json_path = Path(ARCHIVE_AUTO_IMPORT_JSON).expanduser()
    if not json_path.exists():
        log_console(f"Archive auto-import JSON not found: {json_path}", "WARN")
        return

    script_path = ROOT_DIR / "scripts" / "import_chatgpt_archive.py"
    if not script_path.exists():
        log_console(f"Archive importer not found: {script_path}", "WARN")
        return

    def _run():
        cmd = [sys.executable, str(script_path), "--json", str(json_path)]
        if ARCHIVE_AUTO_IMPORT_SINCE_LAST:
            cmd.append("--since-last")
        if ARCHIVE_AUTO_IMPORT_MAX_CODE_RATIO is not None:
            cmd.extend(["--max-code-ratio", str(ARCHIVE_AUTO_IMPORT_MAX_CODE_RATIO)])
        log_console(f"Archive auto-import started: {' '.join(cmd)}", "MEM")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                log_console(f"Archive auto-import failed (code {result.returncode}): {stderr}", "ERR")
            else:
                log_console("Archive auto-import completed.", "MEM")
        except Exception as exc:
            log_console(f"Archive auto-import error: {exc}", "ERR")

    threading.Thread(target=_run, daemon=True).start()


_start_archive_auto_import()


def get_current_model():
    with current_model_lock:
        return current_model


def set_current_model(name: str):
    global current_model
    if name is None:
        return current_model
    clean_name = name.strip()
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


class LlamaSwitchPayload(BaseModel):
    model: str
    mmproj: str | None = None


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


_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _strip_data_url(data_b64: str) -> tuple[str, str | None]:
    if not data_b64:
        return "", None
    if data_b64.startswith("data:"):
        head, _, tail = data_b64.partition(",")
        mime = None
        if head.startswith("data:") and ";base64" in head:
            mime = head.split(":", 1)[1].split(";", 1)[0].strip()
        return tail.strip(), mime
    return data_b64.strip(), None


def _approx_b64_size(raw_b64: str) -> int:
    if not raw_b64:
        return 0
    return int(len(raw_b64) * 3 / 4)


def _normalize_image_payloads(images) -> list[dict]:
    if not images:
        return []
    if len(images) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(f"Too many images (max {MAX_IMAGES_PER_MESSAGE}).")
    normalized = []
    for idx, img in enumerate(images):
        if not isinstance(img, dict):
            raise ValueError("Invalid image payload.")
        raw_b64 = img.get("data_b64") or img.get("data") or ""
        raw_b64 = str(raw_b64)
        raw_b64, inferred_mime = _strip_data_url(raw_b64)
        mime = (img.get("mime") or inferred_mime or "").lower().strip()
        if not raw_b64:
            raise ValueError("Image data is empty.")
        if not mime:
            raise ValueError("Image mime type is required.")
        if mime not in _ALLOWED_IMAGE_MIME:
            raise ValueError(f"Unsupported mime type: {mime}")
        size_bytes = int(img.get("size_bytes") or _approx_b64_size(raw_b64))
        if size_bytes > int(MAX_IMAGE_MB * 1024 * 1024):
            raise ValueError(f"Image exceeds {MAX_IMAGE_MB}MB limit.")
        normalized.append({
            "id": img.get("id") or f"img-{idx}",
            "mime": mime,
            "data_b64": raw_b64,
            "width": img.get("width"),
            "height": img.get("height"),
            "size_bytes": size_bytes,
        })
    return normalized


def _build_vision_content(prompt: str, images: list[dict]):
    parts = []
    if prompt:
        parts.append({"type": "text", "text": prompt})
    else:
        parts.append({"type": "text", "text": ""})
    for img in images or []:
        mime = img.get("mime") or "image/jpeg"
        data_b64 = img.get("data_b64") or ""
        if not data_b64:
            continue
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data_b64}"},
        })
    return parts


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


def _prepend_system_prompt(messages):
    if not SYSTEM_PROMPT:
        return messages
    if messages and messages[0].get("role") == "system" and (messages[0].get("content") or "") == SYSTEM_PROMPT:
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + (messages or [])


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
        messages = _prepend_system_prompt(messages)
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


@app.post("/v1/chat/vision")
async def vision_chat_endpoint(request: Request):
    try:
        data = await request.json()
        prompt = (data.get("prompt") or data.get("text") or "").strip()
        images = _normalize_image_payloads(data.get("images") or [])
        if not prompt and not images:
            return JSONResponse(content={"error": "empty_input", "message": "No prompt or images provided."}, status_code=400)

        session_id = data.get("session_id")
        if session_id is not None and session_exists(session_id):
            current_id = session_id
        else:
            current_id = create_session(None)

        history = load_session_messages(current_id)
        metrics = {}
        context_msgs, _rag_payload = build_rag_context(
            prompt,
            history,
            memory,
            metrics,
            current_id,
            retrieval_cache=None,
        )
        user_msg = {"role": "user", "content": _build_vision_content(prompt, images)}
        messages = _prepend_system_prompt(context_msgs + [user_msg])

        model_name = get_current_model()
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=False,
        )
        ai_text = resp.choices[0].message.content or ""
        store_user_text = prompt or "[image]"
        if images:
            db.save_interaction_with_attachments(current_id, store_user_text, ai_text, images)
        else:
            db.save_interaction(current_id, store_user_text, ai_text)

        payload = _serialize_chat_completion(model_name, ai_text, getattr(resp, "usage", None))
        payload["session_id"] = current_id
        return payload
    except Exception as e:
        log_console(f"Vision chat failed: {e}", "ERR")
        return JSONResponse(content={"error": "vision_chat_failed", "message": str(e)}, status_code=500)


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

def build_rag_context(
    user_text,
    history,
    memory_manager,
    metrics,
    session_id: int | None = None,
    retrieval_cache: dict | None = None,
):
    t_start = time.time()

    def window_messages(msgs, first=10, last=10):
        if len(msgs) <= first + last:
            return msgs
        return msgs[:first] + msgs[-last:]

    history_window = window_messages(history)

    archive_requested, _archive_explicit, archive_query = memory_intents.detect_archive_recall(user_text)

    if retrieval_cache is None:
        # 1) Vector memories (UPDATED: Unpack metadatas + Two-Stage retrieval)
        # Search last 60 days first
        retrieved_docs, distances, metadatas = memory_manager.search(user_text, n_results=3, days_filter=60)

        # 2) SQLite FTS5 keyword recall
        fts_hits = []
        try:
            fts_hits = fts_recursive_search(
                user_text,
                session_id=session_id,
                limit=FTS_MAX_HITS,
                conn=db_conn,
                lock=db_lock,
            )
        except Exception:
            fts_hits = []
    else:
        retrieved_docs = retrieval_cache.get("docs", []) if retrieval_cache else []
        distances = retrieval_cache.get("dists", []) if retrieval_cache else []
        metadatas = retrieval_cache.get("metas", []) if retrieval_cache else []
        fts_hits = retrieval_cache.get("fts_hits", []) if retrieval_cache else []

    probe_s = retrieval_cache.get("probe_s", 0.0) if retrieval_cache else 0.0
    metrics['rag'] = probe_s + (time.time() - t_start)

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

    archive_trigger = "explicit" if archive_requested else None

    archive_valid = []
    archive_payload = []
    archive_session_ids = []
    # Archive recall is explicit-only to avoid background pollution.
    archive_threshold = ARCHIVE_EXPLICIT_THRESHOLD
    if archive_trigger:
        arch_docs, arch_dists, arch_metas = memory_manager.search_archive(archive_query, n_results=3)
        for i, (doc, dist, meta) in enumerate(zip(arch_docs, arch_dists, arch_metas)):
            ts = meta.get("timestamp", 0) if meta else 0
            age_days = (time.time() - ts) / (24 * 3600) if ts > 0 else 0
            if dist < archive_threshold:
                archive_valid.append(doc)
                sess_id = meta.get("session_id") if meta else None
                if sess_id is not None:
                    try:
                        archive_session_ids.append(int(sess_id))
                    except Exception:
                        pass
                archive_payload.append({
                    "archive_doc": doc[:40] + "...",
                    "dist": round(float(dist), 3),
                    "age_days": round(age_days, 1),
                    "threshold": round(float(archive_threshold), 3),
                    "rank": i + 1,
                    "session_id": sess_id,
                })

    archive_fts_lines = []
    archive_fts_hits = []
    if archive_requested and archive_session_ids:
        seen_ids = set()
        unique_sessions = []
        for sid in archive_session_ids:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            unique_sessions.append(sid)
        for sid in unique_sessions[:ARCHIVE_FTS_MAX_SESSIONS]:
            hits = fts_recursive_search(
                archive_query,
                session_id=sid,
                limit=ARCHIVE_FTS_MAX_HITS,
                conn=db_conn,
                lock=db_lock,
            )
            archive_fts_hits.extend(hits)

        seen_rows = set()
        for h in archive_fts_hits:
            role = h.get("role", "?")
            raw_content = h.get("content") or ""
            clean_content = raw_content.strip().replace("\n", " ")
            if len(clean_content) > 240:
                clean_content = clean_content[:240] + "..."
            key = (h.get("session_id"), h.get("created_at"), role, clean_content)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            score = h.get("score")
            if score is None:
                archive_fts_lines.append(f"- [{role}] {clean_content}")
            else:
                archive_fts_lines.append(f"- [{role}] {clean_content} (bm25={score:.2f})")

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

    if archive_requested and (archive_fts_lines or archive_valid):
        if archive_fts_lines:
            archive_block = "\n".join(archive_fts_lines)
            rag_injection.append({
                "role": "system",
                "content": (
                    "Archive recall (ChatGPT history). Semantically matched sessions were narrowed "
                    "via the archive index; exact excerpts are below:\n"
                    f"{archive_block}\n"
                    "(Treat as raw evidence; reconcile with the current turn.)"
                )
            })
        else:
            archive_block = "\n".join([f"- {m}" for m in archive_valid])
            rag_injection.append({
                "role": "system",
                "content": (
                    "Relevant archive memories (ChatGPT history):\n"
                    f"{archive_block}\n"
                    "(Use only if directly relevant; treat as historical context.)"
                )
            })
        rag_payload.append({
            "archive_trigger": archive_trigger,
            "archive_hits": len(archive_valid),
            "archive_fts_hits": len(archive_fts_lines),
        })
        rag_payload.extend(archive_payload)

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


LLAMA_CTL_PATH = (ROOT_DIR / "scripts" / "llama_ctl.sh").resolve()
LLAMA_SUDO_USER = os.environ.get("SIMON_LLAMA_SUDO_USER", "deshev").strip()
LLAMA_READY_TIMEOUT_S = float(os.environ.get("SIMON_LLAMA_READY_TIMEOUT_S", "180").strip() or "180")
LLAMA_CTL_TIMEOUT_S = float(os.environ.get("SIMON_LLAMA_CTL_TIMEOUT_S", "300").strip() or "300")


def _llama_ctl_cmd(args: list[str]) -> list[str]:
    if not LLAMA_CTL_PATH.exists():
        raise RuntimeError(f"llama_ctl.sh not found at: {LLAMA_CTL_PATH}")
    base = [str(LLAMA_CTL_PATH), *args]
    if LLAMA_SUDO_USER:
        # -H sets HOME for the target user; important for rootless toolbox/podman.
        return ["sudo", "-n", "-H", "-u", LLAMA_SUDO_USER, *base]
    return base


def _run_llama_ctl(args: list[str], timeout_s: float | None = None) -> subprocess.CompletedProcess:
    cmd = _llama_ctl_cmd(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s or LLAMA_CTL_TIMEOUT_S)


def _wait_llama_ready(timeout_s: float) -> list[str]:
    deadline = time.time() + max(1.0, timeout_s)
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            data = client.models.list()
            raw_models = getattr(data, "data", data)
            models: list[str] = []
            for m in raw_models:
                if m is None:
                    continue
                if hasattr(m, "id"):
                    models.append(m.id)
                elif isinstance(m, dict) and m.get("id"):
                    models.append(m["id"])
                elif isinstance(m, str):
                    models.append(m)
            if models:
                return models
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"llama-server not ready after {timeout_s:.0f}s: {last_err}")


@app.get("/admin/llama/models")
async def admin_llama_models(refresh: int = 0):
    # refresh is reserved for future caching; for now scans every call.
    try:
        result = await asyncio.to_thread(_run_llama_ctl, ["list", "--json"])
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return JSONResponse(content={"error": "llama_list_failed", "message": msg, "code": result.returncode}, status_code=500)
        return JSONResponse(content=json.loads(result.stdout or "{}"))
    except Exception as exc:
        return JSONResponse(content={"error": "llama_list_error", "message": str(exc)}, status_code=500)


@app.get("/admin/llama/status")
async def admin_llama_status():
    try:
        result = await asyncio.to_thread(_run_llama_ctl, ["status", "--json"])
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return JSONResponse(content={"error": "llama_status_failed", "message": msg, "code": result.returncode}, status_code=500)
        return JSONResponse(content=json.loads(result.stdout or "{}"))
    except Exception as exc:
        return JSONResponse(content={"error": "llama_status_error", "message": str(exc)}, status_code=500)


@app.post("/admin/llama/eject")
async def admin_llama_eject():
    # Eject = stop llama-server process to free VRAM.
    try:
        result = await asyncio.to_thread(_run_llama_ctl, ["stop"])
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return JSONResponse(content={"error": "llama_eject_failed", "message": msg, "code": result.returncode}, status_code=500)
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(content={"error": "llama_eject_error", "message": str(exc)}, status_code=500)


@app.post("/admin/llama/switch")
async def admin_llama_switch(payload: LlamaSwitchPayload):
    model_path = (payload.model or "").strip()
    mmproj_path = (payload.mmproj or "").strip()
    if not model_path:
        return JSONResponse(content={"error": "empty_model", "message": "model is required"}, status_code=400)

    args = ["restart", "--model", model_path]
    if mmproj_path:
        args += ["--mmproj", mmproj_path]
    else:
        # Make sure the script stays non-interactive even if it grows heuristics later.
        args += ["--no-mmproj"]

    try:
        result = await asyncio.to_thread(_run_llama_ctl, args)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return JSONResponse(content={"error": "llama_switch_failed", "message": msg, "code": result.returncode}, status_code=500)

        # After restart, re-sync the OpenAI model id used by the backend.
        models = await asyncio.to_thread(_wait_llama_ready, LLAMA_READY_TIMEOUT_S)
        active = models[0] if models else get_current_model()
        set_current_model(active)
        await asyncio.to_thread(warm_model, active)

        return {"status": "ok", "active_model": active, "models": models}
    except Exception as exc:
        return JSONResponse(content={"error": "llama_switch_error", "message": str(exc)}, status_code=500)


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
        db.sync_mem_db_for_session(current_session_id, limit=MEM_HOT_SESSION_LIMIT, lock=db_lock)
    except Exception as exc:
        log_console(f"Hot memdb sync failed: {exc}", "WARN")
    try:
        await websocket.send_text(f"SYS:SESSION:{current_session_id}")
    except Exception:
        pass

    current_task = None
    stop_event = threading.Event()
    audio_buffer = io.BytesIO()
    inbound_audio_mode = "webm"  # "webm" (browser MediaRecorder) | "pcm16le" (Android AudioRecord)
    inbound_pcm_sr = SAMPLE_RATE
    recording_started_at = None
    recording_stopped_at = None
    first_partial_at = None

    def _pcm16le_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray | None:
        if not pcm_bytes:
            return None
        if len(pcm_bytes) < 2:
            return None
        if len(pcm_bytes) % 2 == 1:
            # Drop trailing byte; avoids np.frombuffer error.
            pcm_bytes = pcm_bytes[:-1]
        try:
            audio_i16 = np.frombuffer(pcm_bytes, dtype="<i2")
            if audio_i16.size == 0:
                return None
            return audio_i16.astype(np.float32) / 32768.0
        except Exception:
            return None

    def _mark_first_partial():
        nonlocal first_partial_at
        if first_partial_at is None:
            first_partial_at = time.time()

    def _reset_recording_markers():
        nonlocal recording_started_at, recording_stopped_at, first_partial_at
        recording_started_at = None
        recording_stopped_at = None
        first_partial_at = None

    live_stt = None
    if LIVE_STT_ENABLED:
        async def _send_live(payload: dict):
            await websocket.send_text(f"LIVE:STT:{json.dumps(payload)}")

        live_stt = LiveTranscriber(
            stt_model=stt_model,
            send_event=_send_live,
            sample_rate=SAMPLE_RATE,
            window_s=LIVE_STT_WINDOW_S,
            step_s=LIVE_STT_STEP_S,
            keep_s=LIVE_STT_KEEP_S,
            min_window_s=LIVE_STT_MIN_WINDOW_S,
            overlap_words=LIVE_STT_OVERLAP_WORDS,
            log_fn=log_console,
            on_first_partial=_mark_first_partial,
        )

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
                    if text_msg.startswith("AUDIO:"):
                        # Client audio transport negotiation.
                        # - "AUDIO:WEBM" (default): client streams MediaRecorder WebM/Opus chunks.
                        # - "AUDIO:PCM16LE:16000": client streams raw PCM16LE mono @ 16kHz.
                        try:
                            parts = [p.strip() for p in text_msg.split(":") if p is not None]
                            # parts[0] == "AUDIO"
                            mode = parts[1].lower() if len(parts) > 1 else ""
                            if mode in {"webm", "default"}:
                                inbound_audio_mode = "webm"
                                inbound_pcm_sr = SAMPLE_RATE
                                await websocket.send_text("SYS:AUDIO:WEBM")
                            elif mode in {"pcm16le", "pcm"}:
                                sr = SAMPLE_RATE
                                if len(parts) >= 3:
                                    try:
                                        sr = int(parts[2])
                                    except Exception:
                                        sr = SAMPLE_RATE
                                if sr != SAMPLE_RATE:
                                    await websocket.send_text(f"SYS:AUDIO:ERR: unsupported sample_rate {sr}; expected {SAMPLE_RATE}")
                                else:
                                    inbound_audio_mode = "pcm16le"
                                    inbound_pcm_sr = sr
                                    await websocket.send_text(f"SYS:AUDIO:PCM16LE:{sr}")
                            else:
                                await websocket.send_text(f"SYS:AUDIO:ERR: unknown mode '{mode}'")
                        except Exception as e:
                            await websocket.send_text(f"SYS:AUDIO:ERR: {e}")
                        continue
                    if text_msg.startswith("SESSION:"):
                        try:
                            new_id = int(text_msg.split("SESSION:", 1)[1].strip())
                            if not session_exists(new_id):
                                new_id = create_session(None)
                            session_history = load_session_messages(new_id)
                            current_session_id = new_id
                            try:
                                db.sync_mem_db_for_session(current_session_id, limit=MEM_HOT_SESSION_LIMIT, lock=db_lock)
                            except Exception as exc:
                                log_console(f"Hot memdb sync failed: {exc}", "WARN")
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
                        recording_stopped_at = time.time()
                        if live_stt:
                            live_stt.stop()
                        audio_bytes = audio_buffer.getvalue()
                        audio_buffer = io.BytesIO()
                        metrics = init_metrics("voice", current_session_id)
                        metrics["audio_mode"] = inbound_audio_mode
                        metrics["record_start"] = recording_started_at
                        metrics["record_end"] = recording_stopped_at
                        metrics["first_partial"] = first_partial_at
                        if recording_started_at and first_partial_at:
                            metrics["partial_latency"] = first_partial_at - recording_started_at
                        metrics["audio_bytes"] = len(audio_bytes)
                        if not audio_bytes:
                            finalize_metrics(metrics, "empty_audio")
                            await websocket.send_text("DONE")
                            _reset_recording_markers()
                            continue

                        t_decode = time.time()
                        if inbound_audio_mode == "pcm16le":
                            audio_np = _pcm16le_bytes_to_float32(audio_bytes)
                        else:
                            audio_np = convert_webm_to_numpy(audio_bytes)
                        metrics["audio_decode"] = time.time() - t_decode
                        if audio_np is None:
                            finalize_metrics(metrics, "decode_failed")
                            await websocket.send_text("DONE")
                            _reset_recording_markers()
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
                        metrics["final_transcript_ready"] = time.time()
                        if recording_stopped_at:
                            metrics["final_latency"] = metrics["final_transcript_ready"] - recording_stopped_at
                        metrics["input_chars"] = len(user_text)
                        if metrics.get("stt"):
                            audio_duration = len(audio_np) / float(SAMPLE_RATE)
                            if audio_duration > 0:
                                metrics["stt_rtf"] = audio_duration / metrics["stt"]

                        if not user_text or user_text in ["You", "you", "Thank you", "MBC", "You."]:
                            finalize_metrics(metrics, "empty_transcript")
                            await websocket.send_text("DONE")
                            _reset_recording_markers()
                            continue

                        if live_stt:
                            await live_stt.send_final(user_text)

                        await websocket.send_text(f"LOG:User: {user_text}")
                        current_task = asyncio.create_task(
                            process_and_stream_response(
                                user_text,
                                websocket,
                                session_history,
                                metrics,
                                stop_event,
                                current_session_id,
                                image_payloads=None,
                                generate_audio=True,
                                client=client,
                                kokoro=kokoro,
                                memory=memory,
                                build_rag_context=build_rag_context,
                                log_console=log_console,
                                get_current_model=get_current_model,
                            )
                        )
                        _reset_recording_markers()
                        continue

                    parsed_payload = None
                    if text_msg and text_msg.lstrip().startswith("{"):
                        try:
                            parsed_payload = json.loads(text_msg)
                        except Exception:
                            parsed_payload = None

                    if isinstance(parsed_payload, dict) and parsed_payload.get("type") in {"chat", "vision"}:
                        try:
                            user_text = (parsed_payload.get("prompt") or parsed_payload.get("text") or "").strip()
                            image_payloads = _normalize_image_payloads(parsed_payload.get("images") or [])
                        except Exception as exc:
                            await websocket.send_text(f"LOG:AI: Error: {exc}")
                            await websocket.send_text("DONE")
                            continue

                        if not user_text and not image_payloads:
                            await websocket.send_text("LOG:AI: Error: empty message.")
                            await websocket.send_text("DONE")
                            continue

                        log_payload = {"text": user_text, "images": image_payloads}
                        await websocket.send_text(f"LOG:User: {json.dumps(log_payload)}")
                        metrics = init_metrics("text", current_session_id)
                        current_task = asyncio.create_task(
                            process_and_stream_response(
                                user_text,
                                websocket,
                                session_history,
                                metrics,
                                stop_event,
                                current_session_id,
                                image_payloads=image_payloads,
                                generate_audio=False,
                                client=client,
                                kokoro=kokoro,
                                memory=memory,
                                build_rag_context=build_rag_context,
                                log_console=log_console,
                                get_current_model=get_current_model,
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
                                    image_payloads=None,
                                    generate_audio=False,
                                    client=client,
                                    kokoro=kokoro,
                                    memory=memory,
                                    build_rag_context=build_rag_context,
                                    log_console=log_console,
                                    get_current_model=get_current_model,
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
                            recording_started_at = time.time()
                            recording_stopped_at = None
                            first_partial_at = None
                            if live_stt:
                                live_stt.reset()
                                await live_stt.send_reset()
                        audio_buffer.write(data)
                        if live_stt:
                            if inbound_audio_mode == "pcm16le":
                                audio_np = _pcm16le_bytes_to_float32(data)
                            else:
                                audio_np = convert_webm_to_numpy(data)
                            if audio_np is not None:
                                live_stt.add_audio(audio_np)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if not QUIET_LOGS:
            print(f"Error: {e}")
    finally:
        if live_stt:
            live_stt.stop()


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
