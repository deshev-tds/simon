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
_ensure_fts5 = db._ensure_fts5


def init_db():
    global db_conn
    db_conn = db.init_db()
    return db_conn


def init_mem_db():
    global mem_conn
    mem_conn = db.init_mem_db()
    return mem_conn


db_conn = init_db()
try:
    _ensure_fts5(db_conn)
except Exception as _e:
    if DEBUG_MODE:
        print(f"[WARN] FTS5 setup failed: {_e}")

mem_conn = init_mem_db()
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


if not TEST_MODE:
    start_mem_threads()

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

MemoryManager = memory_mod.MemoryManager
memory = memory_mod.memory
print("All Models Loaded.")

# --- ARCHIVE MEMORY INTENT ---
ARCHIVE_EXPLICIT_PREFIXES = (
    "archive:",
    "memory:",
    "/archive",
    "/memory",
)

_ARCHIVE_INTENT_PATTERNS = [
    re.compile(r"\b(chatgpt|gpt)\b.*\b(history|archive|conversation|conversations)\b"),
    re.compile(r"\b(past|previous|earlier|last)\b.*\b(chat|conversation|discussion|talk)\b"),
    re.compile(r"\b(do you remember|remember when|what did (we|i|you) (say|discuss|talk(?:ed)? about)|you said|you told me|i told you)\b"),
    re.compile(r"\b(my|our)\b.*\b(history|archive|past chats?)\b"),
    re.compile(r"\b(we talked about|we discussed)\b.*\b(before|earlier|last time|previously)\b"),
]


def _archive_intent(user_text: str):
    if not user_text:
        return False, False, ""
    raw_text = user_text.strip()
    lowered = raw_text.lower()
    for prefix in ARCHIVE_EXPLICIT_PREFIXES:
        if lowered.startswith(prefix):
            trimmed = raw_text[len(prefix):].strip()
            return True, True, trimmed or raw_text
    for pattern in _ARCHIVE_INTENT_PATTERNS:
        if pattern.search(lowered):
            return True, False, raw_text
    return False, False, raw_text

SIMON_TOOLS = tools.SIMON_TOOLS
tool_search_memory = tools.tool_search_memory
tool_analyze_deep = tools.tool_analyze_deep
_safe_args = tools._safe_args
process_and_stream_response = agent.process_and_stream_response




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

    archive_requested, _archive_explicit, archive_query = _archive_intent(user_text)

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

    local_is_weak = False
    if not valid_memories:
        if not distances:
            local_is_weak = True
        elif distances[0] >= LOCAL_WEAK_THRESHOLD:
            local_is_weak = True

    archive_trigger = None
    if archive_requested:
        archive_trigger = "explicit"
    elif local_is_weak:
        archive_trigger = "weak_local"

    archive_valid = []
    archive_payload = []
    archive_age_cutoff_ts = None
    if archive_trigger == "weak_local" and ARCHIVE_WEAK_MAX_AGE_DAYS > 0:
        archive_age_cutoff_ts = time.time() - (ARCHIVE_WEAK_MAX_AGE_DAYS * 24 * 3600)
    if archive_trigger:
        arch_docs, arch_dists, arch_metas = memory_manager.search_archive(archive_query, n_results=3)
        for i, (doc, dist, meta) in enumerate(zip(arch_docs, arch_dists, arch_metas)):
            ts = meta.get("timestamp", 0) if meta else 0
            age_days = (time.time() - ts) / (24 * 3600) if ts > 0 else 0
            if archive_age_cutoff_ts is not None and (not ts or ts < archive_age_cutoff_ts):
                continue
            if dist < ARCHIVE_STRONG_THRESHOLD:
                archive_valid.append(doc)
                archive_payload.append({
                    "archive_doc": doc[:40] + "...",
                    "dist": round(float(dist), 3),
                    "age_days": round(age_days, 1),
                    "rank": i + 1
                })

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

    if archive_valid and (archive_requested or local_is_weak):
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
            "archive_hits": len(archive_valid)
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
                                generate_audio=True,
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
