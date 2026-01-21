import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import asyncio
import time
import soundfile as sf
import io
import re
import threading
from pathlib import Path
import av
import traceback

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
INDEX_HTML = (BASE_DIR / "index.html").read_text(encoding="utf-8")

# --- REVERT: Връщаме стандартния модел за максимална точност ---
WHISPER_MODEL_NAME = "medium" 

TTS_VOICE = "am_fenrir"

MAX_RECENT_MESSAGES = 50
ANCHOR_MESSAGES = 6

# --- GLOBAL MODELS ---
print("Loading AI Models... (High Accuracy Mode)")

# 1. STT - Ползваме int8 за памет, но medium модел
stt_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

# 2. TTS
kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")

# 3. LLM Client
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

print("All Models Loaded.")


def log_console(msg, type="INFO"):
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{type}] {msg}")


def print_perf_report(metrics):
    total_pipeline = metrics['end_time'] - metrics['start_time']
    C_GREEN = '\033[92m'
    C_END = '\033[0m'
    
    print(f"\n{C_GREEN}--- INTERACTION PERF REPORT ---{C_END}")
    print(f"   STT (Whisper):    {metrics['stt']:.3f}s")
    print(f"   TOTAL PROCESS:    {total_pipeline:.3f}s")
    print(f"------------------------------------------\n")


app = FastAPI()

# --- HELPERS ---
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
        if audio_stream is None:
            log_console("No audio stream found.", "DECODE")
            return None

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )

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

    except Exception as e:
        log_console(f"PyAV decode failed: {e}", "DECODE")
        return None


def numpy_to_wav_bytes(audio_np, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


def get_optimized_context(history):
    if len(history) <= (ANCHOR_MESSAGES + MAX_RECENT_MESSAGES):
        return history
    anchor = history[:ANCHOR_MESSAGES]
    recent = history[-MAX_RECENT_MESSAGES:]
    return anchor + recent


async def process_and_stream_response(user_text, websocket, history, metrics, stop_event):
    t_ctx_start = time.time()

    history.append({"role": "user", "content": user_text})
    messages_to_send = get_optimized_context(history)

    metrics['ctx'] = time.time() - t_ctx_start

    if not stop_event.is_set():
        await websocket.send_text("SYS:AI Thinking...")

    q = asyncio.Queue(maxsize=64)
    response_holder = {"text": ""}
    sentence_endings = re.compile(r'[.!?]+')

    async def tts_consumer():
        first_audio_generated = False
        while True:
            if stop_event.is_set(): break
            try:
                item = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if item is None: break
            clean_text = item
            if not clean_text: continue

            samples, sr = await asyncio.to_thread(
                kokoro.create, clean_text, voice=TTS_VOICE, speed=1.0, lang="en-us"
            )

            if stop_event.is_set(): break
            wav_bytes = await asyncio.to_thread(numpy_to_wav_bytes, samples, sr)

            if not first_audio_generated:
                metrics['tts_first'] = time.time() - metrics['start_time']
                first_audio_generated = True

            if not stop_event.is_set():
                await websocket.send_text(f"LOG:AI: {clean_text}")
                await websocket.send_bytes(wav_bytes)

    def llm_producer_threadsafe(loop, stop_evt):
        full_response = ""
        current_sentence = ""
        try:
            stream = client.chat.completions.create(
                model="local-model",
                messages=messages_to_send,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                if stop_evt.is_set():
                    print("LLM Interrupted.")
                    stream.close()
                    break

                delta = chunk.choices[0].delta
                if not getattr(delta, "content", None): continue

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
                if clean_text:
                    asyncio.run_coroutine_threadsafe(q.put(clean_text), loop)

        except Exception as e:
            print(f"LLM Error: {e}")
        finally:
            response_holder["text"] = full_response
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    consumer_task = asyncio.create_task(tts_consumer())
    loop = asyncio.get_running_loop()
    producer_task = asyncio.create_task(asyncio.to_thread(llm_producer_threadsafe, loop, stop_event))

    try:
        await asyncio.gather(producer_task, consumer_task)
    except asyncio.CancelledError:
        print("Tasks Cancelled by User")
        stop_event.set()

    if not stop_event.is_set():
        history.append({"role": "assistant", "content": response_holder["text"]})
        metrics['end_time'] = time.time()
        print_perf_report(metrics)
        await websocket.send_text("DONE")
    else:
        await websocket.send_text("LOG: --- ABORTED ---")


@app.get("/")
async def get():
    return HTMLResponse(INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_console("Client Connected", "NET")
    session_history = []
    
    current_task = None
    stop_event = threading.Event()

    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                
                if "text" in message:
                    if message["text"] == "STOP":
                        log_console("STOP RECEIVED", "CTRL")
                        stop_event.set()
                        if current_task and not current_task.done():
                            current_task.cancel()
                            try:
                                await current_task
                            except asyncio.CancelledError:
                                pass
                        current_task = None
                        continue

                if "bytes" in message:
                    stop_event.clear()
                    if current_task and not current_task.done():
                        current_task.cancel()
                    
                    data = message["bytes"]
                    metrics = {
                        'start_time': time.time(), 'stt': 0, 'ctx': 0, 'ttft': 0,
                        'tts_first': 0, 'total_latency': 0, 'input_tokens': 0, 'end_time': 0
                    }

                    audio_np = convert_webm_to_numpy(data)
                    if audio_np is None:
                        await websocket.send_text("DONE")
                        continue

                    # --- NO TRIMMING: Pass raw audio directly ---
                    # audio_np = trim_leading_trailing_silence(audio_np, SAMPLE_RATE) 

                    t_stt_start = time.time()
                    segments, _ = stt_model.transcribe(
                        audio_np,
                        beam_size=5,          
                        vad_filter=False,     # VAD is OFF
                        language="en",
                        best_of=5,
                        initial_prompt="A raw conversation."
                    )
                    user_text = " ".join([s.text for s in segments]).strip()
                    metrics['stt'] = time.time() - t_stt_start

                    if not user_text:
                        await websocket.send_text("DONE")
                        continue

                    await websocket.send_text(f"LOG:User: {user_text}")
                    
                    current_task = asyncio.create_task(
                        process_and_stream_response(user_text, websocket, session_history, metrics, stop_event)
                    )

    except WebSocketDisconnect:
        log_console("Client Disconnected", "NET")
    except Exception as e:
        log_console(f"Error: {e}", "ERR")
        if DEBUG_MODE:
            log_console(traceback.format_exc(), "ERR")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")