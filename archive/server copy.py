import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import ffmpeg
import numpy as np
import asyncio
import time
import soundfile as sf
import io
import re

# --- AI IMPORTS ---
from faster_whisper import WhisperModel
from openai import OpenAI
from kokoro_onnx import Kokoro

# --- CONFIG & DEBUG ---
SAMPLE_RATE = 16000     
TTS_SAMPLE_RATE = 24000 
DEBUG_MODE = True
LM_STUDIO_URL = "http://localhost:1234/v1"

# --- OPTIMIZATION 1: По-бърз STT ---
# Сваляме данъка от 2.5s на ~0.8s
WHISPER_MODEL_NAME = "distil-medium.en" 

TTS_VOICE = "am_fenrir" 

# --- OPTIMIZATION 2: Smart Context Window ---
MAX_RECENT_MESSAGES = 6 # Пазим последните 8 (активен диалог)
ANCHOR_MESSAGES = 6     # Пазим първите 2 (Setup/Rules)

# --- GLOBAL MODELS ---
print("⏳ Loading AI Models... (Smart Context Mode)")

# 1. STT
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

# --- TELEMETRY HELPER ---
def print_perf_report(metrics):
    total_pipeline = metrics['end_time'] - metrics['start_time']
    
    C_GREEN = '\033[92m'
    C_YELLOW = '\033[93m'
    C_RED = '\033[91m'
    C_END = '\033[0m'
    
    ttft = metrics.get('ttft', 0)
    color = C_GREEN if ttft < 1.0 else (C_YELLOW if ttft < 2.5 else C_RED)

    print(f"\n{color}📊 --- INTERACTION PERF REPORT ---{C_END}")
    print(f"   🎤 STT (Whisper):    {metrics['stt']:.3f}s")
    print(f"   🧠 CTX (Injection):  {metrics['ctx']:.3f}s | Size: ~{metrics['input_tokens']} toks")
    print(f"   💡 LLM (TTFT):       {metrics['ttft']:.3f}s (Time To First Token)")
    print(f"   🗣️ TTS (1st Chunk):  {metrics['tts_first']:.3f}s")
    print(f"   ---------------------------------------")
    print(f"   ⚡ TOTAL LATENCY:    {metrics['total_latency']:.3f}s (End of Speech -> Start of Audio)")
    print(f"   🔄 TOTAL PROCESS:    {total_pipeline:.3f}s")
    print(f"------------------------------------------\n")

app = FastAPI()

# --- HELPERS ---
def convert_webm_to_numpy(webm_bytes):
    try:
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar=str(SAMPLE_RATE))
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        out, err = process.communicate(input=webm_bytes)
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        return None

def numpy_to_wav_bytes(audio_np, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()

# --- SMART CONTEXT LOGIC ---
def get_optimized_context(history):
    # Ако историята е кратка, връщаме всичко
    if len(history) <= (ANCHOR_MESSAGES + MAX_RECENT_MESSAGES):
        return history
    
    # Иначе: Anchor + Gap + Recent
    anchor = history[:ANCHOR_MESSAGES] # Първите 2 (Setup)
    recent = history[-MAX_RECENT_MESSAGES:] # Последните 8
    
    # Можем да върнем директно слеения списък
    return anchor + recent

# --- STREAMING LOGIC ---
async def process_and_stream_response(user_text, websocket, history, metrics):
    
    t_ctx_start = time.time()

    # 1. Add User Input to Global History
    history.append({"role": "user", "content": user_text})
    
    # 2. Prepare Optimized Context (Anchor + Recent)
    messages_to_send = get_optimized_context(history)

    metrics['ctx'] = time.time() - t_ctx_start
    full_context_str = "".join([m['content'] for m in messages_to_send])
    metrics['input_tokens'] = len(full_context_str) // 4

    t_llm_start = time.time()

    # Start Stream
    stream = client.chat.completions.create(
        model="local-model",
        messages=messages_to_send, 
        temperature=0.7,
        stream=True, 
    )

    full_response = ""
    current_sentence = ""
    sentence_endings = re.compile(r'[.!?]+')
    
    first_token_received = False
    first_audio_generated = False

    await websocket.send_text("SYS:AI Thinking...")

    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            
            if not first_token_received:
                metrics['ttft'] = time.time() - t_llm_start
                first_token_received = True

            full_response += token
            current_sentence += token
            
            if sentence_endings.search(token) and len(current_sentence.strip()) > 5:
                raw_text = current_sentence.strip()
                clean_text = re.sub(r'[*#_`~]+', '', raw_text).strip()

                if clean_text:
                    t_tts_start = time.time()
                    samples, sr = kokoro.create(clean_text, voice=TTS_VOICE, speed=1.1, lang="en-us")
                    t_tts_end = time.time()
                    
                    if not first_audio_generated:
                        metrics['tts_first'] = t_tts_end - t_tts_start
                        metrics['total_latency'] = t_tts_end - metrics['start_time'] 
                        first_audio_generated = True
                    
                    wav_bytes = numpy_to_wav_bytes(samples, sr)
                    await websocket.send_text(f"LOG:AI: {clean_text}") # Front-end log stays
                    await websocket.send_bytes(wav_bytes)
                
                current_sentence = "" 

    if current_sentence.strip():
        raw_text = current_sentence.strip()
        clean_text = re.sub(r'[*#_`]+', '', raw_text).strip()
        if clean_text:
            samples, sr = kokoro.create(clean_text, voice=TTS_VOICE, speed=1.1, lang="en-us")
            await websocket.send_text(f"LOG:AI: {clean_text}")
            await websocket.send_bytes(numpy_to_wav_bytes(samples, sr))

    # 3. Add AI Response to Full History
    history.append({"role": "assistant", "content": full_response})
    
    metrics['end_time'] = time.time()
    print_perf_report(metrics)
    
    await websocket.send_text("DONE")

# --- FRONTEND (Unchanged) ---
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Simon neural</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
    <style>
        body { background: #080808; color: #00ff41; font-family: 'Courier New', monospace; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        #console { 
            position: absolute; top: 20px; width: 90%; height: 200px; 
            font-size: 0.9rem; color: #00ff41; overflow-y: auto; text-align: left;
            border-bottom: 1px solid #222; padding-bottom: 10px;
            white-space: pre-wrap;
        }
        #status { margin-bottom: 40px; font-size: 1.2rem; font-weight: bold; letter-spacing: 3px; text-shadow: 0 0 10px #00ff41; }
        
        #mic-btn {
            width: 180px; height: 180px; border-radius: 50%; 
            background: #111; border: 2px solid #333;
            color: #555; font-size: 1.2rem; cursor: pointer; 
            display: flex; align-items: center; justify-content: center; 
            transition: all 0.2s;
            user-select: none; -webkit-tap-highlight-color: transparent;
        }
        
        #mic-btn.active { border-color: #00ff41; color: #00ff41; box-shadow: 0 0 40px rgba(0, 255, 65, 0.2); }
        #mic-btn.recording { background: #200; border-color: #ff0000; color: #ff0000; box-shadow: 0 0 50px rgba(255, 0, 0, 0.5); animation: pulse 1s infinite; }
        #mic-btn.processing { border-color: #fff; color: #fff; box-shadow: 0 0 20px #fff; }

        @keyframes pulse { 0% {transform: scale(1);} 50% {transform: scale(1.02);} 100% {transform: scale(1);} }
    </style>
</head>
<body>
    <div id="console">Initializing Neural Link...</div>
    <div id="status">DISCONNECTED</div>
    <div id="mic-btn">CONNECT</div>
    
    <script>
        const ws = new WebSocket(`wss://${window.location.host}/ws`);
        const btn = document.getElementById('mic-btn');
        const statusEl = document.getElementById('status');
        const consoleEl = document.getElementById('console');
        
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let audioStream = null; 
        const audioQueue = [];
        let isPlaying = false;

        function playNext() {
            if (audioQueue.length === 0) { isPlaying = false; checkIdle(); return; }
            isPlaying = true;
            const audioBlob = audioQueue.shift();
            const audio = new Audio(URL.createObjectURL(audioBlob));
            audio.onended = playNext;
            audio.play().catch(e => log("Playback error: " + e));
        }

        function queueAudio(blob) {
            audioQueue.push(blob);
            if (!isPlaying) playNext();
        }

        function checkIdle() {
             if (!isRecording && audioQueue.length === 0) {
                 statusEl.innerText = "READY";
                 btn.className = "active";
                 btn.innerText = "TALK";
             }
        }

        function log(msg) { 
            consoleEl.innerText += "\\n> " + msg;
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        ws.onopen = () => { statusEl.innerText = "NEURAL LINK ACTIVE"; btn.innerText = "TALK"; btn.classList.add('active'); log("System Online."); };
        ws.onclose = () => { statusEl.innerText = "OFFLINE"; btn.classList.remove('active'); log("Connection Lost."); };

        async function startRecording() {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
                mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });
                audioChunks = [];
                mediaRecorder.ondataavailable = event => { if (event.data.size > 0) audioChunks.push(event.data); };
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioStream.getTracks().forEach(track => track.stop());
                    if (ws.readyState === WebSocket.OPEN) {
                        statusEl.innerText = "PROCESSING";
                        btn.className = "processing";
                        btn.innerText = "•••";
                        ws.send(audioBlob);
                    }
                };
                mediaRecorder.start();
                isRecording = true;
                btn.className = "recording";
                btn.innerText = "STOP";
                statusEl.innerText = "LISTENING";
                log("Mic Active.");
            } catch (err) { log("Mic Error: " + err); }
        }

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!isRecording) startRecording();
            else {
                mediaRecorder.stop();
                isRecording = false;
            }
        });

        ws.onmessage = async (event) => {
            if (typeof event.data === "string") {
                const msg = event.data;
                if (msg.startsWith("LOG:")) {
                    log(msg.substring(4)); 
                } else if (msg === "DONE") { } else if (msg.startsWith("SYS:")) {
                    statusEl.innerText = msg.substring(4);
                }
            } else {
                statusEl.innerText = "SPEAKING";
                queueAudio(event.data);
            }
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_console("Client Connected - New Session Started", "NET")
    session_history = [] 
    
    try:
        while True:
            data = await websocket.receive_bytes()
            metrics = {
                'start_time': time.time(),
                'stt': 0, 'ctx': 0, 'ttft': 0, 
                'tts_first': 0, 'total_latency': 0,
                'input_tokens': 0, 'end_time': 0
            }

            audio_np = convert_webm_to_numpy(data)
            if audio_np is None: continue
            
            t_stt_start = time.time()
            segments, _ = stt_model.transcribe(audio_np, beam_size=1)
            user_text = " ".join([s.text for s in segments]).strip()
            metrics['stt'] = time.time() - t_stt_start

            if not user_text: 
                await websocket.send_text("DONE")
                continue

            await websocket.send_text(f"LOG:User: {user_text}")
            await process_and_stream_response(user_text, websocket, session_history, metrics)

    except Exception as e:
        log_console(f"Error: {e}", "ERR")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")