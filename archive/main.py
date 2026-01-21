import torch
import sounddevice as sd
import numpy as np
import time
from openai import OpenAI
from faster_whisper import WhisperModel

# --- CONFIG ---
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5     # Над това е говор
SILENCE_DURATION = 1.0  # Колко секунди тишина, за да приключим записа
MIN_RECORDING = 0.5     # Да не пращаме празни шумове
CHUNKS_PER_SEC = 32     # При 512 samples/chunk (32ms)

# --- SETUP ---
# 1. Load VAD (супер лек е)
model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=True) # M4 Pro обича ONNX
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model_vad)

# 2. Clients
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
stt_model = WhisperModel("medium.en", device="cpu", compute_type="int8")

# --- STATE ---
is_ai_speaking = False

def speak_response(text):
    global is_ai_speaking
    is_ai_speaking = True
    print(f"🤖 AI: {text}")
    
    # 1. Генериране на аудио (Тук викаш Kokoro/TTS)
    # audio = generate_tts(text)
    
    # 2. Playback (Blocking - чакаме да свърши!)
    # sd.play(audio, 24000)
    # sd.wait() 
    
    # Симулация за теста:
    time.sleep(len(text) * 0.05) 
    
    print("✅ AI done speaking.")
    is_ai_speaking = False
    # Ресетваме VAD-а, за да не "чуе" ехото като нова реч веднага
    vad_iterator.reset_states()

def main_loop():
    print("🎤 Mic Listening... (Silence threshold: 1.0s)")
    
    buffer = []
    silence_chunks = 0
    is_recording_speech = False
    
    # Callback за аудио стрима
    def callback(indata, frames, time, status):
        nonlocal silence_chunks, is_recording_speech, buffer
        
        # 1. ГЛУХ РЕЖИМ: Ако AI говори, игнорираме входа
        if is_ai_speaking:
            return

        # Convert to float32 for VAD
        audio_chunk = indata.flatten()
        
        # 2. VAD Check
        speech_prob = model_vad(torch.from_numpy(audio_chunk), SAMPLE_RATE).item()
        
        if speech_prob > VAD_THRESHOLD:
            is_recording_speech = True
            silence_chunks = 0
            buffer.extend(audio_chunk)
        elif is_recording_speech:
            # Вече сме почнали да записваме, но сега е тихо
            silence_chunks += 1
            buffer.extend(audio_chunk)
            
            # Проверка дали тишината е достатъчно дълга (1 сек)
            chunks_needed = int(SILENCE_DURATION * (SAMPLE_RATE / 512))
            if silence_chunks > chunks_needed:
                # КРАЙ НА ТУРН-а
                process_turn(np.array(buffer))
                # Ресет
                buffer = []
                is_recording_speech = False
                silence_chunks = 0

    # Start Stream
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=512):
        while True:
            sd.sleep(100) # Държи скрипта жив

def process_turn(audio_data):
    if len(audio_data) / SAMPLE_RATE < MIN_RECORDING:
        return # Твърде кратко, сигурно е шум
        
    print("Processing user audio...")
    
    # 1. Whisper Transcribe
    segments, _ = stt_model.transcribe(audio_data, beam_size=5)
    user_text = " ".join([s.text for s in segments]).strip()
    
    if not user_text: return
    print(f"👤 User: {user_text}")

    # 2. Check for Trigger Word (Optional refinement)
    # if "jarvis" not in user_text.lower(): return

    # 3. LLM Request
    stream = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": user_text}],
        stream=True
    )

    # 4. Stream & Speak
    # Тук събираш изречения и ги пращаш на speak_response()
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    
    speak_response(full_response)

if __name__ == "__main__":
    main_loop()