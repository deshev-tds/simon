import io

import av
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from backend.config import ESP_AUDIO_DIR, ESP_AUDIO_MAX_FILES, SAMPLE_RATE


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
    sf.write(buffer, audio_np, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def _convert_to_mp3_bytes(samples: np.ndarray, sample_rate: int):
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


__all__ = [
    "_ensure_frames_list",
    "convert_webm_to_numpy",
    "numpy_to_wav_bytes",
    "_convert_to_mp3_bytes",
    "_trim_old_esp_audio",
    "_analyze_audio",
]
