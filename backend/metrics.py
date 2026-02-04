import math
import threading
import time
import uuid
from collections import deque

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
        "record_start": None,
        "record_end": None,
        "first_partial": None,
        "partial_latency": None,
        "final_transcript_ready": None,
        "final_latency": None,
        "stt_rtf": None,
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


__all__ = [
    "METRICS_HISTORY",
    "METRICS_LOCK",
    "count_tokens",
    "init_metrics",
    "_metric_value",
    "estimate_tokens_from_text",
    "estimate_tokens_from_messages",
    "finalize_metrics",
]
