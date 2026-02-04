import asyncio
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Optional, Tuple

import numpy as np


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _merge_with_overlap(base: str, addition: str, max_words: int) -> str:
    base = _normalize_text(base)
    addition = _normalize_text(addition)
    if not base:
        return addition
    if not addition:
        return base
    base_words = base.split()
    add_words = addition.split()
    max_overlap = min(len(base_words), len(add_words), max_words)
    for overlap in range(max_overlap, 0, -1):
        if base_words[-overlap:] == add_words[:overlap]:
            tail = " ".join(add_words[overlap:])
            return base if not tail else f"{base} {tail}"
    return f"{base} {addition}"


class LiveTranscriber:
    def __init__(
        self,
        stt_model,
        send_event: Callable[[dict], Awaitable[None]],
        sample_rate: int,
        window_s: float,
        step_s: float,
        keep_s: float,
        min_window_s: float = 1.0,
        overlap_words: int = 12,
        log_fn: Optional[Callable[[str, str], None]] = None,
        on_first_partial: Optional[Callable[[], None]] = None,
    ) -> None:
        self._stt_model = stt_model
        self._send_event = send_event
        self._log_fn = log_fn
        self._on_first_partial = on_first_partial

        self._sample_rate = sample_rate
        self._window_s = max(1e-3, window_s)
        self._step_s = max(1e-3, step_s)
        self._keep_s = max(0.0, keep_s)
        self._min_samples = int(max(1e-3, min_window_s) * sample_rate)
        self._max_samples = int(self._window_s * sample_rate)
        self._overlap_words = max(0, overlap_words)

        self._chunks: Deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._total_samples = 0
        self._last_schedule = 0.0

        self._stable_text = ""
        self._draft_text = ""
        self._stable_end = 0.0
        self._last_sent_stable = None
        self._last_sent_draft = None
        self._first_partial_sent = False

        self._active = False
        self._version = 0
        self._pending_snapshot: Optional[Tuple[int, np.ndarray, float, float]] = None
        self._inflight_task: Optional[asyncio.Task] = None

    def reset(self) -> None:
        self._version += 1
        self._active = True
        self._chunks.clear()
        self._ring_samples = 0
        self._total_samples = 0
        self._last_schedule = 0.0
        self._stable_text = ""
        self._draft_text = ""
        self._stable_end = 0.0
        self._last_sent_stable = None
        self._last_sent_draft = None
        self._first_partial_sent = False
        self._pending_snapshot = None

    def stop(self) -> None:
        self._active = False
        self._version += 1
        self._pending_snapshot = None

    async def send_reset(self) -> None:
        await self._emit({"event": "reset"})

    async def send_final(self, text: str) -> None:
        await self._emit({"event": "final", "text": text})

    def add_audio(self, pcm: np.ndarray) -> None:
        if not self._active:
            return
        if pcm is None or pcm.size == 0:
            return
        chunk = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return

        self._chunks.append(chunk)
        self._ring_samples += chunk.size
        self._total_samples += chunk.size
        while self._ring_samples > self._max_samples:
            dropped = self._chunks.popleft()
            self._ring_samples -= dropped.size

        now = time.monotonic()
        if now - self._last_schedule >= self._step_s:
            self._last_schedule = now
            snapshot = self._snapshot()
            if snapshot is not None:
                self._schedule(snapshot)

    def _snapshot(self) -> Optional[Tuple[int, np.ndarray, float, float]]:
        if self._ring_samples < self._min_samples:
            return None
        try:
            audio_np = np.concatenate(list(self._chunks))
        except Exception:
            return None
        if audio_np.size == 0:
            return None
        window_end = self._total_samples / float(self._sample_rate)
        window_start = window_end - (audio_np.size / float(self._sample_rate))
        return (self._version, audio_np, window_start, window_end)

    def _schedule(self, snapshot: Tuple[int, np.ndarray, float, float]) -> None:
        if self._inflight_task and not self._inflight_task.done():
            self._pending_snapshot = snapshot
            return
        self._inflight_task = asyncio.create_task(self._run_snapshot(snapshot))

    async def _run_snapshot(self, snapshot: Tuple[int, np.ndarray, float, float]) -> None:
        version, audio_np, window_start, window_end = snapshot
        if not self._active or version != self._version:
            return

        try:
            def run_transcribe():
                segs, _ = self._stt_model.transcribe(
                    audio_np,
                    beam_size=3,
                    vad_filter=False,
                    language="en",
                    initial_prompt="Use context.",
                )
                return list(segs)

            segments = await asyncio.to_thread(run_transcribe)
        except Exception as exc:
            self._log(f"Live STT failed: {exc}", "WARN")
            segments = []

        if not self._active or version != self._version:
            return

        cutoff = window_end - self._keep_s
        stable_additions = []
        draft_parts = []
        stable_end = self._stable_end

        for seg in segments:
            text = _normalize_text(getattr(seg, "text", ""))
            if not text:
                continue
            abs_start = window_start + float(getattr(seg, "start", 0.0))
            abs_end = window_start + float(getattr(seg, "end", 0.0))
            if abs_end <= cutoff:
                if abs_end > stable_end + 1e-3:
                    stable_additions.append(text)
                    stable_end = abs_end
            else:
                if abs_start < cutoff and abs_end > cutoff:
                    # Keep split segment in draft to allow revisions in the keep zone.
                    draft_parts.append(text)
                else:
                    draft_parts.append(text)

        if stable_additions:
            new_stable = " ".join(stable_additions)
            if self._overlap_words > 0:
                self._stable_text = _merge_with_overlap(self._stable_text, new_stable, self._overlap_words)
            else:
                self._stable_text = _normalize_text(f"{self._stable_text} {new_stable}")
            self._stable_end = stable_end

        draft_text = _normalize_text(" ".join(draft_parts))

        if self._stable_text != self._last_sent_stable or draft_text != self._last_sent_draft:
            self._draft_text = draft_text
            await self._emit(
                {
                    "event": "partial",
                    "stable": self._stable_text,
                    "draft": self._draft_text,
                }
            )
            self._last_sent_stable = self._stable_text
            self._last_sent_draft = self._draft_text
            if not self._first_partial_sent:
                self._first_partial_sent = True
                if self._on_first_partial:
                    self._on_first_partial()

        if self._pending_snapshot:
            next_snapshot = self._pending_snapshot
            self._pending_snapshot = None
            if self._active:
                self._schedule(next_snapshot)

    async def _emit(self, payload: dict) -> None:
        try:
            await self._send_event(payload)
        except Exception as exc:
            self._log(f"Live STT send failed: {exc}", "WARN")

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self._log_fn:
            try:
                self._log_fn(msg, level)
            except Exception:
                pass
