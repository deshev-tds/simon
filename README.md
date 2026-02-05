# Simon: A Recursive, Agentic Voice Assistant

### *Implementation of MIT's "Recursive Language Models" (arXiv:2512.24601)*

**Version:** 2.0 (Refactored for Recursive Agency)
**Author:** [Damyan Deshev]
**Date:** January 2026

---

## 1. Executive Summary

This project is a " fully local, very smart etc. voice assistant"; but is is a practical, rigorous implementation of the **Recursive Language Model (RLM)** paradigm proposed in MIT's recent paper (arXiv:2512.24601).

While traditional Voice AI pipelines function linearly (Listen -> Transcribe -> Retrieve -> Generate -> Speak), "Simon" introduces a **bi-phasic cognitive architecture**. By decoupling the **Thinking Phase** (recursion, tool use, verification) from the **Speaking Phase** (streaming synthesis), this backend allows a local, optimized Large Language Model (LLM) to outperform significantly larger models in reasoning tasks.

This refactor transforms the system from a stochastic parrot into a verified reasoning engine, capable of self-correction and deep analysis before a single byte of audio reaches the user.

---

## 2. The Theoretical Shift: Why Refactor?

The primary motivation for rewriting `server.py` was to move away from "One-Shot RAG" toward "Recursive RAG."

### The Old Way (`server.old.py`)

The legacy system used a standard linear chain:

1. **Input:** User speaks.
2. **Context:** Vector database performs a similarity search.
3. **Generation:** The LLM immediately streams a response based on that context.

**The Flaw:** As noted in the RLM paper, one-shot generation forces the model to hallucinate if the immediate context is imperfect. It cannot "change its mind" or "double-check" a fact once the token stream begins.

### The New Way (`server.py`)

The new architecture implements the **Recursive Controller**. When the user asks a complex question (detected via `AGENT_TRIGGER_KEYWORDS`), the system enters a **Silent Reasoning Loop**.

1. **Recursion:** The model enters a `while` loop (capped at `AGENT_MAX_TURNS`).
2. **Tool Use:** Instead of speaking, the model calls tools (`search_memory`, `analyze_deep_context`).
3. **Observation:** The tool output is fed back into the context window.
4. **Convergence:** Only when the model decides it has sufficient information does it exit the loop and trigger the `generate_audio` stream.

This effectively trades **latency** (milliseconds of silence) for **accuracy** (verified facts), a core tenet of the RLM approach.

---

## 3. Key Architectural Features

### A. Hybrid Memory System (Vectors + FTS5)

*Located in: `MemoryManager` & `fts_search_messages*`

Vector search (ChromaDB) is excellent for semantic meaning ("concepts about projects"), but terrible at specific keyword recall ("what did I say about 'Project Fenrir' specifically?").

To solve this, I implemented a **Dual-Index Memory**:

* **Semantic Layer (ChromaDB):** Retrieves concept-adjacent memories.
* **Exact Layer (SQLite FTS5):** A full-text search engine embedded directly in the message history.

**Why this matters:** The Recursive Agent can now use the `search_memory` tool to perform precise lookups. If the user asks, "What was the error code?", the vector search might fail, but the FTS5 layer will catch "Error 500" via token matching. This is the difference between a "chatty" assistant and a "competent" one.

#### Tiered Storage (Hot / Warm / Cold)

Simon treats memory as tiers rather than a single in-RAM pool:

* **Hot (memdb / RAM):** Only the *current* session, capped to the last N messages (`SIMON_MEM_HOT_SESSION_LIMIT`). This keeps the hot path fast and bounded.
* **Warm (history.db / disk):** Full session history across all chats (SQLite + FTS5). **Primary** source for deep recall outside the prompt window.
* **Cold (corpus.db / disk):** Large documents/knowledge base stored separately with FTS5. Queried only via tools; never loaded into RAM wholesale.

**Implication:** `search_memory(scope="global")` queries disk FTS over **both** Warm (chat history) and Cold (corpus) tiers. Memdb is *not* a deep-history cache.

#### Archive Memory (ChatGPT history)

Simon can ingest a ChatGPT export into a separate vector collection (`chatgpt_archive`) and the SQL session store.
Archive recall is explicit-only (memory intent) to avoid polluting current context. Intent detection is configurable in `backend/memory_intents.py`.

**Recall anchors (examples):**
- English: "do you remember ...", "have we talked about ...", "remember when ..."
- Bulgarian: "помниш ли ...", "спомняш ли си ...", "говорихме ли ..."
- Prefixes: `/archive:`, `/memory:`

When archive recall is triggered, Simon uses the archive vector index to narrow semantically relevant sessions and then pulls exact SQL excerpts via FTS. This keeps recall grounded while still handling fuzzy queries.

* **Explicit recall ("remember" intent):** Controlled by `SIMON_ARCHIVE_EXPLICIT_THRESHOLD`.
* **Implicit recall (weak local):** Disabled for now; `SIMON_ARCHIVE_STRONG_THRESHOLD` and `SIMON_ARCHIVE_WEAK_MAX_AGE_DAYS` are reserved.

**Implication:** Raising the explicit threshold will surface more archive noise; keeping it strict preserves the current-session focus.

**Auto-update (optional):**
- On startup: set `SIMON_ARCHIVE_AUTO_IMPORT=1` and `SIMON_ARCHIVE_JSON_PATH=/path/to/conversations.json` (uses `--since-last` by default; set `SIMON_ARCHIVE_SINCE_LAST=0` to disable).
- Scheduled: `python scripts/import_chatgpt_archive.py --json /path/to/conversations.json --since-last` during quiet hours.

#### Explicit Memories (User-saved)

Simon only writes to the `explicit_memories` collection when the user asks to save a memory. A short LLM pass extracts a concise, stable fact or preference before saving.
Intent patterns for recall/save are configurable in `backend/memory_intents.py`.
Remote embeddings are **opt-in**: set `SIMON_EMBEDDINGS_REMOTE_ALLOWED=1` to allow remote model downloads; otherwise embeddings run in local-only mode.

**Save anchors (examples):**
- English: "remember this", "save this", "make a note"
- Bulgarian: "запомни това", "запомни го", "запиши си"

This collection powers spontaneous similarity recall in normal chats.

### B. The Surgical Agent Loop

*Located in: `process_and_stream_response*`

The refactor introduces a bifurcated pipeline:

* **Fast Path:** For casual chatter ("Hello", "How are you?"), the system bypasses the agentic overhead for instant response.
* **Deep Path:** Triggered by keywords (e.g., "analyze", "deep dive"). This activates the `llm_producer_threadsafe` logic which handles the **Think-Act-Observe** cycle.

**Code Highlight:**
The loop logic explicitly separates "thinking tokens" from "speaking tokens." The user sees "SYS:THINKING" updates on the frontend while the backend is aggressively querying the database, effectively giving the user a window into the machine's "internal monologue" without cluttering the audio stream.

### C. RLM-lite Gate + Evidence Discipline (Context Debt + Retrieval Confidence)

Simon uses an explicit gate to decide when to pay the latency cost of deep/recursive reasoning. The goal is to **avoid** expensive recursion on trivial queries, but **escalate** when the relevant facts likely live outside the current prompt window.

**Signals:**
- **Context debt:** estimated session tokens ÷ window tokens.
- **Recall / complexity intent:** explicit recall phrases (e.g., `/memory:`, `/archive:`, "remember from previous messages") force Deep Mode. Softer recall phrasing only escalates when retrieval is weak, to avoid false positives.
- **Retrieval confidence:** weak FTS (bm25) and/or zero hits; vector is used only as a fallback when FTS is empty/weak.
- **Recent window short-circuit:** if the query overlaps heavily with the last N turns, skip deep mode.
- **Evidence-bound answers:** in deep mode, final answers must be grounded in tool evidence; if no evidence exists, the system replies `not found`. When the query asks for a single value (code/date/amount/email/phone/name/city/formula), the system extracts that value deterministically from evidence lines.

**Evidence layer (unified):**
- Tool output is normalized into a single schema for FTS + vector + corpus hits: `source_type`, `doc_id`, `ts`, `text`, `score/dist`, `entity_hits`.
- The extractor/validator runs once over this shared schema (no special-case branches).
- **Conflict policy:** when multiple matches exist, extraction prefers the most recent evidence by timestamp (recency wins).

**Rule of thumb:**
- Explicit intent → deep mode.
- Explicit recall → deep mode.
- Soft recall + weak retrieval → deep mode.
- Complex + (high debt or weak retrieval) → deep mode.
- High debt override if the user is asking about the past.

**Env knobs:**
- `SIMON_RLM_ENABLED` (default `1`)
- `SIMON_RLM_FALLBACK_ENABLED` (default `1`)
- `SIMON_RLM_MAX_DEBT_RATIO` (default `2.5`)
- `SIMON_RLM_MIN_DEBT_FOR_CHECK` (default `1.2`)
- `SIMON_RLM_RECENT_WINDOW_TURNS` (default `6`)
- `SIMON_RLM_MIN_QUERY_LEN` (default `20`)
- `SIMON_RLM_VECTOR_WEAK_DIST` (default `0.55`)
- `SIMON_RLM_MIN_FTS_HITS` (default `1`)
- `SIMON_RLM_FTS_WEAK_SCORE` (default `-1.0`)
- `SIMON_RLM_MAX_HOPS` (default `3`)
- `SIMON_RLM_DEEP_HISTORY_MAX_MSGS` (default `2`) and `SIMON_RLM_DEEP_HISTORY_MAX_CHARS` (default `600`) to keep Deep Mode context surgical (avoid filler pollution).
- `SIMON_RLM_STREAM` (default `1`) to stream Deep Mode final answers; set `0` for non-stream (debug-friendly) completions.
- `SIMON_RLM_TRACE` (default `0`) to emit `[TRACE]` logs for gate, search, evidence, and fallbacks.

### D. Context Budgeting & Token Management

*Located in: `tool_search_memory` & `MAX_TOOL_OUTPUT_CHARS*`

Recursive models suffer from "Context Explosion." If an agent queries a document and gets 50,000 tokens back, the context window overflows, and the model crashes.

**The Fix:** I implemented strict output capping.
`if len(full_resp) > MAX_TOOL_OUTPUT_CHARS: return full_resp[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"`

This forces the agent to be efficient. If it needs more info, it must refine its search query (recursion) rather than dumping the whole database into RAM.

**Default prompt window (tunable):**
- `ANCHOR_MESSAGES=10`, `MAX_RECENT_MESSAGES=10` (bump up/down based on memory and latency budget).

### E. Vision + Image Messaging (Base64, No Uploads)

Simon now supports **multi-image + prompt** messages for vision-capable models (e.g., Qwen3-VL GGUF). The frontend can capture from camera / gallery / files, shows previews, and sends images as **base64** with the prompt. The backend stores images on disk and injects them into the LLM as multimodal `image_url` parts.

**Transport (WebSocket, primary):**
```json
{
  "type": "chat",
  "prompt": "Describe these photos",
  "images": [
    {
      "mime": "image/jpeg",
      "data_b64": "<base64>",
      "width": 1280,
      "height": 720,
      "size_bytes": 345678
    }
  ]
}
```

**REST fallback:** `POST /v1/chat/vision` with the same payload (plus optional `session_id`).

**Storage:** images are saved to `DATA_DIR/images/{session_id}/{message_id}_{idx}.jpg` and linked via the `message_attachments` table. Session history returns attachments (as base64) so the chat UI can re-render past images.

**Limits (configurable):**
- `SIMON_MAX_IMAGES_PER_MESSAGE` (default `10`)
- `SIMON_MAX_IMAGE_MB` (default `8`)
- `SIMON_MAX_IMAGE_EDGE` (default `2048`)

**Model input format:** the user message becomes a multimodal `content` array with `text` + `image_url` data URLs, which llama.cpp and OpenAI-compatible backends accept.

---

## 4. Resilience Engineering: Preemptive Fixes

In a one-man project, there is no QA team. The code must be defensive by design. Here are three specific scenarios where I anticipated failure and engineered a solution:

### Scenario 1: The "Echo Chamber" Effect

**The Risk:** In a recursive loop, an agent might save its own "thoughts" into long-term memory. If it thinks about a topic for 5 turns, it might save 5 near-identical memories, polluting the vector space.
**The Fix:** **Atomic Deduplication.**
In `MemoryManager.save`, I added a check that queries the vector DB for the *exact text* being saved before writing.
`if dists and dists[0] < 0.2: ... Skipping save.`
This ensures that long-term memory only stores unique information, keeping retrieval sharp.

### Scenario 2: Thread Race Conditions

**The Risk:** `FastAPI` (async) + `Whisper` (CPU bound) + `SQLite` (File I/O) is a recipe for database lock errors.
**The Fix:** **Global Locking Strategy.**
I introduced granular locks (`db_lock`, `METRICS_LOCK`, `self.lock`) wrapping every stateful operation. Specifically, SQLite operations are wrapped in `with db_lock:` to force serialization, preventing the dreaded "Database is locked" error during high-concurrency voice/text interleave.

### Scenario 3: The "Silent Failure" of MP3 Conversion

**The Risk:** The ESP32 client prefers MP3, but `pydub`/`ffmpeg` conversion can fail if system codecs are missing or the audio buffer is malformed.
**The Fix:** **Graceful Fallback Protocols.**
The TTS endpoint (`speech_endpoint`) attempts MP3 conversion first. If it throws an exception, it *silently* catches it, logs the error, and falls back to raw WAV format (which is larger but universally compatible). The user never experiences a failed request, only a slightly larger payload.

---

## 5. Technical Stack

* **Inference:** `faster_whisper` (Int8 Quantization), Local LLM (via LM Studio/OpenAI API standards), `Kokoro-ONNX` (Real-time TTS).
* **Backend:** FastAPI, Python 3.10+, AsyncIO.
* **Data:** ChromaDB (Vector), SQLite (Relational + FTS5).
* **Corpus:** Separate SQLite FTS DB (`corpus.db`) for long-form documents.
* **Protocol:** WebSockets for real-time bi-directional audio streaming.

---

## 6. Testing (Unit + Integration)

Unit tests exercise the API, WebSocket flows, **hot-session memdb sync**, and FTS retrieval. Integration tests run against a live LM Studio server (no LLM mocks) and verify that a "needle" token can be recalled via disk FTS and confirmed by the model response. If LM Studio has only one model loaded, you can leave `SIMON_DEFAULT_LLM_MODEL` empty to let it pick the default.

**Methodology / Principles:**
* **Fast by default:** Unit tests skip vector memory and audio model loads to keep the feedback loop tight.
* **Isolation:** Test data uses a temp data dir and cleans SQLite/memdb state between tests.
* **Causality over magic:** The needle test asserts that the needle is **not** present in the prompt window and **is** present in the RAG payload, then expects the model to echo the exact token.
* **Randomized collision checks:** The needle/marker are UUID-based to avoid accidental matches with persistent stores; the test asserts absence before seeding.
* **Robustness via decoys:** The needle test seeds near-collision markers (mutated) and a same-marker/wrong-token decoy to validate ranking behavior.

**Rationale:**
Unit tests are optimized for fast, deterministic feedback on core logic (API, FTS, memdb). Integration tests are the realism check: they hit a live LM Studio model and verify end-to-end recall behavior. Vector memory is opt-in because it adds external dependencies and variance.

**Test parameters (defaults in `tests/conftest.py`):**
* `SIMON_TEST_USE_FAKE_LLM=1` to force a fake LLM client. Default is **real LLM** for all tests, so LM Studio should be running unless you opt into fake mode.
* `SIMON_SKIP_VECTOR_MEMORY=1` to skip Chroma. Set to `0` to enable vector memory tests.
* `SIMON_SKIP_AUDIO_MODELS=1` to avoid loading TTS/STT in unit tests.
* `SIMON_EMBEDDINGS_REMOTE_ALLOWED=1` if vector memory needs remote embedding model downloads.
* `SIMON_DEFAULT_LLM_MODEL=""` to let LM Studio choose the only loaded model.
* `SIMON_MEM_HOT_SESSION_LIMIT` to cap the in-RAM hot window per session.
* `SIMON_MEM_SEED_LIMIT`, `SIMON_MEM_MAX_ROWS`, `SIMON_MEM_PRUNE_INTERVAL_S` (legacy) if you intentionally enable background memdb seed/prune threads.
* `TOKENIZERS_PARALLELISM=false` is set in tests to avoid fork warnings.
* `SIMON_KEEP_PERMANENT_LOGS=1` to keep logs across restarts (default `0` clears logs on the next `stack.sh` start).

**Runtime logging (stack.sh):**
- Logs live in `.simon-run/` (`backend.log`, `backend_http.log`, `frontend.log`).
- Default behavior: logs are **cleared on next start** (session-only logs).
- Set `SIMON_KEEP_PERMANENT_LOGS=1` to append and keep history across restarts.

**Run the suites:**
* `tests/run_tests.sh` (unit)
* `tests/run_tests.sh --integration` (integration, requires LM Studio at `http://localhost:1234/v1`)
* `tests/run_tests.sh --all`

**Demo (large corpus bridge):**
* `python scripts/demo_rlm_bridge.py --fresh --tokens-per-stage 500000 --probe`
* Options: `--force-gate` (force deep mode), `--seed-vectors` (seed vector DB with facts only), `--no-cleanup` (keep /tmp data).
* The demo prints **retrieval timing** for the live query and shows evidence lines used for extraction.
* Use `scripts/gen_noise_sentences.py` to generate realistic EN/BG filler text for UI testing.

**Needle recall integration test (excerpt):**

```python
# tests/integration/test_memdb_needle.py
needle = f"TOKEN_{uuid.uuid4().hex}"
marker = f"MARKER_{uuid.uuid4().hex}"

# assert absence in DB/Chroma, then seed 50 turns with decoys

ws.send_text(
    f"What token was paired with {marker}? "
    "Read the recalled evidence and reply with the token only."
)
messages = _recv_until_done(ws)

assert any(needle in line for line in preview_lines)  # memdb/FTS recall via RAG payload
assert ai_text.strip() == needle  # model confirms the recalled token
```

---

## 7. Conclusion

This project demonstrates that enterprise-grade AI architecture-specifically the **Recursive Agentic patterns** utilized by top research labs-can be successfully implemented in a resource-constrained, local environment. By carefully managing context, enforcing thread safety, and enabling the model to "think" before it speaks, Simon represents the next generation of personal voice assistants.

*Ready for 2026.*
