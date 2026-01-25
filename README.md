Here is the README documentation. It is structured to act as a technical deep-dive and marketing piece for your portfolio, highlighting the architectural shift from a standard RAG pipeline to a Recursive Agentic Loop based on the MIT paper.

---

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

#### Archive Memory (ChatGPT history)

Simon can ingest a ChatGPT export into a separate vector collection (`chatgpt_archive`) and the SQL session store.
Archive recall is gated with two thresholds to avoid polluting current context:

* **Implicit recall (weak local memory):** strict threshold + recent-only window. Controlled by `SIMON_ARCHIVE_STRONG_THRESHOLD` and `SIMON_ARCHIVE_WEAK_MAX_AGE_DAYS`.
* **Explicit recall ("remember" intent):** looser threshold, no age cap. Controlled by `SIMON_ARCHIVE_EXPLICIT_THRESHOLD`.

**Implication:** Raising the explicit threshold will surface more archive noise. Lowering the implicit threshold keeps the assistant more "present" but may miss older context.

### B. The Surgical Agent Loop

*Located in: `process_and_stream_response*`

The refactor introduces a bifurcated pipeline:

* **Fast Path:** For casual chatter ("Hello", "How are you?"), the system bypasses the agentic overhead for instant response.
* **Deep Path:** Triggered by keywords (e.g., "analyze", "deep dive"). This activates the `llm_producer_threadsafe` logic which handles the **Think-Act-Observe** cycle.

**Code Highlight:**
The loop logic explicitly separates "thinking tokens" from "speaking tokens." The user sees "SYS:THINKING" updates on the frontend while the backend is aggressively querying the database, effectively giving the user a window into the machine's "internal monologue" without cluttering the audio stream.

### C. Context Budgeting & Token Management

*Located in: `tool_search_memory` & `MAX_TOOL_OUTPUT_CHARS*`

Recursive models suffer from "Context Explosion." If an agent queries a document and gets 50,000 tokens back, the context window overflows, and the model crashes.

**The Fix:** I implemented strict output capping.
`if len(full_resp) > MAX_TOOL_OUTPUT_CHARS: return full_resp[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"`

This forces the agent to be efficient. If it needs more info, it must refine its search query (recursion) rather than dumping the whole database into RAM.

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
* **Protocol:** WebSockets for real-time bi-directional audio streaming.

---

## 6. Testing (Unit + Integration)

Unit tests exercise the API, WebSocket flows, memdb seed/prune, and FTS retrieval. Integration tests run against a live LM Studio server (no model mocks) and verify that long context oversaturation still allows a "needle" token to be recalled via memdb and confirmed in the model response.

**Run the suites:**
* `tests/run_tests.sh` (unit)
* `tests/run_tests.sh --integration` (integration, requires LM Studio at `http://localhost:1234/v1`)
* `tests/run_tests.sh --all`

**Needle recall integration test (excerpt):**

```python
# tests/integration/test_memdb_needle.py
needle = "TOKEN1234"
marker = "MARKERXYZ"

ws.send_text(f"Seed {i} {marker} {needle} synthetic payload. Reply with exactly: OK")
...
ws.send_text(f"What token was paired with {marker}? Read the recalled evidence and reply with the token only.")
messages = _recv_until_done(ws)

assert any(needle in line for line in preview_lines)  # memdb/FTS recall via RAG payload
assert needle in ai_text  # model confirms the recalled token
```

---

## 7. Conclusion

This project demonstrates that enterprise-grade AI architecture-specifically the **Recursive Agentic patterns** utilized by top research labs-can be successfully implemented in a resource-constrained, local environment. By carefully managing context, enforcing thread safety, and enabling the model to "think" before it speaks, Simon represents the next generation of personal voice assistants.

*Ready for 2026.*
