import asyncio
import json
import re
import threading
import time

import backend.db as db
import backend.memory_intents as memory_intents
import backend.tools as tools
from backend.audio import numpy_to_wav_bytes
from backend.config import (
    AGENT_MAX_TURNS,
    AGENT_TRIGGER_KEYWORDS,
    DEBUG_MODE,
    MAX_TOOL_OUTPUT_CHARS,
    QUIET_LOGS,
    TTS_VOICE,
)
from backend.metrics import (
    _metric_value,
    estimate_tokens_from_messages,
    estimate_tokens_from_text,
    finalize_metrics,
)

STREAM_FLUSH_CHARS = 24
STREAM_FLUSH_SECS = 0.05
EXPLICIT_MEMORY_MAX_CHARS = 400
SESSION_TITLE_MAX_CHARS = 80
SESSION_TITLE_CONTEXT_MAX_CHARS = 800


def _msg_to_dict(msg):
    if msg is None:
        return {}
    if isinstance(msg, dict):
        out = {"role": msg.get("role"), "content": msg.get("content")}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            out["tool_calls"] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    out["tool_calls"].append({
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                        "function": {
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        },
                    })
                else:
                    fn_obj = getattr(tc, "function", None)
                    out["tool_calls"].append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(fn_obj, "name", None) if fn_obj else None,
                            "arguments": getattr(fn_obj, "arguments", None) if fn_obj else None,
                        },
                    })
        return out

    out = {"role": getattr(msg, "role", None), "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = []
        for tc in tool_calls:
            fn_obj = getattr(tc, "function", None)
            out["tool_calls"].append({
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None),
                "function": {
                    "name": getattr(fn_obj, "name", None) if fn_obj else None,
                    "arguments": getattr(fn_obj, "arguments", None) if fn_obj else None,
                },
            })
    return out


def _print_perf_report(metrics):
    if QUIET_LOGS:
        return
    total_pipeline = metrics["end_time"] - metrics["start_time"]
    c_green = "\033[92m"
    c_end = "\033[0m"
    print(f"\n{c_green}--- RAG PERF REPORT ---{c_end}")
    print(
        f"   Decode: {_metric_value(metrics, 'audio_decode'):.3f}s | "
        f"STT: {_metric_value(metrics, 'stt'):.3f}s | "
        f"MEMORY: {_metric_value(metrics, 'rag'):.3f}s | "
        f"CTX: {_metric_value(metrics, 'ctx'):.3f}s"
    )
    print(
        f"   LLM TTFT: {_metric_value(metrics, 'ttft'):.3f}s | "
        f"LLM Total: {_metric_value(metrics, 'llm_total'):.3f}s"
    )
    print(
        f"   TTS First: {_metric_value(metrics, 'tts_first'):.3f}s | "
        f"TTS Total: {_metric_value(metrics, 'tts_total'):.3f}s"
    )
    print(
        f"   Tokens In: {_metric_value(metrics, 'input_tokens'):.0f} | "
        f"Tokens Out: {_metric_value(metrics, 'output_tokens'):.0f} | "
        f"TPS: {_metric_value(metrics, 'tokens_per_second'):.2f} | "
        f"Overhead: {_metric_value(metrics, 'overhead'):.3f}s"
    )
    print(f"   TOTAL: {total_pipeline:.3f}s")
    print(f"------------------------------------------\n")


def _format_memory_context(history, max_messages=6, max_chars=2000):
    if not history:
        return ""
    window = history[-max_messages:]
    lines = []
    for msg in window:
        role = (msg.get("role") or "unknown").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.title()}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _truncate_text(text, max_chars):
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _clean_session_title(title):
    if not title:
        return ""
    cleaned = title.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" -–—:;,.")
    if len(cleaned) > SESSION_TITLE_MAX_CHARS:
        cleaned = cleaned[:SESSION_TITLE_MAX_CHARS].rstrip()
    return cleaned


def _generate_session_title(user_text, ai_text, client, get_current_model):
    if client is None or get_current_model is None:
        return ""
    user_snip = _truncate_text(user_text, SESSION_TITLE_CONTEXT_MAX_CHARS)
    ai_snip = _truncate_text(ai_text, SESSION_TITLE_CONTEXT_MAX_CHARS)
    if not user_snip and not ai_snip:
        return ""
    try:
        response = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short session title (3-7 words) in the user's language. "
                        "No quotes, no trailing punctuation, no emojis. Output title only."
                    ),
                },
                {"role": "user", "content": f"USER:\n{user_snip}\n\nASSISTANT:\n{ai_snip}"},
            ],
            temperature=0.2,
            max_tokens=32,
        )
        title = (response.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not title:
        return ""
    if title.strip().upper() in {"NONE", "N/A"}:
        return ""
    return _clean_session_title(title)


def _maybe_set_session_title(session_id, user_text, ai_text, client, get_current_model):
    if not session_id:
        return
    try:
        meta = db.get_session_meta(session_id)
    except Exception:
        meta = None
    if meta and meta.get("title"):
        return
    title = _generate_session_title(user_text, ai_text, client, get_current_model)
    if not title:
        return
    try:
        db.set_session_title(session_id, title)
        if DEBUG_MODE and not QUIET_LOGS:
            print(f"[SESSION] Title set: {title}")
    except Exception as exc:
        if DEBUG_MODE and not QUIET_LOGS:
            print(f"[SESSION] Title update failed: {exc}")


def _summarize_explicit_memory(history, client, get_current_model):
    if client is None or get_current_model is None:
        return ""
    context = _format_memory_context(history)
    if not context:
        return ""
    try:
        response = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a single concise memory to store. "
                        "Keep only stable facts, preferences, or commitments worth recalling later. "
                        "Focus on what the user asked to remember; ignore assistant suggestions. "
                        "Use the user's language. If nothing should be saved, reply with NONE."
                    ),
                },
                {"role": "user", "content": f"CONTEXT:\n{context}"},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not summary:
        return ""
    if summary.strip().upper() in {"NONE", "NO", "N/A"}:
        return ""
    if summary.startswith("- "):
        summary = summary[2:].strip()
    if len(summary) > EXPLICIT_MEMORY_MAX_CHARS:
        summary = summary[:EXPLICIT_MEMORY_MAX_CHARS] + "..."
    return summary


def _save_explicit_memory(history, session_id, client, get_current_model, memory):
    summary = _summarize_explicit_memory(history, client, get_current_model)
    if not summary:
        return
    metadata = {
        "source": "explicit",
        "saved_by": "user_intent",
        "timestamp": time.time(),
    }
    try:
        memory.save_explicit(summary, session_id=session_id, metadata=metadata)
        if DEBUG_MODE and not QUIET_LOGS:
            print(f"[MEMORY] Saved explicit memory: {summary[:80]}")
    except Exception as exc:
        if DEBUG_MODE and not QUIET_LOGS:
            print(f"[MEMORY] Explicit save failed: {exc}")


async def process_and_stream_response(
    user_text,
    websocket,
    history,
    metrics,
    stop_event,
    session_id,
    generate_audio=True,
    *,
    client,
    kokoro,
    memory,
    build_rag_context,
    log_console,
    get_current_model,
):
    if memory is None:
        import backend.memory as memory_mod
        memory = memory_mod.memory

    t_ctx_start = time.time()
    is_deep_mode = any(k in user_text.lower() for k in AGENT_TRIGGER_KEYWORDS)

    current_messages = []
    rag_payload = []

    if is_deep_mode:
        log_console("ACTIVATING AGENTIC LOOP", "AGENT")
        await websocket.send_text("SYS:THINKING: Entering Deep Mode...")
        current_messages = [
            {"role": "system", "content": "You are Simon (Deep Mode). Use your tools to verify facts from memory/history before answering. Do not hallucinate."},
            *history[-10:],
            {"role": "user", "content": user_text}
        ]
        tools_list = tools.SIMON_TOOLS
    else:
        context_msgs, rag_payload = build_rag_context(user_text, history, memory, metrics, session_id)
        current_messages = context_msgs + [{"role": "user", "content": user_text}]
        tools_list = None

    metrics["ctx"] = time.time() - t_ctx_start
    metrics["input_chars"] = len(user_text)
    if metrics.get("input_tokens") is None:
        metrics["input_tokens"] = estimate_tokens_from_messages(current_messages)

    if rag_payload:
        await websocket.send_text(f"RAG:{json.dumps(rag_payload)}")

    emit_text_deltas = not generate_audio
    emit_final_text = True
    emit_tts_text = False

    q = asyncio.Queue(maxsize=64) if generate_audio else None
    response_holder = {"text": ""}
    sentence_endings = re.compile(r"[.!?]+")

    async def tts_consumer():
        first_audio_generated = False
        while True:
            if stop_event.is_set():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                break
            clean_text = item
            if not clean_text:
                continue

            wav_bytes = None
            if generate_audio:
                if metrics.get("tts_total") is None:
                    metrics["tts_total"] = 0
                tts_start = time.time()
                samples, sr = await asyncio.to_thread(
                    kokoro.create, clean_text, voice=TTS_VOICE, speed=1.0, lang="en-us"
                )
                if stop_event.is_set():
                    break
                wav_bytes = await asyncio.to_thread(numpy_to_wav_bytes, samples, sr)
                metrics["tts_total"] += time.time() - tts_start

                if not first_audio_generated:
                    metrics["tts_first"] = time.time() - metrics["start_time"]
                    first_audio_generated = True

            if not stop_event.is_set():
                if emit_tts_text:
                    await websocket.send_text(f"LOG:AI: {clean_text}")
                if generate_audio and wav_bytes:
                    await websocket.send_bytes(wav_bytes)

    def llm_producer_threadsafe(loop, stop_evt):
        try:
            model_name = get_current_model()
            log_console(f"Using model: {model_name} | Deep: {is_deep_mode}", "AI")
            llm_start = time.time()
            metrics["_llm_start"] = llm_start

            final_text_buffer = ""
            tool_chars_used = 0

            def enqueue_text(text):
                if q is None:
                    return
                def _do_put():
                    try:
                        q.put_nowait(text)
                    except asyncio.QueueFull:
                        if DEBUG_MODE:
                            log_console("TTS queue full; dropping chunk", "WARN")
                loop.call_soon_threadsafe(_do_put)

            def send_delta(text):
                if not emit_text_deltas or not text or stop_evt.is_set():
                    return
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(f"STREAM:AI:{text}"),
                    loop,
                )

            if is_deep_mode:
                turn_count = 0
                while turn_count < AGENT_MAX_TURNS and not stop_evt.is_set():
                    turn_count += 1

                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=current_messages,
                            temperature=0.7,
                            stream=False,
                            tools=tools_list,
                            tool_choice="auto"
                        )
                    except Exception as e:
                        print(f"Agent Loop Error: {e}")
                        break

                    msg_dict = _msg_to_dict(response.choices[0].message)
                    tool_calls = msg_dict.get("tool_calls") or []
                    if tool_calls:
                        current_messages.append(msg_dict)
                        first_tool = tool_calls[0] if tool_calls else {}
                        first_fn = (first_tool.get("function") or {}).get("name") or first_tool.get("name")
                        if first_fn:
                            log_console(f"Tool Call: {first_fn}", "AGENT")
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Consulting memory..."),
                            loop,
                        )

                        for tool_call in tool_calls:
                            fn = tool_call.get("function") or {}
                            fn_name = fn.get("name") or tool_call.get("name")
                            args = tools._safe_args(fn.get("arguments"))
                            result = "Error: Unknown tool"

                            if fn_name == "search_memory":
                                query = args.get("query", "")
                                scope = args.get("scope", "recent")
                                result = tools.tool_search_memory(
                                    query,
                                    scope,
                                    session_id,
                                    memory=memory,
                                    mem_conn=db.mem_conn,
                                    db_lock=db.db_lock,
                                )
                            elif fn_name == "analyze_deep_context":
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_text("SYS:THINKING: Deep reading transcript..."),
                                    loop,
                                )
                                target_session = args.get("session_id")
                                instruction = args.get("instruction", "")
                                if target_session is None or not instruction:
                                    result = "Error: analyze_deep_context requires session_id and instruction."
                                else:
                                    result = tools.tool_analyze_deep(
                                        client,
                                        get_current_model,
                                        target_session,
                                        instruction,
                                    )

                            result_str = str(result)
                            remain = MAX_TOOL_OUTPUT_CHARS - tool_chars_used
                            if remain <= 0:
                                result_str = "[TOOL_BUDGET_EXCEEDED]"
                            else:
                                result_str = result_str[:remain]
                            tool_chars_used += len(result_str)

                            tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                            tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
                            current_messages.append(tool_msg)
                    else:
                        break

            if not stop_evt.is_set():
                try:
                    final_messages = current_messages
                    if is_deep_mode:
                        final_messages = current_messages + [{
                            "role": "system",
                            "content": "Now produce the final answer for the user. Do not call tools."
                        }]

                    stream_start = time.time()
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=final_messages,
                        temperature=0.7,
                        stream=True
                    )

                    delta_buffer = ""
                    last_flush = time.monotonic()
                    current_sentence = ""
                    for chunk in stream:
                        if stop_evt.is_set():
                            break

                        choices = getattr(chunk, "choices", None)
                        if choices is None and isinstance(chunk, dict):
                            choices = chunk.get("choices")
                        if not choices:
                            continue
                        choice0 = choices[0]
                        delta = getattr(choice0, "delta", None)
                        if delta is None and isinstance(choice0, dict):
                            delta = choice0.get("delta")
                        if not delta:
                            continue
                        token = getattr(delta, "content", None)
                        if token is None and isinstance(delta, dict):
                            token = delta.get("content")
                        if not token:
                            continue
                        if metrics.get("ttft") is None:
                            metrics["ttft"] = time.time() - stream_start

                        final_text_buffer += token
                        if emit_text_deltas:
                            delta_buffer += token
                            now = time.monotonic()
                            if len(delta_buffer) >= STREAM_FLUSH_CHARS or (now - last_flush) >= STREAM_FLUSH_SECS:
                                send_delta(delta_buffer)
                                delta_buffer = ""
                                last_flush = now
                        if generate_audio:
                            current_sentence += token

                        if generate_audio and sentence_endings.search(current_sentence[-2:]) and len(current_sentence.strip()) > 5:
                            raw_t = current_sentence.strip()
                            clean_t = re.sub(r"[*#_`~]+", "", raw_t).strip()
                            if clean_t:
                                enqueue_text(clean_t)
                            current_sentence = ""

                    if emit_text_deltas and delta_buffer and not stop_evt.is_set():
                        send_delta(delta_buffer)
                    if generate_audio and current_sentence.strip() and not stop_evt.is_set():
                        raw_t = current_sentence.strip()
                        clean_t = re.sub(r"[*#_`~]+", "", raw_t).strip()
                        if clean_t:
                            enqueue_text(clean_t)

                except Exception as e:
                    print(f"Streaming Error: {e}")

            response_holder["text"] = final_text_buffer
            if session_id and user_text and final_text_buffer and len(history) == 0:
                threading.Thread(
                    target=_maybe_set_session_title,
                    args=(session_id, user_text, final_text_buffer, client, get_current_model),
                    daemon=True,
                ).start()

        except Exception as e:
            print(f"LLM Error: {e}")
        finally:
            if metrics.get("_llm_start"):
                metrics["llm_total"] = time.time() - metrics["_llm_start"]
            if q is not None:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

    loop = asyncio.get_running_loop()
    consumer_task = asyncio.create_task(tts_consumer()) if generate_audio else None
    producer_task = asyncio.create_task(asyncio.to_thread(llm_producer_threadsafe, loop, stop_event))

    try:
        if consumer_task is not None:
            await asyncio.gather(producer_task, consumer_task)
        else:
            await producer_task
    except asyncio.CancelledError:
        stop_event.set()

    if not stop_event.is_set():
        full_reply = response_holder["text"]
        if full_reply:
            metrics["output_chars"] = len(full_reply)
            if metrics.get("output_tokens") is None:
                metrics["output_tokens"] = estimate_tokens_from_text(full_reply)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": full_reply})

            if memory_intents.detect_memory_save(user_text):
                history_snapshot = list(history)
                threading.Thread(
                    target=_save_explicit_memory,
                    args=(history_snapshot, session_id, client, get_current_model, memory),
                ).start()
            threading.Thread(target=db.save_interaction, args=(session_id, user_text, full_reply)).start()

            metrics["end_time"] = time.time()
            _print_perf_report(metrics)
            finalize_metrics(metrics, "ok")
            if emit_final_text:
                await websocket.send_text(f"LOG:AI: {full_reply}")
            await websocket.send_text("DONE")
        else:
            metrics["end_time"] = time.time()
            finalize_metrics(metrics, "empty_reply")
            await websocket.send_text("DONE")
    else:
        metrics["end_time"] = time.time()
        finalize_metrics(metrics, "aborted")
        await websocket.send_text("LOG: --- ABORTED ---")


__all__ = ["process_and_stream_response"]
