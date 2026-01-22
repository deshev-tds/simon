import json
import time

import backend.db as db
import backend.memory as memory_mod
from backend.config import MAX_TOOL_OUTPUT_CHARS

SIMON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search long-term memory (vectors) and chat history (FTS). Use to find facts, past conversations, or project details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "recent", "session"],
                        "description": "Scope: 'global' (all time), 'recent' (60 days), 'session' (current chat)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_deep_context",
            "description": "Reads a past session transcript. Use ONLY for complex analysis of a specific known session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The ID of the session to analyze."
                    },
                    "instruction": {
                        "type": "string",
                        "description": "What to extract/summarize."
                    }
                },
                "required": ["session_id", "instruction"]
            }
        }
    }
]


def _safe_args(raw_args):
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except Exception:
        return {"_raw": str(raw_args)}


def tool_search_memory(
    query: str,
    scope: str = "recent",
    session_id: int | None = None,
    memory=None,
    mem_conn=None,
    db_lock=None,
):
    days = 60 if scope == "recent" else None
    sess_filter = session_id if scope == "session" else None
    cutoff_ts = time.time() - (60 * 24 * 3600) if scope == "recent" else None

    mem = memory or memory_mod.memory
    conn = mem_conn or db.mem_conn
    lock = db_lock or db.db_lock

    docs, dists, metas = mem.search(query, n_results=3, days_filter=days, session_filter=sess_filter)

    fts_res = []
    if scope in ["recent", "session"]:
        target_sess = session_id if scope == "session" else None
        fts_res = db.fts_recursive_search(
            query,
            session_id=target_sess,
            limit=3,
            cutoff_ts=cutoff_ts,
            conn=conn,
            lock=lock,
        )

    results = []
    for doc, meta in zip(docs, metas):
        sess = meta.get("session_id", "?")
        ts = meta.get("timestamp", 0)
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        results.append(f"[VECTOR] (Sess:{sess}, {date_str}): {doc[:150]}...")

    for item in fts_res:
        role = item["role"]
        content = item["content"].replace("\n", " ")
        sess = item.get("session_id", "?")
        results.append(f"[EXACT] (Sess:{sess}, {role}): {content[:150]}...")

    if not results:
        return "No relevant memories found."

    full_resp = "\n".join(results)
    if len(full_resp) > MAX_TOOL_OUTPUT_CHARS:
        return full_resp[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"
    return full_resp


def tool_analyze_deep(client, get_current_model, session_id: int, instruction: str):
    transcript = db.get_session_transcript(session_id, max_chars=6000)
    if not transcript:
        return "Session not found or empty."
    if client is None or get_current_model is None:
        return "Analysis failed: LLM client not available."

    try:
        sub_response = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": "You are a data extractor. Analyze the transcript based on the instruction. Be brief."},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nINSTRUCTION: {instruction}"}
            ],
            temperature=0.0,
            max_tokens=400
        )
        return f"ANALYSIS RESULT: {sub_response.choices[0].message.content}"
    except Exception as e:
        return f"Analysis failed: {str(e)}"


__all__ = [
    "SIMON_TOOLS",
    "_safe_args",
    "tool_search_memory",
    "tool_analyze_deep",
]
