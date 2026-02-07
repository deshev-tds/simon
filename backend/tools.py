import json
import time

import backend.db as db
import backend.memory as memory_mod
from backend.config import (
    MAX_TOOL_OUTPUT_CHARS,
    QUIET_LOGS,
    RLM_FTS_WEAK_SCORE,
    RLM_MIN_FTS_HITS,
    RLM_TRACE,
)

SIMON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search explicit memories (user-saved vectors) and chat history (FTS). Use to find facts, past conversations, or project details.",
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
    ,
    {
        "type": "function",
        "function": {
            "name": "list_session_files",
            "description": "List files uploaded in the current session (metadata only). Use when the user references an attached file without naming it.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_corpus_doc",
            "description": "Read a corpus document chunk by doc_id (from CORPUS search results). Use to fetch more context than the search excerpt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer"}
                },
                "required": ["doc_id"]
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


def _format_ts(ts: float | int | None) -> tuple[str, int]:
    if ts is None:
        return "ts=?", 0
    try:
        ts_int = int(float(ts) * 1000)
    except (TypeError, ValueError):
        ts_int = 0
    return f"ts={ts_int}", ts_int


def _best_bm25(rows: list[dict]) -> float | None:
    scores = [r.get("score") for r in rows if r.get("score") is not None]
    if not scores:
        return None
    return min(scores)


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
    lock = db_lock or db.db_lock

    fts_res: list[dict] = []
    corpus_res: list[dict] = []
    target_conn = db.db_conn
    # Use hot-session memdb for current session when available.
    if scope == "session" and session_id is not None:
        try:
            if db.mem_conn is not None and db.mem_active_session_id is not None and int(db.mem_active_session_id) == int(session_id):
                target_conn = db.mem_conn
        except Exception:
            target_conn = db.db_conn
    if scope == "session":
        fts_res = db.fts_recursive_search(
            query,
            session_id=session_id,
            limit=3,
            conn=target_conn,
            lock=lock,
        )
        # Also search the CORPUS for this session's uploaded file chunks. We don't have a
        # separate indexed "session_id" column in corpus, so we filter via the header token.
        if session_id is not None:
            corpus_res = db.fts_recursive_search_corpus(
                f"{query} session {int(session_id)}",
                limit=3,
                conn=db.corpus_conn,
                lock=lock,
            )
            if not corpus_res:
                corpus_res = db.fts_recursive_search_corpus(
                    f"session {int(session_id)}",
                    limit=3,
                    conn=db.corpus_conn,
                    lock=lock,
                )
    elif scope == "recent":
        fts_res = db.fts_recursive_search(
            query,
            session_id=None,
            limit=3,
            cutoff_ts=cutoff_ts,
            conn=target_conn,
            lock=lock,
        )
    else:
        fts_res = db.fts_recursive_search(
            query,
            session_id=None,
            limit=3,
            conn=target_conn,
            lock=lock,
        )
        corpus_res = db.fts_recursive_search_corpus(
            query,
            limit=3,
            conn=db.corpus_conn,
            lock=lock,
        )

    best_score = _best_bm25(fts_res) or _best_bm25(corpus_res)
    total_text_hits = len(fts_res) + len(corpus_res)
    fts_is_weak = (
        total_text_hits < RLM_MIN_FTS_HITS
        or best_score is None
        or best_score > RLM_FTS_WEAK_SCORE
    )

    docs, dists, metas = ([], [], [])
    if fts_is_weak:
        docs, dists, metas = mem.search(query, n_results=3, days_filter=days, session_filter=sess_filter)

    if RLM_TRACE and not QUIET_LOGS:
        top_dist = min(dists) if dists else None
        trace = {
            "query": query,
            "scope": scope,
            "fts_hits": len(fts_res),
            "corpus_hits": len(corpus_res),
            "best_bm25": best_score,
            "fts_is_weak": fts_is_weak,
            "vector_hits": len(docs),
            "top_dist": top_dist,
        }
        print(f"[TRACE][search_memory] {json.dumps(trace, ensure_ascii=False)}")

    evidence_items = []

    for doc, dist, meta in zip(docs, dists, metas):
        sess = meta.get("session_id", "?")
        ts = meta.get("timestamp", 0) or 0
        content = doc.replace("\n", " ")
        evidence_items.append({
            "source_type": "vector",
            "doc_id": meta.get("content_hash") or meta.get("id"),
            "ts": int(float(ts) * 1000) if ts else 0,
            "text": content[:200],
            "score": None,
            "distance": dist,
            "session_id": sess,
        })

    for item in fts_res:
        role = item.get("role") or "user"
        content = (item.get("content") or "").replace("\n", " ")
        sess = item.get("session_id", "?")
        created_at = item.get("created_at") or 0
        evidence_items.append({
            "source_type": "fts",
            "doc_id": item.get("id"),
            "ts": int(float(created_at) * 1000) if created_at else 0,
            "text": content[:200],
            "score": item.get("score"),
            "distance": None,
            "session_id": sess,
            "role": role,
        })

    for item in corpus_res:
        content = (item.get("content") or "").replace("\n", " ")
        source = item.get("source") or "corpus"
        created_at = item.get("created_at") or 0
        evidence_items.append({
            "source_type": "corpus",
            "doc_id": item.get("doc_id") or item.get("id"),
            "ts": int(float(created_at) * 1000) if created_at else 0,
            "text": content[:200],
            "score": item.get("score"),
            "distance": None,
            "source": source,
        })

    # Deduplicate by text
    deduped = []
    seen = set()
    for item in evidence_items:
        key = (item.get("text") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if not deduped:
        return "No relevant memories found."

    lines = []
    for item in deduped:
        source_type = (item.get("source_type") or "unknown").upper()
        ts = item.get("ts")
        ts_label = f"ts={ts}" if ts else "ts=?"
        text = (item.get("text") or "").strip()
        if source_type == "VECTOR":
            sess = item.get("session_id", "?")
            date_str = time.strftime("%Y-%m-%d", time.localtime((item.get("ts") or 0) / 1000))
            lines.append(f"[VECTOR] (Sess:{sess}, {date_str}, {ts_label}): {text[:150]}...")
        elif source_type == "FTS":
            sess = item.get("session_id", "?")
            role = item.get("role") or "user"
            date_str = time.strftime("%Y-%m-%d", time.localtime((item.get("ts") or 0) / 1000))
            lines.append(f"[EXACT] (Sess:{sess}, {date_str}, {ts_label}, {role}): {text[:150]}...")
        elif source_type == "CORPUS":
            source = item.get("source") or "corpus"
            date_str = time.strftime("%Y-%m-%d", time.localtime((item.get("ts") or 0) / 1000))
            lines.append(f"[CORPUS] ({source}, {date_str}, {ts_label}): {text[:150]}...")
        else:
            lines.append(f"[{source_type}] ({ts_label}): {text[:150]}...")

    payload = {"items": deduped, "lines": lines}
    payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    full_resp = f"EVIDENCE_JSON: {payload_str}"
    if lines:
        full_resp += "\n" + "\n".join(lines)
    if len(full_resp) > MAX_TOOL_OUTPUT_CHARS:
        return full_resp[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"
    return full_resp


def tool_list_session_files(
    session_id: int | None,
    db_lock=None,
):
    if session_id is None:
        return "Error: session_id is required."
    lock = db_lock or db.db_lock
    try:
        rows = db.list_session_files(int(session_id), limit=50, conn=db.db_conn, lock=lock)
    except Exception as exc:
        return f"Error: list_session_files failed: {exc}"
    if not rows:
        return "No uploaded files found in this session."
    lines = []
    for f in rows[:50]:
        lines.append(f"- {f.get('filename')} (id={f.get('id')}, mime={f.get('mime')}, size={f.get('size_bytes')} bytes)")
    payload = {"files": rows[:50], "lines": lines}
    out = f"FILES_JSON: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    if lines:
        out += "\n" + "\n".join(lines)
    if len(out) > MAX_TOOL_OUTPUT_CHARS:
        return out[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"
    return out


def tool_read_corpus_doc(
    doc_id: int,
    db_lock=None,
):
    lock = db_lock or db.db_lock
    try:
        target_conn = db.corpus_conn
        if target_conn is None:
            return "Error: corpus database is not initialized."
        with lock:
            row = target_conn.execute(
                "SELECT id, source, content, created_at FROM documents WHERE id=?",
                (int(doc_id),),
            ).fetchone()
    except Exception as exc:
        return f"Error: read_corpus_doc failed: {exc}"
    if not row:
        return "not found"
    payload = {
        "id": row[0],
        "source": row[1],
        "created_at": row[3],
        "content": row[2],
    }
    out = f"DOC_JSON: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    if len(out) > MAX_TOOL_OUTPUT_CHARS:
        return out[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"
    return out


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
    "tool_list_session_files",
    "tool_read_corpus_doc",
]
