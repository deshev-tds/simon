import asyncio
import json
import re
import threading
import time
import urllib.request
import ssl

import backend.db as db
import backend.memory_intents as memory_intents
import backend.tools as tools
from backend.audio import numpy_to_wav_bytes
from backend.config import (
    AGENT_MAX_TURNS,
    ANCHOR_MESSAGES,
    DEBUG_MODE,
    FTS_MAX_HITS,
    MAX_RECENT_MESSAGES,
    MAX_TOOL_OUTPUT_CHARS,
    QUIET_LOGS,
    RLM_TRACE,
    RLM_FALLBACK_ENABLED,
    RLM_MAX_HOPS,
    RLM_STREAM,
    RLM_DEEP_HISTORY_MAX_MSGS,
    RLM_DEEP_HISTORY_MAX_CHARS,
    SKIP_VECTOR_MEMORY,
    STREAM_TEXT_WITH_AUDIO,
    SSE_STREAM_FALLBACK,
    LM_STUDIO_URL,
    LLM_TIMEOUT_S,
    SYSTEM_PROMPT,
    TTS_VOICE,
)
from backend.metrics import (
    _metric_value,
    estimate_tokens_from_messages,
    estimate_tokens_from_text,
    finalize_metrics,
)
from backend.rlm_gate import GateContext, RLMGatekeeper

STREAM_FLUSH_CHARS = 24
STREAM_FLUSH_SECS = 0.05
EXPLICIT_MEMORY_MAX_CHARS = 400
SESSION_TITLE_MAX_CHARS = 80
SESSION_TITLE_CONTEXT_MAX_CHARS = 800
TOOL_LOOP_MAX_TOKENS = 256
FILE_EVIDENCE_MAX_DOCS = 8
FILE_EVIDENCE_FALLBACK_CHUNKS_PER_FILE = 3

_UNCERTAIN_PHRASES = (
    "not sure",
    "don't know",
    "dont know",
    "cannot find",
    "can't find",
    "no information",
    "no info",
    "not available",
    "unsure",
    "i'm not sure",
    "im not sure",
)
_CODE_PATTERN = re.compile(
    r"\b(?:[A-Z0-9]{2,}[-_][A-Z0-9]{2,}|[A-Z0-9]*\d[A-Z0-9_-]*)\b"
)
_HAS_DIGIT = re.compile(r"\d")
_HAS_UPPER_SEQ = re.compile(r"[A-Z]{2,}")
_CODE_INTENT = ("code", "token", "key", "id", "auth", "authentication", "password", "api key", "access code", "pin")
_BRIDGE_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "project", "protocol",
    "update", "confidential", "log", "entry", "alpha", "beta", "uses", "code",
}

_PERSON_STOPWORDS = {
    "project", "protocol", "system", "session", "report", "note", "the", "today",
    "yesterday", "this", "that", "these", "those", "log", "entry", "stage",
    "alpha", "beta", "omega",
}
_TITLE_PATTERN = re.compile(r"^(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+", re.I)
_PERSON_PATTERN = re.compile(r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_CITY_PATTERN = re.compile(r"\b(?:in|at|from|near|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_UPLOAD_SOURCE_PATTERN = re.compile(r"^upload:([^:]+):")

_DEEP_SYSTEM_PROMPT = (
    "You are Simon (Deep Mode). You MUST call search_memory before answering. "
    "Use scope='global' unless the user explicitly asks about the current session. "
    "If the user references an attached file without naming it, call list_session_files. "
    "If you need more context from a CORPUS hit, call read_corpus_doc(doc_id). "
    "Answer ONLY from tool evidence above. If no evidence is found, reply exactly: not found. "
    "Do not hallucinate."
)

_TOOL_CALLING_SUPPORTED: bool | None = None


def _tool_calling_enabled() -> bool:
    return _TOOL_CALLING_SUPPORTED is not False


def _maybe_disable_tool_calling(exc: Exception) -> bool:
    global _TOOL_CALLING_SUPPORTED
    text = str(exc or "").lower()
    looks_bad_request = ("400" in text) or ("badrequest" in text) or ("invalid_request" in text)
    looks_tools_related = ("tool" in text) or ("function" in text) or ("tool_choice" in text) or ("tools" in text)
    if looks_bad_request and looks_tools_related:
        _TOOL_CALLING_SUPPORTED = False
        return True
    return False


def _with_system_prompt(messages: list[dict]) -> list[dict]:
    if not SYSTEM_PROMPT:
        return messages
    if messages and messages[0].get("role") == "system" and (messages[0].get("content") or "") == SYSTEM_PROMPT:
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages

_EVIDENCE_PATTERNS = {
    "code": _CODE_PATTERN,
    "date": re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
    "amount": re.compile(r"\b(?:\$|usd|eur|gbp)?\s?\d+(?:\.\d+)?\b", re.I),
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "phone": re.compile(r"\+?\d[\d\s\-]{6,}\d"),
    "formula": re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b"),
}

_EVIDENCE_INTENT_MAP = {
    "code": {"code", "token", "key", "id", "auth", "authentication", "password", "api key", "access code", "pin"},
    "date": {"date", "when", "deadline", "due", "timestamp"},
    "amount": {"price", "cost", "budget", "amount", "total", "usd", "$", "eur", "gbp"},
    "email": {"email", "e-mail", "mail"},
    "phone": {"phone", "call", "number"},
    "name": {"name", "named", "called", "who", "whose", "person", "sister", "brother", "father", "mother", "boss", "leader"},
    "city": {"city", "town", "capital"},
    "formula": {"formula", "chemical", "compound"},
}

_EVIDENCE_QUERY_HINT = {
    "code": "authentication code",
    "date": "date",
    "amount": "amount",
    "email": "email",
    "phone": "phone",
    "name": "name",
    "city": "city",
    "formula": "formula",
}


def _evidence_types_for_query(query: str) -> list[str]:
    q = (query or "").lower()
    words = set(re.findall(r"\b\w+\b", q))
    types = []
    for kind, tokens in _EVIDENCE_INTENT_MAP.items():
        hit = False
        for tok in tokens:
            if " " in tok:
                if tok in q:
                    hit = True
                    break
            else:
                if tok in words:
                    hit = True
                    break
        if hit:
            types.append(kind)
    return types


def _missing_evidence_types(user_text: str, evidence_reason: str) -> list[str]:
    if evidence_reason.startswith("missing_evidence:"):
        payload = evidence_reason.split("missing_evidence:", 1)[1]
        return [p for p in payload.split(",") if p]
    if evidence_reason in ("no_evidence_lines", "no_tool_calls"):
        return _evidence_types_for_query(user_text)
    return []


def _iter_sse_lines(resp):
    for raw in resp:
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not line or not line.startswith("data:"):
            continue
        yield line[5:].strip()


def _stream_sse_chat(url: str, payload: dict, on_token, stop_evt: threading.Event | None, timeout_s: int) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    ctx = None
    if url.startswith("https://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        if ctx:
            resp = urllib.request.urlopen(req, timeout=timeout_s or None, context=ctx)
        else:
            resp = urllib.request.urlopen(req, timeout=timeout_s or None)
    except Exception:
        return False

    saw_delta = False
    try:
        for data_line in _iter_sse_lines(resp):
            if stop_evt is not None and stop_evt.is_set():
                break
            if data_line == "[DONE]":
                break
            try:
                chunk = json.loads(data_line)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content")
            if token:
                saw_delta = True
                on_token(token)
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return saw_delta


def _parse_prefixed_json(text: str, prefix: str) -> dict | None:
    for line in str(text).splitlines():
        if line.startswith(prefix):
            payload_str = line.split(prefix, 1)[1].strip()
            if not payload_str:
                return None
            try:
                return json.loads(payload_str)
            except Exception:
                return None
    return None


def _parse_evidence_json(text: str) -> dict | None:
    return _parse_prefixed_json(text, "EVIDENCE_JSON:")


def _parse_doc_json(text: str) -> dict | None:
    return _parse_prefixed_json(text, "DOC_JSON:")


def _evidence_items_to_lines(items: list[dict]) -> list[str]:
    lines = []
    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        source_type = (item.get("source_type") or "unknown").upper()
        ts = item.get("ts")
        ts_label = f"ts={ts}" if ts else "ts=?"
        lines.append(f"[{source_type}] ({ts_label}): {text[:150]}...")
    return lines


def _line_to_item(line: str) -> dict:
    source_type = "unknown"
    match = re.match(r"^\[([A-Z]+)\]", line)
    if match:
        source_type = match.group(1).lower()
    return {
        "source_type": source_type,
        "doc_id": None,
        "ts": _parse_evidence_ts(line),
        "text": _normalize_evidence_line(line),
        "score": None,
        "distance": None,
    }


def _extract_evidence_items(tool_texts: list[str]) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    lines: list[str] = []
    for text in tool_texts:
        payload = _parse_evidence_json(text)
        if payload:
            payload_items = payload.get("items") or payload.get("evidence") or []
            payload_lines = payload.get("lines") or []
            if isinstance(payload_items, list):
                items.extend(payload_items)
            if isinstance(payload_lines, list):
                lines.extend([l for l in payload_lines if isinstance(l, str)])
            if payload_items and not payload_lines:
                lines.extend(_evidence_items_to_lines(payload_items))
            continue
        doc_payload = _parse_doc_json(text)
        if doc_payload:
            content = str(doc_payload.get("content") or "").replace("\n", " ").strip()
            source = str(doc_payload.get("source") or "corpus")
            ts = 0
            created_at = doc_payload.get("created_at")
            if created_at is not None:
                try:
                    ts = int(float(created_at) * 1000)
                except (TypeError, ValueError):
                    ts = 0
            if content:
                doc_item = {
                    "source_type": "corpus",
                    "doc_id": doc_payload.get("id"),
                    "ts": ts,
                    "text": content[:200],
                    "score": None,
                    "distance": None,
                    "source": source,
                }
                items.append(doc_item)
                lines.append(f"[CORPUS] ({source}, ts={ts or '?'}) : {content[:150]}...")
            continue
        for line in str(text).splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                lines.append(line)
                items.append(_line_to_item(line))
    return items, lines


def _normalize_evidence_line(line: str) -> str:
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", line)
    cleaned = re.sub(r"^\([^)]+\):\s*", "", cleaned)
    return cleaned.strip()


def _parse_evidence_ts(line: str) -> int:
    match = re.search(r"ts=(\d+)", line)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


def _extract_person_name(line: str, query_lower: str) -> str | None:
    candidates = []
    for match in _PERSON_PATTERN.finditer(line):
        name = match.group(0).strip()
        name = _TITLE_PATTERN.sub("", name)
        name = name.replace("'s", "")
        tokens = [t for t in name.split() if t]
        if not tokens:
            continue
        if any(t.lower() in _PERSON_STOPWORDS for t in tokens):
            continue
        if name.lower() in query_lower:
            continue
        candidates.append(name)
    if not candidates:
        return None
    return candidates[-1]


def _extract_city_name(line: str) -> str | None:
    match = _CITY_PATTERN.search(line)
    if not match:
        return None
    city = match.group(1).strip()
    tokens = [t for t in city.split() if t]
    if tokens and any(t.lower() in _PERSON_STOPWORDS for t in tokens):
        return None
    return city


def _item_matches_kind(kind: str, text: str, query_lower: str) -> bool:
    if kind == "name":
        return _extract_person_name(text, query_lower) is not None
    if kind == "city":
        return _extract_city_name(text) is not None
    pattern = _EVIDENCE_PATTERNS.get(kind)
    if pattern is None:
        return False
    return pattern.search(text) is not None


def _has_evidence_type(kind: str, evidence_items: list[dict], query_lower: str) -> bool:
    for item in evidence_items:
        cleaned = (item.get("text") or "").strip()
        if _item_matches_kind(kind, cleaned, query_lower):
            return True
    return False


def _evidence_check(user_text: str, tool_texts: list[str]) -> tuple[bool, str, list[dict], list[str], dict]:
    evidence_items, evidence_lines = _extract_evidence_items(tool_texts)
    required_types = _evidence_types_for_query(user_text)
    matches = {}
    query_lower = (user_text or "").lower()
    for kind in required_types:
        matches[kind] = _has_evidence_type(kind, evidence_items, query_lower)

    if required_types:
        for item in evidence_items:
            cleaned = (item.get("text") or "").strip()
            hits = {}
            for kind in required_types:
                hits[kind] = _item_matches_kind(kind, cleaned, query_lower)
            if hits:
                item["entity_hits"] = hits

    if not evidence_items:
        return False, "no_evidence_lines", evidence_items, evidence_lines, matches
    if required_types:
        missing = [k for k, ok in matches.items() if not ok]
        if missing:
            return False, f"missing_evidence:{','.join(missing)}", evidence_items, evidence_lines, matches
    return True, "evidence_ok", evidence_items, evidence_lines, matches


def _extract_evidence_value(kind: str, evidence_items: list[dict]) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    if kind == "name":
        for idx, item in enumerate(evidence_items):
            cleaned = (item.get("text") or "").strip()
            name = _extract_person_name(cleaned, "")
            if name:
                candidates.append((int(item.get("ts") or 0), idx, name))
    elif kind == "city":
        for idx, item in enumerate(evidence_items):
            cleaned = (item.get("text") or "").strip()
            city = _extract_city_name(cleaned)
            if city:
                candidates.append((int(item.get("ts") or 0), idx, city))
    else:
        pattern = _EVIDENCE_PATTERNS.get(kind)
        if pattern is None:
            return None
        for idx, item in enumerate(evidence_items):
            match = pattern.search((item.get("text") or "").strip())
            if match:
                candidates.append((int(item.get("ts") or 0), idx, match.group(0)))

    if not candidates:
        return None

    if any(ts > 0 for ts, _, _ in candidates):
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[-1][2]
    candidates.sort(key=lambda c: c[1])
    return candidates[-1][2]


def _enforce_evidence_answer(user_text: str, evidence_items: list[dict]) -> tuple[str | None, str | None]:
    required = _evidence_types_for_query(user_text)
    if not required:
        return None, None
    for kind in required:
        value = _extract_evidence_value(kind, evidence_items)
        if value:
            return value, f"extracted_{kind}"
    return "not found", "forced_not_found"


def _extract_upload_file_id(source: str) -> str | None:
    m = _UPLOAD_SOURCE_PATTERN.match(str(source or ""))
    if not m:
        return None
    return m.group(1)


def _force_file_tool_call(
    user_text: str,
    session_id: int,
    file_payloads: list[dict] | None,
    metrics: dict,
):
    if not file_payloads:
        return None

    file_map: dict[str, str] = {}
    for f in file_payloads:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or "").strip()
        if not fid:
            continue
        fname = str(f.get("filename") or f"file-{fid[:8]}").strip()
        file_map[fid] = fname
    if not file_map:
        return None

    items: list[dict] = []

    def _append_item(doc_id, source, content, created_at, score=None):
        fid = _extract_upload_file_id(source)
        if not fid or fid not in file_map:
            return
        text = str(content or "").replace("\n", " ").strip()
        if not text:
            return
        ts = 0
        if created_at is not None:
            try:
                ts = int(float(created_at) * 1000)
            except (TypeError, ValueError):
                ts = 0
        items.append({
            "source_type": "corpus",
            "doc_id": doc_id,
            "ts": ts,
            "text": text[:200],
            "score": score,
            "distance": None,
            "source": source,
            "file_id": fid,
            "filename": file_map.get(fid),
        })

    try:
        if db.corpus_conn is None:
            return None

        query = (user_text or "").strip()
        if query:
            hits = db.fts_recursive_search_corpus(
                query,
                limit=max(16, len(file_map) * 4),
                conn=db.corpus_conn,
                lock=db.db_lock,
            )
            for h in hits:
                source = h.get("source") or ""
                if _extract_upload_file_id(source) not in file_map:
                    continue
                _append_item(
                    h.get("doc_id") or h.get("id"),
                    source,
                    h.get("content"),
                    h.get("created_at"),
                    h.get("score"),
                )
                if len(items) >= FILE_EVIDENCE_MAX_DOCS:
                    break

        if not items:
            with db.db_lock:
                for fid in file_map:
                    rows = db.corpus_conn.execute(
                        "SELECT id, source, content, created_at FROM documents WHERE source LIKE ? ORDER BY id ASC LIMIT ?",
                        (f"upload:{fid}:%", FILE_EVIDENCE_FALLBACK_CHUNKS_PER_FILE),
                    ).fetchall()
                    for row in rows:
                        _append_item(row[0], row[1], row[2], row[3], None)
                        if len(items) >= FILE_EVIDENCE_MAX_DOCS:
                            break
                    if len(items) >= FILE_EVIDENCE_MAX_DOCS:
                        break
    except Exception as exc:
        if RLM_TRACE and not QUIET_LOGS:
            print(f"[TRACE][file_seed] failed: {exc}")
        return None

    if not items:
        metrics.setdefault("file_seed", {})
        metrics["file_seed"] = {
            "ok": False,
            "files": list(file_map.keys()),
            "reason": "no_indexed_chunks",
        }
        return None

    lines = []
    for item in items[:FILE_EVIDENCE_MAX_DOCS]:
        ts = item.get("ts")
        ts_label = f"ts={ts}" if ts else "ts=?"
        fname = item.get("filename") or item.get("file_id") or "file"
        text = (item.get("text") or "").strip()
        lines.append(f"[CORPUS] (file={fname}, {ts_label}): {text[:150]}...")

    payload = {"items": items[:FILE_EVIDENCE_MAX_DOCS], "lines": lines}
    out = f"EVIDENCE_JSON: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    if lines:
        out += "\n" + "\n".join(lines)
    if len(out) > MAX_TOOL_OUTPUT_CHARS:
        out = out[:MAX_TOOL_OUTPUT_CHARS] + "...[TRUNCATED]"

    metrics.setdefault("file_seed", {})
    metrics["file_seed"] = {
        "ok": True,
        "files": list(file_map.keys()),
        "items": len(items[:FILE_EVIDENCE_MAX_DOCS]),
    }
    return out


def _force_tool_call(
    user_text: str,
    session_id: int,
    memory,
    metrics: dict,
    file_payloads: list[dict] | None = None,
):
    file_seed = _force_file_tool_call(user_text, session_id, file_payloads, metrics)
    if file_seed:
        metrics.setdefault("rlm_forced_tool", [])
        metrics["rlm_forced_tool"].append({"query": "__attached_files__", "mode": "files"})
        return file_seed

    query = user_text or ""
    result = tools.tool_search_memory(
        query,
        scope="global",
        session_id=session_id,
        memory=memory,
        db_lock=db.db_lock,
    )
    metrics.setdefault("rlm_forced_tool", [])
    metrics["rlm_forced_tool"].append({"query": query})
    return result


def _extract_bridge_candidates(text: str, query_lower: str) -> list[str]:
    if not text:
        return []
    hyphenated = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", text)
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    seen = set()
    candidates = []
    for cand in hyphenated + proper:
        norm = cand.strip()
        key = norm.lower()
        if not norm or key in seen:
            continue
        if key in query_lower:
            continue
        if key in _BRIDGE_STOPWORDS:
            continue
        seen.add(key)
        candidates.append(norm)
    return candidates


def _maybe_auto_hop(user_text: str, tool_text: str, session_id: int, memory, metrics: dict):
    query_lower = (user_text or "").lower()
    if not any(k in query_lower for k in _CODE_INTENT):
        return None
    items, _ = _extract_evidence_items([tool_text])
    bridge_text = "\n".join([i.get("text", "") for i in items]) if items else tool_text
    if _CODE_PATTERN.search(bridge_text):
        return None

    candidates = _extract_bridge_candidates(bridge_text, query_lower)
    if not candidates:
        return None

    keyword = next((k for k in _CODE_INTENT if k in query_lower), "code")
    bridge = candidates[0]
    hop_query = f"{bridge} {keyword}"
    result = tools.tool_search_memory(
        hop_query,
        scope="global",
        session_id=session_id,
        memory=memory,
        db_lock=db.db_lock,
    )
    metrics.setdefault("rlm_autohop", [])
    metrics["rlm_autohop"].append({"query": hop_query, "bridge": bridge})
    return {"query": hop_query, "result": result}


def _apply_hops(
    user_text: str,
    session_id: int,
    memory,
    metrics: dict,
    current_messages: list,
    all_tool_texts: list,
    tool_chars_used: int,
    evidence_ok: bool,
    evidence_reason: str,
    evidence_items: list[dict],
    evidence_lines: list[str],
    evidence_matches: dict,
):
    if evidence_ok or RLM_MAX_HOPS <= 0:
        return evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches, tool_chars_used

    missing = _missing_evidence_types(user_text, evidence_reason)
    if not missing:
        return evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches, tool_chars_used

    seen_bridges = set()
    prev_missing = set(missing)
    prev_lines = len(evidence_lines)
    hop_count = 0

    while hop_count < RLM_MAX_HOPS and missing:
        candidates = _extract_bridge_candidates("\n".join(all_tool_texts), (user_text or "").lower())
        bridge = next((c for c in candidates if c.lower() not in seen_bridges), None)
        if not bridge:
            break
        seen_bridges.add(bridge.lower())

        hint = _EVIDENCE_QUERY_HINT.get(missing[0], missing[0])
        hop_query = f"{bridge} {hint}".strip()
        hop_result = tools.tool_search_memory(
            hop_query,
            scope="global",
            session_id=session_id,
            memory=memory,
            db_lock=db.db_lock,
        )
        metrics.setdefault("rlm_hops", [])
        metrics["rlm_hops"].append({"bridge": bridge, "query": hop_query})

        result_str = str(hop_result)
        remain = MAX_TOOL_OUTPUT_CHARS - tool_chars_used
        if remain <= 0:
            result_str = "[TOOL_BUDGET_EXCEEDED]"
        else:
            result_str = result_str[:remain]
        tool_chars_used += len(result_str)
        all_tool_texts.append(result_str)
        current_messages.append({
            "role": "tool",
            "tool_call_id": f"hop-{hop_count + 1}",
            "content": result_str,
        })
        hop_count += 1

        evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches = _evidence_check(
            user_text, all_tool_texts
        )
        if evidence_ok:
            break
        missing = _missing_evidence_types(user_text, evidence_reason)
        if len(evidence_lines) <= prev_lines and set(missing) == prev_missing:
            break
        prev_lines = len(evidence_lines)
        prev_missing = set(missing)

    return evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches, tool_chars_used


def _extract_message_text(msg):
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return getattr(msg, "content", "") or ""


def _build_user_message(user_text: str, image_payloads: list[dict] | None, file_payloads: list[dict] | None = None):
    text = user_text or ""
    if file_payloads:
        lines = ["[ATTACHED_FILES]"]
        for f in file_payloads:
            if not isinstance(f, dict):
                continue
            fid = (f.get("id") or "").strip()
            name = (f.get("filename") or "").strip()
            mime = (f.get("mime") or "").strip()
            size_b = f.get("size_bytes")
            lines.append(f"- {name} (id={fid}, mime={mime}, size={size_b} bytes)")
        lines.append("[/ATTACHED_FILES]")
        block = "\n".join(lines)
        text = f"{text}\n\n{block}".strip() if text.strip() else block

    if not image_payloads:
        return {"role": "user", "content": text}
    parts = []
    parts.append({"type": "text", "text": text})
    for img in image_payloads:
        mime = (img.get("mime") or "image/jpeg").strip()
        data_b64 = (img.get("data_b64") or "").strip()
        if not data_b64:
            continue
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data_b64}"},
        })
    return {"role": "user", "content": parts}


def _needs_fallback(user_text: str, reply_text: str, gate_metrics: dict) -> bool:
    if not reply_text:
        return True
    lower = reply_text.lower()
    if any(p in lower for p in _UNCERTAIN_PHRASES):
        return True
    query_lower = (user_text or "").lower()
    if any(k in query_lower for k in _CODE_INTENT):
        if not _CODE_PATTERN.search(reply_text):
            if not (_HAS_DIGIT.search(reply_text) or _HAS_UPPER_SEQ.search(reply_text)):
                return True
    if gate_metrics.get("weak_retrieval") and (gate_metrics.get("is_recall") or gate_metrics.get("is_complex")):
        return True
    return False


def _window_messages(msgs, first=10, last=10):
    if len(msgs) <= first + last:
        return msgs
    return msgs[:first] + msgs[-last:]


def _select_deep_history(history: list[dict]) -> list[dict]:
    if not history or RLM_DEEP_HISTORY_MAX_MSGS <= 0:
        return []
    trimmed = []
    for msg in reversed(history):
        content = (msg.get("content") or "")
        if len(content) > RLM_DEEP_HISTORY_MAX_CHARS:
            continue
        trimmed.append(msg)
        if len(trimmed) >= RLM_DEEP_HISTORY_MAX_MSGS:
            break
    return list(reversed(trimmed))


def _estimate_window_tokens(history):
    if not history:
        return 0
    history_window = _window_messages(history)
    anchor = history_window[:ANCHOR_MESSAGES]
    remaining = history_window[ANCHOR_MESSAGES:]
    recent = remaining[-MAX_RECENT_MESSAGES:] if MAX_RECENT_MESSAGES > 0 else []
    return estimate_tokens_from_messages(anchor + recent)


def _probe_retrieval(user_text, memory, session_id):
    t_start = time.time()
    retrieved_docs, distances, metadatas = memory.search(user_text, n_results=3, days_filter=60)
    try:
        target_conn = db.db_conn
        if db.mem_conn is not None and db.mem_active_session_id is not None and int(db.mem_active_session_id) == int(session_id):
            target_conn = db.mem_conn
        fts_hits = db.fts_recursive_search(
            user_text,
            session_id=session_id,
            limit=FTS_MAX_HITS,
            conn=target_conn,
            lock=db.db_lock,
        )
    except Exception:
        fts_hits = []
    probe_s = time.time() - t_start
    return {
        "docs": retrieved_docs or [],
        "dists": distances or [],
        "metas": metadatas or [],
        "fts_hits": fts_hits or [],
        "probe_s": probe_s,
    }


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


def _trace_rlm_metrics(metrics: dict):
    if not RLM_TRACE or QUIET_LOGS:
        return
    gate = metrics.get("rlm_gate")
    if gate:
        print(f"[TRACE][rlm_gate] {json.dumps(gate, ensure_ascii=False)}")
    evidence = metrics.get("evidence") or {}
    if evidence:
        items = evidence.get("items") or []
        counts = {}
        for item in items:
            src = (item.get("source_type") or "unknown").lower()
            counts[src] = counts.get(src, 0) + 1
        trace = {
            "ok": evidence.get("ok"),
            "reason": evidence.get("reason"),
            "matches": evidence.get("matches"),
            "enforced": evidence.get("enforced"),
            "counts": counts,
            "sample": [i.get("text") for i in items[:3]],
        }
        print(f"[TRACE][evidence] {json.dumps(trace, ensure_ascii=False)}")
    fallback = metrics.get("rlm_fallback")
    if fallback:
        print(f"[TRACE][rlm_fallback] {json.dumps(fallback, ensure_ascii=False)}")


def _trace_final_answer(text: str | None):
    if not RLM_TRACE or QUIET_LOGS:
        return
    if not text:
        return
    preview = text.replace("\n", " ").strip()
    if len(preview) > 240:
        preview = preview[:240] + "…"
    print(f"[TRACE][final_answer] {preview}")


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
    image_payloads=None,
    file_payloads=None,
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

    store_user_text = user_text or ""
    if image_payloads and not store_user_text.strip():
        store_user_text = "[image]"
    if file_payloads and not store_user_text.strip():
        store_user_text = "[file]"

    t_ctx_start = time.time()
    retrieval_cache = _probe_retrieval(user_text, memory, session_id)
    metrics["rag"] = retrieval_cache.get("probe_s", 0.0)
    session_tokens = estimate_tokens_from_messages(history)
    window_tokens = _estimate_window_tokens(history)
    gate_ctx = GateContext(
        user_query=user_text,
        session_tokens=session_tokens,
        window_tokens=window_tokens,
        vector_dists=retrieval_cache.get("dists", []),
        fts_hit_count=len(retrieval_cache.get("fts_hits", [])),
        recent_history=history,
        vector_enabled=not SKIP_VECTOR_MEMORY,
    )
    gate_decision = RLMGatekeeper.evaluate(gate_ctx)
    metrics["rlm_gate"] = {
        "trigger": gate_decision.trigger,
        "reason": gate_decision.reason,
        **(gate_decision.metrics or {}),
    }
    if RLM_TRACE:
        try:
            log_console(f"RLM gate: {json.dumps(metrics['rlm_gate'], ensure_ascii=False)}", "TRACE")
        except Exception:
            log_console(f"RLM gate: {metrics.get('rlm_gate')}", "TRACE")
    forced_by_files = bool(file_payloads)
    if forced_by_files and not gate_decision.trigger:
        metrics.setdefault("rlm_gate", {})
        if isinstance(metrics.get("rlm_gate"), dict):
            metrics["rlm_gate"]["forced_by_files"] = True
    is_deep_mode = gate_decision.trigger or forced_by_files

    current_messages = []
    rag_payload = []

    if is_deep_mode:
        log_console("ACTIVATING AGENTIC LOOP", "AGENT")
        await websocket.send_text("SYS:THINKING: Entering Deep Mode...")
        deep_history = _select_deep_history(history)
        current_messages = _with_system_prompt([
            {"role": "system", "content": _DEEP_SYSTEM_PROMPT},
            *deep_history,
            _build_user_message(user_text, image_payloads, file_payloads),
        ])
        tools_list = tools.SIMON_TOOLS
    else:
        context_msgs, rag_payload = build_rag_context(
            user_text,
            history,
            memory,
            metrics,
            session_id,
            retrieval_cache=retrieval_cache,
        )
        current_messages = _with_system_prompt(context_msgs + [_build_user_message(user_text, image_payloads, file_payloads)])
        tools_list = None

    metrics["ctx"] = time.time() - t_ctx_start
    metrics["input_chars"] = len(user_text)
    if metrics.get("input_tokens") is None:
        metrics["input_tokens"] = estimate_tokens_from_messages(current_messages)

    if rag_payload:
        await websocket.send_text(f"RAG:{json.dumps(rag_payload)}")

    emit_text_deltas = (not generate_audio) or STREAM_TEXT_WITH_AUDIO
    emit_final_text = True
    emit_tts_text = False

    q = asyncio.Queue(maxsize=64) if generate_audio else None
    response_holder = {"text": ""}
    sentence_endings = re.compile(r"[.!?]+")

    def _run_text_fallback(base_text: str, model_name: str, loop, stop_evt) -> tuple[str, str | None]:
        fallback_reason = None
        full_reply = base_text
        if not _needs_fallback(user_text, base_text, gate_decision.metrics or {}):
            metrics["rlm_fallback"] = {"trigger": False, "reason": "base_ok"}
            if isinstance(metrics.get("rlm_gate"), dict):
                metrics["rlm_gate"]["fallback_reason"] = "base_ok"
            return full_reply, fallback_reason

        fallback_reason = "answer_inadequate"
        if not stop_evt.is_set():
            try:
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text("SYS:THINKING: Entering Deep Mode..."),
                    loop,
                )
            except Exception:
                pass

        deep_history = _select_deep_history(history)
        deep_messages = _with_system_prompt([
            {"role": "system", "content": _DEEP_SYSTEM_PROMPT},
            *deep_history,
            _build_user_message(user_text, image_payloads, file_payloads),
        ])
        tool_chars_used = 0
        turn_count = 0
        auto_hop_used = False
        tool_calls_made = False
        all_tool_texts = []
        skip_tool_loop = False
        if file_payloads:
            seeded = _force_file_tool_call(user_text, session_id, file_payloads, metrics)
            if seeded:
                tool_calls_made = True
                seeded = seeded[:MAX_TOOL_OUTPUT_CHARS]
                all_tool_texts.append(seeded)
                tool_chars_used += len(seeded)
                deep_messages.append({
                    "role": "tool",
                    "tool_call_id": "attached-files-seed",
                    "content": seeded,
                })
                seed_ok, _, _, _, _ = _evidence_check(user_text, [seeded])
                skip_tool_loop = seed_ok
                if not stop_evt.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Reading attached files..."),
                            loop,
                        )
                    except Exception:
                        pass

        while (not skip_tool_loop) and turn_count < AGENT_MAX_TURNS and not stop_evt.is_set():
            if not _tool_calling_enabled():
                break
            turn_count += 1
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=deep_messages,
                    temperature=0.7,
                    stream=False,
                    tools=tools.SIMON_TOOLS,
                    tool_choice="auto",
                    max_tokens=TOOL_LOOP_MAX_TOKENS,
                )
            except Exception as exc:
                disabled = _maybe_disable_tool_calling(exc)
                if DEBUG_MODE:
                    if disabled:
                        log_console(f"Agent Loop: disabling tools for provider compatibility ({exc})", "WARN")
                    else:
                        log_console(f"Agent Loop Error: {exc}", "ERR")
                break

            msg_dict = _msg_to_dict(response.choices[0].message)
            tool_calls = msg_dict.get("tool_calls") or []
            if tool_calls:
                tool_calls_made = True
                deep_messages.append(msg_dict)
                if not stop_evt.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Consulting memory..."),
                            loop,
                        )
                    except Exception:
                        pass
                tool_texts = []
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
                            db_lock=db.db_lock,
                        )
                    elif fn_name == "analyze_deep_context":
                        if not stop_evt.is_set():
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_text("SYS:THINKING: Deep reading transcript..."),
                                    loop,
                                )
                            except Exception:
                                pass
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
                    elif fn_name == "list_session_files":
                        result = tools.tool_list_session_files(
                            session_id,
                            db_lock=db.db_lock,
                        )
                    elif fn_name == "read_corpus_doc":
                        doc_id = args.get("doc_id")
                        if doc_id is None:
                            result = "Error: read_corpus_doc requires doc_id."
                        else:
                            result = tools.tool_read_corpus_doc(
                                int(doc_id),
                                db_lock=db.db_lock,
                            )

                    result_str = str(result)
                    remain = MAX_TOOL_OUTPUT_CHARS - tool_chars_used
                    if remain <= 0:
                        result_str = "[TOOL_BUDGET_EXCEEDED]"
                    else:
                        result_str = result_str[:remain]
                    tool_chars_used += len(result_str)
                    tool_texts.append(result_str)

                    tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                    tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
                    deep_messages.append(tool_msg)
                all_tool_texts.extend(tool_texts)
                if not auto_hop_used and tool_texts:
                    hop = _maybe_auto_hop(
                        user_text,
                        "\n".join(tool_texts),
                        session_id,
                        memory,
                        metrics,
                    )
                    if hop:
                        auto_hop_used = True
                        if not stop_evt.is_set():
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_text("SYS:THINKING: Following bridge..."),
                                    loop,
                                )
                            except Exception:
                                pass
                        hop_msg = {
                            "role": "tool",
                            "tool_call_id": f"auto-hop-{turn_count}",
                            "content": str(hop["result"]),
                        }
                        deep_messages.append(hop_msg)
                        all_tool_texts.append(str(hop["result"]))
            else:
                break

        try:
            if not tool_calls_made:
                if not stop_evt.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Consulting memory..."),
                            loop,
                        )
                    except Exception:
                        pass
                forced = _force_tool_call(
                    user_text,
                    session_id,
                    memory,
                    metrics,
                    file_payloads=file_payloads,
                )
                tool_calls_made = True
                forced_text = str(forced)
                all_tool_texts.append(forced_text)
                deep_messages.append({
                    "role": "tool",
                    "tool_call_id": "forced-tool",
                    "content": forced_text[:MAX_TOOL_OUTPUT_CHARS],
                })

            evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches = _evidence_check(
                user_text, all_tool_texts
            )
            evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches, tool_chars_used = _apply_hops(
                user_text,
                session_id,
                memory,
                metrics,
                deep_messages,
                all_tool_texts,
                tool_chars_used,
                evidence_ok,
                evidence_reason,
                evidence_items,
                evidence_lines,
                evidence_matches,
            )
            metrics["evidence"] = {
                "ok": evidence_ok,
                "reason": evidence_reason,
                "lines": evidence_lines[:10],
                "items": evidence_items[:10],
                "matches": evidence_matches,
                "tool_calls": tool_calls_made or bool(metrics.get("rlm_hops")),
            }
            if not (tool_calls_made or metrics.get("rlm_hops")):
                evidence_ok = False
                metrics["evidence"]["ok"] = False
                metrics["evidence"]["reason"] = "no_tool_calls"

            if not evidence_ok:
                full_reply = "not found"
            else:
                enforced_value, enforced_reason = _enforce_evidence_answer(user_text, evidence_items)
                if enforced_reason:
                    metrics["evidence"]["enforced"] = enforced_reason
                    full_reply = enforced_value or "not found"
                else:
                    evidence_block = "\n".join(evidence_lines[:10])
                    final_messages = deep_messages + [{
                        "role": "system",
                        "content": (
                            "Answer ONLY from the tool evidence above. "
                            "Cite evidence verbatim in your reasoning if needed. "
                            "If evidence is missing, reply exactly: not found."
                        ),
                    }]
                    if evidence_block:
                        final_messages.append({
                            "role": "system",
                            "content": f"EVIDENCE LINES:\n{evidence_block}",
                        })
                    final_resp = client.chat.completions.create(
                        model=model_name,
                        messages=final_messages,
                        temperature=0.2,
                        stream=False,
                    )
                    final_msg = _msg_to_dict(final_resp.choices[0].message)
                    full_reply = final_msg.get("content") or _extract_message_text(final_resp.choices[0].message)
        except Exception as exc:
            full_reply = base_text
            if DEBUG_MODE:
                log_console(f"Deep completion failed: {exc}", "ERR")

        metrics["rlm_fallback"] = {
            "trigger": True,
            "reason": fallback_reason,
        }
        if isinstance(metrics.get("rlm_gate"), dict):
            metrics["rlm_gate"]["fallback_reason"] = fallback_reason
        return full_reply, fallback_reason

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
                auto_hop_used = False
                tool_calls_made = False
                all_tool_texts = []
                skip_tool_loop = False
                if file_payloads:
                    seeded = _force_file_tool_call(user_text, session_id, file_payloads, metrics)
                    if seeded:
                        tool_calls_made = True
                        seeded = seeded[:MAX_TOOL_OUTPUT_CHARS]
                        all_tool_texts.append(seeded)
                        tool_chars_used += len(seeded)
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": "attached-files-seed",
                            "content": seeded,
                        })
                        seed_ok, _, _, _, _ = _evidence_check(user_text, [seeded])
                        skip_tool_loop = seed_ok
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Reading attached files..."),
                            loop,
                        )
                while (not skip_tool_loop) and turn_count < AGENT_MAX_TURNS and not stop_evt.is_set():
                    if not _tool_calling_enabled():
                        break
                    turn_count += 1

                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=current_messages,
                            temperature=0.7,
                            stream=False,
                            tools=tools_list,
                            tool_choice="auto",
                            max_tokens=TOOL_LOOP_MAX_TOKENS,
                        )
                    except Exception as e:
                        disabled = _maybe_disable_tool_calling(e)
                        if disabled:
                            log_console(f"Agent Loop: disabling tools for provider compatibility ({e})", "WARN")
                        else:
                            print(f"Agent Loop Error: {e}")
                        break

                    msg_dict = _msg_to_dict(response.choices[0].message)
                    tool_calls = msg_dict.get("tool_calls") or []
                    if tool_calls:
                        tool_calls_made = True
                        current_messages.append(msg_dict)
                        first_tool = tool_calls[0] if tool_calls else {}
                        first_fn = (first_tool.get("function") or {}).get("name") or first_tool.get("name")
                        if first_fn:
                            log_console(f"Tool Call: {first_fn}", "AGENT")
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text("SYS:THINKING: Consulting memory..."),
                            loop,
                        )

                        tool_texts = []
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
                            elif fn_name == "list_session_files":
                                result = tools.tool_list_session_files(
                                    session_id,
                                    db_lock=db.db_lock,
                                )
                            elif fn_name == "read_corpus_doc":
                                doc_id = args.get("doc_id")
                                if doc_id is None:
                                    result = "Error: read_corpus_doc requires doc_id."
                                else:
                                    result = tools.tool_read_corpus_doc(
                                        int(doc_id),
                                        db_lock=db.db_lock,
                                    )

                            result_str = str(result)
                            remain = MAX_TOOL_OUTPUT_CHARS - tool_chars_used
                            if remain <= 0:
                                result_str = "[TOOL_BUDGET_EXCEEDED]"
                            else:
                                result_str = result_str[:remain]
                            tool_chars_used += len(result_str)
                            tool_texts.append(result_str)

                            tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                            tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
                            current_messages.append(tool_msg)
                        all_tool_texts.extend(tool_texts)

                        if not auto_hop_used and tool_texts:
                            hop = _maybe_auto_hop(
                                user_text,
                                "\n".join(tool_texts),
                                session_id,
                                memory,
                                metrics,
                            )
                            if hop:
                                auto_hop_used = True
                                asyncio.run_coroutine_threadsafe(
                                    websocket.send_text("SYS:THINKING: Following bridge..."),
                                    loop,
                                )
                                hop_msg = {
                                    "role": "tool",
                                    "tool_call_id": f"auto-hop-{turn_count}",
                                    "content": str(hop["result"]),
                                }
                                current_messages.append(hop_msg)
                                all_tool_texts.append(str(hop["result"]))
                    else:
                        break

            if not stop_evt.is_set():
                try:
                    final_messages = current_messages
                    evidence_ok = True
                    evidence_items = []
                    evidence_lines = []
                    if is_deep_mode:
                        if not tool_calls_made:
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_text("SYS:THINKING: Consulting memory..."),
                                loop,
                            )
                            forced = _force_tool_call(
                                user_text,
                                session_id,
                                memory,
                                metrics,
                                file_payloads=file_payloads,
                            )
                            tool_calls_made = True
                            forced_text = str(forced)
                            all_tool_texts.append(forced_text)
                            current_messages.append({
                                "role": "tool",
                                "tool_call_id": "forced-tool",
                                "content": forced_text[:MAX_TOOL_OUTPUT_CHARS],
                            })

                        evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches = _evidence_check(
                            user_text, all_tool_texts
                        )
                        evidence_ok, evidence_reason, evidence_items, evidence_lines, evidence_matches, tool_chars_used = _apply_hops(
                            user_text,
                            session_id,
                            memory,
                            metrics,
                            current_messages,
                            all_tool_texts,
                            tool_chars_used,
                            evidence_ok,
                            evidence_reason,
                            evidence_items,
                            evidence_lines,
                            evidence_matches,
                        )
                        metrics["evidence"] = {
                            "ok": evidence_ok,
                            "reason": evidence_reason,
                            "lines": evidence_lines[:10],
                            "items": evidence_items[:10],
                            "matches": evidence_matches,
                            "tool_calls": tool_calls_made or bool(metrics.get("rlm_hops")),
                        }
                        if not (tool_calls_made or metrics.get("rlm_hops")):
                            evidence_ok = False
                            metrics["evidence"]["ok"] = False
                            metrics["evidence"]["reason"] = "no_tool_calls"

                        if not evidence_ok:
                            final_text_buffer = "not found"
                        else:
                            final_messages = current_messages + [{
                                "role": "system",
                                "content": (
                                    "Answer ONLY from the tool evidence above. "
                                    "If evidence is missing, reply exactly: not found. "
                                    "Do not call tools."
                                )
                            }]
                            if evidence_lines:
                                final_messages.append({
                                    "role": "system",
                                    "content": f"EVIDENCE LINES:\n{chr(10).join(evidence_lines[:10])}",
                                })

                    if not is_deep_mode or evidence_ok:
                        stream_start = time.time()
                        use_stream = not (is_deep_mode and not RLM_STREAM)
                        if use_stream:
                            stream = client.chat.completions.create(
                                model=model_name,
                                messages=final_messages,
                                temperature=0.2 if is_deep_mode else 0.7,
                                stream=True
                            )

                            delta_buffer = ""
                            last_flush = time.monotonic()
                            current_sentence = ""
                            saw_delta = False
                            first_token_logged = False

                            def handle_token(token: str):
                                nonlocal final_text_buffer, delta_buffer, last_flush, current_sentence, saw_delta, first_token_logged
                                if metrics.get("ttft") is None:
                                    metrics["ttft"] = time.time() - stream_start
                                saw_delta = True
                                if not first_token_logged:
                                    first_token_logged = True
                                    if RLM_TRACE:
                                        try:
                                            log_console("STREAM: first token emitted", "TRACE")
                                        except Exception:
                                            pass
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
                                    if sentence_endings.search(current_sentence[-2:]) and len(current_sentence.strip()) > 5:
                                        raw_t = current_sentence.strip()
                                        clean_t = re.sub(r"[*#_`~]+", "", raw_t).strip()
                                        if clean_t:
                                            enqueue_text(clean_t)
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
                                handle_token(token)

                            if emit_text_deltas and delta_buffer and not stop_evt.is_set():
                                send_delta(delta_buffer)
                            if generate_audio and current_sentence.strip() and not stop_evt.is_set():
                                raw_t = current_sentence.strip()
                                clean_t = re.sub(r"[*#_`~]+", "", raw_t).strip()
                                if clean_t:
                                    enqueue_text(clean_t)

                            if SSE_STREAM_FALLBACK and not saw_delta and not stop_evt.is_set():
                                if RLM_TRACE:
                                    try:
                                        log_console("STREAM: SSE fallback engaged", "TRACE")
                                    except Exception:
                                        pass
                                stream_start = time.time()
                                delta_buffer = ""
                                last_flush = time.monotonic()
                                current_sentence = ""
                                sse_url = LM_STUDIO_URL.rstrip("/") + "/chat/completions"
                                payload = {
                                    "model": model_name,
                                    "messages": final_messages,
                                    "temperature": 0.2 if is_deep_mode else 0.7,
                                    "stream": True,
                                }
                                saw_delta = _stream_sse_chat(
                                    sse_url,
                                    payload,
                                    handle_token,
                                    stop_evt,
                                    LLM_TIMEOUT_S,
                                )

                                if emit_text_deltas and delta_buffer and not stop_evt.is_set():
                                    send_delta(delta_buffer)
                                if generate_audio and current_sentence.strip() and not stop_evt.is_set():
                                    raw_t = current_sentence.strip()
                                    clean_t = re.sub(r"[*#_`~]+", "", raw_t).strip()
                                    if clean_t:
                                        enqueue_text(clean_t)
                                if not saw_delta and RLM_TRACE:
                                    try:
                                        log_console("STREAM: no deltas after SSE fallback", "TRACE")
                                    except Exception:
                                        pass
                        else:
                            resp = client.chat.completions.create(
                                model=model_name,
                                messages=final_messages,
                                temperature=0.2 if is_deep_mode else 0.7,
                                stream=False
                            )
                            msg = _msg_to_dict(resp.choices[0].message)
                            final_text_buffer = msg.get("content") or _extract_message_text(resp.choices[0].message) or ""
                            if metrics.get("ttft") is None:
                                metrics["ttft"] = time.time() - stream_start
                            if emit_text_deltas and final_text_buffer:
                                send_delta(final_text_buffer)
                            if generate_audio and final_text_buffer:
                                clean_t = re.sub(r"[*#_`~]+", "", final_text_buffer).strip()
                                if clean_t:
                                    enqueue_text(clean_t)

                except Exception as e:
                    print(f"Streaming Error: {e}")

            if (not is_deep_mode) and RLM_FALLBACK_ENABLED and (not generate_audio) and not image_payloads and not file_payloads and not stop_evt.is_set():
                final_text_buffer, _fallback_reason = _run_text_fallback(
                    final_text_buffer,
                    model_name,
                    loop,
                    stop_evt,
                )
            response_holder["text"] = final_text_buffer
            if session_id and store_user_text and final_text_buffer and len(history) == 0:
                threading.Thread(
                    target=_maybe_set_session_title,
                    args=(session_id, store_user_text, final_text_buffer, client, get_current_model),
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
        if is_deep_mode:
            evidence = metrics.get("evidence") or {}
            if evidence.get("ok") and evidence.get("items"):
                enforced, reason = _enforce_evidence_answer(
                    user_text,
                    evidence.get("items", []),
                )
                if reason:
                    evidence["enforced"] = reason
                    metrics["evidence"] = evidence
                    full_reply = enforced or "not found"
        if full_reply:
            metrics["output_chars"] = len(full_reply)
            if metrics.get("output_tokens") is None:
                metrics["output_tokens"] = estimate_tokens_from_text(full_reply)
            history.append({"role": "user", "content": store_user_text})
            history.append({"role": "assistant", "content": full_reply})

            if memory_intents.detect_memory_save(user_text):
                history_snapshot = list(history)
                threading.Thread(
                    target=_save_explicit_memory,
                    args=(history_snapshot, session_id, client, get_current_model, memory),
                ).start()
            if image_payloads:
                file_ids = [f.get("id") for f in (file_payloads or []) if isinstance(f, dict) and f.get("id")]
                threading.Thread(
                    target=db.save_interaction_with_assets,
                    args=(session_id, store_user_text, full_reply, image_payloads, file_ids),
                ).start()
            elif file_payloads:
                file_ids = [f.get("id") for f in (file_payloads or []) if isinstance(f, dict) and f.get("id")]
                threading.Thread(
                    target=db.save_interaction_with_assets,
                    args=(session_id, store_user_text, full_reply, None, file_ids),
                ).start()
            else:
                threading.Thread(target=db.save_interaction, args=(session_id, store_user_text, full_reply)).start()

            metrics["end_time"] = time.time()
            _trace_rlm_metrics(metrics)
            _trace_final_answer(full_reply)
            _print_perf_report(metrics)
            finalize_metrics(metrics, "ok")
            if emit_final_text:
                await websocket.send_text(f"LOG:AI: {full_reply}")
            await websocket.send_text("DONE")
        else:
            metrics["end_time"] = time.time()
            _trace_rlm_metrics(metrics)
            finalize_metrics(metrics, "empty_reply")
            await websocket.send_text("DONE")
    else:
        metrics["end_time"] = time.time()
        _trace_rlm_metrics(metrics)
        finalize_metrics(metrics, "aborted")
        await websocket.send_text("LOG: --- ABORTED ---")


__all__ = ["process_and_stream_response"]
