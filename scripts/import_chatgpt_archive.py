#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import backend.db as db  # noqa: E402
from backend.config import DATA_DIR, DB_PATH  # noqa: E402


IMAGE_TOKEN_RE = re.compile(r"\bturn\\d+image\\d+\\b")
STATE_PATH = DATA_DIR / "chatgpt_archive_state.json"
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
NON_TEXT_CONTENT_TYPES = {
    "code",
    "execution_output",
    "thoughts",
    "user_editable_context",
    "reasoning_recap",
    "tether_quote",
    "tether_browsing_display",
    "app_pairing_content",
    "system_error",
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("", "").replace("", "").replace("", "")
    cleaned = IMAGE_TOKEN_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _extract_text(message: dict) -> str | None:
    if not message:
        return None
    content = message.get("content") or {}
    content_type = content.get("content_type")
    if content_type in NON_TEXT_CONTENT_TYPES:
        return None
    if content_type not in ("text", "multimodal_text"):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts = [p for p in parts if isinstance(p, str)]
    if not text_parts:
        return None
    return _clean_text("\n".join(text_parts))


def _build_main_path(mapping: dict, current_node: str | None) -> list[str]:
    if not current_node or current_node not in mapping:
        return []
    path = []
    seen = set()
    node_id = current_node
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(node_id)
        node_id = mapping[node_id].get("parent")
    path.reverse()
    return path


def _merge_consecutive(messages: list[dict]) -> list[dict]:
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
            msg_ts = msg.get("ts")
            if msg_ts is not None:
                prev_ts = merged[-1].get("ts")
                if prev_ts is None or msg_ts < prev_ts:
                    merged[-1]["ts"] = msg_ts
            continue
        merged.append(dict(msg))
    return merged


def _normalize_timestamps(messages: list[dict], fallback_ts: float | None) -> None:
    if not messages:
        return
    baseline = messages[0].get("ts") or fallback_ts or time.time()
    last_ts = baseline - 0.001
    for msg in messages:
        ts = msg.get("ts")
        if ts is None or ts < last_ts:
            ts = last_ts + 0.001
        msg["ts"] = ts
        last_ts = ts


def _build_turns(messages: list[dict], max_code_ratio: float | None) -> tuple[list[dict], int]:
    turns = []
    current_user = None
    skipped_code = 0
    for msg in messages:
        role = msg["role"]
        if role == "user":
            if current_user is None:
                current_user = {"text": msg["content"], "ts": msg["ts"]}
            else:
                current_user["text"] += "\n" + msg["content"]
            continue
        if role == "assistant" and current_user is not None:
            ratio = _code_ratio(msg["content"])
            if (
                max_code_ratio is not None
                and 0 <= max_code_ratio < 1
                and ratio > max_code_ratio
            ):
                skipped_code += 1
                current_user = None
                continue
            turns.append({
                "user": current_user["text"],
                "assistant": msg["content"],
                "ts": msg["ts"],
            })
            current_user = None
    return turns, skipped_code


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append(text[start:end])
        if end >= total:
            break
        start = max(0, end - overlap)
    return chunks


def _code_ratio(text: str) -> float:
    if not text:
        return 0.0
    total_len = len(text.strip())
    if total_len <= 0:
        return 0.0
    code_len = sum(len(m.group(0)) for m in CODE_FENCE_RE.finditer(text))
    return code_len / total_len


def _wipe_local_data(wipe: bool) -> None:
    if not wipe:
        return
    if DB_PATH.exists():
        DB_PATH.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = DB_PATH.with_name(DB_PATH.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    chroma_dir = DATA_DIR / "simon_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)


def _iter_conversations(data, limit: int | None):
    if limit is None:
        for conv in data:
            yield conv
        return
    count = 0
    for conv in data:
        if count >= limit:
            break
        yield conv
        count += 1


def _parse_since(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if re.fullmatch(r"\d{4}-\d{2}", value):
        dt = datetime.strptime(value, "%Y-%m").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if re.fullmatch(r"\d{4}", value):
        dt = datetime.strptime(value, "%Y").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    try:
        return float(value)
    except Exception:
        return None


def _year_to_ts(year: int | None) -> float | None:
    if year is None:
        return None
    try:
        dt = datetime(int(year), 1, 1, tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def main() -> int:
    default_json = ROOT_DIR.parent / "gpt-persona" / "conversations.json"
    parser = argparse.ArgumentParser(description="Import ChatGPT archive into Simon.")
    parser.add_argument("--json", default=str(default_json), help="Path to conversations.json")
    parser.add_argument("--wipe", action="store_true", help="Wipe local SQLite + Chroma before import")
    parser.add_argument("--skip-vector", action="store_true", help="Skip Chroma import")
    parser.add_argument("--limit", type=int, default=None, help="Max conversations to import")
    parser.add_argument(
        "--since",
        default=None,
        help="Earliest conversation date (YYYY-MM, YYYY-MM-DD, YYYY, or epoch seconds)",
    )
    parser.add_argument("--since-last", action="store_true", help="Use last import timestamp from state file")
    parser.add_argument("--state-file", default=str(STATE_PATH), help="Path to import state file")
    parser.add_argument("--min-year", type=int, default=None, help="Earliest year (YYYY)")
    parser.add_argument("--chunk-size", type=int, default=1400, help="Chroma doc chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chroma chunk overlap")
    parser.add_argument("--batch-size", type=int, default=64, help="Chroma batch size")
    parser.add_argument("--commit-every", type=int, default=25, help="SQL commit cadence")
    parser.add_argument(
        "--max-code-ratio",
        type=float,
        default=0.8,
        help="Skip assistant turns where code fence ratio exceeds this (vectors only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse only; do not write")
    args = parser.parse_args()

    json_path = Path(args.json).expanduser()
    if not json_path.exists():
        print(f"[ERR] conversations.json not found: {json_path}")
        return 1

    if args.wipe:
        print(f"[INFO] Wiping DB at {DB_PATH} and Chroma at {DATA_DIR / 'simon_db'}")
        if not args.dry_run:
            _wipe_local_data(wipe=True)

    if args.dry_run:
        print("[INFO] Dry-run enabled; no data will be written.")

    since_ts = _parse_since(args.since)
    if since_ts is None:
        since_ts = _year_to_ts(args.min_year)
    if args.since and since_ts is None:
        print(f"[ERR] Unable to parse --since value: {args.since}")
        return 1
    if args.since_last and since_ts is None:
        state_path = Path(args.state_file).expanduser()
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                since_ts = float(state.get("last_import_ts")) if state.get("last_import_ts") else None
            except Exception:
                since_ts = None

    conn = None
    if not args.dry_run:
        conn = db.init_db()
        try:
            db._ensure_fts5(conn)
        except Exception as exc:
            print(f"[WARN] FTS5 setup failed: {exc}")

    archive_collection = None
    if not args.skip_vector and not args.dry_run:
        try:
            import backend.memory as memory_mod  # noqa: E402
            existing = getattr(memory_mod, "memory", None)
            if existing is not None and hasattr(existing, "archive_collection"):
                archive_collection = existing.archive_collection
            else:
                memory = memory_mod.MemoryManager()
                archive_collection = memory.archive_collection
        except Exception as exc:
            print(f"[WARN] Vector memory unavailable ({exc}). Skipping Chroma import.")
            archive_collection = None

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_convs = len(data)
    print(f"[INFO] Loaded {total_convs} conversations.")

    conv_count = 0
    msg_count = 0
    vector_count = 0
    skipped_old = 0
    skipped_no_ts = 0
    skipped_code_turns = 0
    latest_seen_ts = None

    batch_docs = []
    batch_metas = []
    batch_ids = []

    def flush_vectors():
        nonlocal vector_count
        if not archive_collection or not batch_docs:
            return
        archive_collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
        )
        vector_count += len(batch_docs)
        batch_docs.clear()
        batch_metas.clear()
        batch_ids.clear()

    def commit_sql():
        if conn is not None:
            conn.commit()

    for idx, conv in enumerate(_iter_conversations(data, args.limit), start=1):
        mapping = conv.get("mapping") or {}
        path = _build_main_path(mapping, conv.get("current_node"))
        if not path:
            continue

        conv_ts = conv.get("update_time") or conv.get("create_time")
        if conv_ts is not None:
            try:
                conv_ts = float(conv_ts)
            except Exception:
                conv_ts = None
        if conv_ts is not None:
            latest_seen_ts = conv_ts if latest_seen_ts is None else max(latest_seen_ts, conv_ts)
        if since_ts is not None:
            if conv_ts is None:
                skipped_no_ts += 1
                continue
            if float(conv_ts) < float(since_ts):
                skipped_old += 1
                continue

        messages = []
        for node_id in path:
            node = mapping.get(node_id) or {}
            msg = node.get("message") or {}
            role = (msg.get("author") or {}).get("role")
            if role not in ("user", "assistant"):
                continue
            text = _extract_text(msg)
            if not text:
                continue
            messages.append({
                "role": role,
                "content": text,
                "ts": msg.get("create_time"),
            })

        if not messages:
            continue

        messages = _merge_consecutive(messages)
        conv_create = conv.get("create_time")
        _normalize_timestamps(messages, conv_create)

        if args.dry_run:
            conv_count += 1
            msg_count += len(messages)
            continue

        title = conv.get("title") or ""
        conv_id = conv.get("conversation_id") or conv.get("id") or ""
        model = conv.get("default_model_slug")
        created_at = conv_create or messages[0]["ts"] or time.time()
        updated_at = conv.get("update_time") or messages[-1]["ts"] or created_at

        metadata = {
            "source": "chatgpt",
            "conversation_id": conv_id,
            "current_node": conv.get("current_node"),
            "create_time": conv.get("create_time"),
            "update_time": conv.get("update_time"),
            "default_model_slug": conv.get("default_model_slug"),
            "is_archived": conv.get("is_archived"),
            "is_starred": conv.get("is_starred"),
            "conversation_origin": conv.get("conversation_origin"),
            "voice": conv.get("voice"),
            "is_do_not_remember": conv.get("is_do_not_remember"),
        }

        cur = conn.execute(
            "INSERT INTO sessions(title, summary, tags, model, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                title,
                "",
                "chatgpt_archive",
                model,
                json.dumps(metadata, ensure_ascii=False),
                float(created_at),
                float(updated_at),
            ),
        )
        session_id = cur.lastrowid

        message_rows = [
            (
                session_id,
                m["role"],
                m["content"],
                0,
                None,
                float(m["ts"]),
            )
            for m in messages
        ]
        conn.executemany(
            "INSERT INTO messages(session_id, role, content, tokens, audio_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            message_rows,
        )
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?",
            (float(messages[-1]["ts"]), session_id),
        )

        conv_count += 1
        msg_count += len(messages)

        if archive_collection:
            turns, skipped_here = _build_turns(messages, args.max_code_ratio)
            for turn_index, turn in enumerate(turns):
                full_text = f"User: {turn['user']}\nAI: {turn['assistant']}"
                chunks = _chunk_text(full_text, args.chunk_size, args.chunk_overlap)
                for chunk_index, chunk in enumerate(chunks):
                    doc_id = f"{conv_id}:{turn_index}:{chunk_index}"
                    batch_docs.append(chunk)
                    batch_metas.append({
                        "source": "chatgpt",
                        "conversation_id": conv_id,
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "chunk_index": chunk_index,
                        "timestamp": float(turn["ts"]),
                    })
                    batch_ids.append(doc_id)
                if len(batch_docs) >= args.batch_size:
                    flush_vectors()
            skipped_code_turns += skipped_here

        if args.commit_every and conv_count % args.commit_every == 0:
            flush_vectors()
            commit_sql()
            print(f"[INFO] Imported {conv_count} sessions, {msg_count} messages...")

    flush_vectors()
    commit_sql()

    if args.since_last and not args.dry_run and latest_seen_ts is not None:
        state_path = Path(args.state_file).expanduser()
        try:
            state_path.write_text(
                json.dumps(
                    {"last_import_ts": latest_seen_ts, "updated_at": time.time()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] Failed to write archive state file: {exc}")

    print(
        "[DONE] Sessions: {0} | Messages: {1} | Archive vectors: {2}".format(
            conv_count, msg_count, vector_count
        )
    )
    if since_ts is not None:
        print(f"[INFO] Skipped (older than since): {skipped_old}")
        print(f"[INFO] Skipped (missing timestamp): {skipped_no_ts}")
    if args.max_code_ratio is not None and 0 <= args.max_code_ratio < 1:
        print(f"[INFO] Skipped code-heavy turns (vectors): {skipped_code_turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
