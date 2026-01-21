import json
import queue
import threading
import textwrap

import pytest


TABLE_WIDTHS = {
    "step": 24,
    "status": 12,
    "detail": 46,
    "raw": 60,
}


def _table_line():
    return "+" + "+".join("-" * (w + 2) for w in TABLE_WIDTHS.values()) + "+"


def _table_print_header():
    line = _table_line()
    print(line, flush=True)
    print(
        "| {step:<{sw}} | {status:<{stw}} | {detail:<{dw}} | {raw:<{rw}} |".format(
            step="Step",
            status="Status",
            detail="Detail",
            raw="Raw Response",
            sw=TABLE_WIDTHS["step"],
            stw=TABLE_WIDTHS["status"],
            dw=TABLE_WIDTHS["detail"],
            rw=TABLE_WIDTHS["raw"],
        ),
        flush=True,
    )
    print(line, flush=True)


def _table_print_row(step, status, detail="", raw=""):
    step_lines = textwrap.wrap(step or "", TABLE_WIDTHS["step"]) or [""]
    status_lines = textwrap.wrap(status or "", TABLE_WIDTHS["status"]) or [""]
    detail_lines = textwrap.wrap(detail or "", TABLE_WIDTHS["detail"]) or [""]
    raw_lines = textwrap.wrap(raw or "", TABLE_WIDTHS["raw"]) or [""]
    rows = max(len(step_lines), len(status_lines), len(detail_lines), len(raw_lines))
    for i in range(rows):
        print(
            "| {step:<{sw}} | {status:<{stw}} | {detail:<{dw}} | {raw:<{rw}} |".format(
                step=step_lines[i] if i < len(step_lines) else "",
                status=status_lines[i] if i < len(status_lines) else "",
                detail=detail_lines[i] if i < len(detail_lines) else "",
                raw=raw_lines[i] if i < len(raw_lines) else "",
                sw=TABLE_WIDTHS["step"],
                stw=TABLE_WIDTHS["status"],
                dw=TABLE_WIDTHS["detail"],
                rw=TABLE_WIDTHS["raw"],
            ),
            flush=True,
        )


def _table_print_footer():
    print(_table_line(), flush=True)


def _receive_with_timeout(ws, timeout_s):
    q = queue.Queue()

    def _worker():
        try:
            q.put(ws.receive())
        except Exception as exc:
            q.put(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        item = q.get(timeout=timeout_s)
    except queue.Empty:
        return None
    if isinstance(item, Exception):
        raise item
    return item


def _ws_connect(app_client, path, timeout_s=5.0):
    q = queue.Queue()

    def _worker():
        try:
            cm = app_client.websocket_connect(path)
            ws = cm.__enter__()
            q.put((cm, ws))
        except Exception as exc:
            q.put(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        item = q.get(timeout=timeout_s)
    except queue.Empty:
        pytest.fail("WebSocket connection timed out.")
    if isinstance(item, Exception):
        raise item
    return item


def _recv_until_done(ws, max_steps=2000, timeout_s=15.0):
    texts = []
    for _ in range(max_steps):
        msg = _receive_with_timeout(ws, timeout_s=timeout_s)
        if msg is None:
            pytest.fail("WebSocket receive timed out while waiting for DONE.")
        text = msg.get("text")
        if text is None:
            continue
        texts.append(text)
        if text == "DONE":
            break
    return texts


def _get_session_id(ws, timeout_s=5.0):
    session_id = None
    for _ in range(10):
        msg = _receive_with_timeout(ws, timeout_s=timeout_s)
        if msg is None:
            break
        text = msg.get("text", "")
        if text.startswith("SYS:SESSION:"):
            session_id = int(text.split("SYS:SESSION:", 1)[1])
            break
    if session_id is None:
        pytest.fail("WebSocket did not provide a session id.")
    return session_id


@pytest.mark.integration
def test_memdb_needle_rag_payload(app_client, server):
    original_threshold = server.RAG_THRESHOLD
    server.RAG_THRESHOLD = 0.0
    cm = None
    try:
        _table_print_header()
        cm, ws = _ws_connect(app_client, "/ws", timeout_s=5.0)
        session_id = _get_session_id(ws)
        _table_print_row("connect ws", "ok", f"session_id={session_id}")

        needle = "TOKEN1234"
        marker = "MARKERXYZ"
        window_first = 10
        window_last = 10
        history_window_size = max(
            window_first + window_last,
            server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES
        )
        total_turns = max(30, (history_window_size // 2) + 6)
        needle_index = max(12, total_turns // 2)

        for i in range(1, total_turns + 1):
            if i == needle_index:
                user_text = (
                    f"Seed {i} {marker} {needle} synthetic payload. "
                    "Reply with exactly: OK"
                )
            else:
                user_text = f"Seed {i} synthetic payload. Reply with exactly: OK"
            ws.send_text(user_text)
            _recv_until_done(ws)
            if i == needle_index:
                _table_print_row(
                    "send needle",
                    "ok",
                    f"sending {needle} with {marker}",
                    user_text,
                )
            if i % 10 == 0 or i == total_turns:
                _table_print_row(
                    "saturate context",
                    "progress",
                    f"sent {i}/{total_turns} messages",
                )

        history = server.load_session_messages(session_id)
        assert len(history) > history_window_size
        assert len(history) > server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES
        _table_print_row(
            "context size",
            "ok",
            f"history={len(history)} window={history_window_size}",
        )

        needle_indices = [idx for idx, msg in enumerate(history) if needle in (msg.get("content") or "")]
        assert needle_indices
        earliest = min(needle_indices)
        assert earliest >= window_first
        assert earliest < (len(history) - window_last)
        _table_print_row("needle outside window", "ok", f"index={earliest}")

        ws.send_text(
            f"What token was paired with {marker}? "
            "Read the recalled evidence and reply with the token only."
        )
        messages = _recv_until_done(ws)

        rag_msgs = [m for m in messages if m.startswith("RAG:")]
        assert rag_msgs, "Missing RAG payload for needle query."

        payload = json.loads(rag_msgs[-1][4:])
        preview_lines = []
        for item in payload:
            if isinstance(item, dict) and "fts_preview" in item:
                preview_lines.extend(item["fts_preview"])

        assert any(needle in line for line in preview_lines), "FTS preview missing needle token."
        _table_print_row(
            "recall from rag",
            "ok",
            f"recalling {needle}",
            rag_msgs[-1],
        )

        ai_chunks = [m for m in messages if m.startswith("LOG:AI:")]
        ai_text = " ".join(m.split("LOG:AI:", 1)[1].strip() for m in ai_chunks)
        assert needle in ai_text
        _table_print_row(
            "recall from model",
            "ok",
            f"recalling {needle}",
            ai_text,
        )
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
        server.RAG_THRESHOLD = original_threshold
        _table_print_footer()
