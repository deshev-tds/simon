import json
import queue
import threading

import pytest


def _receive_with_timeout(ws, timeout_s):
    q = queue.Queue()

    def _worker():
        try:
            q.put(ws.receive())
        except Exception as exc:
            msg = str(exc)
            if "disconnect message" in msg:
                q.put(None)
            else:
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
        cm, ws = _ws_connect(app_client, "/ws", timeout_s=5.0)
        session_id = _get_session_id(ws)

        needle = "TOKEN1234"
        marker = "MARKERXYZ"
        window_first = 10
        window_last = 10
        history_window_size = max(
            window_first + window_last,
            server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES
        )
        total_turns = max(60, history_window_size)
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

        history = server.load_session_messages(session_id)
        assert len(history) > history_window_size
        assert len(history) > server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES
        assert len(history) >= history_window_size * 2

        needle_indices = [idx for idx, msg in enumerate(history) if needle in (msg.get("content") or "")]
        assert needle_indices
        earliest = min(needle_indices)
        assert earliest >= window_first
        assert earliest < (len(history) - window_last)

        query = (
            f"What token was paired with {marker}? "
            "Read the recalled evidence and reply with the token only."
        )
        metrics = server.init_metrics("text", session_id)
        context_msgs, _ = server.build_rag_context(
            query,
            history,
            server.memory,
            metrics,
            session_id,
        )
        non_system_msgs = [m for m in context_msgs if m.get("role") != "system"]
        assert len(non_system_msgs) <= (window_first + window_last)
        assert all(needle not in (m.get("content") or "") for m in non_system_msgs)

        purge_resp = app_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Reset any prior context."},
                    {"role": "user", "content": "OK"},
                ],
                "temperature": 0.0,
            },
        )
        assert purge_resp.status_code == 200
        ws.send_text(query)
        messages = _recv_until_done(ws)

        rag_msgs = [m for m in messages if m.startswith("RAG:")]
        assert rag_msgs, "Missing RAG payload for needle query."

        payload = json.loads(rag_msgs[-1][4:])
        preview_lines = []
        for item in payload:
            if isinstance(item, dict) and "fts_preview" in item:
                preview_lines.extend(item["fts_preview"])

        assert any(needle in line for line in preview_lines), "FTS preview missing needle token."
        rag_hit_line = next((line for line in preview_lines if needle in line), "")

        ai_chunks = [m for m in messages if m.startswith("LOG:AI:")]
        ai_text = " ".join(m.split("LOG:AI:", 1)[1].strip() for m in ai_chunks)
        assert needle in ai_text

        with server.METRICS_LOCK:
            last_metrics = server.METRICS_HISTORY[-1] if server.METRICS_HISTORY else {}
        input_tokens = last_metrics.get("input_tokens", "n/a")

        print(f"[INTEGRATION] tokens_sent={input_tokens}", flush=True)
        print(
            (
                "[INTEGRATION] limits="
                f"anchor={server.ANCHOR_MESSAGES} "
                f"recent={server.MAX_RECENT_MESSAGES} "
                f"window={history_window_size}"
            ),
            flush=True,
        )
        print(f"[INTEGRATION] needle_query={query}", flush=True)
        print(f"[INTEGRATION] rag_hit={rag_hit_line}", flush=True)
        print(f"[INTEGRATION] model_reply={ai_text}", flush=True)
        ws.close()
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
        server.RAG_THRESHOLD = original_threshold
