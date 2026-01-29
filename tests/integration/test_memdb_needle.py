import concurrent.futures
import json

import pytest


_DISCONNECT = object()
_TIMEOUT = object()


def _receive_with_timeout(ws, timeout_s):
    def _worker():
        try:
            return ws.receive()
        except Exception as exc:
            msg = str(exc)
            if "disconnect message" in msg:
                return _DISCONNECT
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker)
        try:
            item = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return _TIMEOUT
    if isinstance(item, Exception):
        raise item
    return item


def _ws_connect(app_client, path, timeout_s=5.0):
    def _worker():
        cm = app_client.websocket_connect(path)
        ws = cm.__enter__()
        return cm, ws

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker)
        try:
            item = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            pytest.fail("WebSocket connection timed out.")
    return item


def _recv_until_done(ws, max_steps=2000, timeout_s=15.0):
    texts = []
    for _ in range(max_steps):
        msg = _receive_with_timeout(ws, timeout_s=timeout_s)
        if msg is _TIMEOUT:
            pytest.fail("WebSocket receive timed out while waiting for DONE.")
        if msg is _DISCONNECT:
            pytest.fail("WebSocket disconnected while waiting for DONE.")
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
        if msg is _TIMEOUT:
            break
        if msg is _DISCONNECT:
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
    original_anchor = server.ANCHOR_MESSAGES
    original_recent = server.MAX_RECENT_MESSAGES
    server.RAG_THRESHOLD = 0.0
    server.ANCHOR_MESSAGES = 5
    server.MAX_RECENT_MESSAGES = 5
    cm = None
    try:
        import uuid

        needle = f"TOKEN_{uuid.uuid4().hex}"
        marker = f"MARKER_{uuid.uuid4().hex}"

        # Assert needle/marker are not already in persistent stores.
        with server.db_lock:
            row = server.db_conn.execute(
                "SELECT 1 FROM messages WHERE content LIKE ? LIMIT 1",
                (f"%{needle}%",)
            ).fetchone()
        assert row is None
        with server.db_lock:
            row = server.db_conn.execute(
                "SELECT 1 FROM messages WHERE content LIKE ? LIMIT 1",
                (f"%{marker}%",)
            ).fetchone()
        assert row is None
        try:
            docs, _, _ = server.memory.search(needle, n_results=1)
            assert not docs
            docs, _, _ = server.memory.search(marker, n_results=1)
            assert not docs
        except Exception:
            # If vector memory is disabled or unavailable, skip.
            pass

        cm, ws = _ws_connect(app_client, "/ws", timeout_s=5.0)
        session_id = _get_session_id(ws)
        window_first = 5
        window_last = 5
        history_window_size = max(
            window_first + window_last,
            server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES
        )
        total_turns = 50
        needle_index = max(12, total_turns // 2)
        decoy_indices = set()
        for offset in (-3, -2, -1, 1, 2, 3, 4, 5):
            idx = needle_index + offset
            if 1 <= idx <= total_turns and idx != needle_index:
                decoy_indices.add(idx)
            if len(decoy_indices) >= 5:
                break

        def _mutate_marker(value: str, replacement: str) -> str:
            if not value:
                return replacement
            if len(value) == 1:
                return replacement
            return value[:-2] + replacement + value[-1]

        for i in range(1, total_turns + 1):
            if i == needle_index:
                user_text = (
                    f"Seed {i} {marker} {needle} synthetic payload. "
                    "Reply with exactly: OK"
                )
            elif i in decoy_indices:
                decoy_char = "x" if marker[-1].lower() != "x" else "y"
                decoy_marker = _mutate_marker(marker, decoy_char)
                user_text = (
                    f"Seed {i} {decoy_marker} NOT_THE_TOKEN synthetic payload. "
                    "Reply with exactly: OK"
                )
            elif i == needle_index + 6 and i <= total_turns:
                user_text = (
                    f"Seed {i} {marker} WRONG_TOKEN synthetic payload. "
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

        # Integration-level coupling: this relies on fts_preview format.
        payload = json.loads(rag_msgs[-1][4:])
        preview_lines = []
        for item in payload:
            if isinstance(item, dict) and "fts_preview" in item:
                preview_lines.extend(item["fts_preview"])

        assert any(needle in line for line in preview_lines), "FTS preview missing needle token."
        rag_hit_line = next((line for line in preview_lines if needle in line), "")

        ai_chunks = [m for m in messages if m.startswith("LOG:AI:")]
        ai_text = " ".join(m.split("LOG:AI:", 1)[1].strip() for m in ai_chunks)
        assert ai_text.strip() == needle

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
        server.ANCHOR_MESSAGES = original_anchor
        server.MAX_RECENT_MESSAGES = original_recent
