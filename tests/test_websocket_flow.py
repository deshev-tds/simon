import json
import time

import numpy as np


def wait_for_condition(fn, timeout_s=3.0, sleep_s=0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(sleep_s)
    return False


def _recv_until_done(ws, max_steps=2000):
    texts = []
    for _ in range(max_steps):
        msg = ws.receive()
        text = msg.get("text")
        if text is None:
            continue
        texts.append(text)
        if text == "DONE":
            break
    return texts


def _get_session_id(ws):
    session_id = None
    for _ in range(5):
        msg = ws.receive()
        text = msg.get("text", "")
        if text.startswith("SYS:SESSION:"):
            session_id = int(text.split("SYS:SESSION:", 1)[1])
            break
    assert session_id is not None
    return session_id


def test_websocket_synthetic_flow(app_client, server):
    with app_client.websocket_connect("/ws") as ws:
        session_id = _get_session_id(ws)

        total_messages = 60
        for i in range(1, total_messages + 1):
            ws.send_text(f"Msg {i} TOKEN{i:04d} synthetic payload.")
            _recv_until_done(ws)

        def counts_ok():
            with server.db_lock:
                disk_count = server.db_conn.execute(
                    "SELECT COUNT(1) FROM messages WHERE session_id=?",
                    (session_id,)
                ).fetchone()[0]
                mem_count = server.mem_conn.execute(
                    "SELECT COUNT(1) FROM messages WHERE session_id=?",
                    (session_id,)
                ).fetchone()[0]
            return disk_count >= total_messages * 2 and mem_count >= total_messages * 2

        assert wait_for_condition(counts_ok, timeout_s=5.0)

        def token_present():
            with server.db_lock:
                row = server.mem_conn.execute(
                    "SELECT 1 FROM messages WHERE session_id=? AND content LIKE ?",
                    (session_id, "%TOKEN0001%")
                ).fetchone()
            return row is not None

        assert wait_for_condition(token_present, timeout_s=3.0)

        ws.send_text("What did I say about TOKEN0001?")
        messages = _recv_until_done(ws)
        rag_msgs = [m for m in messages if m.startswith("RAG:")]
        assert rag_msgs
        payload = json.loads(rag_msgs[-1][4:])
        fts_hits = 0
        for item in payload:
            if isinstance(item, dict) and "fts_hits" in item:
                fts_hits = item["fts_hits"]
        assert fts_hits >= 1

        ws.send_text("research: Use memory to find TOKEN0001")
        messages = _recv_until_done(ws)
        assert any("SYS:THINKING: Consulting memory" in m for m in messages)
        assert server.client.tool_calls_made >= 1


def test_websocket_audio_flow(app_client, server, monkeypatch):
    monkeypatch.setattr(server, "convert_webm_to_numpy", lambda _b: np.zeros(1600, dtype=np.float32))
    with app_client.websocket_connect("/ws") as ws:
        _get_session_id(ws)
        ws.send_bytes(b"dummy")
        ws.send_text("CMD:COMMIT_AUDIO")
        messages = _recv_until_done(ws)
        assert any(m.startswith("LOG:User:") for m in messages)
