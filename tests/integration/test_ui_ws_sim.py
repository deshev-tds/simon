import queue
import threading
import time

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


def _recv_until_done(ws, max_steps=2000, timeout_s=30.0):
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


def _expect_session(ws, expected_id, timeout_s=5.0):
    for _ in range(20):
        msg = _receive_with_timeout(ws, timeout_s=timeout_s)
        if msg is None:
            break
        text = msg.get("text", "")
        if text.startswith("SYS:SESSION:"):
            session_id = int(text.split("SYS:SESSION:", 1)[1])
            if session_id == expected_id:
                return session_id
    pytest.fail(f"WebSocket did not switch to session {expected_id}.")


@pytest.mark.integration
def test_ui_ws_recall_name_bridge(app_client, server):
    """
    UI-simulated recall test (websocket):
    - Seed a fact about Mr. Deshev's mother having another child (Olga Petrova).
    - Push the fact out of the anchor/recent window with noise.
    - Ask a "research:" question to trigger deep mode and evidence-based extraction.
    """
    fact_a = "Update: Mr. Deshev is the boss. Reply: OK"
    fact_b = "Update: Mr. Deshev's mother has another child: Olga Petrova. Reply: OK"
    target = "Olga Petrova"

    noise = [
        f"Note {i}: The pipeline syncs logs in the primary cluster. Reply: OK"
        for i in range(12)
    ]

    with app_client.websocket_connect("/ws") as ws:
        session_id = _get_session_id(ws)

        ws.send_text(fact_a)
        _recv_until_done(ws, timeout_s=90.0)

        ws.send_text(fact_b)
        _recv_until_done(ws, timeout_s=90.0)

        for line in noise:
            ws.send_text(line)
            _recv_until_done(ws, timeout_s=90.0)

    time.sleep(1.0)

    history = server.load_session_messages(session_id)
    assert any(target in (m.get("content") or "") for m in history)
    tail = history[-(server.ANCHOR_MESSAGES + server.MAX_RECENT_MESSAGES):]
    assert all(target not in (m.get("content") or "") for m in tail)

    query = (
        "research: What is the name of the the other child of Mr. Deshev's mother? "
        "Reply with the name only."
    )

    with app_client.websocket_connect("/ws") as ws:
        _get_session_id(ws)
        ws.send_text(f"SESSION:{session_id}")
        _expect_session(ws, session_id)

        ws.send_text(query)
        messages = _recv_until_done(ws, timeout_s=60.0)

    ai_reply = next((m for m in reversed(messages) if m.startswith("LOG:AI:")), "")
    ai_reply_clean = ai_reply.replace("LOG:AI:", "").strip()

    assert ai_reply_clean == target
