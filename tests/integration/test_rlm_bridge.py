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
def test_bridge_multi_hop_reasoning(app_client, server):
    """
    Bridge Test (RLM-lite):
    - Fact A: Omega-Protocol -> CRIMSON-77 (deep history)
    - Distractors push Fact A out of the last 10 messages
    - Fact B: Project Highlander -> Omega-Protocol (recent)
    - Query asks for Project Highlander's code, requiring a multi-hop lookup
    """
    target_code = "CRIMSON-77"
    bridge_entity = "Omega-Protocol"
    target_project = "Project Highlander"

    fact_a = f"CONFIDENTIAL: The {bridge_entity} authentication code is {target_code}."
    fact_b = f"UPDATE: {target_project} relies exclusively on the {bridge_entity} for access control."

    distractors = [
        f"Log Entry {i}: The Alpha-Protocol uses code AZURE-{i}." for i in range(8)
    ]

    with app_client.websocket_connect("/ws") as ws:
        session_id = _get_session_id(ws)
        assert session_id is not None

        ws.send_text(f"{fact_a} Reply: OK")
        _recv_until_done(ws, timeout_s=15.0)

        for d in distractors:
            ws.send_text(f"{d} Reply: OK")
            _recv_until_done(ws, timeout_s=15.0)

        ws.send_text(f"{fact_b} Reply: OK")
        _recv_until_done(ws, timeout_s=15.0)

    time.sleep(1.0)

    history = server.load_session_messages(session_id)
    assert any(target_code in (m.get("content") or "") for m in history)
    tail = history[-10:]
    assert all(target_code not in (m.get("content") or "") for m in tail)
    assert any(bridge_entity in (m.get("content") or "") for m in tail)

    metrics_before = len(server.METRICS_HISTORY)

    query = (
        f"research: You must call the search_memory tool before answering. "
        f"Find the authentication code for {target_project}. "
        f"First confirm which protocol it uses, then search memory for that protocol's code. "
        "Reply with the code only."
    )

    with app_client.websocket_connect("/ws") as ws:
        _get_session_id(ws)
        ws.send_text(f"SESSION:{session_id}")
        _expect_session(ws, session_id)

        ws.send_text(query)
        messages = _recv_until_done(ws, timeout_s=60.0)

    ai_reply = next((m for m in reversed(messages) if m.startswith("LOG:AI:")), "")
    ai_reply_clean = ai_reply.replace("LOG:AI:", "").strip()

    tool_msgs = [m for m in messages if "Consulting memory" in m]
    assert tool_msgs, "Expected search_memory tool usage in deep mode."

    assert target_code in ai_reply_clean, (
        f"Bridge failed. Expected {target_code}, got '{ai_reply_clean}'."
    )

    with server.METRICS_LOCK:
        last_metrics = server.METRICS_HISTORY[-1] if server.METRICS_HISTORY else {}
    if len(server.METRICS_HISTORY) > metrics_before:
        gate = last_metrics.get("rlm_gate", {})
        assert gate.get("trigger") is True
