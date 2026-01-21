import threading
import time


def _wait_for_condition(fn, timeout_s=5.0, sleep_s=0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(sleep_s)
    return False


def test_memdb_seed_and_prune_threads(server):
    server.MEM_SEED_LIMIT = 30
    server.MEM_MAX_ROWS = 10
    server.MEM_PRUNE_INTERVAL_S = 1

    session_id = server.create_session("seed", None, None, None)
    base_ts = time.time() - 3600
    with server.db_lock:
        for i in range(1, 31):
            server.db_conn.execute(
                "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, "user", f"seed TOKEN{i:04d}", 0, base_ts + i)
            )
        server.db_conn.commit()

    with server.db_lock:
        mem_count = server.mem_conn.execute("SELECT COUNT(1) FROM messages").fetchone()[0]
    assert mem_count == 0

    stop_event = threading.Event()
    threads = server.start_mem_threads(stop_event=stop_event)
    assert threads is not None

    def mem_count_now():
        with server.db_lock:
            return server.mem_conn.execute("SELECT COUNT(1) FROM messages").fetchone()[0]

    assert _wait_for_condition(lambda: mem_count_now() >= 30)

    hits = server.fts_search_messages(
        "TOKEN0005",
        session_id=session_id,
        conn=server.mem_conn,
        lock=server.db_lock,
    )
    assert hits

    assert _wait_for_condition(lambda: mem_count_now() <= 10)

    stop_event.set()
    for t in threads:
        t.join(timeout=1)
    server._mem_threads_started = False
