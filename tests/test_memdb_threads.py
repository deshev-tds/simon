import time


def test_memdb_hot_session_sync(server):
    original_limit = server.db.MEM_HOT_SESSION_LIMIT
    server.db.MEM_HOT_SESSION_LIMIT = 10
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

    server.db.sync_mem_db_for_session(session_id, limit=server.db.MEM_HOT_SESSION_LIMIT, lock=server.db_lock)

    with server.db_lock:
        mem_count = server.mem_conn.execute(
            "SELECT COUNT(1) FROM messages WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
    assert mem_count == 10
    assert server.db.mem_active_session_id == session_id

    with server.db_lock:
        row = server.mem_conn.execute(
            "SELECT 1 FROM messages WHERE session_id=? AND content LIKE ?",
            (session_id, "%TOKEN0030%")
        ).fetchone()
    assert row is not None
    with server.db_lock:
        row = server.mem_conn.execute(
            "SELECT 1 FROM messages WHERE session_id=? AND content LIKE ?",
            (session_id, "%TOKEN0001%")
        ).fetchone()
    assert row is None

    other_session = server.create_session("other", None, None, None)
    server.save_interaction(other_session, "hello", "ok")
    with server.db_lock:
        other_rows = server.mem_conn.execute(
            "SELECT COUNT(1) FROM messages WHERE session_id=?",
            (other_session,),
        ).fetchone()[0]
    assert other_rows == 0
    server.db.MEM_HOT_SESSION_LIMIT = original_limit
