import time


def _print_recursion_debug(label, query, debug, results):
    print(f"[FTS-RECURSION] {label} query={query}", flush=True)
    print(
        "[FTS-RECURSION] root_tokens="
        f"{debug.get('root_tokens')} strong_tokens={debug.get('strong_tokens')}",
        flush=True,
    )
    print(
        "[FTS-RECURSION] depth="
        f"{debug.get('depth')} max_queries={debug.get('max_queries')} "
        f"max_branches={debug.get('max_branches')} oversample={debug.get('oversample')} "
        f"min_match_tokens={debug.get('min_match_tokens')}",
        flush=True,
    )
    for item in debug.get("queries", []):
        print(
            "[FTS-RECURSION] query_run="
            f"{item.get('query')} allow_or={item.get('allow_or')} results={item.get('results')}",
            flush=True,
        )
    for item in debug.get("subqueries", []):
        print(
            "[FTS-RECURSION] subqueries depth_left="
            f"{item.get('depth_left')} query={item.get('query')} subs={item.get('subqueries')}",
            flush=True,
        )
    print(
        "[FTS-RECURSION] counts "
        f"raw={debug.get('raw_count')} deduped={debug.get('deduped_count')} "
        f"filtered={debug.get('filtered_count')} returned={debug.get('returned_count')} "
        f"queries_used={debug.get('query_count')}",
        flush=True,
    )
    for row in results:
        print(
            "[FTS-RECURSION] hit role="
            f"{row.get('role')} score={row.get('score')} content={row.get('content')}",
            flush=True,
        )
def _seed_message(server, session_id, content, ts_offset=0.0):
    ts = time.time() + ts_offset
    with server.db_lock:
        server.db_conn.execute(
            "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", content, 0, ts)
        )
        server.db_conn.commit()


def test_fts_recursive_split_terms(server):
    session_id = server.create_session("seed", None, None, None)
    _seed_message(server, session_id, "alpha beta signal one")
    _seed_message(server, session_id, "gamma delta signal two", ts_offset=1.0)

    query = "alpha beta gamma delta"
    strict = server.fts_search_messages(
        query,
        session_id=session_id,
        conn=server.db_conn,
        lock=server.db_lock,
        allow_or_fallback=False,
    )
    print(f"[FTS-RECURSION] strict_query hits={len(strict)} query={query}", flush=True)
    assert strict == []

    debug = {}
    rec = server.fts_recursive_search(
        query,
        session_id=session_id,
        conn=server.db_conn,
        lock=server.db_lock,
        depth=2,
        debug=debug,
    )
    _print_recursion_debug("split_terms", query, debug, rec)
    contents = [r["content"] for r in rec]
    assert any("alpha beta" in c for c in contents)
    assert any("gamma delta" in c for c in contents)


def test_fts_recursive_filters_singletons(server):
    session_id = server.create_session("seed", None, None, None)
    _seed_message(server, session_id, "alpha solo token")
    _seed_message(server, session_id, "alpha beta paired signal", ts_offset=1.0)

    query = "alpha beta gamma"
    debug = {}
    rec = server.fts_recursive_search(
        query,
        session_id=session_id,
        conn=server.db_conn,
        lock=server.db_lock,
        depth=2,
        max_branches=8,
        min_match_tokens=2,
        debug=debug,
    )
    _print_recursion_debug("filter_singletons", query, debug, rec)
    contents = [r["content"] for r in rec]
    assert any("alpha beta paired signal" in c for c in contents)
    assert all("alpha solo token" not in c for c in contents)
