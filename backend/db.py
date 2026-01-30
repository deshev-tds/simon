import re
import sqlite3
import threading
import time

from backend.config import (
    ANCHOR_MESSAGES,
    DB_PATH,
    CORPUS_DB_PATH,
    DEBUG_MODE,
    FTS_MAX_HITS,
    FTS_MIN_TOKEN_LEN,
    FTS_PER_SESSION,
    FTS_RECURSIVE_DEPTH,
    FTS_RECURSIVE_MAX_BRANCHES,
    FTS_RECURSIVE_MAX_QUERIES,
    FTS_RECURSIVE_MIN_MATCH,
    FTS_RECURSIVE_OVERSAMPLE,
    MAX_RECENT_MESSAGES,
    MEM_MAX_ROWS,
    MEM_PRUNE_INTERVAL_S,
    MEM_SEED_LIMIT,
    MEM_HOT_SESSION_LIMIT,
    TEST_MODE,
)

db_lock = threading.Lock()
db_conn = None
mem_conn = None
corpus_conn = None
mem_active_session_id = None

_word_re = re.compile(r"\w+", re.UNICODE)
_STOP_TOKENS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "our", "say", "that", "the", "this", "to", "us", "was",
    "were", "what", "when", "where", "who", "why", "with", "you", "your",
}


def _resolve_conn(conn):
    target = conn or db_conn
    if target is None:
        raise RuntimeError("Database connection is not initialized.")
    return target


def _resolve_lock(lock):
    return lock or db_lock


def init_db(db_path=DB_PATH):
    global db_conn
    needs_reset = False
    existing_conn = None

    if db_path.exists():
        try:
            existing_conn = sqlite3.connect(db_path, check_same_thread=False)
            existing_conn.execute("PRAGMA foreign_keys = ON;")
            existing_conn.execute("PRAGMA journal_mode=WAL;")
            existing_conn.execute("PRAGMA synchronous=NORMAL;")
            existing_conn.execute("PRAGMA busy_timeout=5000;")

            sess_cols = {r[1] for r in existing_conn.execute("PRAGMA table_info('sessions')").fetchall()}
            msg_cols = {r[1] for r in existing_conn.execute("PRAGMA table_info('messages')").fetchall()}
            required_sessions = {"id", "title", "summary", "tags", "model", "metadata", "created_at", "updated_at"}
            required_messages = {"id", "session_id", "role", "content", "tokens", "audio_path", "created_at"}
            if not required_sessions.issubset(sess_cols) or not required_messages.issubset(msg_cols):
                needs_reset = True
        except Exception:
            needs_reset = True

        if needs_reset:
            try:
                existing_conn.close()
            except Exception:
                pass
            db_path.unlink(missing_ok=True)
        else:
            db_conn = existing_conn
            return existing_conn

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            model TEXT,
            metadata TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            updated_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            audio_path TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")

    conn.commit()
    db_conn = conn
    return conn


def _ensure_fts5(conn: sqlite3.Connection):
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __fts5_test")
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] SQLite FTS5 unavailable: {e}")
        return False

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            session_id UNINDEXED,
            role UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    conn.execute("DROP TRIGGER IF EXISTS messages_ai")
    conn.execute("DROP TRIGGER IF EXISTS messages_ad")
    conn.execute("DROP TRIGGER IF EXISTS messages_au")
    conn.execute("""
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, session_id, role)
            VALUES (new.id, new.content, new.session_id, new.role);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER messages_au AFTER UPDATE OF content, session_id, role ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content, session_id, role)
            VALUES (new.id, new.content, new.session_id, new.role);
        END;
    """)

    try:
        cur = conn.execute("SELECT COUNT(1) FROM messages_fts")
        existing = cur.fetchone()[0]
        if existing == 0:
            conn.execute("""
                INSERT INTO messages_fts(rowid, content, session_id, role)
                SELECT id, content, session_id, role FROM messages
            """)
    except Exception:
        pass

    conn.commit()
    return True


def init_corpus_db(db_path=CORPUS_DB_PATH):
    global corpus_conn
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            created_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC)")

    conn.execute("DROP TRIGGER IF EXISTS documents_ai")
    conn.execute("DROP TRIGGER IF EXISTS documents_ad")
    conn.execute("DROP TRIGGER IF EXISTS documents_au")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content,
            source UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)
    conn.execute("""
        CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, content, source)
            VALUES (new.id, new.content, new.source);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
            DELETE FROM documents_fts WHERE rowid = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER documents_au AFTER UPDATE OF content, source ON documents BEGIN
            DELETE FROM documents_fts WHERE rowid = old.id;
            INSERT INTO documents_fts(rowid, content, source)
            VALUES (new.id, new.content, new.source);
        END;
    """)
    conn.commit()
    corpus_conn = conn
    return conn


def init_mem_db():
    global mem_conn
    conn = sqlite3.connect(
        "file:memdb?mode=memory&cache=shared",
        uri=True,
        check_same_thread=False
    )
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA journal_mode=MEMORY;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            model TEXT,
            metadata TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            updated_at REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            audio_path TEXT,
            created_at REAL DEFAULT (strftime('%s','now')),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")

    conn.commit()
    try:
        _ensure_fts5(conn)
    except Exception as _e:
        if DEBUG_MODE:
            print(f"[WARN] In-memory FTS5 setup failed: {_e}")
    mem_conn = conn
    return conn


def reset_mem_db(conn=None, lock=None):
    global mem_active_session_id
    target_conn = conn or mem_conn
    if target_conn is None:
        return
    target_lock = _resolve_lock(lock)
    with target_lock:
        target_conn.execute("DELETE FROM messages")
        target_conn.execute("DELETE FROM sessions")
        target_conn.commit()
    mem_active_session_id = None


def seed_mem_db_for_session(
    session_id: int,
    limit: int = MEM_HOT_SESSION_LIMIT,
    disk_conn=None,
    mem_conn=None,
    lock=None,
):
    if limit <= 0:
        return
    target_lock = _resolve_lock(lock)
    target_mem = mem_conn if mem_conn is not None else globals().get("mem_conn")
    if target_mem is None:
        return
    source_conn = disk_conn or db_conn
    if source_conn is None:
        return

    with target_lock:
        rows = source_conn.execute(
            """
            SELECT session_id, role, content, tokens, audio_path, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(session_id), int(limit)),
        ).fetchall()

        if not rows:
            return
        rows = list(reversed(rows))
        target_mem.executemany(
            "INSERT INTO messages(session_id, role, content, tokens, audio_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        target_mem.commit()


def sync_mem_db_for_session(session_id: int, limit: int = MEM_HOT_SESSION_LIMIT, lock=None):
    global mem_active_session_id
    reset_mem_db(lock=lock)
    seed_mem_db_for_session(session_id, limit=limit, lock=lock)
    mem_active_session_id = int(session_id)


def seed_mem_db_from_disk(disk_conn, mem_conn, limit=MEM_SEED_LIMIT, batch_size=1000, seed_before_ts=None, lock=None):
    if limit <= 0:
        return
    if mem_conn is None:
        return
    target_lock = _resolve_lock(lock)
    ro_conn = None
    src_conn = None
    close_src = False
    try:
        if disk_conn is not None:
            src_conn = disk_conn
        else:
            ro_conn = sqlite3.connect(
                f"file:{DB_PATH}?mode=ro",
                uri=True,
                check_same_thread=False
            )
            ro_conn.execute("PRAGMA busy_timeout=5000;")
            src_conn = ro_conn
            close_src = True
        params = []
        sql = """
            SELECT session_id, role, content, tokens, audio_path, created_at
            FROM messages
        """
        if seed_before_ts is not None:
            sql += " WHERE created_at <= ?"
            params.append(float(seed_before_ts))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        rows = src_conn.execute(sql, tuple(params)).fetchall()
        if not rows:
            return
        rows = list(reversed(rows))
        if DEBUG_MODE:
            print(f"[MEM] Seeding in-memory FTS with {len(rows)} rows...")

        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            with target_lock:
                mem_conn.executemany(
                    "INSERT INTO messages(session_id, role, content, tokens, audio_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    chunk
                )
                mem_conn.commit()
        if DEBUG_MODE:
            print("[MEM] In-memory FTS seed complete.")
    except Exception as _e:
        if DEBUG_MODE:
            print(f"[WARN] In-memory FTS seed failed: {_e}")
    finally:
        try:
            if close_src and src_conn:
                src_conn.close()
        except Exception:
            pass


def prune_mem_db(mem_conn, max_rows=MEM_MAX_ROWS, interval_s=MEM_PRUNE_INTERVAL_S, stop_event=None, lock=None):
    if max_rows <= 0:
        return
    if mem_conn is None:
        return
    if interval_s is None:
        interval_s = MEM_PRUNE_INTERVAL_S
    if TEST_MODE or interval_s < 5:
        interval_s = max(1, interval_s)
    else:
        interval_s = max(5, interval_s)
    target_lock = _resolve_lock(lock)
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        time.sleep(interval_s)
        if stop_event is not None and stop_event.is_set():
            break
        try:
            with target_lock:
                cur = mem_conn.execute("SELECT COUNT(1) FROM messages")
                count = cur.fetchone()[0] or 0
                if count <= max_rows:
                    continue
                cutoff_row = mem_conn.execute(
                    "SELECT created_at FROM messages ORDER BY created_at DESC LIMIT 1 OFFSET ?",
                    (max_rows - 1,)
                ).fetchone()
                if not cutoff_row:
                    continue
                cutoff_ts = cutoff_row[0]
                mem_conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff_ts,))
                mem_conn.commit()
            if DEBUG_MODE:
                print(f"[MEM] Pruned in-memory FTS to {max_rows} rows.")
        except Exception as _e:
            if DEBUG_MODE:
                print(f"[WARN] In-memory FTS prune failed: {_e}")


def start_mem_threads(
    db_conn,
    mem_conn,
    mem_seed_limit=MEM_SEED_LIMIT,
    mem_max_rows=MEM_MAX_ROWS,
    mem_prune_interval_s=MEM_PRUNE_INTERVAL_S,
    stop_event=None,
    db_lock=None,
):
    if db_conn is None or mem_conn is None:
        return None
    if mem_seed_limit <= 0 and mem_max_rows <= 0:
        return None
    target_lock = _resolve_lock(db_lock)
    _mem_seed_before_ts = time.time()
    seed_thread = threading.Thread(
        target=seed_mem_db_from_disk,
        args=(db_conn, mem_conn, mem_seed_limit, 1000, _mem_seed_before_ts),
        kwargs={"lock": target_lock},
        daemon=True
    )
    prune_thread = threading.Thread(
        target=prune_mem_db,
        args=(mem_conn, mem_max_rows, mem_prune_interval_s, stop_event),
        kwargs={"lock": target_lock},
        daemon=True
    )
    seed_thread.start()
    prune_thread.start()
    return seed_thread, prune_thread


def create_session(title=None, model=None, summary=None, tags=None, conn=None, lock=None):
    ts = time.time()
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        cur = target_conn.execute(
            "INSERT INTO sessions(title, summary, tags, model, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, summary or "", tags or "", model, None, ts, ts)
        )
        target_conn.commit()
        return cur.lastrowid


def list_sessions(limit=50, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        cur = target_conn.execute(
            """
            SELECT s.id, COALESCE(s.title, '') as title, COALESCE(s.summary, '') as summary,
                   COALESCE(s.tags, '') as tags, s.model, s.created_at, s.updated_at
            FROM sessions s
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cur.fetchall()
        return [{
            "id": r[0],
            "title": r[1],
            "summary": r[2],
            "tags": r[3],
            "model": r[4],
            "created_at": r[5],
            "updated_at": r[6],
        } for r in rows]


def get_session_meta(session_id, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        cur = target_conn.execute(
            """
            SELECT id, COALESCE(title,''), COALESCE(summary,''), COALESCE(tags,''), model, created_at, updated_at
            FROM sessions WHERE id=?
            """,
            (session_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "summary": row[2],
        "tags": row[3],
        "model": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def session_exists(session_id, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        cur = target_conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,))
        return cur.fetchone() is not None


def touch_session(session_id, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        target_conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), session_id))
        target_conn.commit()


def load_session_messages(session_id, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        cur = target_conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        )
        return [{"role": r[0], "content": r[1]} for r in cur.fetchall()]


def get_session_transcript(session_id, max_chars=6000, conn=None, lock=None):
    try:
        msgs = load_session_messages(session_id, conn=conn, lock=lock)
        if not msgs:
            return ""
        lines = []
        total_len = 0
        for m in msgs:
            line = f"{m['role'].upper()}: {m['content']}"
            if total_len + len(line) > max_chars:
                break
            lines.append(line)
            total_len += len(line)
        return "\n".join(lines)
    except Exception:
        return ""


def save_interaction(session_id, user_text, ai_text, conn=None, mem=None, lock=None):
    ts = time.time()
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    target_mem = mem if mem is not None else mem_conn
    with target_lock:
        target_conn.execute(
            "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", user_text, 0, ts)
        )
        target_conn.execute(
            "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", ai_text, 0, ts)
        )
        target_conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, session_id))
        target_conn.commit()
        if target_mem is not None:
            allow_mem_write = mem is not None or (
                mem_active_session_id is not None and int(session_id) == int(mem_active_session_id)
            )
            if not allow_mem_write:
                return
            try:
                target_mem.execute(
                    "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, "user", user_text, 0, ts)
                )
                target_mem.execute(
                    "INSERT INTO messages(session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, "assistant", ai_text, 0, ts)
                )
                target_mem.commit()
                if MEM_HOT_SESSION_LIMIT > 0:
                    cur = target_mem.execute(
                        "SELECT COUNT(1) FROM messages WHERE session_id=?",
                        (int(session_id),),
                    )
                    count = cur.fetchone()[0] or 0
                    if count > MEM_HOT_SESSION_LIMIT:
                        cutoff_row = target_mem.execute(
                            """
                            SELECT created_at FROM messages
                            WHERE session_id=?
                            ORDER BY created_at DESC
                            LIMIT 1 OFFSET ?
                            """,
                            (int(session_id), int(MEM_HOT_SESSION_LIMIT - 1)),
                        ).fetchone()
                        if cutoff_row:
                            cutoff_ts = cutoff_row[0]
                            target_mem.execute(
                                "DELETE FROM messages WHERE session_id=? AND created_at < ?",
                                (int(session_id), cutoff_ts),
                            )
                            target_mem.commit()
            except Exception as _e:
                if DEBUG_MODE:
                    print(f"[WARN] In-memory DB insert failed: {_e}")


def set_session_title(session_id, title, conn=None, lock=None):
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)
    with target_lock:
        target_conn.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, time.time(), session_id))
        target_conn.commit()


def get_session_window(session_id, anchor=ANCHOR_MESSAGES, recent=MAX_RECENT_MESSAGES, conn=None, lock=None):
    meta = get_session_meta(session_id, conn=conn, lock=lock)
    if not meta:
        return None

    anchor = max(0, anchor)
    recent = max(0, recent)
    target_conn = _resolve_conn(conn)
    target_lock = _resolve_lock(lock)

    with target_lock:
        anchor_rows = target_conn.execute(
            "SELECT id, role, content, created_at, tokens FROM messages WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
            (session_id, anchor)
        ).fetchall()
        recent_rows = target_conn.execute(
            "SELECT id, role, content, created_at, tokens FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, recent)
        ).fetchall()

    anchor_ids = {r[0] for r in anchor_rows}
    recent_rows = [r for r in reversed(recent_rows) if r[0] not in anchor_ids]

    def map_row(row):
        return {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": row[3],
            "tokens": row[4] or 0,
        }

    return {
        "session": meta,
        "anchors": [map_row(r) for r in anchor_rows],
        "recents": [map_row(r) for r in recent_rows],
    }


def _fts_tokenize(text: str) -> list[str]:
    if not text:
        return []
    toks = [t.lower() for t in _word_re.findall(text)]
    toks = [t for t in toks if len(t) >= FTS_MIN_TOKEN_LEN]
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _filter_stop_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP_TOKENS]


def _fts_sanitize_query(text: str, join_op=" ") -> str:
    toks = _fts_tokenize(text)
    return join_op.join(toks[:12])


def _build_subqueries(tokens: list[str], max_branches: int) -> list[str]:
    if not tokens:
        return []
    max_branches = max(1, int(max_branches))
    compound_limit = max(1, max_branches - 1) if max_branches > 1 else 1
    candidates = []
    if len(tokens) >= 2:
        mid = len(tokens) // 2
        left = " ".join(tokens[:mid])
        right = " ".join(tokens[mid:])
        if left:
            candidates.append(left)
        if right and right != left:
            candidates.append(right)
        for i in range(len(tokens) - 1):
            candidates.append(f"{tokens[i]} {tokens[i + 1]}")
    for tok in tokens:
        candidates.append(tok)

    seen = set()
    out = []
    for cand in candidates:
        if not cand or cand in seen:
            continue
        if len(out) < compound_limit:
            seen.add(cand)
            out.append(cand)
            continue
        break

    if len(out) < max_branches:
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= max_branches:
                break
    return out


def fts_search_messages(
    query_text: str,
    session_id: int | None = None,
    limit: int = FTS_MAX_HITS,
    cutoff_ts: float | None = None,
    conn: sqlite3.Connection | None = None,
    lock=None,
    allow_or_fallback: bool = True,
):
    q = _fts_sanitize_query(query_text, " ")
    if not q:
        return []

    def run_query(match_query):
        sql = """
            SELECT m.id, m.role, m.content, m.created_at, m.session_id, bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
        """
        params = [match_query]
        if FTS_PER_SESSION and session_id is not None:
            sql += " AND m.session_id = ?"
            params.append(int(session_id))
        if cutoff_ts is not None:
            sql += " AND m.created_at >= ?"
            params.append(float(cutoff_ts))
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(int(max(1, min(limit, 50))))

        try:
            target_conn = conn or db_conn
            if target_conn is None:
                return []
            target_lock = lock or db_lock
            with target_lock:
                return target_conn.execute(sql, tuple(params)).fetchall()
        except Exception:
            return []

    rows = run_query(q)
    if allow_or_fallback and not rows:
        q_or = _fts_sanitize_query(query_text, " OR ")
        if q_or and q_or != q:
            rows = run_query(q_or)

    return [{
        "id": r[0],
        "role": r[1],
        "content": r[2],
        "created_at": r[3],
        "session_id": r[4],
        "score": float(r[5]) if r[5] is not None else None
    } for r in rows]


def _count_token_matches(text: str, tokens: list[str]) -> int:
    if not text or not tokens:
        return 0
    content_toks = {t.lower() for t in _word_re.findall(text)}
    return sum(1 for tok in tokens if tok in content_toks)


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    merged = {}
    for r in rows:
        key = (r.get("session_id"), r.get("created_at"), r.get("role"), r.get("content"))
        existing = merged.get(key)
        if existing is None:
            merged[key] = r
            continue
        score = r.get("score")
        existing_score = existing.get("score")
        if existing_score is None:
            if score is not None:
                merged[key] = r
            continue
        if score is not None and score < existing_score:
            merged[key] = r
    return list(merged.values())


def _dedupe_corpus_rows(rows: list[dict]) -> list[dict]:
    merged = {}
    for r in rows:
        key = (r.get("source"), r.get("created_at"), r.get("content"))
        existing = merged.get(key)
        if existing is None:
            merged[key] = r
            continue
        score = r.get("score")
        existing_score = existing.get("score")
        if existing_score is None:
            if score is not None:
                merged[key] = r
            continue
        if score is not None and score < existing_score:
            merged[key] = r
    return list(merged.values())


def fts_search_corpus(
    query_text: str,
    limit: int = FTS_MAX_HITS,
    cutoff_ts: float | None = None,
    conn: sqlite3.Connection | None = None,
    lock=None,
    allow_or_fallback: bool = True,
):
    q = _fts_sanitize_query(query_text, " ")
    if not q:
        return []

    def run_query(match_query):
        sql = """
            SELECT d.content, d.created_at, d.id, d.source, bm25(documents_fts) AS score
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
        """
        params = [match_query]
        if cutoff_ts is not None:
            sql += " AND d.created_at >= ?"
            params.append(float(cutoff_ts))
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(int(max(1, min(limit, 50))))

        try:
            target_conn = conn or corpus_conn
            if target_conn is None:
                return []
            target_lock = lock or db_lock
            with target_lock:
                return target_conn.execute(sql, tuple(params)).fetchall()
        except Exception:
            return []

    rows = run_query(q)
    if allow_or_fallback and not rows:
        q_or = _fts_sanitize_query(query_text, " OR ")
        if q_or and q_or != q:
            rows = run_query(q_or)

    return [{
        "content": r[0],
        "created_at": r[1],
        "doc_id": r[2],
        "source": r[3],
        "score": float(r[4]) if r[4] is not None else None,
    } for r in rows]


def fts_recursive_search(
    query_text: str,
    session_id: int | None = None,
    limit: int = FTS_MAX_HITS,
    cutoff_ts: float | None = None,
    conn: sqlite3.Connection | None = None,
    lock=None,
    depth: int = FTS_RECURSIVE_DEPTH,
    max_queries: int = FTS_RECURSIVE_MAX_QUERIES,
    max_branches: int = FTS_RECURSIVE_MAX_BRANCHES,
    oversample: int = FTS_RECURSIVE_OVERSAMPLE,
    min_match_tokens: int = FTS_RECURSIVE_MIN_MATCH,
    debug: dict | None = None,
):
    root_tokens = _fts_tokenize(query_text)
    strong_tokens = _filter_stop_tokens(root_tokens) or root_tokens
    if not strong_tokens:
        return []
    target_conn = conn or db_conn
    target_lock = lock or db_lock
    if target_conn is None:
        return []

    query_state = {"count": 0}
    max_queries = max(1, int(max_queries))
    oversample = max(1, int(oversample))

    if debug is not None:
        debug.setdefault("queries", [])
        debug.setdefault("subqueries", [])
        debug["root_tokens"] = root_tokens
        debug["strong_tokens"] = strong_tokens
        debug["depth"] = int(depth)
        debug["max_queries"] = int(max_queries)
        debug["max_branches"] = int(max_branches)
        debug["oversample"] = int(oversample)
        debug["min_match_tokens"] = int(min_match_tokens)

    def run_query(q, allow_or):
        if query_state["count"] >= max_queries:
            return []
        query_state["count"] += 1
        results = fts_search_messages(
            q,
            session_id=session_id,
            limit=limit * oversample,
            cutoff_ts=cutoff_ts,
            conn=target_conn,
            lock=target_lock,
            allow_or_fallback=allow_or,
        )
        if debug is not None:
            debug["queries"].append({
                "query": q,
                "allow_or": bool(allow_or),
                "results": len(results),
            })
        return results

    def search_recursive(q, depth_left):
        rows = run_query(q, allow_or=False)
        if rows:
            return rows
        if depth_left <= 0:
            return run_query(q, allow_or=True)
        sub_tokens = _filter_stop_tokens(_fts_tokenize(q))
        subqueries = _build_subqueries(sub_tokens or strong_tokens, max_branches)
        if debug is not None:
            debug["subqueries"].append({
                "depth_left": int(depth_left),
                "query": q,
                "subqueries": subqueries,
            })
        acc = []
        for sub in subqueries:
            if query_state["count"] >= max_queries:
                break
            acc.extend(search_recursive(sub, depth_left - 1))
            if len(acc) >= limit * oversample:
                break
        return acc

    rows = search_recursive(query_text, int(depth))
    if debug is not None:
        debug["raw_count"] = len(rows)
    if not rows:
        if debug is not None:
            debug["query_count"] = query_state["count"]
            debug["returned_count"] = 0
        return []

    deduped = _dedupe_rows(rows)
    if debug is not None:
        debug["deduped_count"] = len(deduped)
    if len(strong_tokens) >= 1:
        min_match = min(max(1, int(min_match_tokens)), len(strong_tokens))
        filtered = [
            r for r in deduped
            if _count_token_matches(r.get("content", ""), strong_tokens) >= min_match
        ]
        if filtered:
            deduped = filtered
    if debug is not None:
        debug["filtered_count"] = len(deduped)

    deduped.sort(key=lambda r: (r.get("score") is None, r.get("score") or 0))
    final_rows = deduped[:limit]
    if debug is not None:
        debug["query_count"] = query_state["count"]
        debug["returned_count"] = len(final_rows)
    return final_rows


def fts_recursive_search_corpus(
    query_text: str,
    limit: int = FTS_MAX_HITS,
    cutoff_ts: float | None = None,
    conn: sqlite3.Connection | None = None,
    lock=None,
    depth: int = FTS_RECURSIVE_DEPTH,
    max_queries: int = FTS_RECURSIVE_MAX_QUERIES,
    max_branches: int = FTS_RECURSIVE_MAX_BRANCHES,
    oversample: int = FTS_RECURSIVE_OVERSAMPLE,
    min_match_tokens: int = FTS_RECURSIVE_MIN_MATCH,
    debug: dict | None = None,
):
    root_tokens = _fts_tokenize(query_text)
    strong_tokens = _filter_stop_tokens(root_tokens) or root_tokens
    if not strong_tokens:
        return []
    target_conn = conn or corpus_conn
    target_lock = lock or db_lock
    if target_conn is None:
        return []

    query_state = {"count": 0}
    max_queries = max(1, int(max_queries))
    oversample = max(1, int(oversample))

    if debug is not None:
        debug.setdefault("queries", [])
        debug.setdefault("subqueries", [])
        debug["root_tokens"] = root_tokens
        debug["strong_tokens"] = strong_tokens
        debug["depth"] = int(depth)
        debug["max_queries"] = int(max_queries)
        debug["max_branches"] = int(max_branches)
        debug["oversample"] = int(oversample)
        debug["min_match_tokens"] = int(min_match_tokens)

    def run_query(q, allow_or):
        if query_state["count"] >= max_queries:
            return []
        query_state["count"] += 1
        results = fts_search_corpus(
            q,
            limit=limit * oversample,
            cutoff_ts=cutoff_ts,
            conn=target_conn,
            lock=target_lock,
            allow_or_fallback=allow_or,
        )
        if debug is not None:
            debug["queries"].append({
                "query": q,
                "allow_or": bool(allow_or),
                "results": len(results),
            })
        return results

    def search_recursive(q, depth_left):
        rows = run_query(q, allow_or=False)
        if rows:
            return rows
        if depth_left <= 0:
            return run_query(q, allow_or=True)
        sub_tokens = _filter_stop_tokens(_fts_tokenize(q))
        subqueries = _build_subqueries(sub_tokens or strong_tokens, max_branches)
        if debug is not None:
            debug["subqueries"].append({
                "depth_left": int(depth_left),
                "query": q,
                "subqueries": subqueries,
            })
        acc = []
        for sub in subqueries:
            if query_state["count"] >= max_queries:
                break
            acc.extend(search_recursive(sub, depth_left - 1))
            if len(acc) >= limit * oversample:
                break
        return acc

    rows = search_recursive(query_text, int(depth))
    if debug is not None:
        debug["raw_count"] = len(rows)
    if not rows:
        if debug is not None:
            debug["query_count"] = query_state["count"]
            debug["returned_count"] = 0
        return []

    deduped = _dedupe_corpus_rows(rows)
    if debug is not None:
        debug["deduped_count"] = len(deduped)
    if len(strong_tokens) >= 1:
        min_match = min(max(1, int(min_match_tokens)), len(strong_tokens))
        filtered = [
            r for r in deduped
            if _count_token_matches(r.get("content", ""), strong_tokens) >= min_match
        ]
        if filtered:
            deduped = filtered
    if debug is not None:
        debug["filtered_count"] = len(deduped)

    deduped.sort(key=lambda r: (r.get("score") is None, r.get("score") or 0))
    final_rows = deduped[:limit]
    if debug is not None:
        debug["query_count"] = query_state["count"]
        debug["returned_count"] = len(final_rows)
    return final_rows


__all__ = [
    "db_lock",
    "db_conn",
    "mem_conn",
    "corpus_conn",
    "mem_active_session_id",
    "init_db",
    "_ensure_fts5",
    "init_mem_db",
    "init_corpus_db",
    "reset_mem_db",
    "seed_mem_db_for_session",
    "sync_mem_db_for_session",
    "seed_mem_db_from_disk",
    "prune_mem_db",
    "start_mem_threads",
    "create_session",
    "list_sessions",
    "get_session_meta",
    "session_exists",
    "touch_session",
    "load_session_messages",
    "get_session_transcript",
    "save_interaction",
    "set_session_title",
    "get_session_window",
    "_fts_tokenize",
    "_fts_sanitize_query",
    "fts_search_messages",
    "fts_search_corpus",
    "fts_recursive_search",
    "fts_recursive_search_corpus",
]
