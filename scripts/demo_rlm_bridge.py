#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import signal
import sys
import time
import atexit
import json
import urllib.request
import subprocess
import socket
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SIMON_SKIP_AUDIO_MODELS", "1")
os.environ.setdefault("SIMON_SKIP_VECTOR_MEMORY", "1")

db = None
try:
    from backend.metrics import count_tokens as _count_tokens  # type: ignore
except Exception:
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))


WORDS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "kappa",
    "protocol", "system", "deployment", "authentication", "access", "service",
    "infra", "project", "session", "memory", "context", "bridge", "token",
    "pipeline", "runtime", "latency", "throughput", "cache", "storage",
]

SUBJECTS_EN = [
    "The team", "The system", "The process", "The module", "The server",
    "The client", "The session", "The pipeline", "The agent", "The service",
]
VERBS_EN = [
    "processes", "validates", "collects", "syncs", "routes", "sends",
    "stores", "checks", "executes", "observes",
]
OBJECTS_EN = [
    "requests", "events", "data", "logs", "context", "rules",
    "markers", "session metadata", "payloads",
]
ADVERBIALS_EN = [
    "in real time", "under heavy load", "during deployment",
    "after verification", "before startup", "with default settings",
]
TIME_EN = [
    "Today", "Yesterday", "Earlier", "This morning", "Late last night",
]
PLACE_EN = [
    "in production", "in staging", "on the local machine", "in the primary cluster",
]

TEMPLATES_EN = [
    "{subject} {verb} {object} {adverb}.",
    "{time}, {subject} {verb} {object} {place}.",
    "Report {idx}: {subject} {verb} {object} {adverb}.",
    "Note {idx}: {subject} {verb} {object} {place}.",
]

SUBJECTS_BG = [
    "Екипът", "Системата", "Процесът", "Модулът", "Сървърът", "Клиентът",
    "Сесията", "Потокът", "Агентът", "Пайплайнът",
]
VERBS_BG = [
    "обработва", "валидира", "събира", "синхронизира", "маршрутизира",
    "изпраща", "запазва", "проверява", "изпълнява", "наблюдава",
]
OBJECTS_BG = [
    "заявките", "събитията", "данните", "логовете", "контекста",
    "правилата", "маркерите", "сесийните метаданни",
]
ADVERBIALS_BG = [
    "в реално време", "при висок товар", "по време на деплой",
    "след проверка", "преди стартиране", "със стандартни настройки",
]
TIME_BG = [
    "днес", "вчера", "по-рано", "тази сутрин", "късно вечер",
]
PLACE_BG = [
    "в продукция", "в тестова среда", "на локалната машина",
    "в основния клъстер",
]

TEMPLATES_BG = [
    "{subject} {verb} {object} {adverb}.",
    "{time} {subject} {verb} {object} {place}.",
    "Доклад {idx}: {subject} {verb} {object} {adverb}.",
    "Бележка {idx}: {subject} {verb} {object} {place}.",
]


def _now() -> float:
    return time.time()


def _format_rate(count: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "n/a"
    return f"{count / elapsed:,.1f} tok/s"


def _format_eta(seconds: float) -> str:
    if seconds < 0:
        return "n/a"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def _noise_chunk(tokens: int, rng: random.Random, start_idx: int) -> str:
    words = []
    idx = start_idx
    while len(words) < tokens:
        if rng.random() < 0.5:
            template = rng.choice(TEMPLATES_BG)
            sentence = template.format(
                idx=idx,
                subject=rng.choice(SUBJECTS_BG),
                verb=rng.choice(VERBS_BG),
                object=rng.choice(OBJECTS_BG),
                adverb=rng.choice(ADVERBIALS_BG),
                time=rng.choice(TIME_BG),
                place=rng.choice(PLACE_BG),
            )
        else:
            template = rng.choice(TEMPLATES_EN)
            sentence = template.format(
                idx=idx,
                subject=rng.choice(SUBJECTS_EN),
                verb=rng.choice(VERBS_EN),
                object=rng.choice(OBJECTS_EN),
                adverb=rng.choice(ADVERBIALS_EN),
                time=rng.choice(TIME_EN),
                place=rng.choice(PLACE_EN),
            )
        sentence_words = sentence.split()
        words.extend(sentence_words)
        idx += 1
    return " ".join(words[:tokens])


def _build_filler_for_target(target_tokens: int, rng: random.Random, start_idx: int) -> tuple[str, int]:
    if target_tokens <= 0:
        return "", 0
    # Start slightly under target to reduce overflow risk.
    word_budget = max(20, int(target_tokens * 0.8))
    text = _noise_chunk(word_budget, rng, start_idx)
    tok = _count_tokens(text)

    if tok < target_tokens:
        extra_words = int((target_tokens - tok) * 0.8)
        if extra_words > 0:
            extra_text = _noise_chunk(extra_words, rng, start_idx + 10_000)
            text = f"{text} {extra_text}"
            tok = _count_tokens(text)

    # Trim if we overshot.
    if tok > target_tokens:
        words = text.split()
        if words:
            keep = max(20, int(len(words) * (target_tokens / max(1, tok)) * 0.95))
            text = " ".join(words[:keep])
            tok = _count_tokens(text)
    return text, tok


def _chunk_text_by_tokens(text: str, rng: random.Random, min_tokens: int, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    idx = 0
    total = len(words)
    min_tokens = max(10, min_tokens)
    max_tokens = max(min_tokens, max_tokens)
    while idx < total:
        take = rng.randint(min_tokens, max_tokens)
        chunk_words = words[idx:idx + take]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        idx += take
    return chunks


def _seed_fact_with_noise(conn, source: str, fact: str, noise_tokens: int, chunk_size: int, batch_size: int, rng: random.Random):
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    cursor.execute(
        "INSERT INTO documents(source, content, tokens, created_at) VALUES (?, ?, ?, ?)",
        (source, fact, len(fact.split()), _now()),
    )
    conn.commit()

    print(f"[SEED] {source}: inserted fact line ({len(fact.split())} tokens)")

    remaining = noise_tokens
    total = noise_tokens
    batch = []
    inserted = 0
    seed = 0
    t0 = _now()
    last_print = t0

    while remaining > 0:
        take = min(chunk_size, remaining)
        text = _noise_chunk(take, rng, seed)
        seed += 1
        batch.append((source, text, take, _now()))
        remaining -= take
        inserted += take
        if len(batch) >= batch_size:
            cursor.execute("BEGIN")
            cursor.executemany(
                "INSERT INTO documents(source, content, tokens, created_at) VALUES (?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            batch.clear()
        now = _now()
        if now - last_print >= 2.0:
            elapsed = now - t0
            rate = inserted / elapsed if elapsed > 0 else 0.0
            eta_s = (total - inserted) / rate if rate > 0 else -1
            pct = (inserted / total) * 100 if total else 100.0
            print(
                f"[SEED] {source}: {inserted:,}/{total:,} tokens "
                f"({pct:.1f}%) @ {_format_rate(inserted, now - t0)} "
                f"| ETA {_format_eta(eta_s)}"
            )
            last_print = now

    if batch:
        cursor.execute("BEGIN")
        cursor.executemany(
            "INSERT INTO documents(source, content, tokens, created_at) VALUES (?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        batch.clear()

    elapsed = _now() - t0
    print(f"[SEED] {source}: done {total:,} tokens in {elapsed:.1f}s ({_format_rate(total, elapsed)})")


def _probe(conn, query: str):
    if db is None:
        return
    hits = db.fts_recursive_search_corpus(query, limit=5, conn=conn, lock=db.db_lock)
    print(f"[PROBE] Query: {query}")
    if not hits:
        print("[PROBE] No corpus hits.")
        return
    for h in hits:
        content = (h.get("content") or "").replace("\n", " ")
        snippet = content[:180] + ("..." if len(content) > 180 else "")
        score = h.get("score")
        if score is None:
            print(f"  - {snippet}")
        else:
            print(f"  - {snippet} (bm25={score:.2f})")


async def _ws_roundtrip(ws, text: str, timeout_s: float = 120.0):
    await ws.send(text)
    messages = []
    start = time.time()
    while time.time() - start < timeout_s:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            continue
        messages.append(msg)
        if msg == "DONE":
            break
    return messages


async def _ws_roundtrip_timed(ws, text: str, timeout_s: float = 120.0):
    await ws.send(text)
    messages = []
    start = time.time()
    first_token_ts = None
    while time.time() - start < timeout_s:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            continue
        messages.append(msg)
        if first_token_ts is None and isinstance(msg, str) and msg.startswith("LOG:AI:"):
            first_token_ts = time.time()
        if msg == "DONE":
            break
    elapsed = time.time() - start
    ttft = (first_token_ts - start) if first_token_ts is not None else None
    return messages, elapsed, ttft


def _metrics_url_from_ws(ws_url: str) -> str:
    if ws_url.startswith("wss://"):
        base = "https://" + ws_url[len("wss://"):]
    elif ws_url.startswith("ws://"):
        base = "http://" + ws_url[len("ws://"):]
    else:
        base = ws_url
    base = base.replace("/ws", "")
    return f"{base}/metrics?limit=1"


def _fetch_latest_metrics(ws_url: str):
    url = _metrics_url_from_ws(ws_url)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items") or []
            return items[-1] if items else None
    except Exception:
        return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, timeout_s: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _start_server(data_dir: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["SIMON_DATA_DIR"] = str(data_dir)
    env.setdefault("SIMON_SKIP_AUDIO_MODELS", "1")
    env.setdefault("SIMON_SKIP_VECTOR_MEMORY", "1")
    env["PYTHONPATH"] = str(ROOT_DIR)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo: seed large corpus and verify multi-hop bridge via disk FTS.")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data dir for history/corpus DBs.")
    parser.add_argument("--fresh", action="store_true", help="Delete history.db and corpus.db before seeding.")
    parser.add_argument("--tokens-per-stage", type=int, default=500_000, help="Noise tokens per stage.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Tokens per inserted chunk.")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per executemany batch.")
    parser.add_argument("--probe", action="store_true", help="Run corpus FTS probe after seeding.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for noise generation.")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep demo data dir after exit.")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8001/ws", help="WebSocket URL for live query.")
    parser.add_argument("--no-spawn-server", action="store_true", help="Use existing server instead of launching a demo server.")
    parser.add_argument("--server-port", type=int, default=0, help="Port for spawned demo server (0 = auto).")
    parser.add_argument("--no-kvfill", action="store_true", help="Skip KV-cache fill step.")
    parser.add_argument("--fill-tokens", type=int, default=3500, help="Approx tokens to fill the prompt before query.")
    parser.add_argument("--ctx-limit", type=int, default=4096, help="Model context length for KV demo.")
    parser.add_argument("--kv-rounds", type=int, default=2, help="Number of KV fill rounds before query.")
    parser.add_argument("--kv-target", type=float, default=0.85, help="Target fraction of ctx-limit per round.")
    parser.add_argument("--kv-margin", type=int, default=256, help="Token margin to keep under ctx-limit.")
    parser.add_argument(
        "--kv-mode",
        type=str,
        default="overflow",
        choices=["overflow", "safe"],
        help="overflow: each round aims near ctx-limit (triggers rolling if supported); safe: split across rounds.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What's the authentication code for the system we're deploying?",
        help="Query to run after seeding.",
    )
    parser.add_argument(
        "--force-gate",
        action="store_true",
        help="Prefix the query to force Deep Mode in the demo.",
    )
    parser.add_argument(
        "--force-gate-prefix",
        type=str,
        default="research: ",
        help="Prefix used when --force-gate is enabled.",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default="",
        help="Prefix to prepend to the live query (use for gating).",
    )
    parser.add_argument(
        "--query-suffix",
        type=str,
        default="",
        help="Suffix to append to the live query (set to '' to disable).",
    )
    parser.add_argument(
        "--fact-a",
        type=str,
        default="Project Highlander uses the Omega-Protocol for authentication.",
        help="Fact A (bridge) line to seed into corpus.",
    )
    parser.add_argument(
        "--fact-b",
        type=str,
        default="Omega-Protocol authentication code is CRIMSON-77.",
        help="Fact B (code) line to seed into corpus.",
    )
    parser.add_argument(
        "--seed-vectors",
        action="store_true",
        help="Also seed the vector store with Fact A/B (facts only, no noise).",
    )
    args = parser.parse_args()

    cleanup_enabled = not args.no_cleanup
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("/tmp/simon_demo_data")
    os.environ["SIMON_DATA_DIR"] = str(data_dir)
    from backend import db as db_module  # noqa: E402
    global db
    db = db_module

    data_dir = Path(os.environ.get("SIMON_DATA_DIR", str(ROOT_DIR / "backend" / "data")))
    history_db = data_dir / "history.db"
    corpus_db = data_dir / "corpus.db"
    server_proc = None

    def _cleanup():
        if not cleanup_enabled:
            return
        try:
            if server_proc is not None and server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except Exception:
                    server_proc.kill()
            if data_dir.exists():
                shutil.rmtree(data_dir, ignore_errors=True)
                print(f"[CLEANUP] Removed demo data dir: {data_dir}")
        except Exception as exc:
            print(f"[CLEANUP] Failed: {exc}")

    atexit.register(_cleanup)

    def _handle_signal(signum, _frame):
        print(f"[INFO] Received signal {signum}. Cleaning up...")
        _cleanup()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    total_tokens = args.tokens_per_stage * 2
    print("[INFO] This will take a while; live ETA appears once seeding starts.")
    print(f"[INFO] Total noise tokens to seed: {total_tokens:,}.")
    print(f"[INFO] Using data dir: {data_dir}")
    if cleanup_enabled:
        print("[INFO] Auto-cleanup enabled (use --no-cleanup to keep data).")
    else:
        print("[INFO] Auto-cleanup disabled.")
    print("[STEP 1/4] Init databases")
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        if history_db.exists():
            history_db.unlink()
            print(f"[RESET] Deleted {history_db}")
        if corpus_db.exists():
            corpus_db.unlink()
            print(f"[RESET] Deleted {corpus_db}")

    try:
        db.init_db(history_db)
        corpus_conn = db.init_corpus_db(corpus_db)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize databases: {exc}")
        print("[HINT] If the server is running on the same data dir, stop it or use --data-dir to isolate the demo.")
        if cleanup_enabled:
            try:
                if data_dir.exists():
                    shutil.rmtree(data_dir, ignore_errors=True)
            except Exception:
                pass
        return 1

    rng = random.Random(args.seed)

    fact_a = args.fact_a
    fact_b = args.fact_b

    print("[STEP 2/4] Seed Fact A + noise")
    _seed_fact_with_noise(
        corpus_conn,
        source="stage_a",
        fact=fact_a,
        noise_tokens=args.tokens_per_stage,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        rng=rng,
    )

    print("[STEP 3/4] Seed Fact B + noise")
    _seed_fact_with_noise(
        corpus_conn,
        source="stage_b",
        fact=fact_b,
        noise_tokens=args.tokens_per_stage,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        rng=rng,
    )

    if args.seed_vectors:
        print("[STEP 3.5/4] Seed vector memory (facts only)")
        try:
            from backend import memory as memory_mod  # noqa: E402

            mem = memory_mod.memory
            if mem.__class__.__name__ == "_DummyMemory":
                print("[SEED] Vector memory disabled. Set SIMON_SKIP_VECTOR_MEMORY=0 to enable.")
            else:
                mem.save_explicit(fact_a, session_id=0, metadata={"source": "demo_fact_a"})
                mem.save_explicit(fact_b, session_id=0, metadata={"source": "demo_fact_b"})
                print("[SEED] vector: inserted 2 fact lines")
        except Exception as exc:
            print(f"[SEED] Vector seed failed: {exc}")

    print("[STEP 4/4] Probe")
    if args.probe:
        probe_queries = [
            fact_a,
            fact_b,
            args.query,
        ]
        for query in probe_queries:
            _probe(corpus_conn, query)
    else:
        print("[PROBE] Skipped. Run with --probe to verify FTS hits.")

    print("[DONE] Seeding complete. You can now run the live query in the UI.")
    print(f"Example query: \"{args.query}\"")

    if args.no_kvfill:
        return 0

    print("[LIVE] Attempting KV-fill + query via WebSocket...")
    try:
        import websockets  # type: ignore
    except Exception:
        print("[LIVE] websockets not available. Install with: pip install websockets")
        return 0

    if not args.no_spawn_server:
        port = args.server_port or _find_free_port()
        server_proc = _start_server(data_dir, port)
        metrics_url = f"http://127.0.0.1:{port}/metrics?limit=1"
        if not _wait_for_http(metrics_url, timeout_s=25.0):
            print("[LIVE] Failed to start demo server in time. Check logs.")
            if server_proc is not None:
                try:
                    out = server_proc.stdout.read() if server_proc.stdout else ""
                    if out:
                        print(out.strip())
                except Exception:
                    pass
            return 1
        args.ws_url = f"ws://127.0.0.1:{port}/ws"
        print(f"[LIVE] Demo server up at {args.ws_url}")

    async def _run_live():
        async with websockets.connect(args.ws_url, ping_interval=None) as ws:
            # consume SYS:MODEL and SYS:SESSION
            try:
                for _ in range(2):
                    await ws.recv()
            except Exception:
                pass
            if args.kv_rounds > 0:
                base_target = int(args.ctx_limit * args.kv_target) - args.kv_margin
                base_target = max(256, base_target)
                if args.kv_mode == "safe":
                    base_target = max(256, base_target // max(1, args.kv_rounds))
                print(
                    f"[LIVE] KV-fill mode={args.kv_mode} rounds={args.kv_rounds} "
                    f"target≈{base_target} tokens/round (ctx={args.ctx_limit})."
                )
                if args.kv_mode == "overflow":
                    print("[LIVE] Tip: enable LM Studio context shift for rolling cache.")

                for i in range(args.kv_rounds):
                    filler, est_tok = _build_filler_for_target(base_target, rng, i * 1000)
                    print(f"[LIVE] Filling prompt round {i+1}/{args.kv_rounds} (~{est_tok} tokens)...")
                    chunk_min = max(150, base_target // 6)
                    chunk_max = max(chunk_min + 50, base_target // 3)
                    chunks = _chunk_text_by_tokens(filler, rng, chunk_min, chunk_max)
                    if not chunks:
                        chunks = [filler]
                    total_parts = len(chunks)
                    for part_idx, chunk in enumerate(chunks, start=1):
                        part_tokens = _count_tokens(chunk)
                        filler_msg = (
                            f"FILLER ROUND {i+1}/{args.kv_rounds} PART {part_idx}/{total_parts}: "
                            f"{chunk} Reply with OK."
                        )
                        print(f"[LIVE]  - part {part_idx}/{total_parts} (~{part_tokens} tokens)")
                        await _ws_roundtrip(ws, filler_msg, timeout_s=180.0)
                    metrics = _fetch_latest_metrics(args.ws_url)
                    if metrics:
                        print(
                            f"[LIVE] Round {i+1} metrics: input_tokens={metrics.get('input_tokens')} "
                            f"output_tokens={metrics.get('output_tokens')}"
                        )
                        if metrics.get("output_tokens") is None:
                            print("[LIVE] Warning: output_tokens missing; possible context overflow.")
            else:
                filler = _noise_chunk(args.fill_tokens, rng, 0)
                filler_msg = f"FILLER: {filler} Reply with OK."
                print("[LIVE] Filling prompt...")
                await _ws_roundtrip(ws, filler_msg, timeout_s=180.0)
            print("[LIVE] Sending query...")
            prefix = args.query_prefix
            if args.force_gate:
                prefix = f"{args.force_gate_prefix}{prefix}"
            query_msg = f"{prefix}{args.query}{args.query_suffix}"
            messages, elapsed, ttft = await _ws_roundtrip_timed(ws, query_msg, timeout_s=180.0)
            ai = next((m for m in reversed(messages) if isinstance(m, str) and m.startswith("LOG:AI:")), "")
            if ai:
                print(f"[LIVE] AI: {ai.replace('LOG:AI:', '').strip()}")
            print(f"[LIVE] Processing time: {elapsed:.2f}s")
            if ttft is not None:
                print(f"[LIVE] TTFT: {ttft:.2f}s")
            metrics = _fetch_latest_metrics(args.ws_url)
            if metrics:
                print(f"[LIVE] Metrics: input_tokens={metrics.get('input_tokens')} output_tokens={metrics.get('output_tokens')}")
                if isinstance(metrics.get("rlm_gate"), dict):
                    print(f"[LIVE] Gate: {metrics['rlm_gate']}")
                if isinstance(metrics.get("evidence"), dict):
                    print(f"[LIVE] Evidence: {metrics['evidence']}")
                if isinstance(metrics.get("rlm_forced_tool"), list):
                    print(f"[LIVE] Forced tools: {metrics['rlm_forced_tool']}")

    import asyncio as _asyncio
    _asyncio.run(_run_live())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
