#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("SIMON_SKIP_AUDIO_MODELS", "1")

import backend.server as server  # noqa: E402


INJECTION_PREFIXES = (
    "Relevant past memories:",
    "Relevant archive memories",
    "Keyword recall",
)


def _select_session(session_id: int | None) -> int:
    if session_id is None:
        new_id = server.create_session(None)
        print(f"[INFO] Created session: {new_id}")
        return new_id
    if not server.session_exists(session_id):
        print(f"[WARN] Session {session_id} not found; creating new.")
        new_id = server.create_session(None)
        print(f"[INFO] Created session: {new_id}")
        return new_id
    return session_id


def _print_injections(context_msgs):
    injections = []
    for msg in context_msgs:
        if msg.get("role") != "system":
            continue
        content = msg.get("content") or ""
        if any(content.startswith(prefix) for prefix in INJECTION_PREFIXES):
            injections.append(content)
    if not injections:
        print("[INJECT] none")
        return
    for i, content in enumerate(injections, start=1):
        print(f"[INJECT {i}]")
        print(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive RAG probe using Simon's exact retrieval logic.")
    parser.add_argument("--session-id", type=int, default=None, help="Existing session id to use")
    args = parser.parse_args()

    session_id = _select_session(args.session_id)
    print("[INFO] Enter text to run match -> search -> inject. Use /session <id> to switch, /quit to exit.")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break
        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in {"/quit", "/exit", "quit", "exit"}:
            break
        if lowered.startswith("/session"):
            parts = user_text.split()
            if len(parts) == 2 and parts[1].isdigit():
                session_id = _select_session(int(parts[1]))
                print(f"[INFO] Using session: {session_id}")
            else:
                print("[INFO] Usage: /session <id>")
            continue

        history = server.load_session_messages(session_id)
        metrics = {}
        context_msgs, rag_payload = server.build_rag_context(
            user_text,
            history,
            server.memory,
            metrics,
            session_id,
        )

        print("[MATCH] done")
        print("[SEARCH] rag_payload:")
        print(json.dumps(rag_payload, indent=2))
        _print_injections(context_msgs)
        print(f"[CTX] messages={len(context_msgs)} session={session_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
