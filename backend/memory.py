import hashlib
import os
import threading
import time
import uuid
from difflib import SequenceMatcher

import chromadb
from chromadb.utils import embedding_functions

from backend.config import DATA_DIR, DEBUG_MODE, SKIP_VECTOR_MEMORY, TEST_MODE

EXPLICIT_COLLECTION_NAME = "explicit_memories"
ARCHIVE_COLLECTION_NAME = "chatgpt_archive"


class MemoryManager:
    def __init__(self):
        print("Initializing Vector Memory (ChromaDB)...")
        self.lock = threading.Lock()
        self.chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "simon_db"))
        emb_kwargs = {}
        if not _allow_remote_embeddings():
            emb_kwargs["local_files_only"] = True
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            **emb_kwargs,
        )
        self.explicit_collection = self.chroma_client.get_or_create_collection(
            name=EXPLICIT_COLLECTION_NAME,
            embedding_function=self.emb_fn
        )
        self.collection = self.explicit_collection
        self.archive_collection = self.chroma_client.get_or_create_collection(
            name=ARCHIVE_COLLECTION_NAME,
            embedding_function=self.emb_fn
        )
        print(" Memory Loaded.")

    def search(self, query_text, n_results=3, days_filter=None, session_filter=None):
        return self.search_explicit(query_text, n_results, days_filter, session_filter)

    def search_explicit(self, query_text, n_results=3, days_filter=None, session_filter=None):
        where_clause = {}

        if session_filter is not None:
            try:
                where_clause["session_id"] = int(session_filter)
            except (TypeError, ValueError):
                where_clause["session_id_raw"] = str(session_filter)
        elif days_filter is not None:
            cutoff_ts = time.time() - (days_filter * 24 * 3600)
            where_clause["timestamp"] = {"$gte": cutoff_ts}

        if not where_clause:
            where_clause = None

        with self.lock:
            results = self.explicit_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["documents", "distances", "metadatas"]
            )

        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        if session_filter is None and days_filter is not None:
            if len(docs) < n_results:
                if DEBUG_MODE:
                    print("   [MEMORY] Triggering Deep Recall (Global)...")
                with self.lock:
                    g_res = self.explicit_collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        include=["documents", "distances", "metadatas"]
                    )
                g_docs = g_res["documents"][0] if g_res["documents"] else []
                if len(g_docs) > len(docs):
                    docs, dists, metas = g_docs, g_res["distances"][0], g_res["metadatas"][0]

        return docs, dists, metas

    def search_archive(self, query_text, n_results=3):
        with self.lock:
            results = self.archive_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "distances", "metadatas"]
            )

        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return docs, dists, metas

    def save(self, user_text, ai_text, session_id):
        memory_text = f"User: {user_text} | AI: {ai_text}"
        return self.save_explicit(memory_text, session_id=session_id)

    def save_explicit(self, memory_text, session_id=None, metadata=None):
        if not memory_text:
            return
        norm_text = _normalize_text(memory_text)
        content_hash = _hash_text(norm_text)
        with self.lock:
            try:
                # Exact/normalized dedupe by hash
                exact = self.explicit_collection.get(
                    where={"content_hash": content_hash},
                    include=["ids"],
                )
                if exact and exact.get("ids"):
                    if DEBUG_MODE:
                        print(" Memory duplication detected (exact hash). Skipping save.")
                    return

                # Soft semantic dedupe only if highly lexically similar
                res = self.explicit_collection.query(
                    query_texts=[memory_text],
                    n_results=1,
                    include=["distances", "documents"]
                )
                dists = res["distances"][0] if res.get("distances") else []
                docs = res["documents"][0] if res.get("documents") else []
            except Exception:
                dists = []
                docs = []

            if dists and docs:
                candidate = _normalize_text(docs[0])
                lex_sim = _lexical_similarity(norm_text, candidate)
                if dists[0] < 0.1 and lex_sim >= 0.9:
                    if DEBUG_MODE:
                        print(
                            f" Memory duplication detected (dist={dists[0]:.4f}, lex={lex_sim:.3f}). Skipping save."
                        )
                    return

            mem_meta = {
                "role": "conversation",
                "memory_type": "explicit",
                "timestamp": time.time(),
                "session_id": _coerce_session_id(session_id),
                "session_id_raw": str(session_id) if session_id is not None else "",
                "content_hash": content_hash,
            }
            if metadata:
                mem_meta.update(metadata)

            self.explicit_collection.add(
                documents=[memory_text],
                metadatas=[mem_meta],
                ids=[str(uuid.uuid4())]
            )

    def save_archive(self, memory_text, metadata, memory_id=None):
        if not memory_text:
            return
        mem_id = memory_id or str(uuid.uuid4())
        with self.lock:
            self.archive_collection.add(
                documents=[memory_text],
                metadatas=[metadata or {}],
                ids=[mem_id]
            )


class _DummyMemory:
    def search(self, *args, **kwargs):
        return [], [], []

    def search_explicit(self, *args, **kwargs):
        return [], [], []

    def search_archive(self, *args, **kwargs):
        return [], [], []

    def save(self, *args, **kwargs):
        return None

    def save_explicit(self, *args, **kwargs):
        return None

    def save_archive(self, *args, **kwargs):
        return None


def _allow_remote_embeddings() -> bool:
    if os.environ.get("SIMON_EMBEDDINGS_LOCAL_ONLY") == "1":
        return False
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        return False
    return os.environ.get("SIMON_EMBEDDINGS_REMOTE_ALLOWED") == "1"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lexical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _coerce_session_id(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _init_memory():
    if TEST_MODE or SKIP_VECTOR_MEMORY:
        if DEBUG_MODE:
            print("Vector memory disabled. Using dummy memory.")
        return _DummyMemory()
    try:
        return MemoryManager()
    except Exception as exc:
        print(f"Vector memory unavailable ({exc}). Falling back to dummy memory.")
        return _DummyMemory()


memory = _init_memory()


__all__ = ["MemoryManager", "memory"]
