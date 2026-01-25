import os
import socket
import threading
import time
import uuid

import chromadb
from chromadb.utils import embedding_functions

from backend.config import DATA_DIR, DEBUG_MODE, SKIP_VECTOR_MEMORY, TEST_MODE


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
        self.collection = self.chroma_client.get_or_create_collection(
            name="simon_memories",
            embedding_function=self.emb_fn
        )
        print(" Memory Loaded.")

    def search(self, query_text, n_results=3, days_filter=None, session_filter=None):
        where_clause = {}

        if session_filter is not None:
            where_clause["session_id"] = int(session_filter)
        elif days_filter is not None:
            cutoff_ts = time.time() - (days_filter * 24 * 3600)
            where_clause["timestamp"] = {"$gte": cutoff_ts}

        if not where_clause:
            where_clause = None

        with self.lock:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["documents", "distances", "metadatas"]
            )

        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        if session_filter is None and days_filter is not None:
            need_deep = False
            if len(docs) < n_results:
                need_deep = True
            elif dists and dists[0] > 0.45:
                need_deep = True

            if need_deep:
                if DEBUG_MODE:
                    print("   [MEMORY] Triggering Deep Recall (Global)...")
                with self.lock:
                    g_res = self.collection.query(
                        query_texts=[query_text],
                        n_results=n_results,
                        include=["documents", "distances", "metadatas"]
                    )
                g_docs = g_res["documents"][0] if g_res["documents"] else []
                if len(g_docs) > len(docs):
                    docs, dists, metas = g_docs, g_res["distances"][0], g_res["metadatas"][0]

        return docs, dists, metas

    def save(self, user_text, ai_text, session_id):
        with self.lock:
            try:
                res = self.collection.query(
                    query_texts=[user_text],
                    n_results=1,
                    include=["distances"]
                )
                dists = res["distances"][0] if res.get("distances") else []
            except Exception:
                dists = []

            if dists and dists[0] < 0.2:
                if DEBUG_MODE:
                    print(f" Memory duplication detected (Dist: {dists[0]:.4f}). Skipping save.")
                return

            memory_text = f"User: {user_text} | AI: {ai_text}"
            metadata = {
                "role": "conversation",
                "timestamp": time.time(),
                "session_id": int(session_id) if session_id else 0
            }

            self.collection.add(
                documents=[memory_text],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )


class _DummyMemory:
    def search(self, *args, **kwargs):
        return [], [], []

    def save(self, *args, **kwargs):
        return None


def _allow_remote_embeddings() -> bool:
    if os.environ.get("SIMON_EMBEDDINGS_LOCAL_ONLY") == "1":
        return False
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        return False
    try:
        sock = socket.create_connection(("huggingface.co", 443), timeout=1.0)
        sock.close()
        return True
    except OSError:
        return False


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
