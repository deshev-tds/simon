import importlib
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    data_dir = tmp_path_factory.mktemp("data")
    os.environ.pop("SIMON_TEST_MODE", None)
    os.environ.setdefault("SIMON_SKIP_AUDIO_MODELS", "1")
    os.environ.setdefault("SIMON_SKIP_VECTOR_MEMORY", "1")
    os.environ.setdefault("SIMON_DATA_DIR", str(data_dir))
    os.environ.setdefault("SIMON_RAG_DEBUG_VERBOSE", "1")
    os.environ.setdefault("SIMON_LLM_TIMEOUT_S", "10")
    os.environ.setdefault("SIMON_QUIET_LOGS", "1")
    import backend.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture(scope="session", autouse=True)
def require_lm_studio(server):
    base_url = server.LM_STUDIO_URL.rstrip("/")
    try:
        resp = httpx.get(f"{base_url}/models", timeout=2.0)
        if resp.status_code != 200:
            raise RuntimeError(f"Status {resp.status_code}")
        data = resp.json()
        models = data.get("data") or data.get("models") or []
        model_id = None
        for item in models:
            if isinstance(item, dict) and item.get("id"):
                model_id = item["id"]
                break
            if hasattr(item, "id"):
                model_id = item.id
                break
            if isinstance(item, str):
                model_id = item
                break
        if not model_id:
            raise RuntimeError("No models available")
        server.set_current_model(model_id)
        preflight = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            },
            timeout=5.0,
        )
        if preflight.status_code != 200:
            raise RuntimeError(f"Completions status {preflight.status_code}")
    except Exception as exc:
        pytest.skip(f"LM Studio not reachable at {base_url}: {exc}")


@pytest.fixture()
def app_client(server):
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def clean_state():
    yield
