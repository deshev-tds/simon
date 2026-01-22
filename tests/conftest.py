import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeStream:
    def __init__(self, text, include_empty=True):
        self.text = text
        self.include_empty = include_empty

    def __iter__(self):
        if self.include_empty:
            yield SimpleNamespace(choices=[])
        for token in self.text.split(" "):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token + " "))])


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call-1"):
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class FakeResponse:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]
        self.usage = None


class FakeChatCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model=None, messages=None, stream=False, tools=None, tool_choice=None, **kwargs):
        if stream:
            text = "OK."
            if tools:
                text = "Deep answer OK."
            return FakeStream(text)
        if tools:
            has_tool_result = any(
                isinstance(m, dict) and m.get("role") == "tool"
                for m in (messages or [])
            )
            if not has_tool_result:
                self.parent.tool_calls_made += 1
                tool_call = FakeToolCall(
                    "search_memory",
                    '{"query":"TOKEN0001","scope":"session"}'
                )
                msg = FakeMessage(role="assistant", content=None, tool_calls=[tool_call])
                return FakeResponse(msg)
        msg = FakeMessage(role="assistant", content="hello")
        return FakeResponse(msg)


class FakeModels:
    def list(self):
        return SimpleNamespace(data=[SimpleNamespace(id="fake-model")])


class FakeClient:
    def __init__(self):
        self.tool_calls_made = 0
        self.chat = SimpleNamespace(completions=FakeChatCompletions(self))
        self.models = FakeModels()


class FakeSTT:
    def transcribe(self, *args, **kwargs):
        seg = SimpleNamespace(text="hello from stt")
        return [seg], None


class FakeKokoro:
    def create(self, text, voice=None, speed=1.0, lang=None):
        return np.zeros(160, dtype=np.float32), 16000


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    data_dir = tmp_path_factory.mktemp("data")
    models_dir = tmp_path_factory.mktemp("models")
    os.environ["SIMON_TEST_MODE"] = "1"
    os.environ["SIMON_DATA_DIR"] = str(data_dir)
    os.environ["SIMON_MODELS_DIR"] = str(models_dir)
    os.environ["SIMON_MEM_SEED_LIMIT"] = "0"
    os.environ["SIMON_MEM_MAX_ROWS"] = "100000"
    os.environ["SIMON_MEM_PRUNE_INTERVAL_S"] = "3600"
    import backend.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture()
def app_client(server, monkeypatch):
    monkeypatch.setattr(server, "client", FakeClient())
    monkeypatch.setattr(server, "stt_model", FakeSTT())
    monkeypatch.setattr(server, "kokoro", FakeKokoro())
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def clean_state(server):
    try:
        with server.db_lock:
            server.db_conn.execute("DELETE FROM messages")
            server.db_conn.execute("DELETE FROM sessions")
            server.db_conn.commit()
            server.mem_conn.execute("DELETE FROM messages")
            server.mem_conn.execute("DELETE FROM sessions")
            server.mem_conn.commit()
    except Exception:
        with server.db_lock:
            try:
                server.db_conn.close()
            except Exception:
                pass
            server.db_conn = server.init_db()
            try:
                server._ensure_fts5(server.db_conn)
            except Exception:
                pass
            try:
                server.mem_conn.close()
            except Exception:
                pass
            server.mem_conn = server.init_mem_db()
    server.METRICS_HISTORY.clear()
    yield


def wait_for_condition(fn, timeout_s=3.0, sleep_s=0.05):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(sleep_s)
    return False


def pytest_configure(config):
    config._fts_recursive_stats = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    stats = getattr(config, "_fts_recursive_stats", [])
    if not stats:
        return
    terminalreporter.write_line("[FTS-RECURSION] summary begin")
    for item in stats:
        terminalreporter.write_line(
            "[FTS-RECURSION] case="
            f"{item.get('label')} query={item.get('query')} "
            f"depth={item.get('depth')} queries_used={item.get('queries_used')} "
            f"returned={item.get('returned')}"
        )
        terminalreporter.write_line(
            "[FTS-RECURSION] tokens "
            f"root={item.get('root_tokens')} strong={item.get('strong_tokens')}"
        )
        terminalreporter.write_line(
            "[FTS-RECURSION] counts "
            f"raw={item.get('raw_count')} deduped={item.get('deduped_count')} "
            f"filtered={item.get('filtered_count')} subqueries={item.get('subqueries_count')} "
            f"query_runs={item.get('query_runs')}"
        )
        for hit in item.get("hits", []):
            terminalreporter.write_line(
                "[FTS-RECURSION] hit "
                f"score={hit.get('score')} content={hit.get('content')}"
            )
    terminalreporter.write_line("[FTS-RECURSION] summary end")
