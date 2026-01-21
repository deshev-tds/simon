import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    data_dir = tmp_path_factory.mktemp("data")
    os.environ.pop("SIMON_TEST_MODE", None)
    os.environ.setdefault("SIMON_DATA_DIR", str(data_dir))
    import backend.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture()
def app_client(server):
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def clean_state():
    yield
