import io

import numpy as np
import soundfile as sf


def test_root_and_static_endpoints(app_client):
    resp = app_client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()

    manifest = app_client.get("/manifest.json")
    assert manifest.status_code in (200, 404)

    admin = app_client.get("/admin")
    assert admin.status_code in (200, 404)


def test_models_and_model_switch(app_client, server):
    resp = app_client.get("/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert "models" in payload
    assert "current" in payload
    if getattr(server, "_using_fake_llm", False):
        resp = app_client.post("/model", json={"name": "fake-model"})
        assert resp.status_code == 200
        assert resp.json().get("current") == "fake-model"
        assert server.get_current_model() == "fake-model"
    else:
        resp = app_client.post("/model", json={"name": ""})
        assert resp.status_code == 200
        assert resp.json().get("current") == ""
        assert server.get_current_model() == ""


def test_sessions_crud(app_client):
    resp = app_client.post("/sessions", json={})
    assert resp.status_code == 200
    session_id = resp.json()["id"]

    resp = app_client.get("/sessions")
    assert resp.status_code == 200
    assert any(s["id"] == session_id for s in resp.json()["sessions"])

    resp = app_client.get(f"/sessions/{session_id}/window")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session"]["id"] == session_id


def test_chat_completions(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]}
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "choices" in payload
    assert payload["choices"][0]["message"]["content"] is not None


def test_audio_speech(app_client, server, monkeypatch):
    def _fake_mp3(*args, **kwargs):
        return io.BytesIO(b"MP3")

    monkeypatch.setattr(server, "_convert_to_mp3_bytes", _fake_mp3)
    resp = app_client.post("/v1/audio/speech", json={"input": "test"})
    assert resp.status_code == 200
    assert "audio/mpeg" in resp.headers.get("content-type", "")


def test_audio_transcriptions(app_client):
    wav_io = io.BytesIO()
    sf.write(wav_io, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    wav_io.seek(0)
    resp = app_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_io.read(), "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json().get("text") == "hello from stt"


def test_metrics_endpoint(app_client):
    resp = app_client.get("/metrics")
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
    assert "count" in payload
