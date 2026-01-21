import pytest


@pytest.mark.integration
def test_chat_completion_real_model(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Reply with a short greeting."}],
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert content.strip()
