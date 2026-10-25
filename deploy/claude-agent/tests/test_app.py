import os, sys
os.environ["LITELLM_VIRTUAL_KEY"] = "test"
os.environ["SESSIONS_FILE"] = "/tmp/sessions-test.json"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app as svc  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

calls = []

async def fake(prompt, model, resume):
    calls.append((prompt, model, resume))
    yield "text", "hello"
    yield "session", "sess-1"

svc.run_sdk = fake
c = TestClient(svc.app)

def test_session_resume():
    if os.path.exists(svc.SESSIONS_FILE):
        os.remove(svc.SESSIONS_FILE)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/v1/chat/completions", json=body, headers={"X-Conversation-Id": "c1"})
    assert r.json()["choices"][0]["message"]["content"] == "hello"
    c.post("/v1/chat/completions", json=body, headers={"X-Conversation-Id": "c1"})
    assert calls[0][2] is None and calls[1][2] == "sess-1"

def test_key_fallback_ignores_unrendered_placeholder():
    k = svc.session_key("{{LIBRECHAT_BODY_CONVERSATIONID}}", [{"role": "user", "content": "x"}])
    assert k.startswith("h-")

def test_models():
    assert c.get("/v1/models").json()["data"][0]["id"] == "claude-agent-sdk"
