"""OpenAI-compatible endpoint that answers each chat with a Claude Agent SDK session (issue #1).

LibreChat custom endpoint -> POST /v1/chat/completions -> claude_agent_sdk.query().
The SDK's CLI talks to the LiteLLM gateway (Anthropic wire) with logical tiers, never anthropic.com.
A LibreChat conversation maps to one SDK session (resume), keyed by X-Conversation-Id.
"""
import asyncio
import hashlib
import json
import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://mports-inference-litellm-1:4000")
API_KEY = os.environ["LITELLM_VIRTUAL_KEY"]
TIERS = ["mports/balanced", "mports/fast", "mports/reasoning"]
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mports/balanced")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "12"))
WALL_SECONDS = int(os.environ.get("WALL_SECONDS", "90"))
ALLOWED_TOOLS = [t for t in os.environ.get("ALLOWED_TOOLS", "Read,Glob,Grep").split(",") if t]
WORKDIR = os.environ.get("WORKDIR", "/workspace")
SESSIONS_FILE = os.environ.get("SESSIONS_FILE", "/data/sessions.json")

if "anthropic.com" in BASE_URL:
    raise SystemExit("refusing to point the SDK at anthropic.com directly")

app = FastAPI()
_slots = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", "2")))


def _load() -> dict:
    try:
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(key: str, session_id: str) -> None:
    data = _load()
    data[key] = session_id
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def session_key(conversation_id: str | None, messages: list[dict]) -> str:
    """Stable per-chat key: the header if LibreChat sent one, else a hash of the first user message."""
    if conversation_id and conversation_id.strip() and "{{" not in conversation_id:
        return conversation_id.strip()
    first = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    return "h-" + hashlib.sha256(str(first).encode()).hexdigest()[:24]


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


async def run_sdk(prompt: str, model: str, resume: str | None):
    """Yield ('text', str) / ('session', id). Imported lazily so tests can stub the SDK."""
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                                  TextBlock, ToolUseBlock, query)

    options = ClaudeAgentOptions(
        model=model,
        cwd=WORKDIR,
        resume=resume,
        max_turns=MAX_TURNS,
        allowed_tools=ALLOWED_TOOLS,
        permission_mode="default",
        env={"ANTHROPIC_BASE_URL": BASE_URL, "ANTHROPIC_API_KEY": API_KEY,
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
    )
    async with asyncio.timeout(WALL_SECONDS):
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        yield "text", b.text
                    elif isinstance(b, ToolUseBlock):
                        yield "text", f"\n`[tool: {b.name}]`\n"
            elif isinstance(msg, ResultMessage):
                yield "session", msg.session_id


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "claude-agent-sdk", "object": "model", "owned_by": "mports"}]}


@app.get("/health")
def health():
    return {"ok": True}


def _chunk(cid: str, delta: dict, finish=None) -> str:
    body = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "claude-agent-sdk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n"


@app.post("/v1/chat/completions")
async def chat(body: dict, x_conversation_id: str | None = Header(default=None)):
    messages = body.get("messages") or []
    user = [m for m in messages if m.get("role") == "user"]
    if not user:
        raise HTTPException(400, "no user message")
    key = session_key(x_conversation_id, messages)
    resume = _load().get(key)
    prompt = _text(user[-1].get("content"))
    model = DEFAULT_MODEL
    cid = "chatcmpl-" + uuid.uuid4().hex

    async def events():
        yield _chunk(cid, {"role": "assistant"})
        try:
            async with _slots:
                async for kind, val in run_sdk(prompt, model, resume):
                    if kind == "text":
                        yield _chunk(cid, {"content": val})
                    elif val:
                        _save(key, val)
        except Exception as e:  # surfaced in chat, never swallowed silently
            yield _chunk(cid, {"content": f"\n\n[Agent SDK error: {type(e).__name__}: {e}]"})
        yield _chunk(cid, {}, "stop")
        yield "data: [DONE]\n\n"

    if body.get("stream"):
        return StreamingResponse(events(), media_type="text/event-stream")
    text = ""
    async for line in events():
        if line.startswith("data: {"):
            text += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    return JSONResponse({"id": cid, "object": "chat.completion", "created": int(time.time()),
                         "model": "claude-agent-sdk",
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}]})
