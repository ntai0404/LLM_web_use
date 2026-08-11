import argparse
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gemini_web import AuthRequired, GeminiWebClient, GeminiWebError


load_dotenv()

PROFILE_DIR = os.getenv("GEMINI_PROFILE_DIR", "var/gemini-profile")
HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "120000"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8787"))

client = GeminiWebClient(PROFILE_DIR, HEADLESS, TIMEOUT_MS)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = "gemini-web"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-web"
    messages: list[ChatMessage]
    stream: bool = False


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.role.strip().lower()
        label = {"system": "System", "assistant": "Assistant", "user": "User"}.get(role, m.role)
        parts.append(f"{label}: {m.content}")
    return "\n\n".join(parts)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Lazy browser startup: service can boot even before login/bootstrap.
    yield
    await client.stop()


app = FastAPI(title="Gemini Web Python Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return await client.health()


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        text = await client.ask(req.prompt, req.model)
        return {"model": req.model or "gemini-web", "text": text}
    except AuthRequired as exc:
        raise HTTPException(status_code=401, detail={"error": "AUTH_REQUIRED", "message": str(exc)})
    except (GeminiWebError, Exception) as exc:
        raise HTTPException(status_code=502, detail={"error": "GEMINI_WEB_ERROR", "message": str(exc)})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented in this starter")

    prompt = messages_to_prompt(req.messages)
    try:
        text = await client.ask(prompt, req.model)
    except AuthRequired as exc:
        raise HTTPException(status_code=401, detail={"error": "AUTH_REQUIRED", "message": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": "GEMINI_WEB_ERROR", "message": str(exc)})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Gemini Web Service</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;background:#f6f7f9;color:#111}
.card{background:#fff;border:1px solid #ddd;border-radius:14px;padding:18px;box-shadow:0 4px 18px #0001}
textarea{width:100%;min-height:150px;box-sizing:border-box;padding:12px;font:inherit;border:1px solid #bbb;border-radius:10px}
input{padding:10px;border:1px solid #bbb;border-radius:9px;width:220px}
button{padding:10px 16px;border:0;border-radius:9px;cursor:pointer;font-weight:600}
.row{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}
pre{white-space:pre-wrap;background:#111;color:#eee;padding:14px;border-radius:10px;min-height:120px}
small{color:#666}
</style>
</head>
<body>
<div class="card">
  <h2>Gemini Web Python Service</h2>
  <small>Uses the persistent browser profile in <code>var/gemini-profile</code>.</small>
  <div class="row">
    <input id="model" value="gemini-web" placeholder="model label" />
    <button onclick="checkHealth()">Check health</button>
  </div>
  <textarea id="prompt" placeholder="Type a prompt..."></textarea>
  <div class="row"><button onclick="sendPrompt()">Send</button></div>
  <pre id="out">Ready.</pre>
</div>
<script>
const out = document.getElementById('out');
async function checkHealth(){
  out.textContent='Checking...';
  try{const r=await fetch('/health'); out.textContent=JSON.stringify(await r.json(),null,2)}
  catch(e){out.textContent=String(e)}
}
async function sendPrompt(){
  const prompt=document.getElementById('prompt').value;
  const model=document.getElementById('model').value;
  out.textContent='Calling Gemini Web...';
  try{
    const r=await fetch('/api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt,model})});
    const data=await r.json();
    out.textContent=r.ok ? data.text : JSON.stringify(data,null,2);
  }catch(e){out.textContent=String(e)}
}
</script>
</body>
</html>
"""


async def do_bootstrap():
    await client.bootstrap_login()


def main():
    parser = argparse.ArgumentParser(description="Gemini Web Python Service")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap", help="Open persistent Chrome profile and log in once")
    sub.add_parser("serve", help="Run FastAPI service")
    args = parser.parse_args()

    if args.cmd == "bootstrap":
        asyncio.run(do_bootstrap())
    else:
        uvicorn.run("app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
