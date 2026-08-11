"""Opt-in live Browser Use/OpenAI compatibility checks.

This module is a black-box consumer: it imports no service implementation.
Set RUN_LIVE_GEMINI=1 and BASE_URL before running it.
"""

import asyncio
import base64
import glob
import json
import os
import struct
import tempfile
import urllib.request
import uuid
import zlib

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI") != "1",
    reason="live Gemini quota test is opt-in",
)
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:4444").rstrip("/")


class BrowserAction(BaseModel):
    click_element: dict[str, int]


class BrowserAgentOutput(BaseModel):
    thinking: str
    evaluation_previous_goal: str
    memory: str
    next_goal: str
    action: list[BrowserAction]


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="placeholder",
        timeout=240,
        max_retries=0,
    )


def _http_json(method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        assert response.status == 200
        return json.loads(response.read().decode())


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _solid_red_png(width: int = 96, height: int = 96) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )


def test_g_health_models_and_native_regression():
    health = _http_json("GET", "/health")
    assert health["providers"]["gemini-web"]["status"] == "READY"
    assert health["providers"]["gemini-web"]["auth"] == "ok"
    models = _http_json("GET", "/v1/models")
    assert any(model["id"] == "gemini-web" for model in models["data"])
    nonce = f"LIVE_NATIVE_{uuid.uuid4().hex[:10].upper()}"
    result = _http_json(
        "POST",
        "/api/generate",
        {"model": "gemini-web", "prompt": f"Reply exactly: {nonce}"},
    )
    assert result["text"].strip() == nonce


def test_a_async_openai_default_browser_use_parameters():
    async def run():
        nonce = f"LIVE_OPENAI_{uuid.uuid4().hex[:10].upper()}"
        async with _client() as client:
            response = await client.chat.completions.create(
                model="gemini-web",
                messages=[{"role": "user", "content": f"Reply exactly: {nonce}"}],
                temperature=0,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                max_completion_tokens=128,
                seed=42,
                stream=False,
            )
        assert response.choices[0].message.content.strip() == nonce

    asyncio.run(run())


def test_b_structured_browser_agent_parses_directly():
    async def run():
        schema = BrowserAgentOutput.model_json_schema()
        async with _client() as client:
            response = await client.chat.completions.create(
                model="gemini-web",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "DOM snapshot: <button data-index='17'>Continue</button>. "
                            "Choose the Continue button and produce the requested agent state."
                        ),
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "browser_agent_output",
                        "strict": True,
                        "schema": schema,
                    },
                },
                max_completion_tokens=512,
            )
        parsed = BrowserAgentOutput.model_validate_json(
            response.choices[0].message.content
        )
        assert parsed.action[0].click_element["index"] == 17

    asyncio.run(run())


def test_c_vision_data_url_and_temporary_cleanup():
    async def run():
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "llm_web_openai_*")))
        data_url = "data:image/png;base64," + base64.b64encode(_solid_red_png()).decode()
        async with _client() as client:
            response = await client.chat.completions.create(
                model="gemini-web",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify the dominant image color. Reply exactly: RED",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_completion_tokens=64,
            )
        assert response.choices[0].message.content.strip().upper() == "RED"
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "llm_web_openai_*")))
        assert after == before

    asyncio.run(run())


def test_d_required_tool_call_is_openai_compatible():
    async def run():
        async with _client() as client:
            response = await client.chat.completions.create(
                model="gemini-web",
                messages=[
                    {
                        "role": "user",
                        "content": "Click the Continue button at DOM index 17.",
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "click_element",
                            "description": "Click a DOM element by index",
                            "parameters": {
                                "type": "object",
                                "properties": {"index": {"type": "integer"}},
                                "required": ["index"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                tool_choice="required",
                parallel_tool_calls=False,
                max_completion_tokens=256,
            )
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls[0].function.name == "click_element"
        assert json.loads(choice.message.tool_calls[0].function.arguments)["index"] == 17

    asyncio.run(run())


def test_e_multiturn_preserves_prior_information():
    async def run():
        nonce = f"MEMORY_{uuid.uuid4().hex[:10].upper()}"
        async with _client() as client:
            response = await client.chat.completions.create(
                model="gemini-web",
                messages=[
                    {"role": "system", "content": "Follow exact reply instructions."},
                    {"role": "user", "content": f"Remember this value: {nonce}"},
                    {"role": "assistant", "content": "I will remember it."},
                    {
                        "role": "user",
                        "content": "Reply with the remembered value only.",
                    },
                ],
                temperature=0,
                max_completion_tokens=128,
            )
        assert response.choices[0].message.content.strip() == nonce

    asyncio.run(run())
