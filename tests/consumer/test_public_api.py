"""Black-box public API checks.

This test intentionally imports only Python's HTTP standard library. It must
not depend on service implementation modules.
"""

import json
import os
import urllib.request
import uuid


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:4444").rstrip("/")


def request_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_health_over_http_only():
    status, payload = request_json("GET", "/health")
    assert status == 200
    assert payload["service"] in {"ok", "degraded"}
    assert "gemini-web" in payload["providers"]


def test_models_discovery_over_http_only():
    status, payload = request_json("GET", "/v1/models")
    assert status == 200
    assert payload["object"] == "list"
    assert any(item["id"] == "gemini-web" for item in payload["data"])


def test_native_generate_over_http_only():
    nonce = f"CONSUMER_NATIVE_{uuid.uuid4().hex[:10].upper()}"
    status, payload = request_json(
        "POST",
        "/api/generate",
        {"model": "gemini-web", "prompt": f"Reply exactly: {nonce}"},
    )
    assert status == 200
    assert payload["provider"] == "gemini-web"
    assert payload["text"].strip() == nonce


def test_openai_chat_over_http_only():
    nonce = f"CONSUMER_OPENAI_{uuid.uuid4().hex[:10].upper()}"
    status, payload = request_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": "gemini-web",
            "messages": [{"role": "user", "content": f"Reply exactly: {nonce}"}],
        },
    )
    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"].strip() == nonce
