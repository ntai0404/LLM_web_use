# Public API reference

## Base URL

Local callers use `http://127.0.0.1:4444`. Network callers use
`http://<server-ip>:4444`. The OpenAI-compatible base URL is
`http://<server-ip>:4444/v1`.

The supported integration boundary is HTTP JSON. Consumers must not import
`providers`, `provider_manager`, `gemini_web`, or browser/runtime modules.

## API stability

The following endpoints are the public LLM contract:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process/provider readiness |
| GET | `/v1/models` | OpenAI-style model discovery |
| POST | `/v1/chat/completions` | Preferred OpenAI-compatible generation |
| POST | `/api/generate` | Native stateless generation |
| GET | `/docs` | Interactive OpenAPI documentation |
| GET | `/openapi.json` | Generated OpenAPI schema |

`/api/providers/*` and `/api/estimate` are management/support routes. The
dashboard is served at `/`.

## Models

```http
GET /v1/models
```

The response is an OpenAI-compatible list. The registry always exposes the
provider alias `gemini-web`; live Gemini model descriptors may contain extra
metadata.

```json
{
  "object": "list",
  "data": [
    {"id": "gemini-web", "object": "model", "owned_by": "gemini-web"}
  ]
}
```

## OpenAI-compatible chat

```http
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "gemini-web",
  "messages": [
    {"role": "user", "content": "Reply exactly: HELLO_API"}
  ]
}
```

Supported roles are `system`, `user`, and `assistant`. Requests are
non-streaming; `stream: true` returns `400 UNSUPPORTED_OPTION`. The response
shape is compatible with common OpenAI clients:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "gemini-web",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "HELLO_API"},
    "finish_reason": "stop"
  }],
  "usage": null
}
```

No token count, pricing, or usage value is fabricated.

Python consumer example (the consumer owns its HTTP dependency):

```python
import requests

response = requests.post(
    "http://127.0.0.1:4444/v1/chat/completions",
    json={
        "model": "gemini-web",
        "messages": [{"role": "user", "content": "Hello"}],
    },
    timeout=180,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

OpenAI SDK clients can use `base_url="http://127.0.0.1:4444/v1"` and any
placeholder `api_key` required by the client. The server does not inspect or
require that key. Only the non-streaming request shape above is verified.

Node.js example:

```javascript
const response = await fetch("http://127.0.0.1:4444/v1/chat/completions", {
  method: "POST",
  headers: {"content-type": "application/json"},
  body: JSON.stringify({
    model: "gemini-web",
    messages: [{role: "user", content: "Hello"}]
  })
});
console.log((await response.json()).choices[0].message.content);
```

## Native generate

```http
POST /api/generate
Content-Type: application/json
```

```json
{"model": "gemini-web", "prompt": "Reply exactly: NATIVE_OK"}
```

The response contains `model`, `provider`, `text`, and non-secret `metadata`.
`prompt_file` and project-local image paths are supported for local workflows;
external consumers should normally send `prompt` directly.

## Health

```http
GET /health
```

The process returns HTTP 200 with a body such as:

```json
{
  "service": "ok",
  "providers": {
    "gemini-web": {
      "status": "READY",
      "auth": "ok",
      "browser_headless": true
    }
  }
}
```

`service` becomes `degraded` when a provider is not ready; provider state is
the readiness signal. No cookie, token, or session secret is returned.

## Error contract

Errors are always JSON with a request correlation ID:

```json
{
  "error": "AUTH_REQUIRED",
  "message": "Gemini Web login is required",
  "request_id": "abc-123"
}
```

Important implementation codes:

| HTTP | Error codes |
|---:|---|
| 400 | `INVALID_INPUT`, `UNSUPPORTED_OPTION`, `INVALID_CONTENT_LENGTH` |
| 401 | `AUTH_REQUIRED` |
| 404 | `NOT_FOUND`, `PROVIDER_NOT_FOUND` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 422 | `VALIDATION_ERROR` |
| 502 | `UPSTREAM_ERROR`, `KEEPALIVE_FAILED`, `PROVIDER_TEST_FAILED` |
| 503 | `PROVIDER_UNAVAILABLE` |
| 504 | `GENERATION_TIMEOUT` |
| 500 | `INTERNAL_ERROR` |

## Limits and timeouts

These values are enforced by the current implementation:

- Maximum HTTP request body: 20 MiB (`413 PAYLOAD_TOO_LARGE`).
- Prompt/message content: 4,194,304 characters per content field and in total
  for chat messages.
- Chat messages: 1–128; native images: at most 10.
- Default Gemini operation timeout: 120 seconds (`GEMINI_TIMEOUT_MS`).
- `max_output_tokens`: 1–65,536; `max_input_tokens`: 1–1,048,576.

Do not blindly retry. Validation and `AUTH_REQUIRED` should not be retried
automatically. Provider unavailable, timeout, and upstream errors may be
retried with bounded backoff if the caller can tolerate duplicate work.

## Request IDs

Send `X-Request-ID: <safe-id>` to correlate a request. If omitted or invalid,
the service creates one. The same value is returned in the response header and
the JSON error body. IDs contain only letters, digits, `.`, `_`, and `-`.

## Known limitations

- Browser-backed generation is serial per Gemini profile.
- The Web transport is undocumented and may change upstream.
- Streaming/SSE and authoritative token usage are not implemented.
- API authentication and rate limiting are intentionally not implemented in
  this phase; the service trusts the network it is exposed on.
