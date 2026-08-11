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

Supported roles are `system`, `user`, `assistant`, and `tool`. Message content
may be a string, `null`, or an ordered list of `text` and `image_url` parts.
Assistant `tool_calls`, message `name`, and `tool_call_id` are preserved.
Requests are non-streaming; `stream: true` returns `422 VALIDATION_ERROR`.

The adapter accepts common OpenAI/Browser Use options including `temperature`,
`top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `max_tokens`,
`max_completion_tokens`, `response_format`, `tools`, `tool_choice`, and
`parallel_tool_calls`. `max_completion_tokens` takes precedence over
`max_tokens` and maps to the provider output budget. Sampling values that the
Gemini Web transport cannot apply are accepted as controlled no-ops.

The response shape is compatible with common OpenAI clients:

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

OpenAI SDK and Browser Use clients can use
`base_url="http://127.0.0.1:4444/v1"`, `model="gemini-web"`, and any
placeholder `api_key` required by the client. The server does not inspect or
require that key. Only non-streaming chat completions are supported.

## Vision content parts

`image_url.url` accepts only inline `data:image/png|jpeg|webp|gif;base64,...`
values. Each decoded image is limited to 8 MiB and a request may contain at
most 10 images. MIME and file signatures are validated. Images are decoded to
request-scoped temporary files and removed after success, timeout, or error.
Remote HTTP/HTTPS image URLs are rejected and never downloaded.

## Structured output

Both `{"type":"json_object"}` and OpenAI-style `json_schema` response formats
are supported. The schema is added to the provider prompt, Markdown/thinking
wrappers are removed, JSON is parsed and validated server-side, and at most one
repair generation is attempted. A successful response always exposes valid
JSON in `message.content`; repeated malformed output returns
`502 MALFORMED_UPSTREAM_OUTPUT`.

## Function tools

OpenAI function tools support `tool_choice` values `auto`, `required`, `none`,
and a forced function object. The service asks the model to select from the
provided allowlist, validates the function name and JSON arguments against the
function schema, and returns OpenAI `message.tool_calls` with
`finish_reason="tool_calls"`. The service never executes tools itself.

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
| 400 | `INVALID_CONTENT_LENGTH` |
| 401 | `AUTH_REQUIRED` |
| 429 | `RATE_LIMITED` |
| 404 | `NOT_FOUND`, `PROVIDER_NOT_FOUND` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 422 | `VALIDATION_ERROR`, `INVALID_REQUEST` |
| 502 | `UPSTREAM_ERROR`, `MALFORMED_UPSTREAM_OUTPUT`, `KEEPALIVE_FAILED`, `PROVIDER_TEST_FAILED` |
| 503 | `PROVIDER_UNAVAILABLE`, `PROVIDER_BUSY` |
| 504 | `GENERATION_TIMEOUT` |
| 500 | `INTERNAL_ERROR` |

## Limits and timeouts

These values are enforced by the current implementation:

- Maximum HTTP request body: 20 MiB (`413 PAYLOAD_TOO_LARGE`).
- Prompt/message content: 4,194,304 characters per content field and in total
  for chat messages.
- Chat messages: 1–128; native images: at most 10.
- Default Gemini operation timeout: 120 seconds (`GEMINI_TIMEOUT_MS`).
- Default provider queue wait: 30 seconds (`GEMINI_QUEUE_TIMEOUT_MS`).
- `max_output_tokens`: 1–65,536; `max_input_tokens`: 1–1,048,576.

Do not blindly retry. Validation and `AUTH_REQUIRED` should not be retried
automatically. Provider unavailable, timeout, and upstream errors may be
retried with bounded backoff if the caller can tolerate duplicate work.

## Request IDs

Send `X-Request-ID: <safe-id>` to correlate a request. If omitted or invalid,
the service creates one. The same value is returned in the response header and
the JSON error body. IDs contain only letters, digits, `.`, `_`, and `-`.

## Known limitations

- Browser-backed generation is serial per Gemini profile with a bounded queue wait.
- The Web transport is undocumented and may change upstream.
- Streaming/SSE and authoritative token usage are not implemented.
- API authentication and rate limiting are intentionally not implemented in
  this phase; the service trusts the network it is exposed on.
