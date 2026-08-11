# LLM Web Integration

## What it is

This is a modular HTTP gateway for browser-backed LLM providers. The active
provider is `gemini-web`, which uses an authenticated persistent Chromium/Chrome
profile and Gemini Web's live session. Consumers call the HTTP API; they do not
import browser or provider modules.

## Current provider

- `gemini-web` — Gemini Web through the persistent profile at
  `var/profiles/gemini-main`.
- No official Gemini API, Vertex AI, Gemini CLI, or API key is used.
- The public API has no authentication or rate limiting by design in this
  phase. Expose it only on a trusted network.

## Quick start

Requirements: Python 3.11+, Google Chrome/Chromium, and network access to
Gemini Web.

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
python app.py bootstrap
python app.py serve
```

`bootstrap` opens a browser for the one-time Google/Gemini login. The session
is persisted and reused after service restarts. Production defaults are
`0.0.0.0:4444`; override with `HOST`, `PORT`, or the other variables in
`.env.example`.

## API quick example

Preferred integration path:

```powershell
$body = @{ model = "gemini-web"; messages = @(@{ role = "user"; content = "Reply exactly: API_OK" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:4444/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

Public endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (OpenAI-compatible, non-streaming)
- `POST /api/generate` (native stateless contract)
- `GET /docs` and `GET /openapi.json`

See [API.md](API.md) for the consumer contract and [SETUP.md](SETUP.md) for
reproducible setup, bootstrap, restart, and network checks.

## Runtime notes

The provider serializes browser operations per profile. Requests are stateless
unless the caller includes its own context in the prompt/messages. The service
does not expose cookies, tokens, profile data, or browser storage through HTTP.
`usage` is `null` because the undocumented Web transport does not provide an
authoritative token count.
