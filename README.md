# Gemini Web Python Service

Modular browser-backed LLM provider service. HTTP routes resolve model aliases
through a provider registry; Gemini Web is one provider implementation backed by
an authenticated persistent Chrome context. It does not type into the UI, click
Send, or scrape the DOM.

## 1) Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

If Google Chrome is already installed, the service tries `channel=chrome` first and falls back to Playwright Chromium.

## 2) Login once

```powershell
python app.py bootstrap
```

A browser opens using the configured `GEMINI_PROFILE_DIR`. Sign in to Google/Gemini; the process
detects the authenticated `/app` session automatically and then closes the window.

## 3) Run

```powershell
python app.py serve
```

Open:

- Health: http://127.0.0.1:4444/health
- Models: http://127.0.0.1:4444/v1/models

## 4) Direct API

```powershell
$body = @{
  prompt = "Only reply: OK"
  model = "3.5 Flash-Lite"
  max_input_tokens = 4096
  max_output_tokens = 64
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:4444/api/generate" -Method POST -ContentType "application/json" -Body $body
```

Preflight the same request without inference:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:4444/api/estimate" -Method POST -ContentType "application/json" -Body $body
```

The Gemini Web session observed these canonical model names via RPC `otAQ7b`:

| Web label | Canonical name | Web model ID |
|---|---|---|
| Flash-Lite | 3.5 Flash-Lite | `8c46e95b1a07cecc` |
| Flash | 3.6 Flash | `56fdd199312815e2` |
| Pro | 3.1 Pro | `e6fa609c3fa255c0` |

The 1,048,576 input / 65,536 output limits returned in model profiles are
reference specifications for the corresponding published models. The undocumented
Gemini Web endpoint has not exposed an exact token-count or hard
`maxOutputTokens` field. Therefore the bridge marks token counts as estimates,
uses an upstream output-budget instruction, and applies a local estimated output cap.

## 5) OpenAI-compatible endpoint

```powershell
$body = @{
  model = "gemini-web"
  messages = @(
    @{ role = "user"; content = "Only reply: OPENAI_COMPAT_OK" }
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://127.0.0.1:4444/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body
```

Manual keepalive verification:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:4444/providers/gemini-web/keepalive" -Method POST
```

Gemini's provider-owned policy runs a real keepalive generation daily at 00:00
`Asia/Bangkok`. Other providers can register a different policy or disable it.

## Notes

- Keep `var/` private. It contains provider browser profiles and authenticated state.
- Do not run two Chrome/Playwright processes against the same profile directory at the same time.
- Gemini health, keepalive, and generation are serialized by the provider because one Chrome context owns the live session.
- Model names and request tokens are fetched from the current Gemini `/app` and model-list RPC.
- Latency estimates use persisted per-model runtime samples; prompt and response contents are not stored.
- If Google invalidates the session, rerun `python app.py bootstrap`.
