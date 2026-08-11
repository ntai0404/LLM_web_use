# Gemini Web Python Service

Python-only starter that calls Gemini through the signed-in Gemini web UI using a persistent browser profile.

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

A browser opens using `var/gemini-profile`. Sign in to Google/Gemini, make sure the Gemini prompt box is usable, then press ENTER in the terminal. The browser closes and keeps the session/profile on disk.

## 3) Run

```powershell
python app.py serve
```

Open:

- UI: http://127.0.0.1:8787/
- Health: http://127.0.0.1:8787/health

## 4) Direct API

```powershell
$body = @{ prompt = "Only reply: OK"; model = "gemini-web" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8787/api/generate" -Method POST -ContentType "application/json" -Body $body
```

## 5) OpenAI-compatible endpoint

```powershell
$body = @{
  model = "gemini-web"
  messages = @(
    @{ role = "user"; content = "Only reply: OPENAI_COMPAT_OK" }
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://127.0.0.1:8787/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body
```

## Notes

- Keep `var/gemini-profile` private. It contains authenticated browser state.
- Do not run two Chrome/Playwright processes against the same profile directory at the same time.
- Gemini's web DOM can change. Prompt/response selectors are isolated in `gemini_web.py` so they can be adjusted in one place.
- Requests are serialized with an asyncio lock to avoid two callers typing into the same page simultaneously.
- Each request navigates to a fresh Gemini app URL to reduce accidental cross-request conversation context.
- If Google invalidates the session, rerun `python app.py bootstrap`.
