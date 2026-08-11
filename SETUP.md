# Setup and operations

## Requirements

- Windows 10/11 or a comparable Python 3.11+ environment.
- Google Chrome installed. Playwright tries the installed Chrome channel first
  and can fall back to its managed Chromium browser.
- An authenticated Gemini Web account and Internet access.

This repository does not require a Python virtual environment. Install with
the interpreter that will run the service:

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Configuration

Copy `.env.example` to `.env` when overrides are needed. The important defaults
are:

```text
HOST=0.0.0.0
PORT=4444
GEMINI_PROFILE_DIR=var/profiles/gemini-main
GEMINI_TIMEOUT_MS=120000
HEADLESS=true
BROWSER_CHANNEL=chrome
```

`0.0.0.0:4444` makes the HTTP service reachable from another process, LAN
host, container network, or VM when the operating-system firewall permits TCP
4444. This task does not change firewall rules. The service intentionally has
no API authentication, API key, CORS auth, or rate limiting.

## Bootstrap Gemini login

Run once from the repository root:

```powershell
python app.py bootstrap
```

Complete Google/Gemini login in the opened browser. The authenticated state is
stored in `var/profiles/gemini-main`. Do not commit, copy, or run two browser
processes against that profile at the same time. Bootstrap does not automate
passwords or 2FA.

## Start and verify

```powershell
python app.py serve
```

The final listener should be wildcard-bound:

```text
0.0.0.0:4444
```

Verify locally:

```powershell
Invoke-RestMethod http://127.0.0.1:4444/health
Invoke-RestMethod http://127.0.0.1:4444/v1/models
Invoke-WebRequest http://127.0.0.1:4444/docs -UseBasicParsing
```

From another machine, use the server's LAN address:

```text
http://<server-ip>:4444/health
http://<server-ip>:4444/v1/chat/completions
```

## Restart persistence

Stop the service cleanly, start `python app.py serve` again, and call
`GET /health`. A valid profile should return `gemini-web` with `auth: "ok"`
without running bootstrap again. If Google invalidates the session, the health
state becomes `AUTH_REQUIRED`; bootstrap is then a manual operator action.

## Test commands

Run the unit/contract suite with the normal Python command:

```powershell
python -m pytest
```

The black-box tests in `tests/consumer/` use only HTTP and standard-library
client code. They expect a running service at `BASE_URL` (default
`http://127.0.0.1:4444`):

```powershell
$env:BASE_URL = "http://127.0.0.1:4444"
python -m pytest tests/consumer -q
```

## Troubleshooting

- `port already in use`: inspect the listener with `netstat -ano | findstr 4444`.
- `AUTH_REQUIRED`: run `python app.py bootstrap` and complete login manually.
- `PROVIDER_UNAVAILABLE`: check Chrome/Chromium availability and profile locks.
- `GENERATION_TIMEOUT`: the browser-backed generation exceeded
  `GEMINI_TIMEOUT_MS`; use bounded retry with backoff, never infinite retry.
- `UPSTREAM_ERROR`: Gemini Web returned an error; inspect the non-secret server
  status and retry only when appropriate.
- `http://<server-ip>:4444` unreachable: check the OS firewall/network route;
  this project does not alter firewall settings.

## Error retry guidance

Do not retry validation errors or `AUTH_REQUIRED` automatically. A caller may
retry `PROVIDER_UNAVAILABLE`, `GENERATION_TIMEOUT`, or `UPSTREAM_ERROR` with a
bounded backoff and its own request-id correlation.
