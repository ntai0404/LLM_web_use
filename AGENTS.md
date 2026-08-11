# Repository guide

## Purpose

Browser-backed LLM HTTP gateway. The active provider is Gemini Web through an
authenticated persistent native Chrome profile controlled over a CDP pipe.

## Runtime

- Default bind: `0.0.0.0:4444`
- Public OpenAI-compatible base: `/v1`
- Primary model/provider alias: `gemini-web`
- Persistent auth state: `var/profiles/gemini-main/`
- No API authentication, streaming, or fake usage counts

## Commands

```powershell
python -m pip install -r requirements.txt
python app.py bootstrap
python app.py serve
python -m pytest
```

## API boundary

External projects MUST integrate through HTTP. Prefer `POST /v1/chat/completions`.
Also supported: `GET /health`, `GET /v1/models`, `POST /api/generate`,
`GET /docs`, and `GET /openapi.json`. Do not import `app`, `provider_manager`,
`providers`, `gemini_web`, or `scheduler` into consumer projects.

## Architecture

FastAPI routes resolve model aliases through `ProviderManager` and the provider
registry. `GeminiWebProvider` owns browser/session behavior and its keepalive
policy. Scheduler code remains provider-generic.

## Files

- `app.py`: HTTP contract, lifecycle, CLI.
- `provider_manager.py`: registry and provider routing.
- `providers/`: provider contract and implementations.
- `gemini_web.py`: Gemini browser-backed transport.
- `API.md`: consumer-facing API reference.
- `tests/consumer/`: black-box HTTP-only acceptance checks.

## Security and ignored state

Never commit `.env`, `var/`, browser profiles, cookies, session storage,
auth artifacts, logs, screenshots, `prompt.md`, or `image.png`. Never log or
return cookies, tokens, or browser secrets.

## Runtime acceptance

After code changes, restart the service and verify wildcard listener `0.0.0.0:4444`,
health, model discovery, native generation, OpenAI-compatible generation, and
Gemini persistence without bootstrap. Run `python -m pytest tests/consumer -q`
against the running service for the external integration path.

## Current limitations

The service is non-streaming, serial per browser profile, unauthenticated by
design, and dependent on Gemini Web's undocumented browser transport.
