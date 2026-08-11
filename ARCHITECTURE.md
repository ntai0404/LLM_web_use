# Architecture

## Public gateway boundary

```text
Backend / service / CLI / desktop app
                |
           HTTP JSON
                v
          0.0.0.0:4444
                |
             FastAPI
                |
        ProviderManager/Registry
                |
        +-------+--------+
        v                v
 GeminiWebProvider   future provider
        |
 Persistent browser runtime
        |
 Gemini Web
```

Consumers depend only on the HTTP boundary. They do not import internal
provider, browser, or scheduler modules.

## Layers

### Transport/API

`app.py` owns request validation, request IDs, error normalization, OpenAPI
routes, lifecycle, and the CLI. Public LLM routes are `/health`, `/v1/models`,
`/v1/chat/completions`, and `/api/generate`.

### Provider abstraction

`ProviderManager` resolves a model alias through `ProviderRegistry` and calls a
provider contract (`generate`, `health_check`, `auth_status`, `keepalive`, model
discovery, and management metadata). Adding a provider does not require a new
FastAPI route.

### Gemini browser/session runtime

`GeminiWebProvider` owns the provider operation lock, health state, timeout,
keepalive strategy, and persistent profile policy. `gemini_web.py` owns the
Playwright lifecycle and Gemini Web internal request transport. The service
does not type prompts into the Gemini UI or scrape answer DOM content.

The profile defaults to `var/profiles/gemini-main`. Browser state stays on disk
so a service restart can reuse the authenticated session without copying
cookies or tokens.

### Scheduler

The scheduler only invokes each provider's `keepalive()` according to that
provider's policy. Gemini's default policy is daily at 00:00 in
`Asia/Bangkok`. Provider operations are serialized per provider/profile; no
global lock is shared with future providers.

### Management UI

The dashboard at `/` is a thin server-served HTML/CSS/vanilla-JS management
surface. It uses the same management APIs and is not the integration boundary.

## Runtime behavior

Each generation is stateless from the API perspective. The caller must include
any context it needs in `messages` or `prompt`. A generation timeout returns
`504 GENERATION_TIMEOUT`; authentication loss returns `401 AUTH_REQUIRED`;
provider/runtime failures return `503 PROVIDER_UNAVAILABLE` or
`502 UPSTREAM_ERROR`.

The service binds to `0.0.0.0:4444` by default so other hosts can call it. The
current phase intentionally has no API authentication or rate limiting; the
operator must restrict network exposure externally.

## Deliberately unsupported

- Official Gemini API, Vertex AI, Gemini CLI, and other inference backends.
- API authentication, rate limiting, and reverse-proxy auth in this service.
- Streaming/SSE and authoritative token usage.
- Shared browser profiles between providers.
