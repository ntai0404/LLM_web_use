import argparse
import asyncio
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import Settings
from gemini_web import GeminiWebClient
from openai_compat import (
    ChatCompletionRequest,
    OpenAIOutputError,
    prepare_openai_request,
    repair_prompt,
    validate_structured_output,
    validate_tool_output,
)
from provider_manager import ProviderManager, ProviderNotFound, ProviderRegistry
from providers import (
    GeminiWebProvider,
    KeepalivePolicy,
    ProviderAuthRequired,
    ProviderBusy,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from scheduler import ProviderScheduler


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"
settings = Settings.from_env()
logger = logging.getLogger("llm_web.api")


class StructuredOutputMetrics:
    """Small process-local telemetry for the structured response boundary."""

    _COUNTERS = (
        "structured_requests",
        "first_pass_valid",
        "repaired",
        "failed_validation",
        "upstream_errors",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {name: 0 for name in self._COUNTERS}

    def increment(self, name: str) -> None:
        if name not in self._values:
            raise ValueError(f"unknown structured output metric: {name}")
        with self._lock:
            self._values[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {**self._values, "invalid_http_200": 0}


structured_output_metrics = StructuredOutputMetrics()


def build_manager(config: Settings) -> tuple[ProviderManager, ProviderScheduler]:
    registry = ProviderRegistry()
    gemini = GeminiWebProvider(
        GeminiWebClient(
            config.gemini_profile_dir,
            config.headless,
            config.gemini_timeout_ms,
        ),
        keepalive_policy=KeepalivePolicy(
            enabled=config.keepalive_enabled,
            timezone=config.keepalive_timezone,
            hour=config.keepalive_hour,
            minute=config.keepalive_minute,
        ),
        operation_timeout_seconds=config.gemini_timeout_ms / 1000,
        queue_timeout_seconds=config.gemini_queue_timeout_ms / 1000,
    )
    registry.register(gemini)
    manager = ProviderManager(registry)
    return manager, ProviderScheduler(registry.all())


manager, provider_scheduler = build_manager(settings)
runtime_server: uvicorn.Server | None = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(default=None, min_length=1, max_length=4_194_304)
    prompt_file: str | None = Field(default=None, min_length=1, max_length=4096)
    image: str | None = Field(default=None, min_length=1, max_length=4096)
    images: list[str] = Field(default_factory=list, max_length=10)
    model: str = Field(default="gemini-web", min_length=1, max_length=200)
    max_input_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    max_output_tokens: int | None = Field(default=None, ge=1, le=65_536)

    @field_validator("prompt", "prompt_file", "image")
    @classmethod
    def reject_blank_optional_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("images")
    @classmethod
    def validate_image_paths(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 4096 for item in value):
            raise ValueError("image paths must be non-blank and at most 4096 characters")
        return value

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model must not be blank")
        return normalized


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: Any | None = None


class GenerateResponse(BaseModel):
    model: str
    provider: str
    text: str
    metadata: dict[str, Any]


class ChatCompletionMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChoiceResponse(BaseModel):
    index: int
    message: ChatCompletionMessageResponse
    finish_reason: Literal["stop", "tool_calls"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoiceResponse]
    usage: None = None


class ApiException(Exception):
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        details: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details
        self.headers = headers or {}


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request"},
    401: {"model": ErrorResponse, "description": "Provider authentication required"},
    429: {"model": ErrorResponse, "description": "Provider rate limited"},
    413: {"model": ErrorResponse, "description": "Request body is too large"},
    404: {"model": ErrorResponse, "description": "Provider or route not found"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    502: {"model": ErrorResponse, "description": "Upstream provider failed"},
    503: {"model": ErrorResponse, "description": "Provider runtime unavailable"},
    504: {"model": ErrorResponse, "description": "Provider generation timed out"},
}


def resolve_project_file(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    resolved = candidate.resolve()
    if resolved != PROJECT_DIR and PROJECT_DIR not in resolved.parents:
        raise ValueError(f"File must be inside project directory: {raw_path}")
    if not resolved.is_file():
        raise ValueError(f"File does not exist: {raw_path}")
    return resolved


def resolve_generate_input(request: GenerateRequest) -> tuple[str, list[str]]:
    if request.prompt is not None:
        prompt = request.prompt
    elif request.prompt_file:
        try:
            prompt = resolve_project_file(request.prompt_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("Prompt file must be readable UTF-8 text") from exc
    else:
        raise ValueError("Provide prompt or prompt_file")
    if not prompt.strip():
        raise ValueError("Prompt is empty")
    requested_images = ([request.image] if request.image else []) + request.images
    return prompt, [str(resolve_project_file(path)) for path in requested_images]


def provider_http_error(exc: Exception) -> ApiException:
    if isinstance(exc, ProviderNotFound):
        return ApiException(404, "PROVIDER_NOT_FOUND", str(exc))
    if isinstance(exc, ProviderAuthRequired):
        return ApiException(401, "AUTH_REQUIRED", str(exc))
    if isinstance(exc, ProviderRateLimited):
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else {}
        return ApiException(429, "RATE_LIMITED", str(exc), headers=headers)
    if isinstance(exc, ProviderTimeout):
        return ApiException(504, "GENERATION_TIMEOUT", str(exc))
    if isinstance(exc, ProviderBusy):
        return ApiException(503, "PROVIDER_BUSY", str(exc), headers={"Retry-After": "1"})
    if isinstance(exc, ProviderUnavailable):
        return ApiException(503, "PROVIDER_UNAVAILABLE", str(exc))
    if isinstance(exc, ValueError):
        return ApiException(422, "INVALID_REQUEST", str(exc))
    if isinstance(exc, ProviderError):
        return ApiException(502, "UPSTREAM_ERROR", str(exc))
    return ApiException(500, "INTERNAL_ERROR", "Internal server error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await provider_scheduler.start()
    try:
        yield
    finally:
        await provider_scheduler.stop()
        await manager.shutdown()


app = FastAPI(
    title="Browser-backed LLM Provider Service",
    description=(
        "Local HTTP gateway for browser-backed LLM providers. "
        "Use /api/generate for the native contract or /v1/chat/completions "
        "for OpenAI-compatible clients."
    ),
    version="0.3.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
MAX_REQUEST_BYTES = 20 * 1024 * 1024


def request_id_for(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def error_response(
    request: Request,
    status_code: int,
    error: str,
    message: str,
    details: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = request_id_for(request)
    content: dict[str, Any] = {
        "error": error,
        "message": message,
        "request_id": request_id,
    }
    if details is not None:
        content["details"] = details
    response_headers = {"X-Request-ID": request_id}
    response_headers.update(headers or {})
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=response_headers,
    )


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    supplied = request.headers.get("x-request-id", "")
    request.state.request_id = (
        supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
    )
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return error_response(
                    request,
                    413,
                    "PAYLOAD_TOO_LARGE",
                    f"Request body exceeds {MAX_REQUEST_BYTES} bytes",
                )
        except ValueError:
            return error_response(
                request,
                400,
                "INVALID_CONTENT_LENGTH",
                "Invalid Content-Length header",
            )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ApiException)
async def handle_api_exception(request: Request, exc: ApiException):
    return error_response(
        request, exc.status_code, exc.error, exc.message, exc.details, exc.headers
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    details = [
        {
            "location": [str(item) for item in error.get("loc", ())],
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return error_response(
        request,
        422,
        "VALIDATION_ERROR",
        "Request validation failed",
        details,
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        error = str(exc.detail.get("error") or "HTTP_ERROR")
        message = str(exc.detail.get("message") or "Request failed")
    else:
        error = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            409: "CONFLICT",
        }.get(exc.status_code, "HTTP_ERROR")
        message = str(exc.detail)
    return error_response(request, exc.status_code, error, message)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, _: Exception):
    return error_response(
        request, 500, "INTERNAL_ERROR", "Internal server error"
    )


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get(
    "/health",
    summary="Service and provider readiness",
    description="Returns provider auth/runtime state for service consumers.",
)
async def health():
    payload = await manager.health_check()
    payload["structured_output"] = structured_output_metrics.snapshot()
    return payload


@app.get("/api/providers", summary="Discover registered providers")
async def providers(refresh: bool = False):
    return {
        "service": {
            "host": settings.host,
            "port": settings.port,
            "base_url": f"http://{settings.host}:{settings.port}",
            "version": app.version,
        },
        "providers": await manager.provider_list(refresh=refresh),
    }


@app.get(
    "/api/providers/{provider_name}",
    summary="Get provider metadata",
    responses=ERROR_RESPONSES,
)
async def provider_detail(provider_name: str, refresh: bool = False):
    try:
        return await manager.provider_detail(provider_name, refresh=refresh)
    except Exception as exc:
        raise provider_http_error(exc)


@app.post(
    "/api/providers/{provider_name}/test",
    summary="Run a real provider nonce test",
    responses=ERROR_RESPONSES,
)
async def test_provider(provider_name: str):
    try:
        result = await manager.test_provider(provider_name)
    except Exception as exc:
        raise provider_http_error(exc)
    if not result.get("success"):
        raise ApiException(
            502,
            "PROVIDER_TEST_FAILED",
            "Provider response did not match the verification nonce",
        )
    return result


@app.get("/models", summary="List provider models")
@app.get("/v1/models", summary="List models using OpenAI-compatible shape")
async def models():
    try:
        return {"object": "list", "data": await manager.list_models(refresh=True)}
    except Exception as exc:
        raise provider_http_error(exc)


@app.get("/model-profiles", summary="List model capability reference profiles")
async def model_profiles():
    return {"object": "list", "data": manager.model_profiles()}


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    summary="Generate with a registered provider",
    description="Native stateless generation contract routed by model alias.",
    responses=ERROR_RESPONSES,
)
async def generate(request: GenerateRequest):
    try:
        prompt, files = resolve_generate_input(request)
        result = await manager.generate(
            prompt,
            request.model,
            files=files,
            max_input_tokens=request.max_input_tokens,
            max_output_tokens=request.max_output_tokens,
        )
        return {
            "model": result.model,
            "provider": result.provider,
            "text": result.text,
            "metadata": result.metadata,
        }
    except Exception as exc:
        raise provider_http_error(exc)


@app.post(
    "/api/estimate",
    summary="Estimate request size and latency",
    responses=ERROR_RESPONSES,
)
async def estimate(request: GenerateRequest):
    try:
        prompt, files = resolve_generate_input(request)
        return await manager.estimate(
            prompt,
            request.model,
            files=files,
            max_input_tokens=request.max_input_tokens,
            max_output_tokens=request.max_output_tokens,
        )
    except Exception as exc:
        raise provider_http_error(exc)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create an OpenAI-compatible chat completion",
    description="Supports non-streaming text messages routed through ProviderManager.",
    responses=ERROR_RESPONSES,
)
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    request_id = request_id_for(http_request)
    started = time.perf_counter()
    structured = False
    first_pass_valid = False
    repair_attempted = False
    validation_result = "not_requested"
    try:
        with prepare_openai_request(request) as prepared:
            structured = prepared.tool_plan is not None or prepared.output_schema is not None
            if structured:
                structured_output_metrics.increment("structured_requests")

            def validate_output(text: str):
                if prepared.tool_plan is not None:
                    return validate_tool_output(text, prepared.tool_plan)
                if prepared.output_schema is not None:
                    return (
                        validate_structured_output(
                            text,
                            prepared.output_schema,
                            prepared.require_json_object,
                        ),
                        None,
                        "stop",
                    )
                return text, None, "stop"

            total_timeout = max(0.001, settings.gemini_timeout_ms / 1000)
            try:
                async with asyncio.timeout(total_timeout):
                    async with manager.generation_session(request.model) as generate:
                        result = await generate(
                            prepared.prompt,
                            request.model,
                            files=prepared.files,
                            max_input_tokens=request.max_input_tokens,
                            max_output_tokens=request.output_token_limit,
                        )
                        content: str | None = result.text
                        tool_calls: list[dict[str, Any]] | None = None
                        finish_reason: Literal["stop", "tool_calls"] = "stop"

                        if structured:
                            try:
                                content, tool_calls, finish_reason = validate_output(result.text)
                            except OpenAIOutputError as first_error:
                                if first_error.kind == "upstream":
                                    structured_output_metrics.increment("upstream_errors")
                                    validation_result = "upstream_error"
                                    raise ApiException(
                                        502,
                                        "UPSTREAM_ERROR",
                                        str(first_error),
                                        {"attempts": 1},
                                    ) from first_error

                                repair_attempted = True
                                repaired = await generate(
                                    repair_prompt(
                                        result.text,
                                        first_error,
                                        schema=prepared.output_schema,
                                        tool_plan=prepared.tool_plan,
                                    ),
                                    request.model,
                                    # Repair deliberately excludes original DOM/images.
                                    files=None,
                                    max_input_tokens=request.max_input_tokens,
                                    max_output_tokens=request.output_token_limit,
                                )
                                try:
                                    content, tool_calls, finish_reason = validate_output(repaired.text)
                                except OpenAIOutputError as second_error:
                                    if second_error.kind == "upstream":
                                        structured_output_metrics.increment("upstream_errors")
                                        validation_result = "upstream_error"
                                        error_code = "UPSTREAM_ERROR"
                                    else:
                                        structured_output_metrics.increment("failed_validation")
                                        validation_result = "failed_validation"
                                        error_code = "STRUCTURED_OUTPUT_VALIDATION_FAILED"
                                    raise ApiException(
                                        502,
                                        error_code,
                                        str(second_error),
                                        {
                                            "attempts": 2,
                                            "validation_errors": second_error.errors,
                                        },
                                    ) from second_error
                                structured_output_metrics.increment("repaired")
                                validation_result = "repaired"
                            else:
                                structured_output_metrics.increment("first_pass_valid")
                                first_pass_valid = True
                                validation_result = "first_pass_valid"
            except TimeoutError as exc:
                raise ProviderTimeout(
                    f"LLM-Web request exceeded total deadline of {total_timeout:g} seconds"
                ) from exc
    except Exception as exc:
        if structured and isinstance(exc, ProviderError):
            structured_output_metrics.increment("upstream_errors")
            validation_result = "upstream_error"
        logger.info(
            "request_id=%s provider=gemini-web structured=%s first_pass_valid=%s "
            "repair_attempted=%s validation_result=%s latency_seconds=%.3f",
            request_id,
            structured,
            first_pass_valid,
            repair_attempted,
            validation_result,
            time.perf_counter() - started,
        )
        if isinstance(exc, ApiException):
            raise
        raise provider_http_error(exc)
    logger.info(
        "request_id=%s provider=gemini-web structured=%s first_pass_valid=%s "
        "repair_attempted=%s validation_result=%s latency_seconds=%.3f",
        request_id,
        structured,
        first_pass_valid,
        repair_attempted,
        validation_result,
        time.perf_counter() - started,
    )
    return {
        "id": f"chatcmpl-{request_id[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


@app.post(
    "/providers/{provider_name}/keepalive",
    include_in_schema=False,
    deprecated=True,
)
@app.post(
    "/api/providers/{provider_name}/keepalive",
    summary="Run provider keepalive now",
    responses=ERROR_RESPONSES,
)
async def trigger_keepalive(provider_name: str):
    try:
        result = await manager.trigger_keepalive(provider_name)
    except Exception as exc:
        raise provider_http_error(exc)
    if not result.get("success"):
        provider_status = result.get("status")
        if provider_status == "AUTH_REQUIRED":
            raise ApiException(401, "AUTH_REQUIRED", "Provider login is required")
        raise ApiException(
            502,
            "KEEPALIVE_FAILED",
            "Provider keepalive verification failed",
            {"provider_status": provider_status},
        )
    return result


@app.post("/admin/shutdown", include_in_schema=False)
async def shutdown_service(request: Request) -> dict[str, Any]:
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        raise ApiException(403, "FORBIDDEN", "Shutdown is restricted to loopback clients")
    if runtime_server is None:
        raise ApiException(409, "RUNTIME_NOT_MANAGED", "Runtime server is not managed by app.py")
    runtime_server.should_exit = True
    return {"accepted": True, "graceful": True}


async def do_bootstrap() -> None:
    bootstrap_client = GeminiWebClient(
        settings.gemini_profile_dir,
        headless=False,
        timeout_ms=settings.gemini_timeout_ms,
    )
    await bootstrap_client.bootstrap_login()


async def do_generate(
    prompt_file: str,
    images: list[str],
    model: str,
) -> None:
    prompt_path = resolve_project_file(prompt_file)
    image_paths = [str(resolve_project_file(path)) for path in images]
    cli_manager, _ = build_manager(settings)
    try:
        result = await cli_manager.generate(
            prompt_path.read_text(encoding="utf-8"),
            model,
            files=image_paths,
        )
        print(result.text)
    finally:
        await cli_manager.shutdown()


def run_server() -> None:
    global runtime_server
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
    runtime_server = uvicorn.Server(config)
    runtime_server.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser-backed LLM Provider Service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap", help="Authenticate the persistent Gemini Chrome profile")
    subparsers.add_parser("serve", help="Run the provider service")
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate from a UTF-8 prompt file and optional images",
    )
    generate_parser.add_argument("--prompt-file", default="prompt.md")
    generate_parser.add_argument("--image", action="append", default=[])
    generate_parser.add_argument("--model", default="gemini-web")
    args = parser.parse_args()
    if args.command == "bootstrap":
        asyncio.run(do_bootstrap())
    elif args.command == "generate":
        asyncio.run(do_generate(args.prompt_file, args.image, args.model))
    else:
        run_server()


if __name__ == "__main__":
    main()
