import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import Settings
from gemini_web import GeminiWebClient
from provider_manager import ProviderManager, ProviderNotFound, ProviderRegistry
from providers import (
    GeminiWebProvider,
    KeepalivePolicy,
    ProviderAuthRequired,
    ProviderError,
)
from scheduler import ProviderScheduler


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
settings = Settings.from_env()


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
    )
    registry.register(gemini)
    manager = ProviderManager(registry)
    return manager, ProviderScheduler(registry.all())


manager, provider_scheduler = build_manager(settings)
runtime_server: uvicorn.Server | None = None


class GenerateRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1)
    prompt_file: str | None = None
    image: str | None = None
    images: list[str] = Field(default_factory=list, max_length=10)
    model: str = "gemini-web"
    max_input_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    max_output_tokens: int | None = Field(default=None, ge=1, le=65_536)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-web"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=65_536)
    max_input_tokens: int | None = Field(default=None, ge=1, le=1_048_576)


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.role.strip().lower()
        label = {"system": "System", "assistant": "Assistant", "user": "User"}.get(
            role,
            message.role,
        )
        parts.append(f"{label}: {message.content}")
    return "\n\n".join(parts)


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
        prompt = resolve_project_file(request.prompt_file).read_text(encoding="utf-8")
    else:
        raise ValueError("Provide prompt or prompt_file")
    if not prompt.strip():
        raise ValueError("Prompt is empty")
    requested_images = ([request.image] if request.image else []) + request.images
    return prompt, [str(resolve_project_file(path)) for path in requested_images]


def provider_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ProviderNotFound):
        return HTTPException(
            status_code=404,
            detail={"error": "PROVIDER_NOT_FOUND", "message": str(exc)},
        )
    if isinstance(exc, ProviderAuthRequired):
        return HTTPException(
            status_code=401,
            detail={"error": "AUTH_REQUIRED", "message": str(exc)},
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail={"error": "INVALID_INPUT", "message": str(exc)},
        )
    return HTTPException(
        status_code=502,
        detail={"error": "PROVIDER_ERROR", "message": str(exc)},
    )


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
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return await manager.health_check()


@app.get("/models")
@app.get("/v1/models")
async def models():
    try:
        return {"object": "list", "data": await manager.list_models(refresh=True)}
    except Exception as exc:
        raise provider_http_error(exc)


@app.get("/model-profiles")
async def model_profiles():
    return {"object": "list", "data": manager.model_profiles()}


@app.post("/api/generate")
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


@app.post("/api/estimate")
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented")
    try:
        result = await manager.generate(
            messages_to_prompt(request.messages),
            request.model,
            max_input_tokens=request.max_input_tokens,
            max_output_tokens=request.max_tokens,
        )
    except Exception as exc:
        raise provider_http_error(exc)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


@app.post("/providers/{provider_name}/keepalive")
async def trigger_keepalive(provider_name: str):
    try:
        return await manager.trigger_keepalive(provider_name)
    except Exception as exc:
        raise provider_http_error(exc)


@app.post("/admin/shutdown")
async def shutdown_service() -> dict[str, Any]:
    if runtime_server is None:
        raise HTTPException(status_code=409, detail="Runtime server is not managed by app.py")
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
