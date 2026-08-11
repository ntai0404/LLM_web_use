from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_CONTENT_CHARS = 4_194_304
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES = 10
FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DATA_IMAGE_PATTERN = re.compile(
    r"^data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)
IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class OpenAIOutputError(RuntimeError):
    pass


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str = Field(min_length=1)
    detail: str | None = None


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | str | None = None

    @model_validator(mode="after")
    def validate_part(self):
        if self.type == "text":
            if self.text is None or not self.text.strip():
                raise ValueError("text content part must contain non-blank text")
            if len(self.text) > MAX_CONTENT_CHARS:
                raise ValueError(f"text content part exceeds {MAX_CONTENT_CHARS} characters")
        elif self.image_url is None:
            raise ValueError("image_url content part must contain image_url")
        return self


class FunctionCallInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str | dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not FUNCTION_NAME_PATTERN.fullmatch(value):
            raise ValueError("invalid function name")
        return value


class AssistantToolCallInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCallInput


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[AssistantToolCallInput] | None = None

    @model_validator(mode="after")
    def validate_message(self):
        if isinstance(self.content, str):
            if not self.content.strip():
                raise ValueError("content must not be blank")
            if len(self.content) > MAX_CONTENT_CHARS:
                raise ValueError(f"content exceeds {MAX_CONTENT_CHARS} characters")
        elif isinstance(self.content, list) and not self.content:
            raise ValueError("content part list must not be empty")

        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are valid only on assistant messages")
        if self.name is not None and not FUNCTION_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("invalid message name")
        return self


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not FUNCTION_NAME_PATTERN.fullmatch(value):
            raise ValueError("invalid function name")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            validator_for(value).check_schema(value)
        except Exception as exc:
            raise ValueError(f"invalid function JSON Schema: {exc}") from exc
        return value


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    # OpenAI clients add optional fields over time. Unknown optional values are
    # accepted as controlled no-ops rather than breaking Browser Use with 422.
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="gemini-web", min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=65_536)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=65_536)
    max_input_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    response_format: dict[str, Any] | None = None
    tools: list[ToolDefinition] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value

    @model_validator(mode="after")
    def validate_contract(self):
        if self.stream:
            raise ValueError("stream=true is not implemented")
        text_chars = 0
        image_count = 0
        for message in self.messages:
            if isinstance(message.content, str):
                text_chars += len(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if part.type == "text" and part.text:
                        text_chars += len(part.text)
                    elif part.type == "image_url":
                        image_count += 1
        if text_chars > MAX_CONTENT_CHARS:
            raise ValueError(f"combined message content exceeds {MAX_CONTENT_CHARS} characters")
        if image_count > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images are supported")

        response_type = (self.response_format or {}).get("type", "text")
        if response_type not in {"text", "json_object", "json_schema"}:
            raise ValueError("unsupported response_format type")
        if response_type == "json_schema":
            config = (self.response_format or {}).get("json_schema")
            schema = config.get("schema") if isinstance(config, dict) else None
            if not isinstance(schema, dict):
                raise ValueError("response_format.json_schema.schema must be an object")
            try:
                validator_for(schema).check_schema(schema)
            except Exception as exc:
                raise ValueError(f"invalid response JSON Schema: {exc}") from exc

        tool_names = {tool.function.name for tool in self.tools or []}
        if len(tool_names) != len(self.tools or []):
            raise ValueError("tool function names must be unique")
        choice = self.tool_choice
        if isinstance(choice, str):
            if choice not in {"auto", "required", "none"}:
                raise ValueError("tool_choice must be auto, required, none, or a forced function")
            if choice == "required" and not tool_names:
                raise ValueError("tool_choice=required needs at least one tool")
        elif isinstance(choice, dict):
            function = choice.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if choice.get("type") != "function" or not isinstance(name, str):
                raise ValueError("forced tool_choice must identify a function")
            if name not in tool_names:
                raise ValueError("forced tool_choice function is not in the tool allowlist")
        elif choice is not None:
            raise ValueError("invalid tool_choice")
        if response_type in {"json_object", "json_schema"} and self.tools and choice != "none":
            raise ValueError("response_format and active tool calling cannot be combined")
        return self

    @property
    def output_token_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


@dataclass(frozen=True)
class ToolPlan:
    mode: Literal["auto", "required", "forced"]
    definitions: dict[str, FunctionDefinition]
    forced_name: str | None
    parallel: bool


@dataclass(frozen=True)
class PreparedOpenAIRequest:
    prompt: str
    files: list[str]
    output_schema: dict[str, Any] | None
    require_json_object: bool
    tool_plan: ToolPlan | None
    ignored_options: tuple[str, ...]


def _image_url_value(value: ImageURL | str) -> str:
    return value if isinstance(value, str) else value.url


def _validate_image_signature(mime: str, content: bytes) -> bool:
    if mime == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    if mime == "image/gif":
        return content.startswith((b"GIF87a", b"GIF89a"))
    if mime == "image/webp":
        return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    return False


def _decode_data_image(url: str) -> tuple[str, bytes]:
    match = DATA_IMAGE_PATTERN.fullmatch(url)
    if match is None:
        if url.startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not supported")
        raise ValueError("image_url must be a base64 data URL for PNG, JPEG, WEBP, or GIF")
    mime = match.group(1).lower()
    encoded = match.group(2)
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise ValueError(f"decoded image exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_url contains invalid base64") from exc
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"decoded image must be between 1 and {MAX_IMAGE_BYTES} bytes")
    if not _validate_image_signature(mime, content):
        raise ValueError("image bytes do not match the declared MIME type")
    return mime, content


def _tool_plan(request: ChatCompletionRequest) -> ToolPlan | None:
    if not request.tools or request.tool_choice == "none":
        return None
    definitions = {tool.function.name: tool.function for tool in request.tools}
    choice = request.tool_choice or "auto"
    if isinstance(choice, dict):
        return ToolPlan(
            mode="forced",
            definitions=definitions,
            forced_name=choice["function"]["name"],
            parallel=request.parallel_tool_calls,
        )
    return ToolPlan(
        mode=choice,
        definitions=definitions,
        forced_name=None,
        parallel=request.parallel_tool_calls,
    )


def _structured_config(request: ChatCompletionRequest) -> tuple[dict[str, Any] | None, bool]:
    response_format = request.response_format or {}
    response_type = response_format.get("type", "text")
    if response_type == "json_object":
        return {"type": "object"}, True
    if response_type == "json_schema":
        return response_format["json_schema"]["schema"], False
    return None, False


def _format_function(function: FunctionCallInput) -> dict[str, Any]:
    return {
        "name": function.name,
        "arguments": function.arguments,
    }


@contextmanager
def prepare_openai_request(request: ChatCompletionRequest) -> Iterator[PreparedOpenAIRequest]:
    temp_dir: str | None = None
    files: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for index, message in enumerate(request.messages):
            content_items: list[dict[str, Any]] | None = None
            if isinstance(message.content, str):
                content_items = [{"type": "text", "text": message.content}]
            elif isinstance(message.content, list):
                content_items = []
                for part in message.content:
                    if part.type == "text":
                        content_items.append({"type": "text", "text": part.text})
                        continue
                    assert part.image_url is not None
                    mime, content = _decode_data_image(_image_url_value(part.image_url))
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp(prefix="llm_web_openai_")
                    attachment_number = len(files) + 1
                    path = Path(temp_dir) / f"image_{attachment_number}{IMAGE_EXTENSIONS[mime]}"
                    path.write_bytes(content)
                    files.append(str(path))
                    content_items.append(
                        {
                            "type": "image",
                            "attachment_index": attachment_number,
                            "mime_type": mime,
                        }
                    )
            record: dict[str, Any] = {
                "index": index,
                "role": message.role,
                "content": content_items,
            }
            if message.name is not None:
                record["name"] = message.name
            if message.tool_call_id is not None:
                record["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                record["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": _format_function(call.function),
                    }
                    for call in message.tool_calls
                ]
            records.append(record)

        prompt = (
            "Process the following ordered OpenAI chat transcript. Preserve each role, "
            "message boundary, name, tool_call_id, and tool call exactly. Images are "
            "attached in attachment_index order. Respond to the final conversation state "
            "without mentioning this serialization.\n\n<openai_messages_json>\n"
            + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
            + "\n</openai_messages_json>"
        )

        schema, require_object = _structured_config(request)
        if schema is not None:
            prompt += (
                "\n\nReturn exactly one valid JSON value and no Markdown fence, commentary, "
                "or thinking wrapper. It must satisfy this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )

        plan = _tool_plan(request)
        if plan is not None:
            definitions = [
                {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                }
                for definition in plan.definitions.values()
            ]
            choice_instruction = {
                "auto": "Use a tool only when appropriate; otherwise return content.",
                "required": "You must select at least one allowed function.",
                "forced": f"You must call only the function {plan.forced_name}.",
            }[plan.mode]
            prompt += (
                "\n\nYou may not execute functions. Select only from the allowlist below. "
                f"{choice_instruction} Return exactly one JSON object with keys tool_calls "
                "and content. tool_calls is a list of {name, arguments} objects; arguments "
                "must be a JSON object matching that function's schema. content must be a "
                "string only when tool_calls is empty. Do not use Markdown.\n"
                + json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))
            )

        known = set(ChatCompletionRequest.model_fields)
        ignored = tuple(sorted((request.model_extra or {}).keys() - known))
        yield PreparedOpenAIRequest(
            prompt=prompt,
            files=files,
            output_schema=schema,
            require_json_object=require_object,
            tool_plan=plan,
            ignored_options=ignored,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_json_output(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise OpenAIOutputError("upstream did not return valid JSON")


def validate_structured_output(
    text: str,
    schema: dict[str, Any],
    require_object: bool,
) -> str:
    value = parse_json_output(text)
    if require_object and not isinstance(value, dict):
        raise OpenAIOutputError("upstream JSON response is not an object")
    try:
        validator_for(schema)(schema).validate(value)
    except JsonSchemaValidationError as exc:
        raise OpenAIOutputError(f"upstream JSON failed schema validation: {exc.message}") from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_tool_output(
    text: str,
    plan: ToolPlan,
) -> tuple[str | None, list[dict[str, Any]] | None, Literal["stop", "tool_calls"]]:
    value = parse_json_output(text)
    if not isinstance(value, dict):
        raise OpenAIOutputError("tool selection response must be a JSON object")
    raw_calls = value.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise OpenAIOutputError("tool_calls must be a list")
    if plan.mode in {"required", "forced"} and not raw_calls:
        raise OpenAIOutputError("tool_choice requires a function call")
    if not plan.parallel and len(raw_calls) > 1:
        raise OpenAIOutputError("parallel_tool_calls=false allows at most one call")

    calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise OpenAIOutputError("tool call must be an object")
        name = raw_call.get("name")
        if name not in plan.definitions:
            raise OpenAIOutputError("upstream selected a function outside the allowlist")
        if plan.forced_name is not None and name != plan.forced_name:
            raise OpenAIOutputError("upstream did not select the forced function")
        arguments = raw_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise OpenAIOutputError("tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise OpenAIOutputError("tool arguments must be a JSON object")
        definition = plan.definitions[name]
        try:
            validator_for(definition.parameters)(definition.parameters).validate(arguments)
        except JsonSchemaValidationError as exc:
            raise OpenAIOutputError(
                f"tool arguments failed schema validation: {exc.message}"
            ) from exc
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )

    if calls:
        return None, calls, "tool_calls"
    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OpenAIOutputError("automatic tool response returned neither calls nor content")
    return content.strip(), None, "stop"


def repair_prompt(prompt: str, invalid_output: str, reason: str) -> str:
    clipped = invalid_output[:8000]
    return (
        prompt
        + "\n\nThe previous response was invalid: "
        + reason
        + "\nRepair it now. Return only the required JSON value. Previous response:\n"
        + clipped
    )
