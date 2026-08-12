import base64
import copy
import json
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

import app as app_module
from tests.asgi_client import ASGITestClient
from providers import (
    ProviderBusy,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class CapturingManager:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or ["COMPAT_OK"])
        self.error = error
        self.calls = []
        self.files_seen = []

    async def generate(self, prompt, model, files=None, **options):
        if self.error:
            raise self.error
        files = list(files or [])
        self.calls.append(
            {"prompt": prompt, "model": model, "files": files, "options": options}
        )
        self.files_seen.extend(files)
        for path in files:
            assert os.path.isfile(path)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return SimpleNamespace(
            text=self.responses[index],
            model=model,
            provider="gemini-web",
            metadata={},
        )

    @asynccontextmanager
    async def generation_session(self, model):
        async def generate(prompt, session_model=None, files=None, **options):
            return await self.generate(
                prompt,
                session_model or model,
                files=files,
                **options,
            )

        yield generate


class BrowserAction(BaseModel):
    click_element: dict[str, int]


class BrowserAgentOutput(BaseModel):
    thinking: str
    evaluation_previous_goal: str
    memory: str
    next_goal: str
    action: list[BrowserAction]


class OpenAIBrowserUseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = ASGITestClient(app_module.app)

    def test_default_agent_parameters_are_noop_compatible_and_max_tokens_map(self):
        manager = CapturingManager()
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0,
                    "top_p": 0.9,
                    "frequency_penalty": 0.25,
                    "presence_penalty": -0.25,
                    "seed": 7,
                    "max_tokens": 99,
                    "max_completion_tokens": 77,
                    "stream": False,
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(77, manager.calls[0]["options"]["max_output_tokens"])
        self.assertEqual("COMPAT_OK", response.json()["choices"][0]["message"]["content"])

    def test_structured_output_repairs_once_and_parses_directly_with_pydantic(self):
        expected = {
            "thinking": "Button 17 is the target",
            "evaluation_previous_goal": "Page inspected",
            "memory": "Use the visible login button",
            "next_goal": "Click button 17",
            "action": [{"click_element": {"index": 17}}],
        }
        manager = CapturingManager(
            responses=["not json", f"```json\n{json.dumps(expected)}\n```"]
        )
        schema = BrowserAgentOutput.model_json_schema()
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [
                        {
                            "role": "user",
                            "content": "DOM: <button data-index='17'>Continue</button>",
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "browser_agent_output",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(manager.calls))
        self.assertEqual([], manager.calls[1]["files"])
        self.assertNotIn("DOM: <button", manager.calls[1]["prompt"])
        parsed = BrowserAgentOutput.model_validate_json(
            response.json()["choices"][0]["message"]["content"]
        )
        self.assertEqual(17, parsed.action[0].click_element["index"])

    def test_structured_first_pass_valid_has_no_repair_and_preserves_schema(self):
        expected = {
            "thinking": "Inspect",
            "evaluation_previous_goal": "Ready",
            "memory": "Button visible",
            "next_goal": "Click",
            "action": [{"click_element": {"index": 17}}],
        }
        schema = BrowserAgentOutput.model_json_schema()
        original_schema = copy.deepcopy(schema)
        manager = CapturingManager(responses=[json.dumps(expected)])
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "click 17"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "browser_agent_output",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(manager.calls))
        self.assertEqual(original_schema, schema)
        self.assertIn(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            manager.calls[0]["prompt"],
        )

    def test_invalid_after_one_repair_is_never_http_200(self):
        manager = CapturingManager(responses=["not json", '{"wrong":true}'])
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "click 17"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "browser_agent_output",
                            "strict": True,
                            "schema": BrowserAgentOutput.model_json_schema(),
                        },
                    },
                },
            )

        self.assertEqual(502, response.status_code)
        self.assertEqual(
            "STRUCTURED_OUTPUT_VALIDATION_FAILED",
            response.json()["error"],
        )
        self.assertEqual(2, response.json()["details"]["attempts"])
        self.assertEqual(2, len(manager.calls))

    def test_fake_upstream_prose_empty_and_html_fail_without_repair(self):
        outputs = [
            "I encountered an error. Please try again.",
            "",
            "<!doctype html><html><body>Sign in</body></html>",
        ]
        for output in outputs:
            manager = CapturingManager(responses=[output])
            with self.subTest(output=output[:20]), patch.object(
                app_module, "manager", manager
            ):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gemini-web",
                        "messages": [{"role": "user", "content": "return JSON"}],
                        "response_format": {"type": "json_object"},
                    },
                )
            self.assertEqual(502, response.status_code)
            self.assertEqual("UPSTREAM_ERROR", response.json()["error"])
            self.assertEqual(1, len(manager.calls))

    def test_arbitrary_prose_wrapping_json_requires_repair(self):
        manager = CapturingManager(
            responses=['Here is the result: {"ok":true}', '{"ok":true}']
        )
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {"type": "json_object"},
                },
            )
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(manager.calls))

    def test_known_error_phrase_inside_valid_json_is_not_reclassified(self):
        manager = CapturingManager(responses=['{"message":"Please try again"}'])
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {"type": "json_object"},
                },
            )
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(manager.calls))

    def test_json_object_strips_markdown_and_thinking_wrapper(self):
        manager = CapturingManager(
            responses=['<think>hidden</think>```json\n{"ok":true}\n```']
        )
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {"type": "json_object"},
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"ok": True},
            json.loads(response.json()["choices"][0]["message"]["content"]),
        )

    def test_data_image_is_temporary_and_cleaned_after_request(self):
        image = b"\x89PNG\r\n\x1a\ncompatibility-test"
        data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        manager = CapturingManager()
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the image"},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(manager.files_seen))
        self.assertFalse(os.path.exists(manager.files_seen[0]))
        self.assertNotIn(data_url, manager.calls[0]["prompt"])

    def test_remote_image_url_is_rejected_without_download(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-web",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/private.png"},
                            }
                        ],
                    }
                ],
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("INVALID_REQUEST", response.json()["error"])

    def test_required_tool_call_is_allowlisted_and_arguments_are_json(self):
        manager = CapturingManager(
            responses=[
                json.dumps(
                    {
                        "tool_calls": [
                            {"name": "click_element", "arguments": {"index": 17}}
                        ],
                        "content": None,
                    }
                )
            ]
        )
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "Click Continue"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "click_element",
                                "description": "Click a DOM element",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"}
                                    },
                                    "required": ["index"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                },
            )

        payload = response.json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("tool_calls", payload["choices"][0]["finish_reason"])
        call = payload["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual("click_element", call["function"]["name"])
        self.assertEqual(17, json.loads(call["function"]["arguments"])["index"])

    def test_invalid_tool_name_repairs_once_against_caller_allowlist(self):
        manager = CapturingManager(
            responses=[
                json.dumps(
                    {"tool_calls": [{"name": "not_allowed", "arguments": {}}], "content": None}
                ),
                json.dumps(
                    {
                        "tool_calls": [
                            {"name": "click_element", "arguments": {"index": 17}}
                        ],
                        "content": None,
                    }
                ),
            ]
        )
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "Click Continue"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "click_element",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer"}},
                                    "required": ["index"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    "tool_choice": "required",
                },
            )
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(manager.calls))
        self.assertEqual([], manager.calls[1]["files"])

    def test_invalid_tool_arguments_after_repair_fail_non_2xx(self):
        invalid = json.dumps(
            {
                "tool_calls": [
                    {"name": "click_element", "arguments": {"index": "seventeen"}}
                ],
                "content": None,
            }
        )
        manager = CapturingManager(responses=[invalid, invalid])
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "Click Continue"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "click_element",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer"}},
                                    "required": ["index"],
                                },
                            },
                        }
                    ],
                    "tool_choice": "required",
                },
            )
        self.assertEqual(502, response.status_code)
        self.assertEqual(
            "STRUCTURED_OUTPUT_VALIDATION_FAILED",
            response.json()["error"],
        )

    def test_tool_choice_auto_none_and_forced_are_supported(self):
        tool = {
            "type": "function",
            "function": {
                "name": "click_element",
                "parameters": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            },
        }
        cases = [
            (
                "auto",
                json.dumps({"tool_calls": [], "content": "No click needed"}),
                "stop",
            ),
            ("none", "Plain response", "stop"),
            (
                {"type": "function", "function": {"name": "click_element"}},
                json.dumps(
                    {
                        "tool_calls": [
                            {"name": "click_element", "arguments": {"index": 17}}
                        ],
                        "content": None,
                    }
                ),
                "tool_calls",
            ),
        ]
        for choice, upstream, finish_reason in cases:
            manager = CapturingManager(responses=[upstream])
            with self.subTest(choice=choice), patch.object(app_module, "manager", manager):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gemini-web",
                        "messages": [{"role": "user", "content": "continue"}],
                        "tools": [tool],
                        "tool_choice": choice,
                    },
                )
                self.assertEqual(200, response.status_code)
                self.assertEqual(finish_reason, response.json()["choices"][0]["finish_reason"])

    def test_assistant_tool_call_and_tool_result_are_serialized(self):
        manager = CapturingManager()
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [
                        {"role": "user", "content": "Inspect element"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_previous",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_element",
                                        "arguments": "{\"index\":17}",
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "name": "inspect_element",
                            "tool_call_id": "call_previous",
                            "content": "Button label is Continue",
                        },
                        {"role": "user", "content": "What was the label?"},
                    ],
                },
            )

        self.assertEqual(200, response.status_code)
        prompt = manager.calls[0]["prompt"]
        self.assertIn('"role":"tool"', prompt)
        self.assertIn('"tool_call_id":"call_previous"', prompt)
        self.assertIn('"name":"inspect_element"', prompt)

    def test_forced_tool_must_be_in_allowlist(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-web",
                "messages": [{"role": "user", "content": "do it"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "allowed",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "not_allowed"},
                },
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("VALIDATION_ERROR", response.json()["error"])

    def test_multiturn_roles_and_tool_metadata_remain_ordered(self):
        manager = CapturingManager()
        with patch.object(app_module, "manager", manager):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [
                        {"role": "system", "content": "Remember code BLUE"},
                        {"role": "user", "content": "What is the code?"},
                        {"role": "assistant", "content": "BLUE"},
                        {"role": "user", "content": "Repeat the prior code"},
                    ],
                },
            )

        self.assertEqual(200, response.status_code)
        prompt = manager.calls[0]["prompt"]
        positions = [
            prompt.index('"role":"system"'),
            prompt.index('"role":"user"'),
            prompt.index('"role":"assistant"'),
            prompt.rindex('"role":"user"'),
        ]
        self.assertEqual(sorted(positions), positions)
        self.assertIn("Remember code BLUE", prompt)

    def test_error_status_mapping_for_retry_clients(self):
        cases = [
            (ProviderRateLimited("rate limited", "7"), 429, "RATE_LIMITED", "7"),
            (ProviderUnavailable("browser down"), 503, "PROVIDER_UNAVAILABLE", None),
            (ProviderBusy("queue busy"), 503, "PROVIDER_BUSY", "1"),
            (ProviderTimeout("timeout"), 504, "GENERATION_TIMEOUT", None),
        ]
        for error, status, code, retry_after in cases:
            with self.subTest(code=code), patch.object(
                app_module, "manager", CapturingManager(error=error)
            ):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gemini-web",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                self.assertEqual(status, response.status_code)
                self.assertEqual(code, response.json()["error"])
                if retry_after is not None:
                    self.assertEqual(retry_after, response.headers["retry-after"])


if __name__ == "__main__":
    unittest.main()
