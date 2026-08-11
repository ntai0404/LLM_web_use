import asyncio
import base64
import json
import math
import mimetypes
import os
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Page, Response, async_playwright


GEMINI_URL = "https://gemini.google.com/app"
STREAM_GENERATE_URL = (
    "https://gemini.google.com/_/BardChatUi/data/"
    "assistant.lamda.BardFrontendService/StreamGenerate"
)
UPLOAD_URL = "https://content-push.googleapis.com/upload"
BATCH_EXEC_URL = "https://gemini.google.com/_/BardChatUi/data/batchexecute"


# The canonical names come from Gemini Web's live otAQ7b model-list RPC. Limits
# below are reference specifications for the corresponding Google model, not a
# claim that the undocumented Web endpoint exposes the same hard controls.
MODEL_REFERENCES = {
    "3.5 Flash-Lite": {
        "reference_model_code": "gemini-3.5-flash-lite",
        "input_token_limit_reference": 1_048_576,
        "output_token_limit_reference": 65_536,
        "default_thinking_level_reference": "minimal",
    },
    "3.6 Flash": {
        "reference_model_code": "gemini-3.6-flash",
        "input_token_limit_reference": 1_048_576,
        "output_token_limit_reference": 65_536,
        "default_thinking_level_reference": "medium",
    },
    "3.1 Pro": {
        "reference_model_code": "gemini-3.1-pro-preview",
        "input_token_limit_reference": 1_048_576,
        "output_token_limit_reference": 65_536,
        "default_thinking_level_reference": "high",
    },
}

LAST_OBSERVED_MODEL_IDS = {
    "3.5 Flash-Lite": "8c46e95b1a07cecc",
    "3.6 Flash": "56fdd199312815e2",
    "3.1 Pro": "e6fa609c3fa255c0",
}

# Real end-to-end measurements from this workspace on 2026-08-11. They are
# fallback baselines until enough newer samples exist in var/latency_metrics.json.
BASELINE_LATENCY_SAMPLES = [
    {
        "timestamp": "2026-08-11T00:00:00+00:00",
        "model_id": "8c46e95b1a07cecc",
        "canonical_name": "3.5 Flash-Lite",
        "input_tokens_estimated": 2613,
        "image_count": 1,
        "duration_seconds": 10.29,
        "source": "runtime_multimodal_benchmark",
    },
    {
        "timestamp": "2026-08-11T00:00:00+00:00",
        "model_id": "56fdd199312815e2",
        "canonical_name": "3.6 Flash",
        "input_tokens_estimated": 2613,
        "image_count": 1,
        "duration_seconds": 61.62,
        "source": "runtime_multimodal_benchmark",
    },
    {
        "timestamp": "2026-08-11T00:00:00+00:00",
        "model_id": "e6fa609c3fa255c0",
        "canonical_name": "3.1 Pro",
        "input_tokens_estimated": 2613,
        "image_count": 1,
        "duration_seconds": 49.33,
        "source": "runtime_multimodal_benchmark",
    },
]

TOKEN_ESTIMATOR = "approximate_utf8_bytes_div_4_plus_258_per_768px_image_tile"


def estimate_text_tokens(text: str) -> int:
    """Google documents roughly four characters/token; UTF-8 is safer for VN text."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4)) if text else 0


def _image_dimensions(content: bytes) -> Optional[tuple[int, int]]:
    if content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
        return struct.unpack(">II", content[16:24])
    if content.startswith(b"\xff\xd8"):
        position = 2
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        while position + 4 <= len(content):
            if content[position] != 0xFF:
                position += 1
                continue
            marker = content[position + 1]
            position += 2
            if marker in {0xD8, 0xD9}:
                continue
            if position + 2 > len(content):
                break
            segment_length = int.from_bytes(content[position:position + 2], "big")
            if segment_length < 2 or position + segment_length > len(content):
                break
            if marker in sof_markers and segment_length >= 7:
                height = int.from_bytes(content[position + 3:position + 5], "big")
                width = int.from_bytes(content[position + 5:position + 7], "big")
                return width, height
            position += segment_length
    return None


def estimate_image_tokens(content: bytes) -> tuple[int, Optional[tuple[int, int]]]:
    dimensions = _image_dimensions(content)
    if dimensions is None:
        return 258, None
    width, height = dimensions
    if width <= 384 and height <= 384:
        return 258, dimensions
    tiles = math.ceil(width / 768) * math.ceil(height / 768)
    return max(1, tiles) * 258, dimensions


def truncate_by_estimated_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if estimate_text_tokens(text) <= max_tokens:
        return text, False
    encoded = text.encode("utf-8")[: max_tokens * 4]
    return encoded.decode("utf-8", errors="ignore").rstrip(), True


SESSION_CHECK_SCRIPT = r"""
async () => {
  const response = await fetch("https://gemini.google.com/app", {
    credentials: "include",
    cache: "no-store"
  });
  if (!response.ok) return false;
  const html = await response.text();
  return /"SNlM0e":\s*".+?"/.test(html);
}
"""


MODEL_LIST_SCRIPT = r"""
async ({ endpoint }) => {
  const matchValue = (source, key) => {
    const match = source.match(new RegExp('"' + key + '":\\s*"(.*?)"'));
    return match ? match[1] : null;
  };
  const nested = (value, path, fallback = null) => {
    let current = value;
    for (const key of path) {
      if (current === null || current === undefined || !(key in Object(current))) return fallback;
      current = current[key];
    }
    return current === null || current === undefined ? fallback : current;
  };
  const parseFrames = (raw) => {
    const source = raw.startsWith(")]}'") ? raw.slice(4) : raw;
    let position = 0;
    const frames = [];
    while (position < source.length) {
      while (position < source.length && /\s/.test(source[position])) position += 1;
      const marker = source.slice(position).match(/^(\d+)\n/);
      if (!marker) break;
      const length = Number(marker[1]);
      const contentStart = position + marker[1].length;
      const contentEnd = contentStart + length;
      if (contentEnd > source.length) break;
      const chunk = source.slice(contentStart, contentEnd).trim();
      position = contentEnd;
      if (!chunk) continue;
      try {
        const parsed = JSON.parse(chunk);
        if (Array.isArray(parsed)) frames.push(...parsed);
        else frames.push(parsed);
      } catch (_) {}
    }
    return frames;
  };
  const appResponse = await fetch("https://gemini.google.com/app", {
    credentials: "include",
    cache: "no-store"
  });
  if (!appResponse.ok) throw new Error(`Gemini /app returned HTTP ${appResponse.status}`);
  const appHtml = await appResponse.text();
  const accessToken = matchValue(appHtml, "SNlM0e");
  const buildLabel = matchValue(appHtml, "cfb2h");
  const sessionId = matchValue(appHtml, "FdrFJe");
  const language = matchValue(appHtml, "TuX5cc") || "en";
  if (!accessToken) throw new Error("AUTH_REQUIRED: current Gemini page token is unavailable");

  const query = new URLSearchParams({
    rpcids: "otAQ7b",
    hl: language,
    _reqid: String(Math.floor(10000 + Math.random() * 90000)),
    rt: "c",
    "source-path": "/app"
  });
  if (buildLabel) query.set("bl", buildLabel);
  if (sessionId) query.set("f.sid", sessionId);
  const body = new URLSearchParams({
    at: accessToken,
    "f.req": JSON.stringify([[["otAQ7b", "[]", null, "generic"]]])
  });
  const response = await fetch(`${endpoint}?${query.toString()}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
      "X-Same-Domain": "1",
      "x-goog-ext-525001261-jspb": "[1,null,null,null,null,null,null,null,[4]]",
      "x-goog-ext-73010989-jspb": "[0]",
      "x-goog-ext-73010990-jspb": "[0]"
    },
    body: body.toString()
  });
  if (!response.ok) throw new Error(`Gemini model RPC returned HTTP ${response.status}`);
  const raw = await response.text();
  for (const frame of parseFrames(raw)) {
    const bodyString = nested(frame, [2]);
    if (typeof bodyString !== "string") continue;
    try {
      const rpcBody = JSON.parse(bodyString);
      const rawModels = nested(rpcBody, [15], []);
      const tierFlags = nested(rpcBody, [16], []);
      const capabilityFlags = nested(rpcBody, [17], []);
      let capacity = 1;
      let capacityField = 12;
      if (tierFlags.includes(21)) { capacity = 1; capacityField = 13; }
      else if (tierFlags.includes(22)) { capacity = 2; capacityField = 13; }
      else if (capabilityFlags.includes(115)) capacity = 4;
      else if (tierFlags.includes(16) || capabilityFlags.includes(106)) capacity = 3;
      else if (tierFlags.includes(8) || (!capabilityFlags.includes(106) && capabilityFlags.includes(19))) capacity = 2;
      return rawModels
        .filter((item) => Array.isArray(item) && item[0] && item[1])
        .map((item) => ({
          id: item[0],
          display_name: item[1],
          canonical_name: item[11] || item[1],
          description: item[2] || "",
          capacity,
          capacity_field: capacityField,
          rollout_name: item[8] || null
        }));
    } catch (_) {}
  }
  throw new Error("Gemini model RPC returned no model list");
}
"""


STREAM_GENERATE_SCRIPT = r"""
async ({ prompt, endpoint, uploadEndpoint, files, selectedModel }) => {
  const matchValue = (source, key) => {
    const match = source.match(new RegExp('"' + key + '":\\s*"(.*?)"'));
    return match ? match[1] : null;
  };

  const nested = (value, path, fallback = null) => {
    let current = value;
    for (const key of path) {
      if (current === null || current === undefined || !(key in Object(current))) {
        return fallback;
      }
      current = current[key];
    }
    return current === null || current === undefined ? fallback : current;
  };

  const parseFrames = (raw) => {
    const source = raw.startsWith(")]}'") ? raw.slice(4) : raw;
    let position = 0;
    const frames = [];
    while (position < source.length) {
      while (position < source.length && /\s/.test(source[position])) position += 1;
      const marker = source.slice(position).match(/^(\d+)\n/);
      if (!marker) break;
      const length = Number(marker[1]);
      const contentStart = position + marker[1].length;
      const contentEnd = contentStart + length;
      if (contentEnd > source.length) break;
      const chunk = source.slice(contentStart, contentEnd).trim();
      position = contentEnd;
      if (!chunk) continue;
      try {
        const parsed = JSON.parse(chunk);
        if (Array.isArray(parsed)) frames.push(...parsed);
        else frames.push(parsed);
      } catch (_) {
        // Ignore non-data frames. Never log the response body.
      }
    }
    return frames;
  };

  const appResponse = await fetch("https://gemini.google.com/app", {
    credentials: "include",
    cache: "no-store"
  });
  if (!appResponse.ok) throw new Error(`Gemini /app returned HTTP ${appResponse.status}`);
  const appHtml = await appResponse.text();
  const accessToken = matchValue(appHtml, "SNlM0e");
  const buildLabel = matchValue(appHtml, "cfb2h");
  const sessionId = matchValue(appHtml, "FdrFJe");
  const language = matchValue(appHtml, "TuX5cc") || "en";
  const pushId = matchValue(appHtml, "qKIAYe") || "feeds/mcudyrk2a4khkz";
  if (!accessToken) throw new Error("AUTH_REQUIRED: current Gemini page token is unavailable");

  const fileData = [];
  for (const file of files) {
    const binary = atob(file.base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    const multipart = new FormData();
    multipart.append("file", new Blob([bytes], { type: file.mime }), file.name);
    const uploadResponse = await fetch(uploadEndpoint, {
      method: "POST",
      credentials: "include",
      headers: {
        "X-Tenant-Id": "bard-storage",
        "Push-ID": pushId
      },
      body: multipart
    });
    if (!uploadResponse.ok) {
      throw new Error(`Gemini upload returned HTTP ${uploadResponse.status}`);
    }
    const uploadId = (await uploadResponse.text()).trim();
    if (!uploadId) throw new Error("Gemini upload returned an empty identifier");
    fileData.push([[uploadId], file.name]);
  }

  const inner = Array(69).fill(null);
  inner[0] = [prompt, 0, null, fileData.length ? fileData : null, null, null, 0];
  inner[1] = [language];
  inner[2] = ["", "", "", null, null, null, null, null, null, ""];
  inner[6] = [1];
  inner[7] = 1;
  inner[10] = 1;
  inner[11] = 0;
  inner[17] = [[0]];
  inner[18] = 0;
  inner[27] = 1;
  inner[30] = [4];
  inner[41] = [1];
  inner[45] = 1;
  inner[53] = 0;
  const requestUuid = crypto.randomUUID().toUpperCase();
  inner[59] = requestUuid;
  inner[61] = [];
  inner[68] = 2;

  const query = new URLSearchParams({
    hl: language,
    _reqid: String(Math.floor(10000 + Math.random() * 90000)),
    rt: "c"
  });
  if (buildLabel) query.set("bl", buildLabel);
  if (sessionId) query.set("f.sid", sessionId);

  const requestBody = new URLSearchParams({
    at: accessToken,
    "f.req": JSON.stringify([null, JSON.stringify(inner)])
  });
  const modelHeaders = {
    "x-goog-ext-73010989-jspb": "[0]",
    "x-goog-ext-73010990-jspb": "[0]"
  };
  if (selectedModel) {
    const capacityTail = selectedModel.capacity_field === 13
      ? `null,${selectedModel.capacity}`
      : String(selectedModel.capacity);
    modelHeaders["x-goog-ext-525001261-jspb"] =
      `[1,null,null,null,"${selectedModel.id}",null,null,0,[4],null,null,${capacityTail}]`;
  }
  const response = await fetch(`${endpoint}?${query.toString()}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
      "X-Same-Domain": "1",
      "x-goog-ext-525005358-jspb": JSON.stringify([requestUuid, 1]),
      ...modelHeaders
    },
    body: requestBody.toString()
  });
  if (!response.ok) throw new Error(`StreamGenerate returned HTTP ${response.status}`);

  const responseBody = await response.text();
  let finalText = "";
  let errorCode = null;
  for (const frame of parseFrames(responseBody)) {
    const code = nested(frame, [5, 2, 0, 1, 0]);
    if (code) errorCode = code;
    const innerJson = nested(frame, [2]);
    if (typeof innerJson !== "string") continue;
    try {
      const payload = JSON.parse(innerJson);
      for (const candidate of nested(payload, [4], [])) {
        const text = nested(candidate, [1, 0], "");
        if (typeof text === "string" && text) finalText = text;
      }
    } catch (_) {
      // Ignore status frames that do not contain a candidate.
    }
  }
  if (!finalText) {
    throw new Error(errorCode ? `Gemini stream error ${errorCode}` : "Gemini stream contained no text candidate");
  }
  return {
    text: finalText,
    host: new URL(endpoint).host,
    httpStatus: response.status,
    uploadCount: fileData.length,
    modelId: selectedModel ? selectedModel.id : null,
    modelDisplayName: selectedModel ? selectedModel.display_name : "default"
  };
}
"""


class AuthRequired(RuntimeError):
    pass


class GeminiWebError(RuntimeError):
    pass


class GeminiWebClient:
    def __init__(
        self,
        profile_dir: str = "var/gemini-profile",
        headless: bool = True,
        timeout_ms: int = 120_000,
    ) -> None:
        self.profile_dir = str(Path(profile_dir).resolve())
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._pw = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._lock = asyncio.Lock()
        self.last_request_host: Optional[str] = None
        self.last_request_endpoint: Optional[str] = None
        self.last_http_status: Optional[int] = None
        self.last_upload_host: Optional[str] = None
        self.last_upload_status: Optional[int] = None
        self.last_upload_count = 0
        self.last_model_id: Optional[str] = None
        self.last_model_display_name: Optional[str] = None
        self.last_observed_model_id: Optional[str] = None
        self.last_call_metrics: dict = {}
        self.available_models: list[dict] = []
        self._metrics_path = Path(self.profile_dir).parent / "latency_metrics.json"
        self._latency_samples = self._load_latency_samples()

    def _load_latency_samples(self) -> list[dict]:
        try:
            raw = json.loads(self._metrics_path.read_text(encoding="utf-8"))
            samples = raw.get("samples", []) if isinstance(raw, dict) else []
            loaded = [item for item in samples if isinstance(item, dict)][-200:]
            return loaded or [dict(item) for item in BASELINE_LATENCY_SAMPLES]
        except (OSError, ValueError, TypeError):
            return [dict(item) for item in BASELINE_LATENCY_SAMPLES]

    def _save_latency_sample(self, sample: dict) -> None:
        self._latency_samples.append(sample)
        self._latency_samples = self._latency_samples[-200:]
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "samples": self._latency_samples}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self._metrics_path)

    def latency_estimate(
        self,
        model_id: Optional[str],
        input_tokens: Optional[int] = None,
        image_count: Optional[int] = None,
    ) -> Optional[dict]:
        candidates = [
            sample for sample in self._latency_samples
            if sample.get("model_id") == model_id and sample.get("duration_seconds") is not None
        ]
        if not candidates:
            return None
        if input_tokens is not None:
            target_images = image_count or 0
            candidates.sort(
                key=lambda sample: abs(
                    math.log1p(int(sample.get("input_tokens_estimated") or 0))
                    - math.log1p(input_tokens)
                ) + 0.5 * abs(int(sample.get("image_count") or 0) - target_images)
            )
            matched = candidates[:10]
        else:
            matched = candidates[-10:]
        durations = sorted(float(sample["duration_seconds"]) for sample in matched)
        p90_index = max(0, math.ceil(len(durations) * 0.9) - 1)
        return {
            "median_seconds": round(statistics.median(durations), 2),
            "p90_seconds": round(durations[p90_index], 2),
            "matched_samples": len(matched),
            "total_model_samples": len(candidates),
            "basis": "nearest_runtime_samples" if input_tokens is not None else "recent_runtime_samples",
        }

    def _enrich_models(self, models: list[dict]) -> list[dict]:
        enriched = []
        for model in models:
            item = dict(model)
            canonical_name = str(item.get("canonical_name") or item.get("display_name") or "")
            reference = MODEL_REFERENCES.get(canonical_name)
            if reference:
                item.update(reference)
                item["token_limits_scope"] = "official_model_reference_not_web_endpoint_guarantee"
            item["latency"] = self.latency_estimate(str(item.get("id") or ""))
            enriched.append(item)
        return enriched

    async def start(self) -> None:
        if self._context is not None:
            return

        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        browser_channel = os.getenv("BROWSER_CHANNEL", "chrome").strip() or None
        launch_kwargs = {
            "user_data_dir": self.profile_dir,
            "headless": self.headless,
            "viewport": {"width": 1365, "height": 900},
            "args": ["--no-first-run", "--no-default-browser-check", "--disable-quic"],
        }
        if browser_channel:
            launch_kwargs["channel"] = browser_channel

        try:
            self._context = await self._pw.chromium.launch_persistent_context(**launch_kwargs)
        except Exception:
            launch_kwargs.pop("channel", None)
            self._context = await self._pw.chromium.launch_persistent_context(**launch_kwargs)

        self._context.set_default_timeout(self.timeout_ms)
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        self._page.on("response", self._observe_response)

    def _observe_response(self, response: Response) -> None:
        if response.url.startswith(UPLOAD_URL):
            parsed = urlsplit(response.url)
            self.last_upload_host = parsed.hostname
            self.last_upload_status = response.status
            return
        if not response.url.startswith(STREAM_GENERATE_URL):
            return
        parsed = urlsplit(response.url)
        self.last_request_host = parsed.hostname
        self.last_request_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.last_http_status = response.status
        header = response.request.headers.get("x-goog-ext-525001261-jspb")
        if header:
            try:
                header_data = json.loads(header)
                if isinstance(header_data, list) and len(header_data) > 4:
                    self.last_observed_model_id = header_data[4]
            except (json.JSONDecodeError, TypeError):
                self.last_observed_model_id = None

    async def stop(self) -> None:
        if self._context is not None:
            await self._context.close()
        self._context = None
        self._page = None
        if self._pw is not None:
            await self._pw.stop()
        self._pw = None

    async def bootstrap_login(self) -> None:
        self.headless = False
        await self.start()
        assert self._page is not None
        await self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
        print("\nChrome opened with the persistent Gemini profile.")
        print("Log in to Google/Gemini in that window; this process detects it automatically.")
        deadline = time.monotonic() + 600
        while await self.auth_required():
            if time.monotonic() >= deadline:
                raise AuthRequired("Login was not detected within 10 minutes.")
            await asyncio.sleep(2)
        print(f"Login state saved in: {self.profile_dir}")
        await self.stop()

    async def auth_required(self) -> bool:
        await self.start()
        assert self._page is not None
        if "accounts.google.com" in (self._page.url or "").lower():
            return True
        try:
            return not bool(await self._page.evaluate(SESSION_CHECK_SCRIPT))
        except Exception:
            return True

    async def health(self) -> dict:
        try:
            await self.start()
            assert self._page is not None
            if not self._page.url.startswith("https://gemini.google.com"):
                await self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
            auth = "required" if await self.auth_required() else "ok"
            return {
                "ok": auth == "ok",
                "auth": auth,
                "url": self._page.url,
                "transport": "internal-stream-generate",
                "browser_visible": not self.headless,
                "last_request_host": self.last_request_host,
                "last_request_endpoint": self.last_request_endpoint,
                "last_http_status": self.last_http_status,
                "last_upload_host": self.last_upload_host,
                "last_upload_status": self.last_upload_status,
                "last_upload_count": self.last_upload_count,
                "last_model_id": self.last_model_id,
                "last_model_display_name": self.last_model_display_name,
                "last_observed_model_id": self.last_observed_model_id,
                "last_call_metrics": self.last_call_metrics,
            }
        except Exception as exc:
            return {"ok": False, "auth": "unknown", "error": str(exc)}

    async def list_models(self, refresh: bool = False) -> list[dict]:
        await self.start()
        assert self._page is not None
        if self.available_models and not refresh:
            return self.available_models
        if not self._page.url.startswith("https://gemini.google.com"):
            await self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
        try:
            result = await self._page.evaluate(
                MODEL_LIST_SCRIPT,
                {"endpoint": BATCH_EXEC_URL},
            )
        except Exception as exc:
            raise GeminiWebError(str(exc)) from exc
        if not isinstance(result, list) or not result:
            raise GeminiWebError("Gemini Web returned an empty model list")
        self.available_models = self._enrich_models(result)
        return self.available_models

    async def _select_model(self, model: Optional[str]) -> Optional[dict]:
        requested_model = (model or "").strip()
        if requested_model.lower() in {"", "gemini-web", "default", "auto"}:
            return None
        models = await self.list_models()
        requested_lower = requested_model.lower()
        selected_model = next(
            (
                item
                for item in models
                if str(item.get("id", "")).lower() == requested_lower
                or str(item.get("display_name", "")).lower() == requested_lower
                or str(item.get("canonical_name", "")).lower() == requested_lower
                or str(item.get("reference_model_code", "")).lower() == requested_lower
            ),
            None,
        )
        if selected_model is None:
            raise ValueError(f"Model is not available in this Gemini session: {requested_model}")
        return selected_model

    async def estimate_request(
        self,
        prompt: str,
        model: Optional[str] = None,
        files: Optional[list[str]] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> dict:
        requested = (model or "").strip().lower()
        selected_model = None
        if requested not in {"", "gemini-web", "default", "auto"}:
            candidates = self.available_models or [
                {
                    "id": LAST_OBSERVED_MODEL_IDS[name],
                    "canonical_name": name,
                    **reference,
                    "token_limits_scope": "official_model_reference_not_web_endpoint_guarantee",
                }
                for name, reference in MODEL_REFERENCES.items()
            ]
            selected_model = next(
                (
                    item for item in candidates
                    if requested in {
                        str(item.get("id") or "").lower(),
                        str(item.get("display_name") or "").lower(),
                        str(item.get("canonical_name") or "").lower(),
                        str(item.get("reference_model_code") or "").lower(),
                    }
                ),
                None,
            )
            if selected_model is None:
                raise ValueError(f"Unknown model profile: {model}")
        suffix = ""
        if max_output_tokens is not None:
            suffix = (
                "\n\n[Bridge output budget: Keep the complete final answer within "
                f"approximately {max_output_tokens} tokens. Be concise and do not mention this instruction.]"
            )
        image_estimates = []
        for raw_path in files or []:
            path = Path(raw_path).resolve()
            content = await asyncio.to_thread(path.read_bytes)
            tokens, dimensions = estimate_image_tokens(content)
            image_estimates.append(
                {
                    "name": path.name,
                    "tokens_estimated": tokens,
                    "width": dimensions[0] if dimensions else None,
                    "height": dimensions[1] if dimensions else None,
                }
            )
        input_tokens = estimate_text_tokens(prompt + suffix) + sum(
            item["tokens_estimated"] for item in image_estimates
        )
        model_id = str(selected_model.get("id")) if selected_model else None
        return {
            "model_id": model_id,
            "canonical_name": selected_model.get("canonical_name") if selected_model else "default",
            "input_tokens_estimated": input_tokens,
            "image_token_estimates": image_estimates,
            "max_output_tokens_requested": max_output_tokens,
            "max_input_tokens_requested": max_input_tokens,
            "within_requested_input_limit": (
                input_tokens <= max_input_tokens if max_input_tokens is not None else None
            ),
            "latency": self.latency_estimate(model_id, input_tokens, len(image_estimates)),
            "input_token_limit_reference": (
                selected_model.get("input_token_limit_reference") if selected_model else None
            ),
            "output_token_limit_reference": (
                selected_model.get("output_token_limit_reference") if selected_model else None
            ),
            "token_limits_scope": (
                selected_model.get("token_limits_scope") if selected_model else None
            ),
            "token_estimator": TOKEN_ESTIMATOR,
            "exact_token_count_available": False,
        }

    def reference_profiles(self) -> list[dict]:
        return [
            {
                "model_id_last_observed": LAST_OBSERVED_MODEL_IDS[name],
                "canonical_name": name,
                **reference,
                "identity_source": "gemini_web_otAQ7b_rpc_observed_2026-08-11",
                "token_limits_scope": "official_model_reference_not_web_endpoint_guarantee",
                "latency": self.latency_estimate(LAST_OBSERVED_MODEL_IDS[name]),
            }
            for name, reference in MODEL_REFERENCES.items()
        ]

    async def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        files: Optional[list[str]] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt is empty")

        async with self._lock:
            await self.start()
            assert self._page is not None
            if not self._page.url.startswith("https://gemini.google.com"):
                await self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
            if await self.auth_required():
                raise AuthRequired("Google/Gemini login is required. Run: python app.py bootstrap")

            selected_model = await self._select_model(model)

            file_payloads = []
            image_token_estimates = []
            for raw_path in files or []:
                path = Path(raw_path).resolve()
                if not path.is_file():
                    raise ValueError(f"Image file does not exist: {path}")
                content = await asyncio.to_thread(path.read_bytes)
                image_tokens, dimensions = estimate_image_tokens(content)
                image_token_estimates.append(
                    {
                        "name": path.name,
                        "tokens_estimated": image_tokens,
                        "width": dimensions[0] if dimensions else None,
                        "height": dimensions[1] if dimensions else None,
                    }
                )
                file_payloads.append(
                    {
                        "name": path.name,
                        "mime": mimetypes.guess_type(path.name)[0]
                        or "application/octet-stream",
                        "base64": base64.b64encode(content).decode("ascii"),
                    }
                )

            control_suffix = ""
            if max_output_tokens is not None:
                control_suffix = (
                    "\n\n[Bridge output budget: Keep the complete final answer within "
                    f"approximately {max_output_tokens} tokens. Be concise and do not mention this instruction.]"
                )
            effective_prompt = prompt + control_suffix
            input_tokens_estimated = estimate_text_tokens(effective_prompt) + sum(
                item["tokens_estimated"] for item in image_token_estimates
            )
            model_input_reference = (
                selected_model.get("input_token_limit_reference") if selected_model else None
            )
            effective_input_limit = max_input_tokens or model_input_reference
            if effective_input_limit and input_tokens_estimated > int(effective_input_limit):
                raise ValueError(
                    "Estimated input tokens "
                    f"({input_tokens_estimated}) exceed limit ({effective_input_limit})"
                )

            predicted_latency = self.latency_estimate(
                str(selected_model.get("id")) if selected_model else None,
                input_tokens_estimated,
                len(file_payloads),
            )
            started_at = time.perf_counter()

            try:
                result = await self._page.evaluate(
                    STREAM_GENERATE_SCRIPT,
                    {
                        "prompt": effective_prompt,
                        "endpoint": STREAM_GENERATE_URL,
                        "uploadEndpoint": UPLOAD_URL,
                        "files": file_payloads,
                        "selectedModel": selected_model,
                    },
                )
            except Exception as exc:
                message = str(exc)
                if "AUTH_REQUIRED" in message:
                    raise AuthRequired(
                        "Google/Gemini login is required. Run: python app.py bootstrap"
                    ) from exc
                raise GeminiWebError(message) from exc

            if not isinstance(result, dict) or result.get("host") != "gemini.google.com":
                raise GeminiWebError("Internal Gemini request returned an invalid origin")
            self.last_upload_count = int(result.get("uploadCount") or 0)
            self.last_model_id = result.get("modelId")
            self.last_model_display_name = result.get("modelDisplayName")
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                raise GeminiWebError("Internal Gemini stream returned no text")
            text = text.strip()
            output_truncated = False
            if max_output_tokens is not None:
                text, output_truncated = truncate_by_estimated_tokens(text, max_output_tokens)
            duration_seconds = round(time.perf_counter() - started_at, 3)
            output_tokens_estimated = estimate_text_tokens(text)
            model_id = result.get("modelId")
            canonical_name = selected_model.get("canonical_name") if selected_model else "default"
            sample = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
                "canonical_name": canonical_name,
                "input_tokens_estimated": input_tokens_estimated,
                "output_tokens_estimated": output_tokens_estimated,
                "image_count": len(file_payloads),
                "duration_seconds": duration_seconds,
            }
            self._save_latency_sample(sample)
            self.last_call_metrics = {
                **sample,
                "token_estimator": TOKEN_ESTIMATOR,
                "image_token_estimates": image_token_estimates,
                "max_input_tokens_requested": max_input_tokens,
                "max_output_tokens_requested": max_output_tokens,
                "output_truncated": output_truncated,
                "output_control": (
                    "best_effort_web_instruction_plus_local_estimated_cap"
                    if max_output_tokens is not None else "none"
                ),
                "exact_token_count_available": False,
                "predicted_latency_before_call": predicted_latency,
            }
            return text
