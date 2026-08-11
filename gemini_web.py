import asyncio
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Page, Response, async_playwright


GEMINI_URL = "https://gemini.google.com/app"
STREAM_GENERATE_URL = (
    "https://gemini.google.com/_/BardChatUi/data/"
    "assistant.lamda.BardFrontendService/StreamGenerate"
)


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


STREAM_GENERATE_SCRIPT = r"""
async ({ prompt, endpoint }) => {
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
  if (!accessToken) throw new Error("AUTH_REQUIRED: current Gemini page token is unavailable");

  const inner = Array(69).fill(null);
  inner[0] = [prompt, 0, null, null, null, null, 0];
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
  const response = await fetch(`${endpoint}?${query.toString()}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
      "X-Same-Domain": "1",
      "x-goog-ext-525005358-jspb": JSON.stringify([requestUuid, 1])
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
    httpStatus: response.status
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
            "args": ["--no-first-run", "--no-default-browser-check"],
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
        if not response.url.startswith(STREAM_GENERATE_URL):
            return
        parsed = urlsplit(response.url)
        self.last_request_host = parsed.hostname
        self.last_request_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.last_http_status = response.status

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
        print("Log in to Google/Gemini in that window, then return here.")
        await asyncio.to_thread(input, "Press ENTER after Gemini chat is usable... ")
        if await self.auth_required():
            raise AuthRequired("Gemini still appears unauthenticated.")
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
            }
        except Exception as exc:
            return {"ok": False, "auth": "unknown", "error": str(exc)}

    async def ask(self, prompt: str, model: Optional[str] = None) -> str:
        del model  # Gemini Web selects its current/default model internally.
        if not prompt.strip():
            raise ValueError("Prompt is empty")

        async with self._lock:
            await self.start()
            assert self._page is not None
            if not self._page.url.startswith("https://gemini.google.com"):
                await self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
            if await self.auth_required():
                raise AuthRequired("Google/Gemini login is required. Run: python app.py bootstrap")

            try:
                result = await self._page.evaluate(
                    STREAM_GENERATE_SCRIPT,
                    {"prompt": prompt, "endpoint": STREAM_GENERATE_URL},
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
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                raise GeminiWebError("Internal Gemini stream returned no text")
            return text.strip()
