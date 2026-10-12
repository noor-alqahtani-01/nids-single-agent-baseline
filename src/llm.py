"""Minimal client for any OpenAI-compatible chat completions API, with tool calls.

Settings come from .env (see .env.example): LLM_BASE_URL, LLM_API_KEY,
LLM_MODEL and, optionally, LLM_REASONING_EFFORT.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLM:
    def __init__(self) -> None:
        load_env()
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.effort = os.environ.get("LLM_REASONING_EFFORT", "")
        self.wait = float(os.environ.get("LLM_RETRY_WAIT", "20"))
        self.calls = 0
        self.ready = bool(self.base_url and self.api_key and self.model)

    def chat(self, messages: list, tools: list | None = None) -> dict:
        """Send the conversation; return the assistant message (may hold tool_calls)."""
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.effort:
            payload["reasoning_effort"] = self.effort
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as err:
                if err.code not in (429, 500, 503) or attempt == 4:
                    detail = err.read().decode(errors="replace")[:400]
                    raise RuntimeError(f"HTTP {err.code}: {detail}") from err
                print(f"   (API busy or rate limit reached, waiting {self.wait:.0f} s ...)", flush=True)
                time.sleep(self.wait)
        self.calls += 1
        return data["choices"][0]["message"]
