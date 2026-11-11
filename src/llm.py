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


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up: retrying will not help."""


class LLM:
    def __init__(self) -> None:
        load_env()
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.effort = os.environ.get("LLM_REASONING_EFFORT", "")
        self.wait = float(os.environ.get("LLM_RETRY_WAIT", "20"))
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0          # time spent waiting for the API (latency)
        self.retries = 0
        # Optional prices in USD per 1M tokens (0 on the free tier).
        self.price_in = float(os.environ.get("LLM_PRICE_IN_PER_M", "0") or 0)
        self.price_out = float(os.environ.get("LLM_PRICE_OUT_PER_M", "0") or 0)
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
                t0 = time.time()
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
                self.seconds += time.time() - t0
                break
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")
                daily = err.code == 429 and ("PerDay" in detail or "per day" in detail.lower())
                if daily:
                    raise QuotaExhausted(
                        "The API's daily free quota for this model is used up. It resets at "
                        "midnight Pacific time. Use another model in .env (each model has its own "
                        "quota), enable billing, or run with --policy random.") from err
                if err.code not in (429, 500, 503) or attempt == 4:
                    raise RuntimeError(f"HTTP {err.code}: {detail[:400]}") from err
                self.retries += 1
                print(f"   (API busy or rate limit reached, waiting {self.wait:.0f} s ...)", flush=True)
                time.sleep(self.wait)
        self.calls += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return data["choices"][0]["message"]

    def usage(self) -> dict:
        cost = (self.prompt_tokens * self.price_in + self.completion_tokens * self.price_out) / 1e6
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "cost_usd": round(cost, 4), "llm_seconds": round(self.seconds, 1),
                "rate_limit_retries": self.retries, "model": self.model}
