"""Single defender agent for network intrusion detection (M4 baseline).

Loop for each flow:
    perceive -> read one network flow record
    decide   -> tool 1: classify_flow (trained detector)
                tool 2: lookup_attack (MITRE ATT&CK)
                LLM:    write a short reason for the analyst
    act      -> emit an alert record and append it to the alert log

The verdict always comes from the detector. The LLM only explains it, so it
cannot flip a decision. If no LLM is configured or a call fails, the agent
falls back to a template sentence and keeps going.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

from attack_map import lookup_attack

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "detector.joblib"
BENIGN = "BENIGN"
# Features shown to the LLM and used in the template reason (if present).
KEY_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
]

SYSTEM_PROMPT = (
    "You are a SOC analyst assistant inside a network intrusion detection agent. "
    "A trained classifier has already decided the verdict for one network flow. "
    "Write the reason for the analyst in at most two short sentences, citing the "
    "feature values that are most consistent with the verdict. Do not change or "
    "question the verdict. Reply with the reason only."
)


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (no extra library)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMClient:
    """Minimal client for any OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        load_env()
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.enabled = bool(self.base_url and self.api_key and self.model)
        self.disabled_reason = None if self.enabled else (
            "LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are not all set in .env"
        )

    def complete(self, user_message: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            # Generous cap: "thinking" models spend tokens before they answer.
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4000")),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        # Optional, for thinking models (for example "low"): faster replies.
        effort = os.environ.get("LLM_REASONING_EFFORT", "")
        if effort:
            payload["reasoning_effort"] = effort
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        # Free API tiers limit requests per minute: wait and retry on HTTP 429.
        wait = float(os.environ.get("LLM_RETRY_WAIT", "20"))
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as err:
                if err.code != 429 or attempt == 3:
                    raise
                print(f"          (rate limit reached, waiting {wait:.0f} s ...)", flush=True)
                time.sleep(wait)
        content = data["choices"][0]["message"].get("content")
        if not content or not content.strip():
            raise ValueError("the LLM returned an empty reply")
        return " ".join(content.split())


class NIDSAgent:
    def __init__(self, use_llm: bool = True, log_path: Path | None = None) -> None:
        if not MODEL_PATH.exists():
            raise SystemExit(f"{MODEL_PATH} not found. Run: python src/train.py")
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.model.n_jobs = 1  # one flow at a time: threads only add overhead
        self.features = bundle["features"]
        self.log_path = log_path
        self.llm = LLMClient() if use_llm else None
        self.llm_active = bool(self.llm and self.llm.enabled)
        self.llm_note = None
        if not use_llm:
            self.llm_note = "LLM turned off with --no-llm; using template reasons."
        elif not self.llm_active:
            self.llm_note = f"LLM not configured ({self.llm.disabled_reason}); using template reasons."

    # ---- perceive -------------------------------------------------------
    def perceive(self, flow: dict) -> pd.DataFrame:
        """Keep only the detector's features, in training order."""
        missing = [f for f in self.features if f not in flow]
        if missing:
            raise ValueError(f"Flow is missing features: {missing[:5]}")
        return pd.DataFrame([[flow[f] for f in self.features]], columns=self.features)

    # ---- tools ----------------------------------------------------------
    def classify_flow(self, x: pd.DataFrame) -> dict:
        """Tool 1: the trained detector."""
        proba = self.model.predict_proba(x)[0]
        best = int(proba.argmax())
        return {"attack_type": str(self.model.classes_[best]), "confidence": float(proba[best])}

    # ---- decide ---------------------------------------------------------
    def decide(self, flow: dict) -> dict:
        x = self.perceive(flow)
        result = self.classify_flow(x)
        technique = lookup_attack(result["attack_type"])
        verdict = "benign" if result["attack_type"] == BENIGN else "attack"
        key = {f: flow[f] for f in KEY_FEATURES if f in flow}
        decision = {
            "verdict": verdict,
            "attack_type": result["attack_type"],
            "confidence": round(result["confidence"], 4),
            **technique,
            "key_features": key,
        }
        decision["reason"], decision["reason_source"] = self.explain(decision)
        return decision

    def explain(self, decision: dict) -> tuple[str, str]:
        if self.llm_active:
            try:
                facts = {k: decision[k] for k in
                         ("verdict", "attack_type", "confidence", "technique_id",
                          "technique_name", "key_features")}
                return self.llm.complete(json.dumps(facts, default=str)), "llm"
            except Exception as err:  # any failure: keep the run going
                # Stop calling the LLM after the first failure so the run finishes.
                self.llm_active = False
                self.llm_note = f"LLM call failed ({err}); switched to template reasons."
        return self.template_reason(decision), "template"

    @staticmethod
    def template_reason(d: dict) -> str:
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        feats = ", ".join(f"{k}={fmt(v)}" for k, v in list(d["key_features"].items())[:3])
        if d["verdict"] == "benign":
            return f"Detector classified the flow as benign with confidence {d['confidence']:.2f} ({feats})."
        return (f"Detector classified the flow as {d['attack_type']} with confidence "
                f"{d['confidence']:.2f}, mapped to {d['technique_id']} {d['technique_name']} ({feats}).")

    # ---- act ------------------------------------------------------------
    def act(self, flow_id: int, decision: dict) -> dict:
        record = {"flow_id": flow_id, **decision}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return record

    def handle(self, flow_id: int, flow: dict) -> dict:
        return self.act(flow_id, self.decide(flow))
