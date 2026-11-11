"""Scripted stand-in for the LLM, so the pipeline can be tested without an API
key. It emits valid tool calls by following the toolkit's feedback. Its numbers
are NOT results; use it only with --mock to check that the code runs.
"""

import json

import numpy as np


def _last_tool(messages):
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                return m.get("name"), json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                return m.get("name"), {}
    return None, None


def _names(tools):
    return {t["function"]["name"] for t in tools or []}


def _call(name, args=None):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"mock-{name}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args or {})}}]}


def _pick_changes(info, reasons, rng):
    """Move the features the feedback flags as controllable toward typical benign values."""
    ranges = info.get("allowed_ranges", {})
    budget = int(info.get("max_features_changed_per_attempt", 3))
    movable = [f for f, (lo, hi) in ranges.items() if lo < hi]
    flagged = [r["feature"] for r in reasons or []
               if r.get("attacker_can_change") and r["feature"] in movable]
    rest = [f for f in movable if f not in flagged]
    rng.shuffle(rest)
    out = []
    for f in (flagged + rest)[:budget]:
        lo, hi = ranges[f]
        band = next((r["typical_benign_range"] for r in reasons or [] if r["feature"] == f), None)
        v = float(np.mean(band)) if band else float(np.expm1(rng.uniform(np.log1p(lo), np.log1p(hi))))
        out.append({"feature": f, "value": min(max(v, lo), hi)})
    return out


SEV = {"T1498": "critical", "T1499": "high", "T1110": "high", "T1190": "critical"}
RESP = {"T1498": "Rate-limit the source and enable upstream scrubbing",
        "T1499": "Block the source IP at the firewall and set a connection limit",
        "T1110": "Lock the account and block the source IP",
        "T1190": "Isolate the host and patch the service"}


class MockLLM:
    """Covers the single-agent loop; the multi-agent runner calls the role
    helpers below directly, so the same scripted logic serves both."""
    ready = True

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.model = "mock"

    def usage(self):
        return {"llm_calls": self.calls, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "cost_usd": 0.0, "llm_seconds": 0.0,
                "rate_limit_retries": 0, "model": "mock"}

    def chat(self, messages, tools=None):
        self.calls += 1
        names = _names(tools)
        last, res = _last_tool(messages)
        res = res or {}
        if last is None:
            return _call("get_attack_seed")
        if last in ("get_attack_seed", "submit_variant"):
            if last == "get_attack_seed" and "error" in res:
                return _call("retrain_detector")
            seed_info = res if last == "get_attack_seed" else self._last_seed_info(messages)
            if last == "submit_variant" and (res.get("evaded") or res.get("attempts_left_for_seed", 0) <= 0):
                return _call("get_attack_seed")
            reasons = res.get("detector_now_says") or res.get("rejection_reason")
            return _call("submit_variant", {"seed_id": seed_info.get("seed_id"),
                                            "changes": _pick_changes(seed_info, reasons, self.rng)})
        if last == "retrain_detector":
            return _call("get_attack_seed") if "Round" in str(res.get("next", "")) else _call("triage_alerts")
        if last == "triage_alerts":
            decisions = []
            for a in res.get("alerts", []):
                tech = a["technique"].split()[0]
                decisions.append({"alert_id": a["alert_id"], "severity": SEV.get(tech, "medium"),
                                  "response": RESP.get(tech, "Investigate the flow")})
            return _call("submit_triage", {"decisions": decisions})
        if last == "submit_triage":
            return _call("finish", {"summary": "Scripted test run."})
        return {"role": "assistant", "content": "done"}

    @staticmethod
    def _last_seed_info(messages):
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("name") == "get_attack_seed":
                try:
                    info = json.loads(m["content"])
                    if "seed_id" in info:
                        return info
                except (json.JSONDecodeError, TypeError):
                    pass
        return {}

