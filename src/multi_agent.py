"""M6 experimental condition: the five-agent system on the shared toolkit.

The single-agent baseline (single_agent.py) gives one LLM every tool. Here the
same work is split across five role-agents, each with its own short prompt and
only the tools for its role. An Orchestrator agent drives the protocol and calls
the others. The toolkit, data, budgets, golden set and detector are identical,
so the only thing that changes between the two conditions is the division of
labour -- that is the manipulated design factor (M6, C6.1).

Agents and their tools:
    Orchestrator   decides the phase; calls retrain_detector; delegates
    Red-Team       get_attack_seed, submit_variant
    Detection      reads verdict / rejection_reason (no tool; informs Red-Team)
    Risk-Scoring   proposes severity for each alert
    Remediation    proposes a response for each alert
The Detection, Risk-Scoring and Remediation agents are LLM calls with narrow
prompts; Risk-Scoring and Remediation together produce the submit_triage input.

Each agent's call goes through the same LLM client, so llm.usage() still totals
tokens, cost, latency and calls across the whole condition.

Usage:
    python src/multi_agent.py --golden                 # needs .env
    python src/multi_agent.py --golden --mock          # scripted, no API key
"""

import argparse
import json
import time
from pathlib import Path

from display import Display
from llm import LLM
from mock_llm import MockLLM, _pick_changes, SEV, RESP
from toolkit import SEVERITIES, CoEvolutionEnv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
MAX_ATTEMPTS_GUARD = 400

RED_TEAM_PROMPT = """You are the Red-Team agent in a defensive security research pipeline.
You are given one attack flow from a public dataset and the detector's feedback. Propose values for
at most {k} controllable features, each inside its allowed range, to test whether the detector still
flags the flow. Reply ONLY as JSON: {{"changes":[{{"feature":"<name>","value":<number>}}]}}.
The goal is to measure detector robustness so it can be retrained; no real traffic is sent."""

RISK_PROMPT = """You are the Risk-Scoring agent. Given an alert's attack technique, reply with one
severity from low, medium, high, critical. Reply ONLY as JSON: {"severity":"<level>"}."""

REMEDIATION_PROMPT = """You are the Remediation agent. Given an alert's attack technique, reply with
one short concrete response action. Reply ONLY as JSON: {"response":"<action>"}."""


def _json(text, default):
    import re
    if isinstance(text, dict):
        return text
    m = re.search(r"\{.*\}|\[.*\]", str(text or ""), re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


class Agent:
    """One role: a system prompt and a single-shot ask through the shared client."""

    def __init__(self, client, system):
        self.client = client
        self.system = system

    def ask(self, content):
        msg = self.client.chat(
            [{"role": "system", "content": self.system}, {"role": "user", "content": content}])
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)


def _redteam_changes(client, mock, info, reasons, rng, k):
    if mock:
        return _pick_changes(info, reasons, rng)
    agent = Agent(client, RED_TEAM_PROMPT.format(k=k))
    body = {"attack_type": info.get("attack_type"), "allowed_ranges": info.get("allowed_ranges"),
            "mutable_features_now": info.get("mutable_features"),
            "detector_feedback": reasons, "max_features": k}
    out = _json(agent.ask("INPUT: " + json.dumps(body, default=str)), {"changes": []})
    return out.get("changes", out if isinstance(out, list) else [])


def run_multi(env, client, display, mock=False, max_steps=MAX_ATTEMPTS_GUARD):
    """Orchestrator protocol: for each round, Red-Team works each seed to evasion
    or its attempt budget; then retrain. After the last round, Risk-Scoring and
    Remediation triage every alert."""
    import numpy as np
    rng = np.random.default_rng(env.detector_seed)
    risk = Agent(client, RISK_PROMPT)
    remediation = Agent(client, REMEDIATION_PROMPT)
    guard = 0

    for _ in range(env.rounds):
        while True:
            seed = env.get_attack_seed()
            display.show("get_attack_seed", {}, seed)
            if "error" in seed:
                break
            reasons = seed.get("detector_now_says")
            for _ in range(env.attempts_per_seed):
                guard += 1
                if guard > max_steps:
                    break
                changes = _redteam_changes(client, mock, seed, reasons, rng, env.max_changes)
                res = env.submit_variant(seed["seed_id"], changes)
                display.show("submit_variant", {}, res)
                if "error" in res:
                    break
                if res.get("evaded"):
                    break
                reasons = res.get("rejection_reason")
        out = env.retrain_detector()
        display.show("retrain_detector", {}, out)

    alerts = env.triage_alerts()
    display.show("triage_alerts", {}, alerts)
    decisions = []
    for a in alerts.get("alerts", []):
        tech = a["technique"].split()[0]
        if mock:
            sev, resp = SEV.get(tech, "medium"), RESP.get(tech, "Investigate the flow")
        else:
            info = json.dumps({"technique": a["technique"], "predicted_class": a["predicted_class"]})
            sv = _json(risk.ask("ALERT: " + info), {}).get("severity", "medium")
            sev = sv if sv in SEVERITIES else "medium"
            resp = _json(remediation.ask("ALERT: " + info), {}).get("response", "Investigate the flow")
        decisions.append({"alert_id": a["alert_id"], "severity": sev, "response": resp})
    res = env.submit_triage(decisions)
    display.show("submit_triage", {"decisions": decisions}, res)
    env.finish("Multi-agent run: Orchestrator, Red-Team, Detection, Risk-Scoring, Remediation.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seeds", type=int, default=2, help="attack seeds per round (ignored with --golden)")
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--triage", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detector-seed", type=int, default=None)
    p.add_argument("--max-changes", type=int, default=5)
    p.add_argument("--max-factor", type=float, default=4.0)
    p.add_argument("--golden", action="store_true", help="use the fixed M5 golden set")
    p.add_argument("--mock", action="store_true", help="scripted agents, no API key (testing only)")
    p.add_argument("--out", default=None)
    p.add_argument("--no-color", action="store_true")
    a = p.parse_args()

    start = time.time()
    print("Preparing the environment (loading data, training the starting detector) ...")
    env = CoEvolutionEnv(rounds=a.rounds, seeds_per_round=a.seeds, attempts_per_seed=a.attempts,
                         triage_flows=a.triage, seed=a.seed, max_changes=a.max_changes,
                         max_factor=a.max_factor, golden=a.golden, detector_seed=a.detector_seed)
    display = Display(env, color=False if a.no_color else None)
    display.header("multi-agent", {"rounds": env.rounds, "seeds": env.seeds_per_round,
                                   "attempts": a.attempts, "max_changes": a.max_changes,
                                   "max_factor": a.max_factor}, env.history[0]["test_f1"])
    client = MockLLM(seed=env.detector_seed) if a.mock else LLM()
    if not getattr(client, "ready", False):
        raise SystemExit("The LLM is not configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in "
                         ".env, or run with --mock.")
    run_multi(env, client, display, mock=a.mock)
    usage = client.usage()
    seconds = round(time.time() - start, 1)

    rep = env.report()
    rep.update({"policy": "multi_agent", "condition": "multi_agent", "llm_calls": usage["llm_calls"],
                "usage": usage, "seconds": seconds,
                "config": {"rounds": env.rounds, "seeds_per_round": env.seeds_per_round,
                           "attempts_per_seed": a.attempts, "seed": a.seed, "golden": a.golden,
                           "detector_seed": env.detector_seed, "mock": a.mock,
                           "max_changes": a.max_changes, "max_factor": a.max_factor}})
    OUT_DIR.mkdir(exist_ok=True)
    out = Path(a.out) if a.out else OUT_DIR / "multi_agent_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    display.report(rep, "multi-agent", usage["llm_calls"], seconds, str(out))


if __name__ == "__main__":
    main()
