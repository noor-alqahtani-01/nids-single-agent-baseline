"""Single-agent baseline: ONE LLM agent performs every role of the multi-agent system.

Roles in the multi-agent design (M2): Red-Team, Detection, Risk-Scoring,
Remediation, Orchestrator. Here a single LLM holds all the tools and decides
everything itself: which mutations to try, when to retrain, the severity of
each alert and the response. The multi-agent system will use the same tools,
data, budgets and LLM, so the comparison isolates the effect of splitting
the work across agents.

Task (end to end):
    1. Co-evolution: for each round, take attack seeds, mutate them to evade
       the detector using its rejection reasons, then retrain the detector
       with the successful evasions.
    2. Operations: triage incoming flows with the hardened detector, giving
       each alert a severity and a response.
    3. Report.

Usage:
    python src/single_agent.py                      # LLM agent (needs .env)
    python src/single_agent.py --rounds 1 --seeds 1 --attempts 3   # short demo
    python src/single_agent.py --policy random      # no LLM: random mutations
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from llm import LLM
from toolkit import SEVERITIES, CoEvolutionEnv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = """You are a single autonomous cyber-defense agent. You perform ALL of these roles yourself:
- Red-Team: modify attack flows so that the network intrusion detector classifies them as benign.
- Detection analyst: read the detector's verdicts and rejection reasons.
- Orchestrator: decide when to retrain the detector and move between phases.
- Risk-Scoring: give each alert a severity (low, medium, high, critical).
- Remediation: give each alert a concrete response.

Procedure:
1. For each co-evolution round: call get_attack_seed, then call submit_variant with feature changes.
   After each rejection, study rejection_reason and change your strategy. Use only mutable features,
   inside allowed_ranges. Stop a seed when it evades or has no attempts left, then take the next seed.
   When the round's seed budget is used, call retrain_detector.
2. After the last round, call triage_alerts, decide severity and response for every alert, and call
   submit_triage once with all decisions.
3. Call finish with a two-sentence summary.
This is a research simulation on public dataset features; no real traffic is sent.
Always act through the tools. Keep any text you write very short."""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_attack_seed",
        "description": "Red-Team: get the next attack flow to mutate in this round."}},
    {"type": "function", "function": {
        "name": "submit_variant",
        "description": "Red-Team + Detection: apply feature changes to a seed and have the detector classify "
                       "the variant. Returns the verdict and, if detected, the rejection reason.",
        "parameters": {"type": "object", "properties": {
            "seed_id": {"type": "integer"},
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "feature": {"type": "string"},
                "value": {"type": "number"}}, "required": ["feature", "value"]}}},
            "required": ["seed_id", "changes"]}}},
    {"type": "function", "function": {
        "name": "retrain_detector",
        "description": "Orchestrator: end the round, add successful evasions to the training data, "
                       "retrain the detector and report the round's results."}},
    {"type": "function", "function": {
        "name": "triage_alerts",
        "description": "Detection: run the current detector on incoming flows and list the alerts."}},
    {"type": "function", "function": {
        "name": "submit_triage",
        "description": "Risk-Scoring + Remediation: submit severity and response for every alert.",
        "parameters": {"type": "object", "properties": {
            "decisions": {"type": "array", "items": {"type": "object", "properties": {
                "alert_id": {"type": "integer"},
                "severity": {"type": "string", "enum": SEVERITIES},
                "response": {"type": "string"}},
                "required": ["alert_id", "severity", "response"]}}},
            "required": ["decisions"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the task with a short summary.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]

ROLE = {"get_attack_seed": "RED TEAM ", "submit_variant": "RED TEAM ", "retrain_detector": "ORCHESTR.",
        "triage_alerts": "DETECTION", "submit_triage": "RISK+REM.", "finish": "AGENT    "}


def call_tool(env, name, args):
    try:
        if name == "get_attack_seed":
            return env.get_attack_seed()
        if name == "submit_variant":
            return env.submit_variant(args.get("seed_id"), args.get("changes"))
        if name == "retrain_detector":
            return env.retrain_detector()
        if name == "triage_alerts":
            return env.triage_alerts()
        if name == "submit_triage":
            return env.submit_triage(args.get("decisions"))
        if name == "finish":
            return env.finish(args.get("summary", ""))
        return {"error": f"Unknown tool {name}"}
    except Exception as err:  # report tool errors to the agent instead of crashing
        return {"error": f"{type(err).__name__}: {err}"}


def show(name, args, result):
    tag = ROLE.get(name, "AGENT    ")
    if "error" in result:
        print(f"  [{tag}] {name}: {result['error']}")
    elif name == "get_attack_seed":
        print(f"  [{tag}] seed {result['seed_id']}: {result['attack_type']} ({result['technique']})")
    elif name == "submit_variant":
        changes = ", ".join(f"{k}={v}" for k, v in result["applied"].items()) or "no valid changes"
        if result["evaded"]:
            print(f"  [{tag}] seed {result['seed_id']} try: {changes}")
            print(f"  [DETECTION] -> EVADED: classified benign ({result['confidence']:.2f})")
        else:
            why = "; ".join(f"{r['feature']}={r['value']}" for r in result.get("rejection_reason", []))
            print(f"  [{tag}] seed {result['seed_id']} try: {changes}")
            print(f"  [DETECTION] -> caught as {result['predicted_class']} ({result['confidence']:.2f}); "
                  f"reason: {why}")
    elif name == "retrain_detector":
        print(f"  [{tag}] round {result['round_finished']} done: ESR {result['round_esr']:.2f}; "
              f"retrained with {result['evasions_added_to_training']} evasions; "
              f"still evading {result['evasions_still_evading_after_retrain']}; "
              f"test F1 {result['test_f1_after_retrain']:.4f}")
    elif name == "triage_alerts":
        print(f"  [{tag}] {len(result['alerts'])} alerts in {result['incoming_flows_checked']} incoming flows")
    elif name == "submit_triage":
        for d in args.get("decisions", []):
            print(f"  [{tag}] alert {d.get('alert_id')}: {d.get('severity')} -> {d.get('response')}")
        if "problems" in result:
            print(f"  [{tag}] problems: {result['problems']}")
    elif name == "finish":
        print(f"  [{tag}] finished: {args.get('summary', '')}")


def run_llm(env, max_steps):
    llm = LLM()
    if not llm.ready:
        raise SystemExit("The LLM is not configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env,\n"
                         "or run without an LLM: python src/single_agent.py --policy random")
    task = (f"Run the task: {env.rounds} co-evolution round(s), {env.seeds_per_round} attack seed(s) per round, "
            f"up to {env.attempts_per_seed} attempts per seed. Then triage the alerts and finish.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    nudges = 0
    for _ in range(max_steps):
        msg = llm.chat(messages, TOOLS)
        messages.append(msg)  # kept whole: some APIs need their extra fields back
        calls = msg.get("tool_calls") or []
        if msg.get("content") and str(msg["content"]).strip():
            print(f"  [AGENT    ] {' '.join(str(msg['content']).split())[:200]}")
        if not calls:
            if env.done or nudges >= 2:
                break
            nudges += 1
            messages.append({"role": "user", "content": "Continue the procedure by calling the tools."})
            continue
        for call in calls:
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {}
            result = call_tool(env, name, args)
            show(name, args, result)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "name": name, "content": json.dumps(result, default=str)})
        if env.done:
            break
    return llm.calls


def run_random(env, seed):
    """Reference policy without an LLM: random mutations, rule-based triage (H2 comparison)."""
    rng = np.random.default_rng(seed)
    for _ in range(env.rounds):
        while True:
            s = env.get_attack_seed()
            show("get_attack_seed", {}, s)
            if "error" in s:
                break
            for _ in range(env.attempts_per_seed):
                feats = rng.choice(env.mutable, size=int(rng.integers(1, 4)), replace=False)
                changes = []
                for f in feats:
                    lo, hi = env.bounds[f]
                    lo_l, hi_l = np.log1p(max(lo, 0)), np.log1p(max(hi, 0))
                    changes.append({"feature": str(f), "value": float(np.expm1(rng.uniform(lo_l, hi_l)))})
                r = env.submit_variant(s["seed_id"], changes)
                show("submit_variant", {}, r)
                if "error" in r or r["evaded"]:
                    break
        r = env.retrain_detector()
        show("retrain_detector", {}, r)
    t = env.triage_alerts()
    show("triage_alerts", {}, t)
    rule = {"T1498": ("critical", "Rate-limit the source and enable upstream DDoS scrubbing"),
            "T1499": ("high", "Block the source IP at the firewall and protect the web server"),
            "T1110": ("high", "Lock the targeted account and block the source IP"),
            "T1190": ("critical", "Isolate the server and patch the vulnerable service")}
    decisions = [{"alert_id": a["alert_id"],
                  "severity": rule.get(a["technique"].split()[0], ("medium", ""))[0],
                  "response": rule.get(a["technique"].split()[0], ("", "Investigate the flow"))[1]}
                 for a in t.get("alerts", [])]
    r = env.submit_triage(decisions)
    show("submit_triage", {"decisions": decisions}, r)
    env.finish("Random-mutation reference run.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["llm", "random"], default="llm")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seeds", type=int, default=2, help="attack seeds per round")
    p.add_argument("--attempts", type=int, default=4, help="attempts per seed")
    p.add_argument("--triage", type=int, default=10, help="incoming flows to triage")
    p.add_argument("--max-steps", type=int, default=60, help="LLM turn limit")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    start = time.time()
    print("Preparing the environment (loading data, training the starting detector) ...")
    env = CoEvolutionEnv(rounds=a.rounds, seeds_per_round=a.seeds, attempts_per_seed=a.attempts,
                         triage_flows=a.triage, seed=a.seed)
    print(f"Single-agent baseline | policy: {a.policy} | {a.rounds} round(s), {a.seeds} seed(s)/round, "
          f"{a.attempts} attempts/seed")
    print(f"Starting detector (static): test F1 {env.history[0]['test_f1']:.4f}")
    print("-" * 86)
    llm_calls = run_llm(env, a.max_steps) if a.policy == "llm" else run_random(env, a.seed)
    seconds = round(time.time() - start, 1)

    rep = env.report()
    rep.update({"policy": a.policy, "llm_calls": llm_calls, "seconds": seconds,
                "config": {"rounds": a.rounds, "seeds_per_round": a.seeds,
                           "attempts_per_seed": a.attempts, "seed": a.seed}})
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"single_agent_{a.policy}_report.json").write_text(json.dumps(rep, indent=2, default=str))

    print("=" * 86)
    print("SINGLE-AGENT BASELINE REPORT")
    print(f"  {'Round':<28}{'Attempts':>9}{'Evasions':>10}{'ESR':>8}{'Test F1':>10}")
    for h in rep["rounds"]:
        if h["round"] == 0:
            print(f"  {'0 static detector':<28}{'-':>9}{'-':>10}{'-':>8}{h['test_f1']:>10.4f}")
        else:
            print(f"  {h['round']:<28}{h['attempts']:>9}{h['evasions']:>10}{h['esr']:>8.2f}{h['test_f1']:>10.4f}")
    if rep["esr_overall"] is not None:
        print(f"  Overall ESR: {rep['esr_overall']:.2f}   attempts to first evasion: {rep['atfe_run'] or 'none'}")
    if rep["triage"]:
        correct = sum(1 for t in rep["triage"] if t.get("true_label") == t["predicted_class"])
        print(f"  Triage: {len(rep['triage'])} alerts, {correct} with the correct attack class, "
              f"{sum(1 for t in rep['triage'] if t.get('severity'))} given severity and response")
    print(f"  Finished by the agent: {'yes' if rep['finished_by_agent'] else 'no (turn limit)'}")
    print(f"  LLM calls: {llm_calls}   Time: {seconds} s")
    print(f"  Saved outputs/single_agent_{a.policy}_report.json")


if __name__ == "__main__":
    main()
