"""M5/M6 evaluation harness: run every condition on the golden set across three
seeds and produce one results table with variance, cost, latency and a failure
breakdown.

Conditions
    random        no LLM: random feature changes + rule-based triage (H2 reference)
    single        one LLM agent, all five roles           (M4 baseline design)
    multi         five role-agents on the same toolkit    (M6 treatment)

For each condition and seed (0, 1, 2) the detector is retrained from the same
seed, the golden set is fixed, and the attempt budget is identical, so the only
difference between single and multi is how the work is split (C6.2).

Scoring (C5.3: end-to-end and component level)
    end-to-end   Evasion Success Rate (ESR), cases evaded, convergence round
    attacker     ATFE (attempts to first evasion, censored at budget+1),
                 MER = ATFE(condition) / ATFE(random)
    detector     F1, precision, recall, FPR on the held-out test set
    triage       detection TP/FP/FN/TN on the 20 golden flows; severity accuracy
                 and response validity against the reference rubric
    efficiency   LLM calls, tokens, USD, latency

Reported as mean +/- standard deviation across the three seeds (C5.4, C6.3).

Usage:
    python src/evaluate.py --mock                 # full pipeline test, no API key
    python src/evaluate.py --conditions random    # just the no-LLM reference
    python src/evaluate.py                         # all conditions (needs .env)
    python src/evaluate.py --conditions single multi --seeds 0 1 2
"""

import argparse
import json
import statistics as st
import time
from pathlib import Path

from display import Display
from golden import check as golden_check
from llm import LLM
from mock_llm import MockLLM
from toolkit import BENIGN, CoEvolutionEnv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
EVAL_SEEDS = [0, 1, 2]


class Silent:
    """A no-op display: evaluate.py runs many times, so suppress per-step output."""

    def __getattr__(self, _):
        return lambda *a, **k: None


def _score_triage(rep):
    """Component scoring of the triage phase against the golden reference."""
    truth = {t["alert_id"]: t for t in (rep.get("triage_truth") or [])}
    alerts = {a["alert_id"]: a for a in rep.get("triage", [])}
    tp = fp = fn = tn = sev_ok = sev_n = resp_ok = resp_n = 0
    for aid, t in truth.items():
        is_attack = t["true_label"] != BENIGN
        flagged = aid in alerts
        tp += is_attack and flagged
        fp += (not is_attack) and flagged
        fn += is_attack and not flagged
        tn += (not is_attack) and not flagged
        if is_attack and flagged:
            a = alerts[aid]
            if a.get("severity"):
                sev_n += 1
                sev_ok += str(a.get("severity")) == str(t.get("ref_severity"))
                resp_n += 1
                kws = str(t.get("ref_response_keywords") or "").split("|")
                resp_ok += any(k and k.lower() in str(a.get("response", "")).lower() for k in kws)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"detect_tp": tp, "detect_fp": fp, "detect_fn": fn, "detect_tn": tn,
            "triage_precision": round(prec, 3), "triage_recall": round(rec, 3),
            "severity_accuracy": round(sev_ok / sev_n, 3) if sev_n else None,
            "response_validity": round(resp_ok / resp_n, 3) if resp_n else None}


def _metrics(rep):
    rounds = [h for h in rep["rounds"] if h["round"] > 0]
    last = rep["rounds"][-1]
    budget = rep["config"]["attempts_per_seed"]
    cases = rep.get("cases", [])
    evaded_cases = sum(1 for c in cases if c["evaded"])
    # ATFE per case: attempts to first evasion, censored at budget+1 if never evaded.
    atfe = [c["attempts"] if c["evaded"] else budget + 1 for c in cases]
    atfe_mean = round(sum(atfe) / len(atfe), 3) if atfe else None
    # Convergence: first round where ESR stops improving by >=1pp (else last round).
    conv = rounds[-1]["round"] if rounds else 0
    for i in range(1, len(rounds)):
        if rounds[i - 1]["esr"] - rounds[i]["esr"] < 0.01 and rounds[i]["esr"] <= rounds[i - 1]["esr"]:
            conv = rounds[i]["round"]
            break
    usage = rep.get("usage", {})
    bad = [a for a in rep.get("attempt_log", []) if not a["evaded"]]
    mut_blockers = sum(1 for a in bad if a.get("top_reason_mutable") is False)
    m = {"esr": round(rep["esr_overall"], 4) if rep.get("esr_overall") is not None else 0.0,
         "cases_evaded": evaded_cases, "cases_total": len(cases),
         "atfe": atfe_mean, "atfe_list": atfe,
         "convergence_round": conv,
         "test_f1": last.get("test_f1"), "test_precision": last.get("test_precision"),
         "test_recall": last.get("test_recall"), "test_fpr": last.get("test_fpr"),
         "llm_calls": usage.get("llm_calls", rep.get("llm_calls", 0)),
         "total_tokens": usage.get("total_tokens", 0), "cost_usd": usage.get("cost_usd", 0.0),
         "llm_seconds": usage.get("llm_seconds", 0.0), "wall_seconds": rep.get("seconds"),
         "tool_errors": len(rep.get("tool_errors", [])),
         "failed_attempts": len(bad),
         "failed_on_immutable_feature": mut_blockers}
    m.update(_score_triage(rep))
    return m


def _run_one(condition, seed, mock, max_factor, max_changes):
    env = CoEvolutionEnv(seed=seed, detector_seed=seed, golden=True,
                         max_changes=max_changes, max_factor=max_factor, attempts_per_seed=4)
    display = Silent()
    t0 = time.time()
    if condition == "random":
        from single_agent import run_random
        global_display(display)
        run_random(env, seed)
        usage = {"llm_calls": 0, "total_tokens": 0, "cost_usd": 0.0, "llm_seconds": 0.0}
    elif condition == "single":
        from single_agent import run_llm
        global_display(display)
        client = run_llm(env, max_steps=200, mock=mock)
        usage = client.usage()
    elif condition == "multi":
        from multi_agent import run_multi
        client = MockLLM(seed=seed) if mock else LLM()
        if not getattr(client, "ready", False):
            raise SystemExit("LLM not configured. Set .env or use --mock.")
        run_multi(env, client, display, mock=mock)
        usage = client.usage()
    else:
        raise SystemExit(f"Unknown condition {condition}")
    rep = env.report()
    rep.update({"condition": condition, "usage": usage, "seconds": round(time.time() - t0, 1),
                "config": {"attempts_per_seed": 4, "seed": seed, "golden": True,
                           "detector_seed": seed, "max_changes": max_changes, "max_factor": max_factor}})
    return rep


def global_display(display):
    import single_agent
    single_agent.DISPLAY = display


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": round(st.mean(vals), 3),
            "std": round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=["random", "single", "multi"],
                   choices=["random", "single", "multi"])
    p.add_argument("--seeds", nargs="+", type=int, default=EVAL_SEEDS)
    p.add_argument("--mock", action="store_true", help="scripted agents, no API key (testing only)")
    p.add_argument("--max-changes", type=int, default=5)
    p.add_argument("--max-factor", type=float, default=4.0)
    a = p.parse_args()

    man = golden_check()
    print(f"Golden set OK: {man['attack_cases']['rows']} cases, "
          f"{man['triage_flows']['rows']} triage flows, seeds {a.seeds}")
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    per_seed, summary = {}, {}
    for cond in a.conditions:
        per_seed[cond] = []
        for seed in a.seeds:
            print(f"  {cond:7s} seed {seed} ...", end="", flush=True)
            t = time.time()
            rep = _run_one(cond, seed, a.mock, a.max_factor, a.max_changes)
            (OUT_DIR / f"{cond}_seed{seed}.json").write_text(json.dumps(rep, indent=2, default=str))
            m = _metrics(rep)
            per_seed[cond].append(m)
            print(f" ESR {m['esr']:.0%}, cases {m['cases_evaded']}/{m['cases_total']}, "
                  f"F1 {m['test_f1']}, {time.time() - t:.0f}s")
        keys = ["esr", "cases_evaded", "atfe", "convergence_round", "test_f1", "test_precision",
                "test_recall", "test_fpr", "triage_recall", "severity_accuracy", "response_validity",
                "llm_calls", "total_tokens", "cost_usd", "llm_seconds", "wall_seconds",
                "failed_on_immutable_feature", "tool_errors"]
        summary[cond] = {k: _agg([m[k] for m in per_seed[cond]]) for k in keys}

    # Multi-agent efficiency relative to the random reference (MER on ATFE).
    if "random" in summary:
        base = summary["random"]["atfe"]["mean"]
        for cond in a.conditions:
            atfe = summary[cond]["atfe"]["mean"]
            summary[cond]["mer_vs_random"] = round(atfe / base, 3) if base else None

    results = {"golden_manifest": man, "seeds": a.seeds, "mock": a.mock,
               "threat_model": {"max_features_per_attempt": a.max_changes, "max_factor": a.max_factor},
               "summary": summary, "per_seed": per_seed}
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _write_table(results)
    print(f"\nWrote results/results.json and results/results_table.md")


def _cell(agg):
    if agg["mean"] is None:
        return "-"
    return f"{agg['mean']} +/- {agg['std']}"


def _write_table(results):
    s = results["summary"]
    conds = list(s.keys())
    label = {"random": "Random (no LLM)", "single": "Single agent", "multi": "Multi-agent (5)"}
    rows = [
        ("ESR (evasion success rate)", "esr"),
        ("Cases evaded (of 21)", "cases_evaded"),
        ("ATFE (attempts to first evasion)", "atfe"),
        ("MER vs random", None),
        ("Convergence round", "convergence_round"),
        ("Test F1", "test_f1"),
        ("Test precision", "test_precision"),
        ("Test recall", "test_recall"),
        ("Test FPR", "test_fpr"),
        ("Triage recall (golden)", "triage_recall"),
        ("Severity accuracy", "severity_accuracy"),
        ("Response validity", "response_validity"),
        ("LLM calls", "llm_calls"),
        ("Total tokens", "total_tokens"),
        ("Cost (USD)", "cost_usd"),
        ("LLM latency (s)", "llm_seconds"),
        ("Wall time (s)", "wall_seconds"),
        ("Failed on immutable feature", "failed_on_immutable_feature"),
        ("Protocol errors", "tool_errors"),
    ]
    out = [f"# M5/M6 results (mean +/- sd over seeds {results['seeds']})", ""]
    tm = results["threat_model"]
    out.append(f"Threat model: at most {tm['max_features_per_attempt']} controllable features per "
               f"attempt, each within x{tm['max_factor']} of the original. "
               f"{'MOCK RUN - not real results.' if results['mock'] else ''}")
    out.append("")
    out.append("| Metric | " + " | ".join(label.get(c, c) for c in conds) + " |")
    out.append("|" + " --- |" * (len(conds) + 1))
    for name, key in rows:
        if key is None:
            cells = [str(s[c].get("mer_vs_random", "-")) for c in conds]
        else:
            cells = [_cell(s[c][key]) for c in conds]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
