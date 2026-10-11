"""Run the single-agent baseline end to end on a file of network flows.

Task: triage every flow in the input file and produce an alert report.

Usage:
    python src/run_demo.py --input data/sample_flows.csv
    python src/run_demo.py --input data/sample_flows.csv --no-llm
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from agent import NIDSAgent

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "sample_flows.csv"))
    parser.add_argument("--no-llm", action="store_true",
                        help="Use template reasons instead of calling the LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N flows.")
    args = parser.parse_args()

    flows = pd.read_csv(args.input)
    if args.limit:
        flows = flows.head(args.limit)
    # Ground truth is kept away from the agent and used only for scoring.
    truth = flows["label"].tolist() if "label" in flows.columns else None
    flows = flows.drop(columns=[c for c in ("label", "attack_type") if c in flows.columns])

    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / "alerts.jsonl"
    log_path.write_text("")
    agent = NIDSAgent(use_llm=not args.no_llm, log_path=log_path)

    total = len(flows)
    print(f"Single-agent NIDS baseline: triaging {total} flows from {Path(args.input).name}")
    if agent.llm_note:
        print(f"Note: {agent.llm_note}")
    print("-" * 78)

    records, start = [], time.time()
    for i, flow in enumerate(flows.to_dict(orient="records"), start=1):
        note_before = agent.llm_note
        r = agent.handle(i, flow)
        records.append(r)
        if agent.llm_note != note_before:
            print(f"Note: {agent.llm_note}")
        tag = "ALERT " if r["verdict"] == "attack" else "ok    "
        technique = r["technique_id"] or "-"
        print(f"[{i:>3}/{total}] {tag} {r['attack_type']:<17} {technique:<6} conf {r['confidence']:.2f}")
        print(f"          {r['reason']}")
    seconds = time.time() - start

    alerts = [r for r in records if r["verdict"] == "attack"]
    by_technique = Counter(f"{r['technique_id']} {r['technique_name']}" for r in alerts)
    report = {
        "input": Path(args.input).name,
        "flows": total,
        "alerts": len(alerts),
        "alerts_by_technique": dict(by_technique),
        "reasons_from_llm": sum(r["reason_source"] == "llm" for r in records),
        "seconds": round(seconds, 1),
    }
    if truth is not None:
        pred = [int(r["verdict"] == "attack") for r in records]
        tp = sum(p == 1 and t == 1 for p, t in zip(pred, truth))
        fp = sum(p == 1 and t == 0 for p, t in zip(pred, truth))
        fn = sum(p == 0 and t == 1 for p, t in zip(pred, truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report["metrics"] = {"precision": round(precision, 4), "recall": round(recall, 4),
                             "f1": round(f1, 4), "true_attacks": sum(truth),
                             "missed_attacks": fn, "false_alarms": fp}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print("ALERT REPORT")
    print(f"  Flows processed : {total}")
    print(f"  Alerts raised   : {len(alerts)}")
    for technique, n in by_technique.most_common():
        print(f"    {technique:<42} {n}")
    if "metrics" in report:
        m = report["metrics"]
        print(f"  Against true labels: precision {m['precision']:.4f}  recall {m['recall']:.4f}  F1 {m['f1']:.4f}")
        print(f"    missed attacks: {m['missed_attacks']}   false alarms: {m['false_alarms']}")
    print(f"  Reasons written by LLM: {report['reasons_from_llm']} of {total}")
    print(f"  Time: {seconds:.1f} s")
    print("  Saved outputs/alerts.jsonl and outputs/report.json")


if __name__ == "__main__":
    main()
