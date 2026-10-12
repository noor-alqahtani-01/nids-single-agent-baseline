"""Readable terminal output for the single-agent baseline.

Turns tool results into short, labelled lines with units (seconds, KB, packets),
shows each change as old -> new, explains why a variant was caught, and ends
with a plain-language summary of the run.
"""

import sys

TIME_FEATURES = {"Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Fwd IAT Mean"}  # microseconds
BYTE_FEATURES = {"Total Length of Fwd Packets", "Total Length of Bwd Packets", "Init_Win_bytes_forward",
                 "Fwd Header Length", "Fwd Header Length.1", "Bwd Header Length",
                 "Subflow Fwd Bytes", "Subflow Bwd Bytes", "Max Packet Length",
                 "Bwd Packet Length Max", "Fwd Packet Length Max", "Avg Bwd Segment Size",
                 "Avg Fwd Segment Size", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                 "Average Packet Size", "Packet Length Mean", "min_seg_size_forward"}
SHORT = {"Total Fwd Packets": "packets sent", "Total Backward Packets": "packets received",
         "Total Length of Fwd Packets": "bytes sent", "Total Length of Bwd Packets": "bytes received",
         "Flow Duration": "duration", "Flow IAT Mean": "mean gap", "Flow IAT Max": "max gap",
         "Fwd IAT Mean": "mean send gap", "Init_Win_bytes_forward": "TCP window",
         "Fwd Header Length": "header bytes sent", "Fwd Header Length.1": "header bytes sent",
         "Bwd Header Length": "header bytes received", "Subflow Fwd Bytes": "bytes sent (subflow)",
         "Subflow Bwd Bytes": "bytes received (subflow)", "Subflow Fwd Packets": "packets sent (subflow)",
         "Subflow Bwd Packets": "packets received (subflow)", "Max Packet Length": "largest packet",
         "Bwd Packet Length Max": "largest packet received", "Fwd Packet Length Max": "largest packet sent",
         "Avg Bwd Segment Size": "avg packet received", "Avg Fwd Segment Size": "avg packet sent",
         "min_seg_size_forward": "min segment sent", "Destination Port": "destination port"}
WIDTH = 86


def fmt(feature, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature in TIME_FEATURES:
        if abs(v) >= 1e6:
            return f"{v / 1e6:.1f} s"
        if abs(v) >= 1e3:
            return f"{v / 1e3:.1f} ms"
        return f"{v:.0f} µs"
    if feature in BYTE_FEATURES:
        if abs(v) >= 1e6:
            return f"{v / 1e6:.1f} MB"
        if abs(v) >= 1e4:
            return f"{v / 1e3:.1f} KB"
        return f"{v:.0f} B"
    return f"{v:.0f}" if v.is_integer() or abs(v) >= 100 else f"{v:.2f}"


def name(feature):
    return SHORT.get(feature, feature)


class Display:
    def __init__(self, env, color=None):
        self.env = env
        self.color = sys.stdout.isatty() if color is None else color
        self.current_round = 0
        self.attempt_no = {}

    # ------------------------------------------------------------ styling
    def c(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def red(self, t):
        return self.c(t, "31")

    def green(self, t):
        return self.c(t, "32")

    def yellow(self, t):
        return self.c(t, "33")

    def bold(self, t):
        return self.c(t, "1")

    def dim(self, t):
        return self.c(t, "2")

    def banner(self, title, subtitle=""):
        print()
        print(self.bold("=" * WIDTH))
        print(self.bold(f" {title}"))
        if subtitle:
            print(self.dim(f" {subtitle}"))
        print(self.bold("=" * WIDTH))

    def header(self, policy, cfg, f1):
        self.banner("SINGLE-AGENT BASELINE",
                    "One agent plays all five roles: Red-Team, Detection, Orchestrator, Risk-Scoring, Remediation")
        who = "Gemini LLM agent" if policy == "llm" else "random mutations (no LLM)"
        print(f" Policy       : {who}")
        print(f" Task         : {cfg['rounds']} round(s) x {cfg['seeds']} attack(s) x up to {cfg['attempts']} tries each")
        print(f" Attack rules : change at most {cfg['max_changes']} features per try, "
              f"each within x{cfg['max_factor']:g} of its original value")
        print(f" Detector     : Random Forest, test F1 {f1:.4f} before any attack")

    # ------------------------------------------------------------ events
    def show(self, tool, args, result):
        if "error" in result:
            msg = result["error"]
            if "budget" in msg or "rounds are finished" in msg:
                self.note(f"tool: {msg}")      # normal hand-over between phases
            else:
                print(self.yellow(f"   ! tool refused: {msg}"))
            return
        getattr(self, f"_{tool}", lambda *a: None)(args, result)

    def _get_attack_seed(self, args, r):
        seed = self.env.seeds[r["seed_id"]]
        if seed["round"] != self.current_round:
            self.current_round = seed["round"]
            self.banner(f"ROUND {self.current_round} of {self.env.rounds}: attack and defend",
                        "RED TEAM tries to make an attack look benign; DETECTION judges each try")
        self.attempt_no[r["seed_id"]] = 0
        print()
        print(self.bold(f" Attack {r['seed_id']}: {r['attack_type']}  ({r['technique']})"))

    def _submit_variant(self, args, r):
        sid = r["seed_id"]
        self.attempt_no[sid] = self.attempt_no.get(sid, 0) + 1
        orig = self.env.seeds[sid]["flow"]
        changes = [f"{name(f)} {fmt(f, orig[f])} -> {fmt(f, v)}" for f, v in r["applied"].items()]
        print(f"   [RED TEAM]  try {self.attempt_no[sid]}: " + ("; ".join(changes) or "no valid change"))
        if r["evaded"]:
            print("   [DETECTOR]  " + self.green(f"EVADED: classified as benign (confidence {r['confidence']:.2f})"))
            return
        print("   [DETECTOR]  " + self.red(f"CAUGHT as {r['predicted_class']} (confidence {r['confidence']:.2f})"))
        reasons, seen = [], set()
        for x in r.get("rejection_reason", []):
            if name(x["feature"]) in seen or len(reasons) == 2:
                continue  # e.g. the two copies of the header-length column
            seen.add(name(x["feature"]))
            lo, hi = x["typical_benign_range"]
            fixed = "" if x["attacker_can_change"] else ", attacker cannot change it"
            reasons.append(f"{name(x['feature'])} is {fmt(x['feature'], x['value'])} "
                           f"(benign: {fmt(x['feature'], lo)} to {fmt(x['feature'], hi)}{fixed})")
        if reasons:
            print(self.dim("               why: " + "; ".join(reasons)))

    def _retrain_detector(self, args, r):
        h = next(x for x in self.env.history if x["round"] == r["round_finished"])
        print()
        print(self.bold(f" [ORCHESTRATOR] Round {r['round_finished']} result: "
                        f"{h['evasions']} of {h['attempts']} tries evaded (ESR {r['round_esr']:.0%})"))
        print(f"   Detector retrained with {r['evasions_added_to_training']} evasion(s); "
              f"{r['evasions_still_evading_after_retrain']} still evade after retraining; "
              f"test F1 now {r['test_f1_after_retrain']:.4f}")

    def _triage_alerts(self, args, r):
        self.banner("OPERATIONS: triage new traffic",
                    "DETECTION finds alerts; RISK-SCORING sets severity; REMEDIATION picks a response")
        print(f" {len(r['alerts'])} alerts in {r['incoming_flows_checked']} incoming flows")

    def _submit_triage(self, args, r):
        alerts = {a["alert_id"]: a for a in (self.env.triage_alerts_list or [])}
        print()
        print(f"   {'Alert':<6}{'Detected as':<18}{'Technique':<10}{'Severity':<10}Response")
        print(self.dim("   " + "-" * (WIDTH - 3)))
        for d in args.get("decisions", []):
            a = alerts.get(d.get("alert_id"), {})
            sev = str(d.get("severity", "")).upper()
            sev_txt = self.red(f"{sev:<10}") if sev in ("HIGH", "CRITICAL") else f"{sev:<10}"
            tech = str(a.get("technique", "")).split(" ")[0]
            print(f"   {str(d.get('alert_id')):<6}{str(a.get('predicted_class', '')):<18}{tech:<10}"
                  f"{sev_txt}{d.get('response', '')}")
        if "problems" in r:
            print(self.yellow(f"   ! problems: {r['problems']}"))

    def _finish(self, args, r):
        print()
        print(f" [AGENT] {args.get('summary', '')}")

    def note(self, text):
        print(self.dim(f"   ({text})"))

    # ------------------------------------------------------------ report
    def report(self, rep, policy, llm_calls, seconds, path):
        self.banner("REPORT")
        print(f"   {'Round':<22}{'Tries':>7}{'Evaded':>8}{'ESR':>8}{'Test F1':>10}")
        print(self.dim("   " + "-" * 55))
        for h in rep["rounds"]:
            if h["round"] == 0:
                print(f"   {'0 (static detector)':<22}{'-':>7}{'-':>8}{'-':>8}{h['test_f1']:>10.4f}")
            else:
                print(f"   {str(h['round']):<22}{h['attempts']:>7}{h['evasions']:>8}"
                      f"{h['esr']:>8.0%}{h['test_f1']:>10.4f}")
        print()
        tries, evaded = rep["attempts_total"], rep["evasions_total"]
        f0 = rep["rounds"][0]["test_f1"]
        f1 = rep["rounds"][-1]["test_f1"]
        print(self.bold(" What this means"))
        if tries:
            if evaded:
                print(f"   - The attacker fooled the detector {evaded} time(s) in {tries} tries "
                      f"(ESR {evaded / tries:.0%}); first success after {rep['atfe_run']} tries.")
            else:
                print(f"   - The detector caught all {tries} attack variants (ESR 0%).")
        print(f"   - Detection quality on normal test traffic: F1 {f0:.4f} before, {f1:.4f} after.")
        if rep["triage"]:
            done = sum(1 for t in rep["triage"] if t.get("severity"))
            right = sum(1 for t in rep["triage"] if t.get("true_label") == t["predicted_class"])
            print(f"   - {len(rep['triage'])} new alerts: {right} with the correct attack type, "
                  f"{done} given a severity and a response.")
        who = "the agent" if policy == "llm" else "the random policy"
        status = "completed every step" if rep["finished_by_agent"] else "stopped early (see the messages above)"
        print(f"   - {who.capitalize()} {status}: {llm_calls} LLM call(s), {seconds} s.")
        print(self.dim(f"\n Full report saved to {path}"))
