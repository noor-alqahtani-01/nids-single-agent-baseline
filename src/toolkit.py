"""Shared tools for the co-evolution task.

The single-agent baseline (single_agent.py) and the multi-agent system (later)
must use exactly these tools, data and budgets, so that the only difference
between the two conditions is how the work is split across agents.

Tools, grouped by the role that uses them in the multi-agent design:
    Red-Team      get_attack_seed, submit_variant (mutation)
    Detection     submit_variant (classification + rejection reason)
    Orchestrator  retrain_detector (retraining + evaluation)
    Risk-Scoring  triage_alerts, submit_triage (severity)
    Remediation   submit_triage (response)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from attack_map import lookup_attack

SEED = 42
BENIGN = "BENIGN"
ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "working"
SAMPLE_FLOWS = ROOT / "data" / "sample_flows.csv"

# Features the attacker controls directly (feature-level mutations, M2 scope).
MUTABLE = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Fwd IAT Mean",
    "Init_Win_bytes_forward",
]
INTEGER = {"Flow Duration", "Total Fwd Packets", "Total Backward Packets",
           "Total Length of Fwd Packets", "Total Length of Bwd Packets",
           "Init_Win_bytes_forward"}
SEVERITIES = ["low", "medium", "high", "critical"]
DETECTOR_PARAMS = dict(n_estimators=100, max_depth=25, min_samples_leaf=2,
                       class_weight="balanced", random_state=SEED, n_jobs=-1)


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else round(v, 3)


class CoEvolutionEnv:
    def __init__(self, rounds=2, seeds_per_round=2, attempts_per_seed=4,
                 evasion_weight=50, triage_flows=10, seed=SEED):
        for name in ("train", "test"):
            if not (WORK_DIR / f"{name}.csv.gz").exists():
                raise SystemExit(f"data/working/{name}.csv.gz not found. "
                                 "Run: python src/make_working_set.py")
        self.train = pd.read_csv(WORK_DIR / "train.csv.gz")
        self.test = pd.read_csv(WORK_DIR / "test.csv.gz")
        self.features = [c for c in self.train.columns if c not in ("attack_type", "label")]
        self.mutable = [f for f in MUTABLE if f in self.features]
        self.rounds, self.seeds_per_round = rounds, seeds_per_round
        self.attempts_per_seed, self.evasion_weight = attempts_per_seed, evasion_weight
        self.triage_n = triage_flows
        self.rng = np.random.default_rng(seed)

        X = self.train[self.features]
        self.bounds = {f: (float(X[f].quantile(0.01)), float(X[f].quantile(0.99)))
                       for f in self.mutable}
        benign = np.log1p(X[self.train["attack_type"] == BENIGN].clip(lower=0))
        self.benign_q = {"p25": benign.quantile(0.25), "p50": benign.median(),
                         "p75": benign.quantile(0.75)}
        self.benign_raw = X[self.train["attack_type"] == BENIGN]

        self.detector = self._fit([])
        self.top_features = [self.features[i] for i in
                             np.argsort(self.detector.feature_importances_)[::-1][:20]]
        f1_0 = self._test_f1()

        # Attack seeds: held-out attack flows the starting detector catches.
        attacks = self.test[self.test["attack_type"] != BENIGN]
        pred = self.detector.predict(attacks[self.features])
        caught = attacks[pred != BENIGN]
        self.seed_pool = caught.sample(frac=1, random_state=seed).to_dict(orient="records")

        self.round = 1
        self.history = [{"round": 0, "label": "static (no adaptation)", "test_f1": f1_0}]
        self.seeds, self.variants, self.evasions = {}, {}, []
        self.round_attempts, self.round_evasions, self.round_seeds = 0, 0, 0
        self.total_attempts = 0
        self.first_evasion_attempt = None
        self.triage_alerts_list, self.triage_decisions = None, None
        self.done, self.summary = False, None
        self.log = []

    # ---------------------------------------------------------------- helpers
    def _fit(self, evasions):
        X = self.train[self.features]
        y = self.train["attack_type"].to_numpy(dtype=object)
        w = np.ones(len(X))
        if evasions:
            ev = pd.DataFrame([e["flow"] for e in evasions])[self.features]
            X = pd.concat([X, ev], ignore_index=True)
            y = np.concatenate([y, np.array([e["attack_type"] for e in evasions], dtype=object)])
            w = np.concatenate([w, np.full(len(ev), float(self.evasion_weight))])
        model = RandomForestClassifier(**DETECTOR_PARAMS)
        model.fit(X, y, sample_weight=w)
        return model

    def _test_f1(self):
        pred = self.detector.predict(self.test[self.features])
        return round(float(f1_score(self.test["label"], (pred != BENIGN).astype(int))), 4)

    def _classify(self, flow):
        x = pd.DataFrame([[flow[f] for f in self.features]], columns=self.features)
        proba = self.detector.predict_proba(x)[0]
        best = int(proba.argmax())
        return str(self.detector.classes_[best]), round(float(proba[best]), 3)

    def _rejection_reason(self, flow):
        """White-box feedback: important features furthest from typical benign traffic."""
        scored = []
        for f in self.top_features:
            v = np.log1p(max(float(flow[f]), 0.0))
            iqr = float(self.benign_q["p75"][f] - self.benign_q["p25"][f]) or 1.0
            scored.append((abs(v - float(self.benign_q["p50"][f])) / iqr, f))
        reasons = []
        for _, f in sorted(scored, reverse=True)[:3]:
            lo, hi = self.benign_raw[f].quantile(0.25), self.benign_raw[f].quantile(0.75)
            reasons.append({"feature": f, "value": _num(flow[f]),
                            "typical_benign_range": [_num(lo), _num(hi)],
                            "attacker_can_change": f in self.mutable})
        return reasons

    @staticmethod
    def _recompute(flow):
        """Keep rate features consistent with the counts and duration."""
        seconds = max(float(flow.get("Flow Duration", 1)), 1.0) / 1e6
        fwd, bwd = float(flow.get("Total Fwd Packets", 0)), float(flow.get("Total Backward Packets", 0))
        lf, lb = float(flow.get("Total Length of Fwd Packets", 0)), float(flow.get("Total Length of Bwd Packets", 0))
        for name, value in (("Flow Packets/s", (fwd + bwd) / seconds),
                            ("Flow Bytes/s", (lf + lb) / seconds),
                            ("Fwd Packets/s", fwd / seconds),
                            ("Bwd Packets/s", bwd / seconds)):
            if name in flow:
                flow[name] = value

    def _visible(self, flow):
        return {f: _num(flow[f]) for f in self.mutable}

    # ------------------------------------------------------------------ tools
    def get_attack_seed(self):
        if self.done or self.round > self.rounds:
            return {"error": "All co-evolution rounds are finished. Call triage_alerts next."}
        if self.round_seeds >= self.seeds_per_round:
            return {"error": f"Seed budget for round {self.round} is used up. Call retrain_detector."}
        if not self.seed_pool:
            return {"error": "No attack seeds left. Call retrain_detector."}
        flow = self.seed_pool.pop()
        seed_id = len(self.seeds) + 1
        self.seeds[seed_id] = {"flow": flow, "attempts": 0, "evaded": False,
                               "attack_type": flow["attack_type"], "round": self.round}
        self.round_seeds += 1
        tech = lookup_attack(flow["attack_type"])
        return {"seed_id": seed_id, "attack_type": flow["attack_type"],
                "technique": f"{tech['technique_id']} {tech['technique_name']}",
                "mutable_features": self._visible(flow),
                "allowed_ranges": {f: [_num(lo), _num(hi)] for f, (lo, hi) in self.bounds.items()},
                "attempts_allowed": self.attempts_per_seed,
                "detector_now_says": self._rejection_reason(flow)}

    def submit_variant(self, seed_id, changes):
        seed = self.seeds.get(int(seed_id)) if str(seed_id).isdigit() else None
        if seed is None:
            return {"error": f"Unknown seed_id {seed_id}. Call get_attack_seed first."}
        if seed["round"] != self.round:
            return {"error": "This seed belongs to an earlier round. Call get_attack_seed."}
        if seed["evaded"]:
            return {"error": "This seed already evaded. Call get_attack_seed or retrain_detector."}
        if seed["attempts"] >= self.attempts_per_seed:
            return {"error": "No attempts left for this seed. Call get_attack_seed or retrain_detector."}
        flow = dict(seed["flow"])
        applied, ignored, clipped = {}, [], []
        for ch in changes or []:
            f, v = ch.get("feature"), ch.get("value")
            if f not in self.mutable or not isinstance(v, (int, float)):
                ignored.append(f)
                continue
            lo, hi = self.bounds[f]
            nv = min(max(float(v), lo), hi)
            if nv != float(v):
                clipped.append(f)
            flow[f] = int(round(nv)) if f in INTEGER else nv
            applied[f] = _num(flow[f])
        self._recompute(flow)
        predicted, confidence = self._classify(flow)
        evaded = predicted == BENIGN
        seed["attempts"] += 1
        self.round_attempts += 1
        self.total_attempts += 1
        variant_id = len(self.variants) + 1
        self.variants[variant_id] = flow
        result = {"variant_id": variant_id, "seed_id": int(seed_id), "applied": applied,
                  "detector_verdict": "benign" if evaded else "attack",
                  "predicted_class": predicted, "confidence": confidence,
                  "evaded": evaded,
                  "attempts_left_for_seed": self.attempts_per_seed - seed["attempts"]}
        if ignored:
            result["ignored_changes"] = ignored
        if clipped:
            result["clipped_to_allowed_range"] = clipped
        if evaded:
            seed["evaded"] = True
            self.round_evasions += 1
            self.evasions.append({"flow": flow, "attack_type": seed["attack_type"],
                                  "seed_id": int(seed_id), "round": self.round,
                                  "attempt": seed["attempts"]})
            if self.first_evasion_attempt is None:
                self.first_evasion_attempt = self.total_attempts
        else:
            result["rejection_reason"] = self._rejection_reason(flow)
        return result

    def retrain_detector(self):
        if self.round > self.rounds:
            return {"error": "All rounds are finished. Call triage_alerts."}
        if self.round_attempts == 0:
            return {"error": "No attack attempts in this round yet. Call get_attack_seed first."}
        start = time.time()
        esr = round(self.round_evasions / self.round_attempts, 4)
        self.detector = self._fit(self.evasions)
        still = 0
        for e in self.evasions:
            if self._classify(e["flow"])[0] == BENIGN:
                still += 1
        f1 = self._test_f1()
        self.history.append({"round": self.round, "attempts": self.round_attempts,
                             "evasions": self.round_evasions, "esr": esr, "test_f1": f1,
                             "evasions_total": len(self.evasions),
                             "still_evading_after_retrain": still,
                             "retrain_seconds": round(time.time() - start, 1)})
        finished = self.round
        self.round += 1
        self.round_attempts = self.round_evasions = self.round_seeds = 0
        nxt = (f"Round {self.round} starts: call get_attack_seed." if self.round <= self.rounds
               else "All rounds finished: call triage_alerts.")
        return {"round_finished": finished, "round_esr": esr,
                "evasions_added_to_training": len(self.evasions),
                "evasions_still_evading_after_retrain": still,
                "test_f1_after_retrain": f1, "next": nxt}

    def triage_alerts(self):
        if self.round <= self.rounds:
            return {"error": f"Co-evolution round {self.round} is not finished. Retrain first."}
        flows = pd.read_csv(SAMPLE_FLOWS).head(self.triage_n)
        alerts = []
        for i, row in enumerate(flows.to_dict(orient="records"), start=1):
            predicted, confidence = self._classify(row)
            if predicted == BENIGN:
                continue
            tech = lookup_attack(predicted)
            alerts.append({"alert_id": i, "predicted_class": predicted, "confidence": confidence,
                           "technique": f"{tech['technique_id']} {tech['technique_name']}",
                           "destination_port": _num(row.get("Destination Port", -1)),
                           "flow_packets_per_s": _num(row.get("Flow Packets/s", 0)),
                           "true_label": row.get("attack_type")})
        self.triage_alerts_list = alerts
        visible = [{k: v for k, v in a.items() if k != "true_label"} for a in alerts]
        return {"incoming_flows_checked": len(flows), "alerts": visible,
                "next": "Give each alert a severity (low, medium, high, critical) and a "
                        "response, then call submit_triage."}

    def submit_triage(self, decisions):
        if self.triage_alerts_list is None:
            return {"error": "Call triage_alerts first."}
        ids = {a["alert_id"] for a in self.triage_alerts_list}
        clean, problems = [], []
        for d in decisions or []:
            aid, sev = d.get("alert_id"), str(d.get("severity", "")).lower()
            if aid not in ids or sev not in SEVERITIES:
                problems.append(d)
                continue
            clean.append({"alert_id": aid, "severity": sev, "response": str(d.get("response", ""))})
        missing = sorted(ids - {d["alert_id"] for d in clean})
        self.triage_decisions = clean
        out = {"accepted": len(clean)}
        if problems or missing:
            out["problems"] = {"invalid": problems, "missing_alert_ids": missing}
        else:
            out["next"] = "Call finish with a short summary."
        return out

    def finish(self, summary=""):
        self.done = True
        self.summary = summary
        return {"status": "finished"}

    # --------------------------------------------------------------- reporting
    def report(self):
        rounds = [h for h in self.history if h["round"] > 0]
        attempts = sum(h["attempts"] for h in rounds)
        evasions = sum(h["evasions"] for h in rounds)
        per_seed = [s["attempts"] for s in self.seeds.values() if s["evaded"]]
        return {
            "rounds": self.history,
            "attempts_total": attempts,
            "evasions_total": evasions,
            "esr_overall": round(evasions / attempts, 4) if attempts else None,
            "atfe_run": self.first_evasion_attempt,
            "atfe_mean_per_evading_seed": round(float(np.mean(per_seed)), 2) if per_seed else None,
            "seeds_used": len(self.seeds),
            "triage": [{**a, **next((d for d in (self.triage_decisions or [])
                                     if d["alert_id"] == a["alert_id"]), {})}
                       for a in (self.triage_alerts_list or [])],
            "finished_by_agent": self.done,
            "agent_summary": self.summary,
        }
