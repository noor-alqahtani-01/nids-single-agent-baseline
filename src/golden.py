"""M5 golden set: fixed attack cases and triage flows with reference answers.

Both conditions (single agent, multi-agent) and the random reference are scored
on exactly these cases, for every seed. The files are frozen with SHA-256
hashes in data/golden/manifest.json; evaluate.py refuses to run if a file
changed.

Attack cases (data/golden/attack_cases.csv)
    7 attack types x 3 cases = 21 held-out attack flows from the test split,
    one case of every type per round (3 rounds). Ground truth = the CICIDS2017
    label. A case is kept only if the starting detector of EVERY evaluation
    seed (0, 1, 2) catches it, so every run starts from the same situation:
    all 21 cases detected.

Triage flows (data/golden/triage_flows.csv)
    20 held-out flows (12 attacks, 8 benign), none of them an attack case.
    Reference answers:
      - is_attack and attack_type: the CICIDS2017 label;
      - severity: from the ATT&CK technique, by scope of impact (REFERENCE);
      - response: must contain a keyword of the right response family.

Usage:
    python src/golden.py          # build (only needed once; files are committed)
    python src/golden.py --check  # verify the hashes
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "data" / "golden"
CASES_FILE = GOLDEN_DIR / "attack_cases.csv"
TRIAGE_FILE = GOLDEN_DIR / "triage_flows.csv"
MANIFEST = GOLDEN_DIR / "manifest.json"

EVAL_SEEDS = [0, 1, 2]
ATTACK_TYPES = ["DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
                "FTP-Patator", "SSH-Patator"]          # Heartbleed: too few flows
CASES_PER_TYPE = 3                                    # = number of rounds
TRIAGE_ATTACKS, TRIAGE_BENIGN = 12, 8
SELECT_SEED = 2026

# Reference triage answers per ATT&CK technique.
# Severity follows scope of impact: a whole-network outage (DDoS) or a possible
# data leak (exploit) is critical; one service down (DoS) or account takeover
# risk (brute force) is high. A response matches if it names an action of the
# right family (any keyword, case-insensitive).
REFERENCE = {
    "T1498": ("critical", ["rate-limit", "rate limit", "scrub", "upstream", "isp", "cdn",
                           "mitigat", "blackhole", "null-route", "anti-ddos", "ddos protection"]),
    "T1499": ("high", ["block", "firewall", "rate-limit", "rate limit", "timeout", "waf",
                       "connection limit", "throttl", "drop"]),
    "T1110": ("high", ["lock", "block", "ban", "fail2ban", "mfa", "2fa", "password",
                       "disable", "credential"]),
    "T1190": ("critical", ["patch", "isolate", "update", "upgrade", "revoke"]),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    from sklearn.ensemble import RandomForestClassifier

    from attack_map import lookup_attack
    from toolkit import BENIGN, DETECTOR_PARAMS, WORK_DIR

    train = pd.read_csv(WORK_DIR / "train.csv.gz")
    test = pd.read_csv(WORK_DIR / "test.csv.gz")
    features = [c for c in train.columns if c not in ("attack_type", "label")]
    caught_by_all = np.ones(len(test), dtype=bool)
    for s in EVAL_SEEDS:
        model = RandomForestClassifier(**{**DETECTOR_PARAMS, "random_state": s})
        model.fit(train[features], train["attack_type"])
        pred = model.predict(test[features])
        caught_by_all &= (pred == test["attack_type"].to_numpy()) | (
            (pred != BENIGN) & (test["attack_type"].to_numpy() != BENIGN))
        print(f"detector seed {s}: trained")

    rng = np.random.default_rng(SELECT_SEED)
    cases = []
    for t in ATTACK_TYPES:
        pool = np.flatnonzero((test["attack_type"].to_numpy() == t) & caught_by_all)
        pick = rng.choice(pool, size=CASES_PER_TYPE, replace=False)
        for r, i in enumerate(pick, start=1):
            cases.append({"case_id": None, "round": r, "test_index": int(i)})
    cases.sort(key=lambda c: (c["round"], ATTACK_TYPES.index(test.loc[c["test_index"], "attack_type"])))
    for k, c in enumerate(cases, start=1):
        c["case_id"] = k
    used = {c["test_index"] for c in cases}
    case_df = pd.concat([pd.DataFrame(cases), test.loc[[c["test_index"] for c in cases]]
                         .reset_index(drop=True)], axis=1)

    # Triage: 12 attacks (spread over the types) + 8 benign, all caught/cleared by every seed.
    tri = []
    attack_idx = np.flatnonzero((test["attack_type"].to_numpy() != BENIGN)
                                & (test["attack_type"].to_numpy() != "Heartbleed") & caught_by_all)
    attack_idx = [i for i in attack_idx if i not in used]
    by_type = {t: [i for i in attack_idx if test.loc[i, "attack_type"] == t] for t in ATTACK_TYPES}
    order = [t for _ in range(2) for t in ATTACK_TYPES][:TRIAGE_ATTACKS]
    for t in order:
        i = int(rng.choice([j for j in by_type[t] if j not in tri]))
        tri.append(i)
    benign_idx = np.flatnonzero((test["attack_type"].to_numpy() == BENIGN) & caught_by_all)
    tri += [int(i) for i in rng.choice(benign_idx, size=TRIAGE_BENIGN, replace=False)]
    rng.shuffle(tri)
    tri_df = test.loc[tri].reset_index(drop=True)
    tri_df.insert(0, "test_index", tri)
    tri_df.insert(0, "flow_id", range(1, len(tri_df) + 1))
    sev, kw = [], []
    for t in tri_df["attack_type"]:
        tech = lookup_attack(t)["technique_id"]
        s, k = REFERENCE.get(tech, ("none", []))
        sev.append(s)
        kw.append("|".join(k))
    tri_df["ref_severity"] = sev
    tri_df["ref_response_keywords"] = kw

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(CASES_FILE, index=False)
    tri_df.to_csv(TRIAGE_FILE, index=False)
    manifest = {
        "description": "M5 golden set (see src/golden.py for selection rules)",
        "source": "data/working/test.csv.gz (held-out CICIDS2017 flows)",
        "selection_seed": SELECT_SEED, "eval_seeds": EVAL_SEEDS,
        "attack_cases": {"file": CASES_FILE.name, "rows": len(case_df), "sha256": sha256(CASES_FILE),
                         "per_type": case_df["attack_type"].value_counts().to_dict(),
                         "rounds": int(case_df["round"].max())},
        "triage_flows": {"file": TRIAGE_FILE.name, "rows": len(tri_df), "sha256": sha256(TRIAGE_FILE),
                         "labels": tri_df["attack_type"].value_counts().to_dict()},
        "reference_rubric": {k: {"severity": v[0], "response_keywords": v[1]} for k, v in REFERENCE.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"{len(case_df)} attack cases, {len(tri_df)} triage flows -> {GOLDEN_DIR.relative_to(ROOT)}")


def check() -> dict:
    if not MANIFEST.exists():
        raise SystemExit("data/golden/manifest.json not found. Run: python src/golden.py")
    m = json.loads(MANIFEST.read_text())
    for key, path in (("attack_cases", CASES_FILE), ("triage_flows", TRIAGE_FILE)):
        if sha256(path) != m[key]["sha256"]:
            raise SystemExit(f"{path.name} does not match its SHA-256 in manifest.json: "
                             "the golden set was changed.")
    return m


def load_cases() -> dict:
    """Golden attack cases grouped by round: {round: [flow dict, ...]}."""
    check()
    df = pd.read_csv(CASES_FILE)
    out = {}
    for row in df.to_dict(orient="records"):
        out.setdefault(int(row["round"]), []).append(row)
    return out


def load_triage() -> pd.DataFrame:
    check()
    return pd.read_csv(TRIAGE_FILE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    if p.parse_args().check:
        print("golden set OK:", json.dumps(check()["attack_cases"]["per_type"]))
    else:
        build()
