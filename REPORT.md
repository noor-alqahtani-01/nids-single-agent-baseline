# Multi-Agent Cyber Defense with LLM-Driven Adaptive Red-Team Co-Evolution
## M5 (Evaluation Harness) and M6 (Experimental Condition and Results)

**Team:** Noor Alqahtani, Maryam Medhat, Maryam Akram — CPCS683

---

## 1. Summary

Our system is a network intrusion detection pipeline of five agents — Detection, Risk-Scoring,
Remediation, Red-Team and Orchestrator — that run an attacker–defender co-evolution loop: the
Red-Team agent mutates attack flows to evade the detector, the successful evasions are added to the
training data, the detector is retrained, and the loop repeats. M4 delivered the single-agent
baseline (one LLM agent performing all five roles). M5 builds a fixed, automated evaluation harness
and reports the baseline with variance across three seeds. M6 implements the five-agent treatment
and compares it to the baseline on identical cases, seeds and budgets.

This report documents the harness (M5), the experimental design and the metrics (M6), and the
results available so far. The random, no-LLM reference condition is reported with real numbers; the
single-agent and multi-agent LLM conditions are produced by the same harness from one command once
a Gemini API key is supplied (Section 8).

---

## 2. Threat model (finalised for M5)

M4 found that under a tight constraint (at most 3 features changed, each within ×2 of its original
value) neither the LLM nor random mutation evaded the detector: the evasion success rate (ESR) was
0%. With no evasions, retraining has nothing to learn from, so the co-evolution loop cannot be
studied. M5 therefore finalises the threat model.

**Decision: the attacker may change at most 5 of the 9 controllable features per attempt, each within
×4 of its original value, and within the data's 1st–99th percentile.**

**Justification.** The controllable features are flow-level quantities an attacker can influence
without changing the attack's purpose: duration, packet counts, byte totals, inter-arrival times and
the initial TCP window. Features derived from these (rates, header lengths, per-packet sizes, subflow
counts) are recomputed for every variant, so a mutation cannot leave an inconsistent flow. A ×4
bound is well inside the natural variation of real attacks of the same class: measured on the
training data, the ratio between the 90th and 10th percentile of these features, within a single
attack type, is 11× for DoS GoldenEye, 32× for DDoS, 35× for DoS slowloris and over 100× for DoS
Hulk and FTP-Patator. A variant within ×4 of one real attack flow is therefore still a plausible
flow of the same attack type, not a different attack and not benign traffic. The ×2 bound used in M4
was below this natural spread, which is why it produced no evasions. The limit of 5 features keeps
each attempt a bounded, interpretable change rather than an unconstrained search.

This is the single design decision that the ESR depends on, so it is fixed once, in the harness, and
applied identically to every condition and seed.

---

## 3. M5 — Evaluation harness

### 3.1 Golden set (C5.1)

`data/golden/` holds a frozen evaluation set, selected once with a fixed seed and checked by SHA-256
before every run (`src/golden.py`):

- **Attack cases:** 21 held-out attack flows — 7 attack types × 3 cases, one case of each type per
  round (3 rounds). Ground truth is the CICIDS2017 label. A flow is admitted only if the starting
  detector of **every** evaluation seed (0, 1, 2) catches it, so each run begins from the same state:
  all 21 cases detected. This makes any later evasion attributable to the attacker, not to a case
  that was already misclassified.
- **Triage set:** 20 held-out flows (12 attacks spread over the types, 8 benign), none of them an
  attack case. Each flow carries a reference answer: its true label, a reference severity derived
  from the ATT&CK technique by scope of impact (a network-wide outage or a possible data leak is
  *critical*; a single-service denial or an account-takeover attempt is *high*), and a set of
  response keywords that a correct remediation must contain.

The manifest records the file hashes, the selection seed, the per-type counts and the reference
rubric, so the set is reproducible and tamper-evident.

### 3.2 Automated, reproducible scoring (C5.2)

`src/evaluate.py` runs a condition on the golden set for each seed with no human judgement. The
golden hashes are verified first; the detector's `random_state`, the data sample, the attacker's
stream and the LLM seed are all fixed per seed; LLM responses are requested at temperature 0. Re-running
a condition on a seed reproduces its numbers. All per-attempt records and per-condition metrics are
written to `outputs/` and `results/`.

### 3.3 End-to-end and component-level evaluation (C5.3)

The harness scores the whole pipeline and each component separately:

| Level | What is measured |
| --- | --- |
| End-to-end | ESR, number of golden cases evaded, convergence round |
| Attacker (Red-Team) | ATFE — attempts to first evasion, censored at budget+1; MER (Section 5) |
| Defender (Detection) | F1, precision, recall and false-positive rate on the held-out test set |
| Triage (Risk-Scoring + Remediation) | detection TP/FP/FN/TN on the 20 golden flows; severity accuracy and response validity against the reference rubric |
| Efficiency | LLM calls, tokens, USD, latency |

The Feedback component is evaluated by ablation: the random condition receives the same attempt
budget but no detector feedback, isolating the value of the feedback the LLM conditions use.

### 3.4 Baseline with variance across three seeds (C5.4)

Every metric is reported as mean ± standard deviation over seeds 0, 1 and 2. Section 6 gives the
random reference; the single-agent baseline is produced by the same command (Section 8).

---

## 4. M6 — Experimental condition

### 4.1 Treatment and the manipulated factor (C6.1)

The treatment is the **five-agent system** (`src/multi_agent.py`): an Orchestrator agent drives the
protocol and delegates to a Red-Team agent (mutation), a Detection agent (reads the verdict and
rejection reason and informs the Red-Team), a Risk-Scoring agent (severity) and a Remediation agent
(response). Each agent has its own short prompt and only the tools for its role.

The **manipulated design factor is the division of labour**: the single-agent baseline gives one LLM
every tool and all five roles; the treatment splits the identical tools, data, budgets, golden set
and detector across five specialised agents. Nothing else differs. The comparison therefore isolates
the effect of multi-agent decomposition, which is the project's research question. Splitting the work
is justified on the hypothesis that a narrow prompt per role reduces the reasoning load of each call
and yields more effective mutations and more accurate triage than one agent juggling all roles.

### 4.2 Controlled comparison (C6.2)

All conditions share one toolkit (`src/toolkit.py`). For a given seed the detector is retrained from
that seed, the golden cases and triage flows are identical, the attempt budget is identical (4
attempts per case), and the feedback is identical. The random condition is the no-LLM, no-feedback
reference. The three conditions run on the same three seeds. The only variable across single vs.
multi is the number of agents.

---

## 5. Metrics (operational definitions)

- **ESR (Evasion Success Rate)** = successful evasions ÷ total attempts, over all cases and rounds.
- **Cases evaded** = golden attack cases (of 21) that evaded at least once.
- **ATFE (Attempts To First Evasion)** = attempts a case needed before its first evasion; a case that
  never evaded is censored at budget+1 (= 5). Reported as the mean over cases.
- **MER (Mutation Efficiency Ratio)** = ATFE(condition) ÷ ATFE(random). MER < 1 means the condition
  reaches an evasion in fewer attempts than random search; lower is more efficient.
- **Convergence round** = the first round at which the round ESR stops improving by ≥ 1 percentage
  point (the co-evolution has stabilised).
- **Detector F1 / precision / recall / FPR** = detection quality on the held-out test set after the
  final retraining.
- **Triage recall / severity accuracy / response validity** = component scores against the golden
  reference.
- **Efficiency** = LLM calls, total tokens, cost in USD, and API latency in seconds.

---

## 6. Results

### 6.1 Random reference (real, 3 seeds)

The no-LLM reference ran on the golden set across seeds 0, 1, 2. Numbers are mean ± standard
deviation.

| Metric | Random (no LLM) |
| --- | --- |
| ESR | 0.044 ± 0.014 |
| Cases evaded (of 21) | 3.3 ± 0.9 |
| ATFE (censored at 5) | 4.52 ± 0.20 |
| Convergence round | 3 ± 0 |
| Test F1 | 0.999 ± 0.000 |
| Test precision | 0.998 ± 0.000 |
| Test recall | 0.999 ± 0.000 |
| Test FPR | 0.002 ± 0.000 |
| Triage recall (golden) | 1.00 ± 0.00 |
| LLM calls / cost | 0 / $0.00 |
| Wall time (s) | 16.4 ± 0.2 |

Under the finalised threat model the random attacker evades on about 4% of attempts and on 3 of 21
cases, so the loop is now non-trivial — unlike M4, retraining has evasions to learn from. The
detector's test-set quality is unchanged by retraining (F1 0.999), i.e. hardening against these
evasions does not cost accuracy on normal traffic.

### 6.2 LLM conditions (to be filled from the Colab run)

The single-agent and multi-agent conditions use the Gemini API and are produced by the same harness
(Section 8). On completion, `results/results_table.md` contains all three columns. The hypotheses the
table tests:

- **H1 (co-evolution):** round ESR rises then stabilises; the detector's post-retrain F1 holds.
- **H2 (LLM vs. random):** the LLM conditions reach a lower ATFE and a lower MER than random — the
  feedback is used, not just more tries.
- **H3 (multi vs. single):** the five-agent system achieves higher ESR and/or lower ATFE than the
  single agent, at a cost in LLM calls, tokens and latency that the results table quantifies.

### 6.3 Failure analysis (C6.4)

Across the random runs, 76% of failed attempts (168 of 222) were blocked by a feature the attacker
**cannot** change in our threat model. The features that most often kept a variant detected were
`Bwd Header Length`, `Total Backward Packets`, `Max Packet Length` and `Fwd Header Length` — all
properties of the server's response or of packet structure, which the attacker does not control by
mutating the forward flow. This is the same mechanism M4 identified, now quantified: the detector
leans on immutable structure, so evasion is possible only when a case's detection rests mostly on the
controllable features.

The per-type breakdown agrees. The slow-rate DoS attacks never evaded (DoS Slowhttptest 0/9, DoS
slowloris 0/9): their signature is in timing and response structure that the forward-flow features do
not reach. DDoS evaded most often (3/9), because its controllable volume features carry more of its
signature. This gives the detector's defensible weakness and tells the multi-agent Red-Team where the
headroom is — exactly the information the co-evolution loop should exploit, and a concrete target for
the H3 comparison.

---

## 7. Threats to validity and responsible use

- **Feature-level realism.** Variants are internally consistent flows but are not re-emitted as live
  traffic; we do not prove every variant is sendable end to end. This bounds the external validity of
  the ESR.
- **White-box feedback.** The rejection reason uses the detector's important features. Per M2 this is
  a training-time signal; a deployed attacker would not have it. Results describe robustness under
  training-time adaptation, not a black-box attacker.
- **Dataset.** CICIDS2017 is one dataset and one capture environment; cross-dataset generalisation is
  future work (UNSW-NB15).
- **Responsible use.** All work uses public dataset features in a simulation; no real traffic is
  generated or sent. The purpose is defensive — to harden a detector by anticipating evasion. Cost is
  bounded by the per-case attempt budget and a cached, seeded LLM; the harness reports exact tokens
  and USD so the spend is auditable.

---

## 8. Reproduction

```
# M5 golden set (committed; this only verifies the hashes)
python src/golden.py --check

# Random reference — no API key (real numbers in Section 6.1)
python src/evaluate.py --conditions random

# Full pipeline test without an API key (plumbing only, not results)
python src/evaluate.py --mock

# LLM conditions — set LLM_API_KEY in .env first
python src/evaluate.py --conditions single multi --seeds 0 1 2
# -> results/results.json and results/results_table.md (all three columns)
```

Single runs with readable per-step output: `python src/single_agent.py --golden` and
`python src/multi_agent.py --golden`.
