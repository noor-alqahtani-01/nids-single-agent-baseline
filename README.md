# NIDS Single-Agent Baseline

M4 deliverable for the CPCS683 group project *Multi-Agent Cyber Defense System with LLM-Driven Adaptive Red-Team Co-Evolution*.

**Team:** Noor Alqahtani, Maryam Medhat, Maryam Akram

## What this is

Our full system (M2) is a multi-agent network intrusion detection system with five agents: Detection, Risk-Scoring, Remediation, Red-Team and Orchestrator. They run an attacker–defender co-evolution loop:

> Attack → Detection → Rejection explanation → Attack mutation → Retraining → Repeat

This repository is the **single-agent baseline**: **one LLM agent performs all five roles itself**, with the same tools, data and budgets the multi-agent system will use. Comparing the two isolates the effect of splitting the work across agents.

| Role in the multi-agent design | What the single agent does | Tool it calls |
| --- | --- | --- |
| Red-Team | Takes an attack flow and decides which features to change, and by how much | `get_attack_seed`, `submit_variant` |
| Detection | Reads the detector's verdict and the rejection reason, then adapts | `submit_variant` |
| Orchestrator | Decides when to end a round and retrain the detector with successful evasions | `retrain_detector` |
| Risk-Scoring | Gives each new alert a severity: low, medium, high or critical | `triage_alerts`, `submit_triage` |
| Remediation | Gives each alert a concrete response | `submit_triage` |

The LLM makes every decision. The tools are deterministic and shared, in `src/toolkit.py`.

### Attack constraints

A variant only counts as an evasion if it is still a working attack, so every attempt is limited:

- at most **3 features** changed per attempt;
- each feature stays within **×2 of its original value** and inside the 1st–99th percentile of the data;
- a feature that is zero in the original (for example, no reply packets) stays zero;
- only 9 features the attacker controls can change (duration, packet counts, byte totals, inter-arrival times, initial TCP window). Features computed from them (rates, header lengths, packet-length means, subflow counts) are recomputed automatically so a variant is internally consistent.

Successful evasions are added to the training data with weight 50 before retraining.

## Quick start (no API key)

Requires Python 3.10 or newer and Git. Tested with Python 3.15 on macOS (Apple silicon).

```
git clone https://github.com/noor-alqahtani-01/nids-single-agent-baseline.git
cd nids-single-agent-baseline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/single_agent.py --policy random
```

On Windows, activate with `.venv\Scripts\activate`.

`--policy random` runs the same task with random mutations and rule-based triage instead of the LLM. It needs no key and is also the random-mutation reference for hypothesis H2. The working data set (`data/working/`) is committed, so nothing has to be downloaded.

## Full run with the LLM agent

1. Create a free API key at https://aistudio.google.com (Get API key).
2. Create your settings file and add the key:

```
cp .env.example .env
```

Open `.env` and replace `paste-your-key-here` with your key. `.env` is ignored by Git; never commit it.

3. Run a short task (about 8 LLM calls), then the full task (about 25 LLM calls):

```
python src/single_agent.py --rounds 1 --seeds 1 --attempts 3
python src/single_agent.py
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--rounds` | 2 | co-evolution rounds |
| `--seeds` | 2 | attack seeds per round |
| `--attempts` | 4 | attempts per seed |
| `--max-changes` | 3 | features changed per attempt |
| `--max-factor` | 2 | each feature stays within this factor of its original |
| `--policy` | llm | `llm` or `random` |
| `--seed` | 42 | random seed |

| Setting in `.env` | Meaning |
| --- | --- |
| `LLM_BASE_URL` | Base URL of an OpenAI-compatible API |
| `LLM_API_KEY` | Your own key |
| `LLM_MODEL` | Model name; we used `gemini-3.5-flash-lite` |
| `LLM_REASONING_EFFORT` | Optional, for thinking models (for example `low`) |

### Expected output

Each phase gets a heading. Every try shows the change as old -> new with units, the detector's verdict, and why it was caught. The run ends with a table and a plain-language summary:

```
 ROUND 1 of 2: attack and defend
 Attack 1: DoS slowloris  (T1499 Endpoint Denial of Service)
   [RED TEAM]  try 1: duration 51.8 s -> 60.0 s; packets sent 15 -> 10
   [DETECTOR]  CAUGHT as DoS slowloris (confidence 0.78)
               why: header bytes sent is 331 B (benign: ... , attacker cannot change it); ...
 [ORCHESTRATOR] Round 1 result: 0 of 8 tries evaded (ESR 0%)
 OPERATIONS: triage new traffic
   Alert Detected as       Technique Severity  Response
   4     DDoS              T1498     CRITICAL  Activate DDoS mitigation ...
 REPORT
   Round                   Tries  Evaded     ESR   Test F1
   0 (static detector)         -       -       -    0.9986
   1                           8       0      0%    0.9986
   2                           8       0      0%    0.9986
 What this means
   - The detector caught all 16 attack variants (ESR 0%).
   - Detection quality on normal test traffic: F1 0.9986 before, 0.9986 after.
   - 7 new alerts: 7 with the correct attack type, 7 given a severity and a response.
```

Add `--no-color` for plain text (for example when saving the output to a file).

The full report is saved to `outputs/single_agent_llm_report.json` (or `..._random_report.json`).

## Baseline results (seed 42)

| | LLM single agent | Random mutation |
| --- | --- | --- |
| Attack attempts | 16 | 16 |
| Successful evasions | 0 | 0 |
| ESR | 0.00 | 0.00 |
| Test F1 before / after | 0.9986 / 0.9986 | 0.9986 / 0.9986 |
| Alerts triaged with severity and response | 7 of 7 | 7 of 7 (rules) |
| LLM calls | 25 | 0 |
| Time | 78 s | 3.5 s |

This is one run. Variance across three seeds is reported in M5.

**Reading the result.** Under the constraints above, neither policy evaded the detector. The rejection reasons show why: the detector relies most on features the attacker cannot change in our threat model, such as the maximum packet size the server sends back (`Max Packet Length`, `Bwd Packet Length Max`) and `min_seg_size_forward`. Whether the attacker should control more features is a threat-model decision for M5–M6; without any evasions, retraining has nothing to learn from.

## Static detector reference

The static detector from M2 (trained once on the full data, no adaptation) can also be run alone:

```
python src/run_demo.py --no-llm
```

On 248,691 held-out flows it reaches precision 0.9979, recall 0.9998 and F1 0.9989 (attack vs benign). Full numbers are in `models/metrics.json`.

## Reproduce from the raw data

1. Download `MachineLearningCSV.zip` from https://www.unb.ca/cic/datasets/ids-2017.html
2. Copy these files into `data/raw/`: `Tuesday-WorkingHours.pcap_ISCX.csv`, `Wednesday-workingHours.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
3. Run:

```
python src/prepare_data.py
python src/train.py
python src/make_working_set.py
```

`prepare_data.py` cleans the data (strips column-name spaces, removes Infinity, NaN and duplicate rows) and makes a stratified 80/20 split with seed 42. `train.py` trains the full static detector. `make_working_set.py` writes the smaller class-capped set (60,819 training and 19,955 test flows) that the agent retrains on in a few seconds.

Detector settings, used everywhere: `RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=2, class_weight="balanced", random_state=42)`.

## Known limits

- Feature-level mutations only. A variant is internally consistent, but we do not prove it could be sent as real traffic.
- The rejection reason is white-box feedback (features furthest from typical benign traffic among the detector's 20 most important). Per M2, this is available only during training, not in deployment.
- Free API tiers limit requests per day per model. If the quota runs out, the agent stops, prints the report so far, and says so.
- The triage set is small (10 incoming flows), and the LLM's severity and response are not yet scored against a reference. That belongs to the M5 golden set.
- Heartbleed has very few flows, so results for it are not meaningful.

## ATT&CK mapping

| CICIDS2017 label | Technique | Name |
| --- | --- | --- |
| FTP-Patator, SSH-Patator | T1110 | Brute Force |
| DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest | T1499 | Endpoint Denial of Service |
| DDoS | T1498 | Network Denial of Service |
| Heartbleed | T1190 | Exploit Public-Facing Application |

## Repository layout

```
README.md
requirements.txt
.env.example              LLM settings template
data/
  working/                small train and test sets used by the agent (committed)
  sample_flows.csv        incoming flows for triage, and for run_demo.py
  raw/, processed/        full CICIDS2017 data (not in Git)
models/
  detector.joblib         static detector trained on the full data
  metrics.json            its test-set scores
src/
  single_agent.py         THE BASELINE: one LLM agent, all five roles
  toolkit.py              shared tools, data and budgets (also for the multi-agent system)
  llm.py                  OpenAI-compatible client with tool calls
  display.py              readable terminal output and the run summary
  attack_map.py           attack label to ATT&CK technique
  prepare_data.py         clean and split the raw data
  train.py                train the static detector
  make_working_set.py     build data/working/
  agent.py, run_demo.py   static detector reference run
outputs/                  reports from the last runs (not in Git)
```

## Troubleshooting

| Message | Fix |
| --- | --- |
| `command not found: python` | Activate the environment: `source .venv/bin/activate` |
| `CERTIFICATE_VERIFY_FAILED` (macOS, Python from python.org) | Run `open "/Applications/Python 3.15/Install Certificates.command"` (your version in the path) |
| `The LLM is not configured` | Create `.env` as above, or use `--policy random` |
| `daily free quota ... is used up` | Switch `LLM_MODEL` to another model (each has its own quota), wait for the reset at midnight Pacific time, or use `--policy random` |
| `(API busy or rate limit reached, waiting ...)` | Normal on free tiers: the agent waits and retries |
| `data/working/... not found` | Pull the repository again, or run `python src/make_working_set.py` |
