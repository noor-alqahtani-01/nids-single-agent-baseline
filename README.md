# NIDS Single-Agent Baseline

 group project *Multi-Agent Cyber Defense System with LLM-Driven Adaptive Red-Team Co-Evolution*.

This repository contains one defender agent for network intrusion detection. It reads network flow records, calls a classifier trained once on CICIDS2017, maps each detected attack to a MITRE ATT&CK technique, and writes an alert with a short reason. It is the static-defense control condition that later milestones are compared against.

**Team:** Noor Alqahtani, Maryam Medhat, Maryam Akram

## How the agent works

| Phase | What the agent does |
| --- | --- |
| Perceive | Reads one network flow record (78 CICIDS2017 features) |
| Decide | Tool 1 `classify_flow`: the trained Random Forest returns the attack type and a confidence. Tool 2 `lookup_attack`: returns the MITRE ATT&CK technique. An LLM then writes a short reason for the analyst. |
| Act | Emits a record with verdict, attack type, confidence, technique and reason, and appends it to `outputs/alerts.jsonl` |
| Stop | Ends when every flow in the input file is processed, then prints an alert report |

The verdict always comes from the classifier. The LLM only words the reason, so it cannot change a decision. If no LLM is configured, or a call fails, the agent switches to template reasons and finishes the run.

## Quick start (no dataset download, no API key)

Requires Python 3.10 or newer and Git. Tested with Python 3.15 on macOS (Apple silicon). Not yet tested on Windows or Linux.

```
git clone https://github.com/noor-alqahtani-01/nids-single-agent-baseline.git
cd nids-single-agent-baseline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_demo.py --no-llm
```

On Windows, activate with `.venv\Scripts\activate` instead of the `source` line.

The trained model (`models/detector.joblib`) and 50 sample flows (`data/sample_flows.csv`) are committed, so this runs immediately.

### Expected output

One line per flow with its verdict, followed by this report:

```
ALERT REPORT
  Flows processed : 50
  Alerts raised   : 25
    T1499 Endpoint Denial of Service           16
    T1498 Network Denial of Service            8
    T1110 Brute Force                          1
  Against true labels: precision 1.0000  recall 1.0000  F1 1.0000
    missed attacks: 0   false alarms: 0
  Reasons written by LLM: 0 of 50
  Saved outputs/alerts.jsonl and outputs/report.json
```

## Full mode (reasons written by an LLM)

The agent works with any OpenAI-compatible chat completions API. We used the Google Gemini API.

1. Create an API key at https://aistudio.google.com (Get API key).
2. Copy the example settings file and put your key in it:

```
cp .env.example .env
```

3. Open `.env` in a text editor and replace `paste-your-key-here` with your key. `.env` is ignored by Git; never commit it.
4. Run:

```
python src/run_demo.py --limit 10
```

The report should end with `Reasons written by LLM: 10 of 10`. Each LLM reason takes several seconds, so `--limit` keeps the run short. Without `--limit` all 50 flows are processed.

| Setting in `.env` | Meaning |
| --- | --- |
| `LLM_BASE_URL` | Base URL of the API |
| `LLM_API_KEY` | Your own key |
| `LLM_MODEL` | Model name from your provider |
| `LLM_REASONING_EFFORT` | Optional, for thinking models (for example `low`) |

## Reproduce the model from the raw data

1. Download `MachineLearningCSV.zip` from the CICIDS2017 page: https://www.unb.ca/cic/datasets/ids-2017.html
2. Unzip it and copy these three files into `data/raw/`:
   - `Tuesday-WorkingHours.pcap_ISCX.csv`
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
3. Run:

```
python src/prepare_data.py
python src/train.py
```

`prepare_data.py` strips column-name spaces, removes rows with Infinity or NaN, removes duplicates, and makes a stratified 80/20 split with seed 42. `train.py` trains the Random Forest and writes `models/detector.joblib` and `models/metrics.json`.

## Baseline results

Held-out test set of 248,691 flows (994,764 training flows), seed 42. Full numbers are in `models/metrics.json`.

| Attack vs benign | Value |
| --- | --- |
| Precision | 0.9979 |
| Recall | 0.9998 |
| F1 | 0.9989 |

| Class | Recall | Test flows |
| --- | --- | --- |
| BENIGN | 0.9992 | 182,507 |
| DDoS | 0.9999 | 25,603 |
| DoS GoldenEye | 0.9995 | 2,057 |
| DoS Hulk | 0.9994 | 34,569 |
| DoS Slowhttptest | 0.9904 | 1,046 |
| DoS slowloris | 0.9954 | 1,077 |
| FTP-Patator | 1.0000 | 1,186 |
| Heartbleed | 1.0000 | 2 |
| SSH-Patator | 0.9969 | 644 |

Model: `RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=2, class_weight="balanced", random_state=42)`.

### Known limits of this baseline

- The scores are for traffic drawn from the same distribution as the training data. They say nothing about evasive or mutated attacks, which is the question later milestones test.
- Heartbleed has only 2 test flows, so its recall is not meaningful.
- Only three days of CICIDS2017 are used (brute force, DoS, DDoS). Other attack families are out of scope for M4.
- LLM reasons are written to fit the classifier's verdict. They describe the flow; they are not an explanation of how the classifier decided.

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
.env.example            settings template for the LLM
data/
  sample_flows.csv      50 held-out flows used by the demo
  raw/                  downloaded CICIDS2017 files (not in Git)
  processed/            train.csv and test.csv (not in Git)
models/
  detector.joblib       trained model, feature names, class names
  metrics.json          test-set scores
src/
  prepare_data.py       clean and split the data
  train.py              train and evaluate the detector
  attack_map.py         tool: attack label to ATT&CK technique
  agent.py              the agent: perceive, decide, act
  run_demo.py           end-to-end task and alert report
outputs/                alert log and report from the last run (not in Git)
```

## Troubleshooting

| Message | Fix |
| --- | --- |
| `command not found: python` | The virtual environment is not active in this terminal. Run `source .venv/bin/activate`. |
| `CERTIFICATE_VERIFY_FAILED` in a `Note:` line (macOS, Python from python.org) | Run `open "/Applications/Python 3.15/Install Certificates.command"` (use your Python version in the path), then run the demo again. |
| `Note: LLM not configured ...` | `.env` is missing or incomplete. The run still completes with template reasons. |
| `(rate limit reached, waiting ...)` | The free API tier limits requests per minute. The agent waits and retries; use a smaller `--limit`. |
| `detector.joblib not found` | Run `python src/train.py`, or pull the repository again. |
