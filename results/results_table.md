# M5/M6 results (mean +/- sd over seeds [0, 1, 2])

Threat model: at most 5 controllable features per attempt, each within x4.0 of the original. 

| Metric | Random (no LLM) |
| --- | --- |
| ESR (evasion success rate) | 0.044 +/- 0.014 |
| Cases evaded (of 21) | 3.333 +/- 0.943 |
| ATFE (attempts to first evasion) | 4.524 +/- 0.202 |
| MER vs random | 1.0 |
| Convergence round | 3 +/- 0.0 |
| Test F1 | 0.999 +/- 0.0 |
| Test precision | 0.998 +/- 0.0 |
| Test recall | 0.999 +/- 0.0 |
| Test FPR | 0.002 +/- 0.0 |
| Triage recall (golden) | 1.0 +/- 0.0 |
| Severity accuracy | 1.0 +/- 0.0 |
| Response validity | 1.0 +/- 0.0 |
| LLM calls | 0 +/- 0.0 |
| Total tokens | 0 +/- 0.0 |
| Cost (USD) | 0.0 +/- 0.0 |
| LLM latency (s) | 0.0 +/- 0.0 |
| Wall time (s) | 16.367 +/- 0.236 |
| Failed on immutable feature | 56 +/- 1.633 |
| Protocol errors | 3 +/- 0.0 |
