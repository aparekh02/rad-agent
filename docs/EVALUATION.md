# Demo evaluation: protocol and run log

This file is the full record behind the results in the README: the protocol, written before any run, and every live run. Raw outputs (`*_rounds.jsonl`, `*_scores.csv`, `*_metrics.json`, logs) are in `runs/`. The prompt as frozen before the test run is in `runs/trading_risk_demo.frozen.py`. Before publishing, only local file paths and the data-path default were scrubbed from that file and the logs; prompts, policy and results are unchanged.

## Protocol: rule floor + agent (fixed 2026-10-29 before any run)
- Policy (in `examples/trading_risk_demo.py`):
  - The **rule floor** runs on every update: widen if 1-min vol ≥ 1.5x reference, or within 5 min of a scheduled release.
  - The **agent** may only escalate to flatten, or stand down a vol-triggered widen.
- **Train/control day:** 2024-03-11 (no scheduled event). Look for false flattens and bad stand-downs; prompt changes are allowed here only.
- **Test day:** 2024-03-12 (CPI). At most 2 runs, and every run is recorded below. No prompt changes after the first test run.
- **Pass criteria for the combined policy versus the rule alone on the test day:**
  1. recall no lower than the rule's;
  2. fewer quiet-but-cautious windows (stand-downs that were right);
  3. flatten calls land on severe windows (≥ 3x vol) more often than not.

### Run log
- train1 (2024-03-11): 45 rounds, 0 errors. 0 false flattens, 0 stand-downs: the agent held all 5 vol-widens while the 1-min vol was still high, and all 5 were quiet afterwards (0.78-1.34x). Prompt change: stand down when the 10-second vol is below 1.5x and no alert shows the move still building.
- train2 (2024-03-11): 45 rounds, 0 errors. 4 stand-downs, all quiet afterwards (avg 1.12x); 0 false flattens; quiet-but-cautious 7 (rule) → 3 (combined). **Prompt frozen here.**
- **test1 (2024-03-12, CPI)**: 42 rounds, 0 errors. This was the only test run; no second run was made, because the protocol forbids prompt changes after test 1.

  | Policy | Recall | Precision | Quiet-but-cautious | Flatten calls (severe ≥ 3x after) |
  |---|---|---|---|---|
  | rule alone | 72% | 70% | 10 | 0 |
  | rule + agent (pre-registered) | 56% | 78% | 5 | 4 (4) |
  | *post-hoc: rule + agent escalation only* | *72%* | *70%* | *10* | *4 (4)* |

  Against the pass criteria:
  1. Recall ≥ rule: **FAIL** (56% vs 72%). 5 of its 10 stand-downs turned out risky. The train-day lesson that bursts fade within minutes did not hold after CPI.
  2. Fewer quiet-but-cautious windows: **PASS** (5 vs 10).
  3. Flattens land on severe windows: **PASS** (4 of 4, all at the 12:30 print, followed by 68–170 bps moves).

  The pre-registered policy fails overall. The escalation half works: it adds 4 correct flattens and costs no recall. The stand-down half does not transfer from a quiet day to an event day. The escalation-only row was computed after the fact on test data, so it is a hypothesis to test on new days, not a result.

### Data-layer numbers (the product), same runs
Each cell is train2 / test1.
| Metric | Value |
|---|---|
| Processing cost | 3.6 / 3.5 µs per tick, so ≈ 280k ticks/s per core |
| Fast-path decision | p50 5 µs, p99 27–29 µs |
| Time to first agent decision | 6.3 s / 6.0 s, with 4 research agents running concurrently |
| Data age at agent read | median 1.1–1.7 s at 20x replay; bounded by the demo's 1-minute chunks (3 s wall at 20x), not by the runtime |
| Errors | 0 across 87 agent rounds and 16 research calls |

## Earlier exploratory runs (before the protocol; in-sample, not results)
`runs/live1.log`, `runs/live2_*.log` and their CSVs come from the first stand-alone analyst designs. They exposed the bugs that were then fixed: snapshot-only reads, no vol baseline, the agent ratcheting its own exit conditions, and a look-ahead in the benchmark.
