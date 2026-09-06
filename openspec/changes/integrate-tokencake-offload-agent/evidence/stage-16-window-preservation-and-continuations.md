# Window preservation and continuation workload, 2026-09-06

This candidate restores conservative `all` generation commitments after the
reclaim experiment. Automatic CPU preservation now selects a complete aligned
prefix within CPU capacity and the remaining tool window, including both D2H
and subsequent H2D cost. An explicit positive block limit remains available.
Preservation still requires an eligible completed lifecycle and waiting demand.

The newly authorized `continuation` workload removes the duplicate role prompt
at the five validate-to-repair edges per DAG. Both native and TokenCake receive
this same input-composition change. Graph structure, 24 arrivals, 648 model calls,
271200 output tokens, repository context, and tool durations remain unchanged.
The original source checkout and historical `frozen` profile remain unchanged.
No speculative decoding is used.

Campaign: `/root/autodl-tmp/tokencake-optimization/continuation-01`.
Immutable runtime tree:
`06dc48d48b91e90f8240dacb91cdb92a764b6d5077449f650a8edbf28769ade2`.
Target artifact:
`5844876cf419cf273843d7d764a61497121a1f4528d1aae1d319e5ce6589ff93`.
Observed workload:
`48b9d8615bd260c5297ab38018151afa65e7782e0883b9e9dfadbccd422579aa`.
Native case key:
`82b6dde9d193695c75176a07d31a42530d3cde59f31ab2f42eaaf04bc58acb52`.
Offload-agent case key:
`7e2076567154b2d80f5ba6a8adf05c80299b16b660bfc6c5d940f404d399c601`.

Native uses GPU 0 and offload-agent explicitly uses GPU 1, each with its declared
NUMA affinity and 0.50 GPU memory utilization. These are separate A800 80 GB
devices. Both cases use a fresh process and qualify. Result paths are
`cases/phase-1/native/qps-1.0/launch-0/result.json` and
`cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

| Metric | Native | Offload-agent |
| --- | ---: | ---: |
| Total E2E seconds | 1367.744429 | 1365.696688 |
| Average application latency seconds | 1217.138857 | 1120.224412 |
| Application p95 seconds | 1334.163128 | 1335.473063 |
| Application p99 seconds | 1337.977118 | 1337.288955 |
| Physical / reservation preemptions | 53 / 0 | 0 / 0 |
| Reservation / capacity denials | N/A | 3579 / 1120815 |
| Executed / recomputed tokens | N/A | 3301259 / 0 |
| GPU / CPU admission hit tokens | N/A | 2120928 / 232336 |
| Resume GPU / CPU hit tokens | N/A | 0 / 0 |
| Maximum critical admission wait seconds | N/A | 171.128 |
| Critical admissions waiting at least 180 seconds | N/A | 0 |
| Completed saved GPU blocks | N/A | 14264 |
| Scheduler iteration observations | 33385 | 35139 |

The driver measures E2E around the complete client process; the client's printed
native workload time of 1361.82 seconds excludes its initialization. Relative
improvement uses the same driver metric for both cases: 0.1497%. Average
application latency improves by 7.96%, but tail latency is effectively unchanged.
This experiment does not establish the required 25% E2E reduction. Its fresh
native baseline applies only to this continuation workload.

Both cases complete 24 DAGs, 696 nodes and 648 model calls, with all declared
output tokens. Retries, prompt halvings, terminal preemptions and transfer
failures are zero. TokenCake has no running growth pauses. Denial counters count
repeated scheduling decisions; they are not counts of failed requests. Native
has no TokenCake execution-range or critical-wait instrumentation, so those
fields are unavailable rather than zero.

Verification before freezing:

- Offloading and configuration CPU checks: 127 passed, 3 GPU configuration
  checks deselected, in 27.69 seconds;
  `.venv/optimization-window-prefix-final-unit.xml`.
- Real-model output after CPU restoration and CUDA transfer checks: 6 passed
  in 92.92 seconds; `.venv/optimization-window-prefix-model.xml`.
- Experiment driver, launcher, report and revised workload checks: 47 passed
  in 13.89 seconds; `.venv/optimization-continuation-driver-unit.xml`.
- Final isolated workload probe after typing cleanup: 2 passed in 1.57 seconds;
  `.venv/optimization-continuation-final-unit.xml`.
- Changed-file pre-commit hooks, including type checks: passed.

Tests use `.venv/bin/python -m pytest`. CPU tests set `CUDA_VISIBLE_DEVICES=`;
model tests set `CUDA_VISIBLE_DEVICES=1`, `HF_HUB_OFFLINE=1`, and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`.
The workload probe runs the actual materialized 27-call graph with a test model
peer and checks the five continuation prefixes, unchanged call budgets and
the original tool durations. Its shortened test-only sleeps are not used in
the end-to-end campaign.
