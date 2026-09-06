# Stage 21: Complete Generation Commitments

## Change

Default `generation_reserve_mode` to `all`. Admission logically commits the
remaining declared generation budget of every admitted request, using the
existing native cache-group capacity calculation. Allocation remains incremental.
Agent importance ordering, dynamic reservations and borrowing, tool-window KV
preservation, and native prefill chunking remain enabled.

The experiment driver now supports the requested five application arrival rates
and four component configurations. It records native Prometheus counters and
time series for request concurrency, KV occupancy, and GPU activity.

## Full-DAG Diagnostic

Both qualifying launches used the unchanged `conversation-tools` profile at
QPS 1.0: 24 completed DAGs, 648 model calls, and 155,136 generated tokens. Neither
launch retried a client request or halved a prompt. Speculative decoding was
disabled. These are single-launch observations on different A800 GPUs, not a
confidence interval or a fresh native-vLLM acceptance result.

| Measurement | Selective reclaim | All admitted requests |
| --- | ---: | ---: |
| Total E2E (s) | 663.982608 | 639.216284 |
| Mean application latency (s) | 556.516831 | 535.832208 |
| P95 application latency (s) | 638.931505 | 611.887524 |
| Physical preemptions | 58 | 0 |
| Reservation preemptions | 0 | 0 |
| Actually recomputed tokens | 38,958 | 0 |
| Executed tokens | 1,442,150 | 1,083,016 |
| Maximum critical admission wait (s) | 140.096 | 96.460 |
| Critical admission waits >= 60 s | 91 | 129 |
| Critical admission waits >= 180 s | 0 | 0 |
| Mean annotated critical-node LLM latency (s) | 39.462350 | 36.666057 |
| P95 annotated critical-node LLM latency (s) | 100.680107 | 88.819880 |
| Time-weighted KV occupancy | 90.6725% | 87.3327% |
| Time-weighted GPU activity | 99.0141% | 98.6973% |
| CPU-to-GPU bytes | 302,389,395,456 | 358,506,037,248 |
| CPU-to-GPU summed transfer time (s) | 12.151796 | 14.438256 |

The observed E2E reduction is **3.729966%**. The number of waits above 60 seconds
increased even though the maximum wait and critical-node latencies decreased.
No unfinished request is excluded from either successful launch. Transfer time
is summed across jobs and must not be subtracted from E2E as a serial component.

## Evidence

Artifacts are under `/root/autodl-tmp/tokencake-optimization/`:

- Reclaim: `component-reclaim-01/cases/phase-1/offload-agent/qps-1.0/gpu-0/launch-0/result.json`.
- All: `component-all-01/cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-1/result.json`.
- Excluded startup: `component-all-01/cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

| Artifact | SHA-256 |
| --- | --- |
| Reclaim result | `971c7eecc9c20af2acdf3c7df8841be542725c1fc6fc4500fa9b9062febc2d31` |
| All result | `303eeadf4dd5570a8402dc9df488f19907c36528abe88ac5958ec1b594e25a6d` |
| Reclaim runtime tree | `b6cd62f97601cd9030bd9f9ade18cd8385eaaf75448a7611ff6c5344a1e2f4de` |
| All runtime tree | `a0672d391b65e83abc712d87af17741becc9c491a56a55314eb6c1cf475d954f` |
| Observed workload | `e2de7eb7d6187bc0653c3d8967f769d38aa78da7d47043f0e271711ab9b1e65c` |
| Frozen inputs | `3b8f1be50843e2165cd31a0a7c11b27d0a437df1789eca41129307d713f5e171` |

The first `all` launch failed during engine startup while two servers were
allocating independent 100 GiB CPU KV pools. No workload started. The container
has a 240 GiB memory limit; the host-level free-memory check alone missed this
constraint. Kernel OOM counters were zero, so the precise failure mechanism is
unconfirmed. An isolated retry succeeded. The driver now serializes host-offload
servers while allowing GPU-only cases to overlap, and records cgroup limits.
The excluded launch remains in the ledger. CPU KV capacity stays at 100 GiB.

## Validation

- Scheduling and configuration: 208 passed, 3 deselected in 49.80 s;
  `.venv/component-all-generation-cpu.xml`.
- Experiment driver, launcher, report, and workload: 54 passed in 18.39 s;
  `.venv/component-matrix-final-tools.xml`.
- Both successful full-DAG runs passed completion, token-count, frozen-input,
  resolved-configuration, and artifact-hash checks.

The complete five-QPS, four-component evaluation remains pending. Historical
three-QPS native acceptance is supporting context, not a substitute for it.
