# Phase-1 Native Baseline Queue

All three native v0.22 baseline cases completed the full 24-DAG workload and
qualified under the frozen source analyzer. Each case completed 648 model calls
and all 696 required DAG nodes, with a normal client exit and a live server at
completion. HTTP 400, context-length errors, client retries, prompt halving,
terminal `FINISHED_PREEMPTED`, foreign GPU processes, affinity violations and
monitor failures were all zero.

| QPS | Client process wall time (s) | App p50 (s) | App p95 (s) | Native preemptions |
| --- | --- | --- | --- | --- |
| 1.0 | 1362.843 | 1224.086 | 1328.844 | 56 |
| 0.5 | 1392.543 | 1234.651 | 1335.796 | 45 |
| 0.1 | 1417.694 | 1035.562 | 1184.592 | 54 |

The primary interval is client process launch to exit, including the fixed
application arrival schedule and tool work, and excluding server startup.
Application latency and native requeuing preemptions are diagnostic only.
These are the first qualifying baseline measurements; affected-pair repeats
and final median gates remain pending.

## Inputs and verification

The campaign runs with:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver run \
  /root/autodl-tmp/tokencake-acceptance/phase-1-repaired
```

The official baseline checkout is unchanged at
`0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`. Each case used a fresh server on
GPU0, the frozen NUMA0 CPU set, complete Qwen2.5-14B weights, the 24-application
`code-paper-pressure` workload, seed 42, 0.5 GPU memory utilization, a 32768
model length, and `tp=1`, `pp=1`. TokenCake configuration and KV-offload options
were absent. GPU1 continued its independent agent-only QPS 1.0 case throughout
these baseline runs; actual peer and thermal samples are retained.

After each case's qualification, a read-only audit rehashed every artifact
listed by its result, verified all 24 per-application node checks, the 648 retry
trace entries, zero exclusion counters, and equality between the primary
duration and recorded monotonic client timestamps. All 45 artifact hashes
matched. `stage-8-baseline.json` records the exact result paths, SHA-256 values,
case identities, primary timings and diagnostic summaries; each referenced
result retains the full raw artifact manifest.

QPS 1.0 also retains the earlier interrupted baseline launch described in
`stage-8-reservation-repair.md`. It has used two total launches, including that
exclusion. QPS 0.5 and 0.1 have each used one. No excluded timing is used as a
performance input, and the three-launch limit remains in force.

This records completion of the native queue within task 8.1. The agent-only
queue, target offload-agent queue, required repeats, old/Mooncake references,
and performance gates are still pending. Task 8.1 is not marked complete until
its agent-only cases also qualify. GPU0 has proceeded to offload-agent QPS 1.0
without an added cooldown or thermal qualification rule. AI assistance was used.
