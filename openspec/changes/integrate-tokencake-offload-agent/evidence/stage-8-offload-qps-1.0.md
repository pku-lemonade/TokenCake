# First Full-DAG Offload Result

Phase-1 offload-agent QPS 1.0 completed all 24 applications, 696 DAG nodes and
648 model calls. The unchanged source analyzer qualified every application.
The client exited normally with the server still alive. HTTP 400,
context-length errors, retries, prompt halving, terminal `FINISHED_PREEMPTED`,
foreign GPU processes, affinity violations and monitor failures were all zero.
A read-only audit verified all 15 raw artifact hashes, the resolved server
configuration and equality between the primary duration and the recorded
monotonic client interval.

| Measurement | Result |
| --- | --- |
| Client process wall time | 6213.263 s |
| Application p50 latency | 5197.198 s |
| Application p95 latency | 6183.716 s |
| Native and TokenCake preemptions | 15139 each |
| TokenCake deferrals | 81566 |
| Annotated prefill caps | 245927 |
| D2H completed operations / bytes | 61 / 4108320768 |
| H2D completed operations / bytes | 154 / 11047796736 |
| Completed saved GPU blocks | 1306 |
| Transfer failures | 0 |

This is actual 14B model execution with native CPU offload under the complete
workload, including demand-driven H2D during the run. Transfer bytes include
repeated loads after preemption. The native external-prefix counters for new
requests report 1744 hit tokens out of 4854387 queried tokens; previously
preempted requests have a separate internal accounting path. Neither transfer
bytes nor the low rounded log hit rate should replace the primary wall time.
Native copy timings also exclude scheduling, event and submission overhead.

## Initial Comparisons

| Reference | Reference wall time (s) | Target / reference | Required maximum |
| --- | --- | --- | --- |
| Native v0.22 | 1362.843 | 4.559044 | 0.90 |
| Target agent-only | 6068.311 | 1.023887 | 1.05 |

The native improvement requirement is not met by this initial measurement.
The agent-only ratio is within its threshold but inside the three-percentage-
point repetition band. Evaluating the actual case identities and launch ledger
with the report implementation requests two more offload-agent launches and
one more launch of each QPS 1.0 reference after the primary queues finish.
The earlier native and agent-only exclusions still consume their launch
budgets. Shared offload-agent repeats are reused by both comparisons. Final
median decisions and the latest-old parity comparison remain pending.

All three QPS 1.0 modes generated 271200 tokens. Offload-agent used 5390291
prompt tokens, versus native's 5390297 and agent-only's 5392790. These diagnostic
volumes are close; additional generated work does not explain the measured
slowdown. The source analyzer's accepted contract does not compare output text
across modes. Earlier real-model reload tests independently compare exact
outputs, and CUDA overwrite/reload tests compare preserved tensor contents.

The bounded counters record 192 stale-snapshot observations, 192 prefix-gap
observations and 1002400 snapshot-external frontier observations. They count
observations, including repeats, rather than distinct useful lost blocks.
These values alone do not establish the causal snapshot-known timing gap
required for Phase 2. No Phase-2 implementation or scope expansion is triggered
by this initial result.

## Reproduction And Scope

The command and frozen inputs are those in `stage-8-baseline.md`. The server
uses runtime commit `648289c5c`, both TokenCake switches, `TokenCakeConnector`,
100 GiB native CPU offload, full Qwen2.5-14B weights, 0.5 GPU memory utilization,
32768 model length, `tp=1`, `pp=1`, and the assigned GPU0/NUMA0 CPU set.
GPU1 independently progressed from agent-only QPS 1.0 to QPS 0.5; the actual
peer, process, affinity and thermal samples are retained with this result.
`stage-8-offload-qps-1.0.json` records its result path, SHA-256, implementation
identity and exact values; the referenced result includes the raw manifest.

GPU0 has started offload-agent QPS 0.5 with a fresh server. Task 8.2 remains
pending until the other two full cases qualify; target/reference repeats and
final performance acceptance also remain pending. AI assistance was used.
