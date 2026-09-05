# Primary Native And Agent Queues

Task 8.1 is complete. The official native baseline and migrated agent-only
queues each completed the three prescribed QPS points in descending order.
All six full cases qualified: 144 applications, 4176 DAG nodes and 3888 model
calls in total. Each case used a fresh server and the full Qwen2.5-14B model.

## Agent-Only Results

| QPS | Client wall time (s) | Application p50 (s) | Application p95 (s) | Native / TokenCake preemptions |
| --- | --- | --- | --- | --- |
| 1.0 | 6068.311 | 5172.733 | 6043.801 | 15158 |
| 0.5 | 5454.074 | 4243.416 | 5407.878 | 13561 |
| 0.1 | 5157.303 | 3756.795 | 4938.547 | 15465 |

Each case completed 24 applications, 696 nodes and 648 model calls. The source
analyzer recorded 648 `FINISHED_LENGTH_CAPPED` and 48 `FINISHED_LOCAL`
terminal statuses per case, with no `FINISHED_PREEMPTED`. HTTP 400,
context-length errors, retries, prompt halving, foreign GPU processes,
affinity violations and monitor errors were zero. Clients exited normally,
with their servers still alive. Native and TokenCake final preemption counters
agree in every agent-only case.

A read-only audit rehashed the 90 raw artifacts referenced by these six native
and agent-only results. It also checked every application against its 29
expected nodes, the frozen identities and input contract, resolved model and
memory settings, and exact equality between each primary duration and its
recorded monotonic client interval. The baseline results are recorded in
`stage-8-baseline.json`; `stage-8-agent-queue.json` records all agent-only
result paths, hashes and exact measurements.

## Reproduction And Isolation

The command remains:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver run /root/autodl-tmp/tokencake-acceptance/phase-1-repaired
```

Agent-only uses runtime commit `648289c5c` with scheduling enabled, TokenCake
offload disabled, no KV connector and no native CPU offload. The resolved
configuration retains GPU memory utilization 0.5, maximum model length 32768,
`tp=1` and `pp=1`. GPU1 and its NUMA1 CPU set are fixed by the frozen
campaign. Its independent queue overlaps the actual GPU0 baseline and
offload-agent cases; process, affinity, thermal and concurrent activity
samples remain with each result. GPU1 becomes idle naturally after QPS 0.1.

The earlier excluded QPS 1.0 launch is still charged to the native and agent-only
budgets. Each therefore has used two of its three total launches at QPS 1.0;
the other primary native and agent-only points have used one launch each.
The excluded attempts are not treated as completed full workloads.

## Remaining Acceptance

These are complete workload correctness results. Agent-only's substantial
runtime and preemption counts remain performance concerns. No agent/native
ratio creates an additional hard gate; the required comparisons use
offload-agent against each reference. QPS 1.0 and QPS 0.5 offload-agent initial
results trigger the prescribed affected-pair repetitions. The last primary
offload-agent case, those repeats and both reference queues remain pending.

All reported prompt and generation counts are diagnostic under the unchanged
source analyzer. It does not compare generated text across modes. No smoke
result contributes to this record. AI assistance was used; final performance
acceptance and human review remain outstanding.
