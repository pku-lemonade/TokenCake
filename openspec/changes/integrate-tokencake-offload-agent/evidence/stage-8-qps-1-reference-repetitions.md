# QPS 1.0 Native And Agent Repetitions

The prescribed QPS 1.0 native and agent-only repetitions are complete.
Each reference has used all three permitted launches, including its excluded
launch 0 from the reservation-repair campaign. Launches 1 and 2 qualify.
The excluded launches contribute no timing to either median.

| Mode | Launch 1 client wall (s) | Launch 2 client wall (s) | Qualifying median (s) |
| --- | --- | --- | --- |
| Native v0.22 | 1362.843447 | 1362.990290 | 1362.916869 |
| Agent-only | 6068.310653 | 6306.012673 | 6187.161663 |

All four qualifying cases completed 24 applications, 696 DAG nodes and 648
model calls each. Each has 648 `FINISHED_LENGTH_CAPPED` and 48
`FINISHED_LOCAL` statuses, no `FINISHED_PREEMPTED`, normal client exit and
a live server. HTTP 400, context-length errors, retries, prompt halving,
foreign GPU processes, affinity violations and monitor errors were zero.

The new native result recorded 61 native preemptions; the new agent result
recorded 15786 native and TokenCake preemptions, 75157 deferrals and 243278
prefill caps. Agent-only's high preemption count recurs in the full repeat.
These are diagnostic observations; they do not establish a Phase-2 snapshot
timing cause.

A read-only audit checked the four full DAG results, all 60 referenced raw
artifact hashes, frozen case identities, resolved model/configuration values,
exact monotonic client durations and final metrics. The report module's
`measurements()` function verified contiguous launch ledgers `[0, 1, 2]`,
the carried exclusion charges and the two qualifying inputs per median.
The companion JSON records exact result paths, identities, hashes and budgets.

The baseline remains official v0.22 commit `0b3ba88f1`; the target runtime
remains `648289c5c`. GPU0 ran native while GPU1 ran agent-only, with the
frozen CPU affinity and independent queue behavior. Both used the full
Qwen2.5-14B model, GPU memory utilization 0.5, maximum model length 32768 and
`tp=1, pp=1`. Neither reference enables KV offload. The command remains:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver run /root/autodl-tmp/tokencake-acceptance/phase-1-repaired
```

Task 8.3 is not complete: offload-agent QPS 1.0 and the other affected pairs
still need their prescribed repetitions. These reference medians are not a
final performance-gate verdict. No fourth launch is allowed for either
reference at this QPS. No smoke timing contributes to this record.

AI assistance was used; full performance acceptance and human review remain
outstanding.
