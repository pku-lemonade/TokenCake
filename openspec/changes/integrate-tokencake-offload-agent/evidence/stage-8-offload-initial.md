# Initial Offload-Agent Queue

All three initial Phase-1 offload-agent launches completed the full 24-DAG
workload. QPS 1.0 and 0.5 qualify; QPS 0.1 is excluded for one startup affinity
violation. Task 8.2 remains incomplete until its replacement qualifies.

| QPS | Client wall time (s) | Application p50 (s) | Application p95 (s) | Qualification |
| --- | --- | --- | --- | --- |
| 1.0 | 6213.263 | 5197.198 | 6183.716 | Qualifying |
| 0.5 | 5912.541 | 5336.369 | 5880.463 | Qualifying |
| 0.1 | 4879.520 | 3550.981 | 4643.498 | Excluded; timing is diagnostic only |

Each case completed 24 applications, 696 nodes and 648 model calls, with 648
`FINISHED_LENGTH_CAPPED` and 48 `FINISHED_LOCAL` terminal statuses. There were
no `FINISHED_PREEMPTED` statuses, HTTP 400 or context-length errors, retries,
prompt halvings, fatal transfer errors or foreign GPU processes. All clients
exited normally with their servers alive. A read-only audit verified all 45
referenced raw artifact hashes, all 29 expected nodes per application, the
frozen identities and workload, resolved configuration, and exact equality
between primary durations and their monotonic client intervals.

The frozen runtime remains `648289c5c`: both TokenCake features enabled,
`TokenCakeConnector`, 100 GiB native CPU offload, GPU memory utilization 0.5,
maximum model length 32768, and `tp=1, pp=1`. All cases use GPU0 and its NUMA0
CPU set. The companion JSON records actual GPU1 peer/idle sample counts
inside the primary interval. Counts are observations, not duration estimates.

## Excluded QPS 0.1 Launch

The monitor recorded descendant PID 525032 with affinity `[113]`, outside
the assigned `0-35,72-107` CPU set. This was the only violating sample among
2305 samples. Its monotonic timestamp was 17763523.90189084, approximately
1.846 seconds before client launch. No violating sample occurred inside the
primary client interval. The same sample did not list that PID as a GPU
process. The recorded data do not identify the descendant's command.

The frozen monitor covers startup and the workload. Its exclusion is
therefore retained, and this launch consumes one of the case's three
permitted launches. Its timing is absent from both QPS 0.1 gate inputs in
`reports/0000.json`. No acceptance condition or budget was relaxed. The
report requests one replacement launch.

## Initial Gates And Repetitions

| QPS | Offload-agent / native | Offload-agent / agent | Report status |
| --- | --- | --- | --- |
| 1.0 | 4.559044 | 1.023887 | Affected-pair repetitions required |
| 0.5 | 4.245858 | 1.084059 | Affected-pair repetitions required |
| 0.1 | Unavailable | Unavailable | Missing qualifying target |

The maximum ratios are 0.9 against native and 1.05 against agent-only.
QPS 1.0 fails the native threshold and enters the agent gray band; QPS 0.5
fails both initial thresholds. The prescribed repetitions, shared-case reuse
and launch budgets are recorded by the report, whose path and SHA-256 are in
the companion JSON. These are initial comparisons, not final medians.

Both qualifying cases performed native D2H and H2D transfers. QPS 0.5
recorded 56 D2H operations (1903165440 bytes), 207 H2D operations
(13718519808 bytes), 605 saved GPU blocks and zero transfer failures. Its
14394 native preemptions agree with the TokenCake preemption counter.
Snapshot rejection and transfer counts alone do not establish the causal
condition for Phase 2. Phase 2 remains absent.

The command remains:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver run /root/autodl-tmp/tokencake-acceptance/phase-1-repaired
```

No smoke timing contributes to this record. AI assistance was used.
Performance acceptance, the reference queues and human review remain pending.
