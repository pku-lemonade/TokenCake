# Full-DAG Scheduling Recovery

The corrected agent-only QPS 1.0 case completed all 24 applications, 696 DAG
nodes and 648 model calls. The unchanged source analyzer qualified the result.
The client exited normally with the server still alive; HTTP 400,
context-length errors, client retries, prompt halving, terminal
`FINISHED_PREEMPTED`, GPU contamination, affinity violations and monitor
failures were all zero. All 15 raw artifact hashes matched on re-audit.

This exercises the running-reservation repair from `648289c5c` on the complete
workload that exposed the earlier progress failure. The reservation-denial
reproducer and smaller real-model pressure tests remain supporting evidence;
this result supplies the full-DAG correctness evidence for QPS 1.0.

| Measurement | Result |
| --- | --- |
| Client process wall time | 6068.311 s |
| Application p50 latency | 5172.733 s |
| Application p95 latency | 6043.801 s |
| Native requeuing preemptions | 15158 |
| TokenCake preemption counter | 15158 |
| TokenCake deferrals | 80797 |
| Annotated prefill caps | 241779 |

The first qualifying native QPS 1.0 time was 1362.843 s, so this agent-only
measurement took 4.453 times as long. That descriptive ratio and the large
preemption count record a substantial performance concern. The accepted hard
gates compare target offload-agent against its references and require the
specified affected-pair repeats; those measurements are still pending.

Source inspection confirms that the migrated victim-score rule and its
3000-point margin match the latest source. The source running branch at
`vllm/v1/core/sched/opt_scheduler.py` terminalizes victims as
`FINISHED_PREEMPTED`, whereas this change deliberately retains native
requeuing and recomputation. The upcoming old-reference cases must therefore
establish their own work equivalence before entering a parity gate.

The result was produced by the same `driver run` command and frozen campaign
documented in `stage-8-baseline.md`, on GPU1 and its NUMA1 CPU set. Resolved
configuration confirms scheduling-only enablement, no KV connector or CPU
offload, full 14B weights, 0.5 GPU memory utilization and 32768 model length.
`stage-8-agent-qps-1.0.json` records the exact result identity and SHA-256;
the referenced result contains commands, configuration, raw data and the full
artifact manifest.

The earlier failed launch still consumes one budget entry, so QPS 1.0 has used
two of its three allowed launches. No excluded timing contributes to this
measurement. Agent-only QPS 0.5 has started, and GPU0 is independently executing
offload-agent QPS 1.0. Task 8.1, all offload-agent performance gates and reference
comparisons remain pending. AI assistance was used.
