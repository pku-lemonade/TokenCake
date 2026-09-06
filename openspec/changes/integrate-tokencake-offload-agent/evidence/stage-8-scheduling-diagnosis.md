# Agent Scheduling Regression Diagnosis

The acceptance campaign was stopped on 2026-09-06 at the operator's request.
The immediate concern is scheduling-induced preemption and recomputation.
The migrated policy preserves the old admission, prefill and victim rules
while replacing terminal preemption with native lossless requeuing.
That combination needs admission/progress protection before further acceptance
repetitions are useful. No runtime policy was changed during this diagnosis.

## Completed Workload Evidence

All entries below are completed 24-application, 648-model-call workloads.
Both modes generated 271200 tokens per case and had zero terminal preemptions,
client retries, prompt halvings, context errors or HTTP 400 errors.
QPS 1.0 uses the two qualifying repetitions per reference; the other points
have one qualifying reference each. Excluded attempts supply no timing.

| QPS | Native client wall (s) | Agent client wall (s) | Agent/native | Native preemptions | Agent preemptions |
| --- | ---: | ---: | ---: | --- | --- |
| 1.0 | 1362.917 | 6187.162 | 4.540 | 56, 61 | 15158, 15786 |
| 0.5 | 1392.543 | 5454.074 | 3.917 | 45 | 13561 |
| 0.1 | 1417.694 | 5157.303 | 3.638 | 54 | 15465 |

Agent-only disables CPU offload and has no KV connector. Thus CPU KV transfer
cannot account for its regression. Its QPS 1.0 repeat recorded 75157 deferrals
and 243278 prefill caps. These counters include scheduling decisions that can
be rejected or rolled back; they are not executed GPU-token counts.
Likewise, native prompt and iteration-token metrics do not count all repeated
prefill computation and must not be used to dismiss recomputation overhead.

Resolved native and agent QPS 1.0 configurations both use 3050 GPU blocks of
16 tokens, an 8192-token batch budget, 1024 maximum sequences, prefix caching,
async scheduling and `scheduler_reserve_full_isl=true`. Model length is 32768,
GPU memory utilization is 0.5, and the model is Qwen2.5-14B-Instruct.

## Why Historical Improvement Is Not Directly Comparable

In source commit `7a608a4e5`, `opt_scheduler.py:1334` sets a victim to
`FINISHED_PREEMPTED`, frees its KV and puts it in `preempted_stop_requests`.
At `opt_scheduler.py:1771`, the engine emits a final result and frees the
request. `vllm/v1/request.py:201` maps this status to `FinishReason.STOP`.
`vllm_serving.py:388` accepts the successful completion response and returns
its possibly incomplete text, allowing the DAG to advance without retrying it.

The source repository's `final-report.md:15` explicitly reports 39211
`FINISHED_PREEMPTED`, 12629 `FINISHED_LENGTH_CAPPED` and 3840 local completions
across 80 E1 cases. The historical checker permitted terminal preemption.
Finishing every graph node under that checker did not establish complete,
equal model generation. The raw historical result archive is absent locally;
these aggregate counts are attributed to the committed report, not a new
independent audit of those old runs. Their presence does not quantify how much
of the reported speedup came from early termination versus useful scheduling.

The target's `vllm/v1/core/sched/scheduler.py:1076` instead frees KV, resets
computed progress, marks `PREEMPTED` and requeues the request. It eventually
finishes the requested generation. This is the required vLLM behavior, but it
means an overcommitted request can incur the same allocation/preemption cycle
repeatedly. Reintroducing terminal preemption would break serving correctness.

## Current Failure Mechanism

1. `vllm/tokencake/scheduling.py:426` limits annotated prefills to 256 tokens
   under backlog or pressure. The waiting loop can consequently admit many
   partially computed long requests in one batch.
2. `can_allocate()` at line 311 checks incremental physical allocation against
   shared/reserved capacity. Native `KVCacheManager.allocate_slots()` at
   `vllm/v1/core/kv_cache_manager.py:346` additionally checks whether one full
   input would fit now, but allocates only this step's chunk. It does not reserve
   future space for every already-admitted partial prefill. Multiple long
   requests can therefore pass individually and exceed aggregate future KV
   capacity as they progress.
3. At `vllm/v1/core/sched/scheduler.py:502`, rejection by the reservation policy
   enters the native preemption branch even when physical blocks remain free.
   The policy has no guarantee that an admitted request can reach completion
   before that protection makes its next allocation ineligible.
4. `victim()` at `vllm/tokencake/scheduling.py:448` retains the source's
   3000-point margin. Unless the requesting score exceeds the lowest running
   score by that margin, the requester itself is selected. The waiting score
   then permits re-admission, with no recomputation-cost or preemption-count
   penalty in that per-request order. The request can cycle back into pressure.

The 256 cap and 3000 margin are present in the latest source. This diagnosis
has not found a sign inversion or an accidentally changed constant. The
problem is the policy's behavior with resumable requests and its interaction
with current native allocation.

## Offline Mechanism Check

The companion Python file runs production `AsyncScheduler`, `KVCacheManager`
and `SchedulingController`, using 24 distinct 8192-token prompts, eight agent
types, 500 output tokens each and the same block/batch limits. The model
feedback tokens are synthetic; no model weights are loaded or GPU server
launched. Both clocks advance deterministically for policy/history calculations.
This is supporting mechanism evidence, not a smoke test or performance result.

| Diagnostic variant | First-step admissions | Preemptions | Running reservation denials | Scheduled tokens |
| --- | ---: | ---: | ---: | ---: |
| Native | 1 | 0 | 0 | 208584 |
| Agent | 24 | 64 | 53 | 290206 |
| Agent, cap bypassed | 1 | 1 | 1 | 211480 |
| Agent, native victim | 24 | 54 | 25 | 278487 |
| Agent, reservation checks bypassed | 24 | 46 | 0 | 290558 |

Every variant finished all 24 synthetic requests and produced 12000 tokens.
All 64 agent victims were the requesting request itself. The cap-bypassed
variant still preempted one request after 401 output tokens, showing that
running-request reservation denial remains possible after prefill completes.
Instance-local diagnostic overrides were used only inside this CPU process.
They are not proposed production patches or evidence of a measured speedup.

The offline fixture demonstrates additional computation from these policy
interactions. It does not reproduce the complete DAG, shared real prefixes,
arrival feedback or GPU execution and cannot apportion the 4.540 wall-time
ratio among cap, reservation, victim choice and Python scheduling overhead.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python openspec/changes/integrate-tokencake-offload-agent/evidence/stage-8-scheduling-diagnosis.py
```

## Pause And Next Fix

The driver received SIGTERM and exited normally after cleanup. Agent QPS 0.5
launch 1 and offload-agent QPS 1.0 launch 1 retain
`InterruptedError: Campaign interrupted` and incomplete-workload exclusions.
These are operator cancellations, not complete-workload performance failures.
Their launch ledger entries and raw files remain intact. GPU process queries
confirmed that both devices were free after shutdown and after the CPU probe.
The campaign has not been resumed. Task 8.3 and the remaining acceptance gates
stay incomplete; this diagnosis does not trigger snapshot-frontier Phase 2.

The next implementation should bound simultaneous long prefills by their
aggregate future KV needs and give admitted work a progress path through
reservation changes. Victim choice must consider retained computation and
avoid repeatedly resetting the same work. Each change needs separate real
model verification and complete-DAG comparison before it becomes a claimed
performance improvement. Merely changing the margin or deleting the cap is
not yet a validated fix.

The JSON records exact case metrics, result and source-file hashes, cancellation
records and offline outputs. Prior completed-reference evidence was committed
as `b0a9ab2eb`. AI assistance was used.
