# TokenCake Integration Design

## Context

The target is vLLM v0.22.0 on branch `upstream` (`88e2f9aa7`, based on upstream tag commit `0b3ba88f1`). The behavior source is the clean `../vllm_agent` `submit-impl` HEAD `7a608a4e53ea990b2540c93b4d28cb795b905109`; the latest production optimization path, rather than older documents or dead code, is authoritative. The clean `../vllm-note` `main` HEAD `8b6de0eb9a09ef53f20cf06bd4d17ee264b9c2a7` is a read-only reference for current native vLLM patterns, not the target revision. Source documentation in `tools/tokencake_experiments/{README,STATUS}.md` and `docs/pkua100_cpu_offload_experiment_handoff.md` supplies experiment and implementation evidence. [TokenCake arXiv v4](https://arxiv.org/abs/2510.18586v4), revised 2026-08-21, supplies the frozen algorithmic context, but source HEAD resolves implementation differences.

v0.22 already provides `OffloadingConnector`, a CPU KV manager, asynchronous D2H/H2D workers, load-on-prefix-hit, in-flight store fences, connector metrics, reset handling, and correct request recomputation after preemption. The old fork duplicates most of those facilities and uses request IDs and MCP-specific RPCs that do not fit v0.22. Its useful incremental behavior is dynamic agent scheduling plus pressure-aware preservation of reusable prefixes around tool stalls.

The normal launcher order is generation completion, local post-processing, `stall_started`, tool sleep, `stall_finished`, then downstream DAG execution. Therefore generation state must remain attachable after the vLLM request itself has completed. v0.22 also randomizes its internal request ID, so lifecycle correlation cannot depend on a returned or internal completion ID.

## Goals / Non-Goals

**Goals:**

- Preserve latest-source request scoring, adaptive reservation, offload scoring, backoff, transfer prediction, and lifecycle behavior. Adapt admission, chunking, and victim selection to continued execution after preemption as described below.
- Keep native v0.22 scheduling and CPU offload behavior unchanged when TokenCake is disabled.
- Reuse the native connector data path and keep TokenCake-specific production code small and readable.
- Make failures observable and benchmark comparisons work-equivalent and reproducible.

**Non-Goals:**

- Do not copy `OptScheduler`, the old CPU block pool/manager, swap maps, GPU related-block graph, model-runner transfer events, waiting H2D prefetches, predictive H2D, upload reservations/debt, or inactive compatibility methods.
- Do not add `SchedulingPolicy.AGENT`, TokenCake CLI/environment flags, a TokenCake plugin registry, legacy `/v1/mcp*` routes, or TokenCake debug/reset/status endpoints.
- Do not support or benchmark data parallelism greater than one in the first version; fail startup when either TokenCake switch enables mutable state with effective DP greater than one. Tensor and pipeline parallelism are not intentionally restricted, but are not claimed as validated by the single-GPU acceptance run.
- Do not change benchmark retry count, prompt halving, DAG semantics, text validation, or application timing.

The intentional differences from the latest fork are limited to the v0.22 integration choices in this design: native requeuing preemption and demand-driven H2D replace the old terminal preemption and proactive upload machinery; native tuple tie-breaking, the ordinary-request barrier, and no ordinary borrowing preserve the mixed-traffic ordering contract; aggregate prefill commitments govern admission while admitted requests retain native chunking and physical-capacity progress; Phase 1 uses a completed lifecycle snapshot instead of the global free-LRU frontier; and generic metadata/events replace MCP-specific protocol objects. These are explicit scope decisions rather than claims of byte-for-byte behavioral parity.

## Decisions

### 1. Parse one strict, structured configuration

TokenCake is configured only through `VllmConfig.additional_config["tokencake"]`. A small parser under `vllm/tokencake/` validates known keys once at startup and produces immutable runtime settings. It does not add fields to vLLM's public scheduler policy or duplicate EngineArgs/CLI flags.

```json
{
  "tokencake": {
    "scheduling": {
      "enabled": true,
      "temporal_selection": "first_fit",
      "critical_ratio": 0.75,
      "reserve_ratio_min": 0.05,
      "reserve_ratio_max": 0.30,
      "gpu_usage_low": 0.40,
      "gpu_usage_high": 0.75,
      "reserve_adjustment_step": 0.05
    },
    "offload": {
      "enabled": true,
      "min_gpu_usage": 0.60,
      "high_pressure_gpu_usage": 0.85,
      "score_threshold": 1.0,
      "backoff_steps": 8,
      "eviction_window_blocks": 128,
      "max_relief_blocks": 32,
      "default_stall_s": 1.0,
      "release_lead_s": 0.10,
      "ewma_alpha": 0.5,
      "transfer": {
        "d2h_bandwidth_gbps": 12.0,
        "h2d_bandwidth_gbps": 12.0,
        "d2h_base_time_s": 0.0,
        "h2d_base_time_s": 0.0,
        "submission_time_per_run_s": 0.000003
      }
    }
  }
}
```

Every field is optional and takes the displayed default. Unknown configuration keys, invalid types/ranges, tuning fields supplied under an explicitly disabled section, and inconsistent watermark/ratio bounds fail startup. `offload.enabled=true` additionally requires positive standard `kv_offloading_size`, the native backend, and data-parallel size one. Scheduling and offload remain independent switches. Because the accepted schema keeps `temporal_selection` under `scheduling`, offload-only mode uses fixed `first_fit`; the alternative temporal policies are available only when scheduling is enabled rather than adding a duplicate offload setting.

This strict engine configuration is intentionally different from request metadata: unknown request metadata is preserved and ignored so later producers can extend it without breaking an older server.

### 2. Carry lifecycle metadata through `vllm_xargs`

The OpenAI completion, chat, and responses protocol types are narrowly broadened from scalar-only `vllm_xargs` values to recursive JSON values. The API layer validates the known fields in `vllm_xargs["tokencake"]` before engine admission and then uses the existing `SamplingParams.extra_args` transport. `Request` and prefix-hash identity are not extended or overwritten. Because one lifecycle ID owns one generation snapshot, a TokenCake-annotated API request must expand to exactly one EngineCore request; annotated multi-prompt or multi-choice fan-out is rejected with HTTP 400 while ordinary requests retain native fan-out behavior.

Each target launcher attempt generates `tc-{uuid4.hex}` inside the existing retry loop and sends it as the top-level completion `request_id`. Agent-only and offload-agent also send that value as nested `lifecycle_id`; only offload-agent sends it as the event `lifecycle_id`. Native baseline sends no nested TokenCake metadata or events, and latest-old keeps its original launcher identity. Every target retry gets a new UUID. Application, node, and attempt identifiers remain separate diagnostic fields and never participate in cache hashing or scheduler identity.

The target launcher maps the latest source metadata to these names:

- `type`, `name`, and `priority` become `agent_type`, `agent_name`, and `importance`.
- Application timing becomes `application_started_at_s`, `application_start_offset_s`, and `application_elapsed_s`.
- Graph data uses `depth`, `in_degree`, `out_degree`, `similarity`, `application_max_depth`, `remaining_depth`, `critical_path`, `near_completion`, `join_group`, `dependency_depth`, `fanout_width`, and `memory_weight`.
- Offload intent uses `offload_eligible` and `reusable_prefix`.
- `expected_tool_stall_s` becomes the start event's `estimated_duration_s`.

`workload_profile`, `branch_id`, `branch_group`, `stage_type`, and launcher-local `preserve_llm_output_after_tool` do not enter the core policy. An absent TokenCake namespace is an ordinary v0.22 request. A present namespace requires a valid lifecycle UUID; omitted optional fields take explicit neutral defaults, while malformed known fields return HTTP 400.

### 3. Integrate scheduling directly at a few v0.22 decision points

The scheduler keeps `None` or a lightweight TokenCake scheduling controller. Disabled startup retains the exact native queue classes and native code paths; enabled code uses explicit branches in `scheduler.py` rather than a general-purpose hook framework or null object.

The controller owns score and reservation/accounting state under `vllm/tokencake/`, while `scheduler.py` keeps the short orchestration needed at existing decision points:

- register metadata on admission and release accounting on preemption/finish;
- snapshot waiting/running state and freeze dynamic scores once per scheduler step;
- retain native prefill token clamps and batch budgets, replacing the fixed 256-token cap with aggregate logical capacity admission;
- check new admission immediately before native allocation and commit only after allocation succeeds; retain a progress path for previously admitted requests;
- choose an exact waiting candidate across the native waiting/skipped queues;
- before allocating new blocks, evaluate any active detached preservation snapshot and emit connector-only work when it creates a store job;
- choose a preemption victim, then call v0.22's native `_preempt_request()`.

No public `AGENT` policy and no changes to the generic request-queue factory are needed. In enabled mode an exact selected request can be removed from its native queue; in disabled mode the existing `pop_request()` remains untouched, preserving priority-queue complexity and semantics.

For TokenCake requests, the latest request score is primary and the native tuple `(priority ascending, arrival time ascending, request ID ascending)` is the deterministic tie-break. After native blocked-state and LoRA handling, the scheduler reads the native merged order across `skipped_waiting` and `waiting`; it may dynamically reorder only the contiguous TokenCake segment before the first ordinary-request barrier. This makes the rule that a TokenCake request cannot pass a native-preferred ordinary request executable for both FCFS and priority policies.

All requests share physical capacity. In a mixed workload, every allocation reduces available shared capacity, so an ordinary request can be deferred rather than consume protected capacity; it is never charged to or allowed to borrow a named per-agent reservation. When no queued or running request is annotated, TokenCake reservation and admission logic is bypassed completely. This is an intentional mixed-traffic simplification from the latest fork, which could lend an idle named reservation to an unannotated request.

The reservation calculation preserves the latest source defaults and effective formulas, but block demand and commit accounting use v0.22's multi-cache-group allocation result rather than copying the old single-list block math. TokenCake changes only victim selection inside the native allocation-failure branch. If the victim was already selected earlier in the same scheduler step, the existing running-list removal, token-budget, block, speculative-token, encoder-input, and loop-index rollback is performed before `_preempt_request()` frees KV, marks `PREEMPTED`, resets computed progress, and requeues it. It never produces the old fork's terminal `FINISHED_PREEMPTED` behavior.

The 2026-09-06 optimization revision makes reservations govern new admission.
Already admitted requests allocate through the native physical-capacity path;
growth beyond a revised partition is shared accounting debt that constrains
later admissions until blocks are released. Physical exhaustion still selects
a concrete running beneficiary and victims through native rollback/preemption.
The admission gate uses the native coordinator's full-sequence, multi-group,
recycling-aware requirement and adds the outstanding requirements of every
admitted partial prefill, including asynchronous cache loads. These commitments
are logical only; no full-prompt allocation or separate pool is created.

Waiting order retains the original request score. Victim selection uses a
bounded score neighborhood before comparing recomputation tokens and uniquely
releasable blocks, prefers an adequate single victim when possible, and protects
requests near their generation limit. Scores and time estimates are not added
together. Physically preempted work waits for an improved capacity observation
or completion of competing work before readmission. These cost rules are
candidate policies subject to complete-DAG validation, not measured constants.

The combined candidate restores native prefill chunking without introducing a
replacement numeric cap. The user explicitly deferred single-factor ablations
until after the overall improvement. Model-returned execution ranges count
actual repeated computation, separately from rolled-back scheduling decisions.
Bounded metrics also distinguish physical preemptions, reservation denials,
prefill-capacity denials, resume GPU/CPU matches, and critical admission waits.
For wholly annotated queues without LoRA constraints, each step reuses its
resolved, score-sorted candidate order across capacity denials. Mixed traffic
and LoRA eligibility retain the existing traversal; this optimization does not
change admission or victim policy.

The next combined revision enables `reserve_generation_tokens` by default.
Logical commitments extend to original prompt length plus `max_tokens`, with
native model-length and recycling-aware group limits and speculative lookahead.
Already generated output does not increase the declared total again on resume.
The prefill-only setting remains available for deployments with loose output
bounds. Both modes keep incremental allocation in the native pool.

Offload pressure now checks full admission commitments against uncommitted
capacity and current allocation against physical free capacity. The preservation
window caps the amount observed for protection; it does not disqualify an
otherwise fitting prefill chunk larger than that window. The 128-block window
and 32-block preservation bound remain unchanged for this revision.

### 4. Use one generic, acknowledged lifecycle-event path

The production HTTP surface is only `POST /v1/tokencake/events`, covered by existing `/v1` API-key middleware. It accepts:

```json
{
  "event": "stall_started",
  "lifecycle_id": "tc-...",
  "kind": "stall",
  "estimated_duration_s": 1.0
}
```

or a finish containing only `event=stall_finished` and `lifecycle_id`. `kind` defaults to `stall`; a missing estimate uses `default_stall_s`. The API uses the existing generic EngineCore utility RPC and waits for a typed result, so a successful start response is a barrier before the launcher begins its tool sleep and a successful finish is a barrier before downstream DAG work starts.

The minimal lifecycle state distinguishes generation completion from stall phase and native connector transfer state. Generation finish captures a stable lifecycle-to-prefix snapshot, including only hashes, block/group layout, and the native request context needed by the offload manager, before native request cleanup. It does not retain the `Request` or pin GPU blocks. A later start revalidates hashes/ownership and considers only surviving blocks. Start arriving while generation is still active is recorded; completion then sets one pending post-finish evaluation. After native GPU blocks are freed, that marker counts as scheduler work for exactly one connector-only evaluation and is cleared whether selection succeeds or rejects. This prevents an otherwise idle engine from losing the transition without turning active lifecycle timers into work. If finish follows that accepted start but precedes generation completion, it terminalizes the lifecycle, learns the observed duration once, and prevents later completion from capturing a snapshot or reactivating offload. A native terminal abort/error before successful completion discards partial snapshot state, releases TokenCake scheduling accounting, and enters a tombstone without duration learning; native `PREEMPTED` requeue is not terminal and does not take this path.

The normal state progression is generation complete, start, release/finish, tombstone. The first terminal transition wins: a later generation finish, abort/error, expiry, reset, or repeated event cannot rewrite its terminal cause, accepted-finish flag, timestamps, or history effects. Exact duplicate starts on a live lifecycle and exact duplicate finishes return 200 without resetting timestamps or updating EWMAs twice. A tombstone retains the canonical start fingerprint: replaying that exact accepted start returns 200, while a first or conflicting start against a tombstone returns 409. A finish during its tombstone is idempotent 200 with `duplicate` when it matches an already accepted finish and `late_finish` when any terminalization happened before a first finish, including abort/error, expiry, or reset. Invalid schemas return 400, authentication failure returns the native 401 response, never-seen IDs return 404, a disabled/unavailable TokenCake offload path returns 503, and internal failures return an appropriate non-200 response rather than an error body with HTTP 200. An ID is unknown after tombstone expiry.

Generation-complete state waits a fixed 60 seconds for start. Cleanup is lazy at event handling, request admission/finish, and scheduler entry; no background timer or idle-engine wakeup is added. Expiry removes only TokenCake association/ownership. It neither pins nor evicts native GPU/CPU cache entries.

An active lifecycle safety deadline is `clamp(predicted_duration * 4, 60s, 3600s)`. Terminal state retains a 60-second tombstone. Explicit finish releases TokenCake ownership immediately. Without finish, predicted release occurs early by `max(release_lead_s, 1.25 * estimated H2D duration)`. Release leaves reusable blocks in the native CPU LRU; a later request uses native demand-driven H2D rather than predictive upload.

Duration prediction is 1 second with no signal, the caller estimate when only it exists, the learned EWMA when only history exists, and an equal estimate/history blend when both exist. Observed server-side stall duration updates history once on a valid finish. Shareable history keys are derived from agent type/name and event kind and are held in a private 4096-entry LRU; a lifecycle-ID fallback remains lifecycle-local and is discarded with its tombstone because a UUID-per-attempt value cannot inform another request.

The target launcher's event helper calls `raise_for_status()` so a non-2xx acknowledgement invalidates the case. The historical old launcher stays byte-for-byte unchanged.

### 5. Extend, rather than replace, native CPU offload

`vllm/tokencake/offloading.py` supplies a small connector subclass registered through the existing connector factory. The base `OffloadingConnector` receives only a protected scheduler-construction hook, and `OffloadingConnectorScheduler` receives one protected early-defer predicate checked before store progress advances; their defaults construct the current scheduler and return false. The concrete `CPUOffloadingManager` receives only narrow `retain()` and `release()` operations implemented with its existing per-block reference count; the generic `OffloadingManager` contract is not expanded. Worker-side transfer execution, CPU allocation/LRU, store/load completion, block-reuse fences, HMA support, and native metrics remain the native implementations.

When TokenCake offload is enabled, startup normalizes the native offload connector selection to the TokenCake subclass after validating that standard `kv_offloading_size` and backend settings are valid. When TokenCake offload is absent or disabled, independently configured native CPU offload continues selecting `OffloadingConnector`, including when TokenCake scheduling alone is enabled. Ordinary requests handled by the TokenCake subclass retain native store/load policy.

For ordinary requests, the subclass delegates unchanged native store/load policy. For an annotated request, automatic generation-time stores are always deferred without advancing native store progress. Only after generation completion has captured a detached hash/block/group snapshot, native request cleanup has released its GPU blocks, and an accepted stall is attached may TokenCake selection create a store job; a start received before completion records lifecycle state but cannot enable an automatic store from the live request. Start, or the beginning of a later normal scheduler step while the lifecycle is active, revalidates the detached snapshot and applies the latest pressure, waiting-demand, utility, transfer-cost, eviction-window, relief, threshold, temporal-selection, and rejection-backoff decisions before any new block allocation. The snapshot does not retain a `Request` or pin a block.

When Phase 1 selects surviving blocks, the TokenCake connector scheduler constructs an explicit store job with the native manager's `prepare_store` result, shared job-ID allocator, `TransferJob`, and pending-block fence map. Every detached GPU source block is registered in that fence map at job creation because there is no live request whose later `request_finished()` call can register it. A small detached job-status map supplies keys and native request context on completion so the finished `Request` is not retained; TokenCake completions are consumed there and all ordinary job results continue through the base update path.

Ready keys that already exist in CPU cache are retained immediately. A lifecycle-owned set records only keys for which retain actually incremented the native reference count. A newly prepared key remains at the native not-ready reference value and is never retained until successful `complete_store()` makes it ready; it is then retained only if the lifecycle still owns preservation. Finish or expiry while D2H is in flight therefore leaves a later successful completion unretained in the ordinary native LRU. Release decrements only the lifecycle-owned set, takes and clears that set exactly once, and never assigns the refcount to zero because concurrent native loads or another lifecycle may also hold the key. Thus the active lifecycle protects data with the native eviction refcount but adds no second cache or ownership table in the CPU manager.

The scheduler treats a one-shot post-finish evaluation plus queued or in-flight TokenCake jobs as connector work. Any scheduler step that first publishes one or more unpublished detached store jobs is forced to emit an exclusive connector-only `SchedulerOutput`, even when model work is runnable and regardless of whether registration occurred in event handling or in `schedule()`, because the native worker does not submit those new stores until the following step. At the start of the next model or connector-only step the worker submits the queued D2H; model work may then overlap the transfer, and reuse of a selected GPU source is guarded by the now-effective native pending-job fence. Connector-only work stops as soon as the one-shot evaluation rejects or the job completes/fails, so active lifecycle timers alone never wake or spin an idle engine. HTTP start acknowledgement means the lifecycle transition and any immediately eligible scheduler-side job registration were applied; it does not wait for asynchronous D2H completion.

The latest fork selects reusable blocks from the global free-LRU imminent-eviction frontier using reuse, lineage, and ineffective-preservation metadata owned by its custom block pool. To keep the first implementation small, Phase 1 deliberately limits selection to the completed lifecycle's detached per-group ordered snapshot and never claims to free GPU capacity: those blocks are already free cache entries, and D2H preserves them before overwrite. A bounded, non-mutating BlockPool frontier observation counts hashed entries missing from or lost by the snapshot only for attribution; its result cannot affect Phase-1 selection. It does not add predictive H2D, an independent CPU pool, or a replacement global lineage/reuse tracker.

If an acceptance gate fails after the specified reruns and metrics show that snapshot-known candidates were lost before preservation or missed the exact overwrite window, Phase 2 closes that timing gap by adding only:

- reuse of the Phase-1 read-only BlockPool frontier observation at the native pre-allocation point;
- intersection of exact imminent-overwrite entries with ordered, still-active lifecycle snapshots, selecting only revalidated blocks whose snapshot ancestors remain reachable in CPU/GPU or are selected together;
- connector submission/fencing that orders D2H before compute can overwrite those newly selected sources.

Phase 2 remains behind TokenCake offload enablement and uses the same native CPU manager and worker. It is not implemented merely because Phase 1 differs internally from the old fork; evidence must attribute a gate miss to this timing gap. This is a conservative snapshot-known subset of the latest fork's arbitrary global-frontier behavior, not a claim of exact parity. If a remaining gate miss is attributed to hot frontier blocks outside every retained snapshot, the implementation stops for an explicit decision rather than silently adding the old pool's lineage/reuse/ineffective-history structures.

### 6. Reuse reset and metrics surfaces

No custom reset route is added. After a successful reset with connector reset disabled, native local prefix cache is reset and TokenCake drops only detached GPU snapshot references made invalid by that reset; lifecycle timing, CPU retention, bounded prediction history, and metrics remain. Only after a connector reset succeeds and authoritatively clears CPU blocks regardless of refcount does TokenCake clear connector-owned jobs/references, lifecycle retention ledgers, ownership, active lifecycles, deadlines, and backoff. Cleared active IDs enter tombstones, so late finish remains idempotent and does not release a key that may have been recreated after reset. Learned shareable tool/transfer EWMAs and monotonic metrics survive either reset. A refused local-only reset does not invalidate snapshots, and a refused connector reset does not clear connector-owned TokenCake state; this follows v0.22's component ordering rather than inventing cross-component rollback. Benchmarks use a fresh server for complete isolation.

Native connector D2H/H2D byte, time, and size metrics are retained. TokenCake adds bounded counters/gauges for lifecycle results, policy decisions/reasons, saved blocks, scheduling deferrals/preemptions, and active lifecycle count. Labels are fixed enums only; request IDs and agent names/types never become metric labels. Per-request detail is DEBUG-only and disabled by default. `/metrics` and logs replace old debug/status endpoints.

### 7. Keep the launcher source immutable and make comparisons reproducible

The repository stores a small protocol patch plus a benchmark driver. For target runs the driver creates a disposable checkout/archive of exact source commit `7a608a4e53ea990b2540c93b4d28cb795b905109`, applies the patch, and uses the original dataset by absolute path. Old comparisons execute the original source checkout. The result records source commit, patch hash, dataset hash, server commit, environment, commands, GPU UUID/NUMA affinity, seed/arrival trace, and concurrent peer case.

The patch changes only attempt UUID/request construction, nested metadata names, the event URL/payloads, and HTTP error checking. Every target mode uses a fresh UUID4 top-level request ID per attempt. A small launcher-only switch omits nested TokenCake metadata for the official native baseline, enables matching nested IDs and scheduling metadata for target agent-only/offload-agent, enables lifecycle events only for offload-agent, and leaves the original source launcher untouched for old comparisons. Native baseline and target agent-only use the existing notification-disable path because no offload lifecycle endpoint is active; agent-only still sends scheduling metadata, while baseline sends neither TokenCake metadata nor events. Tool sleep remains unchanged. The patch preserves the old launcher's non-streaming DAG, at-most-four-attempt loop (`while retry_count <= 3`), prompt halving, successor barrier, output analyzer, and outer process timing.

Two repo-local `uv` environments isolate target and source dependencies. The target environment is created through `uv` from the accepted `mooncake_agent`/Torch 2.11-compatible interpreter and package stack without modifying the original environment; target and Mooncake reference commands use that repo-local environment. The old source remains on its compatible Torch 2.6 stack in the second repo-local environment. Provenance records the original `/root/autodl-tmp/conda_envs/mooncake_agent` environment, `mooncake-transfer-engine==0.3.8`, the exact installed wheel (`mooncake_transfer_engine-0.3.8-cp312-cp312-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl`, SHA-256 `37354b69e87f84c3c0162839519041b64b39714adf52bf44eeda37a929a12962`), and tokencake-mooncake configuration commit `696c9a14f30ffeacda1707e8f712b7a214460be6`. No system Python or bare pip command is used.

## Risks / Trade-offs

- [Phase 1 cannot exactly reproduce the old pre-allocation D2H timing] -> Instrument candidate survival and transfer exposure, then add the narrow Phase 2 frontier/fence only when a failed gate is attributed to this gap.
- [The latest fork can rank hot global blocks absent from every active snapshot] -> Keep Phase 2 to a snapshot-known frontier intersection; if evidence shows this excluded surface causes a remaining gate miss, report it and request approval before adding a bounded lineage/reuse tracker.
- [Native correct preemption may be slower than the old fork because it completes work the old fork terminated] -> Require old `FINISHED_PREEMPTED=0` for the hard old-parity gate; otherwise report the run as work-inequivalent and request a comparison decision.
- [Conditional scheduler branches can add tiny disabled overhead] -> Avoid null objects and per-request callbacks when disabled, preserve native queues/pop paths, assert that no controller/scan/subclass is active, and run a migrated-disabled control only if later attribution needs it.
- [Enabled candidate selection may scan queues] -> Freeze scores once per step and keep exact-removal logic enabled-only; the latest old implementation already rebuilt its heap on peek/pop, so this does not worsen its asymptotic enabled path.
- [Metadata/event loss can leak state or produce false benchmark success] -> Use bounded lazy expiry, typed acknowledgements, real HTTP errors, and client-side status checking.
- [Concurrent GPU experiments can contaminate timing] -> Pin each server/client pair to the agreed GPU UUID and NUMA CPUs, record the peer case, and invalidate only third-party GPU process overlap. Temperature is diagnostic only as requested.
- [The accepted parallel matrix compares fixed modes across two A800s and changing peer load] -> Treat the two A800s as directly comparable under the user's single-GPU replacement rule, isolate their NUMA CPU sets, record actual peer/idle state, and report this as residual measurement risk rather than silently adding counterbalancing or cooldown runs.
- [Data-parallel event ownership is ambiguous] -> Fail closed for `data_parallel_size > 1` in the first version and keep lifecycle routing internal so owner routing can be added later without a public schema change.

## Migration Plan

1. Establish clean source/target environments and capture model, dataset, GPU, dependency, and commit provenance.
2. Add structured configuration, request/event validation, lifecycle state, utility acknowledgements, and focused unit tests.
3. Add minimal scheduler integration and verify disabled native scheduler behavior plus agent scheduling invariants.
4. Add the TokenCake native-connector extension, reset/metrics integration, and focused connector/lifecycle tests.
5. Apply the target launcher patch only to a disposable source checkout and run the primary A800 end-to-end matrix from QPS 1.0 to 0.5 to 0.1.
6. Rerun only failed or near-boundary pairs as specified. Add Phase 2 only on the defined attribution evidence, then repeat affected target pairs.
7. Run latest-old offload-agent and Mooncake reference cases after target primary cases, produce a gate report, and run changed-file lint/type/unit checks.

Rollback is configuration-first: omit `additional_config["tokencake"]` to restore native runtime behavior. Code rollback removes the registered subclass and the guarded scheduler/API integration without changing stored model data or native cache formats.
