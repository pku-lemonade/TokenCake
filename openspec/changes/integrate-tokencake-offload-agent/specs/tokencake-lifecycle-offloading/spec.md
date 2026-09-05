## Purpose

Define a bounded lifecycle and selective KV-preservation contract for external tool stalls while retaining vLLM v0.22 native CPU offload, demand restore, cache management, and disabled behavior.

## ADDED Requirements

### Requirement: TokenCake request metadata uses one nested extension object
The OpenAI completion, chat, and responses request schemas SHALL accept recursive JSON values in `vllm_xargs` and SHALL carry TokenCake metadata in `vllm_xargs["tokencake"]` to `SamplingParams.extra_args["tokencake"]`. A present TokenCake object SHALL contain a `lifecycle_id` formatted as `tc-` followed by the 32 hexadecimal digits of a UUID4. The same value SHALL be supplied as the input request's top-level `request_id`. A TokenCake-annotated API request SHALL expand to exactly one EngineCore generation request; annotated multi-prompt or multi-choice fan-out SHALL be rejected rather than ambiguously sharing one lifecycle. Ordinary requests retain native fan-out behavior.

The recognized metadata fields SHALL be `lifecycle_id`, `agent_type`, `agent_name`, `importance`, `depth`, `in_degree`, `out_degree`, `similarity`, `application_started_at_s`, `application_start_offset_s`, `application_elapsed_s`, `application_max_depth`, `remaining_depth`, `critical_path`, `near_completion`, `join_group`, `dependency_depth`, `fanout_width`, `memory_weight`, `offload_eligible`, and `reusable_prefix`. Omitted optional fields SHALL take neutral defaults. Unknown nested metadata SHALL be preserved and ignored. Known fields with invalid types, ranges, or relationships SHALL be rejected before engine admission. If a duplicate generation lifecycle ID nevertheless reaches EngineCore, it SHALL fail through the native request-error path without overwriting or merging the existing lifecycle; this extension SHALL NOT add a synchronous generation-reservation RPC or promise a pre-response HTTP status for that case.

#### Scenario: Valid nested metadata is admitted
- **WHEN** a request contains a valid TokenCake object and matching top-level request ID
- **THEN** the same lifecycle and agent metadata reaches TokenCake coordination through the existing extra-arguments transport
- **AND** no TokenCake field is inserted into native KV-transfer parameters or prefix hashes

#### Scenario: Nested metadata is extensible
- **WHEN** a valid TokenCake object contains an unknown nested field
- **THEN** the request is accepted and the unknown field is preserved but ignored by this implementation

#### Scenario: Known metadata is malformed
- **WHEN** a known TokenCake field has an invalid type or value, the lifecycle ID is not `tc-<uuid4-hex>`, or the top-level and nested IDs differ
- **THEN** the generation request returns HTTP `400 Bad Request`
- **AND** no TokenCake lifecycle state is created

#### Scenario: An annotated request would fan out
- **WHEN** a TokenCake-annotated API request would create more than one EngineCore generation request
- **THEN** the generation request returns HTTP `400 Bad Request`
- **AND** no partial lifecycle state or child generation is created

#### Scenario: TokenCake metadata is absent
- **WHEN** a request has no `vllm_xargs["tokencake"]`
- **THEN** it is accepted as an ordinary v0.22 request
- **AND** it creates no TokenCake lifecycle or proactive-offload state

#### Scenario: Duplicate generation identity reaches the core
- **WHEN** a generation carries a lifecycle ID already owned by another live generation or lifecycle
- **THEN** the new request fails without changing or attaching to the existing lifecycle
- **AND** ordinary v0.22 internal request-ID randomization remains unchanged

### Requirement: Every target launcher attempt has a distinct attempt identity
The adapted target launcher SHALL generate a new UUID4-derived attempt ID inside the existing retry loop for every LLM attempt and use it as the top-level request ID. Target agent-only and offload-agent modes SHALL also use that exact value as the nested lifecycle ID. Only event-enabled offload-agent attempts SHALL send `stall_started` and `stall_finished`, using the same value. Native baseline SHALL send neither nested TokenCake metadata nor lifecycle events, and latest-old comparisons SHALL retain their unchanged identity and protocol behavior. Application, node, and attempt trace fields SHALL remain separate diagnostics and SHALL NOT be used as the uniqueness mechanism.

#### Scenario: An offload-agent attempt succeeds
- **WHEN** a target offload-agent LLM attempt succeeds and enters a tool stall
- **THEN** its top-level request ID, nested lifecycle ID, start event, and finish event all carry the same `tc-<uuid4-hex>` value

#### Scenario: An agent-only attempt is constructed
- **WHEN** a target agent-only LLM attempt is constructed
- **THEN** its top-level request ID and nested lifecycle ID carry the same `tc-<uuid4-hex>` value
- **AND** it sends no lifecycle event

#### Scenario: A native-baseline attempt is constructed
- **WHEN** a target native-baseline LLM attempt is constructed
- **THEN** its top-level request ID is a fresh `tc-<uuid4-hex>` value
- **AND** it sends no nested TokenCake metadata or lifecycle event

#### Scenario: An attempt is retried
- **WHEN** a target attempt fails or times out and the existing retry loop starts another attempt
- **THEN** the retry receives a new top-level attempt UUID and, when annotated, the same new nested lifecycle UUID
- **AND** in annotated modes, the earlier and later attempts cannot attach to each other's events or KV state

### Requirement: TokenCake offload has a strict structured configuration
TokenCake offload SHALL be configured only by `additional_config["tokencake"]["offload"]`. Its fields and defaults SHALL be:

```json
{
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
```

All fields SHALL be optional when the containing TokenCake object is present. TokenCake offload and scheduling SHALL remain independent switches. Unknown keys, invalid types or ranges, inconsistent watermarks, and tuning fields supplied under an explicitly disabled offload section SHALL fail startup. Enabling TokenCake offload SHALL require `data_parallel_size=1`, positive standard `kv_offloading_size`, the native backend, and the v0.22 `OffloadingConnector` implementation rather than a different backend or the simple-offload fallback. Offload-only mode SHALL use fixed `first_fit`; selecting another accepted temporal policy requires scheduling to be enabled because `temporal_selection` remains a scheduling setting.

#### Scenario: TokenCake configuration is absent
- **WHEN** `additional_config` has no `tokencake` object
- **THEN** both TokenCake scheduling and TokenCake offload are disabled
- **AND** native CPU offload remains independently configurable

#### Scenario: Agent-only mode is configured
- **WHEN** scheduling is enabled, TokenCake offload is explicitly disabled without offload tuning fields, and standard `kv_offloading_size` is absent
- **THEN** agent scheduling starts without a CPU-offload connector or lifecycle-event requirement

#### Scenario: Agent scheduling is combined with native CPU offload
- **WHEN** scheduling is enabled, TokenCake offload is explicitly disabled, and valid standard native CPU offload is independently configured
- **THEN** agent scheduling starts with the original `OffloadingConnector`
- **AND** the TokenCake connector subclass and lifecycle-event requirement remain disabled

#### Scenario: Offload-only mode is configured
- **WHEN** TokenCake offload is enabled and scheduling is explicitly disabled without scheduling tuning fields
- **THEN** the TokenCake connector uses fixed `first_fit` selection and no agent scheduling state is created

#### Scenario: Offload-agent mode is configured
- **WHEN** both TokenCake switches and all native offload prerequisites are valid
- **THEN** the server selects the TokenCake extension of native `OffloadingConnector`

#### Scenario: An offload prerequisite is missing
- **WHEN** TokenCake offload is enabled with non-positive CPU capacity, a non-native or incompatible connector, or data parallelism greater than one
- **THEN** startup fails with an actionable validation error instead of silently choosing a fallback

### Requirement: Lifecycle events use one generic authenticated endpoint
The service SHALL provide only `POST /v1/tokencake/events` for TokenCake lifecycle events and SHALL apply the existing `/v1` authentication middleware. The endpoint SHALL accept `stall_started` and `stall_finished`. Every event SHALL contain a valid `lifecycle_id`. A start MAY contain `kind`, which defaults to `stall`, and `estimated_duration_s`, which defaults to configured `default_stall_s`. A finish SHALL require no fields beyond event and lifecycle ID. Unknown or malformed event fields SHALL be schema errors.

#### Scenario: A valid start is submitted
- **WHEN** an authenticated client posts `stall_started` for a known lifecycle with an optional kind and duration estimate
- **THEN** the event is forwarded to EngineCore using the generic utility path

#### Scenario: A valid finish is submitted
- **WHEN** an authenticated client posts `stall_finished` for an active lifecycle
- **THEN** the server measures the observed duration and applies the terminal transition

#### Scenario: Event schema is invalid
- **WHEN** an event has malformed JSON, an unsupported event name, an invalid ID, an invalid duration, or an unknown field
- **THEN** the endpoint returns HTTP `400 Bad Request`
- **AND** lifecycle state remains unchanged

#### Scenario: Authentication fails
- **WHEN** API-key authentication is configured and the event request lacks valid credentials
- **THEN** the endpoint returns the same HTTP `401 Unauthorized` response as other protected `/v1` endpoints

#### Scenario: TokenCake offload is unavailable
- **WHEN** the route is called while TokenCake offload is disabled or its backend is unavailable
- **THEN** the endpoint returns HTTP `503 Service Unavailable`

### Requirement: Event success acknowledges an applied core transition
The HTTP handler SHALL wait for a typed EngineCore utility result. HTTP `200 OK` SHALL mean the core applied the transition or recognized an allowed idempotent event; queueing the RPC alone SHALL NOT count as success. The acknowledgement barrier covers coordination state, not completion of an asynchronous transfer.

#### Scenario: A new transition is applied
- **WHEN** EngineCore applies a valid start or finish
- **THEN** the endpoint returns HTTP `200 OK` with the lifecycle ID, event, resulting state, and an `applied` disposition

#### Scenario: An exact live start or any accepted finish is repeated
- **WHEN** the same canonical start and immutable payload are submitted again for a live lifecycle, or an already accepted finish is submitted again while its live record or tombstone remains
- **THEN** the endpoint returns HTTP `200 OK` with a `duplicate` disposition
- **AND** it does not repeat transfer scheduling, timestamps, EWMA learning, or other transition side effects

#### Scenario: A finish arrives during a tombstone
- **WHEN** `stall_finished` names a lifecycle with an unexpired terminal tombstone
- **THEN** the endpoint returns HTTP `200 OK` with `duplicate` if a finish was already accepted, otherwise `late_finish`
- **AND** it does not reactivate the lifecycle
- **AND** it does not refresh the tombstone timestamp, extend its expiry, or update duration history

#### Scenario: Identity or state conflicts
- **WHEN** the same ID carries conflicting immutable data, changes an accepted start payload, or requests a non-idempotent transition from terminal state
- **THEN** the endpoint returns HTTP `409 Conflict`
- **AND** existing lifecycle state remains unchanged

#### Scenario: The lifecycle is unknown
- **WHEN** an event names an ID with no live record and no unexpired tombstone
- **THEN** the endpoint returns HTTP `404 Not Found`

#### Scenario: Core or backend is unavailable
- **WHEN** the handler cannot obtain an EngineCore acknowledgement because the core or offload backend is unavailable
- **THEN** it returns HTTP `503 Service Unavailable`

#### Scenario: Internal event execution fails
- **WHEN** an unexpected internal failure occurs while applying an event
- **THEN** the endpoint returns HTTP `500 Internal Server Error`
- **AND** it does not return an error payload with HTTP 200

### Requirement: Completion and stall start attach by lifecycle ID
Generation completion and stall start SHALL be tracked as independent facts. Before native request cleanup, TokenCake SHALL capture the lifecycle's prefix hashes, candidate block/group identity, and native offload request context without retaining the finished `Request` or pinning GPU blocks. The lifecycle becomes a D2H candidate only when generation completion and an accepted start have both occurred. Candidate identity SHALL be revalidated before use. If start already exists at completion, completion SHALL set one pending post-finish evaluation that counts as scheduler work only until a single evaluation runs after native GPU-block release; that evaluation SHALL clear the marker whether it selects or rejects. If an accepted finish terminalizes the lifecycle before generation completion, later completion SHALL neither capture a snapshot nor reactivate TokenCake state.

A native terminal abort or error before successful generation completion SHALL discard partial prefix state, release TokenCake scheduling accounting or ownership held for that attempt, and create a 60-second tombstone without training duration history only if the lifecycle is not already terminal. The first terminal transition SHALL remain authoritative: later generation completion, abort/error, expiry, reset, or repeated events SHALL NOT rewrite its cause, accepted-finish flag, timestamps, or history effects. Native `PREEMPTED` requeue SHALL NOT be treated as terminal. A retry uses its independently generated lifecycle UUID.

#### Scenario: Completion precedes start
- **WHEN** generation completes and start arrives within the attach window
- **THEN** the start attaches by lifecycle ID and surviving reusable blocks become eligible for selection

#### Scenario: Start precedes completion
- **WHEN** a valid start is applied while generation is still active
- **THEN** the state records the start and marks one post-finish evaluation when generation completes
- **AND** after native block release, the next connector-only step revalidates the snapshot and clears the marker

#### Scenario: Started generation is the last model request
- **WHEN** a started lifecycle completes while no other model request can keep EngineCore active
- **THEN** its one-shot post-finish evaluation keeps EngineCore active until that evaluation runs once
- **AND** a rejected evaluation leaves no job or marker that can spin the idle engine

#### Scenario: Finish precedes an accepted start
- **WHEN** finish names a known non-terminal lifecycle that has no accepted start
- **THEN** the endpoint returns HTTP `409 Conflict`
- **AND** no duration sample or transfer is recorded

#### Scenario: Finish follows start but precedes generation completion
- **WHEN** an accepted start is followed by finish while its generation is still active
- **THEN** the finish returns HTTP `200 OK` with an `applied` disposition, records the observed duration exactly once, and creates a tombstone without D2H
- **AND** later successful generation completion produces its normal generation result but does not capture a TokenCake snapshot or reactivate the lifecycle

#### Scenario: A recorded block was reused
- **WHEN** a captured block no longer matches its recorded hash or ownership at selection time
- **THEN** that block is excluded without reading or copying stale KV data

#### Scenario: Generation terminates before start
- **WHEN** a generation aborts or errors before successful completion and before accepting start
- **THEN** TokenCake retains no partial prefix or unbounded live lifecycle state
- **AND** it releases attempt accounting and creates a tombstone

#### Scenario: Generation terminates after start
- **WHEN** start was accepted, no finish was accepted, and the generation later aborts or errors before successful completion
- **THEN** TokenCake terminalizes the lifecycle without D2H or duration learning
- **AND** a finish during the tombstone returns the existing `late_finish` disposition

#### Scenario: Generation terminates after an accepted early finish
- **WHEN** start and finish were accepted before generation completion and the generation later aborts or errors
- **THEN** the accepted-finish tombstone and its one duration update remain unchanged
- **AND** a repeated finish retains the `duplicate` disposition

#### Scenario: Generation is preempted and requeued
- **WHEN** native scheduling marks the generation `PREEMPTED` for recomputation
- **THEN** TokenCake preserves the lifecycle association for the requeued request and does not create a tombstone

### Requirement: Lifecycle state has fixed bounded expiry
A completion awaiting start SHALL have a fixed 60-second attach window measured from generation completion. An active lifecycle SHALL have a safety deadline of `clamp(predicted_duration_s * 4, 60s, 3600s)` measured from the applied start. Every terminal lifecycle SHALL retain a 60-second tombstone without KV ownership. These values SHALL be internal constants rather than public configuration.

Expiry SHALL be checked lazily during event, request, and scheduling activity and SHALL NOT add a background timer or keep an otherwise idle EngineCore awake. Cleanup SHALL remove TokenCake state and ownership only; it SHALL NOT forcibly evict native GPU or CPU cache data.

#### Scenario: Start arrives within the attach window
- **WHEN** start is applied less than 60 seconds after completion
- **THEN** the lifecycle attaches and the completion-wait deadline no longer applies

#### Scenario: Start does not arrive
- **WHEN** at least 60 seconds pass after completion and later activity performs lazy cleanup
- **THEN** attachable state is removed and a 60-second tombstone is created

#### Scenario: Start targets a tombstone
- **WHEN** start names an ID whose tombstone has not expired
- **THEN** an exact replay of a start previously accepted for that lifecycle returns HTTP `200 OK` with `duplicate`
- **AND** a first start or a conflicting start payload returns HTTP `409 Conflict`
- **AND** neither case reactivates the lifecycle
- **AND** neither case refreshes the tombstone timestamp or extends its expiry
- **AND** neither case updates duration history

#### Scenario: Active safety time expires
- **WHEN** an active lifecycle passes its derived safety deadline without finish and later activity performs cleanup
- **THEN** TokenCake releases its ownership and creates a tombstone without predictive H2D
- **AND** native in-flight transfer fences remain authoritative

#### Scenario: Tombstone expires
- **WHEN** 60 seconds pass after terminalization and later activity removes the tombstone
- **THEN** subsequent events for that ID return HTTP `404 Not Found`

### Requirement: Duration prediction follows the latest four-case rule
For a lifecycle's internal prediction key, TokenCake SHALL predict one second when neither a caller estimate nor learned history exists, use the estimate when only it exists, use the EWMA when only history exists, and use `0.5 * estimate + 0.5 * history` when both exist. A valid first-observed finish SHALL measure duration server-side with a monotonic clock and update history using configured `ewma_alpha`. Duplicate, late, conflicting, expired, or start-less finishes SHALL NOT update history.

The prediction key SHALL be derived internally from agent type, then agent name, plus event kind. Shareable prediction history SHALL use a private 4096-entry least-recently-used bound. When neither agent type nor agent name exists, lifecycle identity SHALL be a non-sharing fallback whose sample is discarded with the lifecycle tombstone rather than retained in shared history. The system SHALL NOT add another public request field.

#### Scenario: No prediction signal exists
- **WHEN** a valid start has neither a duration estimate nor matching learned history
- **THEN** its predicted duration is one second

#### Scenario: Exactly one signal exists
- **WHEN** a valid start has only a caller estimate or only matching history
- **THEN** its predicted duration equals the available signal

#### Scenario: Both signals exist
- **WHEN** a valid start has a caller estimate and matching history
- **THEN** its predicted duration is the equal-weight blend of those signals

#### Scenario: A valid finish trains history
- **WHEN** the first valid finish for an active lifecycle is applied
- **THEN** the measured duration updates its prediction key's EWMA exactly once

#### Scenario: Shareable duration history reaches its bound
- **WHEN** learning a new shareable key would exceed 4096 retained keys
- **THEN** TokenCake evicts the least-recently-used history entry before recording the new one

#### Scenario: A lifecycle has no shareable prediction identity
- **WHEN** a lifecycle has neither agent type nor agent name
- **THEN** its UUID-keyed duration state is removed when the expired tombstone is lazily collected and cannot grow process-lifetime shared history

### Requirement: Ownership release does not trigger predictive H2D
TokenCake SHALL retain successfully preserved ready CPU keys through the concrete native CPU manager's existing eviction reference count, without extending the generic offloading-manager interface or creating another cache. The CPU manager SHALL increment only keys that exist and are ready and SHALL return the successfully retained set; it SHALL NOT change a not-ready store key. Each lifecycle SHALL own that returned set outside the manager. An explicit finish SHALL atomically take, clear, and release every TokenCake-retained reference exactly once by decrementing rather than resetting its refcount. If finish is absent, ownership SHALL be released at the predicted finish time minus `max(release_lead_s, 1.25 * estimated H2D duration)`. Early predicted release SHALL leave the lifecycle record available for finish and duration learning until its safety deadline. Released CPU data SHALL remain in the native LRU and SHALL NOT reserve GPU blocks or initiate H2D. Ownership release SHALL NOT discard bookkeeping or fences required to settle an already queued or in-flight native D2H job.

#### Scenario: Finish arrives before predicted release
- **WHEN** a valid finish is applied while TokenCake still owns preserved blocks
- **THEN** ownership is released immediately without initiating H2D

#### Scenario: Predicted release becomes due
- **WHEN** the calculated early-release deadline passes without finish
- **THEN** TokenCake releases ownership while leaving native CPU cache policy authoritative
- **AND** the lifecycle may still accept its first valid finish for duration learning

#### Scenario: Finish occurs while D2H is in flight
- **WHEN** explicit finish releases a lifecycle whose native D2H job has not reached a terminal result
- **THEN** lifecycle ownership is released immediately while detached job bookkeeping and source-reuse fences remain until native completion or failure
- **AND** a later successful store completion leaves its keys ready but unretained in the native CPU LRU

#### Scenario: A selected key is already ready in CPU cache
- **WHEN** TokenCake selects a key that the native CPU manager already stores and the lifecycle still owns preservation
- **THEN** TokenCake retains that ready key without issuing a redundant D2H transfer
- **AND** lifecycle release returns exactly that retained reference to normal native LRU eligibility

#### Scenario: A selected key is still being stored
- **WHEN** native store preparation has created a not-ready CPU key
- **THEN** TokenCake does not retain or otherwise alter that key's not-ready reference value
- **AND** it retains the key only after successful completion and only while lifecycle ownership remains active

#### Scenario: No later request reuses the prefix
- **WHEN** ownership has been released and no request demands the CPU-resident prefix
- **THEN** no H2D transfer occurs

### Requirement: TokenCake performs selective proactive D2H through native offload
TokenCake SHALL consider proactive D2H only for an opted-in, offload-eligible, actively stalled lifecycle with still-valid reusable prefix blocks. Phase 1 SHALL apply the latest source's GPU-pressure, waiting-demand, utility, transfer-cost, eviction-window, maximum-relief, score-threshold, temporal-selection, ancestor-reachability, and rejection-backoff behavior to the lifecycle's detached completion snapshot. It SHALL preserve only block sets that can participate in a continuous native prefix hit, request asynchronous copies from the native CPU-offload manager and worker, retain successfully ready CPU keys with the manager's existing reference count until ownership release, and retain native source-reuse fences. The copied GPU entries are already free cache entries; preservation does not create additional physical GPU capacity.

For annotated requests, the connector SHALL always defer automatic native generation-time stores without advancing their native store-progress index. A start received before completion SHALL only record lifecycle state. Only after generation completion captures the detached snapshot, native cleanup releases the request's GPU blocks, and both completion and start are present MAY the one-shot post-finish evaluation or subsequent normal scheduler activity select from that snapshot before new block allocation. When selection creates an explicit native store job, it SHALL use the native manager preparation result, shared job-ID space, worker metadata, and pending-source fence; every detached GPU source ID SHALL enter that fence at job creation rather than waiting for a finished-request callback. A detached completion record SHALL allow the scheduler-side manager to complete the store and conditionally retain its ready keys without retaining the finished `Request`. The one-shot evaluation and queued/in-flight jobs SHALL count as scheduler work. Any scheduler step that first publishes one or more unpublished detached store jobs SHALL publish only connector metadata and SHALL schedule no model tokens, even when model work is runnable and regardless of whether registration occurred in event handling or in `schedule()`, because the native worker cannot submit those new jobs until the following step. The next model or connector-only step SHALL submit them before model compute or any fence wait. Lifecycle timers or a merely active stall SHALL NOT count as scheduler work. The event acknowledgement covers core transition and scheduler-side job registration, not asynchronous transfer completion.

Phase 1 SHALL make a bounded, non-mutating observation of the global free-LRU imminent-eviction frontier only for attribution counters. Those observed blocks SHALL NOT become Phase-1 store candidates unless they are independently present in the active lifecycle snapshot. A failed performance gate MAY enable Phase 2 only for snapshot-known candidates that were lost before preservation or missed the exact overwrite window. Phase 2 SHALL intersect the exact imminent-overwrite frontier with ordered, still-active snapshots and require every selected block's snapshot ancestors to remain CPU/GPU reachable or be selected together. It SHALL NOT add arbitrary global selection or a lineage/reuse tracker without a separate explicit decision.

#### Scenario: Waiting demand appears after start
- **WHEN** start was accepted without a useful waiting request and a later scheduler step introduces useful waiting demand while the lifecycle remains active
- **THEN** TokenCake reevaluates the still-valid snapshot before allocating new blocks in that step
- **AND** a selected store job is registered before any selected source can be reused

#### Scenario: A useful candidate is selected
- **WHEN** a valid stalled prefix passes benefit, pressure, demand, capacity, and safety gates
- **THEN** TokenCake selects at most the configured relief limit and submits native asynchronous D2H

#### Scenario: Start arrives while EngineCore is otherwise idle
- **WHEN** an accepted start makes a detached snapshot immediately eligible and no model request is runnable
- **THEN** the queued store job keeps EngineCore active for connector-only no-forward steps
- **AND** the job reaches the native worker without a background timer, a new worker transfer implementation, or an unrelated dummy model batch

#### Scenario: An unpublished detached job exists while model work is runnable
- **WHEN** a detached store job registered by event handling or scheduler selection is first published in a step that could otherwise schedule model tokens
- **THEN** that publication step emits an exclusive connector-only no-forward output and does not allocate or compute model work
- **AND** the following step submits the queued D2H before model compute, after which native fencing permits safe overlap

#### Scenario: A stall is active but has no selected job
- **WHEN** policy rejects or defers all snapshot blocks and no model request or connector job remains
- **THEN** the lifecycle remains lazily tracked without keeping EngineCore awake

#### Scenario: No waiting work benefits
- **WHEN** no runnable waiting allocation would overwrite reusable entries that preservation could benefit
- **THEN** TokenCake skips D2H and records a bounded reason

#### Scenario: An ordinary request uses the TokenCake connector
- **WHEN** TokenCake offload is enabled but a request has no TokenCake metadata
- **THEN** its automatic store/load decisions, store-progress index, connector metadata, and completion path are semantically identical to the original `OffloadingConnector`

#### Scenario: The global frontier contains a candidate outside the snapshot
- **WHEN** Phase 1 observes a hashed, potentially reusable imminent-eviction entry that is absent from every active lifecycle snapshot
- **THEN** it increments only a bounded attribution counter and does not select or transfer that candidate

#### Scenario: A candidate is unsafe or unprofitable
- **WHEN** its blocks are invalid or prefix-unreachable, CPU capacity is unavailable, transfer cost exceeds benefit, pressure is too low, or rejection backoff applies
- **THEN** TokenCake skips those blocks without changing native cache ownership

#### Scenario: A source block is about to be reused
- **WHEN** native allocation would reuse a block involved in an incomplete TokenCake D2H
- **THEN** the job has already been submitted by the preceding exclusive publication step
- **AND** native connector fencing prevents compute from overwriting it before the required transfer dependency is satisfied

#### Scenario: An asynchronous native transfer fails after acknowledgement
- **WHEN** the native worker reports or raises a D2H or H2D transfer failure after the lifecycle event was acknowledged
- **THEN** TokenCake preserves the native fail-closed connector behavior and the affected end-to-end case is non-qualifying
- **AND** the system does not claim that the earlier HTTP acknowledgement represented transfer success

### Requirement: H2D remains native and demand driven
TokenCake SHALL NOT implement predictive upload, proactive H2D, upload reservation/debt, or a second CPU cache. A later request with a matching prefix SHALL use the v0.22 native lookup, allocation, asynchronous H2D, completion, and miss/recompute paths.

#### Scenario: A later request finds a CPU prefix
- **WHEN** native prefix lookup finds a continuous usable prefix preserved by TokenCake
- **THEN** native demand-driven H2D restores it and native completion fencing gates model use

#### Scenario: A later request misses
- **WHEN** native lookup does not find a continuous usable prefix
- **THEN** the request follows native recomputation behavior

#### Scenario: Native offload-only mode runs
- **WHEN** native CPU offload is configured without TokenCake
- **THEN** the original `OffloadingConnector` store, load, eviction, reset, and metric behavior remains unchanged

### Requirement: Native reset semantics remain authoritative
TokenCake SHALL add no reset endpoint. After a successful `/reset_prefix_cache` with `reset_external=false`, TokenCake SHALL discard only detached GPU snapshot references invalidated by the native local reset and SHALL retain lifecycle timing, prediction history, and metrics. With `reset_external=true`, only after the native connector reset succeeds SHALL TokenCake additionally clear connector-owned jobs/references, block ownership, active lifecycles, deadlines, and rejection backoff. Learned shareable duration/transfer EWMAs and monotonic process metrics SHALL persist. Cleared live IDs SHALL enter tombstones. A refused local-only reset SHALL NOT invalidate snapshots, and a refused connector reset SHALL NOT clear connector-owned TokenCake state. The implementation SHALL preserve v0.22's ordered component results rather than attempt cross-component rollback when local and connector outcomes differ.

#### Scenario: Local-only reset succeeds
- **WHEN** `/reset_prefix_cache` succeeds with `reset_external=false`
- **THEN** native local cache reset behavior is preserved
- **AND** invalid detached GPU snapshot references are discarded while lifecycle timing, predictions, and metrics remain unchanged

#### Scenario: External reset succeeds
- **WHEN** `/reset_prefix_cache` succeeds with `reset_external=true`
- **THEN** native external cache/jobs and TokenCake live coordination state are cleared consistently
- **AND** late finish for a cleared lifecycle remains idempotent during its tombstone
- **AND** the cleared lifecycle retention ledger cannot release a same-key CPU entry created after reset

#### Scenario: A reset component is refused
- **WHEN** native running-request rules refuse a local-only reset or connector rules refuse an external reset
- **THEN** TokenCake retains the snapshot or connector-owned state governed by that refused component
- **AND** it does not roll back a different component that v0.22 already reset successfully

### Requirement: TokenCake observability is bounded
The standard `/metrics` surface SHALL retain native D2H/H2D byte, time, and size metrics and SHALL expose bounded TokenCake lifecycle outcomes, active-state counts, selection decisions/reasons, saved-block counts, and scheduling deferral/preemption counters. Metric labels SHALL contain bounded enums only and SHALL NOT contain lifecycle IDs, prompts, agent names, agent types, or other request values. Per-request details SHALL be DEBUG-only and disabled by default.

#### Scenario: An event or decision occurs
- **WHEN** TokenCake applies an event or makes an offload/scheduling decision
- **THEN** the matching bounded counter or gauge is updated

#### Scenario: Native transfer executes
- **WHEN** TokenCake selection causes native D2H or a later request causes native H2D
- **THEN** native connector transfer metrics retain their existing meaning

#### Scenario: A metrics endpoint is scraped
- **WHEN** a client reads `/metrics`
- **THEN** no request-specific unbounded value appears as a metric label

### Requirement: Legacy MCP production interfaces are not provided
The system SHALL NOT add `/v1/mcp`, `/v1/mcp/finished`, `/v1/mcp/health`, `/v1/mcp/debug`, or `/v1/mcp/reset_prefix_cache`. `/health`, `/metrics`, and `/reset_prefix_cache` retain their existing general meanings.

#### Scenario: A legacy MCP route is requested
- **WHEN** a client requests an old MCP-specific path
- **THEN** it receives HTTP `404 Not Found`
- **AND** no TokenCake or cache state changes
