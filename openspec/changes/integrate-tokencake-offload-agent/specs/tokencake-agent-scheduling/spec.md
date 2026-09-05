## Purpose

Define an opt-in agent-aware scheduling capability that uses structured TokenCake metadata to prioritize multi-agent DAG work and protect KV-cache capacity while preserving native vLLM behavior for ordinary and disabled workloads.

## ADDED Requirements

### Requirement: Structured TokenCake configuration
The system SHALL configure TokenCake agent scheduling exclusively from an object under `additional_config["tokencake"]`. When that TokenCake object is present, the scheduling fields and defaults SHALL be `enabled=true`, `temporal_selection="first_fit"`, `critical_ratio=0.75`, `reserve_ratio_min=0.05`, `reserve_ratio_max=0.30`, `gpu_usage_low=0.40`, `gpu_usage_high=0.75`, and `reserve_adjustment_step=0.05`. Supported temporal-selection values SHALL retain the latest source behavior for `first_fit`, `best_fit`, and `priority_first` when scheduling is enabled. Because this setting remains in the scheduling section, TokenCake offload with scheduling disabled SHALL use `first_fit` and SHALL NOT introduce a second offload-specific setting. The system SHALL NOT add a TokenCake-specific command-line flag or an `agent` value to the public vLLM scheduling policy.

Unknown keys, invalid types/ranges, inconsistent minimum/maximum or low/high values, and tuning fields explicitly supplied while scheduling is disabled SHALL fail engine startup instead of being ignored or coerced.

#### Scenario: Enable agent scheduling with structured configuration
- **WHEN** `additional_config["tokencake"]` is an object that explicitly enables agent scheduling and all supplied values are valid
- **THEN** the system activates TokenCake agent scheduling while retaining `fcfs` and `priority` as the only public vLLM scheduling policies

#### Scenario: Reject malformed TokenCake configuration
- **WHEN** `additional_config["tokencake"]` is not an object or contains an invalid agent-scheduling value
- **THEN** engine initialization fails with an error that identifies the invalid TokenCake configuration instead of partially enabling the capability

#### Scenario: Apply scheduling defaults
- **WHEN** a TokenCake object is present and omits one or more scheduling fields without explicitly disabling scheduling
- **THEN** the system applies the declared defaults for those omitted fields

#### Scenario: Do not consume legacy configuration in the engine
- **WHEN** a legacy TokenCake flag or environment variable is present but structured TokenCake configuration does not enable agent scheduling
- **THEN** that legacy value does not activate or configure TokenCake agent scheduling

### Requirement: Structured request metadata
The system SHALL accept per-request TokenCake agent metadata only from the `tokencake` object in the request's extra arguments. TokenCake metadata SHALL remain separate from the native request priority and native KV-transfer parameters.

#### Scenario: Accept an annotated agent request
- **WHEN** a request includes a valid `extra_args["tokencake"]` object containing agent and DAG scheduling metadata
- **THEN** the scheduler makes that metadata available to dynamic ordering, reservation, and admission decisions without rewriting the request's native priority

#### Scenario: Keep connector parameters independent
- **WHEN** a request contains both `extra_args["tokencake"]` and native KV-transfer parameters
- **THEN** each namespace retains its own meaning and TokenCake metadata does not replace or reinterpret the native KV-transfer parameters

### Requirement: Latest dynamic agent ordering
When TokenCake agent scheduling is enabled, the system SHALL order annotated requests using the latest TokenCake dynamic scoring behavior. The score SHALL incorporate TokenCake `importance`, DAG depth and remaining depth, in-degree and out-degree, similarity and parallel-join pressure, application age and queue wait, progress toward completion, critical-path and near-completion indicators, join and dependency information, fan-out, and memory weight. A higher TokenCake score SHALL represent more important work.

After native blocked-state and LoRA eligibility handling, the scheduler SHALL derive the configured native merged order across the skipped and waiting queues. It SHALL dynamically reorder only the contiguous annotated segment before the first ordinary request in that order. The first ordinary request is a barrier that later annotated requests SHALL NOT cross.

#### Scenario: Prefer higher-value DAG work
- **WHEN** two schedulable annotated requests in the same reorderable segment have different TokenCake scores in the same scheduling step
- **THEN** the request with the higher score is considered before the request with the lower score

#### Scenario: Refresh time-dependent priority
- **WHEN** queue wait, application age, progress, or other request-scoring state changes between scheduling steps
- **THEN** the system recomputes the affected ordering before selecting work for the next step

#### Scenario: Preserve a stable order within one step
- **WHEN** an annotated request is inspected and then removed during one scheduling step without an intervening state update
- **THEN** the score used for that step remains stable so that removal selects the same request that was inspected

#### Scenario: Resolve equal scores deterministically
- **WHEN** annotated requests have equal dynamic scores
- **THEN** the system orders them by lower native priority, earlier arrival time, and then a stable request identifier

#### Scenario: Preserve an ordinary-request barrier
- **WHEN** native merged order contains an annotated request, then an ordinary request, then another annotated request with a higher TokenCake score
- **THEN** TokenCake may reorder annotated work before the ordinary request but SHALL NOT move the later annotated request across that ordinary request
- **AND** the rule applies across both waiting queues under FCFS and native priority policy

### Requirement: Annotated long prefills are bounded under pressure
When TokenCake agent scheduling is enabled, an annotated request asking to schedule more than 256 new tokens in one step SHALL be capped to 256 after the existing native long-prefill, token-budget, and model-length clamps if either the waiting queue is non-empty, the running count is at least `max(2, floor(max_num_running_reqs / 2))`, or GPU KV-cache usage is at least `0.80`. The cap SHALL never increase a native token allowance. It SHALL be bypassed for ordinary requests and whenever TokenCake agent scheduling is disabled; the fixed source-compatible cap and pressure threshold SHALL NOT add public configuration fields.

#### Scenario: Cap an annotated prefill under backlog
- **WHEN** an annotated request has more than 256 natively allowed new tokens and either accepted backlog condition is true
- **THEN** that scheduler step admits at most 256 new tokens for the request before KV allocation

#### Scenario: Cap an annotated prefill under KV pressure
- **WHEN** an annotated request has more than 256 natively allowed new tokens, no backlog is present, and GPU KV-cache usage is at least `0.80`
- **THEN** that scheduler step admits at most 256 new tokens for the request before KV allocation

#### Scenario: Preserve the native allowance outside the cap
- **WHEN** the request is ordinary, TokenCake scheduling is disabled, no accepted pressure/backlog condition exists, or the native allowance is already at most 256 tokens
- **THEN** TokenCake does not change the native per-step token allowance

### Requirement: Adaptive agent capacity reservation
The system SHALL maintain a bounded KV-capacity reservation for the currently important annotated agent types. It SHALL derive agent-type importance from TokenCake request importance, waiting demand and delay, deferrals, preemptions, observed execution history, and graph characteristics; adjust the total reservation according to GPU KV-cache pressure; and partition reserved capacity among important agent types according to their current importance and memory demand.

#### Scenario: Increase protection under high pressure
- **WHEN** GPU KV-cache usage reaches the configured high-pressure threshold and important annotated work is active
- **THEN** the reservation is increased by the configured bounded adjustment without exceeding its configured maximum

#### Scenario: Reduce protection under low pressure
- **WHEN** GPU KV-cache usage reaches the configured low-pressure threshold
- **THEN** the reservation is reduced by the configured bounded adjustment without falling below its configured minimum

#### Scenario: Protect a waiting important agent type
- **WHEN** an important annotated agent type has waiting work and another request would consume capacity reserved for that agent type
- **THEN** the reserved capacity remains available to the important waiting work

#### Scenario: Borrow idle reserved capacity
- **WHEN** an annotated request lacks shared capacity, reserved capacity is idle, and no request from the reservation owner is waiting for it
- **THEN** the scheduler allows that annotated work to borrow the otherwise idle reserved capacity

### Requirement: Agent-aware admission control
The system SHALL apply TokenCake admission control before native KV-cache allocation whenever a mixed workload has active or waiting annotated work. Admission SHALL account for the request's incremental KV demand, current shared capacity, applicable reserved capacity, and native physical capacity. Every physical allocation SHALL reduce available shared capacity. Annotated reservation owners MAY consume their applicable reservation; ordinary requests SHALL use shared capacity only. A reservation denial SHALL defer the request rather than report it as completed or failed. When all queued and running work is ordinary, the TokenCake admission path SHALL be bypassed.

#### Scenario: Admit work that fits shared capacity
- **WHEN** an annotated request's incremental KV demand fits within available shared capacity and native allocation constraints
- **THEN** the request is admitted and its actual allocation is charged to shared capacity

#### Scenario: Admit important work from its reservation
- **WHEN** an important annotated request does not fit shared capacity but fits its available reserved capacity and native allocation constraints
- **THEN** the request is admitted and its actual allocation is charged to the corresponding reservation

#### Scenario: Defer work that would violate protection
- **WHEN** admitting an annotated request would consume capacity protected for waiting important work
- **THEN** the request remains pending for a later scheduling step and no terminal result is emitted

#### Scenario: Release accounted capacity
- **WHEN** allocated KV blocks are freed because a request is preempted, finishes, or is aborted
- **THEN** the scheduler releases the matching shared and reserved accounting exactly once

#### Scenario: Account for ordinary allocation in a mixed workload
- **WHEN** an ordinary request is admitted while annotated work is active or waiting
- **THEN** its actual allocation reduces shared available capacity
- **AND** it neither consumes nor borrows a named per-agent reservation

#### Scenario: Bypass admission for an all-ordinary workload
- **WHEN** no queued or running request contains TokenCake metadata
- **THEN** the scheduler performs no TokenCake reservation or admission calculation

### Requirement: Native fallback for ordinary requests
Requests without `extra_args["tokencake"]` SHALL remain valid ordinary vLLM requests. Their native priority interpretation, relative ordering among ordinary requests, request lifecycle, and KV-transfer behavior SHALL be preserved. Annotated requests and protected capacity MAY affect when ordinary requests run under shared contention, but the system SHALL NOT synthesize TokenCake metadata for them or reject them because metadata is absent.

#### Scenario: Schedule an all-ordinary workload
- **WHEN** TokenCake agent scheduling is enabled but no queued or running request contains TokenCake metadata
- **THEN** scheduling and preemption decisions are semantically identical to the configured native vLLM policy

#### Scenario: Schedule a mixed workload
- **WHEN** annotated and ordinary requests are queued together
- **THEN** ordinary requests remain eligible through native rules, preserve their native relative ordering, and require no TokenCake metadata
- **AND** an annotated request does not pass an ordinary candidate that the configured native policy places ahead of it
- **AND** ordinary requests use shared capacity without consuming or borrowing per-agent reservations

#### Scenario: Preserve native priority meaning
- **WHEN** ordinary requests use the native `priority` policy
- **THEN** a lower numerical native priority remains more important and is not interpreted as a TokenCake score

### Requirement: Native preemption and requeue semantics
TokenCake victim selection SHALL integrate with native vLLM preemption semantics. A preempted request SHALL release its KV allocation, reset recomputable progress as required by native vLLM, transition to the native preempted state, and return to the waiting queue rather than terminate.

#### Scenario: Preempt a lower-value running request
- **WHEN** native allocation cannot proceed and TokenCake selects a running request as the victim
- **THEN** the system performs native preemption, requeues the victim, and permits it to resume in a later scheduling step

#### Scenario: Roll back a victim already scheduled in the current step
- **WHEN** TokenCake selects a victim whose tokens or encoder/speculative work were already added to the current scheduler output
- **THEN** the scheduler performs the same running-list, token-budget, new-block, speculative-token, encoder-input, and loop-index rollback as the native allocation-failure branch before native preemption

#### Scenario: Do not expose preemption as completion
- **WHEN** a request is preempted for scheduler capacity
- **THEN** the system emits no terminal completion or failure response solely because of that preemption

#### Scenario: Avoid an unjustified agent preemption
- **WHEN** the requesting work does not exceed the best candidate victim by the fixed TokenCake preemption margin
- **THEN** the scheduler does not displace that candidate on behalf of the requesting work

### Requirement: Data-parallel fail-closed behavior
TokenCake agent scheduling SHALL support only an effective data-parallel world size of one until its mutable scheduling and reservation state has a defined cross-rank consistency protocol.

#### Scenario: Reject unsupported data parallelism
- **WHEN** TokenCake agent scheduling is enabled with an effective data-parallel world size greater than one
- **THEN** engine initialization fails before serving requests with an actionable unsupported-configuration error

#### Scenario: Preserve native data parallelism when disabled
- **WHEN** the effective data-parallel world size is greater than one and both TokenCake scheduling and TokenCake offload are absent or disabled
- **THEN** the system applies the native vLLM data-parallel validation and behavior without a TokenCake-specific failure

### Requirement: Disabled scheduling preserves native scheduler semantics
When TokenCake is absent or agent scheduling is disabled, the system SHALL preserve native vLLM queue construction, scheduling order, admission, allocation, preemption, and request lifecycle. It SHALL create no TokenCake scheduling state and SHALL perform no TokenCake score or reservation work. Native connector construction and behavior SHALL also remain unchanged when TokenCake is absent or both TokenCake switches are disabled; when only scheduling is disabled, the independent TokenCake offload switch MAY select its connector extension.

#### Scenario: TokenCake configuration is absent
- **WHEN** `additional_config` has no `tokencake` object
- **THEN** the scheduler behaves semantically the same as unmodified vLLM for the same configuration, request sequence, and runtime state

#### Scenario: Agent scheduling is explicitly disabled
- **WHEN** structured TokenCake configuration is present but agent scheduling is disabled
- **THEN** native scheduler behavior is preserved and no agent queue, dynamic scoring, reservation, or admission policy is activated
- **AND** connector behavior is determined only by the independent offload setting

#### Scenario: Metadata is present while scheduling is disabled
- **WHEN** a request carries `extra_args["tokencake"]` but TokenCake agent scheduling is disabled
- **THEN** the metadata does not change native scheduling or preemption behavior
- **AND** it affects KV transfer only if the independently configured TokenCake offload switch is enabled
