# TokenCake

TokenCake adds optional agent scheduling and selective CPU preservation of KV
prefixes during application stalls. Scheduling and preservation are independently
enabled through `additional_config["tokencake"]`. Requests opt in through the
`vllm_xargs` extension of the completion, chat completion, or responses API.

## Server configuration

Enable both components with the native CPU offload backend:

```bash
VLLM_USE_SIMPLE_KV_OFFLOAD=0 vllm serve Qwen/Qwen2.5-14B-Instruct \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --kv-offloading-size 100 \
    --kv-offloading-backend native \
    --additional-config '{"tokencake":{}}'
```

The CPU capacity is specified in GiB and must fit at least one KV block. Offload
requires the native CPU manager and its registered connector; another connector,
custom manager, or simple-offload fallback is rejected at startup. Enabling
either component requires `data_parallel_size=1`.

| Mode | `--additional-config` | Native CPU offload required |
| --- | --- | --- |
| Scheduling and offload | `'{"tokencake":{}}'` | Yes |
| Scheduling only | `'{"tokencake":{"offload":{"enabled":false}}}'` | No |
| Offload only | `'{"tokencake":{"scheduling":{"enabled":false}}}'` | Yes |
| Both disabled | `'{"tokencake":{"scheduling":{"enabled":false},"offload":{"enabled":false}}}'` | No |

Omitting the namespace also disables TokenCake. Unknown configuration keys,
invalid types or ranges, and tuning fields in a disabled section are rejected.
Offload-only mode uses `first_fit` temporal selection. See
[TokenCakeConfig][vllm.tokencake.config.TokenCakeConfig] for the complete schema
and defaults. The legacy agent CLI flags and environment variables do not enable
these components.

Reservations prioritize new admissions. Requests that have already been admitted
continue to grow, and subsequent admissions account for their remaining capacity
commitments. By default, `scheduling.reserve_generation_tokens` commits a
physical-reclaim beneficiary's declared generation budget, with
`scheduling.generation_reserve_mode="reclaim"`.

The `reclaim` mode commits all admitted input growth, and a request
needing physical preemption also commits its remaining generation. Victim
selection seeks enough space for this commitment, and new admissions respect
it until the beneficiary finishes or is preempted. Running requests retain
native physical progress; reservation changes affect subsequent admissions.
`reclaim_beneficiary` counts newly protected beneficiaries.

The optional `progress` mode commits all admitted input growth together with
enough generation capacity for at least one admitted request to finish. Other
requests can extend their KV only while this completion guarantee remains
affordable. Requests that temporarily wait retain their computed KV. If previously
admitted native work already exceeds this bound, a concrete finisher can recover
physical space through native preemption; displaced requests remain pending.

The `all` mode commits every admitted request's declared generation budget.
It is also the conservative fallback for speculative lookahead and cache groups
other than full attention. Setting `reserve_generation_tokens` to `false` commits
only known input. All modes use native multi-group capacity calculations and
allocate KV incrementally. Large unused generation budgets can reduce concurrency
in `all` mode. `generation_progress_deferred` counts temporary growth deferrals,
separately from physical and reservation preemptions.
The `tokencake_critical_growth_wait_max_seconds` gauge tracks the longest
observed critical growth wait, including waits that are still in progress.

After checking normal reservation eligibility in agent-score order, the scheduler
can lend otherwise idle reservations to a waiting request that fits the remaining
physical and committed capacity. It still checks full demand before admitting a
borrower. A preempted generation remains pending and resumes to completion.
Within a similar-importance victim band, selection prefers enough releasable
capacity at a lower recomputation cost and protects requests near completion.
When CPU offloading is enabled, this cost excludes the currently restorable
prefix, using native cache-group alignment. The estimate is read-only and grants
no credit for pending transfers. CPU cache eviction before readmission can still
require native recomputation; the request remains pending until it completes.
For full attention without lookahead or encoder inputs, requests first check
committed capacity using native multi-group demand and GPU prefix hits. A
request that cannot fit waits before querying or touching the CPU cache. Once
capacity becomes available, native lookup and asynchronous restoration proceed.
During normal admission, a request may also borrow an idle reservation when its
effective score exceeds every waiting owner it borrows from by at least
`scheduling.priority_borrow_score_margin` (500 points by default). This prevents
a lower-priority owner from taking capacity that an important request needs to
advance. Setting the margin to zero disables this early borrowing exception.

When decode requests are running, `scheduling.decode_prefill_token_budget` limits
the total annotated prefill work per step if set to a positive value. Its default
is zero, preserving the native token budget. The full 24-DAG QPS 1.0 experiment
with a 1024-token limit regressed E2E. Pure prefill, encoder inputs, non-chunked
prefill, and Mamba block alignment retain their native scheduling rules.

Under high committed-capacity pressure, annotated requests within
`scheduling.cache_affinity_score_band` (500 score points by default) can prefer
prefixes held by running requests. This reduces incremental physical KV demand
within bounded importance bands. Shared blocks must at least cover the new
capacity demand, so short common headers alone do not change ordering.
Larger score differences keep their order;
ordinary-request barriers and all admission checks still apply. Setting the band
to zero disables this preference. GPU shared/exclusive hit-block counters separate
concurrent sharing from reuse of free cache entries.

By default, `scheduling.inherit_join_priority` gives live branches with the same
positive `application_started_at_s` and nonempty `join_group` their group's
highest live agent score. This lets unfinished branches advance alongside a
critical peer toward the same DAG join. Original agent scores break ties within
the group. Inheritance expires when the high-scored peer leaves the scheduler;
missing group metadata and disabled inheritance preserve individual scores.

## Request metadata

Generate a fresh ID of the form `tc-` followed by lowercase UUID4 hex for every
generation attempt, including retries. Supply it both as the top-level
`request_id` and as `vllm_xargs.tokencake.lifecycle_id`:

```json
{
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "request_id": "tc-4c3d2e1f001142228333abcdef123456",
  "prompt": "Write a Python function that sums a list of integers.",
  "max_tokens": 128,
  "vllm_xargs": {
    "tokencake": {
      "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456",
      "agent_type": "programmer",
      "importance": 3.0,
      "offload_eligible": true,
      "reusable_prefix": true
    }
  }
}
```

The example ID illustrates the format; generate a new one for each attempt.
Keep the input ID for subsequent events: the completion API's response ID has
its own native prefix. An annotated request must expand into exactly one
EngineCore generation, so multiple prompts, `n > 1`, beam search, and responses
API built-in tools that can generate additional model calls are rejected.
Streaming a single generation is supported.

Only `lifecycle_id` is required inside the metadata namespace. Known fields are
validated; unrecognized nested JSON fields are preserved and ignored by
TokenCake. The graph and application timing fields are defined by
[TokenCakeMetadata][vllm.tokencake.protocol.TokenCakeMetadata]. Preservation
requires both `offload_eligible` and `reusable_prefix` to be true.

Unannotated requests keep native ordering and prefill behavior. When the
TokenCake connector is enabled, they also retain native automatic CPU store and
load behavior. Annotated requests use lifecycle-selected stores; scheduling-only
clients do not send lifecycle events.

## Lifecycle events

After a successful generation, POST the following body to
`/v1/tokencake/events` before starting the tool or other blocking work:

```json
{
  "event": "stall_started",
  "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456",
  "kind": "file_write",
  "estimated_duration_s": 5.0
}
```

Wait for the successful HTTP response before starting that work. After it
finishes, POST the following body to the same endpoint and wait for its response
before submitting a successor generation:

```json
{
  "event": "stall_finished",
  "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456"
}
```

Events use the server's normal API-key authentication. Treat non-2xx responses
as failures. A successful start acknowledges the core state transition and any
scheduler-side transfer registration; asynchronous D2H may still be pending.
Identical repeated starts and finishes are idempotent while their lifecycle
record remains available. Conflicting starts return 409, unknown IDs return
404, malformed bodies return 400, and disabled or unavailable offload returns
503. Internal execution failures return a sanitized 500 response.

`kind` defaults to `stall`. Omit `estimated_duration_s` when no estimate is
available; an explicit value must be positive and finite. Prediction uses the
estimate or duration history when only one is available, averages them when
both are available, and defaults to one second when neither is available.
A start may arrive before generation completes, but preservation can begin only
after the completed request releases its GPU blocks. Aborted or failed
generations cannot be reactivated by a later event.

## Cache ownership and observation

Preservation selects a bounded set of valid blocks from the completed
generation's ordered snapshot. Full-attention blocks may still be shared by
another running request: the completed prefix remains immutable, and copying
it does not change GPU ownership. Other cache types require free GPU entries.
Copying preserves contents without increasing physical GPU capacity. Native
transfer fences protect the source bytes against reuse until D2H completes.
H2D occurs only when a later request needs a matching prefix.

The default `offload.max_relief_blocks=0` selects an aligned prefix within the
available CPU capacity and predicted tool window. Estimated D2H plus H2D time
must leave at least twice that transfer time, or 50 milliseconds, in the window.
Blocks already on CPU still count toward the restore estimate. A positive value
keeps an explicit block cap, also bounded by the current waiting demand. Native
multi-group alignment and transfer fences apply to both modes.

The lifecycle temporarily retains ready CPU keys using the native manager's
eviction references. Finish, expiry, or predicted ownership release drops those
references without evicting cached data. Start must attach within 60 seconds of
generation completion. Active stalls have a safety deadline of four times the
predicted duration, bounded to 60 seconds through one hour; terminal records
remain for 60 seconds for duplicate and late-event handling.

Existing prefix-cache reset semantics apply. A local reset invalidates detached
snapshots; a successful external reset also clears retained ownership and
terminalizes lifecycles. Neither reset clears duration history or cumulative
metrics.

`/metrics` exposes bounded TokenCake lifecycle, decision, saved-block, and
scheduling counters alongside native offload transfer metrics. Successful event
responses alone do not establish that data was saved or reused. Check completed
saved blocks and native D2H/H2D metrics for those outcomes.

Real-model validation currently covers a single GPU with `tp=1`, `pp=1`.
Native multi-group and sliding-window transfers also have CUDA data-integrity
coverage; model-level TP/PP execution and performance remain unvalidated.
