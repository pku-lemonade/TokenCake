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
commitments. By default, `scheduling.reserve_generation_tokens` commits the known
input plus the declared generation budget; setting it to `false` commits only
the known input. Both use the native multi-group capacity calculation and allocate
KV incrementally. Large unused generation budgets can reduce concurrency.

After checking normal reservation eligibility in agent-score order, the scheduler
can lend otherwise idle reservations to a waiting request that fits the remaining
physical and committed capacity. It still checks full demand before admitting a
borrower. A preempted generation remains pending and resumes to completion.

When decode requests are running, `scheduling.decode_prefill_token_budget` limits
the total annotated prefill work per step (1024 tokens by default). Pure prefill
uses the native token budget. Setting this value to zero disables the additional
limit. Encoder inputs, non-chunked prefill, and Mamba block alignment retain their
native scheduling rules.

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
generation's ordered snapshot. Those GPU cache entries are already free;
copying them to CPU preserves their contents without increasing physical GPU
capacity. Native transfer fences protect their bytes until D2H completes.
H2D occurs only when a later request needs a matching prefix.

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
