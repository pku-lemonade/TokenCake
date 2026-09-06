# TokenCake Integration Proposal

## Why

TokenCake's latest agent-aware scheduling and selective KV-cache preservation improve multi-agent DAG makespan, but the implementation in `../vllm_agent` forks older vLLM internals and duplicates CPU-offload machinery. The useful behavior needs to be migrated onto vLLM v0.22.0 with a small, upstream-oriented change that preserves native behavior when disabled and reuses the native CPU-offload connector.

## What Changes

- Add opt-in TokenCake agent scheduling configured only through structured `additional_config["tokencake"]`, while leaving the public vLLM `fcfs` and `priority` scheduler policies unchanged.
- Add a generic `POST /v1/tokencake/events` lifecycle interface and structured request metadata in `vllm_xargs["tokencake"]`, keyed by a client-generated UUID lifecycle ID independent of vLLM's internal request ID.
- Extend v0.22's native CPU-offload connector with event-triggered selective proactive D2H from a bounded completion snapshot, retain selected CPU entries through the manager's existing eviction refcount, and retain native demand-driven H2D, asynchronous transfers, completion fences, cache management, and preemption semantics.
- Keep TokenCake implementation code under `vllm/tokencake/` and add only direct, narrow integration logic to the v0.22 scheduler, OpenAI protocol, EngineCore utility path, connector factory, reset, and metrics surfaces.
- Do not migrate the old custom CPU block pool, swap-map output fields, model-runner transfer path, predictive H2D/upload reservations, MCP-specific EngineCore request types, debug endpoints, or request-terminating preemption behavior.
- Adapt the latest old benchmark launcher through a small, target-owned patch applied to a disposable checkout; keep the old source checkout unchanged for historical comparisons.
- Validate native baseline, target agent-only, target offload-agent, latest-old offload-agent, and current Mooncake behavior with the agreed A800 end-to-end matrix. The revised primary gate requires offload-agent total E2E to improve on native vLLM by at least 25% at every QPS.
- Adapt reservation, aggregate prefill admission, chunking, and victim selection to requests that must resume after preemption. Complete the combined optimization first; defer single-factor ablations until afterward, as confirmed on 2026-09-06.
- Extend logical admission to declared generation budgets and correct preservation-window demand accounting, validating each combined stage with complete DAG execution. The user authorized autonomous strategy iteration and a code commit after each validated stage on 2026-09-06.
- Observe the global imminent-eviction frontier without using it for Phase-1 selection; if Phase 1 misses a performance gate and metrics attribute the miss to delayed preservation of snapshot-known blocks, allow the conditional second-stage frontier/snapshot-intersection hook without replacing native offload or rebuilding the old global lineage graph.

## Capabilities

### New Capabilities

- `tokencake-agent-scheduling`: Structured TokenCake metadata, dynamic agent ordering, adaptive KV-capacity reservation, admission control, and native preemption integration.
- `tokencake-lifecycle-offloading`: Generic lifecycle events and selective proactive D2H layered on v0.22 native CPU offload with demand-driven H2D and bounded lifecycle state.
- `tokencake-performance-validation`: Reproducible launcher adaptation, environment provenance, correctness checks, and end-to-end performance acceptance criteria.

### Modified Capabilities

None. This repository has no existing OpenSpec capabilities.

## Impact

- Runtime code: `vllm/tokencake/`, `vllm/v1/core/sched/scheduler.py`, the existing v1 offloading connector/factory and reset paths, EngineCore utility handling, OpenAI request/event protocol, and bounded metrics.
- Public surface: one opt-in structured `additional_config` namespace, nested JSON support in `vllm_xargs`, and one authenticated lifecycle-event endpoint. No TokenCake-specific CLI flags or public scheduler policy are added.
- Compatibility: with TokenCake absent or both switches disabled, v0.22 creates its native queues and connector unchanged. Disabling only scheduling preserves native queue/order/admission/preemption behavior while allowing the independent TokenCake offload switch. Ordinary requests retain their native request and priority semantics; enabled mixed-workload protection may change when they run under contention.
- Validation systems: two repo-local `uv` environments, the unchanged latest `../vllm_agent` source, the existing `mooncake_agent` runtime, A800 GPU/NUMA affinity, the transferred `agentcodeclean_new.json` workload, and benchmark result/provenance artifacts.
