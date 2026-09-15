# TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2510.18586-b31b1b)](https://arxiv.org/abs/2510.18586)
[![Venue](https://img.shields.io/badge/EuroSys-27-4b6cb7)](https://arxiv.org/abs/2510.18586)

TokenCake targets multi-agent workloads, in which an application usually
alternates between model calls and external tools. Every new call carries more
of the conversation history, while an agent waiting for a tool may hold a large
KV cache that is temporarily inactive. Under concurrency, these workloads
create both spatial contention (useful prefixes are evicted) and temporal
under-utilization (idle prefixes occupy GPU memory).

TokenCake co-optimizes scheduling and KV-cache management around the agent
workflow:

- **Spatial scheduling.** Admission accounts for the input and declared future
  generation of accepted requests. A hybrid priority uses agent importance,
  dependency-graph structure, runtime progress, and prefix affinity. Dynamic
  reservations protect critical-path agents while allowing idle capacity to be
  borrowed by other agent types.
- **Temporal scheduling.** Tool lifecycle events identify periods in which a
  prefix is likely to be reused. TokenCake opportunistically preserves eligible
  GPU KV blocks in CPU memory and restores them for a later call, reducing
  prefill work after a long tool stall.
- **Implementation on vLLM.** TokenCake reuses vLLM's PagedAttention,
  prefix-cache, chunked-prefill, CPU-offload, and asynchronous transfer paths;
  the model execution kernels are unchanged.

The paper describes predictive uploading to hide transfer latency. The current
implementation uses vLLM v0.22.0 and performs event-driven preservation with
demand-triggered CPU-to-GPU recovery; predictive prefetching and further policy
tuning remain active development work.

## Paper and project status

TokenCake is described in the paper [“TokenCake: A KV-Cache-centric Serving
Framework for LLM-based Multi-Agent Applications”](https://arxiv.org/abs/2510.18586),
which has been accepted to **EuroSys '27**.

The paper's evaluation used an earlier vLLM baseline. Its headline results are
therefore historical paper results: more than **47.06% lower end-to-end
latency** and up to **16.9% higher effective GPU-memory utilization** relative
to vLLM. This repository contains an ongoing migration of the implementation
and experiments to target **vLLM v0.22.0**. The measurements below are from
that implementation and should be interpreted separately from the paper's
original numbers.

## Results from the v0.22.0 migration

### Static multi-agent DAG workload

The workload contains 24 complete DAGs (648 model calls and 155,136 generated
tokens). Each DAG has 29 nodes and 36 dependency edges and includes planning,
parallel code generation, review, repair, and joins. We compare native vLLM
(`base`) with the combined TokenCake configuration (`agent_offload`) at the
same offered application rate. Times are wall-clock seconds; the reduction is
`1 - TokenCake / native`.

| Offered QPS | Native E2E | TokenCake E2E | E2E reduction |
| ---: | ---: | ---: | ---: |
| 0.05 | 875.74 | **676.86** | **22.71%** |
| 0.10 | 940.76 | **671.28** | **28.65%** |
| 0.20 | 938.79 | **640.45** | **31.78%** |
| 0.50 | 938.99 | **648.05** | **30.98%** |
| 1.00 | 960.44 | **631.77** | **34.22%** |

At 1.0 QPS, application P95 latency decreased from 929.12 s to 604.50 s
(**34.94%**). In the same run, first-pass prefill work decreased from
3,282,597 to 952,780 tokens (**70.97%**), input-cache reuse increased from
44.12% to 83.78%, request queueing decreased from 62.97 s to 28.78 s, and the
combined configuration experienced zero preemptions (41 for native vLLM).
The prefill figure counts the first input processing for each call; it does not
claim that all possible recomputation decreased by 70.97%.

The QPS 1.0 component comparison shows how the two mechanisms contribute:

| Configuration | E2E | Reduction vs. native | First-pass prefill tokens | Preemptions |
| --- | ---: | ---: | ---: | ---: |
| Native (`base`) | 960.44 s | — | 3,282,597 | 41 |
| Scheduling only (`agent`) | 837.78 s | 12.77% | 2,705,768 | 0 |
| Tool-window preservation only (`offload`) | 691.01 s | 28.05% | 1,039,577 | 58 |
| Combined (`agent_offload`) | **631.77 s** | **34.22%** | **952,780** | **0** |

The component improvements overlap and are not additive. In the combined run,
GPU-to-CPU and CPU-to-GPU transfer jobs took 2.105 s and 13.656 s in aggregate;
these jobs can overlap with model execution. Both offload configurations fit in
the configured 100 GiB CPU KV pool without capacity rejection or transfer
failure.

### Agent coding workload

We also ran the first 20 SWE-bench Verified tasks with mini-swe-agent 2.4.6.
Each comparison used the same task set, prompts, and 1,800 s time budget, but
the agents executed real repository edits and tests, so their subsequent
requests and total generated work followed different trajectories.

| Metric | Native vLLM | TokenCake | Change |
| --- | ---: | ---: | ---: |
| Output throughput (token/s/GPU) | 46.27 | **59.02** | **+27.6%** |
| Mean inter-token interval | 46.80 ms | **34.25 ms** | **−26.8%** |
| Mean TPOT | 51.73 ms | **36.55 ms** | **−29.3%** |
| Preemptions | 11 | **0** | — |
| Peak GPU KV usage | 100% | **91.83%** | −8.17 percentage points |
| Mean request queueing | 18.06 s | 34.65 s | increased |

Input-cache reuse in this run changed only from 9.97% to 10.39%; 123,472 input
tokens were recovered from CPU KV. Thus the coding result should be read as a
throughput and scheduling measurement for this run, rather than as evidence
that CPU KV alone explains the gain. The different agent trajectories also
mean that this number is not an equal-workload acceleration factor.

### Shared experimental setup

| Setting | Value |
| --- | --- |
| Model | Qwen2.5-14B-Instruct, BF16 |
| GPU | NVIDIA A800-SXM4-80GB, TP=1, PP=1 |
| GPU memory utilization / maximum context | 0.5 / 32,768 tokens |
| GPU KV pool | 3,050 blocks, 16 tokens per block |
| Batch token budget | 8,192 (native chunked prefill retained) |
| CPU KV capacity | 100 GiB for offload configurations |

These are development measurements from a finite campaign. The static DAG
experiments use repeated runs only at selected QPS points, and the coding
agents produce non-identical traces. Use the scripts and frozen workloads in
[`tools/tokencake_experiments`](tools/tokencake_experiments) when collecting new
measurements.

## Quick start

Install the repository in a vLLM-compatible `uv` environment:

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
```

The current vLLM-based implementation exposes TokenCake through
`additional_config`. Launch the vLLM V1 server; the following example enables
both TokenCake scheduling and lifecycle-based offload:

```bash
VLLM_USE_SIMPLE_KV_OFFLOAD=0 \
  .venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --additional-config '{"tokencake":{}}' \
  --kv-offloading-size 100 \
  --kv-offloading-backend native
```

The two components can be enabled independently. Use these argument fragments
with the command above:

```bash
# Scheduling only
--additional-config '{"tokencake":{"offload":{"enabled":false}}}'

# Lifecycle-based KV preservation only
--additional-config '{"tokencake":{"scheduling":{"enabled":false}}}' \
--kv-offloading-size 100 --kv-offloading-backend native
```

Requests that participate in agent-aware scheduling carry metadata in
`vllm_xargs.tokencake`. The `lifecycle_id` must be a `tc-` prefixed UUID4 and
must match the top-level `request_id`:

```json
{
  "request_id": "tc-123e4567e89b42d3a456426614174000",
  "vllm_xargs": {
    "tokencake": {
      "lifecycle_id": "tc-123e4567e89b42d3a456426614174000",
      "agent_type": "coder",
      "importance": 1.0,
      "critical_path": true,
      "join_group": "application-7"
    }
  }
}
```

The client reports tool stalls to `/v1/tokencake/events` so the temporal
policy can evaluate a preservation window:

```json
{
  "event": "stall_started",
  "lifecycle_id": "tc-123e4567e89b42d3a456426614174000",
  "kind": "tool",
  "estimated_duration_s": 10.0
}
```

Send `{"event":"stall_finished", "lifecycle_id":"..."}` when the tool
returns. Offload requires the native CPU offload backend, a positive
`--kv-offloading-size`, and `VLLM_USE_SIMPLE_KV_OFFLOAD=0`. The current
implementation also requires `data_parallel_size=1`.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`vllm/tokencake`](vllm/tokencake) | Configuration, request metadata, lifecycle events, scheduling, offload policy, and metrics |
| [`tests/tokencake`](tests/tokencake) | Unit, serving, lifecycle, scheduling, and transfer tests |
| [`tools/tokencake_experiments`](tools/tokencake_experiments) | Frozen workloads, launchers, campaign drivers, analysis, and reports |
| [`tools/tokencake_experiments/datasets`](tools/tokencake_experiments/datasets) | Checked-in DAG and agent workload definitions |

The main integration points are the V1 scheduler, the native CPU-offloading
connector, the OpenAI-compatible serving routes, and the EngineCore lifecycle
channel. Requests without TokenCake metadata continue to use the native vLLM
ordering and prefill behavior.

For hardware setup, provenance checks, frozen datasets, and the full acceptance
campaign, see the [experiment README](tools/tokencake_experiments/README.md).

## Citation

If you use TokenCake, please cite:

```bibtex
@article{bian2025tokencake,
  title={Tokencake: A kv-cache-centric serving framework for llm-based multi-agent applications},
  author={Bian, Zhuohang and Wu, Feiyang and Li, Zhuoran and Ma, Teng and Zhuo, Youwei},
  journal={arXiv preprint arXiv:2510.18586},
  year={2025}
}
```

The current TokenCake implementation uses vLLM and retains the upstream
project's Apache-2.0 license. See [vLLM](https://github.com/vllm-project/vllm)
for upstream documentation and contribution guidance.
