# TokenCake Experiments

These tools run fixed multi-agent DAG workloads against the current repository.
The native baseline uses the same checkout with TokenCake scheduling and KV
offloading disabled. Historical measurements in the project README retain
their original configurations; new runs record their own code and environment.

The experiment client, datasets, and result analyzer are included here. No
separate TokenCake checkout or particular Git commit is required. GPU indices,
model location, interpreter, CPU affinity, and memory budgets are configurable.

## Setup

Run commands from the repository root. Use an environment compatible with the
installed GPU driver and the current vLLM checkout:

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
uv pip install --python .venv/bin/python httpx aiohttp psutil prometheus-client pydantic numpy matplotlib
```

Pass a local model directory or a Hugging Face model ID with `--model`.
Preparation writes the run configuration without loading weights or checking
CUDA extensions, installed package versions, Git state, or model checksums.
The serving process loads the selected model when the experiment starts.

## Plan and run

The default matrix compares native vLLM, scheduling only, and combined
scheduling/offload at QPS 1.0, 0.5, and 0.1. `--components` selects all four
modes at QPS 1.0, 0.5, 0.2, 0.1, and 0.05, for 20 cases.

```bash
# Planning prints commands without loading a model or inspecting GPUs.
.venv/bin/python -m tools.tokencake_experiments.driver plan \
  --components --model /path/to/model --gpus 0

# Use a fresh output directory for every configuration.
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /path/to/runs/components \
  --components --model /path/to/model --gpus 0 \
  --gpu-memory-utilization 0.5 --max-model-len 32768 \
  --cpu-offload-gib 16 --workload-profile conversation-tools

.venv/bin/python -m tools.tokencake_experiments.driver run-cases \
  /path/to/runs/components
.venv/bin/python -m tools.tokencake_experiments.driver report \
  /path/to/runs/components
```

Choose CPU KV capacity for the available host memory and workload using
`--cpu-offload-gib`. There is no required host-memory size or preflight quota check.

`--gpus` takes physical indices from `nvidia-smi`. One GPU runs all cases
sequentially. Multiple indices distribute comparison modes across GPUs.
GPU discovery and monitoring are optional observations; a missing `nvidia-smi`
or an occupied GPU does not block launching. The serving process reports actual
allocation or device errors. Cases assigned to different GPUs can run concurrently.

CPU affinity is inherited by default. `--cpus 0-7,16-23` optionally binds the
server and client to an allowed CPU set; GPU indices do not imply NUMA nodes.
`--python /path/to/environment/.venv/bin/python` selects the interpreter used
by both the server and the DAG client.
`--dtype` defaults to `auto`; select `float16` or `bfloat16` when needed for
the model and GPU.

A subset can be prepared with explicit modes and QPS:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /path/to/runs/comparison --model /path/to/model --gpus 0 \
  --mode native --mode agent --mode offload-agent --qps 1.0
```

`run` and `run-cases` execute the selected cases once. There is no automatic
threshold-driven repetition or per-case launch cap. Reports retain failed and
interrupted launches and show measured medians and speedups without acceptance
thresholds.
Any positive QPS is accepted by `--qps`.

## Workload and snapshots

The bundled datasets contain 24 applications: 29 nodes per DAG,
27 model calls, and 36 edges. Tools have fixed simulated waits and results;
model outputs are generated during the run and passed to downstream nodes.
See [datasets/README.md](datasets/README.md) for the four input profiles.

`conversation-tools` is the default. `--workload-dataset /path/to/dataset.json`
selects a custom JSON workload. Application count, task name and seed come from
that file; preparation does not require the original count, profile or seed.

Preparation copies the current `dataset.py`, `dataset_client.py`, `analysis.py`,
and selected JSON file to the run's `launcher/` directory. `--snapshot-target`
also copies the current `vllm/` runtime, including native extensions. Without
that option the serving process imports this checkout directly. Both native
and TokenCake modes use the selected runtime.

Run records describe the selected settings and inputs. Source hashes and
identifiers are informational: edits to code, data, configuration or installed
packages do not trigger integrity checks before launching or reporting.

## Optional Mooncake comparison

Mooncake is needed only when explicitly selecting `--mode mooncake`. Install
its compatible runtime in the selected environment and provide a configuration
JSON for the native vLLM Mooncake store connector:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /path/to/runs/mooncake --model /path/to/model --gpus 0 \
  --mode native --mode offload-agent --mode mooncake --qps 1.0 \
  --mooncake-config /path/to/server_mooncake.json \
  --mooncake-master /path/to/environment/.venv/bin/mooncake_master
```

The configuration is copied into the run record. Each Mooncake case starts a
local master on a free port and overrides `master_server_address`. These options
are optional when using an externally configured Mooncake service. Other
connector settings, including store and buffer capacity, come from the supplied
JSON. Ordinary native/TokenCake runs do not import or require Mooncake.

## Analyze and plot

Analyze collected cases, including a subset of modes and QPS values:

```bash
.venv/bin/python -m tools.tokencake_experiments.component_graph \
  /path/to/runs/components/launcher /path/to/runs/components/graph.json
.venv/bin/python -m tools.tokencake_experiments.component_analysis \
  /path/to/runs/components --graph /path/to/runs/components/graph.json \
  --output /path/to/runs/components/analysis.json
.venv/bin/python -m tools.tokencake_experiments.component_matrix \
  /path/to/runs/components/analysis.json /path/to/runs/components/tables
.venv/bin/python -m tools.tokencake_experiments.component_plots \
  /path/to/runs/components/analysis.json /path/to/runs/components/figures
```

The local analyzer verifies all applications and DAG nodes, frozen workload
contracts, generation budgets, terminal status, and retry-free execution.
Reports preserve unavailable metrics as N/A. Tables and plots use the available
cases without requiring 20 groups or matching environment/configuration hashes. CSV/Markdown tables and PNG/PDF figures cover latency, throughput,
KV occupancy, component effects, prefill sources, and application latency CDFs.

Client wall time excludes server startup. First-prefill token sources are not
a count of every recomputation. KV occupancy, GPU hardware utilization, and
useful-compute occupancy are separate metrics. Plot ranges are observed minima
and maxima, not confidence intervals.

## Real agent benchmarks

See [agent_bench/README.md](agent_bench/README.md) for BFCL and mini-swe-agent
workloads that execute real tools and repository edits.
