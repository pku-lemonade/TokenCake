# Real Agent Benchmarks

This directory runs BFCL multi-turn tool calling and mini-swe-agent tasks with
local SWE-bench grading. Model requests and tools execute during each run.
Both the native baseline and TokenCake load the current repository; the baseline
disables TokenCake scheduling and KV offloading.

## Environment and inputs

Run commands from the repository root. Follow the
[experiment setup instructions](../README.md#setup) for the serving environment.
Choose a separate data directory, shown below as `/path/to/agent-bench`, and
place the sources for the benchmarks you use under that directory:

| Source directory | Used for |
| --- | --- |
| `sources/gorilla` | BFCL task data, reference answers, Qwen handler, and grader |
| `sources/mini-swe-agent` | SWE agent loop and XML prompt configuration |
| `sources/SWE-bench` | SWE grading implementation |
| `sources/swe-bench-tasks` | SWE-bench Verified tasks and environment recipes |

Install the corresponding agent, tokenizer, and grading dependencies in the
interpreter that runs them. Copying source directories alone does not install
mini-swe-agent or SWE-bench. Interpreter selection is:

- Model service: `--python`, defaulting to the interpreter running `inputs.py`.
- Agent workers and BFCL grading: `<platform>/.venv/bin/python`, falling back to
  the current interpreter when that executable is absent.
- SWE preparation and grading: `<platform>/environments/grading/.venv/bin/python`,
  then `<platform>/.venv/bin/python`, then the current interpreter.

Separate environments are optional. The manifest records inputs and settings;
preparation does not require dependency lockfiles, a clean Git checkout, model
checksums, or manual confirmation. Prepare only the data and runtime dependencies
for the benchmark you use.

```bash
.venv/bin/python -m tools.tokencake_experiments.agent_bench.inputs \
  --platform /path/to/agent-bench \
  --output /path/to/agent-bench/manifests/run-01 \
  --model /path/to/model --gpus 0 \
  --python /path/to/server/.venv/bin/python \
  --cpu-offload-gib 16 --gpu-memory-utilization 0.5 \
  --seed 42 --workers 8 --execution sequential
```

`--model` accepts a local model directory or a Hugging Face model ID. Manifest
preparation does not check the local model cache. Workers use the selected model
for tokenization and share the service's `--max-model-len` setting. The BFCL
adapter uses the official Qwen handler and its prompt format, so select a
compatible model. No particular source revision, GPU model, or original machine
path is required.

Available modes use the following names in `--modes`:

| Mode | TokenCake scheduling | CPU KV offloading |
| --- | --- | --- |
| `base` | Disabled | Disabled |
| `agent` | Enabled | Disabled |
| `offload` | Disabled | Enabled |
| `agent_offload` | Enabled | Enabled |

`--execution sequential --gpus 2` assigns all modes to the selected GPU.
`--execution parallel_pair --gpus 1 3` assigns `base` and `agent_offload` to GPUs
1 and 3 respectively. Parallel execution supports up to two modes per invocation,
using distinct GPUs from the manifest's `pilot.gpu_by_mode` mapping.
GPU indices follow the physical indices reported by `nvidia-smi`; each service
uses one GPU with tensor and pipeline parallel sizes of one. CPU affinity is
inherited unless `--cpus` is supplied. `--dtype` defaults to `auto` and can be set
to `float16` or `bfloat16`. Parallel services share host memory, including their
CPU KV pools.

## SWE task environments

SWE tasks execute in local task environments. The preparation command adapts the
official Python, dependency, and test recipes and saves a snapshot of each
prepared repository and environment. Existing directories with the same layout
can also be used: each task directory contains `repo/`, `.venv/`, and
`recipe.json`. Snapshots are restored before execution and grading when present;
preparation status and hashes are not launch prerequisites. The agent receives
the problem statement; reference patches, grading scripts, and target test lists
are not included in its task input.

Prepare a separate set of task directories for each mode in the example:

```bash
for mode in base agent_offload; do
  .venv/bin/python -m tools.tokencake_experiments.agent_bench.environments \
    --platform /path/to/agent-bench \
    --manifest /path/to/agent-bench/manifests/run-01/manifest.json \
    --directory "/path/to/agent-bench/environments/run-01/$mode" \
    --output "/path/to/agent-bench/preparation/run-01/$mode" \
    --setup-jobs 2 --subset pilot --prepare-only
done

# Optional: prepare a separate grading environment and run reference diagnostics.
.venv/bin/python -m tools.tokencake_experiments.agent_bench.environments \
  --platform /path/to/agent-bench \
  --manifest /path/to/agent-bench/manifests/run-01/manifest.json \
  --directory /path/to/agent-bench/environments/run-01/grading \
  --output /path/to/agent-bench/preparation/run-01/grading \
  --setup-jobs 2 --subset pilot
```

Some historical tasks use Conda recipes. Select
`--environment-manager miniconda` and make the chosen `conda` executable available
on `PATH`. Container paths in the recipes are translated to task-local paths;
the host supplies native system libraries. Without `--prepare-only`, preparation
also tests the original version and reference fix. These results are diagnostic
and do not gate subsequent experiments.

## Execution, grading, and reporting

Use fresh output directories for preparation, execution, and grading. For SWE,
`--environment-root` identifies the parent containing `<mode>/<task_id>/`.

```bash
.venv/bin/python -m tools.tokencake_experiments.agent_bench.experiment \
  --platform /path/to/agent-bench \
  --manifest /path/to/agent-bench/manifests/run-01/manifest.json \
  --output /path/to/agent-bench/results/run-01 \
  --benchmark swe --subset pilot --modes base agent_offload \
  --environment-root /path/to/agent-bench/environments/run-01 \
  --budget-seconds 1800 \
  --budget-ledger /path/to/agent-bench/results/budget.jsonl

# Reuse each mode's task environments for grading.
.venv/bin/python -m tools.tokencake_experiments.agent_bench.score \
  --platform /path/to/agent-bench \
  --experiment /path/to/agent-bench/results/run-01 \
  --output /path/to/agent-bench/grading/run-01

.venv/bin/python -m tools.tokencake_experiments.agent_bench.report \
  --experiment /path/to/agent-bench/results/run-01 \
  --scores /path/to/agent-bench/grading/run-01/scores.json \
  --output /path/to/agent-bench/results/run-01/report.json
```

To use the optional separate grading environments above, pass
`--environment-root /path/to/agent-bench/environments/run-01/grading` to `score`.
That directory contains `<task_id>/` directly, without a mode subdirectory.

For BFCL, use `--benchmark bfcl` and omit `--environment-root`; SWE task
environments are unnecessary. The input collector recognizes `multi_turn_base`
and `multi_turn_long_context` when their data files are present. The `pilot`
subset deterministically selects up to two tasks per BFCL category or twenty SWE
tasks using `--seed`. The `full` subset reads all tasks from the selected data
files. BFCL grading also reads the corresponding files under `possible_answer/`.

Tasks use closed-loop arrivals with `--workers` concurrent workers per mode.
`--budget-seconds` limits the invocation's measurement time: sequential modes
share that allowance, while a parallel pair shares one wall-clock interval.
The ledger also tracks cumulative measurement time against
`pilot.measurement_budget_s` in the manifest, which defaults to 43,200 seconds
(12 hours) and can be edited. Service startup, environment preparation, and
offline grading are outside the measurement window. The ledger can be reused
after interruption; it counts recorded completed intervals.

Each task's `terminal.json` records its final status, and `journal.jsonl` records
model requests and tool events. Quality calculations include every selected
task; paired latency comparisons use tasks resolved by both modes. Runs can
produce different trajectories and amounts of work, so token throughput gains
alone do not measure speedup for equal work. Server throughput uses cumulative
token counter differences over the measurement window. Missing metrics remain
unavailable, and missing task scores are recorded as `score_missing`.

GPU occupancy, resource quotas, and changes to code or dependencies do not gate
launching. GPU discovery and metric collection are optional observations.
Grading and reporting do not require matching manifest or input hashes.

## Modules

| Module | Purpose |
| --- | --- |
| `inputs.py` | Record selected inputs and runtime settings |
| `experiment.py`, `parallel.py`, `budget.py` | Sequential or parallel execution, concurrency, and measurement budgets |
| `service.py`, `observe.py` | Serve models from the current repository and collect system metrics |
| `worker.py`, `transport.py` | Task processes, requests, tool events, timeouts, and logs |
| `bfcl.py`, `mini.py` | Adapt the official agent workflows |
| `environments.py`, `swe_local.py` | Prepare task environments, restore snapshots, and adapt test recipes |
| `score.py` | Grade completed runs offline |
| `report.py`, `report_metrics.py`, `throughput.py` | Report quality, latency, throughput, and runtime diagnostics |
| `smoke.py` | Optional live service protocol check using the same model and GPU options as input preparation |
