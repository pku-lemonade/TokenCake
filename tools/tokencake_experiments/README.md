# TokenCake Migration Experiments

This target-owned tooling records the inputs to
`integrate-tokencake-offload-agent`. Runtime preflight and GPU transfer tests
are supporting evidence. Only the specified complete 24-DAG cases qualify
correctness and performance acceptance.

## Environments

Both repo-local uv environments use Python 3.12.9 from the accepted
`/root/autodl-tmp/conda_envs/mooncake_agent` environment. Its packages are a
read-only base; overrides are installed locally. This avoids duplicating the
CUDA package stack on the small root filesystem and never installs into the
original environment. The source environment overrides Torch and its
version-sensitive dependencies independently of the target environment.

Run the following from the target repository:

```bash
uv venv .venv --python /root/autodl-tmp/conda_envs/mooncake_agent/bin/python --system-site-packages
uv pip install --python .venv/bin/python -r requirements/lint.txt
uv pip install --python .venv/bin/python 'ninja==1.13.2'
uv pip install --python .venv/bin/python --no-deps /root/autodl-tmp/mooncake-api-check.TdlTNm/mooncake_transfer_engine-0.3.8-cp312-cp312-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/root/autodl-tmp/vllm-wheels/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl uv pip install --python .venv/bin/python --no-deps -e . --torch-backend=auto
.venv/bin/pre-commit install
export PATH="$PWD/.venv/bin:$PATH"

uv venv .venv/source/.venv --python /root/autodl-tmp/conda_envs/mooncake_agent/bin/python --system-site-packages
uv pip install --python .venv/source/.venv/bin/python --no-deps --torch-backend cu124 'torch==2.6.0' 'torchaudio==2.6.0' 'torchvision==0.21.0' 'triton==3.2.0' 'compressed-tensors==0.9.3' 'depyf==0.18.0' 'llguidance==0.7.30' 'lm-format-enforcer==0.10.11' 'xgrammar==0.1.18' 'numba==0.61.2' 'llvmlite==0.44.0' 'nvidia-cusparselt-cu12==0.6.2' 'sympy==1.13.1'
```

Source server/client commands must set their working directory and
`PYTHONPATH` to the unchanged source checkout. The source's transferred native
extension is retained and checked by the real D2H/H2D kernel tests. The target
uses the frozen official v0.22 wheel's native extensions.

The explicit local Mooncake wheel installation also prevents a legacy
`engine.cpython-312-x86_64-linux-gnu.so` in the original environment from
shadowing Mooncake 0.3.8's `engine.so`.

## Official Baseline

```bash
git worktree add --detach .venv/baseline 0b3ba88f1
unzip -n /root/autodl-tmp/vllm-wheels/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl 'vllm/*.so' 'vllm/_version.py' -d .venv/baseline
```

Run baseline commands with `.venv/bin/python`, working directory and
`PYTHONPATH` pointing at `.venv/baseline`. Check its commit and clean tracked
worktree before each launch. Supply no TokenCake configuration, scheduler
override, or KV-offload option.

## Provenance

```bash
.venv/bin/python -m tools.tokencake_experiments.provenance \
  --model /root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  --original-environment /root/autodl-tmp/conda_envs/mooncake_agent \
  --mooncake-wheel /root/autodl-tmp/mooncake-api-check.TdlTNm/mooncake_transfer_engine-0.3.8-cp312-cp312-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl \
  --vllm-wheel /root/autodl-tmp/vllm-wheels/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl \
  --output /path/to/new/provenance.json

.venv/bin/python -m tools.tokencake_experiments.verify_environment \
  --role target --checkout . --output /path/to/new/target-runtime.json
.venv/source/.venv/bin/python -m tools.tokencake_experiments.verify_environment \
  --role source --checkout ../vllm_agent --output /path/to/new/source-runtime.json
.venv/bin/python -m tools.tokencake_experiments.verify_environment \
  --role baseline --checkout .venv/baseline --output /path/to/new/baseline-runtime.json
```

Evidence files are created exclusively so that a later command cannot
overwrite an earlier record. The collector compares source helper bytes to
the frozen source commit, checks the accepted dataset and Mooncake wheel
hashes, and hashes every model shard referenced by the safetensors index.

## Stage 1 Validation

The target native CPU/GPU worker tests passed all 24 cases on GPU0, including
both transfer directions, shared host memory, multi-group layout and partial
blocks:

```bash
CUDA_VISIBLE_DEVICES=GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f .venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_gpu_worker.py -v --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-1-gpu-transfers.xml
```

From the source checkout, both source multi-layer transfer tests passed on GPU1:

```bash
CUDA_VISIBLE_DEVICES=GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087 PYTHONDONTWRITEBYTECODE=1 /root/TokenCake/vLLM-TokenCake-upstream/.venv/source/.venv/bin/python -m pytest tests/kernels/test_cache.py::test_swap_blocks_multi_d2h_h2d_layers_and_components tests/kernels/test_cache.py::test_swap_blocks_multi_fragmented_uva_kernel -p no:cacheprovider -v --junitxml=/root/TokenCake/vLLM-TokenCake-upstream/openspec/changes/integrate-tokencake-offload-agent/evidence/stage-1-source-transfers.xml
```

These tests compare actual transferred tensor contents. They establish the
native data-path prerequisites and do not establish full-DAG correctness or
any performance gate. Source and baseline tracked worktrees remained clean.

## Stage 2 Validation

See the change's `evidence/stage-2-validation.md` for commands and results.
The model-backed HTTP tests use the full local Qwen2.5-14B-Instruct weights
and compare actual generated text through completion, chat, and responses
routes under all four enablement combinations. They establish API and
connector-construction correctness for this stage; the accepted 24-DAG
performance matrix remains a separate requirement.

The v0.22 device-mapping implementation requires integer CUDA device indices
when importing model kernels. Resolve each frozen GPU UUID to its current
index before launching a server and verify the actual process UUID afterward.
The stage-2 run used index 0, verified against the frozen GPU0 UUID.
Runtime kernel compilation also requires the repo-local `ninja` executable
on `PATH`; installing precompiled vLLM extensions alone does not supply it.

## Acceptance Campaign

This target-owned driver implements the fixed acceptance matrix in
`openspec/changes/integrate-tokencake-offload-agent`. It requires the two declared
A800s, their recorded NUMA CPU sets, the local 14B model, the frozen source
checkout, and the isolated target/source environments described in that change.

```bash
.venv/bin/python -m tools.tokencake_experiments.driver plan
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /root/autodl-tmp/tokencake-acceptance/phase-1
.venv/bin/python -m tools.tokencake_experiments.driver run \
  /root/autodl-tmp/tokencake-acceptance/phase-1
.venv/bin/python -m tools.tokencake_experiments.driver report \
  /root/autodl-tmp/tokencake-acceptance/phase-1
```

`plan` only renders the fifteen initial cases. `prepare` verifies hardware,
packages, extensions, model shards, dataset and repository state, materializes
disposable launchers, and checks that target and source environments construct
the identical 24-DAG workload and arrival traces. The destination must be new.
There are no workload truncation or smoke options in this driver.

For an uncommitted optimization candidate, `--snapshot-target` freezes the
runtime files, native extensions, and patch in a new campaign directory.
The baseline checkout remains the native reference. A diagnostic subset can
still execute the full 24-DAG workload at one QPS:

```bash
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /root/autodl-tmp/tokencake-optimization/candidate \
  --snapshot-target --mode native --mode agent --mode offload-agent --qps 1.0
.venv/bin/python -m tools.tokencake_experiments.driver run-cases \
  /root/autodl-tmp/tokencake-optimization/candidate
```

`run-cases` launches each prepared identity once and retains the normal
correctness, isolation, and provenance checks. It does not launch automatic
repeats or unprepared QPS points. Partial reports list missing comparisons and
cannot pass the complete acceptance matrix. Newly prepared campaigns require
25% native-relative E2E reduction; older frozen campaigns keep their original
threshold. Optional `--gpu 0` or `--gpu 1` creates an explicit placement identity;
omitting it preserves the historical identity representation.

The optional `--workload-profile continuation` applies `continuation.patch`
to disposable launchers for an explicitly requested native/TokenCake subset.
It removes the second copy of the role instruction on the five same-role
validation-to-repair edges. Each DAG still has 27 model calls, the same output
budgets, tool windows, graph structure, dataset and arrivals. The frozen
contract records `input_composition="same-role-prefix-v1"`; launcher hashes
also include the revised composer. Source and reference environments must
construct matching revised contracts. The source checkout remains unchanged.

```bash
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /root/autodl-tmp/tokencake-optimization/continuation \
  --snapshot-target --workload-profile continuation \
  --mode native --mode offload-agent --qps 1.0
```

This profile requires a fresh native baseline on the same revised workload.
Old-workload timing is not a valid reference for its percentage improvement.
Generated downstream inputs can vary with model and tool outputs, so equality
is established for the workload construction rules, initial inputs, declared
budgets and arrival trace, rather than every generated downstream byte.

`--workload-profile conversation` applies `conversation.patch` instead. Every
model node appends its role instruction after the existing conversation. Joins
deduplicate the common chunk prefix and append branch results in predecessor
name order. Tools with model successors keep their generated output before the
tool result and declare that prefix reusable and eligible for preservation.
The policy still evaluates actual waiting pressure and tool duration before
copying. The graph, call count, output budgets and tool times are unchanged.
Its contract is `input_composition="append-only-conversation-v1"`, and its own
fresh native baseline is required. This profile can have longer downstream
inputs because it retains tool-call text that older profiles discarded.

`--workload-profile conversation-tools` adds `tool-budget.patch` after the
conversation patch. It limits the 13 tool-instruction calls per DAG to 128
output tokens. Planning and review retain 200 tokens; code validation and
repair retain 400 tokens. The full 24-DAG workload still executes 648 model
calls, with 155136 total output tokens instead of 271200. Its contract records
`tool_instruction_max_tokens=128`. Tool durations, graph, arrivals and initial
repository context are unchanged. Both native and TokenCake use this budget;
percentage improvements require a new native baseline on this exact profile.

`run` starts a fresh server per case and executes the two independent queues.
Only affected comparison members receive repeats, including excluded launches
in the three-launch maximum. Mooncake has one launch per QPS. Both clients and
servers use their assigned GPU/NUMA affinity. Ctrl-C, SIGTERM and SIGHUP stop
the queues and clean up launched process groups. A campaign lock prevents two
drivers or a reporting process from modifying a live launch ledger. A resumed
campaign retains interrupted launches as exclusions and verifies its frozen
inputs before continuing.

Environment identity includes the read-only parent package stack as well as
repo-local overrides. Frozen JSON inputs are hashed and checked before every
launch, and recorded application arrivals must match the source-generated
trace for that QPS.

When a correctness repair interrupts a campaign before any qualifying result,
`prepare NEW_DIRECTORY --prior-exclusions OLD_DIRECTORY` carries its excluded
launch charges forward. Each original result identity and artifact path stays
intact; only its budget attribution points at the repaired implementation.
The phase, mode, QPS, workload, configuration and environment must match.
Qualifying results cannot use this recovery path. Initial queues still start
at high QPS, with the carried launches deducted from their existing limits.

By default, the target client applies `launcher.patch` to exact source commit
`7a608a4e53ea990b2540c93b4d28cb795b905109`; only request/event protocol changes
are applied. The wrapper aliases the relocated tokenizer import to the target
vLLM tokenizer package. The source DAG execution and analyzer stay unchanged.
Latest-old runs use the original checkout and source environment. The current
Mooncake reference uses the target environment and the unpatched source client
with its existing notification-disable flag.

Mooncake configuration is read from frozen tokencake-mooncake commit
`696c9a14f30ffeacda1707e8f712b7a214460be6`, with a fresh local master address,
a 100 GiB embedded store and a 1 GiB local buffer, using TCP/P2PHANDSHAKE. The
source configuration's 64 MiB example pool is unsuitable for the full workload;
the old helper's 1 TiB default exceeds available host memory with the concurrent
100 GiB old offload server. The explicit capacity and configuration overrides
are included in the plan and identity. No SSD or external Mooncake service is
used. Current native Mooncake operation counters supply store/error evidence;
the original analyzer output is retained alongside this telemetry adaptation.

Each case directory contains its launch identity, commands, full server config,
before/after metrics, raw client/server output, completed application results,
request retry traces, process/affinity/thermal samples and concurrent peer
identity. `result.json` hashes the available artifacts and records qualification
and exclusions. Campaign reports include the exact result paths used in each
median. The primary interval runs from client process launch to exit, excluding
server initialization. Latencies, tokens, retries, transfers and thermal samples
are diagnostic; only `performance.total_e2e_s` drives the specified hard gates.

A work-inequivalent old result or an exhausted non-qualifying budget remains an
explicit unresolved decision. The report functions support separately identified
Phase-2 results and require a recorded reference-equivalence proof before reuse.
They never pool Phase-1 target measurements with a changed Phase-2 target.
Phase-2 execution requires the attribution and scope decision specified by the
change; it is not an automatic tuning loop.
