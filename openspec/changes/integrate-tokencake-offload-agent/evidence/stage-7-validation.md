# Stage 7: Focused Verification

The combined configuration, API, lifecycle, scheduling, connector, CPU-manager,
CUDA transfer, launcher and report suite passed all 420 tests in 430.80 seconds.
It includes actual full-weight Qwen2.5-14B generation, preemption recovery and
native CPU D2H/H2D output recovery, not a smoke workload.

Review then confined the running-request index lookup to TokenCake-selected
preemption victims, preserving the disabled native priority path's existing
cost. A mixed-dictionary type inference error in an offload model test was fixed
by using the same explicit string prompt in both request bodies.

After these edits, 46 focused native/TokenCake preemption, priority, ordinary
and disabled-path tests passed in 13.10 seconds. The actual 14B priority
contention/recovery test passed in 43.47 seconds, and the two actual 14B offload
and reload tests passed in 90.83 seconds.

## Commands And Artifacts

Commands ran from the target root. GPU/model tests used:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/tokencake \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_offload/cpu/test_manager.py -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-7-focused.xml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling.py tests/v1/core/test_scheduler.py \
  -k 'preempt or priority or disabled or ordinary' -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-7-preemption.xml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling_serving.py -k priority -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-7-priority-model.xml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_offloading_serving.py -q \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-7-offload-model.xml
```

Corresponding `stage-7-*-output.txt` files retain the command output. The
14B rechecks ran on separate GPUs; these are correctness tests, not timed
acceptance cases or performance comparison inputs.

All changed `vllm/`, `tools/`, `tests/tokencake/` and `pyproject.toml` files
relative to `c5838ce19` were supplied explicitly to `.venv/bin/pre-commit run
--files`. Every applicable hook passed, including Ruff, mypy for Python 3.10,
SPDX, forbidden imports and configuration checks. Output is retained in
`stage-7-precommit-output.txt`. The initial aggregate mypy error and one automatic
formatting pass preceded this successful final run. `git diff --check` and
`openspec validate integrate-tokencake-offload-agent --strict` also passed.

## Hardware And Runtime Freeze

```bash
.venv/bin/python -m tools.tokencake_experiments.driver prepare \
  /root/autodl-tmp/tokencake-acceptance/phase-1-final
```

The final read-only preflight passed. Both A800 UUIDs and NUMA CPU sets match;
ports 8055 and 8056 were available, no external GPU process was present, and
1,047,354,142,720 host bytes were available. All model shards and the declared
dataset checksum were verified. Target and official baseline imported Torch
2.11.0+cu130 and the expected native extensions; unchanged source imported its
Torch 2.6.0+cu124 stack. Mooncake 0.3.8 and its extension hashes were verified.
The baseline resolved to native FCFS with no additional config or KV offload.

Both environments generated identical 24-DAG contracts and arrival traces;
the workload contract hash is
`1590183398f961e31fa1253252e7624a5ef8a64d1406c00610a4587f999f60ed`.
The frozen records are copied to `stage-7-preflight/`; full preparation logs
and disposable launchers remain under the absolute campaign directory above.

The driver now includes both the read-only base package stack and local
overrides in environment identity and verifies them before execution. It
checks frozen JSON input hashes before each launch and audits the actual
recorded application arrival offsets against the source-generated trace.
Credential filtering preserves performance knobs containing `TOKENS`.
The final driver/report regression passed 31 tests in 7.14 seconds, recorded
in `stage-7-environment-final.xml`, with all applicable hooks passing.

Earlier preparation-only records at sibling `phase-1` and `phase-1-v2`
directories are retained. Neither launched a model server or consumed a
benchmark launch. The final experiment runtime is the committed vLLM tree of
`1cde5a084`; subsequent preflight/tooling evidence does not change that runtime.

Full 24-DAG performance acceptance is still pending. AI assistance was used.
