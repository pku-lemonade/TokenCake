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

The hardware/runtime freeze is the remaining Stage-7 step. Full 24-DAG
performance acceptance is still pending. AI assistance was used.
