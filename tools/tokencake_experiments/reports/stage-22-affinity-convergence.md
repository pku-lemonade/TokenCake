# Stage 22: Shared-Prefix Score Range

## Decision

Keep `cache_affinity_score_band=500` with complete generation commitments.
Increasing the range to 1000 did not establish an additional total-E2E benefit.
Proceed to the complete five-QPS, four-component evaluation without changing the
workload. This is convergence of the tested scheduling parameters, not a claim
that every possible optimization has been exhausted.

## Full-DAG Diagnostic

Each launch completed the unchanged 24-DAG `conversation-tools` workload at
QPS 1.0 on GPU1, with 648 model calls, 155,136 generated tokens, no client retries,
no prompt halving, and speculative decoding disabled. All generation budgets
were logically committed at admission. Only the shared-prefix score range
changed between the frozen runtime trees.

| Measurement | Range 500 | Range 1000 |
| --- | ---: | ---: |
| Total E2E (s) | 639.216284 | 638.154387 |
| Mean application latency (s) | 535.832208 | 516.114070 |
| P50 application latency (s) | 526.048088 | 551.683004 |
| P95 application latency (s) | 611.887524 | 604.542138 |
| Physical / reservation preemptions | 0 / 0 | 0 / 0 |
| Actually recomputed tokens | 0 | 0 |
| Executed tokens | 1,083,016 | 1,115,572 |
| First-prefill computed tokens | 928,528 | 961,084 |
| Critical waits >= 60 s | 129 | 92 |
| Maximum critical admission wait (s) | 96.460 | 92.765 |
| Mean GPU activity | 98.6973% | 98.7404% |
| Mean GPU SM clock (MHz) | 1389.167 | 1388.098 |

The E2E reduction was 0.166125%, while executed work increased by 32,556 tokens.
The lower mean and P95 latency are useful observations, but the higher P50 and
single-launch design do not establish a stable overall improvement. Retain the
narrower preference range and its existing importance protection. GPU activity
near 99% and zero preemption-related recomputation suggest that further tuning
of these two parameters has limited immediate headroom on this workload; GPU
activity alone does not prove a hardware throughput limit.

## Evidence

Artifacts are under `/root/autodl-tmp/tokencake-optimization/`:

- Range 500: `component-all-01/cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-1/result.json`.
- Range 1000: `component-affinity-01/cases/phase-1/offload-agent/qps-1.0/gpu-1/launch-0/result.json`.

| Artifact | SHA-256 |
| --- | --- |
| Range 1000 result | `fa385b7565d1ee598149d37bf1302c53cbbab085cfeea4cfdffb099019cd8e98` |
| Range 1000 runtime | `b321a9694b9e2535aa02a600cd737a60bf81c6aad370992a3765bf3cbfe72983` |
| Range 500 runtime | `a0672d391b65e83abc712d87af17741becc9c491a56a55314eb6c1cf475d954f` |
| Frozen inputs | `3b8f1be50843e2165cd31a0a7c11b27d0a437df1789eca41129307d713f5e171` |

The focused shared-prefix scheduling tests passed 45 cases, including the
extended-range boundary and completion behavior across both queue policies and
multiple cache groups (`.venv/component-affinity-cpu.xml`). The additional
boundary tests remain; the experimental default change was reverted.

## Final-Version Validation

- 464 CPU tests passed, 3 deselected, in 112.92 s. Coverage includes scheduling,
  offload policy, native CPU ownership and connector behavior, configuration,
  workload integrity, and the experiment/report tooling.
  Record: `.venv/component-final-cpu.xml`.
- Four real Qwen2.5-14B service tests passed in 173.45 s. They exercise both
  scheduling policies, pressure recovery, preserved-prefix reload with and
  without agent scheduling, output checks, and external cache reset.
  Record: `.venv/component-final-serving.xml`.
- All applicable pre-commit checks passed, including Ruff, local mypy,
  Markdown, SPDX, and configuration validation.
- The extracted diagnostic data, including the earlier excluded startup, is
  `/root/autodl-tmp/tokencake-optimization/component-diagnostics.json`.

Validation commands:

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_component_analysis.py \
  tests/tokencake/test_experiment_driver.py \
  tests/tokencake/test_experiment_launcher.py \
  tests/tokencake/test_experiment_report.py \
  tests/tokencake/test_experiment_workload.py \
  tests/tokencake/test_scheduling.py tests/tokencake/test_offloading.py \
  tests/tokencake/test_config.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  -q -k 'not data_parallel_rejected' --junitxml=.venv/component-final-cpu.xml

env PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 \
  VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling_serving.py \
  tests/tokencake/test_offloading_serving.py \
  -q --junitxml=.venv/component-final-serving.xml
```
