# Stage 4: Agent Scheduling

Implemented the reachable source request score, type importance, critical-type
selection, 500-step reserve adjustment, partitioning, annotated borrowing,
ordinary shared-only admission, frozen per-step ordering, 256-token pressure cap,
and native preemption/requeue integration. Native queue classes and disabled
scheduling paths remain in use.

Allocation demand uses the native multi-group coordinator. Physical blocks carry
one capacity charge across shared prefix references; native reference counts
determine release. There are no Request fields or native priority rewrites.
Source preemption uses a fixed 3000-point margin. Target preemption retains the
native rollback and additionally restores the current candidate's pending
encoder budget when a previously scheduled agent victim is removed.

## Source Comparison

The frozen source environment executed the actual source methods to generate
`tests/tokencake/scheduling_reference.json`, including 32 request scores and four
agent histories, their critical types and exact reservation partition.

```bash
PYTHONPATH=/root/TokenCake/vllm_agent \
  /root/TokenCake/vLLM-TokenCake-upstream/.venv/source/.venv/bin/python \
  /root/TokenCake/vLLM-TokenCake-upstream/tools/tokencake_experiments/scheduling_reference.py \
  --output /root/TokenCake/vLLM-TokenCake-upstream/tests/tokencake/scheduling_reference.json
```

The exporter verifies source revision
`7a608a4e53ea990b2540c93b4d28cb795b905109`. Source git status remains clean.

## Execution Evidence

| Validation | Result | Artifact |
| --- | --- | --- |
| 14B model, 12 concurrent mixed requests under 320-block KV capacity, FCFS and priority; real preemption counter increase, bounded prefills, 128-token completion and serial recovery | 2 passed, 88.85 s | `stage-4-pressure-model.xml` |
| 14B model, three generation APIs, independent configuration modes, authenticated lifecycle events, streaming/abort, duplicate association and resets | 40 passed, 188.01 s | `stage-4-api-model-regression.xml` |
| Golden scores, multi-group demand, shared prefixes, mixed admission, barriers, LoRA eligibility, encoder/speculative/token rollback, reserve bounds | 36 passed, 9.41 s | `stage-4-scheduling.xml` |
| Native scheduler priority/preemption/add/finish regression | 26 passed, 9.11 s | `stage-4-native-scheduler-fixed-env.xml` |
| Configuration, protocol, lifecycle and event regression | 193 passed, 44.53 s | `stage-4-config-lifecycle-regression.xml` |

The GPU commands used `PATH="$PWD/.venv/bin:$PATH"`, integer device indices
resolved to the frozen A800 devices, and
`VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling_serving.py -v -s -x \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-4-pressure-model.xml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/tokencake/test_serving.py tests/tokencake/test_event_serving.py -v -s -x \
  --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-4-api-model-regression.xml
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_scheduling.py -x -q
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/v1/core/test_scheduler.py -q \
  -k 'priority or preempt or add_requests or finish_request'
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_config.py \
  tests/tokencake/test_protocol.py tests/tokencake/test_lifecycle.py \
  tests/tokencake/test_events.py -q
```

Changed-file pre-commit checks passed, including Ruff and mypy. Unit checks are
supporting evidence; real-model executions establish this stage's runtime
correctness. These are not the full-DAG performance gates in stages 8 and 9.

## Environment Repair

The inherited xFormers package targeted Torch 2.6, preventing LLaVA model-class
loading. Installed `xformers==0.0.35` into the target local `.venv` with
`uv pip install --python .venv/bin/python --no-deps xformers==0.0.35`; its declared
Torch requirement is `>=2.10`. Downloaded the missing LLaVA tokenizer files through
`huggingface_hub` using `HF_ENDPOINT=https://hf-mirror.com`. After these repairs,
all 26 native scheduler tests passed. Earlier failed environment attempts are
retained under `.venv/tokencake-stage-4-attempts/`. The original environment and
source checkout were not changed.
