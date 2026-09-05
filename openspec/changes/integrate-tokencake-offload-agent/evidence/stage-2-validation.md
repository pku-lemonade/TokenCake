# Stage 2 Validation

Configuration and request metadata are implemented. The TokenCake connector
currently constructs the original native scheduler and worker; lifecycle
selection and scheduling policy are implemented in subsequent stages.

## Model-Backed Checks

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 VLLM_TEST_MODEL=/root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct .venv/bin/python -m pytest tests/tokencake/test_serving.py -v -s -x --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-2-http-model-final.xml
```

Result: **32 passed** in 125.90 seconds. Every configuration used a fresh
server with all eight original model shards, TP=PP=DP=1, eager execution,
maximum context 4096, GPU memory utilization 0.5, and at most eight sequences.
Offload-enabled configurations used 1 GiB native CPU offload and
`VLLM_USE_SIMPLE_KV_OFFLOAD=0`. The server selected the registered TokenCake
connector on both scheduler and worker sides in both offload modes.

The checks compare actual generated text for ordinary requests and four
concurrent annotated requests per API, verify complete streaming responses,
check matching IDs and strict field types, and exercise ordinary multi-choice
and multi-prompt generation alongside annotated HTTP 400 rejection.

GPU index 0 was verified as UUID
`GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f`. Both GPUs were idle before launch;
the server process tree shut down afterward. These are functional checks,
not full-DAG correctness or performance acceptance. No benchmark launch
budget was consumed.

## Supporting Checks

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_config.py tests/tokencake/test_protocol.py tests/v1/kv_connector/unit/offloading_connector -q --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-2-config-protocol-native-offline.xml
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/tokencake/test_config.py tests/tokencake/test_protocol.py -q --junitxml=openspec/changes/integrate-tokencake-offload-agent/evidence/stage-2-config-protocol-final.xml
```

Results: **213 passed** (including native connector regression tests), then
**152 passed** after adding coverage and compatibility for chat `n=null`.
Offline mode uses the existing native OPT test-model cache and avoids
unnecessary network metadata requests. Applicable changed-file pre-commit
checks, including mypy, pass.

Early attempts identified an unsupported UUID-form CUDA device selector,
a removed logging CLI flag, missing `ninja`, and a test combining greedy
sampling with `n=2`. The final invocation resolves all four. The local uv
environment now additionally installs `ninja==1.13.2`; the original shared
environment remains unchanged. The native greedy-sampling restriction is
preserved. Source and baseline tracked worktrees remain clean.

JUnit artifacts are retained alongside this report. Full local command logs
use the same basenames with `.log` extensions. The implementation used AI
assistance and still requires human review before an upstream submission.
